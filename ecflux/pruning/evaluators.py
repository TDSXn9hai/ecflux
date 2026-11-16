"""APRA 质量反馈信号。

- fid          : 报告原设定，验证集 prompt 20 步采样 -> VAE 解码 -> 与参考图像的 FID（需 VAE、参考图、torchmetrics）
- velocity_mse : 代理指标，固定 (x_t, t, c) 上当前模型与原始模型速度场的 MSE（无需采样，适合快速实验 / 冒烟）
- flow_loss    : 代理指标，固定 batch 上的 Rectified Flow 损失
- traj_velocity_mse : 沿原始模型 CFG 采样轨迹的每一步，比较当前模型与原始模型的引导后速度场（teacher forcing）
- hps          : 固定 prompt / seed 采样 -> VAE 解码 -> HPSv2.1，返回 原始模型均值 - 当前均值
所有 evaluator 返回「越小越好」的标量。eval.aux_metrics 非空时用 MultiEvaluator 同时记录多个指标（.last）。
"""

import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

from ..flux_utils import (decode_latents, dit_forward, euler_sample, fm_forward, load_scheduler, pack_latents,
                          prepare_latent_ids, prepare_text_ids, to_pil)

logger = logging.getLogger("ecflux")


def _to(fb, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in fb.items()}


class VelocityMSEEvaluator:
    def __init__(self, ref_model, batches: list[dict], device, scale: float = 100.0):
        self.batches = [{k: (v.cpu() if torch.is_tensor(v) else v) for k, v in b.items()} for b in batches]
        self.device, self.scale = device, scale
        with torch.no_grad():
            self.refs = [fm_forward(ref_model, _to(b, device)).float().cpu() for b in self.batches]

    @torch.no_grad()
    def __call__(self, model) -> float:
        self.per_item = [self.scale * F.mse_loss(fm_forward(model, _to(b, self.device)).float().cpu(), r).item()
                         for b, r in zip(self.batches, self.refs)]  # 逐 batch，用于估计标准误
        return sum(self.per_item) / len(self.per_item)


class FlowLossEvaluator:
    def __init__(self, batches: list[dict], device, scale: float = 100.0):
        self.batches, self.device, self.scale = batches, device, scale

    @torch.no_grad()
    def __call__(self, model) -> float:
        tot = 0.0
        for b in self.batches:
            b = _to(b, self.device)
            tot += F.mse_loss(fm_forward(model, b).float(), b["target"].float()).item()
        return self.scale * tot / len(self.batches)


class FIDEvaluator:
    """prompt_embeds: 预先编码好的验证集 prompt（list of (1,L,D) 或 (B,L,D)），real_images: 参考图迭代器 [0,1]。"""

    def __init__(self, vae, prompt_embeds: list[torch.Tensor], real_images, device, height=512, width=512,
                 steps=20, guidance_scale=4.0, negative_embeds=None, scheduler=None, seed=0, feature=2048):
        from torchmetrics.image.fid import FrechetInceptionDistance

        self.fid = FrechetInceptionDistance(feature=feature, normalize=True, reset_real_features=False).to(device)
        for imgs in real_images:
            self.fid.update(imgs.to(device), real=True)
        self.vae, self.embeds, self.device = vae, prompt_embeds, device
        self.h, self.w, self.steps, self.cfg = height, width, steps, guidance_scale
        self.neg, self.scheduler, self.seed = negative_embeds, scheduler, seed

    @torch.no_grad()
    def __call__(self, model) -> float:
        self.fid.reset()
        g = torch.Generator().manual_seed(self.seed)
        for emb in self.embeds:
            emb = emb.to(self.device)
            neg = None if self.neg is None else self.neg.to(self.device).expand(emb.shape[0], -1, -1)
            lat, ids = euler_sample(model, emb, self.h, self.w, self.steps, self.cfg, neg, self.scheduler, g)
            imgs = decode_latents(self.vae, lat.to(self.vae.dtype), ids)
            self.fid.update(F.interpolate(imgs, size=(299, 299), mode="bilinear"), real=False)
        return float(self.fid.compute())


def _cfg_velocity(model, x, t, emb, neg, img_ids, txt_ids, neg_txt_ids, cfg):
    dtype = next(model.parameters()).dtype
    v = dit_forward(model, x.to(dtype), t.to(dtype), emb.to(dtype), img_ids, txt_ids)
    if neg is not None:
        v_neg = dit_forward(model, x.to(dtype), t.to(dtype), neg.to(dtype), img_ids, neg_txt_ids)
        v = v_neg + cfg * (v - v_neg)
    return v.float()


class TrajVelocityEvaluator:
    """prompt_embeds: list of (B,L,D)。用原始模型做一次 CFG Euler 采样（与 euler_sample 相同的调度），记录每一步的
    输入 x_t 与引导后速度场；评估时在相同 x_t 上前向当前模型，返回所有步、所有 prompt 的平均 MSE × scale。"""

    def __init__(self, ref_model, prompt_embeds, negative_embeds, device, height=512, width=512, steps=20,
                 guidance_scale=4.0, scheduler=None, seed=0, scale: float = 100.0):
        self.device, self.scale, self.cfg = device, scale, guidance_scale
        scheduler = scheduler or load_scheduler(None)
        g = torch.Generator().manual_seed(seed)
        c = ref_model.config.in_channels
        h, w = 2 * (int(height) // 16) // 2, 2 * (int(width) // 16) // 2
        self.items = []
        with torch.no_grad():
            for emb in prompt_embeds:
                emb = emb.to(device)
                bsz = emb.shape[0]
                neg = None if negative_embeds is None else negative_embeds.to(device).expand(bsz, -1, -1)
                lat = torch.randn((bsz, c, h, w), generator=g, dtype=torch.float32).to(device)
                img_ids, txt_ids = prepare_latent_ids(lat).to(device), prepare_text_ids(emb).to(device)
                neg_txt_ids = None if neg is None else prepare_text_ids(neg).to(device)
                lat = pack_latents(lat)
                scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), device=device,
                                        mu=compute_empirical_mu(image_seq_len=lat.shape[1], num_steps=steps))
                scheduler.set_begin_index(0)
                states = []
                for t in scheduler.timesteps:
                    tt = (t.expand(bsz) / 1000).float()
                    v = _cfg_velocity(ref_model, lat, tt, emb, neg, img_ids, txt_ids, neg_txt_ids, guidance_scale)
                    states.append((lat.cpu(), tt.cpu(), v.cpu()))
                    lat = scheduler.step(v, t, lat.float(), return_dict=False)[0]
                self.items.append({"emb": emb.cpu(), "neg": None if neg is None else neg.cpu(), "img_ids": img_ids,
                                   "txt_ids": txt_ids, "neg_txt_ids": neg_txt_ids, "states": states})

    @torch.no_grad()
    def __call__(self, model) -> float:
        per = []  # [prompt batch][step]
        for it in self.items:
            emb = it["emb"].to(self.device)
            neg = None if it["neg"] is None else it["neg"].to(self.device)
            row = []
            for x, t, v_ref in it["states"]:
                v = _cfg_velocity(model, x.to(self.device), t.to(self.device), emb, neg, it["img_ids"],
                                  it["txt_ids"], it["neg_txt_ids"], self.cfg)
                row.append(self.scale * F.mse_loss(v.cpu(), v_ref).item())
            per.append(row)
        self.per_item = [sum(r) / len(r) for r in per]           # 逐 prompt batch，用于估计标准误
        self.per_step = [sum(c) / len(c) for c in zip(*per)]     # 逐采样步，看误差集中在哪些 t
        return sum(self.per_item) / len(self.per_item)


class HPSEvaluator:
    """固定 prompt / seed 的 20 步 CFG 采样 -> VAE 解码 -> HPSv2.1。返回 base - mean（越小越好），.mean 为本次均值。"""

    def __init__(self, ref_model, vae, prompts: list[str], prompt_embeds, device, height=512, width=512, steps=20,
                 guidance_scale=4.0, negative_embeds=None, scheduler=None, seed=0):
        from ..metrics import HPSv2Scorer

        self.scorer = HPSv2Scorer()
        self.vae, self.prompts, self.embeds, self.device = vae, prompts, prompt_embeds, device
        self.h, self.w, self.steps, self.cfg = height, width, steps, guidance_scale
        self.neg, self.scheduler, self.seed = negative_embeds, scheduler, seed
        self.base = self.score(ref_model)
        self.mean = self.base
        logger.info(f"[HPSEvaluator] base HPSv2 = {self.base:.4f} on {len(prompts)} prompts")

    @torch.no_grad()
    def score(self, model) -> float:
        g = torch.Generator().manual_seed(self.seed)
        scores, k = [], 0
        for emb in self.embeds:
            emb = emb.to(self.device)
            neg = None if self.neg is None else self.neg.to(self.device).expand(emb.shape[0], -1, -1)
            lat, ids = euler_sample(model, emb, self.h, self.w, self.steps, self.cfg, neg, self.scheduler, g)
            imgs = to_pil(decode_latents(self.vae, lat.to(self.vae.dtype), ids))
            scores += self.scorer(imgs, self.prompts[k:k + len(imgs)])
            k += len(imgs)
        self.per_item = scores
        return sum(scores) / len(scores)

    def __call__(self, model) -> float:
        self.mean = self.score(model)
        return self.base - self.mean


class DriftReference:
    """漂移校正参照：不剪枝的 4B 用与 APRA 相同的恢复训练（每 chunk_steps 步一个 checkpoint，scripts/drift_ref.py 生成）。
    剪枝模型训练了 n 步时，与 ref_dir/ckpt_{n // chunk_steps} 比较：
      traj_ref = 在参照模型的 CFG 采样轨迹上，剪枝模型与参照模型引导后速度场的 MSE × scale
      hps_ref  = 参照模型 HPSv2 均值（ckpt 的 metrics.json）- 剪枝模型 HPSv2 均值
    这样恢复训练本身造成的偏移（风格、HPS 下降）被扣除，只剩剪枝带来的损伤。checkpoint 未就绪时轮询等待。"""

    def __init__(self, ref_dir, chunk_steps, prompt_embeds, negative_embeds, device, wait_s=60, **sample_kw):
        self.ref_dir, self.chunk, self.device, self.wait_s = ref_dir, int(chunk_steps), device, wait_s
        self.embeds, self.neg, self.kw = prompt_embeds, negative_embeds, sample_kw
        self.steps, self.cache = 0, {}

    def set_trained_steps(self, n: int):
        if n % self.chunk:
            raise ValueError(f"trained steps {n} 不是参照 checkpoint 间隔 {self.chunk} 的整数倍")
        self.steps = n

    def _ckpt(self, k):
        d = os.path.join(self.ref_dir, f"ckpt_{k}")
        while not os.path.exists(os.path.join(d, "metrics.json")):
            logger.info(f"[drift] waiting for {d}")
            time.sleep(self.wait_s)
        return d

    def get(self, k):
        if k not in self.cache:
            from ..flux_utils import load_transformer

            d = self._ckpt(k)
            ref = load_transformer(d).to(self.device).eval().requires_grad_(False)
            ev = TrajVelocityEvaluator(ref, self.embeds, self.neg, self.device, **self.kw)
            del ref
            torch.cuda.empty_cache()
            self.cache = {k: (ev, json.load(open(os.path.join(d, "metrics.json"))))}  # 只保留当前 k，控制内存
        return self.cache[k]


class MultiEvaluator:
    """主指标 + 辅助指标：返回主指标，.last 记录本次所有指标（HPS 额外记录绝对均值 hps_abs）。
    设置 drift（DriftReference）时额外记录 traj_ref / hps_ref（相对同步数恢复训练的未剪枝参照）。"""

    def __init__(self, evaluators: dict, primary: str, drift: DriftReference | None = None):
        self.evaluators, self.primary, self.last, self.drift = evaluators, primary, {}, drift

    def set_trained_steps(self, n: int):
        if self.drift is not None:
            self.drift.set_trained_steps(n)

    def __call__(self, model) -> float:
        self.last = {}
        for name, ev in self.evaluators.items():
            t0 = time.perf_counter()
            self.last[name] = ev(model)
            self.last[f"{name}_time_s"] = time.perf_counter() - t0
            if isinstance(ev, HPSEvaluator):
                self.last["hps_abs"] = ev.mean
            per = getattr(ev, "per_item", None)
            if per and len(per) > 1:
                self.last[f"{name}_se"] = float(np.std(per, ddof=1) / np.sqrt(len(per)))
            if getattr(ev, "per_step", None):
                self.last[f"{name}_per_step"] = ev.per_step
        if self.drift is not None:
            k = self.drift.steps // self.drift.chunk
            self.last["ref_ckpt"] = k
            if k == 0:  # 尚未做恢复训练：参照即原始模型
                self.last["traj_ref"], self.last["hps_ref"] = self.last["traj_velocity_mse"], self.last["hps"]
            else:
                t0 = time.perf_counter()
                ev, meta = self.drift.get(k)
                self.last["traj_ref"] = ev(model)
                self.last["hps_ref"] = meta["hps_abs"] - self.last["hps_abs"]
                self.last["traj_ref_time_s"] = time.perf_counter() - t0
        logger.info(f"[eval] { {k: v for k, v in self.last.items() if not k.endswith('per_step')} }")
        return self.last[self.primary]


def build_evaluator(kind: str, **kw):
    if kind == "velocity_mse":
        return VelocityMSEEvaluator(kw["ref_model"], kw["batches"], kw["device"], kw.get("scale", 100.0))
    if kind == "flow_loss":
        return FlowLossEvaluator(kw["batches"], kw["device"], kw.get("scale", 100.0))
    if kind == "fid":
        return FIDEvaluator(**{k: v for k, v in kw.items() if k not in ("ref_model", "batches", "scale")})
    if kind == "traj_velocity_mse":
        return TrajVelocityEvaluator(**{k: v for k, v in kw.items() if k not in ("batches", "vae", "prompts")})
    if kind == "hps":
        return HPSEvaluator(**{k: v for k, v in kw.items() if k not in ("batches", "scale")})
    raise ValueError(kind)
