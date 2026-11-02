"""4.3 语义对齐知识蒸馏训练器。

教师（默认 FLUX.2 Klein 9B，冻结）两种工作方式：
- online : 教师与学生同卡，逐 batch 前向（显存足够或教师用 fp8/offload 时）
- offline: 先用 scripts/cache_teacher.py 缓存教师速度场 / 隐藏状态 / 注意力图，训练时不加载教师（报告 4.3.1 方案）
"""

import itertools
import logging
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset

from ..flux_utils import add_noise, fm_forward, prepare_fm_batch, prepare_text_ids
from ..hooks import FeatureRecorder, block_map
from ..pruning.prune_ops import save_pruned_transformer
from ..utils import Tracker, build_optimizer, build_scheduler, count_params, peak_memory_gb
from .losses import DistillLoss, HiddenProjectors

logger = logging.getLogger("ecflux")

DEFAULT_STUDENT_HIDDEN = ["double:3", "double:4", "single:11"]  # G2 末尾、G3 末尾、G5 中间


def model_depth(model_or_cfg) -> tuple[int, int]:
    c = model_or_cfg.config if hasattr(model_or_cfg, "config") else model_or_cfg
    return int(c["num_layers"]), int(c["num_single_layers"])


def auto_teacher_layers(student_layers: list[str], s_depth, t_depth) -> list[str]:
    """按相对深度把学生（原始编号）层位映射到教师层位。"""
    out = []
    for spec in student_layers:
        kind, idx = spec.split(":")
        ns, nt = (s_depth[0], t_depth[0]) if kind == "double" else (s_depth[1], t_depth[1])
        out.append(f"{kind}:{min(nt - 1, round((int(idx) + 1) / ns * nt) - 1)}")
    return out


def hidden_dim(model_or_cfg) -> int:
    c = model_or_cfg.config if hasattr(model_or_cfg, "config") else model_or_cfg
    return int(c["num_attention_heads"]) * int(c["attention_head_dim"])


class OfflineTeacherDataset(Dataset):
    """读取 cache_teacher.py 的输出：每个样本含 K 次 (t, noise) 抽样及对应的教师目标，训练时随机取一组。"""

    def __init__(self, root: str, seed: int = 0):
        import glob

        self.files = sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))
        if not self.files:
            raise FileNotFoundError(root)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = torch.load(self.files[i], map_location="cpu", weights_only=False)
        draw = self.rng.choice(d["teacher"])
        out = {k: d[k] for k in ("latents", "prompt_embeds", "caption") if k in d}
        out["t"], out["noise"], out["v_teacher"] = draw["t"], draw["noise"], draw["v"]
        for k, v in draw.get("hidden", {}).items():
            out[f"th:{k}"] = v
        for k, v in draw.get("attn", {}).items():
            out[f"ta:{k}"] = v
        return out


def collate_any(batch):
    out = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        out[k] = torch.stack([torch.as_tensor(v) for v in vals]) if torch.is_tensor(vals[0]) or isinstance(vals[0], float) else vals
    return out


def infinite(loader):
    while True:
        for b in loader:
            yield b


class Distiller:
    def __init__(self, student, teacher, cfg: dict, device):
        self.student, self.teacher, self.cfg, self.device = student, teacher, cfg, device
        dcfg = cfg["distill"]
        self.s_hidden = dcfg.get("student_hidden_layers") or DEFAULT_STUDENT_HIDDEN
        # 学生层位若已被块剪枝删除则跳过
        bm = block_map(student)
        self.s_hidden = [s for s in self.s_hidden if int(s.split(":")[1]) in bm[s.split(":")[0]]]
        t_cfg = teacher.config if teacher is not None else cfg["teacher"]["config"]
        s_depth = model_depth(student.config)
        t_layers = dcfg.get("teacher_hidden_layers", "auto")
        self.t_hidden = auto_teacher_layers(self.s_hidden, s_depth, model_depth(t_cfg)) if t_layers == "auto" else t_layers
        k = int(dcfg.get("attn_k", 2))
        self.attn_layers = [l for l in range(k) if l in bm["double"]]
        self.projectors = HiddenProjectors(self.s_hidden, hidden_dim(student.config), hidden_dim(t_cfg)).to(device)
        self.loss = DistillLoss(dcfg, self.projectors, self.s_hidden, self.t_hidden, self.attn_layers)
        self.query_stride = int(dcfg.get("attn_query_stride", 2))
        self.t_embeds_key = cfg.get("teacher", {}).get("embeds_key", "prompt_embeds")
        # 学生侧记录器在整个训练期间常驻：梯度检查点在反向时会重算块内前向，记录逻辑必须与正向一致
        w = self.loss.weights(10**9)
        self.s_rec = FeatureRecorder(student, self.s_hidden if w["hid"] > 0 else [],
                                     self.attn_layers if w["attn"] > 0 else [], query_stride=self.query_stride)
        self.s_rec.__enter__()
        logger.info(f"hidden align: student {self.s_hidden} <-> teacher {self.t_hidden}; attn layers {self.attn_layers}")

    def head_index(self):
        """教师为原始未剪枝同构模型时，可按原始头编号逐头对齐（attn_head_mode=match）。"""
        units = getattr(self.student, "ecflux_units", None)
        if not units or self.cfg["distill"].get("attn_head_mode", "mean") != "match":
            return None
        return {l: units[("double", l)]["heads"] for l in self.attn_layers if ("double", l) in units}

    @torch.no_grad()
    def teacher_targets(self, fb: dict, batch: dict):
        if "v_teacher" in batch:  # offline
            v_t = batch["v_teacher"].to(self.device)
            hid = {n: batch[f"th:{n}"] for n in self.t_hidden if f"th:{n}" in batch}
            att = {l: batch[f"ta:{l}"] for l in self.attn_layers if f"ta:{l}" in batch}
            return v_t, hid, att
        emb = batch.get(self.t_embeds_key, batch.get("prompt_embeds")).to(self.device, fb["prompt_embeds"].dtype)
        txt_ids = prepare_text_ids(emb).to(self.device)
        use_attn = self.loss.weights(10**9)["attn"] > 0
        with FeatureRecorder(self.teacher, self.t_hidden, self.attn_layers if use_attn else [],
                             query_stride=self.query_stride, detach=True) as rec:
            v_t = fm_forward(self.teacher, fb, prompt_embeds=emb, txt_ids=txt_ids)
        return v_t, dict(rec.hidden), dict(rec.attn)

    def compute(self, batch: dict, step: int, generator=None):
        dcfg = self.cfg["distill"]
        dtype = next(self.student.parameters()).dtype
        fb = prepare_fm_batch(batch, self.device, dtype, generator, self.cfg.get("timesteps"))
        v_t, t_hid, t_att = self.teacher_targets(fb, batch)
        x_in = fb["x_t"]
        sigma = float(dcfg.get("robust_sigma", 0.01))
        if dcfg.get("robust", True) and sigma > 0:
            x_in = x_in + sigma * torch.randn_like(x_in)
        self.s_rec.clear()
        v_s = fm_forward(self.student, fb, x_t=x_in)
        return self.loss(step, v_s, v_t, self.s_rec, t_hid, t_att, fb["target"], self.head_index())

    def close(self):
        self.s_rec.__exit__(None, None, None)


def train(student, teacher, dataset, cfg: dict, device, out_dir: str, encoder=None):
    """encoder 非空时（data.mode=online），dataset 输出原始图文，每个 micro-batch 先在线编码再蒸馏。"""
    tcfg = cfg["train"]
    student.to(device).train()
    student.requires_grad_(True)
    if teacher is not None:
        teacher.to(device).eval().requires_grad_(False)
    if tcfg.get("gradient_checkpointing", True):
        student.enable_gradient_checkpointing()
    distiller = Distiller(student, teacher, cfg, device)
    params = list(student.parameters()) + list(distiller.projectors.parameters())
    opt = build_optimizer(params, tcfg)
    steps = int(tcfg.get("max_steps", 10000))
    sched = build_scheduler(opt, steps, int(tcfg.get("warmup_steps", 200)))
    micro = int(tcfg.get("micro_batch_size", 1))
    accum = max(1, int(tcfg.get("global_batch_size", 64)) // micro)
    loader = DataLoader(dataset, batch_size=micro, shuffle=True, num_workers=int(tcfg.get("num_workers", 2)),
                        collate_fn=collate_any, drop_last=True)
    it = infinite(loader)
    tracker = Tracker(cfg.get("wandb"), cfg)
    g = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    mp = tcfg.get("mixed_precision", "bf16")
    use_amp = device.type == "cuda" and mp != "no"
    amp_dtype = torch.float16 if mp == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and mp == "fp16")
    logger.info(f"student params={count_params(student) / 1e6:.1f}M, accum={accum}, steps={steps}")
    os.makedirs(out_dir, exist_ok=True)
    for step in range(steps):
        agg = {}
        for _ in range(accum):
            batch = next(it)
            if encoder is not None:
                batch = encoder(batch)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                loss, logs = distiller.compute(batch, step, g)
            scaler.scale(loss / accum).backward()
            for k, v in logs.items():
                agg[k] = agg.get(k, 0.0) + v / accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, float(tcfg.get("max_grad_norm", 1.0)))
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % int(tcfg.get("log_every", 10)) == 0 or step == steps - 1:
            agg.update({"lr": sched.get_last_lr()[0], "peak_mem_gb": peak_memory_gb(device)})
            tracker.log(agg, step=step)
        save_every = int(tcfg.get("save_every", 2000))
        if save_every > 0 and (step + 1) % save_every == 0 and step + 1 < steps:
            save_checkpoint(student, distiller, os.path.join(out_dir, f"step_{step + 1}"), cfg)
    distiller.close()
    if tcfg.get("gradient_checkpointing", True):
        student.disable_gradient_checkpointing()
    save_checkpoint(student, distiller, os.path.join(out_dir, "final"), cfg)
    tracker.finish()
    return distiller


def save_checkpoint(student, distiller, path, cfg):
    save_pruned_transformer(student, path, extra={"distill": cfg.get("distill", {})})
    torch.save(distiller.projectors.state_dict(), os.path.join(path, "projectors.pt"))
    logger.info(f"saved checkpoint -> {path}")


@torch.no_grad()
def teacher_draws(teacher, sample: dict, num_draws: int, t_hidden: list[str], attn_layers: list[int], device,
                  dtype, generator=None, timestep_cfg=None, embeds_key="prompt_embeds", query_stride: int = 2,
                  store_dtype=torch.float16) -> list[dict]:
    """离线缓存：对一个样本抽 K 组 (t, ε)，记录教师 v_T、隐藏状态与注意力图（头维度取均值以节省磁盘）。"""
    batch = {"latents": sample["latents"][None]}
    emb_key = embeds_key if embeds_key in sample else "prompt_embeds"
    batch["prompt_embeds"] = sample[emb_key][None]
    draws = []
    for _ in range(num_draws):
        fb = prepare_fm_batch(batch, device, dtype, generator, timestep_cfg)
        with FeatureRecorder(teacher, t_hidden, attn_layers, query_stride=query_stride, detach=True) as rec:
            v = fm_forward(teacher, fb)
        draws.append({
            "t": fb["t"][0].float().cpu(), "noise": fb["noise"][0].to(store_dtype).cpu(), "v": v[0].to(store_dtype).cpu(),
            "hidden": {k: h[0].to(store_dtype).cpu() for k, h in rec.hidden.items()},
            "attn": {k: a[0].mean(0, keepdim=True).to(store_dtype).cpu() for k, a in rec.attn.items()},
        })
    return draws
