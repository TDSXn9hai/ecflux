"""APRA 质量反馈信号。

- fid          : 报告原设定，验证集 prompt 20 步采样 -> VAE 解码 -> 与参考图像的 FID（需 VAE、参考图、torchmetrics）
- velocity_mse : 代理指标，固定 (x_t, t, c) 上当前模型与原始模型速度场的 MSE（无需采样，适合快速实验 / 冒烟）
- flow_loss    : 代理指标，固定 batch 上的 Rectified Flow 损失
所有 evaluator 返回「越小越好」的标量。
"""

import logging

import torch
import torch.nn.functional as F

from ..flux_utils import decode_latents, euler_sample, fm_forward

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
        tot = 0.0
        for b, r in zip(self.batches, self.refs):
            tot += F.mse_loss(fm_forward(model, _to(b, self.device)).float().cpu(), r).item()
        return self.scale * tot / len(self.batches)


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


def build_evaluator(kind: str, **kw):
    if kind == "velocity_mse":
        return VelocityMSEEvaluator(kw["ref_model"], kw["batches"], kw["device"], kw.get("scale", 100.0))
    if kind == "flow_loss":
        return FlowLossEvaluator(kw["batches"], kw["device"], kw.get("scale", 100.0))
    if kind == "fid":
        return FIDEvaluator(**{k: v for k, v in kw.items() if k not in ("ref_model", "batches", "scale")})
    raise ValueError(kind)
