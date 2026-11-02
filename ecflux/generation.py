"""文生图推理封装：Text Encoder（冻结）+ DiT（原始 / 剪枝 / 蒸馏）+ VAE，与 Flux2KleinPipeline 数值一致。"""

import logging
import time

import torch

from .flux_utils import TextEncoder, decode_latents, euler_sample, load_scheduler, load_transformer, load_vae, to_pil

logger = logging.getLogger("ecflux")


class T2IGenerator:
    def __init__(self, pipeline_path: str, transformer_path: str | None = None, device="cuda", dtype=torch.bfloat16,
                 max_sequence_length: int = 512, text_encoder: TextEncoder | None = None):
        self.device, self.dtype = torch.device(device), dtype
        self.text_encoder = text_encoder or TextEncoder(pipeline_path, device, dtype, max_sequence_length)
        self.transformer = load_transformer(transformer_path or pipeline_path, dtype=dtype).to(self.device).eval()
        self.vae = load_vae(pipeline_path, dtype=torch.float32, device=self.device)
        self.scheduler = load_scheduler(pipeline_path)
        self._neg = None

    def negative_embeds(self):
        if self._neg is None:
            self._neg = self.text_encoder.encode([""])
        return self._neg

    @torch.no_grad()
    def __call__(self, prompts: list[str], height=1024, width=1024, steps=50, guidance_scale=4.0, seed=0,
                 prompt_embeds=None, return_timing=False):
        t0 = time.perf_counter()
        emb = prompt_embeds if prompt_embeds is not None else self.text_encoder.encode(prompts)
        neg = self.negative_embeds().expand(emb.shape[0], -1, -1) if guidance_scale > 1 else None
        g = torch.Generator().manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat, ids = euler_sample(self.transformer, emb, height, width, steps, guidance_scale, neg, self.scheduler, g)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        imgs = to_pil(decode_latents(self.vae, lat, ids))
        if return_timing:
            return imgs, {"text_s": t1 - t0, "dit_s": t2 - t1, "total_s": time.perf_counter() - t0}
        return imgs
