"""表2 评估指标：FID、CLIP Score、HPSv2、参数量 / 峰值显存 / 推理延迟。
GenEval 与 T2I-CompBench 需使用官方仓库评测，scripts/generate.py 负责按其目录格式导出图片。"""

import glob
import os

import torch
import torch.nn.functional as F


class ClipScorer:
    """CLIP Score = 100 · max(cos(E_img, E_txt), 0)，默认 ViT-L/14。"""

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    @torch.no_grad()
    def __call__(self, images, prompts) -> list[float]:
        inp = self.proc(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model(**inp)
        s = F.cosine_similarity(out.image_embeds, out.text_embeds, dim=-1)
        return (100 * s.clamp_min(0)).tolist()


class HPSv2Scorer:
    """依赖官方 hpsv2 包（pip install hpsv2）；首次调用自动下载 HPS v2.1 权重。"""

    def __init__(self, version="v2.1"):
        import hpsv2  # noqa: F401

        self.version = version

    def __call__(self, images, prompts) -> list[float]:
        import hpsv2

        return [float(hpsv2.score(img, p, hps_version=self.version)[0]) for img, p in zip(images, prompts)]


def square_resize(img, size: int):
    """短边缩放 + 中心裁剪为正方形，再缩放到 size（FID 预处理；COCO 原图非正方形，直接拉伸会让分布失真）。"""
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return img.crop((left, top, left + s, top + s)).resize((size, size), 3)  # 3 = BICUBIC


def load_images(folder: str):
    from PIL import Image

    files = sorted(f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    return files, (Image.open(f).convert("RGB") for f in files)


@torch.no_grad()
def fid_between_folders(gen_dir: str, ref_dir: str, device="cuda", batch=32, size=299) -> float:
    import numpy as np
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    for real, d in ((True, ref_dir), (False, gen_dir)):
        _, it = load_images(d)
        buf = []
        for img in it:
            buf.append(torch.from_numpy(np.asarray(square_resize(img, size))).permute(2, 0, 1).float() / 255)
            if len(buf) == batch:
                fid.update(torch.stack(buf).to(device), real=real)
                buf = []
        if buf:
            fid.update(torch.stack(buf).to(device), real=real)
    return float(fid.compute())


@torch.no_grad()
def efficiency(model, example_inputs: dict, steps: int = 20, warmup: int = 2) -> dict:
    """DiT 参数量、峰值显存与 steps 次前向的延迟（单次请求，CFG 时请把 steps×2 传入）。"""
    from .utils import count_params

    dev = next(model.parameters()).device
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    for _ in range(warmup):
        model(**example_inputs, return_dict=False)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            model(**example_inputs, return_dict=False)
        end.record()
        torch.cuda.synchronize(dev)
        ms = start.elapsed_time(end)
        mem = torch.cuda.max_memory_allocated(dev) / 1024**3
    else:
        import time

        t = time.perf_counter()
        for _ in range(steps):
            model(**example_inputs, return_dict=False)
        ms, mem = (time.perf_counter() - t) * 1000, float("nan")
    return {"params_M": count_params(model) / 1e6, "peak_mem_GB": mem, f"latency_{steps}steps_s": ms / 1000}
