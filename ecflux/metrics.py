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

        local = os.path.join(os.path.dirname(os.path.dirname(__file__)), "weights", "clip-vit-large-patch14")
        if model_name == "openai/clip-vit-large-patch14" and os.path.isdir(local):
            model_name = local
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
    """HPS v2.1（官方 hpsv2 包的模型与预处理）。官方 img_score.score 每次调用都会重新 torch.load 整个 checkpoint，
    逐张打 1000 张图要几十分钟；这里只加载一次并按 batch 计算，结果与官方逐张打分一致（同为 amp 下的图文内积）。"""

    def __init__(self, version="v2.1", device="cuda", batch_size=16):
        import sys
        import types

        # hpsv2 1.2.0 vendors an unused ``from turtle import forward`` that
        # requires tkinter.  Provide a tiny shim so headless servers can import it.
        if "turtle" not in sys.modules:
            turtle = types.ModuleType("turtle")
            turtle.forward = lambda *args, **kwargs: None
            sys.modules["turtle"] = turtle
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

        checkpoint = os.environ.get("HPSV2_CHECKPOINT") or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "weights", "hpsv2", "HPS_v2.1_compressed.pt")
        if not os.path.isfile(checkpoint):
            import huggingface_hub
            from hpsv2.utils import hps_version_map

            checkpoint = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[version])
        # HPSv2.1_compressed.pt already contains the complete fine-tuned state_dict;
        # skip the redundant OpenCLIP base weights (architecture is fully specified by ``ViT-H-14``).
        model, _, self.preprocess = create_model_and_transforms(
            "ViT-H-14", None, precision="amp", device=device, jit=False, force_quick_gelu=False,
            force_custom_text=False, force_patch_dropout=False, force_image_size=None, pretrained_image=False,
            image_mean=None, image_std=None, light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False)
        model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
        self.model = model.to(device).eval()
        self.tokenizer = get_tokenizer("ViT-H-14")
        self.device, self.version, self.batch_size = device, version, batch_size

    @torch.no_grad()
    def __call__(self, images, prompts) -> list[float]:
        out = []
        for i in range(0, len(images), self.batch_size):
            img = torch.stack([self.preprocess(x) for x in images[i:i + self.batch_size]]).to(self.device)
            txt = self.tokenizer(list(prompts[i:i + self.batch_size])).to(self.device)
            with torch.autocast("cuda"):
                o = self.model(img, txt)
                out += (o["image_features"] * o["text_features"]).sum(-1).float().cpu().tolist()
        return out


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
