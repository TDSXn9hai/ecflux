"""数据：原始图文对读取、VAE/文本编码离线缓存、缓存数据集。

缓存格式：<cache_dir>/<shard>/<idx>.pt，每个文件是一个 dict：
    latents          (C, h, w)  归一化 patchified latents（fp16/bf16）
    prompt_embeds    (L, D)     学生文本编码器输出（可选；text_cache=false 时不存）
    caption          str
    teacher_prompt_embeds (L, D_T)  教师文本编码器输出（可选，师生文本编码器不同时使用）

data.mode=online 时不读缓存：直接读取原始图文（data.sources），训练时由 OnlineEncoder 在线做 VAE / 文本编码，
产出与缓存相同的 dict，不占用缓存磁盘（代价是训练时多加载 VAE 与文本编码器，并多一次编码前向）。
"""

import glob
import json
import logging
import os
import random

import torch
from torch.utils.data import Dataset

logger = logging.getLogger("ecflux")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


class ImageTextDataset(Dataset):
    """原始图文对。支持两种格式：
    1) 目录：xxx.jpg + 同名 xxx.txt（img2dataset 默认输出格式即为此）
    2) jsonl：每行 {"image": 路径, "caption": 文本}
    可选 min_aesthetic 过滤（jsonl 中带 aesthetic 字段，或 img2dataset 的同名 .json 中带 AESTHETIC_SCORE）。
    """

    def __init__(self, source: str, resolution: int = 512, min_aesthetic: float | None = None, max_samples=None):
        from torchvision import transforms

        self.items = []
        if source.endswith(".jsonl"):
            base = os.path.dirname(source)
            with open(source, encoding="utf-8") as f:
                for line in f:
                    d = json.loads(line)
                    if min_aesthetic is not None and d.get("aesthetic", 1e9) < min_aesthetic:
                        continue
                    img = d["image"] if os.path.isabs(d["image"]) else os.path.join(base, d["image"])
                    self.items.append((img, d["caption"]))
        else:
            for p in sorted(glob.glob(os.path.join(source, "**", "*"), recursive=True)):
                if not p.lower().endswith(IMG_EXTS):
                    continue
                stem = os.path.splitext(p)[0]
                if not os.path.exists(stem + ".txt"):
                    continue
                if min_aesthetic is not None and os.path.exists(stem + ".json"):
                    with open(stem + ".json", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("AESTHETIC_SCORE", meta.get("aesthetic", 1e9)) < min_aesthetic:
                        continue
                with open(stem + ".txt", encoding="utf-8") as f:
                    self.items.append((p, f.read().strip()))
        if max_samples:
            self.items = self.items[: int(max_samples)]
        self.tf = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        path, cap = self.items[i]
        img = Image.open(path).convert("RGB")
        return {"pixel_values": self.tf(img), "caption": cap, "path": path}


class CachedLatentDataset(Dataset):
    """读取 cache_dataset.py 生成的缓存。多个缓存目录可按比例混合（报告 4.3.1：混入约 20% 教师合成样本）。"""

    def __init__(self, roots: str | list[str], mix_ratios: list[float] | None = None, length: int | None = None,
                 seed: int = 0):
        roots = [roots] if isinstance(roots, str) else list(roots)
        self.files = [sorted(glob.glob(os.path.join(r, "**", "*.pt"), recursive=True)) for r in roots]
        for r, fs in zip(roots, self.files):
            if not fs:
                raise FileNotFoundError(f"no cached samples under {r}")
        self.mix = mix_ratios or [len(f) for f in self.files]
        s = float(sum(self.mix))
        self.mix = [m / s for m in self.mix]
        self.length = length or sum(len(f) for f in self.files)
        self.rng = random.Random(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        if len(self.files) == 1:
            path = self.files[0][i % len(self.files[0])]
        else:
            src = self.rng.choices(range(len(self.files)), weights=self.mix)[0]
            path = self.rng.choice(self.files[src])
        return torch.load(path, map_location="cpu", weights_only=False)


def collate_cached(batch: list[dict]) -> dict:
    out = {"caption": [b["caption"] for b in batch], "latents": torch.stack([b["latents"] for b in batch])}
    for k in ("prompt_embeds", "teacher_prompt_embeds"):
        if all(k in b for b in batch):
            out[k] = torch.stack([b[k] for b in batch])
    return out


def collate_raw(batch: list[dict]) -> dict:
    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "caption": [b["caption"] for b in batch]}


class MixedDataset(Dataset):
    """多个数据集按比例采样（与 CachedLatentDataset 的多目录混合规则一致）。"""

    def __init__(self, datasets: list[Dataset], mix_ratios: list[float] | None = None, seed: int = 0):
        self.sets = datasets
        mix = mix_ratios or [len(s) for s in datasets]
        self.mix = [m / float(sum(mix)) for m in mix]
        self.seed = seed

    def __len__(self):
        return sum(len(s) for s in self.sets)

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)  # 按索引取种子，DataLoader 多 worker 间不会重复采样
        s = self.sets[rng.choices(range(len(self.sets)), weights=self.mix)[0]]
        return s[rng.randrange(len(s))]


class OnlineEncoder:
    """原始图文 batch（pixel_values + caption）-> 与缓存格式相同的 batch，全程在显存中，不落盘。"""

    def __init__(self, vae, text_encoder, teacher_text_encoder=None):
        self.vae, self.te, self.tte = vae, text_encoder, teacher_text_encoder

    @torch.no_grad()
    def __call__(self, batch: dict) -> dict:
        from .flux_utils import encode_images

        caps = list(batch["caption"])
        out = {"caption": caps, "latents": encode_images(self.vae, batch["pixel_values"]),
               "prompt_embeds": self.te.encode(caps)}
        if self.tte is not None:
            out["teacher_prompt_embeds"] = self.tte.encode(caps)
        return out


def build_train_data(cfg: dict, device, dtype, teacher_pipeline: str | None = None):
    """按 data.mode 构造训练数据，返回 (dataset, encoder, collate_fn)。

    cache ：读取 cache_dataset.py 的缓存，encoder 为 None；
    online：读取 data.sources 中的原始图文，encoder 负责在线编码（teacher_pipeline 非空时额外编码教师文本嵌入）。
    """
    d = cfg["data"]
    mode = d.get("mode", "cache")
    if mode == "cache":
        return CachedLatentDataset(d["cache_dirs"], d.get("mix_ratios")), None, collate_cached
    if mode != "online":
        raise ValueError(f"unknown data.mode {mode}")
    from .flux_utils import TextEncoder, load_vae

    srcs = d["sources"]
    srcs = [srcs] if isinstance(srcs, str) else list(srcs)
    sets = [ImageTextDataset(s, d.get("resolution", 512), d.get("min_aesthetic"), d.get("max_samples")) for s in srcs]
    for s, ds in zip(srcs, sets):
        if len(ds) == 0:
            raise FileNotFoundError(f"no image-text pairs under {s}")
    ds = sets[0] if len(sets) == 1 else MixedDataset(sets, d.get("mix_ratios"), seed=int(cfg.get("seed", 0)))
    edev = d.get("encoder_device") or device
    pipe, seq = cfg["paths"]["student_pipeline"], d["max_sequence_length"]
    vae = load_vae(pipe, torch.float32, edev)
    te = TextEncoder(pipe, edev, dtype, seq)
    tte = TextEncoder(teacher_pipeline, edev, dtype, seq) if teacher_pipeline else None
    logger.info(f"online data: {len(ds)} image-text pairs from {srcs}, encoders on {edev}")
    return ds, OnlineEncoder(vae, te, tte), collate_raw


class PromptDataset(Dataset):
    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, i):
        return self.prompts[i]


class SyntheticCachedDataset(Dataset):
    """随机张量数据，仅用于冒烟测试（与缓存格式一致）。"""

    def __init__(self, n=8, channels=16, hw=4, seq_len=8, embed_dim=32, teacher_embed_dim=None, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = []
        for i in range(n):
            d = {
                "latents": torch.randn(channels, hw, hw, generator=g),
                "prompt_embeds": torch.randn(seq_len, embed_dim, generator=g),
                "caption": f"a photo of {i} red cubes on the left of a blue sphere",
            }
            if teacher_embed_dim:
                d["teacher_prompt_embeds"] = torch.randn(seq_len, teacher_embed_dim, generator=g)
            self.items.append(d)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]
