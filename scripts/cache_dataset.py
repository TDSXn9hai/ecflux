"""离线缓存 VAE latents 与文本嵌入（报告 4.1：提前完成 VAE 编码与文本编码的缓存）。

阶段 1（学生）：python scripts/cache_dataset.py --source data/laion_aes65 --out data/cache/laion_aes65 \
                 --pipeline weights/FLUX.2-klein-base-4B --min_aesthetic 6.5 --max_samples 150000
阶段 2（教师文本嵌入，可选）：--stage teacher_text --pipeline weights/FLUX.2-klein-base-9B
"""

import glob
import os

import torch
from _common import base_parser, init
from torch.utils.data import DataLoader

from ecflux.data import ImageTextDataset
from ecflux.flux_utils import TextEncoder, encode_images, load_vae
from ecflux.utils import get_dtype


def main():
    p = base_parser(__doc__)
    p.add_argument("--source", help="图片目录（xxx.jpg + xxx.txt）或 jsonl")
    p.add_argument("--pipeline", required=True)
    p.add_argument("--stage", choices=["student", "teacher_text"], default="student")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--min_aesthetic", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--shard_size", type=int, default=1000)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    init(args)
    dtype = get_dtype(args.dtype)

    if args.stage == "teacher_text":
        te = TextEncoder(args.pipeline, args.device, dtype, args.max_sequence_length)
        files = sorted(glob.glob(os.path.join(args.out, "**", "*.pt"), recursive=True))
        for i in range(0, len(files), args.batch_size):
            chunk = files[i:i + args.batch_size]
            items = [torch.load(f, weights_only=False) for f in chunk]
            emb = te.encode([d["caption"] for d in items]).cpu()
            for f, d, e in zip(chunk, items, emb):
                d["teacher_prompt_embeds"] = e.to(dtype)
                torch.save(d, f)
        print(f"added teacher_prompt_embeds to {len(files)} samples")
        return

    ds = ImageTextDataset(args.source, args.resolution, args.min_aesthetic, args.max_samples)
    print(f"{len(ds)} image-text pairs")
    vae = load_vae(args.pipeline, torch.float32, args.device)
    te = TextEncoder(args.pipeline, args.device, dtype, args.max_sequence_length)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=4,
                    collate_fn=lambda b: {"pixel_values": torch.stack([x["pixel_values"] for x in b]),
                                          "caption": [x["caption"] for x in b]})
    idx = 0
    for batch in dl:
        lat = encode_images(vae, batch["pixel_values"]).cpu()
        emb = te.encode(batch["caption"]).cpu()
        for j, cap in enumerate(batch["caption"]):
            shard = os.path.join(args.out, f"{idx // args.shard_size:05d}")
            os.makedirs(shard, exist_ok=True)
            d = {"latents": lat[j].to(dtype), "prompt_embeds": emb[j].to(dtype), "caption": cap}
            torch.save(d, os.path.join(shard, f"{idx:08d}.pt"))
            idx += 1
    print(f"cached {idx} samples -> {args.out}")


if __name__ == "__main__":
    main()
