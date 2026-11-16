"""离线缓存 VAE latents 与文本嵌入（报告 4.1：提前完成 VAE 编码与文本编码的缓存）。

阶段 1（学生）：python scripts/cache_dataset.py --source data/laion_aes65 --out data/cache/laion_aes65 \
                 --pipeline weights/FLUX.2-klein-base-4B --min_aesthetic 6.5 --max_samples 150000
阶段 2（教师文本嵌入，可选）：--stage teacher_text --pipeline weights/FLUX.2-klein-base-9B
"""

import glob
import os

import torch
from _common import base_parser, init
from torch.utils.data import DataLoader, Dataset

from ecflux.data import ImageTextDataset
from ecflux.flux_utils import TextEncoder, encode_images, load_vae
from ecflux.utils import get_dtype


class IndexedDataset(Dataset):
    """Apply a global sample index to a subset without changing cache filenames."""

    def __init__(self, dataset: Dataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position: int) -> dict:
        index = self.indices[position]
        item = self.dataset[index]
        item["_index"] = index
        return item


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
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    init(args)
    dtype = get_dtype(args.dtype)

    if args.stage == "teacher_text":
        te = TextEncoder(args.pipeline, args.device, dtype, args.max_sequence_length)
        files = sorted(glob.glob(os.path.join(args.out, "**", "*.pt"), recursive=True))
        files = [
            path for index, path in enumerate(files)
            if index % args.num_shards == args.shard_id
        ]
        todo = []
        for f in files:
            d = torch.load(f, map_location="cpu", weights_only=False)
            if args.overwrite or "teacher_prompt_embeds" not in d:
                todo.append(f)
        for i in range(0, len(files), args.batch_size):
            chunk = todo[i:i + args.batch_size]
            if not chunk:
                continue
            items = [torch.load(f, weights_only=False) for f in chunk]
            emb = te.encode([d["caption"] for d in items]).cpu()
            for f, d, e in zip(chunk, items, emb):
                d["teacher_prompt_embeds"] = e.to(dtype).clone()
                torch.save(d, f)
        print(f"added teacher_prompt_embeds to {len(todo)} samples")
        return

    ds = ImageTextDataset(args.source, args.resolution, args.min_aesthetic, args.max_samples)
    existing_paths = [] if args.overwrite else glob.glob(os.path.join(args.out, "**", "*.pt"), recursive=True)
    existing = {int(os.path.splitext(os.path.basename(path))[0]) for path in existing_paths}
    indices = [
        index for index in range(len(ds))
        if index % args.num_shards == args.shard_id and index not in existing
    ]
    print(
        f"{len(indices)} image-text pairs remaining "
        f"(source={len(ds)}, shard={args.shard_id}/{args.num_shards}, existing={len(existing)})"
    )
    if not indices:
        return
    indexed = IndexedDataset(ds, indices)
    vae = load_vae(args.pipeline, torch.float32, args.device)
    te = TextEncoder(args.pipeline, args.device, dtype, args.max_sequence_length)
    dl = DataLoader(indexed, batch_size=args.batch_size, num_workers=args.num_workers,
                    collate_fn=lambda b: {"pixel_values": torch.stack([x["pixel_values"] for x in b]),
                                          "caption": [x["caption"] for x in b],
                                          "index": [x["_index"] for x in b]})
    done = 0
    for batch in dl:
        lat = encode_images(vae, batch["pixel_values"]).cpu()
        emb = te.encode(batch["caption"]).cpu()
        for j, (idx, cap) in enumerate(zip(batch["index"], batch["caption"])):
            shard = os.path.join(args.out, f"{idx // args.shard_size:05d}")
            os.makedirs(shard, exist_ok=True)
            # clone() is required: slice views otherwise serialize the whole batch
            # storage in every .pt file (31 MB instead of ~2.2 MB for L=128).
            d = {
                "latents": lat[j].to(dtype).clone(),
                "prompt_embeds": emb[j].to(dtype).clone(),
                "caption": cap,
            }
            torch.save(d, os.path.join(shard, f"{idx:08d}.pt"))
            done += 1
        if done % (args.batch_size * 10) < args.batch_size:
            print(f"cached +{done}/{len(indices)} samples -> {args.out}", flush=True)
    print(f"cached +{done}/{len(indices)} samples -> {args.out}")


if __name__ == "__main__":
    main()
