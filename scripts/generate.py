"""文生图推理（原始 4B / 剪枝 / 蒸馏 / 9B 教师），可按 GenEval 或 T2I-CompBench 目录格式导出。

python scripts/generate.py --pipeline weights/FLUX.2-klein-base-4B --transformer outputs/distill/full/final \
    --prompts data/prompts.txt --out outputs/gen/full --steps 20
python scripts/generate.py ... --format geneval --prompts geneval/prompts/evaluation_metadata.jsonl
多卡分片：每张卡 --shard_id i --num_shards n（文件名保持全局下标），全部结束后 --merge --out <同一目录> 合并 index / timing
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ecflux.generation import T2IGenerator  # noqa: E402
from ecflux.utils import get_dtype, read_prompts, save_json, seed_everything, setup_logging  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline", required=True, help="提供 text_encoder / vae / scheduler 的管线目录")
    p.add_argument("--transformer", default=None, help="剪枝或蒸馏后的 DiT 目录；缺省使用管线自带 DiT")
    p.add_argument("--prompts", required=True, help=".txt / .jsonl（geneval 格式为 evaluation_metadata.jsonl）")
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["plain", "geneval", "t2icompbench"], default="plain")
    p.add_argument("--n_per_prompt", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="只合并 --out 下各分片的 index / timing，不生成")
    a = p.parse_args()
    setup_logging()
    if a.merge:
        return merge_shards(a.out)
    seed_everything(a.seed)
    if a.format == "geneval":
        metas = [json.loads(l) for l in open(a.prompts, encoding="utf-8") if l.strip()]
        prompts = [m["prompt"] for m in metas]
    else:
        prompts, metas = read_prompts(a.prompts), None
    prompts = prompts[: a.limit] if a.limit else prompts
    per = -(-len(prompts) // a.num_shards)  # 连续切分，idx 保持全局下标
    lo, hi = a.shard_id * per, min(len(prompts), (a.shard_id + 1) * per)
    suffix = f"_shard{a.shard_id}" if a.num_shards > 1 else ""

    gen = T2IGenerator(a.pipeline, a.transformer, a.device, get_dtype(a.dtype), a.max_sequence_length)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    os.makedirs(a.out, exist_ok=True)
    timings, index = [], []
    for i in range(lo, hi, a.batch_size):
        chunk = prompts[i:min(i + a.batch_size, hi)]
        for k in range(a.n_per_prompt):
            imgs, t = gen(chunk, a.height, a.width, a.steps, a.guidance_scale, seed=a.seed + k, return_timing=True)
            timings.append(t)
            for j, (pr, img) in enumerate(zip(chunk, imgs)):
                idx = i + j
                if a.format == "geneval":  # <out>/<idx:05d>/metadata.jsonl + samples/<k:04d>.png
                    d = os.path.join(a.out, f"{idx:05d}")
                    os.makedirs(os.path.join(d, "samples"), exist_ok=True)
                    with open(os.path.join(d, "metadata.jsonl"), "w", encoding="utf-8") as f:
                        f.write(json.dumps(metas[idx], ensure_ascii=False) + "\n")
                    path = os.path.join(d, "samples", f"{k:04d}.png")
                elif a.format == "t2icompbench":  # <out>/samples/<prompt>_<idx:06d>.png
                    os.makedirs(os.path.join(a.out, "samples"), exist_ok=True)
                    path = os.path.join(a.out, "samples", f"{pr}_{idx * a.n_per_prompt + k:06d}.png")
                else:
                    path = os.path.join(a.out, f"{idx:05d}_{k}.png")
                img.save(path)
                index.append({"idx": idx, "k": k, "prompt": pr, "path": path})
    with open(os.path.join(a.out, f"index{suffix}.jsonl"), "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = max(1, len(timings))
    save_json({"avg_" + k: sum(t[k] for t in timings) / n for k in timings[0]} | {
        "peak_mem_GB": torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None,
        "steps": a.steps, "resolution": [a.height, a.width], "batch_size": a.batch_size, "n_calls": len(timings)},
        os.path.join(a.out, f"timing{suffix}.json"))
    print(f"generated {len(index)} images -> {a.out}")


def merge_shards(out: str):
    idx_files = sorted(glob.glob(os.path.join(out, "index_shard*.jsonl")))
    if not idx_files:
        raise FileNotFoundError(f"no index_shard*.jsonl under {out}")
    rows = [json.loads(l) for f in idx_files for l in open(f, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: (r["idx"], r["k"]))
    with open(os.path.join(out, "index.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ts = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(out, "timing_shard*.json")))]
    n = sum(t["n_calls"] for t in ts)
    merged = {k: sum(t[k] * t["n_calls"] for t in ts) / n for k in ts[0] if k.startswith("avg_")}
    merged |= {"peak_mem_GB": max(t["peak_mem_GB"] or 0 for t in ts), "steps": ts[0]["steps"],
               "resolution": ts[0]["resolution"], "batch_size": ts[0]["batch_size"], "n_calls": n, "shards": len(ts)}
    save_json(merged, os.path.join(out, "timing.json"))
    print(f"merged {len(idx_files)} shards, {len(rows)} images -> {out}/index.jsonl")


if __name__ == "__main__":
    main()
