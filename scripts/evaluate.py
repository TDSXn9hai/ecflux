"""表2 指标评估：CLIP Score / HPSv2（逐 prompt 输出，供路由标签使用）、FID、DiT 效率。

python scripts/evaluate.py --images outputs/gen/full --metrics clip hps --out outputs/eval/full
python scripts/evaluate.py --images outputs/gen/full --ref_dir data/coco30k --metrics fid --out outputs/eval/full
python scripts/evaluate.py --transformer outputs/distill/full/final --metrics efficiency --out outputs/eval/full
GenEval / T2I-CompBench 请用 generate.py --format geneval/t2icompbench 导出后运行官方评测脚本。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ecflux.utils import get_dtype, save_json, setup_logging  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", help="generate.py 输出目录（含 index.jsonl）")
    p.add_argument("--ref_dir", help="FID 参考图目录")
    p.add_argument("--transformer", help="效率评估的 DiT（管线目录或剪枝目录）")
    p.add_argument("--metrics", nargs="+", default=["clip", "hps"], choices=["clip", "hps", "fid", "efficiency"])
    p.add_argument("--clip_model", default="openai/clip-vit-large-patch14")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--text_len", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    setup_logging()
    os.makedirs(a.out, exist_ok=True)
    summary = {}

    if {"clip", "hps"} & set(a.metrics):
        from PIL import Image

        from ecflux.metrics import ClipScorer, HPSv2Scorer

        index = [json.loads(l) for l in open(os.path.join(a.images, "index.jsonl"), encoding="utf-8")]
        scorers = {}
        if "clip" in a.metrics:
            scorers["clip"] = ClipScorer(a.clip_model, a.device)
        if "hps" in a.metrics:
            scorers["hps"] = HPSv2Scorer()
        per = {}
        for r in index:
            img = Image.open(r["path"]).convert("RGB")
            d = per.setdefault(r["idx"], {"prompt": r["prompt"]})
            for name, s in scorers.items():
                d.setdefault(name, []).append(s([img], [r["prompt"]])[0])
        rows = []
        for idx in sorted(per):
            d = per[idx]
            rows.append({"idx": idx, "prompt": d["prompt"], **{k: sum(d[k]) / len(d[k]) for k in scorers}})
        with open(os.path.join(a.out, "per_prompt.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for k in scorers:
            summary[k] = sum(r[k] for r in rows) / len(rows)

    if "fid" in a.metrics:
        from ecflux.metrics import fid_between_folders

        summary["fid"] = fid_between_folders(a.images, a.ref_dir, a.device)

    if "efficiency" in a.metrics:
        from ecflux.flux_utils import load_transformer, prepare_latent_ids, prepare_text_ids
        from ecflux.metrics import efficiency

        m = load_transformer(a.transformer, dtype=get_dtype(a.dtype)).to(a.device).eval()
        hw = a.resolution // 16
        lat = torch.zeros(1, m.config.in_channels, hw, hw, device=a.device)
        emb = torch.zeros(1, a.text_len, m.config.joint_attention_dim, device=a.device, dtype=get_dtype(a.dtype))
        inputs = {"hidden_states": lat.flatten(2).transpose(1, 2).to(emb.dtype),
                  "timestep": torch.full((1,), 0.5, device=a.device, dtype=emb.dtype),
                  "encoder_hidden_states": emb, "img_ids": prepare_latent_ids(lat).to(a.device),
                  "txt_ids": prepare_text_ids(emb).to(a.device)}
        summary["efficiency"] = efficiency(m, inputs, steps=a.steps)

    save_json(summary, os.path.join(a.out, "summary.json"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
