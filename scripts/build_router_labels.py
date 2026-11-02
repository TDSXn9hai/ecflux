"""4.4.1 构造复杂度标签：端侧学生与云端教师在同一批 prompt 上的质量差距。

python scripts/build_router_labels.py --edge outputs/eval/edge/per_prompt.jsonl \
    --cloud outputs/eval/cloud/per_prompt.jsonl --config configs/router.yaml --out data/router
可选 --geneval_edge / --geneval_cloud：GenEval 官方 results.jsonl（含 correct 字段），按 prompt 取平均正确率。
"""

import json
import os
from collections import defaultdict

import numpy as np
from _common import base_parser, init

from ecflux.router.labels import make_labels


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def geneval_by_prompt(path):
    acc = defaultdict(list)
    for r in read_jsonl(path):
        acc[r["prompt"]].append(float(r["correct"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def main():
    p = base_parser(__doc__, "configs/router.yaml")
    p.add_argument("--edge", required=True)
    p.add_argument("--cloud", required=True)
    p.add_argument("--geneval_edge")
    p.add_argument("--geneval_cloud")
    args = p.parse_args()
    cfg = init(args)
    edge = {r["prompt"]: r for r in read_jsonl(args.edge)}
    cloud = {r["prompt"]: r for r in read_jsonl(args.cloud)}
    ge = geneval_by_prompt(args.geneval_edge) if args.geneval_edge else {}
    gc = geneval_by_prompt(args.geneval_cloud) if args.geneval_cloud else {}
    recs = []
    for prompt in edge:
        if prompt not in cloud:
            continue
        r = {"prompt": prompt}
        for m in ("clip", "hps"):
            if m in edge[prompt] and m in cloud[prompt]:
                r[f"{m}_edge"], r[f"{m}_cloud"] = edge[prompt][m], cloud[prompt][m]
        if prompt in ge and prompt in gc:
            r["geneval_edge"], r["geneval_cloud"] = ge[prompt], gc[prompt]
        recs.append(r)
    lc = cfg["labels"]
    y, gap = make_labels(recs, lc.get("weights"), lc.get("norm", "zscore"), lc.get("threshold", 0.0),
                         lc.get("threshold_mode", "value"))
    out = args.out or "data/router"
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "labels.jsonl"), "w", encoding="utf-8") as f:
        for r, yi, gi in zip(recs, y, gap):
            f.write(json.dumps({**r, "gap": float(gi), "label": int(yi)}, ensure_ascii=False) + "\n")
    print(f"{len(recs)} prompts, complex ratio={np.mean(y):.3f} -> {out}/labels.jsonl")


if __name__ == "__main__":
    main()
