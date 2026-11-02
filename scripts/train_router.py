"""4.4.1 训练任务复杂度判别器并在验证集上按云端预算校准阈值。

python scripts/train_router.py --config configs/router.yaml --labels data/router/labels.jsonl --out outputs/router/default
python scripts/train_router.py ... --set router.features=struct          # 特征消融
"""

import csv
import json
import os

import numpy as np
import torch
from _common import base_parser, init

from ecflux.flux_utils import TextEncoder, masked_mean_pool
from ecflux.router.features import struct_feature_tensor
from ecflux.router.model import ComplexityRouter
from ecflux.router.train import calibrate_threshold, classification_report, roc_auc, tradeoff_curve, train_router
from ecflux.utils import get_device, get_dtype, save_json


@torch.no_grad()
def pooled_embeddings(prompts, pipeline, device, dtype, max_len, bs=16):
    te = TextEncoder(pipeline, device, dtype, max_len)
    out = []
    for i in range(0, len(prompts), bs):
        e, m = te.encode(prompts[i:i + bs], return_mask=True)
        out.append(masked_mean_pool(e, m).cpu())
    te.offload()
    return torch.cat(out)


def main():
    p = base_parser(__doc__, "configs/router.yaml")
    p.add_argument("--labels", required=True)
    p.add_argument("--emb_cache", default=None, help="缓存 pooled 嵌入的 .pt，避免重复编码")
    args = p.parse_args()
    cfg = init(args)
    out = args.out or "outputs/router/default"
    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
    prompts = [r["prompt"] for r in rows]
    y = torch.tensor([r["label"] for r in rows])
    rc = cfg["router"]

    emb = None
    if "emb" in rc["features"]:
        cache = args.emb_cache or os.path.join(out, "pooled_emb.pt")
        if os.path.exists(cache):
            emb = torch.load(cache)
        else:
            emb = pooled_embeddings(prompts, cfg["paths"]["student_pipeline"], get_device(cfg.get("device")),
                                    get_dtype(cfg.get("dtype", "bf16")), cfg["data"]["max_sequence_length"])
            torch.save(emb, cache)
    model = ComplexityRouter(emb.shape[1] if emb is not None else 0, rc["features"], rc["hidden"], rc["dropout"])
    x = model.build_input(emb, struct_feature_tensor(prompts))

    g = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(len(rows), generator=g)
    n_va = int(len(rows) * rc.get("val_ratio", 0.2))
    va, tr = perm[:n_va], perm[n_va:]
    train_router(model, x[tr], y[tr], x[va], y[va], {**rc, "seed": cfg["seed"]})
    pv = model.prob(x[va]).numpy()
    yv = y[va].numpy()
    cc = cfg["calibration"]
    tau = calibrate_threshold(pv, yv, cc.get("cloud_budget"), cc.get("objective", "f1"))
    report = {"val_auc": roc_auc(yv, pv), **classification_report(yv, pv, tau)}

    q_e = np.array([rows[i].get("hps_edge", np.nan) for i in va.tolist()])
    q_c = np.array([rows[i].get("hps_cloud", np.nan) for i in va.tolist()])
    curve = tradeoff_curve(pv, yv, q_e if not np.isnan(q_e).any() else None, q_c if not np.isnan(q_c).any() else None)
    with open(os.path.join(out, "tradeoff_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(curve[0].keys()))
        w.writeheader()
        w.writerows(curve)
    model.save(os.path.join(out, "router.pt"), extra={"tau": tau, "report": report})
    save_json(report, os.path.join(out, "report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
