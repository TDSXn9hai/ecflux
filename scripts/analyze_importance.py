"""阶段1.1：三粒度重要性统计与可视化（不剪枝）。

python scripts/analyze_importance.py --config configs/prune.yaml --out outputs/stage1/importance --seeds 0 1
产物：importance_seed{s}.pt、summary.json、heads.png、text_ratio.png、neurons.png、block_sim.png
"""

import itertools
import os

import torch
from _common import base_parser, init
from torch.utils.data import DataLoader

from ecflux.data import build_train_data
from ecflux.flux_utils import load_transformer, prepare_fm_batch
from ecflux.pruning.importance import collect_importance
from ecflux.pruning.pruner import DEFAULT_GROUPS, group_of
from ecflux.utils import get_device, get_dtype, save_json

TAUS = (0.88, 0.90, 0.92, 0.94)


def calib_batches(ds, encoder, collate, n, bs, device, dtype, seed, tcfg):
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed),
                    collate_fn=collate, num_workers=2, drop_last=True)
    return [prepare_fm_batch(b if encoder is None else encoder(b), device, dtype, g, tcfg)
            for b in itertools.islice(dl, n)]


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = a.float().argsort().argsort().float(), b.float().argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))


def keystr(k) -> str:
    return ":".join(str(x) for x in k)


def to_cpu(imp: dict) -> dict:
    out = {}
    for name, d in imp.items():
        out[name] = {keystr(k): (v.detach().float().cpu() if torch.is_tensor(v) else float(v)) for k, v in d.items()}
    return out


def summarize(imps: list[dict], groups: list, pcfg: dict) -> dict:
    imp = imps[0]
    thr = float(pcfg.get("cross_ratio_threshold", 1.0))
    heads = {}
    for k, s in imp["heads"].items():
        r = imp["head_text_ratio"].get(k)
        heads[k] = {"score_mean": float(s.mean()), "score_cv": float(s.std() / s.mean().clamp_min(1e-12)),
                    "min_over_max": float(s.min() / s.max().clamp_min(1e-12))}
        if r is not None:
            heads[k] |= {"n_cross": int((r > thr).sum()), "n_heads": int(r.numel()),
                         "text_ratio_mean": float(r.mean())}
    neurons = {}
    for k, s in imp["neurons"].items():
        kind, orig = k.split(":")[:2]
        n = s / s.max().clamp_min(1e-12)
        neurons[k] = {"group": group_of(groups, kind, int(orig)), "n": int(s.numel()),
                      "frac_below_0.05max": float((n < 0.05).float().mean()), "frac_below_0.1max": float((n < 0.1).float().mean()),
                      "q10_over_median": float(s.quantile(0.1) / s.median().clamp_min(1e-12))}
    sims = {k: float(v) for k, v in imp["block_sim"].items()}
    out = {"heads": heads, "neurons": neurons, "block_sim": sims,
           "blocks_over_tau": {str(t): [k for k, v in sims.items() if v > t] for t in TAUS},
           "cross_ratio_threshold": thr}
    if len(imps) > 1:
        a, b = imps[0], imps[1]
        hs = {k: spearman(a["heads"][k], b["heads"][k]) for k in a["heads"]}
        ns = {k: spearman(a["neurons"][k], b["neurons"][k]) for k in a["neurons"]}
        rs = {k: spearman(a["head_text_ratio"][k], b["head_text_ratio"][k]) for k in a["head_text_ratio"]}
        stat = lambda d: {"mean": sum(d.values()) / len(d), "min": min(d.values()), "per_layer": d}  # noqa: E731
        out["seed_stability_spearman"] = {"heads": stat(hs), "neurons": stat(ns), "head_text_ratio": stat(rs)}
    return out


def order_key(k: str):
    kind, orig = k.split(":")[:2]
    return (kind != "double", int(orig), k)


def plot(imp: dict, summary: dict, out: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 1) 注意力头梯度敏感度热图（每个 block 内按最大值归一化）
    keys = sorted(imp["heads"], key=order_key)
    mat = np.stack([(imp["heads"][k] / imp["heads"][k].max().clamp_min(1e-12)).numpy() for k in keys])
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(keys)), keys, fontsize=6)
    ax.set_xlabel("head index")
    ax.set_title("head grad sensitivity (normalized per block)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "heads.png"), dpi=150)
    plt.close(fig)

    # 2) 文本注意力占比直方图
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in ("double", "single"):
        v = [imp["head_text_ratio"][k] for k in imp["head_text_ratio"] if k.startswith(kind)]
        if v:
            ax.hist(np.log10(torch.cat(v).clamp_min(1e-6).numpy()), bins=60, alpha=0.6, label=kind)
    ax.axvline(np.log10(summary["cross_ratio_threshold"]), color="k", ls="--", label="threshold")
    ax.set_xlabel("log10(text attention mass / uniform)")
    ax.set_ylabel("#heads")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "text_ratio.png"), dpi=150)
    plt.close(fig)

    # 3) 各组 FFN 激活分布（按层最大值归一化后合并）
    by_g = {}
    for k, s in imp["neurons"].items():
        g = summary["neurons"][k]["group"] or "none"
        by_g.setdefault(g, []).append((s / s.max().clamp_min(1e-12)).numpy())
    names = sorted(by_g)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([np.concatenate(by_g[g]) for g in names], showfliers=False)
    ax.set_xticks(range(1, len(names) + 1), names)
    ax.set_ylabel("neuron activation L1 / layer max")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "neurons.png"), dpi=150)
    plt.close(fig)

    # 4) 相邻 Single Block 相似度
    sims = sorted(summary["block_sim"].items(), key=lambda kv: int(kv[0].split(":")[0]))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([k for k, _ in sims], [v for _, v in sims], marker="o")
    for t in TAUS:
        ax.axhline(t, ls=":", lw=0.8, color="gray")
    ax.set_xlabel("single block pair (a:b)")
    ax.set_ylabel("cosine similarity")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "block_sim.png"), dpi=150)
    plt.close(fig)


def main():
    p = base_parser(__doc__, "configs/prune.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="校准数据种子；两个以上时计算排序稳定性")
    args = p.parse_args()
    cfg = init(args)
    out = args.out or os.path.join(cfg["paths"]["output_root"], "stage1", "importance")
    device, dtype = get_device(cfg.get("device")), get_dtype(cfg.get("dtype", "bf16"))
    groups = cfg.get("groups") or DEFAULT_GROUPS
    pcfg, c = cfg["prune"], cfg["calib"]

    model = load_transformer(cfg["paths"]["student_pipeline"], dtype=dtype).to(device).eval().requires_grad_(False)
    ds, encoder, collate = build_train_data(cfg, device, dtype)
    all_batches = [calib_batches(ds, encoder, collate, c["num_batches"], c["batch_size"], device, dtype, s,
                                 cfg.get("timesteps")) for s in args.seeds]
    del encoder
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    imps = []
    for s, batches in zip(args.seeds, all_batches):
        imp = to_cpu(collect_importance(model, batches, pcfg, seed=s))
        torch.save(imp, os.path.join(out, f"importance_seed{s}.pt"))
        imps.append(imp)
    summary = summarize(imps, groups, pcfg)
    save_json(summary, os.path.join(out, "summary.json"))
    plot(imps[0], summary, out)
    print(f"blocks over tau: { {t: len(v) for t, v in summary['blocks_over_tau'].items()} }; saved to {out}")
    if "seed_stability_spearman" in summary:
        st = summary["seed_stability_spearman"]
        print({k: round(v["mean"], 3) for k, v in st.items()})


if __name__ == "__main__":
    main()
