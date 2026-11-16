"""阶段1 v2 汇总、指标校验与可视化。

python scripts/plot_stage1.py --root outputs/stage1_v2
读取 base/s{42,43,44}、sens/G*_r*、human_labels.json，输出：
  summary.json             每个敏感度配置的全部指标（相对未剪枝 seed 42 的变化量）
  metric_validation.json   各候选指标与人工严重程度的一致性、噪声与耗时
  fig_sens_curves.png      各指标随剪枝率变化的分组曲线（阴影为 ±1σ / ±2σ）
  fig_metric_vs_human_<labels>.png  指标对人工严重程度的散点（每个标签文件一张）
  fig_rank_table_<labels>.png      各配置在各指标下的归一化热力表（含人工标签）
  fig_traj_per_step.png    轨迹速度场误差随采样步的分布
  fig_hps_per_prompt.png   逐 prompt ΔHPS 分布
  samples/grid_p*.jpg      同 prompt、同 seed 的 G×r 样例拼图（第一列为未剪枝）
"""

import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from scipy import stats  # noqa: E402

GROUPS = ["G1", "G2", "G3", "G4", "G5", "G6"]
RATES = [0.1, 0.2, 0.3]
# 候选指标：key -> (图中名称, 是否为本次比较的主指标候选)
METRICS = {
    "vel_mse": ("velocity MSE x100 (1-step, train batches)", True),
    "traj_vel_mse": ("trajectory velocity MSE x100 (CFG, 20 steps)", True),
    "d_hps": ("HPSv2 drop (1000 prompts)", True),
    "d_hps200": ("HPSv2 drop (first 200 prompts)", True),
    "d_fid": ("FID increase (1000, D_val)", False),
    "d_clip": ("CLIP drop (1000)", False),
}


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_eval(d):
    s = json.load(open(f"{d}/eval/summary.json"))
    per = {r["idx"]: r for r in jl(f"{d}/eval/per_prompt.jsonl")}
    t = json.load(open(f"{d}/gen/timing.json")) if os.path.exists(f"{d}/gen/timing.json") else {}
    return s, per, t


def collect(root):
    base = {}
    for d in sorted(glob.glob(f"{root}/base/s*")):
        if os.path.exists(f"{d}/eval/summary.json"):
            base[int(os.path.basename(d)[1:])] = load_eval(d)
    if 42 not in base:
        raise SystemExit("need base/s42 first")
    b_s, b_per, b_t = base[42]
    idx200 = sorted(b_per)[:200]
    b_hps200 = np.mean([b_per[i]["hps"] for i in idx200])
    rows = []
    for d in sorted(glob.glob(f"{root}/sens/G*_r*")):
        if not os.path.exists(f"{d}/eval/summary.json"):
            continue
        g, r = os.path.basename(d).split("_r")
        h = json.load(open(f"{d}/history.json"))
        aux = h["rounds"][0].get("aux", {})
        s, per, t = load_eval(d)
        common = sorted(set(per) & set(b_per))
        diff = np.array([b_per[i]["hps"] - per[i]["hps"] for i in common])  # 同 prompt、同 seed 的成对差
        rows.append({
            "name": f"{g}_r{r}", "group": g, "rate": float(r), "param_ratio": h["rounds"][0]["param_ratio"],
            "vel_mse": aux.get("velocity_mse", h["rounds"][0]["score"]), "vel_mse_se": aux.get("velocity_mse_se"),
            "traj_vel_mse": aux.get("traj_velocity_mse"), "traj_vel_mse_se": aux.get("traj_velocity_mse_se"),
            "traj_per_step": aux.get("traj_velocity_mse_per_step"),
            "vel_time_s": aux.get("velocity_mse_time_s"), "traj_time_s": aux.get("traj_velocity_mse_time_s"),
            "fid": s["fid"], "clip": s["clip"], "hps": s["hps"],
            "d_fid": s["fid"] - b_s["fid"], "d_clip": b_s["clip"] - s["clip"], "d_hps": b_s["hps"] - s["hps"],
            "d_hps200": b_hps200 - np.mean([per[i]["hps"] for i in idx200]),
            "hps_paired_se": float(diff.std(ddof=1) / math.sqrt(len(diff))),
            "hps200_paired_se": float(diff[:200].std(ddof=1) / math.sqrt(200)),
            "hps_frac_worse": float((diff > 0).mean()), "hps_diff": diff.tolist(),
            "gen_s_per_img": t.get("avg_total_s"),
        })
    return base, rows, idx200


def noise(base, idx200):
    """未剪枝模型在 3 个种子下的波动（σ），用于把各指标的变化量换算成「几个 σ」。"""
    seeds = sorted(base)
    out = {"seeds": seeds}
    if len(seeds) >= 2:
        f = [base[s][0]["fid"] for s in seeds]
        c = [base[s][0]["clip"] for s in seeds]
        h = [base[s][0]["hps"] for s in seeds]
        h2 = [np.mean([base[s][1][i]["hps"] for i in idx200]) for s in seeds]
        out |= {"fid": f, "clip": c, "hps": h, "hps200": h2, "sigma": {
            "d_fid": float(np.std(f, ddof=1)), "d_clip": float(np.std(c, ddof=1)),
            "d_hps": float(np.std(h, ddof=1)), "d_hps200": float(np.std(h2, ddof=1))}}
    return out


def _auc(bad, good):
    if not bad or not good:
        return None
    return float(np.mean([(b[1] > g[1]) + 0.5 * (b[1] == g[1]) for b in bad for g in good]))


def validate(rows, labels, sig):
    """各指标与人工严重程度的一致性。sev0 = 无明显退化，sev2+ = 明显退化及以上。"""
    lab = {k: v for k, v in labels.items() if v["severity"] is not None}
    res = {}
    for m, (title, _) in METRICS.items():
        pts = [(r["name"], r[m], lab[r["name"]]["severity"], lab[r["name"]]["source"]) for r in rows
               if r["name"] in lab and r.get(m) is not None]
        if len(pts) < 4:
            continue
        x = np.array([p[1] for p in pts]); y = np.array([p[2] for p in pts])
        obs = [p for p in pts if p[3] != "inferred"]  # 直接观察（用户或助手），不含单调推断
        good = [p for p in pts if p[2] == 0]; bad = [p for p in pts if p[2] >= 2]
        auc = np.mean([(b[1] > g[1]) + 0.5 * (b[1] == g[1]) for b in bad for g in good]) if good and bad else None
        swapped = [f"{g[0]} >= {b[0]}" for b in bad for g in good if g[1] >= b[1]]
        margin = min(b[1] for b in bad) - max(g[1] for g in good) if good and bad else None
        if m in sig:
            s = sig[m]
        else:  # 速度场指标：取各配置标准误的中位数（评估集抽样误差）
            se = [r.get(f"{m}_se") for r in rows if r.get(f"{m}_se")]
            s = float(np.median(se)) if se else None
        mild = [p for p in pts if p[3] != "inferred" and 1.5 <= p[2] <= 2]  # 刚出现「明显退化」的观察点
        r = {
            "title": title, "n": len(pts),
            "spearman": float(stats.spearmanr(x, y).statistic), "kendall_tau_b": float(stats.kendalltau(x, y).statistic),
            "spearman_observed": float(stats.spearmanr([p[1] for p in obs], [p[2] for p in obs]).statistic)
            if len(obs) >= 4 else None,
            "auc_sev0_vs_sev2plus": None if auc is None else float(auc),
            "auc_sev0_vs_sev1.5plus": _auc([p for p in pts if p[2] >= 1.5], good),
            "swapped_pairs": swapped, "margin": margin, "sigma": s,
            "margin_over_sigma": None if (margin is None or not s) else margin / s,
            "mild_degradation_over_sigma": {p[0]: (p[1] / s if s else None) for p in mild},
        }
        if good and bad and margin is not None and margin > 0:
            r["threshold_midpoint"] = (min(b[1] for b in bad) + max(g[1] for g in good)) / 2
        res[m] = r
    return res


def cost(rows):
    """单次评估耗时（秒）。HPS / FID 按 generate.py 单张耗时外推（HPSv2 打分本身约 0.1s/张，忽略）。"""
    g = np.median([r["gen_s_per_img"] for r in rows if r.get("gen_s_per_img")])
    return {"vel_mse": float(np.median([r["vel_time_s"] for r in rows if r.get("vel_time_s")])),
            "traj_vel_mse": float(np.median([r["traj_time_s"] for r in rows if r.get("traj_time_s")])),
            "d_hps200": float(200 * g), "d_hps": float(1000 * g), "d_fid": float(1000 * g), "d_clip": float(1000 * g),
            "gen_s_per_img": float(g)}


def fig_curves(rows, sig, val, out):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (m, (title, _)) in zip(axes.flat, METRICS.items()):
        for g in GROUPS:
            rs = sorted((r for r in rows if r["group"] == g and r.get(m) is not None), key=lambda r: r["rate"])
            if rs:
                ax.plot([r["rate"] for r in rs], [r[m] for r in rs], marker="o", label=g)
        s = val.get(m, {}).get("sigma") or sig.get(m)
        if s:
            ax.axhspan(-s, s, color="gray", alpha=0.25, label="±1σ")
            ax.axhspan(-2 * s, 2 * s, color="gray", alpha=0.12)
        if val.get(m, {}).get("threshold_midpoint") is not None:
            ax.axhline(val[m]["threshold_midpoint"], ls="--", c="k", lw=0.8, label="sep. threshold")
        ax.set(xlabel="prune rate (single group, no recovery)", title=title, xticks=RATES)
        ax.set_yscale("symlog", linthresh=max(1e-3, (s or 0.01) * 2))  # G1 崩溃时其余组仍可读
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Per-group sensitivity, all candidate metrics (higher = worse)")
    fig.tight_layout(); fig.savefig(f"{out}/fig_sens_curves.png", dpi=140); plt.close(fig)


def fig_vs_human(rows, labels, val, out, tag):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    colors = dict(zip(GROUPS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ax, (m, (title, _)) in zip(axes.flat, METRICS.items()):
        for r in rows:
            l = labels.get(r["name"], {})
            if l.get("severity") is None or r.get(m) is None:
                continue
            mk = "^" if l["source"] == "inferred" else "o"
            jitter = (RATES.index(r["rate"]) - 1) * 0.08
            ax.scatter(l["severity"] + jitter, r[m], c=colors[r["group"]], marker=mk, s=50)
            ax.annotate(r["name"], (l["severity"] + jitter, r[m]), fontsize=7, xytext=(3, 2), textcoords="offset points")
        v = val.get(m, {})
        ax.set(xlabel="human severity (o observed, ^ inferred)", xticks=[0, 1, 1.5, 2, 2.5, 3],
               title=f"{title}\nSpearman={v.get('spearman', float('nan')):.2f}  AUC={v.get('auc_sev0_vs_sev2plus') or float('nan'):.2f}")
        ax.set_yscale("symlog", linthresh=max(1e-3, (v.get("sigma") or 0.01) * 2))
        ax.grid(alpha=0.3)
    fig.tight_layout(); fig.suptitle(f"labels: {tag}", y=1.02)
    fig.savefig(f"{out}/fig_metric_vs_human_{tag}.png", dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_rank_table(rows, labels, out, tag):
    names = [f"{g}_r{r}" for r in RATES for g in GROUPS]
    by = {r["name"]: r for r in rows}
    names = [n for n in names if n in by]
    cols = ["human"] + list(METRICS)
    mat = np.full((len(names), len(cols)), np.nan)
    for i, n in enumerate(names):
        sv = labels.get(n, {}).get("severity")
        mat[i, 0] = np.nan if sv is None else sv / 3
        for j, m in enumerate(METRICS, 1):
            vals = np.array([by[k][m] for k in names])
            mat[i, j] = stats.rankdata(vals)[i] / len(vals)  # 归一化排序，便于跨指标比较
    fig, ax = plt.subplots(figsize=(10, 0.38 * len(names) + 1.5))
    ax.imshow(mat, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    for i, n in enumerate(names):
        sv = labels.get(n, {}).get("severity")
        ax.text(0, i, "?" if sv is None else f"{sv:g}", ha="center", va="center", fontsize=8)
        for j, m in enumerate(METRICS, 1):
            ax.text(j, i, f"{by[n][m]:.3g}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols)), ["human\nseverity"] + [m for m in METRICS], fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title(f"value (text) and normalized rank (color) per config; human = {tag}")
    fig.tight_layout(); fig.savefig(f"{out}/fig_rank_table_{tag}.png", dpi=140); plt.close(fig)


def fig_traj_steps(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, rate in zip(axes, RATES):
        for g in GROUPS:
            r = next((r for r in rows if r["group"] == g and r["rate"] == rate and r.get("traj_per_step")), None)
            if r:
                ax.plot(range(1, len(r["traj_per_step"]) + 1), r["traj_per_step"], marker=".", label=g)
        ax.set(xlabel="sampling step (1 = pure noise)", title=f"r={rate}", yscale="log")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("guided velocity MSE x100"); axes[0].legend(fontsize=8)
    fig.suptitle("Trajectory velocity error per sampling step")
    fig.tight_layout(); fig.savefig(f"{out}/fig_traj_per_step.png", dpi=140); plt.close(fig)


def fig_hps_prompts(rows, labels, sig, out):
    names = [f"{g}_r{r}" for g in GROUPS for r in RATES]
    by = {r["name"]: r for r in rows}
    names = [n for n in names if n in by]
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.boxplot([by[n]["hps_diff"] for n in names], showfliers=False)
    for i, n in enumerate(names, 1):
        sv = labels.get(n, {}).get("severity")
        ax.text(i, ax.get_ylim()[1], "?" if sv is None else f"sev {sv:g}", ha="center", va="bottom", fontsize=7)
        ax.text(i, ax.get_ylim()[0], f"{100 * by[n]['hps_frac_worse']:.0f}%", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, c="k", lw=0.8)
    ax.set_xticks(range(1, len(names) + 1), names, rotation=45, fontsize=8)
    ax.set(ylabel="per-prompt HPSv2 drop vs unpruned (same seed)",
           title="Per-prompt HPSv2 drop (bottom: % prompts worse; top: human severity)")
    fig.tight_layout(); fig.savefig(f"{out}/fig_hps_per_prompt.png", dpi=140); plt.close(fig)


def sample_grids(root, rows, base, idxs, out, size=192):
    """每条 prompt 一张图：行 = G1..G6，列 = 未剪枝 / r0.1 / r0.2 / r0.3（同 seed 42）。"""
    os.makedirs(f"{out}/samples", exist_ok=True)
    path = lambda d, i: f"{d}/gen/{i:05d}_0.png"  # noqa: E731
    prompts = {r["idx"]: r["prompt"] for r in jl(f"{root}/base/s42/gen/index.jsonl")}
    pad = 18
    for i in idxs:
        W, H = size * 4, (size + pad) * len(GROUPS) + 40
        canvas = Image.new("RGB", (W + 40, H), "white")
        d = ImageDraw.Draw(canvas)
        d.text((4, 4), prompts[i][:150], fill="black")
        for gi, g in enumerate(GROUPS):
            y = 40 + gi * (size + pad)
            d.text((4, y + size // 2), g, fill="black")
            for ci, src in enumerate([f"{root}/base/s42"] + [f"{root}/sens/{g}_r{r}" for r in RATES]):
                p = path(src, i)
                if os.path.exists(p):
                    canvas.paste(Image.open(p).convert("RGB").resize((size, size)), (40 + ci * size, y))
                d.text((40 + ci * size + 2, y + size + 2), (["unpruned"] + [f"r{r}" for r in RATES])[ci], fill="black")
        canvas.save(f"{out}/samples/grid_p{i:04d}.jpg", quality=90)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/stage1_v2")
    p.add_argument("--labels", nargs="+", default=["human_labels", "visual_labels_v2"],
                   help="root 下的标签文件名（不含 .json）；第一个用于曲线图上的分界阈值")
    p.add_argument("--sample_idx", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8, 13, 21])
    a = p.parse_args()
    sets = {n: json.load(open(f"{a.root}/{n}.json"))["labels"] for n in a.labels if os.path.exists(f"{a.root}/{n}.json")}
    base, rows, idx200 = collect(a.root)
    nz = noise(base, idx200)
    sig = nz.get("sigma", {})
    cs = cost(rows) if rows else {}
    vals = {}
    for n, labels in sets.items():
        vals[n] = validate(rows, labels, sig)
        for m in vals[n]:
            vals[n][m]["eval_time_s"] = cs.get(m)
        fig_vs_human(rows, labels, vals[n], a.root, n)
        fig_rank_table(rows, labels, a.root, n)
    json.dump({"base": {s: base[s][0] for s in base}, "noise": nz,
               "rows": [{k: v for k, v in r.items() if k != "hps_diff"} for r in rows]},
              open(f"{a.root}/summary.json", "w"), indent=2)
    json.dump({"by_label_set": vals, "cost_s": cs}, open(f"{a.root}/metric_validation.json", "w"), indent=2,
              ensure_ascii=False)
    first = next(iter(sets))
    fig_curves(rows, sig, vals[first], a.root)
    fig_traj_steps(rows, a.root)
    fig_hps_prompts(rows, sets[first], sig, a.root)
    sample_grids(a.root, rows, base, a.sample_idx, a.root)
    for n, val in vals.items():
        print(f"== {n}")
        for m, v in val.items():
            print(f"{m:14s} spearman={v['spearman']:+.3f} kendall={v['kendall_tau_b']:+.3f} "
                  f"AUC={v['auc_sev0_vs_sev2plus']} margin/σ={v['margin_over_sigma']} swapped={v['swapped_pairs']}")


if __name__ == "__main__":
    main()
