"""阶段2 v2 汇总与可视化。

python scripts/plot_stage2.py --root outputs/stage2_v2
读取 <root>/<run>/{history.json, eval_dval/summary.json, eff_512/summary.json, gen_dval/}，以及 stage1_v2 的未剪枝基线，输出：
  summary.md / summary.json     每个实验的参数量、删块、APRA 轮次、双闸门指标（aux_final）、D_val FID / CLIP / HPSv2、延迟
  fig_apra_<run>.png            APRA 轨迹：每轮 γ_g、两个闸门指标的 Δ（含阈值）、参数比例
  fig_pareto.png                参数量 vs 质量（轨迹速度场 MSE / D_val HPSv2 下降 / D_val FID 上升）
  fig_ablations.png             各消融维度的对比柱状图（误差线为 full 与 full_rep 估计的运行间 σ）
  fig_layers.png                逐层保留头数 / FFN 宽度比例热力图（被删除的 Single Block 记 0）
  samples/compare_p*.jpg        同 prompt、同 seed 的样例对比
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

GATES = ("traj_velocity_mse", "hps")
GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5", "G6"]
# 消融维度 -> 参与比较的实验（full 为主方法）
ABLATIONS = {
    "APRA": ["full", "fixed_ratio", "uniform_rate"],
    "head criterion": ["full", "head_grad_only", "head_mag", "random"],
    "neuron criterion": ["full", "neuron_act"],
    "modality-aware": ["full", "no_modality"],
    "block stage": ["full", "block_before"],
    "block tau": ["tau_0.88", "tau_0.90", "full", "tau_0.94"],
    "granularity": ["full", "no_block", "neuron_only", "head_only"],
}
SAMPLE_RUNS = ["full", "fixed_ratio", "uniform_rate", "random", "no_modality", "head_only"]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def load_runs(root):
    runs = {}
    for hp in sorted(glob.glob(f"{root}/*/history.json")):
        d = os.path.dirname(hp)
        n = os.path.basename(d)
        if n.startswith(("calib_", "_dry")):
            continue
        h = json.load(open(hp))
        ev = jload(f"{d}/eval_dval/summary.json") or {}
        ef = (jload(f"{d}/eff_512/summary.json") or {}).get("efficiency", {})
        fin = h.get("aux_final", {})
        p = h.get("params_final") or h.get("efficiency", {}).get("params_M", 0) * 1e6
        runs[n] = {
            "name": n, "dir": d, "history": h, "params_B": p / 1e9, "param_ratio": p / h["params_orig"],
            "removed": h.get("removed_single_blocks", []),
            "accepted": sum(not r.get("rejected") for r in h["rounds"]), "n_rounds": len(h["rounds"]),
            "traj": fin.get("traj_velocity_mse"), "hps200_drop": fin.get("hps"),
            "fid": ev.get("fid"), "clip": ev.get("clip"), "hps": ev.get("hps"),
            "latency_s": ef.get("latency_40steps_s"), "peak_mem_GB": ef.get("peak_mem_GB"),
        }
    return runs


def load_base(stage1_root):
    b = jload(f"{stage1_root}/base/s42/eval/summary.json")
    s1 = jload(f"{stage1_root}/summary.json") or {}
    return b, s1.get("noise", {}).get("sigma", {})


def add_deltas(runs, base):
    for r in runs.values():
        if base and r["fid"] is not None:
            r["d_fid"] = r["fid"] - base["fid"]
            r["d_hps"] = base["hps"] - r["hps"]
            r["d_clip"] = base["clip"] - r["clip"]


def run_sigma(runs):
    """full 与 full_rep 只差恢复训练种子：两次结果之差 / √2 作为单次运行的标准差估计（只有两个样本，很粗）。"""
    a, b = runs.get("full"), runs.get("full_rep")
    if not (a and b):
        return {}
    out = {}
    for k in ("traj", "hps200_drop", "d_fid", "d_hps", "d_clip", "params_B"):
        if a.get(k) is not None and b.get(k) is not None:
            out[k] = abs(a[k] - b[k]) / math.sqrt(2)
    return out


def write_summary(runs, base, sig_seed, sig_run, root):
    head = ("| 实验 | DiT 参数 | 参数比例 | 删除的 Single Block | APRA 接受/总 | 轨迹速度场 MSE | HPSv2 下降（200） | "
            "D_val FID (Δ) | CLIP (Δ) | HPSv2 1000 (Δ) | 40 次前向 @512 |")
    rows = [head, "|" + "---|" * 11]
    f = lambda v, fmt: "-" if v is None else format(v, fmt)  # noqa: E731
    order = ["full", "full_rep"] + sorted(n for n in runs if n not in ("full", "full_rep"))
    for n in order:
        if n not in runs:
            continue
        r = runs[n]
        fid = "-" if r["fid"] is None else f'{r["fid"]:.2f} ({r["d_fid"]:+.2f})'
        clip = "-" if r["clip"] is None else f'{r["clip"]:.2f} ({-r["d_clip"]:+.2f})'
        hps = "-" if r["hps"] is None else f'{r["hps"]:.4f} ({-r["d_hps"]:+.4f})'
        rows.append(f'| {n} | {r["params_B"]:.3f}B | {r["param_ratio"]:.3f} | {r["removed"]} | {r["accepted"]}/{r["n_rounds"]} | '
                    f'{f(r["traj"], ".2f")} | {f(r["hps200_drop"], ".4f")} | {fid} | {clip} | {hps} | {f(r["latency_s"], ".2f")} |')
    txt = ["未剪枝 4B：3.876B" + (f'，D_val FID {base["fid"]:.2f} / CLIP {base["clip"]:.2f} / HPSv2 {base["hps"]:.4f}'
                                  f'（seed 42；种子 σ：FID {sig_seed.get("d_fid", float("nan")):.2f}，'
                                  f'HPSv2 {sig_seed.get("d_hps", float("nan")):.4f}）' if base else ""),
           "轨迹速度场 MSE 与 HPSv2 下降（200）为最终模型（含块剪枝与恢复训练）相对未剪枝模型的值，取自 history.json 的 aux_final。",
           ""]
    if sig_run:
        txt.append("运行间 σ（full vs full_rep，仅两个样本）：" + "，".join(f"{k} {v:.4g}" for k, v in sig_run.items()))
        txt.append("")
    open(f"{root}/summary.md", "w", encoding="utf-8").write("\n".join(txt + rows) + "\n")
    keep = {n: {k: v for k, v in r.items() if k not in ("history", "dir")} for n, r in runs.items()}
    json.dump({"base": base, "sigma_seed": sig_seed, "sigma_run": sig_run, "runs": keep},
              open(f"{root}/summary.json", "w"), indent=2)
    print("\n".join(txt + rows))


def fig_apra(r, gates, root):
    h = r["history"]
    rounds = h["rounds"]
    if not rounds:
        return
    t = [x["round"] for x in rounds]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for g in GROUP_ORDER:
        if g in rounds[0]["rates"]:
            axes[0].plot(t, [x["rates"][g] for x in rounds], marker="o", label=g)
    axes[0].set(xlabel="APRA round", ylabel="prune rate γ_g (used in round)", title="group rates")
    axes[0].legend(fontsize=8)
    gate_names = list((gates or {}).keys()) or ["traj_velocity_mse", "hps"]
    titles = {"traj_ref": "Δ trajectory velocity MSE x100 (vs recovered ref)",
              "hps_ref": "Δ HPSv2 drop (200 prompts, vs recovered ref)",
              "traj_velocity_mse": "Δ trajectory velocity MSE x100 (vs original)",
              "hps": "Δ HPSv2 drop (200 prompts, vs original)"}
    for ax, m in zip((axes[1], axes[2]), gate_names[:2]):
        title = titles.get(m, m)
        d = [x.get("gate_deltas", {}).get(m) for x in rounds]
        cum = [x.get("gate_cumulative", {}).get(m) for x in rounds]
        if any(v is not None for v in d):
            if any(v is not None for v in cum):
                ax.plot(t, cum, marker="s", c="tab:purple", label="cumulative since round 0")
                cap = (gates or {}).get(m, {}).get("high")
                if cap:
                    ax.axhline(cap, c="tab:purple", ls="-.", lw=1, label="cumulative cap")
            c = ["tab:red" if x.get("rejected") else "tab:green" for x in rounds]
            ax.bar(t, [v or 0 for v in d], color=c)
            g = (gates or {}).get(m)
            if g:
                ax.axhline(g["low"], ls=":", c="k", lw=1, label="low")
                ax.axhline(g["high"], ls="--", c="k", lw=1, label="high")
                ax.legend(fontsize=8)
        ax.set(xlabel="APRA round (red = rolled back)", title=title)
    axes[3].plot([0] + t, [1.0] + [x["param_ratio"] for x in rounds], marker="o", label="after round")
    acc, cur = [1.0], 1.0
    for x in rounds:
        cur = cur if x.get("rejected") else x["param_ratio"]
        acc.append(cur)
    axes[3].plot([0] + t, acc, marker="s", ls="--", label="kept (after rollback)")
    axes[3].axhline(r["param_ratio"], c="gray", ls=":", label=f"final w/ blocks {r['param_ratio']:.3f}")
    axes[3].set(xlabel="APRA round", ylabel="DiT param ratio", title="compression")
    axes[3].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"APRA trajectory: {r['name']}")
    fig.tight_layout(); fig.savefig(f"{root}/fig_apra_{r['name']}.png", dpi=130); plt.close(fig)


def fig_pareto(runs, sig_run, root):
    panels = [("traj", "final trajectory velocity MSE x100"), ("d_hps", "D_val HPSv2 drop (1000)"),
              ("d_fid", "D_val FID increase (1000)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (k, title) in zip(axes, panels):
        for n, r in runs.items():
            if r.get(k) is None:
                continue
            main = n in ("full", "full_rep")
            ax.errorbar(r["params_B"], r[k], yerr=sig_run.get(k), fmt="*" if main else "o",
                        ms=12 if main else 6, c="tab:red" if main else "tab:blue", capsize=2)
            ax.annotate(n, (r["params_B"], r[k]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set(xlabel="DiT params (B)", ylabel=title + " (lower = better)", title=title)
        ax.grid(alpha=0.3)
    fig.suptitle("Compression vs quality (error bar = run-to-run σ from full / full_rep)")
    fig.tight_layout(); fig.savefig(f"{root}/fig_pareto.png", dpi=130); plt.close(fig)


def fig_ablations(runs, sig_run, root):
    metrics = [("traj", "traj vel MSE"), ("d_hps", "HPSv2 drop (D_val)"), ("d_fid", "FID increase (D_val)"),
               ("params_B", "params (B)")]
    dims = [(k, [n for n in v if n in runs]) for k, v in ABLATIONS.items()]
    dims = [(k, v) for k, v in dims if len(v) >= 2]
    if not dims:
        return
    fig, axes = plt.subplots(len(dims), len(metrics), figsize=(4.2 * len(metrics), 2.6 * len(dims)), squeeze=False)
    for i, (dim, names) in enumerate(dims):
        for j, (k, title) in enumerate(metrics):
            ax = axes[i][j]
            vals = [runs[n].get(k) for n in names]
            x = np.arange(len(names))
            ax.bar(x, [v if v is not None else np.nan for v in vals], yerr=sig_run.get(k), capsize=3,
                   color=["tab:red" if n == "full" else "tab:blue" for n in names])
            ax.set_xticks(x, names, rotation=20, fontsize=7)
            if j == 0:
                ax.set_ylabel(dim)
            if i == 0:
                ax.set_title(title)
            ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Ablations (red = full; error bar = run-to-run σ, two-sample estimate)")
    fig.tight_layout(); fig.savefig(f"{root}/fig_ablations.png", dpi=130); plt.close(fig)


def _layer_fracs(h):
    st = h.get("structure", {})
    cfg = st.get("base_config", {})
    nh = cfg.get("num_attention_heads", 24)
    n_double, n_single = cfg.get("num_layers", 5), cfg.get("num_single_layers", 20)
    heads, ffn = np.zeros(n_double + n_single), np.zeros(n_double + n_single)
    for kind, off, key, okey in (("double", 0, "ff", "orig_ff"), ("single", n_double, "mlp", "orig_mlp")):
        for s in st.get(kind, []):
            u = s.get("units", {})
            heads[off + s["orig"]] = s["heads"] / u.get("orig_heads", nh)
            ffn[off + s["orig"]] = s[key] / u.get(okey, s[key])
    return heads, ffn, n_double, n_single


def fig_layers(runs, root):
    names = [n for n in ["full", "full_rep", "fixed_ratio", "uniform_rate", "random", "head_grad_only", "no_modality",
                         "neuron_act", "block_before"] if n in runs and runs[n]["history"].get("structure")]
    if not names:
        return
    H, F = [], []
    for n in names:
        hh, ff, nd, ns = _layer_fracs(runs[n]["history"])
        H.append(hh); F.append(ff)
    labels = [f"D{i}" for i in range(nd)] + [f"S{i}" for i in range(ns)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 1.0 + 0.45 * len(names) * 2))
    for ax, M, title in ((axes[0], H, "kept attention heads / original"), (axes[1], F, "kept FFN width / original")):
        im = ax.imshow(np.array(M), cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)), labels, fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.axvline(nd - 0.5, c="w", lw=1)
        ax.set_title(title + " (0 = removed block)")
        fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(); fig.savefig(f"{root}/fig_layers.png", dpi=130); plt.close(fig)


def sample_compare(runs, stage1_root, root, idxs, size=224):
    cols = [("unpruned", f"{stage1_root}/base/s42/gen")] + [(n, f"{runs[n]['dir']}/gen_dval") for n in SAMPLE_RUNS
                                                              if n in runs and os.path.isdir(f"{runs[n]['dir']}/gen_dval")]
    if len(cols) < 2:
        return
    os.makedirs(f"{root}/samples", exist_ok=True)
    pad = 16
    canvas = Image.new("RGB", (size * len(cols), (size + pad) * len(idxs) + pad), "white")
    d = ImageDraw.Draw(canvas)
    for ci, (n, _) in enumerate(cols):
        d.text((ci * size + 3, 2), n, fill="black")
    for ri, i in enumerate(idxs):
        for ci, (_, g) in enumerate(cols):
            p = f"{g}/{i:05d}_0.png"
            if os.path.exists(p):
                canvas.paste(Image.open(p).convert("RGB").resize((size, size)), (ci * size, pad + ri * (size + pad)))
    canvas.save(f"{root}/samples/compare.jpg", quality=90)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/stage2_v2")
    p.add_argument("--stage1_root", default="outputs/stage1_v2")
    p.add_argument("--sample_idx", type=int, nargs="+", default=[0, 1, 2, 3, 5, 8, 13, 21])
    a = p.parse_args()
    import yaml

    gates = yaml.safe_load(open("configs/prune.yaml"))["apra"].get("gates")
    runs = load_runs(a.root)
    base, sig_seed = load_base(a.stage1_root)
    add_deltas(runs, base)
    sig_run = run_sigma(runs)
    write_summary(runs, base, sig_seed, sig_run, a.root)
    for n in ("full", "full_rep", "head_only", "neuron_act"):
        if n in runs:
            fig_apra(runs[n], gates, a.root)
    fig_pareto(runs, sig_run, a.root)
    fig_ablations(runs, sig_run, a.root)
    fig_layers(runs, a.root)
    sample_compare(runs, a.stage1_root, a.root, a.sample_idx)


if __name__ == "__main__":
    main()
