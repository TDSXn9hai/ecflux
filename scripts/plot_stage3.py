"""阶段3 块剪枝恢复实验的图表（数据来自 scripts/eval_stage3.py 与 run_stage3.sh dval）。

python scripts/plot_stage3.py --root outputs/stage3_blockrec
产物：
  fig_recovery.png             HPSv2(200) 下降与轨迹速度场 MSE 随蒸馏步数的变化，横线为原始 4B / 9B 教师
  samples/compare_step{0,2000,4000}.jpg   同 prompt、同 seed：4B / 9B / full / no_block / fixed_ratio
  summary.json                 逐步数指标 + final 的 D_val 1000 条 CLIP / HPSv2（含 stage2 未蒸馏值）
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

NAMES = ["full", "no_block", "fixed_ratio"]
SAMPLE_IDX = [0, 1, 2, 3, 5, 8, 13, 21, 11, 22, 29, 31]


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fig_recovery(runs, base, teacher, root):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for n, m in runs.items():
        steps = sorted(int(s) for s in m)
        axes[0].plot(steps, [m[str(s)]["hps_drop"] for s in steps], "o-", label=n)
        axes[1].plot(steps, [m[str(s)]["traj"] for s in steps], "o-", label=n)
    axes[0].axhline(0, color="gray", ls="--", lw=1, label="unpruned 4B")
    if teacher is not None:
        axes[0].axhline(base - teacher, color="green", ls=":", lw=1, label="9B teacher")
    axes[0].set(xlabel="distill steps", ylabel="HPSv2 drop vs 4B (200 prompts)", title="quality gap (lower is better)")
    axes[1].set(xlabel="distill steps", ylabel="traj velocity MSE vs 4B", title="distance to 4B trajectory (reference only)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{root}/fig_recovery.png", dpi=130)
    plt.close(fig)


def compare(root, step, size=224):
    cols = [("4B", f"{root}/_base4b/samples"), ("9B", f"{root}/_teacher9b/samples")]
    cols += [(n, f"{root}/{n}/samples/step_{step}") for n in NAMES]
    cols = [(n, d) for n, d in cols if os.path.isdir(d)]
    pad = 16
    canvas = Image.new("RGB", (size * len(cols), (size + pad) * len(SAMPLE_IDX) + pad), "white")
    d = ImageDraw.Draw(canvas)
    for ci, (n, _) in enumerate(cols):
        d.text((ci * size + 3, 2), f"{n} @{step}" if n in NAMES else n, fill="black")
    for ri, i in enumerate(SAMPLE_IDX):
        for ci, (_, g) in enumerate(cols):
            p = f"{g}/{i:05d}.png"
            if os.path.exists(p):
                canvas.paste(Image.open(p).convert("RGB").resize((size, size)), (ci * size, pad + ri * (size + pad)))
    os.makedirs(f"{root}/samples", exist_ok=True)
    canvas.save(f"{root}/samples/compare_step{step}.jpg", quality=90)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/stage3_blockrec")
    p.add_argument("--src", default="outputs/stage2_v2")
    p.add_argument("--stage1_root", default="outputs/stage1_v2")
    a = p.parse_args()
    base = jload(f"{a.root}/_base4b/metrics.json")["hps_abs"]
    teacher = (jload(f"{a.root}/_teacher9b/metrics.json") or {}).get("hps_abs")
    runs = {n: jload(f"{a.root}/{n}/metrics.json") for n in NAMES}
    runs = {n: m for n, m in runs.items() if m}
    fig_recovery(runs, base, teacher, a.root)
    for step in (0, 2000, 4000):
        compare(a.root, step)
    dval_base = jload(f"{a.stage1_root}/base/s42/eval/summary.json")
    dval = {}
    for n in runs:
        before, after = jload(f"{a.src}/{n}/eval_dval/summary.json"), jload(f"{a.root}/{n}/eval_dval/summary.json")
        dval[n] = {"before": before, "after": after}
    out = {"hps200_base4b": base, "hps200_teacher9b": teacher, "runs": runs, "dval_base4b": dval_base, "dval": dval}
    json.dump(out, open(f"{a.root}/summary.json", "w"), indent=2)
    print(json.dumps({n: {s: round(v["hps_drop"], 4) for s, v in m.items()} for n, m in runs.items()}, indent=1))


if __name__ == "__main__":
    main()
