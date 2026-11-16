"""阶段3 块剪枝恢复实验的逐 checkpoint 评估（与 APRA / drift_ref 相同的采样设置：D_val 前 200 条，512px，20 步，CFG 4.0，seed 42）。

python scripts/eval_stage3.py --root outputs/stage3_blockrec --names full no_block fixed_ratio
python scripts/eval_stage3.py --root outputs/stage3_blockrec --baselines   # 原始 4B 与 9B 教师（恢复上限）

对每个模型的 step_0（= stage2_v2 的剪枝模型，未蒸馏）、step_1000/2000/3000、final（= step_4000）记录：
  hps_abs    : HPSv2.1 200 条均值；hps_drop = 原始 4B 均值 - hps_abs
  traj       : 沿原始 4B CFG 采样轨迹（前 32 条 prompt）的引导后速度场 MSE × 100（蒸馏目标是 9B，只作参考）
产物：<root>/<name>/metrics.json（按步数），<root>/<name>/samples/step_<n>/<idx>.png（SAMPLE_IDX 对应的样例图）。
"""

import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ecflux.flux_utils import TextEncoder, decode_latents, euler_sample, load_scheduler, load_transformer, load_vae, to_pil  # noqa: E402
from ecflux.metrics import HPSv2Scorer  # noqa: E402
from ecflux.pruning.evaluators import TrajVelocityEvaluator  # noqa: E402
from ecflux.utils import read_prompts, save_json, setup_logging  # noqa: E402

PIPE4, PIPE9 = "weights/FLUX.2-klein-base-4B", "weights/FLUX.2-klein-base-9B"
# stage2 compare.jpg 的 8 条 + 4 条人物 prompt（11 头巾女性、22 旗袍女性、29 NYPD 男性、31 男孩与男人）
SAMPLE_IDX = [0, 1, 2, 3, 5, 8, 13, 21, 11, 22, 29, 31]
N_HPS, N_TRAJ, SEED = 200, 32, 42
SAMP = dict(height=512, width=512, num_steps=20, guidance_scale=4.0)


def encode(pipe, prompts, device):
    te = TextEncoder(pipe, device, torch.bfloat16, 512)
    hps_emb = [te.encode(prompts[i:i + 4]).cpu() for i in range(0, N_HPS, 4)]
    traj_emb = [te.encode(prompts[i:i + 4]).cpu() for i in range(0, N_TRAJ, 4)]
    neg = te.encode([""]).cpu()
    del te
    gc.collect()
    torch.cuda.empty_cache()
    return hps_emb, traj_emb, neg


@torch.no_grad()
def hps_and_samples(model, vae, scorer, prompts, embeds, neg, device, sample_dir):
    """与 HPSEvaluator.score 相同的采样顺序（一个 generator 依次跨 batch），额外保存 SAMPLE_IDX 的图。"""
    os.makedirs(sample_dir, exist_ok=True)
    g = torch.Generator().manual_seed(SEED)
    sched = load_scheduler(PIPE4)
    scores, k = [], 0
    for emb in embeds:
        emb = emb.to(device)
        n = neg.to(device).expand(emb.shape[0], -1, -1)
        lat, ids = euler_sample(model, emb, negative_embeds=n, scheduler=sched, generator=g, **SAMP)
        imgs = to_pil(decode_latents(vae, lat.to(vae.dtype), ids))
        scores += scorer(imgs, prompts[k:k + len(imgs)])
        for j, img in enumerate(imgs):
            if k + j in SAMPLE_IDX:
                img.save(os.path.join(sample_dir, f"{k + j:05d}.png"))
        k += len(imgs)
    return sum(scores) / len(scores)


def checkpoints(root, name):
    out = [(0, f"outputs/stage2_v2/{name}/pruned")]
    out += [(s, f"{root}/{name}/step_{s}") for s in (1000, 2000, 3000)]
    out.append((4000, f"{root}/{name}/final"))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/stage3_blockrec")
    p.add_argument("--names", nargs="*", default=[])
    p.add_argument("--baselines", action="store_true")
    a = p.parse_args()
    setup_logging()
    device = torch.device("cuda")
    prompts = read_prompts("data/val_prompts.txt")
    scorer = HPSv2Scorer()

    if a.baselines:
        for tag, pipe in (("_base4b", PIPE4), ("_teacher9b", PIPE9)):
            mp = f"{a.root}/{tag}/metrics.json"
            if os.path.exists(mp):
                continue
            hps_emb, _, neg = encode(pipe, prompts, device)
            vae = load_vae(pipe, torch.float32, device)
            model = load_transformer(pipe).to(device).eval().requires_grad_(False)
            hps = hps_and_samples(model, vae, scorer, prompts, hps_emb, neg, device, f"{a.root}/{tag}/samples")
            save_json({"hps_abs": hps}, mp)
            print(f"{tag}: hps={hps:.4f}")
            del model, vae
            gc.collect()
            torch.cuda.empty_cache()
        if not a.names:
            return

    hps_emb, traj_emb, neg = encode(PIPE4, prompts, device)
    vae = load_vae(PIPE4, torch.float32, device)
    orig = load_transformer(PIPE4).to(device).eval().requires_grad_(False)
    traj_ev = TrajVelocityEvaluator(orig, traj_emb, neg, device, height=512, width=512, steps=20, guidance_scale=4.0,
                                    scheduler=load_scheduler(PIPE4), seed=SEED)
    base_p = f"{a.root}/_base4b/metrics.json"
    if os.path.exists(base_p):
        base = json.load(open(base_p))["hps_abs"]
    else:
        base = hps_and_samples(orig, vae, scorer, prompts, hps_emb, neg, device, f"{a.root}/_base4b/samples")
        save_json({"hps_abs": base}, base_p)
    del orig
    gc.collect()
    torch.cuda.empty_cache()

    for name in a.names:
        mp = f"{a.root}/{name}/metrics.json"
        res = json.load(open(mp)) if os.path.exists(mp) else {}
        for step, path in checkpoints(a.root, name):
            if str(step) in res or not os.path.exists(os.path.join(path, "prune_spec.json")):
                continue
            model = load_transformer(path).to(device).eval().requires_grad_(False)
            hps = hps_and_samples(model, vae, scorer, prompts, hps_emb, neg, device,
                                  f"{a.root}/{name}/samples/step_{step}")
            traj = traj_ev(model)
            res[str(step)] = {"hps_abs": hps, "hps_drop": base - hps, "traj": traj, "path": path}
            save_json(res, mp)
            print(f"{name} step {step}: hps={hps:.4f} (drop {base - hps:+.4f}) traj={traj:.2f}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
