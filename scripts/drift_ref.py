"""漂移校正参照（stage2 v2）：不剪枝的 4B 按 APRA 的恢复训练设置连续训练，每 recover.steps 步保存一个 checkpoint，
并记录该 checkpoint 相对原始模型的 HPSv2 均值与轨迹速度场 MSE（恢复训练本身带来的偏移曲线）。
APRA 设置 eval.drift_ref=<out> 后，剪枝模型训练了 n 步时与 <out>/ckpt_{n // recover.steps} 比较（见 DriftReference）。

python scripts/drift_ref.py --config configs/prune.yaml --out outputs/stage2_v2/drift_ref --chunks 6
产物：ckpt_{k}/{prune_spec.json, model.safetensors, metrics.json}，k = 1..chunks；metrics.json 最后写入（作为就绪标记）。
"""

import itertools
import os

import torch
from _common import base_parser, init
from prune import fm_iter

from ecflux.data import build_train_data
from ecflux.flux_utils import TextEncoder, load_scheduler, load_transformer, load_vae
from ecflux.pruning.apra import flow_finetune
from ecflux.pruning.evaluators import HPSEvaluator, TrajVelocityEvaluator
from ecflux.pruning.pruner import init_unit_meta
from ecflux.pruning.prune_ops import save_pruned_transformer
from ecflux.utils import Tracker, get_device, get_dtype, read_prompts, save_json


def main():
    p = base_parser(__doc__, "configs/prune.yaml")
    p.add_argument("--chunks", type=int, default=6, help="checkpoint 个数（APRA 5 轮 + 块剪枝恢复 = 6）")
    args = p.parse_args()
    cfg = init(args)
    out = args.out
    device, dtype = get_device(cfg.get("device")), get_dtype(cfg.get("dtype", "bf16"))
    tracker = Tracker(cfg.get("wandb"), cfg)
    model = load_transformer(cfg["paths"]["student_pipeline"], dtype=dtype).to(device).eval().requires_grad_(False)
    init_unit_meta(model)
    ds, encoder, collate = build_train_data(cfg, device, dtype)
    rcfg = cfg["recover"]
    recover = fm_iter(ds, rcfg.get("batch_size", 1), device, dtype, rcfg.get("seed", cfg["seed"]), cfg.get("timesteps"),
                      encoder=encoder, collate=collate)

    e = cfg["eval"]
    te = encoder.te if encoder is not None else \
        TextEncoder(cfg["paths"]["student_pipeline"], device, dtype, cfg["data"]["max_sequence_length"])
    prompts = read_prompts(e["prompts"])
    enc = lambda n: [te.encode(prompts[i:i + 4]).cpu() for i in range(0, n, 4)]  # noqa: E731
    hps_emb, traj_emb, neg = enc(e.get("hps_prompts", 200)), enc(e.get("traj_prompts", 32)), te.encode([""]).cpu()
    samp = dict(height=e["height"], width=e["width"], steps=e["steps"], guidance_scale=e["guidance_scale"],
                scheduler=load_scheduler(cfg["paths"]["student_pipeline"]), seed=cfg["seed"])
    vae = load_vae(cfg["paths"]["student_pipeline"], torch.float32, device)
    hps_ev = HPSEvaluator(model, vae, prompts[: e.get("hps_prompts", 200)], hps_emb, device, negative_embeds=neg, **samp)
    traj_ev = TrajVelocityEvaluator(model, traj_emb, neg, device, **samp)
    save_json({"hps_abs": hps_ev.base, "traj_vs_orig": 0.0, "steps": 0}, os.path.join(out, "orig_metrics.json"))

    steps = int(rcfg.get("steps", 2000))
    for k in itertools.count(1):
        if k > args.chunks:
            break
        d = os.path.join(out, f"ckpt_{k}")
        if os.path.exists(os.path.join(d, "metrics.json")):  # 断点续跑：加载已有 checkpoint，数据顺序从头重新开始
            model = load_transformer(d, dtype=dtype).to(device).eval().requires_grad_(False)
            init_unit_meta(model)
            continue
        flow_finetune(model, recover, steps, rcfg, tracker, f"drift_c{k}")
        save_pruned_transformer(model, d, extra={"drift_ref": {"chunk": k, "steps": k * steps}})
        hps_abs = hps_ev.score(model)
        m = {"chunk": k, "steps": k * steps, "hps_abs": hps_abs, "hps_drop_vs_orig": hps_ev.base - hps_abs,
             "traj_vs_orig": traj_ev(model), "traj_vs_orig_per_step": traj_ev.per_step}
        save_json(m, os.path.join(d, "metrics.json"))
        print(f"ckpt_{k}: steps={k * steps} hps={hps_abs:.4f} (drop {m['hps_drop_vs_orig']:+.4f}) traj={m['traj_vs_orig']:.3f}")
    tracker.finish()


if __name__ == "__main__":
    main()
