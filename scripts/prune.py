"""4.2 多粒度结构化压缩（三粒度剪枝 + APRA + 块级剪枝 + 恢复训练）。

python scripts/prune.py --config configs/prune.yaml --out outputs/prune/default
python scripts/prune.py --config configs/prune.yaml --out outputs/prune/no_apra --set apra.enabled=false
"""

import itertools
import os

import torch
from _common import base_parser, init
from torch.utils.data import DataLoader

from ecflux.data import build_train_data
from ecflux.flux_utils import TextEncoder, load_scheduler, load_transformer, load_vae, prepare_fm_batch
from ecflux.metrics import efficiency
from ecflux.pruning.apra import compress
from ecflux.pruning.evaluators import DriftReference, MultiEvaluator, build_evaluator
from ecflux.pruning.prune_ops import save_pruned_transformer
from ecflux.utils import Tracker, count_params, get_device, get_dtype, read_prompts, save_json


def fm_iter(ds, bs, device, dtype, seed, tcfg, shuffle=True, encoder=None, collate=None):
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=collate, num_workers=2, drop_last=True)
    while True:
        for b in dl:
            yield prepare_fm_batch(b if encoder is None else encoder(b), device, dtype, g, tcfg)


def reference_images(folder, size, bs=16):
    import numpy as np

    from ecflux.metrics import load_images, square_resize

    _, it = load_images(folder)
    buf = []
    for img in it:
        buf.append(torch.from_numpy(np.asarray(square_resize(img, 299))).permute(2, 0, 1).float() / 255)
        if len(buf) == bs:
            yield torch.stack(buf)
            buf = []
    if buf:
        yield torch.stack(buf)


def main():
    args = base_parser(__doc__, "configs/prune.yaml").parse_args()
    cfg = init(args)
    out = args.out or os.path.join(cfg["paths"]["output_root"], "prune", "default")
    device, dtype = get_device(cfg.get("device")), get_dtype(cfg.get("dtype", "bf16"))
    tracker = Tracker(cfg.get("wandb"), cfg)

    model = load_transformer(cfg["paths"]["student_pipeline"], dtype=dtype).to(device).eval().requires_grad_(False)
    n0 = count_params(model)
    ds, encoder, collate = build_train_data(cfg, device, dtype)
    dk = {"encoder": encoder, "collate": collate}
    c = cfg["calib"]
    calib = list(itertools.islice(fm_iter(ds, c["batch_size"], device, dtype, 1234, cfg.get("timesteps"), False, **dk),
                                  c["num_batches"]))
    # recover.seed 只改恢复训练的数据顺序与噪声，评估用的 prompt / seed 保持不变（标定运行间波动用）
    recover = fm_iter(ds, cfg["recover"].get("batch_size", 1), device, dtype, cfg["recover"].get("seed", cfg["seed"]),
                      cfg.get("timesteps"), **dk)

    e = cfg["eval"]
    names = [e["metric"]] + [m for m in e.get("aux_metrics") or [] if m != e["metric"]]
    sched = load_scheduler(cfg["paths"]["student_pipeline"])
    samp = dict(device=device, height=e["height"], width=e["width"], steps=e["steps"],
                guidance_scale=e["guidance_scale"], scheduler=sched, seed=cfg["seed"])
    prompts = read_prompts(e["prompts"])
    enc = {}
    if {"fid", "hps", "traj_velocity_mse"} & set(names):
        # 在线数据模式下复用已加载的文本编码器（恢复训练还要用，不能 offload）
        te = encoder.te if encoder is not None else \
            TextEncoder(cfg["paths"]["student_pipeline"], device, dtype, cfg["data"]["max_sequence_length"])
        for key, n in (("fid", e["num_prompts"]), ("hps", e.get("hps_prompts", 200)),
                       ("traj_velocity_mse", e.get("traj_prompts", 16))):
            if key in names:
                enc[key] = [te.encode(prompts[i:i + 4]).cpu() for i in range(0, n, 4)]
        neg = te.encode([""]).cpu()
        if encoder is None:
            te.offload()
        del te
    vae = load_vae(cfg["paths"]["student_pipeline"], torch.float32, device) if {"fid", "hps"} & set(names) else None
    evaluators = {}
    for name in names:
        if name == "fid":
            evaluators[name] = build_evaluator(
                "fid", vae=vae, prompt_embeds=enc["fid"], negative_embeds=neg,
                real_images=reference_images(e["reference_dir"], e["height"]), **samp)
        elif name == "hps":
            evaluators[name] = build_evaluator("hps", ref_model=model, vae=vae, prompts=prompts[: e.get("hps_prompts", 200)],
                                               prompt_embeds=enc["hps"], negative_embeds=neg, **samp)
        elif name == "traj_velocity_mse":
            evaluators[name] = build_evaluator(name, ref_model=model, prompt_embeds=enc[name], negative_embeds=neg,
                                               **samp)
        else:
            ev_batches = list(itertools.islice(fm_iter(ds, 1, device, dtype, 999, cfg.get("timesteps"), False, **dk),
                                               e.get("proxy_batches", 16)))
            evaluators[name] = build_evaluator(name, ref_model=model, batches=ev_batches, device=device)
    drift = None
    if e.get("drift_ref"):  # 漂移校正：与同步数恢复训练的未剪枝参照比较（scripts/drift_ref.py 生成）
        if "traj_velocity_mse" not in names or "hps" not in names:
            raise ValueError("eval.drift_ref 需要 traj_velocity_mse 与 hps 都在 eval.metric / aux_metrics 中")
        drift = DriftReference(e["drift_ref"], cfg["recover"].get("steps", 2000), enc["traj_velocity_mse"], neg,
                               **{k: v for k, v in samp.items()})
    evaluator = evaluators[names[0]] if len(names) == 1 and drift is None else MultiEvaluator(evaluators, names[0], drift)

    history = compress(model, calib, recover, evaluator, cfg, tracker)
    save_pruned_transformer(model, os.path.join(out, "pruned"), extra={"history": {k: v for k, v in history.items()
                                                                                 if k != "structure"}})
    fb = calib[0]
    eff = efficiency(model, {"hidden_states": fb["x_t"][:1], "timestep": fb["t"][:1],
                             "encoder_hidden_states": fb["prompt_embeds"][:1], "img_ids": fb["img_ids"][:1],
                             "txt_ids": fb["txt_ids"][:1]}, steps=20)
    history["efficiency"] = eff
    save_json(history, os.path.join(out, "history.json"))
    print(f"params {n0 / 1e9:.3f}B -> {count_params(model) / 1e9:.3f}B; saved to {out}/pruned")
    tracker.finish()


if __name__ == "__main__":
    main()
