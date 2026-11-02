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
from ecflux.pruning.evaluators import build_evaluator
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
    recover = fm_iter(ds, cfg["recover"].get("batch_size", 1), device, dtype, cfg["seed"], cfg.get("timesteps"), **dk)

    e = cfg["eval"]
    if e["metric"] == "fid":
        # 在线数据模式下复用已加载的文本编码器（恢复训练还要用，不能 offload）
        te = encoder.te if encoder is not None else \
            TextEncoder(cfg["paths"]["student_pipeline"], device, dtype, cfg["data"]["max_sequence_length"])
        prompts = read_prompts(e["prompts"])[: e["num_prompts"]]
        embeds = [te.encode(prompts[i:i + 4]).cpu() for i in range(0, len(prompts), 4)]
        neg = te.encode([""]).cpu()
        if encoder is None:
            te.offload()
        del te
        evaluator = build_evaluator(
            "fid", vae=load_vae(cfg["paths"]["student_pipeline"], torch.float32, device), prompt_embeds=embeds,
            real_images=reference_images(e["reference_dir"], e["height"]), device=device, height=e["height"],
            width=e["width"], steps=e["steps"], guidance_scale=e["guidance_scale"], negative_embeds=neg,
            scheduler=load_scheduler(cfg["paths"]["student_pipeline"]), seed=cfg["seed"])
    else:
        ev_batches = list(itertools.islice(fm_iter(ds, 1, device, dtype, 999, cfg.get("timesteps"), False, **dk),
                                           e.get("proxy_batches", 16)))
        evaluator = build_evaluator(e["metric"], ref_model=model, batches=ev_batches, device=device)

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
