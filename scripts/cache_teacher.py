"""离线缓存教师输出（报告 4.3.1：教师参数冻结，全部输出离线一次性生成，训练时无需加载教师）。

python scripts/cache_teacher.py --config configs/distill.yaml --out data/teacher_cache
每个样本保存 K 组 (t, ε, v_T, 隐藏状态, 头平均后的注意力图)。磁盘占用约
K × (N_img×C + N_img×D_T×3 + Nq×T) × 2 bytes / 样本，请先用 --max_samples 估算。
"""

import glob
import os

import torch
from _common import base_parser, init

from ecflux.distill.trainer import DEFAULT_STUDENT_HIDDEN, auto_teacher_layers, model_depth, teacher_draws
from ecflux.flux_utils import load_transformer
from ecflux.utils import get_device, get_dtype, load_json


def main():
    p = base_parser(__doc__, "configs/distill.yaml")
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()
    cfg = init(args)
    out = args.out or cfg["teacher"]["offline_dir"]
    device, dtype = get_device(cfg.get("device")), get_dtype(cfg.get("dtype", "bf16"))
    teacher = load_transformer(cfg["paths"]["teacher_pipeline"], dtype=dtype).to(device).eval().requires_grad_(False)
    spec = load_json(os.path.join(cfg["student"], "prune_spec.json"))
    dcfg = cfg["distill"]
    s_layers = dcfg.get("student_hidden_layers") or DEFAULT_STUDENT_HIDDEN
    t_layers = dcfg.get("teacher_hidden_layers", "auto")
    if t_layers == "auto":
        t_layers = auto_teacher_layers(s_layers, model_depth(spec["base_config"]), model_depth(teacher.config))
    attn_layers = list(range(int(dcfg.get("attn_k", 2)))) if dcfg.get("use_attn", True) else []
    if not dcfg.get("use_hid", True):
        t_layers = []
    g = torch.Generator().manual_seed(cfg["seed"])
    n = 0
    for root in cfg["data"]["cache_dirs"]:
        for f in sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)):
            if args.max_samples and n >= args.max_samples:
                break
            s = torch.load(f, map_location="cpu", weights_only=False)
            s["teacher"] = teacher_draws(teacher, s, int(cfg["teacher"].get("num_draws", 2)), t_layers, attn_layers,
                                         device, dtype, g, cfg.get("timesteps"), cfg["teacher"]["embeds_key"],
                                         int(dcfg.get("attn_query_stride", 2)))
            dst = os.path.join(out, os.path.relpath(f, root))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            torch.save(s, dst)
            n += 1
    print(f"cached teacher targets for {n} samples -> {out} (teacher layers {t_layers}, attn {attn_layers})")


if __name__ == "__main__":
    main()
