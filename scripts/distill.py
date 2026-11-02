"""4.3 语义对齐知识蒸馏（学生 = 剪枝模型，教师 = FLUX.2 Klein 9B）。

python scripts/distill.py --config configs/distill.yaml --out outputs/distill/full
python scripts/distill.py --config configs/distill.yaml --out outputs/distill/no_attn --set distill.use_attn=false
"""

import os

from _common import base_parser, init

from ecflux.data import build_train_data
from ecflux.distill.trainer import OfflineTeacherDataset, train
from ecflux.flux_utils import load_transformer
from ecflux.utils import get_device, get_dtype, load_json


def main():
    args = base_parser(__doc__, "configs/distill.yaml").parse_args()
    cfg = init(args)
    out = args.out or os.path.join(cfg["paths"]["output_root"], "distill", "default")
    device = get_device(cfg.get("device"))
    # 学生以 fp32 主权重 + autocast(bf16/fp16) 训练；显存紧张时可 --set dtype=bf16 直接以 bf16 训练
    student_dtype = get_dtype(cfg.get("student_dtype", cfg.get("dtype", "bf16")))
    student = load_transformer(cfg["student"], dtype=student_dtype)
    tmode = cfg["teacher"]["mode"]
    encoder = None
    if tmode == "online":
        dtype = get_dtype(cfg.get("dtype", "bf16"))
        teacher = load_transformer(cfg["paths"]["teacher_pipeline"], dtype=dtype)
        # data.mode=online 且教师用独立文本编码器（9B 教师为 Qwen3-8B）时，同时在线编码教师文本嵌入
        t_text = cfg["paths"]["teacher_pipeline"] if cfg["teacher"].get("embeds_key") == "teacher_prompt_embeds" else None
        dataset, encoder, _ = build_train_data(cfg, device, dtype, teacher_pipeline=t_text)
    elif tmode == "offline":
        teacher = None
        tc = cfg["teacher"].get("config")
        if isinstance(tc, str):
            cfg["teacher"]["config"] = load_json(tc)
        elif tc is None:
            cfg["teacher"]["config"] = load_json(os.path.join(cfg["paths"]["teacher_pipeline"], "transformer", "config.json"))
        dataset = OfflineTeacherDataset(cfg["teacher"]["offline_dir"], seed=cfg["seed"])
    else:
        raise ValueError(tmode)
    train(student, teacher, dataset, cfg, device, out, encoder=encoder)
    print(f"done -> {out}/final")


if __name__ == "__main__":
    main()
