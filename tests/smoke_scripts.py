"""脚本级冒烟测试：在临时目录构造极小「管线」与合成缓存，依次调用 scripts/*.py（CPU、fp32、各 1~2 步）。

python tests/smoke_scripts.py
"""

import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from ecflux.data import SyntheticCachedDataset  # noqa: E402
from ecflux.flux_utils import tiny_transformer  # noqa: E402

PY = sys.executable


def run(*args):
    cmd = [PY, *args]
    print("$", " ".join(os.path.relpath(a, ROOT) if os.path.isabs(a) and a.startswith(ROOT) else a for a in cmd))
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-5000:])
        raise SystemExit(f"FAILED: {args[0]}")
    print("   " + (r.stdout.strip().splitlines() or [""])[-1][:200])


def main():
    with tempfile.TemporaryDirectory() as tmp:
        stu, tea = os.path.join(tmp, "student"), os.path.join(tmp, "teacher")
        tiny_transformer(heads=8, head_dim=16, joint_dim=32, seed=0).save_pretrained(os.path.join(stu, "transformer"))
        tiny_transformer(num_layers=6, num_single_layers=24, heads=6, head_dim=16, joint_dim=48, seed=1) \
            .save_pretrained(os.path.join(tea, "transformer"))
        cache = os.path.join(tmp, "cache", "00000")
        os.makedirs(cache)
        for i, d in enumerate(SyntheticCachedDataset(n=8, embed_dim=32, teacher_embed_dim=48)):
            torch.save(d, os.path.join(cache, f"{i:08d}.pt"))
        common = [f"paths.student_pipeline={stu}", f"paths.teacher_pipeline={tea}", "device=cpu", "dtype=fp32",
                  "data.mode=cache", f"data.cache_dirs=[{os.path.dirname(cache)}]", "wandb.enabled=false"]

        out_p = os.path.join(tmp, "prune")
        run("scripts/prune.py", "--config", "configs/prune.yaml", "--out", out_p, "--set", *common,
            "eval.metric=velocity_mse", "eval.proxy_batches=2", "calib.num_batches=2", "calib.batch_size=2",
            "recover.steps=1", "recover.block_steps=1", "recover.grad_accum=1", "recover.use_8bit_adam=false",
            "apra.rounds=2", "block.tau=-1", "block.max_blocks=1")
        pruned = os.path.join(out_p, "pruned")
        assert os.path.exists(os.path.join(pruned, "prune_spec.json"))

        dist_common = [*common, f"student={pruned}", "train.max_steps=2", "train.global_batch_size=2",
                       "train.micro_batch_size=1", "train.num_workers=0", "train.use_8bit_adam=false",
                       "train.save_every=0", "distill.ramp_start=0", "distill.ramp_end=1"]
        run("scripts/distill.py", "--config", "configs/distill.yaml", "--out", os.path.join(tmp, "d_online"),
            "--set", *dist_common, "teacher.mode=online")
        off = os.path.join(tmp, "tcache")
        run("scripts/cache_teacher.py", "--config", "configs/distill.yaml", "--out", off, "--set", *dist_common)
        run("scripts/distill.py", "--config", "configs/distill.yaml", "--out", os.path.join(tmp, "d_offline"),
            "--set", *dist_common, "teacher.mode=offline", f"teacher.offline_dir={off}")

        run("scripts/evaluate.py", "--transformer", os.path.join(tmp, "d_offline", "final"), "--metrics", "efficiency",
            "--device", "cpu", "--dtype", "fp32", "--resolution", "64", "--text_len", "8", "--steps", "2",
            "--out", os.path.join(tmp, "eval"))

        # 路由：构造逐 prompt 分数 -> 标签 -> 训练（struct 特征，无需文本编码器）
        edge, cloud = os.path.join(tmp, "edge.jsonl"), os.path.join(tmp, "cloud.jsonl")
        prompts = [f"a cat number {i}" for i in range(20)] + \
                  [f'{i} red cubes left of two blue spheres and a sign saying "HI"' for i in range(20)]
        with open(edge, "w") as fe, open(cloud, "w") as fc:
            for i, p in enumerate(prompts):
                hard = "cubes" in p
                fe.write(json.dumps({"idx": i, "prompt": p, "clip": 30 - 3 * hard, "hps": 0.27 - 0.02 * hard}) + "\n")
                fc.write(json.dumps({"idx": i, "prompt": p, "clip": 31.0, "hps": 0.28}) + "\n")
        lab = os.path.join(tmp, "router")
        run("scripts/build_router_labels.py", "--config", "configs/router.yaml", "--edge", edge, "--cloud", cloud,
            "--out", lab)
        run("scripts/train_router.py", "--config", "configs/router.yaml", "--labels", os.path.join(lab, "labels.jsonl"),
            "--out", os.path.join(tmp, "router_out"), "--set", "router.features=struct", "router.epochs=5",
            "calibration.cloud_budget=0.5", "device=cpu")
    print("\nSCRIPT SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
