#!/usr/bin/env bash
# 恢复训练后的阈值标定梯度：新初值 × s 剪一轮 + 2000 步恢复（recover.seed=1，与 calib_seed1 对齐），再生成 24 张样例供人工标注
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_MODULE_LOADING=LAZY HF_HOME=$PWD/weights/hf_home TORCH_HOME=$PWD/weights/torch_home HF_HUB_OFFLINE=1
PY=.venv/bin/python
one() {
  local s=$1 gpu=$2 o=outputs/stage2_v2/calib_scale$1
  mkdir -p $o
  g=$($PY - "$s" <<'PYEOF'
import sys, yaml
v = float(sys.argv[1]); gs = yaml.safe_load(open("configs/prune.yaml"))["groups"]
for g in gs:
    g["init"] = round(min(0.5, g["init"] * v), 4); g.pop("max", None)
print(yaml.safe_dump(gs, default_flow_style=True, width=10**6).strip())
PYEOF
)
  [[ -f $o/history.json ]] || CUDA_VISIBLE_DEVICES=$gpu $PY scripts/prune.py --config configs/prune.yaml --out $o --set "groups=$g" \
    apra.rounds=1 apra.rollback=false apra.target_param_ratio=null prune.prune_blocks=false recover.seed=1 > $o/prune.log 2>&1
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/generate.py --pipeline weights/FLUX.2-klein-base-4B --transformer $o/pruned \
    --prompts data/val_prompts.txt --out $o/gen_s --height 512 --width 512 --steps 20 --guidance_scale 4.0 --seed 42 --limit 24 > $o/gen.log 2>&1
  echo "done scale $s"
}
( one 0.5 2; one 2.0 2 ) &
( one 1.5 3; one 1.0 3 ) &
wait
