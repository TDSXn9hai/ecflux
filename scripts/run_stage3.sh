#!/usr/bin/env bash
# 阶段3 块剪枝恢复实验：同一蒸馏预算下蒸馏 full（删 4 块）/ no_block（不删块）/ fixed_ratio（删 4 块），
# 看 full 与 no_block 的质量差距在蒸馏中能否缩小。输出 outputs/stage3_blockrec/<name>/，step_{1000,2000,3000} 与 final。
#   bash scripts/run_stage3.sh train       # GPU1-3 各蒸馏一个模型（并行，4000 步 × batch 16，约 9h）
#   bash scripts/run_stage3.sh eval        # 逐 checkpoint 的 HPSv2 200 / 轨迹速度场 MSE + 9B 上限（scripts/eval_stage3.py，GPU1-3）
#   bash scripts/run_stage3.sh dval        # final 在 D_val 1000 条上的 CLIP / HPSv2（GPU1-3 各一个模型）
#   bash scripts/run_stage3.sh plot        # fig_recovery.png + 对比拼图（scripts/plot_stage3.py）
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_MODULE_LOADING=LAZY
export HF_HOME=$PWD/weights/hf_home TORCH_HOME=$PWD/weights/torch_home HF_HUB_OFFLINE=1
PY=${PY:-.venv/bin/python}
OUT=${OUT:-outputs/stage3_blockrec}
SRC=outputs/stage2_v2
PIPE=weights/FLUX.2-klein-base-4B
NAMES=(full no_block fixed_ratio)
WHAT=${1:-train}
mkdir -p $OUT

if [[ $WHAT == train ]]; then
  GPUS=(${GPUS:-1 2 3})
  for i in "${!NAMES[@]}"; do
    name=${NAMES[$i]}; g=${GPUS[$i]}
    [[ -f $OUT/$name/final/prune_spec.json ]] && { echo "skip $name"; continue; }
    mkdir -p $OUT/$name
    ( echo "[$(date +%F' '%T)] start $name on GPU$g"
      CUDA_VISIBLE_DEVICES=$g $PY scripts/distill.py --config configs/distill.yaml --out $OUT/$name \
        --set student=$SRC/$name/pruned train.max_steps=4000 train.global_batch_size=16 \
        train.save_every=1000 distill.ramp_start=500 distill.ramp_end=2500 > $OUT/$name/distill.log 2>&1 \
        && echo "[$(date +%F' '%T)] done $name" || echo "FAILED $name (see $OUT/$name/distill.log)" ) &
  done
  wait
fi

if [[ $WHAT == eval ]]; then
  # 第一张卡先测原始 4B 与 9B 教师，再各卡评估一个模型
  GPUS=(${GPUS:-1 2 3})
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/eval_stage3.py --root $OUT --baselines > $OUT/eval_baselines.log 2>&1
  for i in "${!NAMES[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY scripts/eval_stage3.py --root $OUT --names ${NAMES[$i]} \
      > $OUT/eval_${NAMES[$i]}.log 2>&1 &
  done
  wait
fi

if [[ $WHAT == dval ]]; then
  GPUS=(${GPUS:-1 2 3})
  for i in "${!NAMES[@]}"; do
    name=${NAMES[$i]}; g=${GPUS[$i]}; d=$OUT/$name
    [[ -f $d/eval_dval/summary.json ]] && continue
    ( [[ -f $d/gen_dval/index.jsonl ]] || CUDA_VISIBLE_DEVICES=$g $PY scripts/generate.py --pipeline $PIPE \
        --transformer "$d/final" --prompts data/val_prompts.txt --out "$d/gen_dval" --height 512 --width 512 \
        --steps 20 --guidance_scale 4.0 --seed 42 > "$d/gen_dval.log" 2>&1
      CUDA_VISIBLE_DEVICES=$g $PY scripts/evaluate.py --images "$d/gen_dval" --ref_dir data/val_reference_images \
        --metrics clip hps --out "$d/eval_dval" > "$d/eval_dval.log" 2>&1
      echo "evaluated $name" ) &
  done
  wait
fi

if [[ $WHAT == plot ]]; then
  $PY scripts/plot_stage3.py --root $OUT --src $SRC
fi
