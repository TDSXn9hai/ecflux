#!/usr/bin/env bash
# 阶段1 v2：剪枝前分析 + 敏感度主指标校验（默认只用 GPU1-3）。各步骤可单独运行：
#   bash scripts/run_stage1.sh importance   # 1.1 三粒度重要性统计与可视化（两组校准种子）
#   bash scripts/run_stage1.sh base         # 未剪枝 4B 在 D_val 上 seed 42/43/44 各生成 1000 张：FID / HPSv2 / CLIP 的种子波动
#   bash scripts/run_stage1.sh sens         # 1.2 逐组敏感度：G1..G6 单独剪 0.1/0.2/0.3，无恢复训练；
#                                           #     每个配置记录单步 / 轨迹速度场 MSE（prune.py）+ D_val 1000 张的 FID / HPSv2 / CLIP
#   bash scripts/run_stage1.sh plot         # 汇总 summary.json、指标校验（对照 human_labels.json）与全部图表
#   bash scripts/run_stage1.sh apra_debug   # 1.3 APRA 调通（velocity_mse、2 轮、200 步恢复），记录耗时
# 敏感度实验复现旧阶段1（人工观察所用）的剪枝判据：头=梯度敏感度（Double Block 也用梯度）、神经元=激活 L1、
# 文本注意力占比不去 padding、跨模态阈值 1.0。生成设置与旧版相同（generate.py，batch 1，seed 42）。
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_MODULE_LOADING=LAZY
export HF_HOME=$PWD/weights/hf_home TORCH_HOME=$PWD/weights/torch_home HF_HUB_OFFLINE=1
PY=${PY:-.venv/bin/python}
GPUS=(${GPUS:-1 2 3})
OUT=${OUT:-outputs/stage1_v2}
P=configs/prune.yaml
PIPE=weights/FLUX.2-klein-base-4B
RES="--height 512 --width 512 --steps 20 --guidance_scale 4.0"
V1_CRITERIA="prune.double_head_metric=grad prune.neuron_metric=activation prune.mask_text_padding=false prune.cross_ratio_quantile=null"
WHAT=${1:-all}
mkdir -p $OUT

declare -A BLOCKS=(
  [G1]="double: [0, 1], single: []"
  [G2]="double: [2, 3], single: []"
  [G3]="double: [4], single: []"
  [G4]="double: [], single: [0, 1, 2, 3, 4, 5, 6, 7]"
  [G5]="double: [], single: [8, 9, 10, 11, 12, 13, 14, 15]"
  [G6]="double: [], single: [16, 17, 18, 19]"
)

# gen_eval <transformer|""> <seed> <dir> <gpu>：D_val 1000 张 -> FID / CLIP / HPSv2（逐 prompt）
gen_eval() {
  local tr=$1 seed=$2 d=$3 gpu=$4
  local targ=(); [[ -n $tr ]] && targ=(--transformer "$tr")
  if [[ ! -f $d/gen/index.jsonl ]]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/generate.py --pipeline $PIPE "${targ[@]}" --prompts data/val_prompts.txt \
      --out "$d/gen" $RES --seed "$seed" > "$d/gen.log" 2>&1
  fi
  [[ -f $d/eval/summary.json ]] || CUDA_VISIBLE_DEVICES=$gpu $PY scripts/evaluate.py --images "$d/gen" \
    --ref_dir data/val_reference_images --metrics fid clip hps --out "$d/eval" > "$d/eval.log" 2>&1
}

base_one() {  # 未剪枝基线（seed 42 也是敏感度实验的参照）
  local seed=$1 gpu=$2 d=$OUT/base/s$1
  [[ -f $d/eval/summary.json ]] && { echo "skip $d"; return; }
  mkdir -p "$d"; gen_eval "" "$seed" "$d" "$gpu"; echo "done $d"
}

# 单个敏感度实验：剪枝（单卡，记录速度场指标）-> 生成 D_val -> FID / CLIP / HPSv2
sens_one() {
  local g=$1 r=$2 gpu=$3 o=$OUT/sens/${1}_r$2
  [[ -f $o/eval/summary.json ]] && { echo "skip $o"; return; }
  mkdir -p "$o"
  if [[ ! -f $o/history.json ]]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/prune.py --config $P --out "$o" --set \
      "groups=[{name: $g, ${BLOCKS[$g]}, init: $r, min: $r, max: $r}]" $V1_CRITERIA \
      apra.rounds=1 apra.rollback=false prune.prune_blocks=false recover.steps=0 eval.metric=velocity_mse \
      "eval.aux_metrics=[traj_velocity_mse]" eval.traj_prompts=32 > "$o/prune.log" 2>&1
  fi
  gen_eval "$o/pruned" 42 "$o" "$gpu"
  [[ -f $o/eval/summary.json ]] && rm -rf "$o/pruned"   # 剪枝权重只用于生成（结构见 history.json）
  echo "done $o"
}

if [[ $WHAT == importance || $WHAT == all ]]; then
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/analyze_importance.py --config $P --out $OUT/importance --seeds 0 1 \
    > $OUT/importance.log 2>&1
fi

if [[ $WHAT == base || $WHAT == sens || $WHAT == all ]]; then
  jobs_list=()
  for s in 42 43 44; do jobs_list+=("base $s"); done
  if [[ $WHAT != base ]]; then
    for r in 0.2 0.3 0.1; do for g in G1 G2 G3 G4 G5 G6; do jobs_list+=("sens $g $r"); done; done
  fi
  # 每张卡一个队列，按顺序取任务（人工标注覆盖的 r=0.2/0.3 优先）
  for i in "${!GPUS[@]}"; do
    (
      for ((j = i; j < ${#jobs_list[@]}; j += ${#GPUS[@]})); do
        set -- ${jobs_list[$j]}
        if [[ $1 == base ]]; then base_one "$2" "${GPUS[$i]}" || echo "FAILED base $2"
        else sens_one "$2" "$3" "${GPUS[$i]}" || echo "FAILED $2 $3"; fi
      done
    ) &
  done
  wait
fi

if [[ $WHAT == plot || $WHAT == all ]]; then
  $PY scripts/plot_stage1.py --root $OUT
fi

if [[ $WHAT == apra_debug ]]; then
  o=$OUT/apra_debug
  mkdir -p $o
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/prune.py --config $P --out $o --set eval.metric=velocity_mse \
    apra.rounds=2 recover.steps=200 recover.block_steps=200 recover.log_every=10 2>&1 | tee $o/prune.log
fi
