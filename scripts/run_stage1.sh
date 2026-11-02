#!/usr/bin/env bash
# 阶段1：剪枝前分析与调通。各步骤可单独运行：
#   bash scripts/run_stage1.sh importance   # 1.1 三粒度重要性统计与可视化（两组校准种子）
#   bash scripts/run_stage1.sh sens         # 1.2 逐组敏感度：G1..G6 单独剪 0.1/0.2/0.3，无恢复训练
#   bash scripts/run_stage1.sh apra_debug   # 1.3 APRA 调通（velocity_mse、2 轮、200 步恢复），记录耗时
# 1.2 的 FID 与阶段0 的 fidvar/gen_s42 使用同一套生成设置（generate.py，seed 42），ΔFID 以其为基线。
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_MODULE_LOADING=LAZY
export HF_HOME=$PWD/weights/hf_home TORCH_HOME=$PWD/weights/torch_home
PY=${PY:-../.venv/bin/python}
GPUS=(${GPUS:-0 1 2 3})
OUT=outputs/stage1
P=configs/prune.yaml
PIPE=weights/FLUX.2-klein-base-4B
RES="--height 512 --width 512 --steps 20 --guidance_scale 4.0 --seed 42"
WHAT=${1:-all}

declare -A BLOCKS=(
  [G1]="double: [0, 1], single: []"
  [G2]="double: [2, 3], single: []"
  [G3]="double: [4], single: []"
  [G4]="double: [], single: [0, 1, 2, 3, 4, 5, 6, 7]"
  [G5]="double: [], single: [8, 9, 10, 11, 12, 13, 14, 15]"
  [G6]="double: [], single: [16, 17, 18, 19]"
)

if [[ $WHAT == importance || $WHAT == all ]]; then
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/analyze_importance.py --config $P --out $OUT/importance --seeds 0 1
fi

# 单个敏感度实验：剪枝（单卡）-> 生成 D_val（单卡）-> FID
sens_one() {
  local g=$1 r=$2 gpu=$3 o=$OUT/sens/${1}_r$2
  [[ -f $o/eval/summary.json ]] && { echo "skip $o"; return; }
  mkdir -p "$o"
  if [[ ! -f $o/pruned/prune_spec.json ]]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/prune.py --config $P --out "$o" --set \
      "groups=[{name: $g, ${BLOCKS[$g]}, init: $r, min: $r, max: $r}]" \
      apra.rounds=1 prune.prune_blocks=false recover.steps=0 eval.metric=velocity_mse > "$o/prune.log" 2>&1
  fi
  if [[ ! -f $o/gen/index.jsonl ]]; then
    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/generate.py --pipeline $PIPE --transformer "$o/pruned" \
      --prompts data/val_prompts.txt --out "$o/gen" $RES > "$o/gen.log" 2>&1
  fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/evaluate.py --images "$o/gen" --ref_dir data/val_reference_images \
    --metrics fid --out "$o/eval" > "$o/eval.log" 2>&1
  rm -rf "$o/pruned"   # 剪枝权重只用于生成，FID 出来后删除以省磁盘（结构见 history.json）
}

if [[ $WHAT == sens || $WHAT == all ]]; then
  jobs_list=()
  for g in G1 G2 G3 G4 G5 G6; do for r in 0.1 0.2 0.3; do jobs_list+=("$g $r"); done; done
  # 每张卡一个队列，按顺序取任务
  for i in "${!GPUS[@]}"; do
    (
      for ((j = i; j < ${#jobs_list[@]}; j += ${#GPUS[@]})); do
        sens_one ${jobs_list[$j]} "${GPUS[$i]}" || echo "FAILED ${jobs_list[$j]}"
      done
    ) &
  done
  wait
  $PY - <<'EOF'
import glob, json, os
base = json.load(open("outputs/stage0/fidvar/summary.json"))
rows = []
for d in sorted(glob.glob("outputs/stage1/sens/G*_r*")):
    g, r = os.path.basename(d).split("_r")
    h = json.load(open(f"{d}/history.json"))
    fid = json.load(open(f"{d}/eval/summary.json"))["fid"]
    rd = h["rounds"][0]
    rows.append({"group": g, "rate": float(r), "param_ratio": rd["param_ratio"], "vel_mse": rd["score"],
                 "fid": fid, "dfid": fid - base["fid"]["42"]})
json.dump({"fid_base": base["fid"]["42"], "fid_std": base["std"], "rows": rows},
          open("outputs/stage1/sens/summary.json", "w"), indent=2)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for g in sorted({x["group"] for x in rows}):
    rs = sorted((x for x in rows if x["group"] == g), key=lambda x: x["rate"])
    axes[0].errorbar([x["rate"] for x in rs], [x["dfid"] for x in rs], yerr=base["std"], marker="o", capsize=3, label=g)
    axes[1].plot([x["rate"] for x in rs], [x["vel_mse"] for x in rs], marker="o", label=g)
axes[0].set(xlabel="prune rate", ylabel="ΔFID vs unpruned (D_val)", title="per-group sensitivity (no recovery)")
axes[0].set_yscale("symlog", linthresh=2)  # G1 崩溃（ΔFID>200）时其余组仍可读
axes[1].set(xlabel="prune rate", ylabel="velocity MSE ×100", yscale="log")
axes[0].legend(); fig.tight_layout(); fig.savefig("outputs/stage1/sens/sensitivity.png", dpi=150)
for x in rows:
    print(x)
EOF
fi

if [[ $WHAT == apra_debug || $WHAT == all ]]; then
  o=$OUT/apra_debug
  mkdir -p $o
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/prune.py --config $P --out $o --set eval.metric=velocity_mse \
    apra.rounds=2 recover.steps=200 recover.block_steps=200 recover.log_every=10 2>&1 | tee $o/prune.log
fi
