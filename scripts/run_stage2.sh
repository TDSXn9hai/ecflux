#!/usr/bin/env bash
# 阶段2 v2：剪枝主实验 + 核心消融（默认用 GPU1-3）。每个实验一个目录 outputs/stage2_v2/<name>/，已完成的自动跳过；
# 开始前在 <name>/.claim 占位，多张卡同时跑同一模式时不会重复执行同一实验（失败会删除占位以便重试）。
# APRA 反馈为双闸门（轨迹速度场 MSE + HPSv2 200 条，见 configs/prune.yaml 的 eval / apra.gates，依据 outputs/stage1_v2）。
#   GPUS="1 2 3" bash scripts/run_stage2.sh calib  # 阈值标定：新初值剪一轮 + 2000 步恢复，recover.seed=1/2/3 各一次（并行）
#   GPU=1 bash scripts/run_stage2.sh drift     # 漂移校正参照（scripts/drift_ref.py），其余实验的每轮评估会等待对应 checkpoint
#   GPU=1 bash scripts/run_stage2.sh main      # Ours-PruneOnly 主实验（APRA 双闸门 + 块剪枝）+ 换恢复种子的重复运行 full_rep
#   GPU=3 bash scripts/run_stage2.sh indep     # 不依赖主实验结果的消融：粒度 / 重要性判据 / 模态感知
#   GPU=2 bash scripts/run_stage2.sh matched   # 依赖主实验参数量的消融：固定比例一次性剪枝 / 统一剪枝率（参数量对齐）
#   GPU=3 bash scripts/run_stage2.sh block     # 块剪枝时机与 τ（优先级最低）
#   GPU=2 bash scripts/run_stage2.sh queue     # 依次执行 indep、block、matched（多张卡各起一个即可并行分担；full 未完成时跳过 matched）
#   GPU=2 bash scripts/run_stage2.sh eval      # 对所有已完成实验在 D_val 上生成并评估 FID / CLIP / HPSv2
#   bash scripts/run_stage2.sh summary         # 汇总成 outputs/stage2_v2/summary.md 并画图（scripts/plot_stage2.py）
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_MODULE_LOADING=LAZY
export HF_HOME=$PWD/weights/hf_home TORCH_HOME=$PWD/weights/torch_home HF_HUB_OFFLINE=1
PY=${PY:-.venv/bin/python}
GPU=${GPU:-1}
OUT=${OUT:-outputs/stage2_v2}
P=configs/prune.yaml
PIPE=weights/FLUX.2-klein-base-4B
TARGET=${TARGET:-0.75}   # 细粒度阶段的目标参数比例；之后块剪枝再删约 3 个 Single Block，最终约 0.65（≈2.5B，开题设定）
COMMON="apra.target_param_ratio=$TARGET"
WHAT=${1:-main}
mkdir -p $OUT

# claim <name>：原子地占位（mkdir 成功即占到）
claim() { mkdir -p "$OUT/$1" && mkdir "$OUT/$1/.claim" 2>/dev/null; }

# run_prune <name> [--set 覆盖...]
run_prune() {
  local name=$1; shift
  local o=$OUT/$name
  if [[ -f $o/pruned/prune_spec.json ]]; then echo "skip $name"; return; fi
  [[ -n ${CLAIMED:-} ]] || claim "$name" || { echo "skip $name (claimed by another worker)"; return; }
  echo "[$(date +%F' '%T)] start $name on GPU$GPU"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/prune.py --config $P --out "$o" --set $COMMON "$@" > "$o/prune.log" 2>&1 \
    || { echo "FAILED $name (see $o/prune.log)"; rmdir "$o/.claim"; return 0; }
  echo "[$(date +%F' '%T)] done $name"
}

# fine_ratio <name>：细粒度阶段（块剪枝前）最后一个被接受的轮次的参数比例
fine_ratio() {
  $PY -c "
import json; h = json.load(open('$OUT/$1/history.json'))
print([r for r in h['rounds'] if not r.get('rejected')][-1]['param_ratio'])"
}

# dry_ratio <groups-yaml>：不训练、不评估，只按给定剪枝率剪一次，返回参数比例
dry_ratio() {
  local d=$OUT/_dry_gpu$GPU
  rm -rf "$d"
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/prune.py --config $P --out "$d" --set "groups=$1" apra.enabled=false \
    prune.prune_blocks=false recover.steps=0 eval.metric=velocity_mse "eval.aux_metrics=[]" apra.gates=null \
    eval.proxy_batches=1 calib.num_batches=8 prune.head_ablation_batches=2 > /dev/null 2>&1
  $PY -c "import json; print(json.load(open('$d/history.json'))['rounds'][0]['param_ratio'])"
}

# groups_scaled <s> / groups_uniform <u>：生成 --set groups=... 用的 YAML
groups_yaml() {
  $PY - "$1" "$2" <<'PYEOF'
import sys, yaml
mode, v = sys.argv[1], float(sys.argv[2])
gs = yaml.safe_load(open("configs/prune.yaml"))["groups"]
for g in gs:
    g["init"] = round(min(0.5, g["init"] * v) if mode == "scale" else v, 4)
print(yaml.safe_dump(gs, default_flow_style=True, width=10**6).strip())
PYEOF
}

# 二分搜索使细粒度参数比例与主实验一致；输出 groups YAML
match_groups() {
  local mode=$1 target=$2 lo hi mid r g
  if [[ $mode == scale ]]; then lo=0.5; hi=6.0; else lo=0.02; hi=0.5; fi
  for _ in 1 2 3 4 5 6 7; do
    mid=$($PY -c "print(($lo + $hi) / 2)")
    g=$(groups_yaml "$mode" "$mid")
    r=$(dry_ratio "$g")
    echo "  match $mode=$mid -> ratio $r (target $target)" >&2
    if $PY -c "import sys; sys.exit(0 if $r > $target else 1)"; then lo=$mid; else hi=$mid; fi
  done
  groups_yaml "$mode" "$($PY -c "print(($lo + $hi) / 2)")"
}

if [[ $WHAT == calib ]]; then
  GPUS=(${GPUS:-1 2 3}); s=1
  for g in "${GPUS[@]}"; do
    ( GPU=$g; run_prune calib_seed$s apra.rounds=1 apra.rollback=false apra.target_param_ratio=null \
        prune.prune_blocks=false recover.seed=$s ) &
    s=$((s + 1))
  done
  wait
  $PY - <<'PYEOF'
import glob, json, statistics as st
rows = []
for d in sorted(glob.glob("outputs/stage2_v2/calib_seed*/history.json")):
    r = json.load(open(d))["rounds"][0]
    rows.append({"run": d.split("/")[-2], "param_ratio": r["param_ratio"], "traj_velocity_mse": r["aux"]["traj_velocity_mse"],
                 "hps": r["aux"]["hps"], "hps_abs": r["aux"]["hps_abs"]})
out = {"rows": rows}
if len(rows) >= 2:
    sd = {m: st.stdev([x[m] for x in rows]) for m in ("traj_velocity_mse", "hps")}
    out |= {"sigma": sd, "suggested_gates": {m: {"low": sd[m], "high": 3 * sd[m]} for m in sd}}
json.dump(out, open("outputs/stage2_v2/calib_summary.json", "w"), indent=2)
print(json.dumps(out, indent=2))
PYEOF
fi

if [[ $WHAT == drift ]]; then
  # 漂移校正参照：不剪枝、按恢复训练设置连续训练 6 × 2000 步，每段一个 checkpoint（APRA 在 eval.drift_ref 中等待使用）
  mkdir -p $OUT/drift_ref
  CUDA_VISIBLE_DEVICES=$GPU $PY scripts/drift_ref.py --config $P --out $OUT/drift_ref --chunks 6 > $OUT/drift_ref/run.log 2>&1
fi

if [[ $WHAT == main ]]; then
  run_prune full
  run_prune full_rep recover.seed=43   # 只换恢复训练种子，估计整条 APRA 流程的运行间波动
fi

if [[ $WHAT == indep || $WHAT == queue ]]; then
  # 粒度
  run_prune head_only    prune.prune_neurons=false prune.prune_blocks=false
  run_prune neuron_only  prune.prune_heads=false prune.prune_blocks=false
  run_prune no_block     prune.prune_blocks=false
  # 重要性判据（magnitude / random 时 Double Block 也不用逐头删除评分）
  run_prune head_grad_only prune.double_head_metric=grad       # 开题原判据：Double Block 也用梯度敏感度
  run_prune head_mag     prune.head_metric=magnitude
  run_prune neuron_act   prune.neuron_metric=activation   # 开题原神经元判据（激活 L1）
  run_prune random       prune.head_metric=random prune.neuron_metric=random
  # 模态感知
  run_prune no_modality  prune.modality_aware=false
fi

if [[ $WHAT == block || $WHAT == queue ]]; then
  run_prune block_before block.stage=before
  # max_blocks 放开到 6：旧版删块数被 max_blocks=4 截住，τ 在 0.88–0.94 内不起作用
  for TAU in 0.88 0.90 0.94; do run_prune tau_$TAU block.tau=$TAU block.max_blocks=6; done
fi

if [[ $WHAT == matched || $WHAT == queue ]] && [[ ! -f $OUT/full/history.json ]]; then
  echo "need $OUT/full first, skip matched"
elif [[ $WHAT == matched || $WHAT == queue ]]; then
  T=$(fine_ratio full)
  echo "main fine-stage param ratio: $T"
  # 只有一轮，rollback 必须关闭，否则这一轮超阈值时会被撤回、只剩块剪枝
  for spec in "fixed_ratio scale" "uniform_rate uniform"; do
    set -- $spec
    [[ -f $OUT/$1/pruned/prune_spec.json ]] && { echo "skip $1"; continue; }
    claim "$1" || { echo "skip $1 (claimed by another worker)"; continue; }   # 先占位再搜索剪枝率，避免重复做 dry run
    g=$(match_groups "$2" "$T")
    CLAIMED=1 run_prune "$1" apra.enabled=false apra.rollback=false apra.target_param_ratio=null "groups=$g"
  done
  rm -rf "$OUT/_dry_gpu$GPU"
fi

if [[ $WHAT == eval ]]; then
  for d in $OUT/*/; do
    n=$(basename "$d"); [[ -f $d/pruned/prune_spec.json ]] || continue
    [[ $n == calib_* || $n == _dry* ]] && continue   # 标定运行只用于估计阈值
    [[ -f $d/eval_dval/summary.json ]] && continue
    mkdir "$d/.eval_claim" 2>/dev/null || continue   # 另一张卡正在评估
    [[ -f $d/gen_dval/index.jsonl ]] || CUDA_VISIBLE_DEVICES=$GPU $PY scripts/generate.py --pipeline $PIPE \
      --transformer "$d/pruned" --prompts data/val_prompts.txt --out "$d/gen_dval" --height 512 --width 512 \
      --steps 20 --guidance_scale 4.0 --seed 42 > "$d/gen_dval.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/evaluate.py --images "$d/gen_dval" --ref_dir data/val_reference_images \
      --metrics fid clip hps --out "$d/eval_dval" > "$d/eval_dval.log" 2>&1
    CUDA_VISIBLE_DEVICES=$GPU $PY scripts/evaluate.py --transformer "$d/pruned" --metrics efficiency --resolution 512 \
      --steps 40 --out "$d/eff_512" > /dev/null 2>&1
    [[ -f $d/eval_dval/summary.json ]] || rmdir "$d/.eval_claim"   # 失败时释放占位
    echo "evaluated $n"
  done
fi

if [[ $WHAT == summary ]]; then
  $PY scripts/plot_stage2.py --root $OUT   # summary.md + 全部图表
fi
