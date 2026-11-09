#!/usr/bin/env bash
# 阶段2：剪枝主实验 + 核心消融（只用 GPU2/GPU3）。每个实验一个目录 outputs/stage2/<name>/，已完成的自动跳过；
# 开始前在 <name>/.claim 占位，多张卡同时跑同一模式时不会重复执行同一实验（失败会删除占位以便重试）。
#   GPU=2 bash scripts/run_stage2.sh main      # Ours-PruneOnly 主实验（APRA + FID 反馈 + 块剪枝）
#   GPU=3 bash scripts/run_stage2.sh indep     # 不依赖主实验结果的消融：粒度 / 重要性判据 / 模态感知
#   GPU=2 bash scripts/run_stage2.sh matched   # 依赖主实验参数量的消融：固定比例一次性剪枝 / 统一剪枝率（参数量对齐）
#   GPU=3 bash scripts/run_stage2.sh block     # 块剪枝时机与 τ（优先级最低）
#   GPU=2 bash scripts/run_stage2.sh queue     # 依次执行 matched、indep、block（两张卡各起一个即可并行分担）
#   GPU=2 bash scripts/run_stage2.sh eval      # 对所有已完成实验在 D_val 上生成并评估 FID / CLIP / HPSv2
#   GPU=2 bash scripts/run_stage2.sh summary   # 汇总成 outputs/stage2/summary.md
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_MODULE_LOADING=LAZY
export HF_HOME=$PWD/weights/hf_home TORCH_HOME=$PWD/weights/torch_home HF_HUB_OFFLINE=1
PY=${PY:-../.venv/bin/python}
GPU=${GPU:-2}
OUT=outputs/stage2
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
    prune.prune_blocks=false recover.steps=0 eval.metric=velocity_mse eval.proxy_batches=1 calib.num_batches=8 \
    prune.head_ablation_batches=2 > /dev/null 2>&1
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

if [[ $WHAT == main ]]; then
  run_prune full
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

if [[ $WHAT == matched || $WHAT == queue ]]; then
  [[ -f $OUT/full/history.json ]] || { echo "need $OUT/full first"; exit 1; }
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

if [[ $WHAT == block || $WHAT == queue ]]; then
  run_prune block_before block.stage=before
  for TAU in 0.88 0.90 0.94; do run_prune tau_$TAU block.tau=$TAU; done
fi

if [[ $WHAT == eval ]]; then
  for d in $OUT/*/; do
    n=$(basename "$d"); [[ -f $d/pruned/prune_spec.json ]] || continue
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
  $PY - <<'PYEOF'
import glob, json, os
base_fid = json.load(open("outputs/stage0/fidvar/summary.json"))["fid"]["42"]
rows = ["| 实验 | DiT 参数 | 参数比例 | 删除的 Single Block | D_val FID (Δ) | CLIP | HPSv2 | 40 次前向 @512 | APRA 轮次（接受/总） |",
        "|---|---|---|---|---|---|---|---|---|"]
for d in sorted(glob.glob("outputs/stage2/*/history.json")):
    d = os.path.dirname(d); n = os.path.basename(d)
    h = json.load(open(f"{d}/history.json"))
    ev = json.load(open(f"{d}/eval_dval/summary.json")) if os.path.exists(f"{d}/eval_dval/summary.json") else {}
    ef = json.load(open(f"{d}/eff_512/summary.json"))["efficiency"] if os.path.exists(f"{d}/eff_512/summary.json") else {}
    p = h.get("efficiency", {}).get("params_M") or ef.get("params_M")
    acc = sum(not r.get("rejected") for r in h["rounds"])
    fid = f'{ev["fid"]:.2f} ({ev["fid"] - base_fid:+.2f})' if "fid" in ev else "-"
    lat = ef.get("latency_40steps_s")
    rows.append(f'| {n} | {p / 1e3:.3f}B | {p / h["params_orig"] * 1e6:.3f} | {h.get("removed_single_blocks", [])} | {fid} | '
                f'{ev.get("clip", float("nan")):.2f} | {ev.get("hps", float("nan")):.4f} | {lat if lat is None else round(lat, 2)} | {acc}/{len(h["rounds"])} |')
txt = f"未剪枝 4B：3.876B，D_val FID {base_fid:.2f}（σ=0.73）\n\n" + "\n".join(rows) + "\n"
open("outputs/stage2/summary.md", "w").write(txt); print(txt)
PYEOF
fi
