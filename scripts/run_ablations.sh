#!/usr/bin/env bash
# 消融实验一键脚本（按需注释）。每个实验输出独立目录，结果见 */history.json、*/final、outputs/eval/*/summary.json
# 用法: bash scripts/run_ablations.sh [prune|distill|router|all]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
WHAT=${1:-all}
P=configs/prune.yaml
D=configs/distill.yaml
R=configs/router.yaml

if [[ $WHAT == prune || $WHAT == all ]]; then
  # ---- 4.2 剪枝：粒度 / 重要性判据 / 模态感知 / APRA / 块剪枝时机
  $PY scripts/prune.py --config $P --out outputs/prune/full
  $PY scripts/prune.py --config $P --out outputs/prune/head_only    --set prune.prune_neurons=false prune.prune_blocks=false
  $PY scripts/prune.py --config $P --out outputs/prune/neuron_only  --set prune.prune_heads=false prune.prune_blocks=false
  $PY scripts/prune.py --config $P --out outputs/prune/no_block     --set prune.prune_blocks=false
  $PY scripts/prune.py --config $P --out outputs/prune/head_mag     --set prune.head_metric=magnitude
  $PY scripts/prune.py --config $P --out outputs/prune/neuron_mag   --set prune.neuron_metric=magnitude
  $PY scripts/prune.py --config $P --out outputs/prune/random       --set prune.head_metric=random prune.neuron_metric=random
  $PY scripts/prune.py --config $P --out outputs/prune/no_modality  --set prune.modality_aware=false
  $PY scripts/prune.py --config $P --out outputs/prune/fixed_ratio  --set apra.enabled=false
  $PY scripts/prune.py --config $P --out outputs/prune/uniform_rate --set apra.uniform_rate=0.2
  $PY scripts/prune.py --config $P --out outputs/prune/apra_pergroup --set apra.feedback=per_group
  $PY scripts/prune.py --config $P --out outputs/prune/apra_rollback --set apra.rollback=true
  $PY scripts/prune.py --config $P --out outputs/prune/block_before --set block.stage=before
  for TAU in 0.88 0.90 0.94; do
    $PY scripts/prune.py --config $P --out outputs/prune/tau_$TAU --set block.tau=$TAU
  done
fi

if [[ $WHAT == distill || $WHAT == all ]]; then
  # ---- 4.3 蒸馏：三级损失逐项移除 / 鲁棒扰动 / 递进权重 / k / 对齐形式
  S="student=outputs/prune/full/pruned"
  $PY scripts/distill.py --config $D --out outputs/distill/full      --set $S
  $PY scripts/distill.py --config $D --out outputs/distill/no_vel    --set $S distill.use_vel=false distill.lambda_gt=1.0
  $PY scripts/distill.py --config $D --out outputs/distill/no_hid    --set $S distill.use_hid=false
  $PY scripts/distill.py --config $D --out outputs/distill/no_attn   --set $S distill.use_attn=false
  $PY scripts/distill.py --config $D --out outputs/distill/vel_only  --set $S distill.use_hid=false distill.use_attn=false
  $PY scripts/distill.py --config $D --out outputs/distill/no_robust --set $S distill.robust=false
  $PY scripts/distill.py --config $D --out outputs/distill/no_progressive --set $S distill.progressive=false
  $PY scripts/distill.py --config $D --out outputs/distill/hid_mse   --set $S distill.hidden_kind=mse
  $PY scripts/distill.py --config $D --out outputs/distill/attn_kl   --set $S distill.attn_kind=kl
  for K in 1 3 5; do
    $PY scripts/distill.py --config $D --out outputs/distill/attn_k$K --set $S distill.attn_k=$K
  done
  # Ours-PruneOnly 基线 = outputs/prune/full/pruned（不蒸馏），评估时与 outputs/distill/*/final 对比
fi

if [[ $WHAT == router || $WHAT == all ]]; then
  # ---- 4.4 判别器：输入特征 / 预算；协同：基线与级联
  L=data/router/labels.jsonl
  for F in emb struct emb+struct; do
    $PY scripts/train_router.py --config $R --labels $L --out outputs/router/feat_$F --set router.features=$F
  done
  for B in 0.1 0.2 0.3 0.5; do
    $PY scripts/train_router.py --config $R --labels $L --out outputs/router/budget_$B --set calibration.cloud_budget=$B
  done
  E="--edge_transformer outputs/distill/full/final --prompts data/test_prompts.txt"
  $PY scripts/collab_infer.py --config $R $E --out outputs/collab/all_edge  --set collab.mode=edge
  $PY scripts/collab_infer.py --config $R $E --out outputs/collab/all_cloud --set collab.mode=cloud
  $PY scripts/collab_infer.py --config $R $E --out outputs/collab/random30  --set collab.mode=random collab.random_ratio=0.3
  $PY scripts/collab_infer.py --config $R $E --router outputs/router/feat_emb+struct/router.pt --out outputs/collab/router_cascade
  $PY scripts/collab_infer.py --config $R $E --router outputs/router/feat_emb+struct/router.pt --out outputs/collab/router_nocascade --set collab.cascade=false
fi
