#!/usr/bin/env bash
# 阶段0：评估管线定标（FID 波动 -> 基线全量评估）。各步骤可单独运行：
#   bash scripts/run_stage0.sh fidvar      # 0.1 FID 随机波动（4B，D_val，seed 42/43/44 + 42 重复）
#   bash scripts/run_stage0.sh baseline    # 0.3 4B / 9B 在 COCO-30k、HPSv2、GenEval、T2I-CompBench 上生成并打分
#   bash scripts/run_stage0.sh bench       # 0.3 用 .venv-bench 跑 GenEval / T2I-CompBench 官方评测（需先 baseline）
#   bash scripts/run_stage0.sh efficiency  # 0.3 DiT 参数量 / 峰值显存 / 20 步延迟（512 与 1024）
# 环境变量：GPUS="0 1 2 3"（分片用的卡），MODELS="4B 9B"，PY=python 解释器
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_MODULE_LOADING=LAZY   # torch 2.14 在本机不设置时报 CUDA devices busy or unavailable
export HF_HOME=$PWD/weights/hf_home TORCH_HOME=$PWD/weights/torch_home
export HF_HUB_OFFLINE=1   # 评估模型都已缓存在 weights/hf_home；不设置时 transformers 会联网检查并超时
PY=${PY:-../.venv/bin/python}
GPUS=(${GPUS:-0 1 2 3})
MODELS=${MODELS:-4B 9B}
OUT=outputs/stage0
RES="--height 512 --width 512 --steps 20 --guidance_scale 4.0"
WHAT=${1:-all}

# gen_sharded <pipeline> <prompts> <out> [额外参数...]：按 GPUS 分片并行生成，结束后合并 index / timing
gen_sharded() {
  local pipe=$1 prompts=$2 out=$3; shift 3
  if [[ -f $out/index.jsonl ]]; then echo "skip $out (exists)"; return; fi
  mkdir -p "$out"
  local n=${#GPUS[@]} pids=()
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY scripts/generate.py --pipeline "$pipe" --prompts "$prompts" --out "$out" \
      $RES --shard_id "$i" --num_shards "$n" "$@" > "$out/gen_shard$i.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  $PY scripts/generate.py --pipeline "$pipe" --prompts "$prompts" --out "$out" --merge
}

# score <images> <out> <metrics...> [--ref_dir dir]：单卡打分
score() {
  local img=$1 out=$2; shift 2
  if [[ -f $out/summary.json ]]; then echo "skip $out (exists)"; return; fi
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/evaluate.py --images "$img" --out "$out" --metrics "$@"
}

pipe_of() { echo "weights/FLUX.2-klein-base-$1"; }

if [[ $WHAT == fidvar || $WHAT == all ]]; then
  for tag in 42 43 44 42rep; do
    seed=${tag%rep}
    gen_sharded "$(pipe_of 4B)" data/val_prompts.txt $OUT/fidvar/gen_s$tag --seed "$seed"
    score $OUT/fidvar/gen_s$tag $OUT/fidvar/eval_s$tag fid --ref_dir data/val_reference_images
  done
  $PY - <<'EOF'
import json, statistics as st
d = {t: json.load(open(f"outputs/stage0/fidvar/eval_s{t}/summary.json"))["fid"] for t in ("42", "43", "44", "42rep")}
v = [d["42"], d["43"], d["44"]]
r = {"fid": d, "mean": st.mean(v), "std": st.stdev(v), "repeat_diff": abs(d["42"] - d["42rep"])}
json.dump(r, open("outputs/stage0/fidvar/summary.json", "w"), indent=2)
print(r)
EOF
fi

if [[ $WHAT == baseline || $WHAT == all ]]; then
  B=data/benchmarks
  for M in $MODELS; do
    P=$(pipe_of "$M")
    G=$OUT/baseline/$M
    gen_sharded "$P" data/coco30k.txt $G/coco30k
    score $G/coco30k $G/eval_coco30k fid clip --ref_dir data/coco30k_images
    for S in anime concept-art paintings photo; do
      gen_sharded "$P" $B/hpsv2/$S.txt $G/hpsv2_$S
      score $G/hpsv2_$S $G/eval_hpsv2_$S hps
    done
    gen_sharded "$P" $B/geneval/prompts/evaluation_metadata.jsonl $G/geneval --format geneval --n_per_prompt 4
    for C in spatial non_spatial complex color shape texture; do
      gen_sharded "$P" $B/T2I-CompBench/examples/dataset/${C}_val.txt $G/t2i_$C --format t2icompbench --n_per_prompt 4
    done
  done
fi

# GenEval / T2I-CompBench 官方评测（.venv-bench：torch 2.1.2 + mmcv-full 1.7.2 + mmdet 2.28 + detectron2）
if [[ $WHAT == bench || $WHAT == all ]]; then
  BP=${BENCH_PY:-$(cd .. && pwd)/.venv-bench/bin/python}
  SP=$($BP -c "import os, mmdet; print(os.path.dirname(mmdet.__file__))")
  T=$PWD/data/benchmarks/T2I-CompBench
  export HF_HUB_OFFLINE=1   # bert-base-uncased 已离线放在 weights/hf_home
  for M in $MODELS; do
    G=$PWD/$OUT/baseline/$M
    if [[ ! -f $G/eval_geneval/results.jsonl ]]; then
      (cd data/benchmarks/geneval && CUDA_VISIBLE_DEVICES=${GPUS[0]} $BP evaluation/evaluate_images.py "$G/geneval" \
        --outfile "$G/eval_geneval/results.jsonl" --model-path "$PWD/../../../weights/benchmark_eval" \
        --model-config "$SP/.mim/configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py")
    fi
    (cd data/benchmarks/geneval && $BP evaluation/summary_scores.py "$G/eval_geneval/results.jsonl") > "$G/eval_geneval/summary.txt"
    for C in color shape texture complex; do
      [[ -f $G/t2i_$C/annotation_blip/vqa_result.json ]] || \
        (cd $T/BLIPvqa_eval && CUDA_VISIBLE_DEVICES=${GPUS[0]} $BP BLIP_vqa.py --out_dir="$G/t2i_$C/")
    done
    for C in spatial complex; do
      extra=(); [[ $C == complex ]] && extra=(--complex=True)
      [[ -f $G/t2i_$C/labels/annotation_obj_detection_2d/vqa_result.json ]] || \
        (cd $T/UniDet_eval && CUDA_VISIBLE_DEVICES=${GPUS[0]} $BP 2D_spatial_eval.py --outpath="$G/t2i_$C/" "${extra[@]}")
    done
    for C in non_spatial complex; do
      extra=(); [[ $C == complex ]] && extra=(--complex=True)
      [[ -f $G/t2i_$C/annotation_clip/vqa_result.json ]] || \
        (cd $T && CUDA_VISIBLE_DEVICES=${GPUS[0]} $BP CLIPScore_eval/CLIP_similarity.py --outpath="$G/t2i_$C/" "${extra[@]}")
    done
    # 官方 3_in_1.py 写死每条 prompt 10 张图；这里按相同加权规则、以 4 张/条重算（question_id = prompt 下标 × 4 + k）
    $PY - "$G/t2i_complex" "$T/examples/dataset" 4 <<'EOF'
import json, os, sys
out, data, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
ld = lambda p: [float(x["answer"]) for x in sorted(json.load(open(os.path.join(out, p))), key=lambda x: x["question_id"])]
attr, spat, act = ld("annotation_blip/vqa_result.json"), ld("labels/annotation_obj_detection_2d/vqa_result.json"), \
    ld("annotation_clip/vqa_result.json")
rd = lambda f: [l.strip("\n").split(".")[0].lower() for l in open(os.path.join(data, f))]
sp, ac, prompts = set(rd("complex_val_spatial.txt")), set(rd("complex_val_action.txt")), rd("complex_val.txt")
tot = []
for i, p in enumerate(prompts):
    for j in range(i * n, (i + 1) * n):
        tot.append((spat[j] + attr[j]) / 2 if p in sp else (act[j] + attr[j]) / 2 if p in ac else (attr[j] + spat[j] + act[j]) / 3)
os.makedirs(os.path.join(out, "annotation_3_in_1"), exist_ok=True)
json.dump([{"question_id": i, "answer": s} for i, s in enumerate(tot)], open(os.path.join(out, "annotation_3_in_1/vqa_result.json"), "w"))
open(os.path.join(out, "annotation_3_in_1/avg_score.txt"), "w").write(f"{sum(tot) / len(tot)}\n")
print("3-in-1 avg:", sum(tot) / len(tot))
EOF
  done
fi

if [[ $WHAT == efficiency || $WHAT == all ]]; then
  for M in $MODELS; do
    for R in 512 1024; do
      o=$OUT/efficiency/${M}_$R
      [[ -f $o/summary.json ]] && continue
      CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/evaluate.py --transformer "$(pipe_of "$M")" --metrics efficiency \
        --resolution $R --steps 40 --out "$o"   # CFG：20 步 × 2 次前向
    done
  done
fi
