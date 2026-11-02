#!/usr/bin/env bash
# ecflux 一键安装脚本：配置环境 + 下载权重 + 下载/构建数据集。
#
#   bash install.sh [env] [weights] [data] [all]      # 默认 all
#
# 常用环境变量：
#   HF_TOKEN                9B 教师为 gated 仓库，设置后才会下载；未设置则跳过
#   ECFLUX_TORCH_CUDA       torch 的 CUDA 轮子变体，默认 cu126（README 的 cu130 在旧驱动上不可用）
#   ECFLUX_VENV             虚拟环境路径，默认 <仓库根>/.venv
#   ECFLUX_PROXY            外网不通时使用的 HTTP 代理（如内网 http://host:port）；默认不使用代理
#   ECFLUX_FETCH_TRAINSET   =1 时下载 306GB 的 Pexels 训练集替代并打分；默认不下
#   ECFLUX_HF_ENDPOINT      备选 HF 镜像，如 https://hf-mirror.com
#
# 本脚本只在本仓库目录内创建 .venv 和下载文件，不改动机器上其他环境。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

VENV="${ECFLUX_VENV:-$HERE/.venv}"
PY="$VENV/bin/python"
PYVER="${ECFLUX_PYVER:-3.11}"
TORCH_CUDA="${ECFLUX_TORCH_CUDA:-cu126}"
PROXY="${ECFLUX_PROXY:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HERE/.uv_cache}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export HF_HOME="${HF_HOME:-$HERE/weights/hf_home}"
export TORCH_HOME="${TORCH_HOME:-$HERE/weights/torch_home}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
[ -n "${ECFLUX_HF_ENDPOINT:-}" ] && export HF_ENDPOINT="$ECFLUX_HF_ENDPOINT"

log() { printf '\n\033[1;34m[ecflux]\033[0m %s\n' "$*"; }
die() { printf '\n\033[1;31m[ecflux ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

command -v uv  >/dev/null 2>&1 || die "需要 uv（https://docs.astral.sh/uv/）。先安装 uv 再运行本脚本。"
command -v git >/dev/null 2>&1 || die "需要 git。"

# 外网探测：huggingface 与 pytorch 两个域都要可达。任一直连不通就改用代理
setup_net() {
  local probe="https://huggingface.co https://download.pytorch.org/whl/cu126/torch/"
  local u
  reach() { curl -sI --retry 2 --retry-all-errors --max-time 15 ${1:+-x "$1"} "$2" >/dev/null 2>&1; }

  local ok_direct=1
  for u in $probe; do reach "" "$u" || ok_direct=0; done
  if [ "$ok_direct" = 1 ]; then log "外网直连可用"; return; fi

  if [ -z "$PROXY" ]; then
    log "外网直连不完整且未设置 ECFLUX_PROXY；如在内网请 export ECFLUX_PROXY=http://host:port 后重试（也可设 ECFLUX_HF_ENDPOINT 用镜像）"
    return
  fi
  local ok_proxy=1
  for u in $probe; do reach "$PROXY" "$u" || ok_proxy=0; done
  if [ "$ok_proxy" = 1 ]; then
    export https_proxy="$PROXY" http_proxy="$PROXY" HTTPS_PROXY="$PROXY" HTTP_PROXY="$PROXY"
    log "使用代理 $PROXY"
  else
    log "代理 $PROXY 未完全连通，继续尝试（可设 ECFLUX_HF_ENDPOINT 用镜像）"
  fi
}
# ───────────────────────── 环境 ─────────────────────────
stage_env() {
  log "创建虚拟环境 $VENV (python $PYVER)"
  uv venv "$VENV" --python "$PYVER"

  log "安装 torch 2.14.1 / torchvision 0.29.1 ($TORCH_CUDA)"
  uv pip install --python "$PY" torch==2.14.1 torchvision==0.29.1 \
    --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match

  log "安装 requirements.txt（torch/torchvision/hpsv2 单独处理）"
  grep -vE '^(torch==|torchvision==|hpsv2==)' requirements.txt > "$HERE/.req_core.txt"
  uv pip install --python "$PY" -r "$HERE/.req_core.txt" \
    "torch==2.14.1+$TORCH_CUDA" "torchvision==0.29.1+$TORCH_CUDA" \
    --index-url https://pypi.org/simple \
    --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA" --index-strategy unsafe-best-match
  rm -f "$HERE/.req_core.txt"

  log "安装 hpsv2（--no-deps，其依赖较旧，手动补运行依赖）"
  uv pip install --python "$PY" hpsv2==1.2.0 --no-deps
  uv pip freeze --python "$PY" | grep -E '^(torch|torchvision|protobuf|transformers|huggingface)' > "$HERE/.con.txt"
  uv pip install --python "$PY" ftfy einops pandas braceexpand fsspec sentencepiece timm \
    webdataset pyarrow clint regex -c "$HERE/.con.txt" \
    --index-url https://pypi.org/simple \
    --extra-index-url "https://download.pytorch.org/whl/$TORCH_CUDA" --index-strategy unsafe-best-match
  rm -f "$HERE/.con.txt"

  log "补齐 hpsv2 缺失的 BPE 词表"
  local d
  d="$("$PY" -c 'import os,hpsv2;print(os.path.join(os.path.dirname(hpsv2.__file__),"src","open_clip"))')"
  local bpe="$d/bpe_simple_vocab_16e6.txt.gz"
  if [ ! -s "$bpe" ]; then
    curl -fSL --retry 5 -o "$bpe" \
      https://github.com/mlfoundations/open_clip/raw/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz \
      || log "BPE 词表下载失败，HPSv2 import 会报错，可稍后手动补到 $bpe"
  fi

  log "校验 import"
  CUDA_MODULE_LOADING=LAZY "$PY" - <<'PYEOF'
import torch, diffusers, transformers, hpsv2, bitsandbytes, torchmetrics  # noqa
from diffusers import Flux2Transformer2DModel, Flux2KleinPipeline, AutoencoderKLFlux2  # noqa
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("diffusers", diffusers.__version__, "transformers", transformers.__version__)
PYEOF
  log "环境就绪。GPU 运行任何脚本前请先: export CUDA_MODULE_LOADING=LAZY"
}
# ───────────────────────── 权重 ─────────────────────────
stage_weights() {
  [ -x "$PY" ] || die "未找到 $PY，请先运行 env 阶段"
  mkdir -p weights

  if [ -d weights/FLUX.2-klein-base-4B/transformer ]; then
    log "跳过 student（weights/FLUX.2-klein-base-4B 已存在）"
  else
    log "下载 student 4B 管线 (≈16GB)"
    PY="$PY" bash scripts/download_weights.sh student
  fi

  if [ -d "$HF_HOME/hub/models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K" ] \
     && [ -f "$TORCH_HOME/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth" ]; then
    log "跳过 eval 评测模型（已存在）"
  else
    log "下载 eval 评测模型 CLIP/HPSv2.1/OpenCLIP-H/FID (≈7.4GB)"
    PY="$PY" bash scripts/download_weights.sh eval
  fi

  if [ -f weights/benchmark_eval/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth ]; then
    log "跳过 GenEval/T2I-CompBench 检测器（已存在）"
  else
    log "下载 GenEval/T2I-CompBench 评测器 (HF 镜像 a12s12/geneval_detection, ≈4GB)"
    "$PY" -c "from huggingface_hub import snapshot_download as s; s('a12s12/geneval_detection', local_dir='weights/benchmark_eval')"
  fi

  if [ -f weights/aesthetic/sac+logos+ava1-l14-linearMSE.pth ]; then
    log "跳过 LAION 美学预测器 v2（已存在）"
  else
    log "下载 LAION 美学预测器 v2（Pexels 训练集打分用）"
    mkdir -p weights/aesthetic
    curl -fSL --retry 5 -o weights/aesthetic/sac+logos+ava1-l14-linearMSE.pth \
      "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
  fi

  if [ -d weights/FLUX.2-klein-base-9B/transformer ]; then
    log "跳过 teacher（weights/FLUX.2-klein-base-9B 已存在）"
  elif [ -n "${HF_TOKEN:-}" ]; then
    log "下载 teacher 9B 管线 (gated, ≈35GB)"
    PY="$PY" bash scripts/download_weights.sh teacher
  else
    log "未设置 HF_TOKEN，跳过 9B 教师。授权后: export HF_TOKEN=hf_xxx 再重跑 weights 阶段"
  fi
  df -h "$HERE" | tail -1
}
# ───────────────────────── 数据 ─────────────────────────
stage_data() {
  [ -x "$PY" ] || die "未找到 $PY，请先运行 env 阶段"
  mkdir -p data/raw

  if [ -f data/coco30k.txt ] && [ "$(ls data/coco30k_images 2>/dev/null | wc -l)" -ge 30000 ]; then
    log "跳过 COCO-30k（已存在）"
  else
    log "下载并导出 COCO-30k（表2 的 FID/CLIP 参考集）"
    "$PY" -c "from huggingface_hub import snapshot_download as s; s('sayakpaul/coco-30-val-2014', repo_type='dataset', local_dir='data/raw/coco-30-val-2014', allow_patterns=['data/*.parquet','README.md'])"
    "$PY" - <<'PYEOF'
import pyarrow.parquet as pq, glob, os, json
os.makedirs('data/coco30k_images', exist_ok=True)
caps, meta, i = [], [], 0
for f in sorted(glob.glob('data/raw/coco-30-val-2014/data/*.parquet')):
    pf = pq.ParquetFile(f)
    for rg in range(pf.num_row_groups):
        for d in pf.read_row_group(rg).to_pylist():
            name = f'{i:05d}.jpg'
            open(f'data/coco30k_images/{name}', 'wb').write(d['image']['bytes'])
            cap = ' '.join(d['caption'].split())
            caps.append(cap); meta.append({'idx': i, 'file': name, 'coco_file': d['image']['path'], 'caption': cap}); i += 1
open('data/coco30k.txt', 'w').write('\n'.join(caps) + '\n')
with open('data/coco30k_meta.jsonl', 'w') as w:
    for m in meta: w.write(json.dumps(m, ensure_ascii=False) + '\n')
print('COCO-30k:', i, 'pairs')
PYEOF
  fi

  if [ -f data/val_prompts.txt ] && [ "$(ls data/val_reference_images 2>/dev/null | wc -l)" -ge 1000 ]; then
    log "跳过 APRA 验证集（已存在）"
  else
    log "构建 APRA 验证集 1000 对（取自 COCO Karpathy test，与 COCO-30k 不重叠）"
    "$PY" -c "from huggingface_hub import snapshot_download as s; s('nlphuji/mscoco_2014_5k_test_image_text_retrieval', repo_type='dataset', local_dir='data/raw/coco_karpathy_test5k')"
    "$PY" - <<'PYEOF'
import csv, json, zipfile, os, ast, random
seen = {json.loads(l)['coco_file'] for l in open('data/coco30k_meta.jsonl')} if os.path.exists('data/coco30k_meta.jsonl') else set()
rows = list(csv.DictReader(open('data/raw/coco_karpathy_test5k/test_5k_mscoco_2014.csv')))
cand = [r for r in rows if r['filename'] not in seen]
random.Random(42).shuffle(cand)
sel = cand[:1000]
z = zipfile.ZipFile('data/raw/coco_karpathy_test5k/images_mscoco_2014_5k_test.zip')
names = {os.path.basename(n): n for n in z.namelist() if n.endswith('.jpg')}
os.makedirs('data/val_reference_images', exist_ok=True)
prompts, meta = [], []
for i, r in enumerate(sel):
    cap = ' '.join(ast.literal_eval(r['raw'])[0].split())
    open(f'data/val_reference_images/{i:04d}.jpg', 'wb').write(z.read(names[r['filename']]))
    prompts.append(cap); meta.append({'idx': i, 'file': f'{i:04d}.jpg', 'coco_file': r['filename'], 'caption': cap})
open('data/val_prompts.txt', 'w').write('\n'.join(prompts) + '\n')
with open('data/val_meta.jsonl', 'w') as w:
    for m in meta: w.write(json.dumps(m, ensure_ascii=False) + '\n')
print('val set:', len(prompts), 'pairs')
PYEOF
  fi
  stage_data_bench
  [ "${ECFLUX_FETCH_TRAINSET:-0}" = "1" ] && stage_data_trainset || log "跳过训练集替代（设 ECFLUX_FETCH_TRAINSET=1 启用，约 306GB 下载）"
  df -h "$HERE" | tail -1
}
# benchmark prompt / 评测仓库
stage_data_bench() {
  mkdir -p data/benchmarks
  if [ -d data/benchmarks/geneval ]; then log "跳过 GenEval 仓库（已存在）"; else
    log "clone GenEval"; git clone --depth 1 https://github.com/djghosh13/geneval.git data/benchmarks/geneval; fi
  if [ -d data/benchmarks/T2I-CompBench ]; then log "跳过 T2I-CompBench 仓库（已存在）"; else
    log "clone T2I-CompBench"; git clone --depth 1 https://github.com/Karine-Huang/T2I-CompBench.git data/benchmarks/T2I-CompBench; fi

  if [ -f data/benchmarks/hpsv2/photo.txt ]; then log "跳过 HPSv2 benchmark prompt（已存在）"; else
    log "下载 HPSv2 benchmark prompt（四类各 800 条）"
    "$PY" -c "from huggingface_hub import snapshot_download as s; s('zhwang/HPDv2', repo_type='dataset', local_dir='data/benchmarks/hpsv2', allow_patterns=['benchmark/*.json','README.md'])"
    "$PY" - <<'PYEOF'
import json
for s in ['anime', 'concept-art', 'paintings', 'photo']:
    p = json.load(open(f'data/benchmarks/hpsv2/benchmark/{s}.json'))
    open(f'data/benchmarks/hpsv2/{s}.txt', 'w').write('\n'.join(' '.join(x.split()) for x in p) + '\n')
print('HPSv2 prompts ready')
PYEOF
  fi

  if [ -f data/router_prompts.txt ] && [ -f data/test_prompts.txt ]; then
    log "跳过 router/test prompt（已存在）"
  else
    log "从 T2I-CompBench 生成 router/test prompt 列表（train/val 不重叠）"
    "$PY" - <<'PYEOF'
D = 'data/benchmarks/T2I-CompBench/examples/dataset'
cats = ['color', 'shape', 'texture', 'spatial', 'non_spatial', 'complex', 'numeracy', '3d_spatial']
def load(split):
    seen, out = set(), []
    for c in cats:
        for l in open(f'{D}/{c}_{split}.txt', encoding='utf-8'):
            p = ' '.join(l.split())
            if p and p.lower() not in seen:
                seen.add(p.lower()); out.append(p)
    return out
tr, va = load('train'), load('val')
vs = {p.lower() for p in va}
tr = [p for p in tr if p.lower() not in vs]
open('data/router_prompts.txt', 'w').write('\n'.join(tr) + '\n')
open('data/test_prompts.txt', 'w').write('\n'.join(va) + '\n')
print('router_prompts', len(tr), 'test_prompts', len(va))
PYEOF
  fi
}
# 训练集替代：i1 Pexels 512px (1M) 下载 → 美学打分 → 取高分 15 万
stage_data_trainset() {
  [ -f weights/aesthetic/sac+logos+ava1-l14-linearMSE.pth ] || die "缺美学预测器权重，请先运行 weights 阶段"
  if [ -f data/pexels_aes/index.jsonl ]; then log "跳过训练集（data/pexels_aes 已存在）"; return; fi
  log "下载 i1 Pexels 512px（约 306GB，断点续传，失败自动重试）"
  for i in $(seq 1 40); do
    n=$(ls data/raw/i1_pexels_512 2>/dev/null | grep -c tfrecord || true)
    [ "$n" -ge 128 ] && break
    timeout 600 "$PY" -c "from huggingface_hub import snapshot_download as s; s('i1-datasets/i1-pexels-512-resolution-1m-tfrecord', repo_type='dataset', local_dir='data/raw/i1_pexels_512', max_workers=11)" || true
  done
  log "美学打分（需 GPU；CUDA_MODULE_LOADING=LAZY）"
  CUDA_MODULE_LOADING=LAZY HF_HUB_OFFLINE=1 "$PY" data/prepare_pexels_aes.py score --src data/raw/i1_pexels_512 --out data/raw/i1_pexels_512_scores
  log "导出高分 15 万 → data/pexels_aes"
  "$PY" data/prepare_pexels_aes.py export --src data/raw/i1_pexels_512 --scores data/raw/i1_pexels_512_scores --out data/pexels_aes --num 150000
  log "训练集就绪。原始 tfrecord data/raw/i1_pexels_512 可删除以省空间"
}
# ───────────────────────── 主流程 ─────────────────────────
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(all)
run=""
for s in "${STAGES[@]}"; do
  case "$s" in
    all) run="env weights data"; break ;;
    env|weights|data) run="$run $s" ;;
    *) die "未知参数: $s（可用 env / weights / data / all）" ;;
  esac
done

setup_net
for s in $run; do
  case "$s" in
    env)     stage_env ;;
    weights) stage_weights ;;
    data)    stage_data ;;
  esac
done
log "全部完成：$run"





