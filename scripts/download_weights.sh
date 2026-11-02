#!/usr/bin/env bash
# 下载权重。国内网络可设置 https_proxy 或 HF_ENDPOINT=https://hf-mirror.com
# 用法: bash scripts/download_weights.sh [student|teacher|eval|all]
#   student : FLUX.2 Klein base 4B 管线（DiT 7.8GB + Qwen3-4B 8.1GB + VAE 0.17GB）≈ 16GB
#   teacher : FLUX.2 Klein base 9B 管线（DiT 18.2GB + Qwen3-8B 16.4GB + VAE 0.17GB）≈ 35GB，gated
#   eval    : CLIP ViT-L/14 (1.7GB) + HPSv2.1 (1.7GB) + OpenCLIP ViT-H-14 (3.9GB) + FID Inception (0.1GB)
# 仓库根目录下的 BFL 原始格式单文件（flux-2-klein-*.safetensors）与 diffusers 子目录内容重复，不下载。
set -euo pipefail
cd "$(dirname "$0")/.."
# 评估模型的缓存放在数据盘；使用 evaluate.py / collab_infer.py 前也需要 export 这两个变量
export HF_HOME=${HF_HOME:-$PWD/weights/hf_home}
export TORCH_HOME=${TORCH_HOME:-$PWD/weights/torch_home}
PY=${PY:-python}
PIPE='["model_index.json","scheduler/*","tokenizer/*","text_encoder/*","transformer/*","vae/*"]'

student() { $PY -c "from huggingface_hub import snapshot_download as s; s('black-forest-labs/FLUX.2-klein-base-4B', local_dir='weights/FLUX.2-klein-base-4B', allow_patterns=$PIPE)"; }
teacher() {
  # 需先在 https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B 登录并同意许可，再 hf auth login（或 export HF_TOKEN=...）
  $PY -c "from huggingface_hub import snapshot_download as s; s('black-forest-labs/FLUX.2-klein-base-9B', local_dir='weights/FLUX.2-klein-base-9B', allow_patterns=$PIPE)"
}
evalw() {
  $PY - <<'PYEOF'
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("openai/clip-vit-large-patch14", allow_patterns=["*.json", "*.txt", "model.safetensors"])
hf_hub_download("xswu/HPSv2", "HPS_v2.1_compressed.pt")
hf_hub_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "open_clip_pytorch_model.bin")
PYEOF
  mkdir -p "$TORCH_HOME/hub/checkpoints"
  curl -L -o "$TORCH_HOME/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth" \
    https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth
}

case ${1:-student} in
  student) student ;;
  teacher) teacher ;;
  eval) evalw ;;
  all) student; evalw; teacher ;;
  *) echo "usage: $0 [student|teacher|eval|all]"; exit 1 ;;
esac
du -sh weights/* 2>/dev/null; df -h .
