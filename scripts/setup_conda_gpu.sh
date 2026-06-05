#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-gaze_affect_clip}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
PYTORCH_CUDA="${PYTORCH_CUDA:-cu121}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda not found. Load conda first, then rerun this script."
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[conda] env already exists: $ENV_NAME"
else
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

python -m pip install -U pip wheel "setuptools<82"
python -m pip install \
  --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}" \
  --extra-index-url https://pypi.org/simple \
  torch torchvision torchaudio
python -m pip install -r requirements-gpu.txt

python - <<'PY'
import torch
import transformers
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY
