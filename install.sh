#!/usr/bin/env bash
# install.sh - one-shot setup for this repo
#
# Fresh setup:
#   bash install.sh
#
# Fast re-run (no pip reinstall):
#   SKIP_DEPS=1 bash install.sh
#
# Options via env vars:
#   WITH_ET1=1         # also set up ET model 1 assets (default: 0)
#   FORCE_DATA=1       # rebuild English dataset even if already present
#   ET2_CHECKPOINT=... # ET2 checkpoint base path (default: ./checkpoints/et_predictor2_seed123)
#   DATA_DIR=...       # output folder for fold csv files (default: ./data)
#   DATA_SEED=...      # split seed for fold1/fold2 (default: 42)
#   DATA_ZIP_URL=...   # Google Drive zip URL for English TSV bundle
#   DATA_ZIP_NAME=...  # local filename for downloaded zip (default: english_va_bundle.zip)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] python/python3 not found"
  exit 1
fi

SKIP_DEPS="${SKIP_DEPS:-0}"
WITH_ET1="${WITH_ET1:-0}"
FORCE_DATA="${FORCE_DATA:-0}"
ET2_CHECKPOINT="${ET2_CHECKPOINT:-./checkpoints/et_predictor2_seed123}"
DATA_DIR="${DATA_DIR:-./data}"
DATA_SEED="${DATA_SEED:-42}"
DATA_ZIP_URL="${DATA_ZIP_URL:-https://drive.google.com/file/d/1xXM32nva_4I3EAVAOrQ84L16f-LjsJbj/view?usp=sharing}"
DATA_ZIP_NAME="${DATA_ZIP_NAME:-english_va_bundle.zip}"

echo "============================================================"
echo " VA+Gaze one-shot install"
echo " Repo: $REPO_ROOT"
echo " Python: $($PYTHON_BIN --version 2>/dev/null || echo "$PYTHON_BIN")"
echo "============================================================"

echo
echo "[1/3] Python dependencies"
if [[ "$SKIP_DEPS" == "1" ]]; then
  echo "  - skip (SKIP_DEPS=1)"
else
  "$PYTHON_BIN" -m pip install -U pip setuptools wheel
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

echo
echo "[2/3] ET setup (ET2 auto-download if missing)"
ET_ARGS=(--skip-install --et2-checkpoint "$ET2_CHECKPOINT")
if [[ "$WITH_ET1" != "1" ]]; then
  ET_ARGS+=(--skip-et1)
fi
"$PYTHON_BIN" setup_et_models.py "${ET_ARGS[@]}"

echo
echo "[3/3] English dataset build"
DATA_ARGS=(--output-dir "$DATA_DIR" --seed "$DATA_SEED")
DATA_ARGS+=(--gdrive-zip-url "$DATA_ZIP_URL" --gdrive-zip-name "$DATA_ZIP_NAME")
if [[ "$FORCE_DATA" == "1" ]]; then
  DATA_ARGS+=(--force)
fi
"$PYTHON_BIN" prepare_english_data.py "${DATA_ARGS[@]}"

echo
echo "Done."
echo
echo "Train example:"
echo "python train_model.py xlmroberta-large mse --use-gaze-concat --et2-checkpoint $ET2_CHECKPOINT --features-used 1,1,1,1,1 --fp-dropout 0.1,0.3 --batch-size 8 --maxlen 200 --optim adamw_torch"
