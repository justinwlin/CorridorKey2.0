#!/usr/bin/env bash
# Prepares a rented CUDA Linux machine to run CorridorKey. remote/run.py runs
# this over SSH after uploading the code to $1. Safe to re-run.
set -euo pipefail
cd "$1"

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "[setup] installing ffmpeg"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
fi

python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; \
print('[setup] torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

echo "[setup] installing Python requirements"
# Rented machines are single-purpose and ship PyTorch in the system Python,
# which Ubuntu 24.04 otherwise refuses to pip install into (PEP 668).
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install -q --root-user-action=ignore -r requirements.txt

# Machines sharing a network volume share the weights; only one downloads.
echo "[setup] checking model weights"
flock .weights.lock python scripts/download_weights.py
