#!/usr/bin/env bash
# Prepares a CUDA Linux machine to run CorridorKey. remote/run.py runs this
# over SSH after uploading the code to $1. $2 is the Python that has CUDA
# PyTorch. Safe to re-run.
set -euo pipefail
cd "$1"
PY="${2:-python}"

if ! command -v ffmpeg >/dev/null 2>&1; then
    if [ "$(id -u)" != 0 ]; then
        echo "[setup] ffmpeg is missing. Install it on this machine and retry." >&2
        exit 1
    fi
    echo "[setup] installing ffmpeg"
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
fi

"$PY" -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; \
print('[setup] torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

echo "[setup] installing Python requirements"
# Rented machines are single-purpose and ship PyTorch in the system Python,
# which Ubuntu 24.04 otherwise refuses to pip install into (PEP 668). On your
# own machines, point --ssh_python at a virtual environment instead.
export PIP_BREAK_SYSTEM_PACKAGES=1
# Keep the machine's PyTorch. Unconstrained, pip has replaced the image's
# torch 2.9.1 with a newer release that CorridorKey is not tested with.
"$PY" - > .torch-constraints.txt <<'PYEOF'
import importlib.metadata as md
for name in ("torch", "torchvision"):
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        pass
PYEOF
echo "[setup] keeping $(tr '\n' ' ' < .torch-constraints.txt)"
"$PY" -m pip install -q --root-user-action=ignore -c .torch-constraints.txt -r requirements.txt
"$PY" -c "import torch; print('[setup] torch after install:', torch.__version__)"

# Machines sharing a network volume share the weights; only one downloads.
echo "[setup] checking model weights"
flock .weights.lock "$PY" scripts/download_weights.py
