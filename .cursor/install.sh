#!/usr/bin/env bash
# Build the .venv that nbdev-test and evals/e2e.py run in. Reruns reuse whatever is already there.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12 --allow-existing .venv

# Every runtime is an optional extra, and CI needs all of them to run the notebook tests. coreml is
# macOS-only, so 04_apple stays #| eval: false here.
uv pip install -e '.[litert,onnx,hub,video,audio]' nbdev notebook ipykernel

# evals/e2e.py checks anya's preprocessing against timm's and HuggingFace's own transforms. CPU wheels:
# the default index ships the CUDA build, which is 2GB of no use here.
uv pip install transformers
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

.venv/bin/nbdev-install-hooks
