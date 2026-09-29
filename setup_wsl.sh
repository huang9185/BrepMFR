#!/usr/bin/env bash
# Set up the BrepMFR training environment inside WSL2 Ubuntu (or any Linux x86-64 box with an NVIDIA GPU).
# Usage (from the repo folder, inside WSL):   bash setup_wsl.sh
# Needs: Miniconda/Anaconda installed in WSL, and `nvidia-smi` working inside WSL.
set -euo pipefail

ENV_NAME=${ENV_NAME:-brep_mfr}

# fairseq builds C++ extensions, so a compiler is required.
if ! command -v g++ >/dev/null 2>&1; then
  echo ">> Installing build tools (needs sudo)"
  sudo apt-get update && sudo apt-get install -y build-essential
fi

# Make `conda activate` work inside a script.
eval "$(conda shell.bash hook)"

if conda env list | grep -qE "^${ENV_NAME}\s"; then
  echo ">> Conda env '${ENV_NAME}' already exists; reusing it"
else
  conda create -n "${ENV_NAME}" python=3.9 -y
fi
conda activate "${ENV_NAME}"

# fairseq 0.12.x cannot be installed with pip >= 24.1 (omegaconf metadata check).
python -m pip install "pip==24.0" "setuptools<70" wheel cython

# PyTorch 1.13.1 with its own CUDA 11.7 runtime (no system CUDA toolkit needed).
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117

# DGL 1.0.0 built for CUDA 11.7.
pip install dgl==1.0.0+cu117 -f https://data.dgl.ai/wheels/cu117/repo.html

# Pins from environment.yml, plus packages the code imports but environment.yml omits.
pip install numpy==1.23.5 pytorch-lightning==1.7.1 "torchmetrics<0.12" torch-geometric==2.3.1 \
  prefetch_generator scipy tqdm tensorboard prettytable matplotlib

pip install fairseq==0.12.2
# fairseq may upgrade numpy; the code and DGL 1.0 need numpy 1.23.
pip install numpy==1.23.5

python tools/check_env.py
