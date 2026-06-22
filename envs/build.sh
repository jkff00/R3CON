#!/usr/bin/env bash
set -euo pipefail

ROOT="${PWD}"
CONDA_PATH="$(conda info --base)"

source "${CONDA_PATH}/etc/profile.d/conda.sh"

# create env
if ! conda env list | awk '{print $1}' | grep -qx "R3CON"; then
    conda create -y -n R3CON -c conda-forge python=3.9 "cmake<4" ninja
fi

conda activate R3CON

export PYTHONNOUSERSITE=True

# make sure pip belongs to this env
which python
which pip
python -m pip --version

# install habitat simulator 0.2.4
mkdir -p "${ROOT}/simulator"
cd "${ROOT}/simulator"

if [ ! -d "habitat-sim" ]; then
    git clone git@github.com:liren-jin/habitat-sim.git
else
    echo "habitat-sim already exists, skip git clone"
fi

cd habitat-sim

if [ -d "build" ]; then
    echo "Remove existing habitat-sim build directory"
    rm -rf build
fi
python -m pip install -r requirements.txt
python setup.py install --headless --bullet

# install PyTorch cu118
python -m pip install --no-cache-dir --force-reinstall \
    torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# lock torch stack, prevent requirements from upgrading it
cat > "${ROOT}/envs/torch-cu118-constraints.txt" <<'EOF'
torch==2.1.2+cu118
torchvision==0.16.2+cu118
torchaudio==2.1.2+cu118
triton==2.1.0
EOF

# install R3CON package support without changing torch
cd "${ROOT}"

python -m pip install -r "${ROOT}/envs/requirements.txt" \
    -c "${ROOT}/envs/torch-cu118-constraints.txt" \
    --extra-index-url https://download.pytorch.org/whl/cu118

# check torch version immediately
python - <<'PY'
import torch, torchvision, torchaudio
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("torch cuda:", torch.version.cuda)
assert torch.__version__.startswith("2.1.2"), torch.__version__
assert torchvision.__version__.startswith("0.16.2"), torchvision.__version__
assert torchaudio.__version__.startswith("2.1.2"), torchaudio.__version__
assert torch.version.cuda == "11.8", torch.version.cuda
PY

# compile local gaussian rasterizer
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

pip install ${ROOT}/envs/360-dn-diff-gaussian-rasterization

python -m pip check
