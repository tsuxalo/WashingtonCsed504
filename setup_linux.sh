#!/usr/bin/env bash
# setup_linux.sh -- Environment setup for CSED 504 (Computer Vision + NLP)
# Linux and WSL2 -- NVIDIA CUDA 12.8, Blackwell-compatible PyTorch wheels.
# CPU-only Linux works too: the cu128 wheel imports fine and falls back to CPU.
#
# Usage:
#   conda activate base
#   bash setup_linux.sh
#
# Like the Windows script, EVERYTHING is installed with pip so PyTorch is the only
# provider of the Intel OpenMP runtime -- the all-pip approach that avoids the
# duplicate-libiomp OMP Error #15 and keeps all three platforms consistent.

set -euo pipefail
ENV_NAME=uw-csed504

echo "=== CSED 504 environment setup (Linux) ==="

# -- Step 1: faster solver -----------------------------------------------------
conda config --set solver libmamba

# -- Step 2: (re)create environment --------------------------------------------
echo "Removing old $ENV_NAME environment (if any)..."
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
echo "Creating $ENV_NAME with Python 3.12..."
conda create -n "$ENV_NAME" python=3.12 -y

# Activate inside this non-interactive script via conda's shell hook.
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# -- Step 3: PyTorch cu128 FIRST (so torch-dependent pkgs don't pull CPU torch) -
echo "Installing PyTorch with CUDA 12.8 (Blackwell-compatible)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# -- Step 4: scientific Python stack (pip -> OpenBLAS, no second libiomp) -------
echo "Installing scientific Python stack..."
pip install numpy pandas matplotlib seaborn scikit-learn tqdm \
    jupyter ipykernel pillow imageio

# -- Step 5: CV and NLP extras -------------------------------------------------
echo "Installing CV and NLP extras..."
pip install torchmetrics transformers datasets accelerate \
    sentencepiece protobuf "huggingface_hub[hf_xet]"

# -- Step 6: register Jupyter kernel -------------------------------------------
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# -- Step 7: pin all same-architecture GPUs, then verify -----------------------
echo "Configuring GPU visibility..."
GPU_CHECK="$(cd "$(dirname "$0")" && pwd)/src/common/gpu_check.py"
CVD="$(python "$GPU_CHECK" --print-cvd 2>/dev/null || true)"
if [ -n "${CVD}" ]; then
    ACT_D="$CONDA_PREFIX/etc/conda/activate.d"
    DEACT_D="$CONDA_PREFIX/etc/conda/deactivate.d"
    mkdir -p "$ACT_D" "$DEACT_D"
    printf 'export CUDA_VISIBLE_DEVICES=%s\n' "$CVD" > "$ACT_D/cuda_visible_devices.sh"
    printf 'unset CUDA_VISIBLE_DEVICES\n'             > "$DEACT_D/cuda_visible_devices.sh"
    echo "  Pinned CUDA_VISIBLE_DEVICES=$CVD in env '$ENV_NAME' (activate.d)."
    echo "  Re-activate the env for it to take effect:  conda deactivate; conda activate $ENV_NAME"
else
    echo "  No NVIDIA GPUs detected (or no restriction needed) -- skipping GPU pinning."
fi

echo "Verifying..."
python "$GPU_CHECK"

echo ""
echo "========================================"
echo "Environment '$ENV_NAME' is ready."
echo "Activate : conda activate $ENV_NAME"
echo 'Verify   : python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"'
echo "========================================"
