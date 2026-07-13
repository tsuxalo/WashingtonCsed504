#!/usr/bin/env bash
# setup_mac.sh — Environment setup for CSED 504 (Computer Vision + NLP)
# macOS / Linux — CPU or MPS (Apple Silicon) PyTorch
#
# Packages installed:
#   • PyTorch + torchvision + torchaudio (CPU/MPS)   — GPU acceleration on Apple Silicon
#   • transformers, datasets, accelerate              — HuggingFace NLP stack
#   • Pillow, scikit-learn                            — CV and ML extras
#   • numpy, pandas, matplotlib, seaborn              — scientific Python
#   • jupyter, ipykernel, tqdm                        — notebook tooling
#
# Usage:
#   conda activate base
#   bash setup_mac.sh

set -euo pipefail

echo "Setting up uw-csed504 environment..."

# Make conda installs faster (do this first)
conda config --set solver libmamba

# Remove existing environment if it exists
echo "Removing existing environment (if any)..."
conda env remove -n uw-csed504 -y 2>/dev/null || true

# Create conda environment with Python 3.12 (matches Colab)
echo "Creating new environment with Python 3.12..."
conda create -n uw-csed504 python=3.12 -y

# Activate the environment
eval "$(conda shell.bash hook)"
conda activate uw-csed504

# Install PyTorch — CPU/MPS build.  No CUDA on macOS; MPS is picked up automatically by PyTorch
# at runtime when running on Apple Silicon.
echo "Installing PyTorch (CPU/MPS)..."
conda install -y pytorch torchvision torchaudio -c pytorch

# Install core scientific Python stack
echo "Installing scientific Python packages..."
conda install -y numpy pandas matplotlib seaborn scikit-learn jupyter ipykernel tqdm pillow

# Install HuggingFace NLP stack + CV extras via pip — conda versions lag behind and have
# dependency conflicts with some HuggingFace packages.  torchmetrics and imageio are kept
# in sync with setup_windows.ps1 so both platforms have the same packages for the assignments.
echo "Installing NLP / HuggingFace + CV packages..."
pip install torchmetrics imageio transformers datasets accelerate sentencepiece protobuf "huggingface_hub[hf_xet]"

# Register kernel for Jupyter / VS Code
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name uw-csed504 --display-name "Python (uw-csed504)"

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo "Activate:  conda activate uw-csed504"
echo "Verify:    python -c \"import torch; print(f'MPS: {torch.backends.mps.is_available()}')\""
echo "========================================"
