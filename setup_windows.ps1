# setup_windows.ps1 -- Environment setup for CSED 504 (Computer Vision + NLP)
# Windows -- NVIDIA CUDA 12.8, Blackwell-compatible PyTorch wheels
#
# Run this from the Anaconda Prompt (or Anaconda PowerShell Prompt):
#   cd O:\Sources\GitHub\TrueRottweiler\WashingtonCsed504
#   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#
# All packages are installed into the uw-csed504 conda env via
# "conda run -n uw-csed504 pip install", so the activation state of the calling
# shell does not matter and no global Python is touched.
#
# ---------------------------------------------------------------------------
# Why everything is installed with pip (this is the OMP Error #15 fix)
# ---------------------------------------------------------------------------
# OMP Error #15 -- "Initializing libiomp5md.dll, but found libiomp5md.dll already
# initialized" -- is NOT a threading problem and is NOT fixed by lowering
# OMP_NUM_THREADS / MKL_NUM_THREADS.  It happens when TWO copies of the Intel
# OpenMP runtime (libiomp5md.dll) are loaded into one process: one from conda's
# MKL-backed numpy/scipy, and one from PyTorch's own wheel.
#
# The fix is to make PyTorch the ONLY provider of libiomp5md.dll.  We do that by
# installing the entire stack with pip (whose numpy/scikit-learn ship OpenBLAS,
# not MKL), and using conda solely to create the Python interpreter.  This mirrors
# the known-good uw-csed503 environment, which has exactly one libiomp5md.dll
# (torch's) and never raises OMP #15.
# ---------------------------------------------------------------------------

$ErrorActionPreference = 'Stop'
$ENV_NAME = 'uw-csed504'

Write-Host ''
Write-Host '=== CSED 504 environment setup ===' -ForegroundColor Cyan

# -- Resolve conda -------------------------------------------------------------------------------
# In the Anaconda Prompt, CONDA_EXE is already exported and inherited by this shell.
$condaExe = if ($env:CONDA_EXE) { $env:CONDA_EXE } else { (Get-Command conda -ErrorAction SilentlyContinue).Source }
if (-not $condaExe -or -not (Test-Path $condaExe)) {
    throw "conda not found. Run this script from the Anaconda Prompt so CONDA_EXE is set."
}
Write-Host "Using conda: $condaExe" -ForegroundColor DarkGray

# -- Step 1: faster solver -----------------------------------------------------------------------
& $condaExe config --set solver libmamba

# -- Step 2: (re)create environment --------------------------------------------------------------
Write-Host "Removing old $ENV_NAME environment (if any)..." -ForegroundColor Yellow
& $condaExe env remove -n $ENV_NAME -y 2>$null

Write-Host "Creating $ENV_NAME with Python 3.12..." -ForegroundColor Green
& $condaExe create -n $ENV_NAME python=3.12 -y

# -- Step 3: PyTorch cu128 FIRST (Blackwell sm_120 compatible) -----------------------------------
# Install torch before the rest so that torch-dependent packages (accelerate,
# torchmetrics) see it already satisfied and do NOT pull a CPU-only torch from
# PyPI.  conda's pytorch-cuda packages don't ship sm_120 kernels yet, so we use
# the official cu128 wheel index.
Write-Host 'Installing PyTorch with CUDA 12.8 (Blackwell-compatible)...' -ForegroundColor Green
& $condaExe run -n $ENV_NAME pip install `
    torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu128

# -- Step 4: scientific Python stack (pip -> OpenBLAS, no second libiomp5md.dll) ------------------
Write-Host 'Installing scientific Python stack...' -ForegroundColor Green
& $condaExe run -n $ENV_NAME pip install `
    numpy pandas matplotlib seaborn scikit-learn tqdm `
    jupyter ipykernel pillow imageio

# -- Step 5: CV and NLP extras -------------------------------------------------------------------
Write-Host 'Installing CV and NLP extras...' -ForegroundColor Green
& $condaExe run -n $ENV_NAME pip install `
    torchmetrics transformers datasets accelerate `
    sentencepiece protobuf 'huggingface_hub[hf_xet]'

# -- Step 6: register Jupyter kernel -------------------------------------------------------------
Write-Host 'Registering Jupyter kernel...' -ForegroundColor Green
& $condaExe run -n $ENV_NAME python -m ipykernel install `
    --user --name $ENV_NAME --display-name "Python ($ENV_NAME)"

# -- Step 7: pin GPU visibility (whatever GPUs are present) and verify ----------------------------
# cuda_check.ps1 writes CUDA_VISIBLE_DEVICES into the env's activate.d, exposing
# every usable GPU (all same-architecture cards on the dual-GPU workstation, or the
# single card on a laptop), then verifies with PyTorch.  It needs the env's python.
#
# We resolve the env prefix with "conda run" instead of "conda activate": activating
# a conda env inside a non-interactive script runs conda's PowerShell hook, whose
# embedded "conda activate base" line calls conda with empty _CE_M/_CE_CONDA args and
# fails with 'invalid choice: ""'.  "conda run" needs no activation and is reliable.
Write-Host 'Configuring GPU visibility...' -ForegroundColor Green
$envPrefix = (& $condaExe run -n $ENV_NAME python -c "import sys; print(sys.prefix)" |
              Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1).Trim()
& "$PSScriptRoot\cuda_check.ps1" -EnvName $ENV_NAME -CondaPrefix $envPrefix

# -- Done ----------------------------------------------------------------------------------------
# Report what was actually pinned, so the message is correct on any machine
# (dual-GPU workstation, single-GPU laptop, or CPU-only).  --print-cvd emits the
# comma-joined UUIDs cuda_check.ps1 pinned; the GPU count is one more than the
# number of commas.
$gpuCheckPy = Join-Path $PSScriptRoot 'src\common\gpu_check.py'
$gpuCount = 0
if ($envPrefix -and (Test-Path (Join-Path $envPrefix 'python.exe')) -and (Test-Path $gpuCheckPy)) {
    $cvd = (& (Join-Path $envPrefix 'python.exe') $gpuCheckPy --print-cvd 2>$null |
            Where-Object { $_ -and $_.Trim() } | Select-Object -Last 1)
    if ($cvd) { $gpuCount = ($cvd.Trim() -split ',').Count }
}

Write-Host ''
Write-Host '========================================' -ForegroundColor Cyan
Write-Host "Environment '$ENV_NAME' is ready." -ForegroundColor Green
Write-Host '========================================' -ForegroundColor Cyan
Write-Host "Activate : conda activate $ENV_NAME"
Write-Host 'Verify   : python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"'
Write-Host 'Re-check GPUs anytime : .\cuda_check.ps1'
Write-Host ''
if ($gpuCount -gt 1) {
    Write-Host "$gpuCount same-architecture GPUs are pinned via activate.d. Wrap models with"
    Write-Host 'gpu_check.get_data_parallel_model(model, DEVICE) to train across all of them.'
} elseif ($gpuCount -eq 1) {
    Write-Host '1 GPU is pinned via activate.d. get_device() will select it automatically;'
    Write-Host 'get_data_parallel_model() is a no-op on a single GPU (returns the model unchanged).'
} else {
    Write-Host 'No NVIDIA GPU was detected - training will fall back to CPU (or MPS on Apple Silicon).'
}
Write-Host '========================================' -ForegroundColor Cyan
