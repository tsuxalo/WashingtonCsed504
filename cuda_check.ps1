# cuda_check.ps1 -- Detect GPUs, pin CUDA_VISIBLE_DEVICES, and verify PyTorch.
#
# Can be run standalone at any time to (re)configure GPU visibility -- e.g. after
# adding or removing a GPU:
#   conda activate uw-csed504
#   .\cuda_check.ps1
#
# Also called by setup_windows.ps1 after PyTorch is installed.
#
# Optional parameter: the conda environment name to configure (default uw-csed504).
param(
    [string]$EnvName = "uw-csed504"
)

# ---------------------------------------------------------------------------
# Multi-GPU visibility.
#
# The GPU selection logic lives in src/common/gpu_check.py so notebooks and this
# script make identical choices.  Selection rules:
#   * Single GPU                        -> use it
#   * Multiple GPUs, same architecture  -> use ALL   (DataParallel across them)
#   * Mixed architectures               -> use only the best-architecture group
#
# This workstation has two RTX PRO 6000 Blackwell Max-Q cards (both sm_120), so
# gpu_check.py selects BOTH and torch.cuda.device_count() reports 2.
#
# We enumerate hardware with nvidia-smi (via gpu_check.py), NOT PyTorch: torch's
# CUDA indices are already filtered by any existing CUDA_VISIBLE_DEVICES, so it
# cannot reliably see all physical devices at setup time.
#
# CUDA_VISIBLE_DEVICES is written into the conda env's activate.d/deactivate.d so
# it is set ONLY while uw-csed504 is active.  Other CUDA apps (Ollama, etc.) keep
# seeing all GPUs.  We never write to the machine-wide user environment.
# ---------------------------------------------------------------------------
Write-Host "Configuring CUDA device visibility..." -ForegroundColor Green

# Resolve conda and the env's python.  conda activate sets these even when the
# executables are not on PATH in the calling shell.
$_condaCmd = Get-Command conda -ErrorAction SilentlyContinue
$condaExe  = if ($env:CONDA_EXE)    { $env:CONDA_EXE }    elseif ($_condaCmd) { $_condaCmd.Source } else { $null }
# Only ever use Python from the conda prefix; never the Windows Store launcher alias.
$pythonExe = if ($env:CONDA_PREFIX) { Join-Path $env:CONDA_PREFIX 'python.exe' } else { $null }

if (-not $condaExe) {
    Write-Host "  WARNING: conda not found - run this from a conda-activated terminal." -ForegroundColor Yellow
    Write-Host "    e.g.  conda activate $EnvName ; then re-run .\cuda_check.ps1" -ForegroundColor Yellow
}
if (-not $pythonExe) {
    Write-Host "  WARNING: CONDA_PREFIX not set - GPU verification will be skipped." -ForegroundColor Yellow
    Write-Host "    Activate the environment first:  conda activate $EnvName" -ForegroundColor Yellow
}

# gpu_check.py lives in src/common relative to this script.
$gpuCheckPy = Join-Path $PSScriptRoot "src\common\gpu_check.py"

if (-not (Test-Path $gpuCheckPy)) {
    Write-Host "  WARNING: gpu_check.py not found at $gpuCheckPy - skipping GPU configuration." -ForegroundColor Yellow
} elseif (-not $pythonExe -or -not (Test-Path $pythonExe)) {
    Write-Host "  WARNING: Python not available - skipping GPU configuration (activate the conda env first)." -ForegroundColor Yellow
} else {
    # --print-cvd emits ONLY the optimal CUDA_VISIBLE_DEVICES value (comma-joined
    # GPU UUIDs) with no other output, so it is safe to capture.  UUIDs are
    # hardware-stable: they survive driver updates, reboots, and slot reorders.
    $cvd = (& $pythonExe $gpuCheckPy --print-cvd 2>$null | Select-Object -Last 1)
    if ($cvd) { $cvd = $cvd.Trim() }

    if ($cvd) {
        $envRoot     = Join-Path (Split-Path $pythonExe) "etc\conda"
        $activateD   = Join-Path $envRoot "activate.d"
        $deactivateD = Join-Path $envRoot "deactivate.d"
        New-Item -ItemType Directory -Path $activateD, $deactivateD -Force | Out-Null

        # PowerShell
        Set-Content (Join-Path $activateD   "cuda_visible_devices.ps1") "`$Env:CUDA_VISIBLE_DEVICES = '$cvd'"
        Set-Content (Join-Path $deactivateD "cuda_visible_devices.ps1") 'Remove-Item Env:\CUDA_VISIBLE_DEVICES -ErrorAction SilentlyContinue'
        # cmd.exe
        Set-Content (Join-Path $activateD   "cuda_visible_devices.bat") "@SET CUDA_VISIBLE_DEVICES=$cvd"
        Set-Content (Join-Path $deactivateD "cuda_visible_devices.bat") "@SET CUDA_VISIBLE_DEVICES="
        # bash/sh
        Set-Content (Join-Path $activateD   "cuda_visible_devices.sh") "export CUDA_VISIBLE_DEVICES=$cvd"
        Set-Content (Join-Path $deactivateD "cuda_visible_devices.sh") "unset CUDA_VISIBLE_DEVICES"

        $count = ($cvd -split ',').Count
        Write-Host "  Pinned $count GPU(s) in conda env '$EnvName' (activate.d)." -ForegroundColor Green
        Write-Host "  CUDA_VISIBLE_DEVICES=$cvd" -ForegroundColor Cyan
        Write-Host "  Re-activate the env (conda deactivate; conda activate $EnvName) for it to take effect." -ForegroundColor Cyan
    } else {
        Write-Host "  No NVIDIA GPUs found or no restriction needed - CUDA_VISIBLE_DEVICES not set." -ForegroundColor Yellow
    }
}

# ---------------------------------------------------------------------------
# Verification summary (delegated to gpu_check.py, which respects the CVD we set).
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Verifying GPU support..." -ForegroundColor Yellow
if (-not $pythonExe -or -not (Test-Path $pythonExe)) {
    Write-Host "  Skipping Python verification - activate the conda environment first." -ForegroundColor Yellow
} elseif (-not (Test-Path $gpuCheckPy)) {
    Write-Host "  Skipping Python verification - gpu_check.py not found." -ForegroundColor Yellow
} else {
    & $pythonExe $gpuCheckPy --smoke-test
}
