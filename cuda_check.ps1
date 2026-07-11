# cuda_check.ps1 -- Detect GPUs, pin CUDA_VISIBLE_DEVICES, and verify PyTorch.
#
# Can be run standalone at any time to (re)configure GPU visibility -- e.g. after
# adding or removing a GPU:
#   conda activate uw-csed504
#   .\cuda_check.ps1
#
# Also called by setup_windows.ps1 after PyTorch is installed.
#
# Parameters:
#   -EnvName      conda environment name to configure (default uw-csed504); used
#                 only for messages.
#   -CondaPrefix  full path to the env's prefix (the folder containing python.exe).
#                 When omitted, $env:CONDA_PREFIX is used -- which is what you get
#                 after "conda activate uw-csed504" in an interactive shell.  The
#                 setup script passes this explicitly because activating a conda
#                 env inside a non-interactive script is unreliable.
param(
    [string]$EnvName = "uw-csed504",
    [string]$CondaPrefix = ""
)

# ---------------------------------------------------------------------------
# Multi-GPU visibility.
#
# The GPU selection logic lives in src/common/gpu_check.py so notebooks and this
# script make identical choices.  Cards are ranked by (VRAM, compute capability) --
# more memory wins, newer architecture only breaks ties (a 32 GB Ampere beats an
# 8 GB Ada, because VRAM matters more than generation for training).  Rules:
#   * Single GPU                          -> use it
#   * Several IDENTICAL cards             -> use ALL   (matched DataParallel)
#     (same VRAM AND same architecture)
#   * Any mismatch (different VRAM, or    -> use ONLY the single most powerful card
#     same VRAM but different arch)
#
# The exact selection depends on the machine: the dual RTX PRO 6000 workstation
# (two matched sm_120 cards) selects BOTH and reports device_count() == 2; a single-
# GPU laptop (e.g. an RTX 2000 Ada, sm_89) selects that one card; and a mixed rig
# (e.g. a 96 GB RTX PRO 6000 + a 32 GB RTX 5000 Ada) selects the RTX PRO 6000 alone,
# because DataParallel across mismatched cards stalls on the weaker card and OOMs the
# smaller one.  gpu_check.py decides from the hardware actually present -- nothing
# here is hard-coded to a specific GPU count or model.
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
# Prefer an explicitly passed -CondaPrefix (used by setup_windows.ps1, which cannot
# rely on interactive "conda activate"); fall back to the active env's CONDA_PREFIX.
$prefix    = if ($CondaPrefix)      { $CondaPrefix }      elseif ($env:CONDA_PREFIX) { $env:CONDA_PREFIX } else { $null }
$pythonExe = if ($prefix)           { Join-Path $prefix 'python.exe' } else { $null }

# conda itself is only needed to locate the env's python when it wasn't passed in.
# If we already have a prefix (explicit -CondaPrefix or an active env), conda's
# absence from PATH is irrelevant, so don't cry wolf.
if (-not $condaExe -and -not $prefix) {
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
