# fix_conda_powershell.ps1 -- Repair "conda activate" on Windows PowerShell.
#
# The bug
# -------
# conda 24.x emits  $Env:_CE_M = ""  and  $Env:_CE_CONDA = ""  from
# `conda shell.powershell activate <env>`, and its Conda.psm1 then calls
#   & $Env:CONDA_EXE $Env:_CE_M $Env:_CE_CONDA shell.powershell activate <env>
# passing those EMPTY STRINGS to conda.exe as empty positional arguments.  conda
# reads the first empty arg as the subcommand and dies with:
#   conda-script.py: error: argument COMMAND: invalid choice: '' (choose from ...)
#   Invoke-Expression: Cannot bind argument to parameter 'Command' because it is an
#   empty string.
# The first activation of a session (into base) works because the vars aren't set
# yet; every `conda activate <other-env>` afterward fails.
#
# The fix
# -------
# Patch Conda.psm1 to drop _CE_M / _CE_CONDA whenever they are empty, right before
# each call that would pass them to conda.exe.  PowerShell then omits absent vars
# from the argument list and activation works for every env, every switch.
#
# This script is idempotent, keeps a one-time pristine backup, and verifies the
# patched module parses before installing it.  Re-run it whenever a `conda update`
# overwrites Conda.psm1 and the bug comes back.
#
# Usage (from an Anaconda PowerShell prompt, in this repo):
#   powershell -ExecutionPolicy Bypass -File .\fix_conda_powershell.ps1
#   powershell -ExecutionPolicy Bypass -File .\fix_conda_powershell.ps1 -Restore
#   powershell -ExecutionPolicy Bypass -File .\fix_conda_powershell.ps1 -Psm1Path 'C:\path\to\Conda.psm1'

param(
    [string]$Psm1Path = "",   # override Conda.psm1 location if auto-detect fails
    [switch]$Restore          # restore the pristine backup instead of patching
)

$ErrorActionPreference = 'Stop'
$MARKER = 'CSED504-conda-fix'   # tag on inserted lines -> makes patching idempotent

# -- Locate Conda.psm1 ---------------------------------------------------------------------------
function Resolve-Psm1 {
    param([string]$Override)
    if ($Override) {
        if (-not (Test-Path $Override)) { throw "No Conda.psm1 at -Psm1Path '$Override'." }
        return (Resolve-Path $Override).Path
    }
    # conda.exe usually lives at <base>\Scripts\conda.exe (or <base>\condabin\conda.bat);
    # the module is at <base>\shell\condabin\Conda.psm1 either way (base = two levels up).
    $condaExe = if ($env:CONDA_EXE) { $env:CONDA_EXE }
                else { (Get-Command conda -ErrorAction SilentlyContinue).Source }
    if ($condaExe) {
        $base = Split-Path (Split-Path $condaExe)
        $cand = Join-Path $base 'shell\condabin\Conda.psm1'
        if (Test-Path $cand) { return $cand }
    }
    foreach ($root in @("$env:USERPROFILE\anaconda3", "$env:USERPROFILE\miniconda3",
                        "$env:LOCALAPPDATA\anaconda3", "$env:LOCALAPPDATA\miniconda3")) {
        $cand = Join-Path $root 'shell\condabin\Conda.psm1'
        if (Test-Path $cand) { return $cand }
    }
    throw "Could not locate Conda.psm1. Re-run with -Psm1Path 'C:\path\to\Conda.psm1'."
}

$psm1 = Resolve-Psm1 -Override $Psm1Path
$bak  = "$psm1.orig-backup"
Write-Host "Conda.psm1 : $psm1" -ForegroundColor DarkGray

# -- Restore mode --------------------------------------------------------------------------------
if ($Restore) {
    if (-not (Test-Path $bak)) { throw "No pristine backup at '$bak' to restore from." }
    Copy-Item $bak $psm1 -Force
    Write-Host "Restored pristine Conda.psm1 from backup. Open a new terminal to reload it." -ForegroundColor Green
    return
}

# -- Already patched? ----------------------------------------------------------------------------
if ((Get-Content -LiteralPath $psm1 -Raw) -match [regex]::Escape($MARKER)) {
    Write-Host "Already patched (marker present) - nothing to do." -ForegroundColor Yellow
    Write-Host "  (Use -Restore to revert, then re-run to re-apply from a clean copy.)" -ForegroundColor DarkGray
    return
}

# -- One-time pristine backup --------------------------------------------------------------------
if (-not (Test-Path $bak)) {
    Copy-Item $psm1 $bak
    Write-Host "Backed up pristine module -> $bak" -ForegroundColor DarkGray
}

# -- Patch: insert an empty-var guard before every conda.exe call that passes the CE vars --------
$callPattern = '&\s*\$Env:CONDA_EXE\s+\$Env:_CE_M\s+\$Env:_CE_CONDA'
$lines   = Get-Content -LiteralPath $psm1
$out     = New-Object System.Collections.Generic.List[string]
$patched = 0
foreach ($line in $lines) {
    if ($line -match $callPattern) {
        $indent = ([regex]::Match($line, '^\s*')).Value
        $out.Add($indent + "if ((Test-Path Env:\_CE_M)     -and '' -eq `$Env:_CE_M)     { Remove-Item Env:\_CE_M }     # $MARKER")
        $out.Add($indent + "if ((Test-Path Env:\_CE_CONDA) -and '' -eq `$Env:_CE_CONDA) { Remove-Item Env:\_CE_CONDA } # $MARKER")
        $patched++
    }
    $out.Add($line)
}

if ($patched -eq 0) {
    Write-Host "No matching call sites found - your Conda.psm1 may already be fixed upstream." -ForegroundColor Yellow
    Write-Host "Nothing changed." -ForegroundColor Yellow
    return
}

# -- Verify the patched text parses, then install it atomically ----------------------------------
$tmp  = "$psm1.tmp"
$text = ($out -join "`r`n") + "`r`n"
[System.IO.File]::WriteAllText($tmp, $text, (New-Object System.Text.UTF8Encoding($false)))  # UTF-8, no BOM

$errs = $null
[System.Management.Automation.Language.Parser]::ParseFile($tmp, [ref]$null, [ref]$errs) | Out-Null
if ($errs) {
    Remove-Item $tmp -ErrorAction SilentlyContinue
    throw "Patched module failed to parse ($($errs.Count) error(s)); left Conda.psm1 untouched."
}
Move-Item $tmp $psm1 -Force

Write-Host ""
Write-Host "Patched $patched call site(s) in Conda.psm1." -ForegroundColor Green
Write-Host "Open a NEW terminal, then: conda activate uw-csed504" -ForegroundColor Green
Write-Host ""
Write-Host "For the CURRENT shell (no restart), clear the empties once:" -ForegroundColor DarkGray
Write-Host "  Remove-Item Env:\_CE_M, Env:\_CE_CONDA -ErrorAction SilentlyContinue" -ForegroundColor DarkGray
Write-Host "Revert anytime: .\fix_conda_powershell.ps1 -Restore" -ForegroundColor DarkGray
