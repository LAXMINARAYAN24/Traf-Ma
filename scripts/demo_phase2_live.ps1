param(
    [int]$Vehicles = 20,
    [int]$Seed = 42,
    [double]$DurationSeconds = 180
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Project virtual-environment Python not found: $python"
}

$runId = "phase2_live_demo_{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $PID
Push-Location $root
try {
    & $python -m trafma simulate `
        --config configs\phase1.yaml `
        --vehicles $Vehicles `
        --seed $Seed `
        --duration $DurationSeconds `
        --run-id $runId `
        --output-dir runs\phase2 `
        --online-federation `
        --learning-config configs\learning.yaml
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host "Live federation artifacts: runs\phase2\$runId"
}
finally {
    Pop-Location
}
