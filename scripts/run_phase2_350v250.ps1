param(
    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
$datasetManifest = Join-Path $root "data\phase2_350clients_1000windows_feasibility\dataset_manifest.json"
if (-not (Test-Path $python)) {
    throw "Project virtual-environment Python was not found: $python"
}
if (-not (Test-Path $datasetManifest)) {
    throw "The 350v250 dataset is missing. Collect it once with configs/data_350clients_1000windows_feasibility.yaml before training."
}

$comparisonRoot = "runs/phase2_350v250_20261105_223954"
$runRoot = Join-Path $comparisonRoot "repeats/$(Get-Date -Format 'yyyyMMdd_HHmmss')"
if (Test-Path $runRoot) {
    throw "This repeat folder already exists; start again in a new second: $runRoot"
}
$configDir = Join-Path $runRoot "configs"
New-Item -ItemType Directory -Path $configDir -Force | Out-Null
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

function Save-RunConfig {
    param(
        [string]$Source,
        [string]$Destination,
        [string]$OutputDir,
        [string]$RsuUpdate = ""
    )

    $content = Get-Content -LiteralPath $Source -Raw
    $content = $content -replace '(?m)^output_dir:\s*.*$', "output_dir: $OutputDir"
    if ($RsuUpdate -ne "") {
        $content = $content -replace '(?m)^  rsu_self_update:\s*.*$', "  rsu_self_update: $RsuUpdate"
    }
    [System.IO.File]::WriteAllText($Destination, $content, $utf8NoBom)
}

function Get-OnlyRunPath {
    param([string]$OutputRoot)
    $items = @(Get-ChildItem -LiteralPath $OutputRoot -Directory)
    if ($items.Count -ne 1) {
        throw "Expected one experiment folder in $OutputRoot; found $($items.Count)."
    }
    return $items[0].FullName
}

Save-RunConfig "configs/learning_350v250_central.yaml" `
    "$configDir/central.yaml" "$runRoot/central"
Save-RunConfig "configs/learning_350v250_flat.yaml" `
    "$configDir/flat.yaml" "$runRoot/flat"
Save-RunConfig "configs/learning_350v250_hierarchical.yaml" `
    "$configDir/hierarchical_rsu_off.yaml" "$runRoot/hierarchical_rsu_off" "false"
Save-RunConfig "configs/learning_350v250_hierarchical.yaml" `
    "$configDir/hierarchical_rsu_on.yaml" "$runRoot/hierarchical_rsu_on" "true"

Write-Host "Repeat outputs and config snapshots inside the existing 350v250 run: $runRoot"

& $python -m trafma train --method central --config "$configDir/central.yaml" --device $Device
if ($LASTEXITCODE -ne 0) { throw "Central GRU failed." }
$centralRun = Get-OnlyRunPath "$runRoot/central"

& $python -m trafma fedavg --config "$configDir/flat.yaml" --threads 0
if ($LASTEXITCODE -ne 0) { throw "Flat FedAvg failed." }
$flatRun = Get-OnlyRunPath "$runRoot/flat"

& $python -m trafma fedavg --hierarchical --config "$configDir/hierarchical_rsu_off.yaml" --rsu-self-update false --threads 0
if ($LASTEXITCODE -ne 0) { throw "Hierarchical FedAvg with RSU self-update off failed." }
$hierOffRun = Get-OnlyRunPath "$runRoot/hierarchical_rsu_off"

& $python -m trafma fedavg --hierarchical --config "$configDir/hierarchical_rsu_on.yaml" --rsu-self-update true --threads 0
if ($LASTEXITCODE -ne 0) { throw "Hierarchical FedAvg with RSU self-update on failed." }
$hierOnRun = Get-OnlyRunPath "$runRoot/hierarchical_rsu_on"

$report = "$runRoot/report"
& $python -m trafma run-report `
    --experiment $centralRun `
    --experiment $flatRun `
    --experiment $hierOffRun `
    --experiment $hierOnRun `
    --output $report
if ($LASTEXITCODE -ne 0) { throw "Run report generation failed." }

Write-Host "Report saved to: $report"
