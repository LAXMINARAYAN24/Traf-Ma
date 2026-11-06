$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$demoName = "demo_$stamp"
$demoRoot = Join-Path "runs" $demoName
$datasetId = "demo_dataset_$stamp"
$datasetPath = Join-Path "data" $datasetId
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

if (-not (Test-Path $py)) { throw "Project venv Python not found." }
if (Test-Path $demoRoot) { throw "Demo output already exists: $demoRoot" }
if (Test-Path $datasetPath) { throw "Demo dataset already exists: $datasetPath" }

$configDir = Join-Path $demoRoot "configs"
$simOut = Join-Path $demoRoot "simulation"
New-Item -ItemType Directory -Path $configDir, $simOut -Force | Out-Null

# Separate 20-vehicle GUI simulation and its plots.
$simId = "demo_gui_$stamp"
& $py -m trafma simulate --config configs/phase1.yaml --vehicles 20 --seed 42 `
    --duration 180 --gui --run-id $simId --output-dir $simOut
if ($LASTEXITCODE -ne 0) { throw "GUI simulation failed." }

$simRun = Join-Path $simOut $simId
& $py -m trafma report --run $simRun
if ($LASTEXITCODE -ne 0) { throw "Simulation report failed." }

# Ten fresh runs: 7 train, 1 validation, 2 test.
$dataConfig = Join-Path $configDir "demo_data.yaml"
$dataYaml = @"
dataset_id: $datasetId
output_dir: data
jobs: 1
base:
  run:
    duration_s: 600
    wall_timeout_s: 900
  labels:
    min_windows: 1
scenarios:
  - {name: demo, vehicles: 20, profile: balanced}
splits:
  train: [94101, 94102, 94103, 94104, 94105, 94106, 94107]
  val: [94108]
  test: [94109, 94110]
"@
[System.IO.File]::WriteAllText($dataConfig, $dataYaml, $utf8NoBom)

& $py -m trafma collect --config $dataConfig
if ($LASTEXITCODE -ne 0) { throw "Demo dataset collection failed." }

function Write-LearningConfig {
    param(
        [string]$Path,
        [string]$OutputDir,
        [int]$MaxClients,
        [string]$ClientTypes = "",
        [string]$RsuSelfUpdate = "true",
        [bool]$Central = $false
    )

    $trainBudget = ""
    if ($Central) {
        $trainBudget = @"
training_windows_by_type:
  vehicle: 200
  rsu: 40
"@
    }

    $clientTypesLine = ""
    if ($ClientTypes) { $clientTypesLine = "  client_types: [$ClientTypes]`n" }

    $outputYaml = $OutputDir.Replace('\', '/')
    $yaml = @"
dataset: data/$datasetId
output_dir: $outputYaml
seed: 20261106
evaluation_seed: 20261106
evaluation_windows_per_run: 10
$trainBudget
baseline:
  method: pooled_signal_aware_ridge
  ridge_alpha: 1.0
  minimum_client_windows: 1
training:
  seed: 20261106
  device: auto
  num_threads: 0
  deterministic: true
  hidden_size: 32
  layers: 1
  dropout: 0.0
  shared_size: 16
  learning_rate: 0.001
  batch_size: 8
  epochs: 10
  early_stopping_patience: 3
  regression_loss_weight: 1.0
  classification_loss_weight: 1.0
  max_train_windows: 240
  max_validation_windows: 20
  local_epochs: 1
federation:
  rounds: 10
  max_clients: $MaxClients
  cohort_seed: 20261106
  max_windows_per_client: 10
  max_rsu_windows_per_group: 10
  require_full_client_budget: false
  deadline_s: 300.0
  live_round_interval_s: 60.0
  transient_vehicle_participation_rate: 0.75
  aggregation: sample_weighted_model_weights
  rsu_self_update: $RsuSelfUpdate
$clientTypesLine"@
    [System.IO.File]::WriteAllText($Path, $yaml, $utf8NoBom)
}

$centralConfig = Join-Path $configDir "central.yaml"
$flatConfig = Join-Path $configDir "flat.yaml"
$hierOffConfig = Join-Path $configDir "hierarchical_off.yaml"
$hierOnConfig = Join-Path $configDir "hierarchical_on.yaml"

$centralOut = Join-Path $demoRoot "central"
$flatOut = Join-Path $demoRoot "flat"
$hierOffOut = Join-Path $demoRoot "hierarchical_off"
$hierOnOut = Join-Path $demoRoot "hierarchical_on"

Write-LearningConfig $centralConfig $centralOut 24 -Central $true
Write-LearningConfig $flatConfig $flatOut 20 "vehicle" "false"
Write-LearningConfig $hierOffConfig $hierOffOut 24 "" "false"
Write-LearningConfig $hierOnConfig $hierOnOut 24 "" "true"

function Get-OnlyRunPath {
    param([string]$Path)
    $items = @(Get-ChildItem -LiteralPath $Path -Directory)
    if ($items.Count -ne 1) { throw "Expected one run in $Path; found $($items.Count)." }
    return $items[0].FullName
}

& $py -m trafma train --method central --config $centralConfig --device auto
if ($LASTEXITCODE -ne 0) { throw "Central GRU failed." }
$centralRun = Get-OnlyRunPath $centralOut

& $py -m trafma fedavg --config $flatConfig --threads 0
if ($LASTEXITCODE -ne 0) { throw "Flat FedAvg failed." }
$flatRun = Get-OnlyRunPath $flatOut

& $py -m trafma fedavg --hierarchical --config $hierOffConfig `
    --rsu-self-update false --threads 0
if ($LASTEXITCODE -ne 0) { throw "Hierarchical FedAvg with RSU self-update off failed." }
$hierOffRun = Get-OnlyRunPath $hierOffOut

& $py -m trafma fedavg --hierarchical --config $hierOnConfig `
    --rsu-self-update true --threads 0
if ($LASTEXITCODE -ne 0) { throw "Hierarchical FedAvg with RSU self-update on failed." }
$hierOnRun = Get-OnlyRunPath $hierOnOut

$reportOut = Join-Path $demoRoot "report"
& $py -m trafma run-report --experiment $centralRun --experiment $flatRun `
    --experiment $hierOffRun --experiment $hierOnRun --output $reportOut
if ($LASTEXITCODE -ne 0) { throw "Comparison report failed." }

Write-Host "Simulation plots: $simRun\report"
Write-Host "All-method comparison report: $reportOut\report.html"