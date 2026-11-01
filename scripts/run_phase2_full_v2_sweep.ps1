param(
    [int[]]$Seeds = @(20261031, 20261032, 20261033, 20261034, 20261035),
    [ValidateSet("central", "flat", "iid", "hierarchical-true", "hierarchical-false")]
    [string[]]$Methods = @("central", "flat", "iid", "hierarchical-true", "hierarchical-false")
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Project virtual-environment Python was not found: $python"
}
if (-not (Test-Path "data\phase2_v2\dataset_manifest.json")) {
    throw "Full-v2 dataset manifest was not found."
}

function Get-CompletedExperiment {
    param(
        [string]$Prefix,
        [int]$Seed,
        [string]$ExpectedMethod,
        [Nullable[bool]]$ExpectedSelfUpdate
    )

    $candidates = Get-ChildItem "runs\phase2" -Directory -Filter "${Prefix}_phase2_v2_seed${Seed}_*"
    foreach ($candidate in $candidates) {
        $manifestPath = Join-Path $candidate.FullName "manifest.json"
        $metricsPath = Join-Path $candidate.FullName "metrics.json"
        $predictionsPath = Join-Path $candidate.FullName "predictions.csv"
        if (-not ((Test-Path $manifestPath) -and (Test-Path $metricsPath) -and (Test-Path $predictionsPath))) {
            continue
        }

        try {
            $manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json
        }
        catch {
            continue
        }
        if ($manifest.dataset_id -ne "phase2_v2" -or
            [int]$manifest.seed -ne $Seed -or
            $manifest.method -ne $ExpectedMethod) {
            continue
        }
        if ($null -ne $ExpectedSelfUpdate -and
            [bool]$manifest.rsu_self_update -ne [bool]$ExpectedSelfUpdate) {
            continue
        }
        return $candidate.FullName
    }
    return $null
}

$plans = @{
    central = @{
        Prefix = "central_gru"
        ExpectedMethod = "central_pooled_offline_gru"
        Arguments = { param($seed) @("-m", "trafma", "train", "--config", "configs/learning.yaml", "--dataset", "data/phase2_v2", "--method", "central", "--training-seed", "$seed") }
        SelfUpdate = $null
    }
    flat = @{
        Prefix = "flat_fedavg"
        ExpectedMethod = "flat_sample_weighted_fedavg"
        Arguments = { param($seed) @("-m", "trafma", "fedavg", "--config", "configs/learning.yaml", "--dataset", "data/phase2_v2", "--training-seed", "$seed") }
        SelfUpdate = $null
    }
    iid = @{
        Prefix = "flat_fedavg_iid"
        ExpectedMethod = "flat_synthetic_iid_fedavg"
        Arguments = { param($seed) @("-m", "trafma", "fedavg", "--config", "configs/learning.yaml", "--dataset", "data/phase2_v2", "--iid-control", "--training-seed", "$seed") }
        SelfUpdate = $null
    }
    "hierarchical-true" = @{
        Prefix = "hierarchical_fedavg"
        ExpectedMethod = "hierarchical_sample_weighted_fedavg"
        Arguments = { param($seed) @("-m", "trafma", "fedavg", "--config", "configs/learning.yaml", "--dataset", "data/phase2_v2", "--hierarchical", "--rsu-self-update", "true", "--training-seed", "$seed") }
        SelfUpdate = $true
    }
    "hierarchical-false" = @{
        Prefix = "hierarchical_fedavg"
        ExpectedMethod = "hierarchical_sample_weighted_fedavg"
        Arguments = { param($seed) @("-m", "trafma", "fedavg", "--config", "configs/learning.yaml", "--dataset", "data/phase2_v2", "--hierarchical", "--rsu-self-update", "false", "--training-seed", "$seed") }
        SelfUpdate = $false
    }
}

foreach ($seed in $Seeds) {
    if ($seed -ge 400 -and $seed -le 404) {
        throw "Seeds 400-404 are reserved and must remain unused."
    }
}

foreach ($method in $Methods) {
    $plan = $plans[$method]
    foreach ($seed in $Seeds) {
        $completedRun = Get-CompletedExperiment -Prefix $plan.Prefix -Seed $seed `
            -ExpectedMethod $plan.ExpectedMethod -ExpectedSelfUpdate $plan.SelfUpdate
        if ($null -ne $completedRun) {
            Write-Output "Already complete: $method seed $seed"
            continue
        }

        Write-Output "Starting full-v2 $method seed $seed"
        $arguments = & $plan.Arguments $seed
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment failed: $method seed $seed (exit code $LASTEXITCODE). Rerun this script to continue."
        }
    }
}

$uniqueSeeds = @($Seeds | Sort-Object -Unique)
if ($uniqueSeeds.Count -lt 5) {
    Write-Output "Sweep commands finished. Paired statistics need at least five distinct seeds per method."
    exit 0
}

$statsExperiments = @()
foreach ($method in $Methods) {
    $plan = $plans[$method]
    foreach ($seed in $uniqueSeeds) {
        $completedRun = Get-CompletedExperiment -Prefix $plan.Prefix -Seed $seed `
            -ExpectedMethod $plan.ExpectedMethod -ExpectedSelfUpdate $plan.SelfUpdate
        if ($null -eq $completedRun) {
            throw "Cannot run paired statistics: missing complete $method experiment for seed $seed."
        }
        $statsExperiments += $completedRun
    }
}

$baseline = "runs\phase2\baseline_phase2_v2_20261031T164408Z"
if (-not (Test-Path (Join-Path $baseline "predictions.csv"))) {
    throw "Full-v2 baseline predictions were not found: $baseline"
}
$statsOutput = "runs\phase2\stats_phase2_v2_$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
$statsArguments = @("-m", "trafma", "phase2-stats", "--baseline", $baseline,
    "--output", $statsOutput, "--bootstrap-replicates", "10000",
    "--bootstrap-seed", "20261031")
foreach ($experiment in $statsExperiments) {
    $statsArguments += @("--experiment", $experiment)
}
Write-Output "Creating paired run-level statistics: $statsOutput"
& $python @statsArguments
if ($LASTEXITCODE -ne 0) {
    throw "Paired statistics failed (exit code $LASTEXITCODE)."
}
