[CmdletBinding()]
param(
    [string]$ConfigPath = (Join-Path $PSScriptRoot "..\configs\experiments.json"),
    [int]$Repeats = 3,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [string]$ContainerName = "vllm-serving-lab-server",
    [switch]$SkipExisting
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigPath = (Resolve-Path $ConfigPath).Path

if ($Repeats -le 0) {
    throw "Repeats must be positive."
}
if ($RunId -notmatch '^[A-Za-z0-9_.-]+$') {
    throw "RunId contains unsafe characters."
}

$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$rawOutput = Join-Path $ProjectRoot "results\raw\$RunId"
New-Item -ItemType Directory -Force -Path $rawOutput | Out-Null

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

try {
    foreach ($profileName in @("serial", "standard", "prefix")) {
        $profile = $config.server_profiles.$profileName
        $benchmarks = @($config.benchmarks | Where-Object { $_.profile -eq $profileName })
        if ($SkipExisting) {
            $benchmarks = @($benchmarks | Where-Object {
                $benchmarkName = $_.name
                $missing = $false
                foreach ($number in 1..$Repeats) {
                    if (-not (Test-Path -LiteralPath (Join-Path $rawOutput ("{0}_run{1}.json" -f $benchmarkName, $number)))) {
                        $missing = $true
                    }
                }
                $missing
            })
        }
        if ($benchmarks.Count -eq 0) {
            continue
        }

        & (Join-Path $PSScriptRoot "Start-VllmServer.ps1") `
            -Model $config.model `
            -Image $config.image `
            -MaxNumSeqs $profile.max_num_seqs `
            -EnablePrefixCaching:$profile.prefix_caching `
            -Port $config.port `
            -MaxModelLen $config.server_defaults.max_model_len `
            -GpuMemoryUtilization $config.server_defaults.gpu_memory_utilization `
            -Dtype $config.server_defaults.dtype `
            -ContainerName $ContainerName | Out-Null

        foreach ($benchmark in $benchmarks) {
            foreach ($runNumber in 1..$Repeats) {
                $outputPath = Join-Path $rawOutput ("{0}_run{1}.json" -f $benchmark.name, $runNumber)
                if ($SkipExisting -and (Test-Path -LiteralPath $outputPath)) {
                    Write-Output "Skipping existing $outputPath"
                    continue
                }
                $prefixArgument = if ($profile.prefix_caching) { "--prefix-caching" } else { "--no-prefix-caching" }

                python -m vllm_serving_lab.benchmark `
                    --base-url "http://127.0.0.1:$($config.port)" `
                    --model $config.model `
                    --workload $benchmark.workload `
                    --config-name $benchmark.name `
                    --concurrency $benchmark.concurrency `
                    --requests $benchmark.requests `
                    --warmup $benchmark.warmup `
                    --output-tokens $benchmark.output_tokens `
                    --seed (2026 + $runNumber - 1) `
                    --server-max-num-seqs $profile.max_num_seqs `
                    --server-image $config.image `
                    --run-number $runNumber `
                    $prefixArgument `
                    --output $outputPath

                if ($LASTEXITCODE -ne 0) {
                    throw "Benchmark $($benchmark.name) run $runNumber failed."
                }
            }
        }

        & (Join-Path $PSScriptRoot "Stop-VllmServer.ps1") -ContainerName $ContainerName
    }

    python -m vllm_serving_lab.summarize `
        --input-dir $rawOutput `
        --csv (Join-Path $ProjectRoot "results\summary.csv") `
        --report (Join-Path $ProjectRoot "results\report.md")

    if ($LASTEXITCODE -ne 0) {
        throw "Result aggregation failed."
    }

    Write-Output "Raw results: $rawOutput"
    Write-Output "Summary: $(Join-Path $ProjectRoot 'results\summary.csv')"
    Write-Output "Report: $(Join-Path $ProjectRoot 'results\report.md')"
}
finally {
    try {
        & (Join-Path $PSScriptRoot "Stop-VllmServer.ps1") -ContainerName $ContainerName
    }
    catch {
        Write-Warning $_
    }
    $env:PYTHONPATH = $previousPythonPath
}
