[CmdletBinding()]
param(
    [int]$Requests = 60,
    [int]$Concurrency = 8,
    [int]$Port = 8000,
    [int]$GpuCapacityMiB = 256,
    [int]$CpuCapacityMiB = 512,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [string]$Python,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($RunId -notmatch '^[A-Za-z0-9_.-]+$') { throw "RunId contains unsafe characters." }
if (-not $Python) { $Python = Join-Path $projectRoot ".venv\bin\python.exe"; if (-not (Test-Path $Python)) { $Python = "python" } }
$outputDir = Join-Path $projectRoot "results\kv-tiers\$RunId"; New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$model = "Qwen/Qwen2.5-1.5B-Instruct"; $image = "vllm/vllm-openai:v0.10.2"; $container = "vllm-serving-lab-kv-tier"
$previousPythonPath = $env:PYTHONPATH; $env:PYTHONPATH = Join-Path $projectRoot "src"
try {
    foreach ($tier in @(@{ Name = "gpu-only"; L2 = $false }, @{ Name = "gpu-cpu-l2"; L2 = $true })) {
        $tierDir = Join-Path $outputDir $tier.Name; New-Item -ItemType Directory -Force -Path $tierDir | Out-Null
        $l2Path = Join-Path $tierDir "cpu-l2"
        $start = @{ Model=$model; Image=$image; MaxNumSeqs=8; MaxModelLen=2048; GpuMemoryUtilization=0.85; KvCacheMemoryBytes=([long]$GpuCapacityMiB*1024*1024); EnablePrefixCaching=$true; Port=$Port; ContainerName=$container; Offline=$Offline }
        if ($tier.L2) { $start.EnableCpuKvCache=$true; $start.CpuKvCacheBytes=([long]$CpuCapacityMiB*1024*1024); $start.CpuKvAdmissionPolicy="lru"; $start.CpuKvCachePath=$l2Path }
        & (Join-Path $PSScriptRoot "Start-VllmServer.ps1") @start | Out-Null
        try {
            & $Python -m vllm_serving_lab.benchmark --base-url "http://127.0.0.1:$Port" --model $model `
                --workload coding-agent --config-name $tier.Name --concurrency $Concurrency --requests $Requests `
                --warmup 8 --output-tokens 32 --server-max-num-seqs 8 --server-image $image --run-number 1 `
                --prefix-caching --output (Join-Path $tierDir "run1.json")
            if ($LASTEXITCODE -ne 0) { throw "KV tier benchmark failed: $($tier.Name)." }
        }
        finally {
            & (Join-Path $PSScriptRoot "Stop-VllmServer.ps1") -ContainerName $container
            if ($tier.L2) {
                $files = @(Get-ChildItem -LiteralPath $l2Path -Recurse -File -ErrorAction SilentlyContinue)
                [ordered]@{ path = $l2Path; file_count = $files.Count; bytes = [long](($files | Measure-Object Length -Sum).Sum) } |
                    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $tierDir "l2_footprint.json") -Encoding utf8
            }
        }
    }
    Write-Output "KV tier artifacts: $outputDir"
}
finally { $env:PYTHONPATH = $previousPythonPath }
