[CmdletBinding()]
param(
    [int[]]$CapacitiesMiB = @(128, 256, 512),
    [int]$Requests = 60,
    [int]$Repeats = 1,
    [int]$Concurrency = 8,
    [int]$Sessions = 4,
    [int]$Rounds = 4,
    [int]$BackgroundUniquePrefixes = 2,
    [int]$IdleGapMs = 0,
    [int]$Port = 8000,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [string]$Python,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($RunId -notmatch '^[A-Za-z0-9_.-]+$') { throw "RunId contains unsafe characters." }
if (-not $CapacitiesMiB -or ($CapacitiesMiB | Where-Object { $_ -le 0 })) { throw "CapacitiesMiB must be positive." }
if (-not $Python) { $Python = Join-Path $projectRoot ".venv\bin\python.exe"; if (-not (Test-Path $Python)) { $Python = "python" } }
$outputDir = Join-Path $projectRoot "results\kv-capacity\$RunId"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$model = "Qwen/Qwen2.5-1.5B-Instruct"; $image = "vllm/vllm-openai:v0.10.2"; $container = "vllm-serving-lab-kv-capacity"
$previousPythonPath = $env:PYTHONPATH; $env:PYTHONPATH = Join-Path $projectRoot "src"
try {
    foreach ($capacity in $CapacitiesMiB) {
        $bytes = [long]$capacity * 1024 * 1024
        & (Join-Path $PSScriptRoot "Start-VllmServer.ps1") -Model $model -Image $image -MaxNumSeqs 8 `
            -MaxModelLen 2048 -GpuMemoryUtilization 0.85 -KvCacheMemoryBytes $bytes `
            -EnablePrefixCaching -Port $Port -ContainerName $container -Offline:$Offline | Out-Null
        try {
            $capacityDir = Join-Path $outputDir ("{0}MiB" -f $capacity)
            New-Item -ItemType Directory -Force -Path $capacityDir | Out-Null
            for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
                $output = Join-Path $capacityDir ("run{0}.json" -f $repeat)
                & $Python -m vllm_serving_lab.benchmark --base-url "http://127.0.0.1:$Port" `
                    --model $model --workload session-chat --config-name ("kv-{0}MiB" -f $capacity) `
                    --concurrency $Concurrency --requests $Requests --warmup 8 --output-tokens 32 `
                    --server-max-num-seqs 8 --server-image $image --run-number $repeat `
                    --sessions $Sessions --rounds $Rounds --idle-gap-ms $IdleGapMs `
                    --background-unique-prefixes $BackgroundUniquePrefixes `
                    --prefix-caching --output $output
                if ($LASTEXITCODE -ne 0) { throw "KV capacity benchmark failed at ${capacity}MiB repeat $repeat." }
            }
        }
        finally { & (Join-Path $PSScriptRoot "Stop-VllmServer.ps1") -ContainerName $container }
    }
    Write-Output "KV capacity artifacts: $outputDir"
}
finally { $env:PYTHONPATH = $previousPythonPath }
