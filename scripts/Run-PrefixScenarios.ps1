[CmdletBinding()]
param(
    [int[]]$IdleGapsMs = @(0, 30000),
    [int[]]$BackgroundUniquePrefixes = @(0, 4),
    [int]$Requests = 32,
    [int]$Sessions = 8,
    [int]$Rounds = 4,
    [int]$Repeats = 1,
    [int]$Port = 8000,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [string]$Python,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($RunId -notmatch '^[A-Za-z0-9_.-]+$') { throw "RunId contains unsafe characters." }
if (-not $Python) { $Python = Join-Path $projectRoot ".venv\bin\python.exe"; if (-not (Test-Path $Python)) { $Python = "python" } }
$outputDir = Join-Path $projectRoot "results\prefix-scenarios\$RunId"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
$model = "Qwen/Qwen2.5-1.5B-Instruct"; $image = "vllm/vllm-openai:v0.10.2"; $container = "vllm-serving-lab-prefix"
$previousPythonPath = $env:PYTHONPATH; $env:PYTHONPATH = Join-Path $projectRoot "src"
try {
    foreach ($gap in $IdleGapsMs) {
        foreach ($background in $BackgroundUniquePrefixes) {
            foreach ($prefixCaching in @($false, $true)) {
                $mode = if ($prefixCaching) { "on" } else { "off" }
                $name = "gap${gap}_bg${background}_pc${mode}"
                $scenarioDir = Join-Path $outputDir $name; New-Item -ItemType Directory -Force -Path $scenarioDir | Out-Null
                & (Join-Path $PSScriptRoot "Start-VllmServer.ps1") -Model $model -Image $image -MaxNumSeqs 8 `
                    -MaxModelLen 2048 -GpuMemoryUtilization 0.85 -EnablePrefixCaching:$prefixCaching `
                    -Port $Port -ContainerName $container -Offline:$Offline | Out-Null
                try {
                    for ($repeat = 1; $repeat -le $Repeats; $repeat++) {
                        & $Python -m vllm_serving_lab.benchmark --base-url "http://127.0.0.1:$Port" --model $model `
                            --workload session-chat --config-name $name --concurrency 8 --requests $Requests `
                            --warmup 8 --output-tokens 32 --server-max-num-seqs 8 --server-image $image `
                            --run-number $repeat --sessions $Sessions --rounds $Rounds --idle-gap-ms $gap `
                            --background-unique-prefixes $background --prefix-caching:$prefixCaching `
                            --output (Join-Path $scenarioDir ("run{0}.json" -f $repeat))
                        if ($LASTEXITCODE -ne 0) { throw "Prefix scenario failed: $name repeat $repeat." }
                    }
                }
                finally { & (Join-Path $PSScriptRoot "Stop-VllmServer.ps1") -ContainerName $container }
            }
        }
    }
    Write-Output "Prefix scenario artifacts: $outputDir"
}
finally { $env:PYTHONPATH = $previousPythonPath }
