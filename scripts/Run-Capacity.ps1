[CmdletBinding()]
param(
    [double[]]$Rates = @(1, 2, 4),
    [int]$Repeats = 3,
    [int]$Requests = 300,
    [double]$MinDuration = 120,
    [double]$TtftSloMs = 1000,
    [double]$TpotSloMs = 50,
    [int]$OutputTokens = 64,
    [int]$Port = 8000,
    [double]$GpuMemoryUtilization = 0.50,
    [string]$RunId = (Get-Date -Format "yyyyMMdd-HHmmss"),
    [string]$Python,
    [switch]$Offline,
    [switch]$UseExistingServer
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ($RunId -notmatch '^[A-Za-z0-9_.-]+$') { throw "RunId contains unsafe characters." }
$outputDir = Join-Path $projectRoot "results\capacity\$RunId"
if (Test-Path -LiteralPath $outputDir) { throw "Run directory already exists; choose a new RunId." }
if (-not $Python) {
    $Python = @(".venv\Scripts\python.exe", ".venv\bin\python.exe") |
        ForEach-Object { Join-Path $projectRoot $_ } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
    if (-not $Python) { $Python = "python" }
}

$model = "Qwen/Qwen2.5-1.5B-Instruct"
$image = "vllm/vllm-openai:v0.10.2"
$container = "vllm-serving-lab-capacity"
$previousPythonPath = $env:PYTHONPATH
$previousTemp = $env:TEMP
$previousTmp = $env:TMP
$tempDir = Join-Path $projectRoot ".cache\tmp"
New-Item -ItemType Directory -Force -Path $outputDir, $tempDir | Out-Null
$env:PYTHONPATH = Join-Path $projectRoot "src"
$env:TEMP = $tempDir
$env:TMP = $tempDir

try {
    if (-not $UseExistingServer) {
        & (Join-Path $PSScriptRoot "Start-VllmServer.ps1") -Model $model -Image $image `
            -MaxNumSeqs 8 -MaxModelLen 2048 -GpuMemoryUtilization $GpuMemoryUtilization `
            -Port $Port -ContainerName $container -Offline:$Offline | Out-Null
    }
    $inspection = docker inspect $container | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect the capacity server." }
    [ordered]@{
        container = $container
        image = $inspection[0].Config.Image
        image_id = $inspection[0].Image
        command = $inspection[0].Config.Cmd
        mounts = @($inspection[0].Mounts | Select-Object Source, Destination)
        started_at = $inspection[0].State.StartedAt
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $outputDir "server.json") -Encoding utf8

    $invariant = [System.Globalization.CultureInfo]::InvariantCulture
    $arguments = @("-m", "vllm_serving_lab.capacity", "--base-url", "http://127.0.0.1:$Port", "--model", $model,
        "--repeats", "$Repeats", "--requests", "$Requests", "--min-duration", $MinDuration.ToString($invariant),
        "--ttft-slo-ms", $TtftSloMs.ToString($invariant), "--tpot-slo-ms", $TpotSloMs.ToString($invariant),
        "--output-tokens", "$OutputTokens", "--output-dir", $outputDir, "--rates")
    $arguments += @($Rates | ForEach-Object { $_.ToString($invariant) })
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Capacity sweep failed; completed runs remain in $outputDir." }
    Write-Output "Capacity report: $(Join-Path $outputDir 'report.md')"
}
finally {
    if (-not $UseExistingServer) {
        & (Join-Path $PSScriptRoot "Stop-VllmServer.ps1") -ContainerName $container
    }
    $env:PYTHONPATH = $previousPythonPath
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}
