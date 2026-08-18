[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [string]$Image,

    [Parameter(Mandatory = $true)]
    [int]$MaxNumSeqs,

    [switch]$EnablePrefixCaching,

    [int]$Port = 8000,
    [int]$MaxModelLen = 2048,
    [double]$GpuMemoryUtilization = 0.85,
    [string]$Dtype = "half",
    [string]$ContainerName = "vllm-serving-lab-server",
    [string]$HuggingFaceCache,
    [string]$VllmCache,
    [int]$StartupTimeoutSeconds = 1800
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($HuggingFaceCache)) {
    $HuggingFaceCache = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.cache\huggingface"))
}
if ([string]::IsNullOrWhiteSpace($VllmCache)) {
    $VllmCache = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.cache\vllm"))
}

if ($ContainerName -notmatch '^vllm-serving-lab-[A-Za-z0-9_.-]+$') {
    throw "ContainerName must start with 'vllm-serving-lab-' and contain only safe characters."
}

docker version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not available. Start Docker Desktop with Linux containers enabled."
}

$existing = docker ps --all --quiet --filter "name=^/$ContainerName$"
if ($existing) {
    docker rm --force $ContainerName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to remove existing container $ContainerName."
    }
}

New-Item -ItemType Directory -Force -Path $HuggingFaceCache | Out-Null
New-Item -ItemType Directory -Force -Path $VllmCache | Out-Null

$dockerArgs = @(
    "run",
    "--detach",
    "--gpus", "all",
    "--ipc=host",
    "--name", $ContainerName,
    "--publish", "${Port}:8000",
    "--volume", "${HuggingFaceCache}:/root/.cache/huggingface",
    "--volume", "${VllmCache}:/root/.cache/vllm",
    $Image,
    "--model", $Model,
    "--served-model-name", $Model,
    "--dtype", $Dtype,
    "--max-model-len", $MaxModelLen,
    "--gpu-memory-utilization", $GpuMemoryUtilization,
    "--max-num-seqs", $MaxNumSeqs
)

if ($EnablePrefixCaching) {
    $dockerArgs += "--enable-prefix-caching"
}
else {
    $dockerArgs += "--no-enable-prefix-caching"
}

$containerId = docker @dockerArgs
if ($LASTEXITCODE -ne 0) {
    throw "Failed to start the vLLM container."
}

$deadline = (Get-Date).AddSeconds($StartupTimeoutSeconds)
$healthUrl = "http://127.0.0.1:$Port/health"
while ((Get-Date) -lt $deadline) {
    $running = docker inspect --format "{{.State.Running}}" $ContainerName 2>$null
    if ($running -ne "true") {
        docker logs --tail 200 $ContainerName
        throw "The vLLM container exited before becoming healthy."
    }

    try {
        $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -eq 200) {
            Write-Output $containerId
            return
        }
    }
    catch {
        Start-Sleep -Seconds 5
    }
}

docker logs --tail 200 $ContainerName
throw "Timed out waiting for $healthUrl."
