[CmdletBinding()]
param(
    [string]$Repository = "vllm-serving-lab-gateway",
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_.-]*$')]
    [string]$Tag = "v1"
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$gatewayDir = Join-Path $projectRoot "gateway"
$image = "${Repository}:${Tag}"

docker version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Docker is not available. Start Docker Desktop with Linux containers enabled."
}
if (-not (Test-Path (Join-Path $gatewayDir "Dockerfile"))) {
    throw "Gateway Dockerfile was not found at $gatewayDir."
}

docker build --build-arg "APP_VERSION=$Tag" --tag $image $gatewayDir
if ($LASTEXITCODE -ne 0) {
    throw "Failed to build $image."
}

Write-Output $image
