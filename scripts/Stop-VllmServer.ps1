[CmdletBinding()]
param(
    [string]$ContainerName = "vllm-serving-lab-server"
)

$ErrorActionPreference = "Stop"

if ($ContainerName -notmatch '^vllm-serving-lab-[A-Za-z0-9_.-]+$') {
    throw "ContainerName must start with 'vllm-serving-lab-' and contain only safe characters."
}

$existing = docker ps --all --quiet --filter "name=^/$ContainerName$"
if ($existing) {
    docker rm --force $ContainerName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to remove container $ContainerName."
    }
}

