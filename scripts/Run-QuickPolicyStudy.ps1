param(
    [string]$Output = "results/cache-policy/20261010-policy-study-quick",
    [string]$NativeDockerExe = "E:/Docker/DockerDesktop/resources/bin/docker.exe"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:POLICY_DOCKER_NATIVE_EXE = $NativeDockerExe
$env:POLICY_DOCKER_RETRY_LOG = Join-Path $repo ".cache/policy-docker-retry.log"
$env:GOMEMLIMIT = "64MiB"
$env:GOMAXPROCS = "2"
$env:GOGC = "20"
$adapter = Join-Path $repo ".cache/policy-docker"
$env:PATH = "$adapter;$env:PATH"
$python = Join-Path $repo ".venv/bin/python.exe"
& $python -u (Join-Path $repo "scripts/Quick-PolicyStudy.py") --output (Join-Path $repo $Output)
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
