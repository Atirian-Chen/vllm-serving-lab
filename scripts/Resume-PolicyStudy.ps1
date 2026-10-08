[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$NativeDockerExe,
    [string]$PreflightOutput = 'results/cache-policy/20261008-preflight-v2',
    [string]$Output = 'results/cache-policy/20261008-policy-study',
    [string]$Branch = 'feat/cache-replacement-policy-lab',
    [ValidateRange(15, 300)]
    [int]$ProbeIntervalSeconds = 30
)
$ErrorActionPreference = 'Stop'
$policyRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $policyRoot
$NativeDockerExe = [System.IO.Path]::GetFullPath($NativeDockerExe)
if (-not (Test-Path -LiteralPath $NativeDockerExe -PathType Leaf)) { throw 'Original Docker executable is missing.' }
if ((git branch --show-current).Trim() -ne $Branch) { throw 'Study branch changed.' }

# One host waiter owns the eventual runner. Neither reboot Windows nor change
# Docker/WSL configuration here. All completed measurements remain checkpoints.
$policyMutex = [System.Threading.Mutex]::new($false, 'Local\VllmServingLabPolicyStudyResume')
$policyOwnedMutex = $false
$policyStatePath = Join-Path $policyRoot '.cache/policy-study-waiter.json'
$policyRecord = @{pid=$PID;started_utc=[DateTime]::UtcNow.ToString('o');status='waiting-for-engine';
    branch=$Branch;formal=$Output;preflight=$PreflightOutput;native_docker=$NativeDockerExe;probe_attempts=0}
try {
    try { $policyOwnedMutex = $policyMutex.WaitOne(0) }
    catch [System.Threading.AbandonedMutexException] { $policyOwnedMutex = $true }
    if (-not $policyOwnedMutex) { throw 'A policy study waiter already owns this batch.' }
    $policyRecord | ConvertTo-Json | Set-Content -LiteralPath $policyStatePath -Encoding utf8

    while ($true) {
        $policyStart = [System.Diagnostics.ProcessStartInfo]::new($NativeDockerExe)
        # This filter forces a real engine request. Unfiltered ps, version and
        # inspect can be served from stale Docker Desktop API cache.
        $policyStart.Arguments = 'ps --all --quiet --filter "name=^/vllm-serving-lab-cache-policy$"'
        $policyStart.UseShellExecute = $false
        $policyStart.CreateNoWindow = $true
        $policyStart.RedirectStandardOutput = $true
        $policyStart.RedirectStandardError = $true
        $policyStart.Environment['GOMEMLIMIT'] = '64MiB'
        $policyStart.Environment['GOMAXPROCS'] = '2'
        $policyStart.Environment['GOGC'] = '25'
        $policyProbe = [System.Diagnostics.Process]::Start($policyStart)
        try {
            $policyStdout = $policyProbe.StandardOutput.ReadToEndAsync()
            $policyStderr = $policyProbe.StandardError.ReadToEndAsync()
            $policyTimedOut = -not $policyProbe.WaitForExit(10000)
            if ($policyTimedOut) { $policyProbe.Kill() }
            $policyProbe.WaitForExit()
            $policyError = $policyStderr.GetAwaiter().GetResult()
            $null = $policyStdout.GetAwaiter().GetResult()
            $policyHealthy = -not $policyTimedOut -and $policyProbe.ExitCode -eq 0
            $policyRecord.probe_attempts++
            $policyRecord.last_probe_utc = [DateTime]::UtcNow.ToString('o')
            $policyRecord.last_probe_timeout = $policyTimedOut
            $policyRecord.last_probe_exit = $policyProbe.ExitCode
            $policyRecord.last_probe_error = $policyError.Substring(0, [Math]::Min(2048, $policyError.Length))
            $policyRecord | ConvertTo-Json | Set-Content -LiteralPath $policyStatePath -Encoding utf8
        } finally { $policyProbe.Dispose() }
        if ($policyHealthy) { break }
        $policyDelay = [int][Math]::Min(300, $ProbeIntervalSeconds * [Math]::Pow(2, [Math]::Min(4, $policyRecord.probe_attempts - 1)))
        $policyRecord.next_probe_delay_seconds = $policyDelay
        $policyRecord | ConvertTo-Json | Set-Content -LiteralPath $policyStatePath -Encoding utf8
        Start-Sleep -Seconds $policyDelay
    }

    $policyAdapterDirectory = Join-Path $policyRoot '.cache/policy-docker'
    New-Item -ItemType Directory -Path $policyAdapterDirectory -Force | Out-Null
    $policyAdapter = Join-Path $policyAdapterDirectory 'docker.exe'
    $policyCompiler = Join-Path $env:SystemRoot 'Microsoft.NET/Framework64/v4.0.30319/csc.exe'
    & $policyCompiler /nologo /target:exe "/out:$policyAdapter" (Join-Path $PSScriptRoot 'PolicyDockerRetry.cs')
    if ($LASTEXITCODE -ne 0) { throw 'Read-only query adapter compilation failed.' }
    $env:POLICY_DOCKER_NATIVE_EXE = $NativeDockerExe
    $env:POLICY_DOCKER_RETRY_LOG = Join-Path $policyRoot '.cache/policy-docker-retry.log'
    $env:GOMEMLIMIT = '64MiB'
    $env:GOMAXPROCS = '2'
    $env:GOGC = '25'
    $env:PATH = $policyAdapterDirectory + ';' + $env:PATH
    $policyRecord.status = 'running'
    $policyRecord.runner_started_utc = [DateTime]::UtcNow.ToString('o')
    $policyRecord.adapter_source_sha256 = (Get-FileHash -LiteralPath 'scripts/PolicyDockerRetry.cs' -Algorithm SHA256).Hash
    $policyRecord.adapter_binary_sha256 = (Get-FileHash -LiteralPath $policyAdapter -Algorithm SHA256).Hash
    $policyRecord | ConvertTo-Json | Set-Content -LiteralPath $policyStatePath -Encoding utf8
    $policyRecord | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Output 'controller-resume4.json') -Encoding utf8
    Write-Output "Engine responded; resuming the frozen batch and report/push workflow."
    & (Join-Path $PSScriptRoot 'Complete-PolicyStudy.ps1') -PreflightOutput $PreflightOutput -Output $Output -Branch $Branch
    if (-not (Test-Path -LiteralPath '.cache/policy-study-completion.json')) { throw 'Runner returned without a publication completion marker.' }
    $policyRecord.status = 'complete'
    $policyRecord.completed_utc = [DateTime]::UtcNow.ToString('o')
    $policyRecord | ConvertTo-Json | Set-Content -LiteralPath $policyStatePath -Encoding utf8
} catch {
    if ($policyOwnedMutex) {
        $policyRecord.status = 'failed'
        $policyRecord.error = $_.Exception.Message
        $policyRecord | ConvertTo-Json | Set-Content -LiteralPath $policyStatePath -Encoding utf8
    }
    throw
} finally {
    if ($policyOwnedMutex) { $policyMutex.ReleaseMutex() }
    $policyMutex.Dispose()
}
