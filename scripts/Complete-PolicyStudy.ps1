[CmdletBinding()]
param(
    [string]$PreflightOutput = 'results/cache-policy/20261008-preflight-v2',
    [string]$Output = 'results/cache-policy/20261008-policy-study',
    [string]$Branch = 'feat/cache-replacement-policy-lab'
)
$ErrorActionPreference = 'Stop'
$taskRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Set-Location -LiteralPath $taskRoot
$taskPython = Join-Path $taskRoot '.venv/bin/python.exe'
$env:GIT_TERMINAL_PROMPT = '0'
$env:GCM_INTERACTIVE = 'never'
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class PolicyStudyWake {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
[PolicyStudyWake]::SetThreadExecutionState([uint32]2147483649) | Out-Null
try {
if ((git branch --show-current).Trim() -ne $Branch) { throw 'Experiment branch changed; refusing to publish another branch.' }

& $taskPython -u -m vllm_serving_lab.policy_study --output $PreflightOutput --phase preflight
if ($LASTEXITCODE -ne 0) { throw 'Preflight failed; formal GPU measurements not started.' }
& $taskPython -u -m vllm_serving_lab.policy_study --output $Output --phase all
if ($LASTEXITCODE -ne 0) { throw 'Formal study interrupted; preserved checkpoints can be resumed.' }

$totalPath = Join-Path $taskRoot 'results/EXPERIMENT-REPORT.md'
$batchName = Split-Path $Output -Leaf
$totalText = Get-Content -LiteralPath $totalPath -Raw -Encoding utf8
if ($totalText.Contains("## 13. CPU L2 替换策略实验（$batchName）")) {
    & $taskPython -u -m vllm_serving_lab.policy_report --input $Output
} else {
    & $taskPython -u -m vllm_serving_lab.policy_report --input $Output --update-total
}
if ($LASTEXITCODE -ne 0) { throw 'Data audit rejected publication.' }
if ((git branch --show-current).Trim() -ne $Branch) { throw 'Experiment branch changed before publication.' }

$publishPaths = @('src/vllm_serving_lab/replacement.py','src/vllm_serving_lab/policy_connector.py',
    'src/vllm_serving_lab/policy_workloads.py','src/vllm_serving_lab/policy_study.py',
    'src/vllm_serving_lab/policy_report.py','src/vllm_serving_lab/client.py',
    'src/vllm_serving_lab/workloads.py','src/vllm_serving_lab/capacity.py',
    'scripts/Start-VllmServer.ps1','scripts/Complete-PolicyStudy.ps1',
    'tests/test_replacement.py','tests/test_policy_workloads.py','tests/test_policy_study.py','tests/test_policy_connector.py','tests/test_client.py',
    'configs/cache_policy_tokens.json','docs/cache-policy-plan.md',
    '.gitignore','README.md','README_CN.md','results/EXPERIMENT-REPORT.md',$Output)
$files = Get-ChildItem -LiteralPath $Output -Recurse -File
if ($files | Where-Object { $_.Length -ge 95000000 }) { throw 'A publication file exceeds the safe GitHub blob size.' }
git add -- $publishPaths
if ($LASTEXITCODE -ne 0) { throw 'Git staging failed.' }
$stagedPaths = @(git diff --cached --name-only)
foreach ($stagedPath in $stagedPaths) {
    $allowed = $false
    foreach ($publishPath in $publishPaths) {
        if ($stagedPath -eq $publishPath -or $stagedPath.StartsWith($publishPath.TrimEnd('/')+'/')) { $allowed = $true; break }
    }
    if (-not $allowed) { throw "Unrelated staged path detected: $stagedPath" }
}
git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    git commit -m 'feat: benchmark LRU LFU ARC CPU KV replacement and SLO capacity'
    if ($LASTEXITCODE -ne 0) { throw 'Git commit failed.' }
}
git push --set-upstream origin $Branch
if ($LASTEXITCODE -ne 0) { throw 'GitHub push failed; completed experiment and reports remain local.' }
$completion = @{ status='complete'; branch=$Branch; commit=(git rev-parse HEAD).Trim(); report="$Output/REPORT.md" }
$completion | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRoot '.cache/policy-study-completion.json') -Encoding utf8
Write-Output ($completion | ConvertTo-Json -Compress)
} finally {
    [PolicyStudyWake]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
