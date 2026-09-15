[CmdletBinding()]
param(
    [string]$Namespace = "vllm-lab",
    [string]$Deployment = "vllm-gateway",
    [string]$Service = "vllm-gateway",
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"

kubectl version --client=true | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "kubectl is not available. Install kubectl and configure a cluster context."
}

kubectl -n $Namespace rollout status "deployment/$Deployment" --timeout="${TimeoutSeconds}s"
if ($LASTEXITCODE -ne 0) {
    throw "Deployment $Deployment did not become ready."
}

kubectl -n $Namespace get pods -l app.kubernetes.io/name=$Deployment -o wide
kubectl -n $Namespace get service $Service
kubectl -n $Namespace get endpointslice -l kubernetes.io/service-name=$Service

Write-Output "Gateway is ready. To exercise it locally, run:"
Write-Output "kubectl -n $Namespace port-forward service/$Service 18000:8000"
