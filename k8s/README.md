# Kubernetes Gateway Extension

This directory deploys a small CPU FastAPI gateway in front of the existing vLLM Docker server. Kubernetes manages the gateway; the GPU vLLM process remains outside the cluster for this first extension.

## Prerequisites

- Docker Desktop with Linux containers and a Kubernetes context, or a local kind/minikube cluster.
- `kubectl` configured for the selected cluster.
- The existing vLLM container serving `http://127.0.0.1:8000/health`.
- A local gateway image loaded into the selected cluster.

The current ConfigMap uses Docker Desktop's `host.docker.internal` address. Start the vLLM container with `-PublishAddress 0.0.0.0` for the local Kubernetes demo:

```powershell
.\scripts\Start-VllmServer.ps1 `
  -Model Qwen/Qwen2.5-1.5B-Instruct `
  -Image vllm/vllm-openai:v0.10.2 `
  -MaxNumSeqs 8 `
  -PublishAddress 0.0.0.0
```

The default remains `127.0.0.1`, so existing benchmark runs keep their original binding behavior.

Before applying the gateway, verify that a Pod can reach the host service:

```powershell
kubectl -n default run netcheck --rm -it --restart=Never `
  --image=curlimages/curl:8.10.1 -- `
  curl -fsS http://host.docker.internal:8000/health
```

If the selected cluster does not provide this DNS name, replace `VLLM_BASE_URL` in `configmap.yaml` with the host address documented by that cluster. Do not commit a machine-specific IP address.

## Build and deploy

```powershell
.\scripts\Build-Gateway.ps1 -Tag v1
kubectl kustomize .\k8s
kubectl apply -k .\k8s
kubectl -n vllm-lab rollout status deployment/vllm-gateway --timeout=120s
kubectl -n vllm-lab get pods -o wide
kubectl -n vllm-lab get endpointslice -l kubernetes.io/service-name=vllm-gateway
```

Docker Desktop can use the locally built image. For kind, load it explicitly:

```powershell
kind load docker-image vllm-serving-lab-gateway:v1
```

For minikube, use `minikube image load vllm-serving-lab-gateway:v1`.

## Checks

Run a request from inside the cluster. This tests the Service DNS path rather than a local tunnel:

```powershell
kubectl -n vllm-lab run curl --rm -it --restart=Never `
  --image=curlimages/curl:8.10.1 -- `
  curl -fsS http://vllm-gateway:8000/health/ready
```

Use `/debug/info` and multiple new connections to observe more than one gateway Pod. `kubectl port-forward service/vllm-gateway ...` is useful for a local protocol smoke test, but it selects a backend and does not prove Service distribution.

Delete one gateway Pod and observe Deployment recovery:

```powershell
$pod = kubectl -n vllm-lab get pods -l app.kubernetes.io/name=vllm-gateway -o jsonpath='{.items[0].metadata.name}'
kubectl -n vllm-lab delete pod $pod
kubectl -n vllm-lab get pods -l app.kubernetes.io/name=vllm-gateway -w
```

Probe behavior can be exercised without changing the Deployment:

```powershell
kubectl -n vllm-lab exec $pod -- touch /tmp/not-ready
kubectl -n vllm-lab get endpointslice -l kubernetes.io/service-name=vllm-gateway
kubectl -n vllm-lab exec $pod -- rm /tmp/not-ready

kubectl -n vllm-lab exec $pod -- touch /tmp/not-live
kubectl -n vllm-lab get pod $pod -w
```

The first flag removes the Pod from ready endpoints without restarting it. The second should increase the container restart count. These files are local test controls, not production health logic.

## Rollout and rollback

Build a second image with a version embedded in `/debug/info`, then update and undo the Deployment:

```powershell
.\scripts\Build-Gateway.ps1 -Tag v2
kubectl -n vllm-lab set image deployment/vllm-gateway gateway=vllm-serving-lab-gateway:v2
kubectl -n vllm-lab rollout status deployment/vllm-gateway --timeout=120s
kubectl -n vllm-lab rollout history deployment/vllm-gateway
kubectl -n vllm-lab rollout undo deployment/vllm-gateway
kubectl -n vllm-lab rollout status deployment/vllm-gateway --timeout=120s
```

Use short requests for the rollout check. Long SSE requests can outlive the 30-second termination grace period and are not used to claim unconditional zero downtime.

## Scope

This is a local single-node Kubernetes deployment of two CPU gateway replicas in front of a host Docker vLLM server. It demonstrates Service discovery, health probes, Deployment recovery, resource declarations and image rollout. It does not demonstrate GPU scheduling, node-level high availability, production authentication, or a linear inference throughput increase.
