# Kubernetes Gateway Extension Report

Status: implementation complete; cluster execution pending local Docker/Kubernetes availability.

Branch: `feat/kubernetes-gateway`

## What was added

- `gateway/app.py`: FastAPI gateway for the existing `POST /v1/completions` endpoint, including normal and SSE responses.
- `gateway/Dockerfile` and `gateway/requirements.txt`: independent gateway image and dependencies.
- `GET /health/live`: process-only liveness check.
- `GET /health/ready`: short-timeout check of the upstream vLLM `/health` endpoint.
- `GET /debug/info`: Pod name and image version for Service-distribution and rollout checks.
- `/tmp/not-ready` and `/tmp/not-live`: per-Pod local controls for probe demonstrations.
- `k8s/`: Namespace, ConfigMap, two-replica Deployment, ClusterIP Service and kustomization.
- `scripts/Build-Gateway.ps1`: local image build with an embedded version.
- `scripts/Verify-K8s.ps1`: rollout and endpoint inspection.
- `scripts/Start-VllmServer.ps1`: optional `-PublishAddress`, preserving the old loopback default.
- `tests/test_gateway.py`: focused tests for forwarding, SSE cleanup, health checks and debug identity.

## Verification completed in this workspace

| Check | Result |
|---|---|
| Gateway unit tests | 5 passed |
| Gateway Python compilation | Passed |
| `kubectl kustomize k8s` | Passed |
| PowerShell parser checks for build/verify scripts | Passed |
| Existing vLLM benchmark suite | 19 passed, 5 skipped with an isolated temporary directory |
| Docker image build | Not run: Docker daemon unavailable |
| `kubectl apply` / live Pod checks | Not run: no Kubernetes context configured |

The live checks must be run on a machine with Docker Desktop or another local Kubernetes runtime. Until then, no latency, recovery-time or rollout-error numbers are claimed.

## Expected acceptance evidence

1. Two gateway Pods reach `Ready` and the Service has two ready endpoints.
2. A cluster-internal request reaches the real vLLM `/health` and `/v1/completions` paths.
3. New cluster-internal connections observe both Pod names through `/debug/info`.
4. Deleting one Pod returns the Deployment to two Ready replicas.
5. Readiness failure removes one endpoint without a restart; liveness failure increments `restartCount`.
6. v1 -> v2 -> v1 completes as a rolling update and rollback.

## Interpretation boundary

The extension adds service lifecycle and operational evidence around the existing vLLM benchmark. It does not change vLLM scheduling, KV cache behavior or GPU capacity. Any reported gateway overhead, recovery duration, error count or rollout result must include the cluster type, model, request shape, concurrency and whether a local port-forward was used.
