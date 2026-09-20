# Kubernetes Gateway Extension Report

Status: implementation complete; static and contract checks passed; live Docker/Kubernetes execution is blocked by the local Docker Engine.

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
| Docker image build | Blocked: `desktop-linux` client context selected, but `docker info`/`docker version` Server did not respond within 15 s after Docker Desktop restart and WSL reset |
| `kubectl apply` / live Pod checks | Not run: `kubectl config get-contexts` has no current Kubernetes context |

## Executed sub-experiments (2026-09-21)

These are deliberately split so the interview story distinguishes application behavior from cluster behavior.

### K8S-1 — Gateway contract behavior

Command: `python -m pytest -q tests/test_gateway.py` in the isolated gateway test environment.

Result: **5 passed**. The checks cover normal completion forwarding, SSE forwarding and cleanup, upstream readiness timeout/error handling, and Pod identity output. This verifies the gateway semantics without claiming Kubernetes scheduling or network behavior.

### K8S-2 — Kubernetes packaging and rollout definition

Command: `kubectl kustomize k8s`.

Result: **PASS**. The rendered objects contain one Namespace, ConfigMap, two-replica Deployment, and ClusterIP Service; the Deployment includes rolling-update settings, readiness/liveness probes, resource requests/limits, and the upstream URL configuration. PowerShell parser checks for the build, verify, and vLLM start scripts also passed.

### K8S-3 — Docker image/container smoke

Attempted to start `E:\Docker\DockerDesktop\Docker Desktop.exe`, selected context `desktop-linux`, reset WSL with `wsl --shutdown`, and retried the Engine. `wsl -l -v` still reports both `Ubuntu-24.04` and `docker-desktop` as `Stopped`; `docker info` and the Server half of `docker version` did not return within the bounded timeout. Therefore image build, gateway container startup, and host-to-container completion requests were **not executable on this workstation**.

### K8S-4 — Live Kubernetes acceptance

Result: **not run**. `kubectl config get-contexts` has no current context, so applying `k8s/` would not target a cluster. No Pod readiness, Service endpoint distribution, failure recovery, rollout, or rollback latency is claimed.

The blocked K8S-3/K8S-4 status is an environment result, not an application pass. On a machine with a healthy Docker Engine and Kubernetes context, run `scripts/Build-Gateway.ps1`, `kubectl apply -k k8s`, and `scripts/Verify-K8s.ps1`; then replace these two blocked rows with measured image, Pod, endpoint, recovery, and rollout data.

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
