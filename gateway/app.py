"""Small Kubernetes-friendly proxy in front of an OpenAI-compatible vLLM server."""

from __future__ import annotations

import os
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse


NOT_READY_FLAG = Path("/tmp/not-ready")
NOT_LIVE_FLAG = Path("/tmp/not-live")
_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _flag_exists(path: Path) -> bool:
    return path.exists()


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _response_headers(response: httpx.Response) -> dict[str, str]:
    """Keep application headers while letting Starlette manage framing."""
    excluded = _HOP_BY_HOP_HEADERS | {"content-length"}
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in excluded
    }


def _request_headers(request: Request) -> dict[str, str]:
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length", "connection"}
    }


def create_app(
    *,
    client: httpx.AsyncClient | None = None,
    base_url: str | None = None,
    version: str | None = None,
    readiness_timeout: float | None = None,
) -> FastAPI:
    """Create the gateway application.

    ``client`` is injectable so tests can use ``httpx.MockTransport``. In normal
    operation the single client is shared by all requests and closed by the
    application lifespan.
    """
    upstream_url = (base_url or os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
    app_version = version or os.getenv("APP_VERSION", "dev")
    ready_timeout = (
        readiness_timeout
        if readiness_timeout is not None
        else _float_env("READINESS_TIMEOUT_SECONDS", 2.0)
    )

    owns_client = client is None
    shared_client = client or httpx.AsyncClient(
        base_url=upstream_url,
        timeout=httpx.Timeout(_float_env("UPSTREAM_TIMEOUT_SECONDS", 120.0), connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        yield
        if owns_client:
            await shared_client.aclose()

    app = FastAPI(title="vLLM Kubernetes gateway", version=app_version, lifespan=lifespan)
    # Set state eagerly so ASGI tests that do not run lifespan still work.
    app.state.http_client = shared_client
    app.state.upstream_url = upstream_url
    app.state.version = app_version
    app.state.readiness_timeout = ready_timeout

    @app.get("/health/live")
    async def health_live() -> Response:
        if _flag_exists(NOT_LIVE_FLAG):
            return JSONResponse({"status": "not_live"}, status_code=500)
        return JSONResponse({"status": "ok"})

    @app.get("/health/ready")
    async def health_ready() -> Response:
        if _flag_exists(NOT_READY_FLAG):
            return JSONResponse({"status": "not_ready", "reason": "test flag"}, status_code=503)

        try:
            upstream = await app.state.http_client.get(
                "/health",
                timeout=httpx.Timeout(app.state.readiness_timeout),
            )
        except httpx.TimeoutException:
            return JSONResponse({"status": "not_ready", "reason": "upstream timeout"}, status_code=503)
        except httpx.RequestError as error:
            return JSONResponse(
                {"status": "not_ready", "reason": f"upstream unavailable: {type(error).__name__}"},
                status_code=503,
            )

        if upstream.is_success:
            return JSONResponse({"status": "ready"})
        return JSONResponse(
            {"status": "not_ready", "reason": f"upstream status {upstream.status_code}"},
            status_code=503,
        )

    @app.get("/debug/info")
    async def debug_info() -> JSONResponse:
        return JSONResponse(
            {
                "pod_name": os.getenv("HOSTNAME", socket.gethostname()),
                "version": app.state.version,
            }
        )

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        body = await request.body()
        try:
            payload = await request.json()
        except ValueError:
            payload = {}
        streaming = payload.get("stream") is True if isinstance(payload, dict) else False
        headers = _request_headers(request)

        if not streaming:
            try:
                upstream = await app.state.http_client.post(
                    "/v1/completions",
                    content=body,
                    headers=headers,
                )
            except httpx.RequestError as error:
                return JSONResponse(
                    {"error": {"message": f"upstream unavailable: {error}"}},
                    status_code=502,
                )
            return Response(
                content=upstream.content,
                status_code=upstream.status_code,
                headers=_response_headers(upstream),
            )

        stream_context = app.state.http_client.stream(
            "POST",
            "/v1/completions",
            content=body,
            headers=headers,
        )
        try:
            upstream = await stream_context.__aenter__()
        except httpx.RequestError as error:
            return JSONResponse(
                {"error": {"message": f"upstream unavailable: {error}"}},
                status_code=502,
            )

        async def iterator() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await stream_context.__aexit__(None, None, None)

        return StreamingResponse(
            iterator(),
            status_code=upstream.status_code,
            headers=_response_headers(upstream),
        )

    return app


app = create_app()
