import asyncio
import json

import httpx

import gateway.app as gateway_app
from gateway.app import create_app


class ChunkedStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


def run_request(app, method: str, path: str, **kwargs):
    async def exercise():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(exercise())


def test_proxy_forwards_non_stream_completion():
    seen: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, headers={"content-type": "application/json"}, json={"id": "req-1"})

    upstream = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://vllm",
    )
    app = create_app(client=upstream, base_url="http://vllm", version="test")
    response = run_request(
        app,
        "POST",
        "/v1/completions",
        headers={"authorization": "Bearer test"},
        json={"model": "demo", "prompt": "hello", "stream": False},
    )

    assert response.status_code == 200
    assert response.json() == {"id": "req-1"}
    assert seen == {
        "path": "/v1/completions",
        "body": {"model": "demo", "prompt": "hello", "stream": False},
    }
    asyncio.run(upstream.aclose())


def test_proxy_streams_sse_bytes_and_closes_upstream():
    stream = ChunkedStream([b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm")
    app = create_app(client=upstream, base_url="http://vllm")
    response = run_request(
        app,
        "POST",
        "/v1/completions",
        json={"model": "demo", "prompt": "hello", "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"
    assert stream.closed
    asyncio.run(upstream.aclose())


def test_health_live_does_not_call_upstream(monkeypatch):
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm")
    app = create_app(client=upstream, base_url="http://vllm")
    monkeypatch.setattr(gateway_app, "_flag_exists", lambda path: True)

    response = run_request(app, "GET", "/health/live")

    assert response.status_code == 500
    assert response.json()["status"] == "not_live"
    assert calls == 0
    asyncio.run(upstream.aclose())


def test_health_ready_checks_upstream_and_supports_not_ready_flag(monkeypatch):
    calls = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200)

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://vllm")
    app = create_app(client=upstream, base_url="http://vllm", readiness_timeout=0.1)

    monkeypatch.setattr(gateway_app, "_flag_exists", lambda path: False)
    response = run_request(app, "GET", "/health/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    assert calls == ["/health"]

    monkeypatch.setattr(gateway_app, "_flag_exists", lambda path: True)
    response = run_request(app, "GET", "/health/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert calls == ["/health"]
    asyncio.run(upstream.aclose())


def test_debug_info_returns_pod_and_version(monkeypatch):
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)), base_url="http://vllm")
    app = create_app(client=upstream, base_url="http://vllm", version="v2")
    monkeypatch.setenv("HOSTNAME", "gateway-abc")

    response = run_request(app, "GET", "/debug/info")

    assert response.status_code == 200
    assert response.json() == {"pod_name": "gateway-abc", "version": "v2"}
    asyncio.run(upstream.aclose())
