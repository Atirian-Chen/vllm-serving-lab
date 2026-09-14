from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from .metrics import RequestResult
from .workloads import WorkloadItem


@dataclass(frozen=True)
class ClientSettings:
    base_url: str
    model: str
    timeout_s: float = 600.0
    api_key: str | None = None
    ignore_eos: bool = True


class StreamingCompletionClient:
    def __init__(
        self,
        settings: ClientSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        max_connections: int = 32,
    ) -> None:
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_s, connect=30.0),
            limits=httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections),
            transport=transport,
        )

    async def __aenter__(self) -> "StreamingCompletionClient":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self._client.aclose()

    async def metrics_snapshot(self) -> str | None:
        """Return the raw Prometheus endpoint when the server exposes it."""
        try:
            response = await self._client.get("/metrics")
            response.raise_for_status()
            return response.text
        except Exception:
            return None

    async def generate(self, item: WorkloadItem, seed: int) -> RequestResult:
        payload = {
            "model": self._settings.model,
            "prompt": item.prompt,
            "max_tokens": item.output_tokens,
            "temperature": 0,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
            "ignore_eos": self._settings.ignore_eos,
        }
        if item.cache_salt is not None:
            payload["cache_salt"] = item.cache_salt

        started = time.perf_counter()
        first_token_at: float | None = None
        prompt_tokens: int | None = None
        output_tokens: int | None = None
        status_code: int | None = None

        try:
            async with self._client.stream("POST", "/v1/completions", json=payload) as response:
                status_code = response.status_code
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue

                    event = json.loads(data)
                    usage = event.get("usage")
                    if usage:
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)

                    for choice in event.get("choices", []):
                        if choice.get("text") and first_token_at is None:
                            first_token_at = time.perf_counter()

            finished = time.perf_counter()
            if first_token_at is None:
                raise RuntimeError("stream completed without a non-empty token event")
            if prompt_tokens is None or output_tokens is None:
                raise RuntimeError("stream completed without usage; the server must support include_usage")

            latency_ms = (finished - started) * 1000
            ttft_ms = (first_token_at - started) * 1000
            tpot_ms = (
                (latency_ms - ttft_ms) / (output_tokens - 1)
                if output_tokens > 1
                else None
            )
            return RequestResult(
                request_id=item.request_id,
                target_prompt_words=item.target_prompt_words,
                ok=True,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft_ms=round(ttft_ms, 6),
                latency_ms=round(latency_ms, 6),
                tpot_ms=round(tpot_ms, 6) if tpot_ms is not None else None,
                http_status=status_code,
                error=None,
                session_id=item.session_id, prefix_id=item.prefix_id,
                expected_shared_tokens=item.expected_shared_tokens,
                reuse_distance=item.reuse_distance, idle_gap_ms=item.idle_gap_ms,
                role=item.role, background_unique_prefixes=item.background_unique_prefixes,
                tenant_id=item.tenant_id, cache_salt=item.cache_salt,
            )
        except Exception as error:  # Request-level failures belong in the result artifact.
            return RequestResult(
                request_id=item.request_id,
                target_prompt_words=item.target_prompt_words,
                ok=False,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                ttft_ms=None,
                latency_ms=round((time.perf_counter() - started) * 1000, 6),
                tpot_ms=None,
                http_status=status_code,
                error=f"{type(error).__name__}: {error}",
                session_id=item.session_id, prefix_id=item.prefix_id,
                expected_shared_tokens=item.expected_shared_tokens,
                reuse_distance=item.reuse_distance, idle_gap_ms=item.idle_gap_ms,
                role=item.role, background_unique_prefixes=item.background_unique_prefixes,
                tenant_id=item.tenant_id, cache_salt=item.cache_salt,
            )
