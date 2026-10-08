import asyncio
import json

import httpx

from vllm_serving_lab.client import ClientSettings, StreamingCompletionClient
from vllm_serving_lab.workloads import WorkloadItem


def test_streaming_client_records_usage_and_ttft() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream_options"] == {"include_usage": True}
        body = "\n".join(
            [
                'data: {"choices":[{"text":"hello"}],"usage":null}',
                'data: {"choices":[{"text":" world"}],"usage":null}',
                'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    async def exercise() -> None:
        settings = ClientSettings("http://test", "test-model")
        transport = httpx.MockTransport(handler)
        item = WorkloadItem("request-1", "prompt", 10, 2)
        async with StreamingCompletionClient(settings, transport=transport) as client:
            result = await client.generate(item, seed=2026)

        assert result.ok
        assert result.prompt_tokens == 10
        assert result.output_tokens == 2
        assert result.ttft_ms is not None
        assert result.latency_ms is not None

    asyncio.run(exercise())


def test_streaming_client_keeps_request_failure_in_result() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    async def exercise() -> None:
        settings = ClientSettings("http://test", "test-model")
        transport = httpx.MockTransport(handler)
        item = WorkloadItem("request-1", "prompt", 10, 2)
        async with StreamingCompletionClient(settings, transport=transport) as client:
            result = await client.generate(item, seed=2026)

        assert not result.ok
        assert result.http_status == 500
        assert "HTTPStatusError" in (result.error or "")

    asyncio.run(exercise())


def test_exact_token_trace_rejects_missing_done_and_truncated_output():
    async def exercise():
        for done, output in ((False, 2), (True, 1)):
            async def handler(request):
                assert request.headers["X-Request-Id"] == "exact"
                body = ['data: {"choices":[{"text":"x"}]}',
                        'data: '+json.dumps({"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":output}})]
                if done: body.append('data: [DONE]')
                return httpx.Response(200,content='\n'.join(body))
            async with StreamingCompletionClient(ClientSettings('http://test','model'),
                        transport=httpx.MockTransport(handler)) as client:
                result = await client.generate(WorkloadItem('exact',[1,2,3],0,2),7)
            assert not result.ok
            assert 'contract' in result.error
    asyncio.run(exercise())
