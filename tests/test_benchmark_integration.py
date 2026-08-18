import argparse
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm_serving_lab.benchmark import run


class MockVllmHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        content_length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(content_length))
        prompt_tokens = len(payload["prompt"].split())
        completion_tokens = int(payload["max_tokens"])
        body = "\n".join(
            [
                'data: {"choices":[{"text":"token"}],"usage":null}',
                'data: {"choices":[{"text":" stream"}],"usage":null}',
                "data: "
                + json.dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                ),
                "data: [DONE]",
                "",
            ]
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_benchmark_writes_request_and_summary_artifact(tmp_path) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), MockVllmHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "run.json"
    args = argparse.Namespace(
        base_url=f"http://127.0.0.1:{server.server_port}",
        model="mock-model",
        workload="mixed",
        config_name="integration",
        concurrency=2,
        requests=6,
        warmup=2,
        output_tokens=4,
        seed=2026,
        timeout=10.0,
        gpu_sample_interval=60.0,
        api_key=None,
        server_max_num_seqs=2,
        server_image="mock-image",
        run_number=1,
        prefix_caching=False,
        ignore_eos=True,
        output=str(output),
    )

    try:
        artifact = asyncio.run(run(args))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert output.exists()
    assert artifact["summary"]["success_count"] == 6
    assert artifact["summary"]["total_output_tokens"] == 24
    assert len(artifact["requests"]) == 6

