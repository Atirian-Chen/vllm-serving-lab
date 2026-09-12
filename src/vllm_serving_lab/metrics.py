from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor
from statistics import mean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RequestResult:
    request_id: str
    target_prompt_words: int
    ok: bool
    prompt_tokens: int | None
    output_tokens: int | None
    ttft_ms: float | None
    latency_ms: float | None
    tpot_ms: float | None
    http_status: int | None
    error: str | None
    session_id: str | None = None
    prefix_id: str | None = None
    expected_shared_tokens: int = 0
    reuse_distance: int | None = None
    idle_gap_ms: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GpuSample:
    memory_used_mib: float
    utilization_percent: float


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _round_optional(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def summarize_results(
    results: Iterable[RequestResult],
    wall_time_s: float,
    gpu_samples: Sequence[GpuSample] = (),
) -> dict[str, int | float | None]:
    result_list = list(results)
    successful = [result for result in result_list if result.ok]
    if wall_time_s <= 0:
        raise ValueError("wall_time_s must be positive")

    ttfts = [result.ttft_ms for result in successful if result.ttft_ms is not None]
    latencies = [result.latency_ms for result in successful if result.latency_ms is not None]
    tpots = [result.tpot_ms for result in successful if result.tpot_ms is not None]
    prompt_tokens = sum(result.prompt_tokens or 0 for result in successful)
    output_tokens = sum(result.output_tokens or 0 for result in successful)

    return {
        "request_count": len(result_list),
        "success_count": len(successful),
        "failure_count": len(result_list) - len(successful),
        "success_rate": round(len(successful) / len(result_list), 6) if result_list else 0.0,
        "wall_time_s": round(wall_time_s, 6),
        "request_throughput_per_s": round(len(successful) / wall_time_s, 4),
        "effective_input_tokens_per_s": round(prompt_tokens / wall_time_s, 4),
        "output_tokens_per_s": round(output_tokens / wall_time_s, 4),
        "total_prompt_tokens": prompt_tokens,
        "total_output_tokens": output_tokens,
        "ttft_p50_ms": _round_optional(percentile(ttfts, 0.50)),
        "ttft_p95_ms": _round_optional(percentile(ttfts, 0.95)),
        "latency_p50_ms": _round_optional(percentile(latencies, 0.50)),
        "latency_p95_ms": _round_optional(percentile(latencies, 0.95)),
        "tpot_p50_ms": _round_optional(percentile(tpots, 0.50)),
        "tpot_p95_ms": _round_optional(percentile(tpots, 0.95)),
        "gpu_memory_peak_mib": _round_optional(
            max((sample.memory_used_mib for sample in gpu_samples), default=None)
        ),
        "gpu_utilization_mean_percent": _round_optional(
            mean(sample.utilization_percent for sample in gpu_samples) if gpu_samples else None
        ),
        "gpu_sample_count": len(gpu_samples),
    }
