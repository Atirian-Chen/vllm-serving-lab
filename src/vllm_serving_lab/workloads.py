from __future__ import annotations

import random
from dataclasses import dataclass


COMMON_WORDS = (
    "model service request cache memory token batch prompt output latency system "
    "data user compute engine scheduler block sequence context stream response "
    "server client method result analysis stable resource performance test"
).split()


@dataclass(frozen=True)
class WorkloadItem:
    request_id: str
    prompt: str
    target_prompt_words: int
    output_tokens: int


def _word_block(word_count: int, offset: int = 0) -> str:
    return " ".join(COMMON_WORDS[(offset + index) % len(COMMON_WORDS)] for index in range(word_count))


def build_mixed_workload(count: int, output_tokens: int, seed: int) -> list[WorkloadItem]:
    rng = random.Random(seed)
    target_sizes = [128, 512, 1024]
    items: list[WorkloadItem] = []

    for index in range(count):
        target_words = target_sizes[index % len(target_sizes)]
        offset = rng.randrange(len(COMMON_WORDS))
        unique_header = f"Request {index:04d} seed {seed}. "
        instruction = "Summarize the following technical context in plain language. Context: "
        prompt = unique_header + instruction + _word_block(target_words, offset)
        items.append(
            WorkloadItem(
                request_id=f"mixed-{index:04d}",
                prompt=prompt,
                target_prompt_words=target_words,
                output_tokens=output_tokens,
            )
        )

    rng.shuffle(items)
    return items


def build_shared_prefix_workload(count: int, output_tokens: int, seed: int) -> list[WorkloadItem]:
    rng = random.Random(seed)
    shared_words = 960
    common_prefix = (
        "Use the following shared service specification when answering every request. "
        + _word_block(shared_words)
        + "\nQuestion: "
    )
    items: list[WorkloadItem] = []

    for index in range(count):
        suffix_words = 32 + (index % 17)
        suffix = _word_block(suffix_words, rng.randrange(len(COMMON_WORDS)))
        prompt = f"{common_prefix}case {index:04d}: {suffix}"
        items.append(
            WorkloadItem(
                request_id=f"prefix-{index:04d}",
                prompt=prompt,
                target_prompt_words=shared_words + suffix_words,
                output_tokens=output_tokens,
            )
        )

    return items


def build_workload(kind: str, count: int, output_tokens: int, seed: int) -> list[WorkloadItem]:
    if count <= 0:
        raise ValueError("count must be positive")
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    if kind == "mixed":
        return build_mixed_workload(count, output_tokens, seed)
    if kind == "shared-prefix":
        return build_shared_prefix_workload(count, output_tokens, seed)
    raise ValueError(f"unsupported workload: {kind}")

