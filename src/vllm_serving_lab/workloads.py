from __future__ import annotations

import math
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
    session_id: str | None = None
    prefix_id: str | None = None
    expected_shared_tokens: int = 0
    reuse_distance: int | None = None
    idle_gap_ms: int = 0
    role: str = "foreground"
    background_unique_prefixes: int = 0
    tenant_id: str | None = None
    cache_salt: str | None = None


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
                session_id=f"session-{index % 8:02d}",
                prefix_id="shared-spec",
                expected_shared_tokens=shared_words,
            )
        )

    return items


def build_coding_agent_workload(
    count: int, output_tokens: int, seed: int, tenants: int = 8
) -> list[WorkloadItem]:
    """Model a multi-tenant coding/agent service with hot and cold prefixes.

    Each tenant owns a stable system/tool/repository prefix. Requests are emitted
    in Zipf-like order so reuse distance and cache pressure are measurable.
    """
    if tenants <= 0:
        raise ValueError("tenants must be positive")
    rng = random.Random(seed)
    prefixes = []
    prefix_words = 512
    for tenant in range(tenants):
        prefix = (
            f"Tenant {tenant:03d} coding agent system and tool contract. "
            "Repository summary and safe execution policy. "
            + _word_block(prefix_words, tenant)
        )
        prefixes.append(prefix)
    items: list[WorkloadItem] = []
    last_seen: dict[int, int] = {}
    for index in range(count):
        rank = min(tenants - 1, int((rng.random() ** 2) * tenants))
        suffix = (
            f"\nTask {index:04d}: inspect changed files, run tests, and propose a patch. "
            + _word_block(48 + index % 16, rng.randrange(len(COMMON_WORDS)))
        )
        reuse_distance = None if rank not in last_seen else index - last_seen[rank]
        last_seen[rank] = index
        items.append(WorkloadItem(
            request_id=f"agent-{index:04d}", prompt=prefixes[rank] + suffix,
            target_prompt_words=prefix_words + 60 + index % 16,
            output_tokens=output_tokens, session_id=f"tenant-{rank:03d}",
            prefix_id=f"tenant-prefix-{rank:03d}", expected_shared_tokens=prefix_words,
            reuse_distance=reuse_distance,
        ))
    return items


def build_tenant_shared_prefix_workload(
    count: int,
    output_tokens: int,
    seed: int,
    tenants: int = 4,
    rounds: int = 8,
    background_unique_prefixes: int = 2,
    tenant_namespace: bool = False,
) -> list[WorkloadItem]:
    """Build a shared-pool multi-tenant trace with optional cache isolation.

    All tenants use the same public policy/tool prefix, while their private
    suffix and append-only history remain distinct.  This makes a global cache
    able to share the public prefix, whereas ``tenant_namespace`` sends a
    per-tenant cache salt without changing the prompt bytes.
    """
    if tenants <= 0 or rounds <= 0:
        raise ValueError("tenants and rounds must be positive")
    if background_unique_prefixes < 0:
        raise ValueError("background_unique_prefixes must be nonnegative")

    rng = random.Random(seed)
    total_foreground = min(
        tenants * rounds,
        max(1, math.ceil(count / (background_unique_prefixes + 1))),
    )
    public_prefix = (
        "Shared customer support policy, tool contract, and retrieval rules. "
        + _word_block(512)
    )
    histories: dict[int, str] = {tenant: "" for tenant in range(tenants)}
    items: list[WorkloadItem] = []
    index = 0

    for round_index in range(rounds):
        for tenant in range(tenants):
            if index >= total_foreground:
                break
            for background in range(background_unique_prefixes):
                bg_id = f"tenant-bg-{seed}-{round_index:02d}-{tenant:02d}-{background:03d}"
                bg_prompt = (
                    f"One-shot tenant request {bg_id}. "
                    + _word_block(512 + background % 32,
                                  rng.randrange(len(COMMON_WORDS)))
                    + "\nQuestion: summarize the request."
                )
                items.append(WorkloadItem(
                    request_id=bg_id,
                    prompt=bg_prompt,
                    target_prompt_words=len(bg_prompt.split()),
                    output_tokens=output_tokens,
                    session_id=bg_id,
                    prefix_id=bg_id,
                    role="background",
                    background_unique_prefixes=background_unique_prefixes,
                    tenant_id=f"tenant-{tenant:02d}",
                    cache_salt=(bg_id if tenant_namespace else None),
                ))

            tenant_id = f"tenant-{tenant:02d}"
            private_prefix = (
                f"\nPrivate workspace for {tenant_id}. "
                "Do not share tenant-specific records."
            )
            suffix = (
                f"\nTurn {round_index}: diagnose the issue and propose the next action. "
                + _word_block(32, rng.randrange(len(COMMON_WORDS)))
            )
            prompt_prefix = public_prefix + private_prefix + histories[tenant]
            prompt = prompt_prefix + suffix
            items.append(WorkloadItem(
                request_id=f"tenant-{tenant:02d}-round-{round_index:02d}",
                prompt=prompt,
                target_prompt_words=len(prompt.split()),
                output_tokens=output_tokens,
                session_id=tenant_id,
                prefix_id="public-support-policy",
                expected_shared_tokens=len(public_prefix.split()),
                reuse_distance=(None if round_index == 0
                                 else background_unique_prefixes * tenants + tenants),
                role="foreground",
                background_unique_prefixes=background_unique_prefixes,
                tenant_id=tenant_id,
                cache_salt=(tenant_id if tenant_namespace else None),
            ))
            histories[tenant] += suffix + "\nAssistant: " + _word_block(16, tenant)
            index += 1
        if index >= total_foreground:
            break

    while len(items) < count:
        bg_id = f"tenant-bg-tail-{len(items):04d}"
        prompt = f"One-shot cold request {bg_id}. " + _word_block(256, rng.randrange(len(COMMON_WORDS)))
        items.append(WorkloadItem(
            request_id=bg_id,
            prompt=prompt,
            target_prompt_words=len(prompt.split()),
            output_tokens=output_tokens,
            session_id=bg_id,
            prefix_id=bg_id,
            role="background",
            background_unique_prefixes=background_unique_prefixes,
            tenant_id="cold",
            cache_salt=(bg_id if tenant_namespace else None),
        ))
    return items[:count]


def build_session_chat_workload(
    count: int,
    output_tokens: int,
    seed: int,
    sessions: int = 8,
    rounds: int = 4,
    idle_gap_ms: int = 0,
    background_unique_prefixes: int = 0,
) -> list[WorkloadItem]:
    """Build a multi-turn support-chat trace with controllable cache pressure.

    Each session keeps a stable system prompt and growing conversation history.
    Optional unique background prompts approximate unrelated tenant traffic that
    can evict otherwise idle prefixes from a finite GPU pool.
    """
    if sessions <= 0 or rounds <= 0:
        raise ValueError("sessions and rounds must be positive")
    if idle_gap_ms < 0 or background_unique_prefixes < 0:
        raise ValueError("idle_gap_ms and background_unique_prefixes must be nonnegative")
    rng = random.Random(seed)
    total_foreground = min(sessions * rounds, max(1, math.ceil(count / (background_unique_prefixes + 1))))
    items: list[WorkloadItem] = []
    histories: dict[int, str] = {session: "" for session in range(sessions)}
    last_seen: dict[int, int] = {}
    index = 0
    for round_index in range(rounds):
        for session in range(sessions):
            if index >= total_foreground:
                break
            for background in range(background_unique_prefixes):
                bg_id = f"chat-bg-{seed}-{round_index:02d}-{session:02d}-{background:03d}"
                bg_prompt = (
                    f"Unrelated tenant {bg_id} system policy. "
                    + _word_block(512 + background % 32, rng.randrange(len(COMMON_WORDS)))
                    + "\nQuestion: summarize this unrelated request."
                )
                items.append(WorkloadItem(
                    request_id=bg_id, prompt=bg_prompt,
                    target_prompt_words=len(bg_prompt.split()), output_tokens=output_tokens,
                    session_id=bg_id, prefix_id=bg_id, expected_shared_tokens=0,
                    role="background", background_unique_prefixes=background_unique_prefixes,
                ))
            prefix = (
                f"Support session {seed}-{session:03d}. Follow the support policy and tools. "
                + _word_block(512, session)
                + histories[session]
            )
            suffix = f"\nUser turn {round_index}: diagnose the issue and propose the next action. " + _word_block(
                32, rng.randrange(len(COMMON_WORDS))
            )
            reuse_distance = None if session not in last_seen else len(items) - last_seen[session]
            last_seen[session] = len(items)
            items.append(WorkloadItem(
                request_id=f"chat-{session:02d}-round-{round_index:02d}",
                prompt=prefix + suffix, target_prompt_words=len((prefix + suffix).split()),
                output_tokens=output_tokens, session_id=f"session-{session:02d}",
                prefix_id=f"session-prefix-{session:02d}", expected_shared_tokens=len(prefix.split()),
                # The first turn is a cold start; apply the idle gap only when
                # returning to a session whose prefix has been seen before.
                reuse_distance=reuse_distance,
                idle_gap_ms=idle_gap_ms if reuse_distance is not None else 0,
                role="foreground", background_unique_prefixes=background_unique_prefixes,
            ))
            # Replay fixed prior turns so token-identical history survives reuse.
            histories[session] += suffix + "\nAssistant: " + _word_block(16, session)
            index += 1
        if index >= total_foreground:
            break
    # Keep the public contract that ``count`` means total requests. If the
    # requested count exceeds the planned foreground trace, fill with unique
    # background requests rather than silently returning fewer records.
    while len(items) < count:
        background = len(items)
        bg_id = f"chat-bg-tail-{background:04d}"
        items.append(WorkloadItem(
            request_id=bg_id,
            prompt=f"Unrelated tenant {bg_id} system policy. " + _word_block(256, rng.randrange(len(COMMON_WORDS))),
            target_prompt_words=256, output_tokens=output_tokens,
            session_id=bg_id, prefix_id=bg_id, role="background",
            background_unique_prefixes=background_unique_prefixes,
        ))
    return items[:count]


def build_workload(kind: str, count: int, output_tokens: int, seed: int, **options: object) -> list[WorkloadItem]:
    if count <= 0:
        raise ValueError("count must be positive")
    if output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    if kind == "mixed":
        return build_mixed_workload(count, output_tokens, seed)
    if kind == "shared-prefix":
        return build_shared_prefix_workload(count, output_tokens, seed)
    if kind == "coding-agent":
        return build_coding_agent_workload(count, output_tokens, seed)
    if kind == "session-chat":
        return build_session_chat_workload(
            count, output_tokens, seed,
            sessions=int(options.get("sessions", 8)),
            rounds=int(options.get("rounds", 4)),
            idle_gap_ms=int(options.get("idle_gap_ms", 0)),
            background_unique_prefixes=int(options.get("background_unique_prefixes", 0)),
        )
    if kind == "tenant-shared":
        return build_tenant_shared_prefix_workload(
            count,
            output_tokens,
            seed,
            tenants=int(options.get("tenants", 4)),
            rounds=int(options.get("rounds", 8)),
            background_unique_prefixes=int(options.get("background_unique_prefixes", 2)),
            tenant_namespace=bool(options.get("tenant_namespace", False)),
        )
    raise ValueError(f"unsupported workload: {kind}")
