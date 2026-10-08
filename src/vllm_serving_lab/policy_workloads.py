"""Exact-token, paired traces for the CPU cache replacement study."""
import hashlib
import json
import random
from .workloads import WorkloadItem


def build_trace(scenario, length, count, output_tokens, seed, alphabet):
    if scenario not in {"stable", "scan", "shift", "unique"}:
        raise ValueError("Unknown policy scenario")
    if length not in {640, 1664} or count < 1:
        raise ValueError("Expected a declared request length and positive count")
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** 1.1 for i in range(32)]
    items, previous = [], {}
    body = alphabet["body"]
    chars = alphabet["hex"]
    for index in range(count):
        phase = min(2, 3 * index // count)
        role = "hot"
        if scenario == "unique":
            key, role = f"cold-{index}", "cold"
        elif scenario == "scan":
            step = index % 320
            if 128 <= step < 192:
                key, role = f"scan-{index}", "scan"
            else:
                key = f"hot-{(step if step < 128 else step - 192) % 32}"
        elif rng.random() < .8:
            hot = rng.choices(range(32), weights=weights)[0]
            key = f"hot-{phase if scenario == 'shift' else 0}-{hot}"
        else:
            key, role = f"cold-{index}", "cold"
        identity = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
        header = [chars[c] for c in identity[:16]]
        prefix = header + [body[i % len(body)] for i in range(496)]
        suffix_hash = hashlib.sha256(f"{seed}:{index}:suffix".encode()).hexdigest()
        suffix = [chars[c] for c in suffix_hash[:16]]
        suffix += [body[(i + index) % len(body)] for i in range(length - 528)]
        items.append(WorkloadItem(request_id=f"policy-{index}", prompt=prefix + suffix,
                     target_prompt_words=0, output_tokens=output_tokens, prefix_id=key,
                     expected_shared_tokens=512,
                     reuse_distance=index-previous[key] if key in previous else None,
                     session_id=str(phase), role=role))
        previous[key] = index
    return items


def trace_hash(items):
    return hashlib.sha256(json.dumps([i.prompt for i in items], separators=(",", ":")).encode()).hexdigest()
