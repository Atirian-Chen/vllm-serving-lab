import random
import pytest
from vllm_serving_lab.replacement import ReplacementCache


def test_lru_and_lfu_choose_different_victims():
    lru, lfu = (ReplacementCache(p, 2) for p in ("lru", "lfu"))
    for key in "aaaab":
        lru.access(key)
        lfu.access(key)
    assert lru.access("c")[1] == ["a"]
    assert lfu.access("c")[1] == ["b"]
    assert lfu.access("b")[1] == ["c"]  # Evicted frequency starts at one.


def test_arc_ghost_feedback_changes_target_and_preserves_frequent_entry():
    cache = ReplacementCache("arc", 2)
    for key in "aab":
        cache.access(key)
    assert cache.access("c")[1] == ["b"]
    assert "b" in cache.b1 and "a" in cache.t2
    cache.access("b")
    assert cache.p == 1 and "a" in cache.b2
    cache.access("a")
    assert cache.p == 0


@pytest.mark.parametrize("policy", ["lru", "lfu", "arc"])
@pytest.mark.parametrize("size", [1, 4, 9, 36])
def test_long_mixed_trace_respects_directory_and_payload_budgets(policy, size):
    cache = ReplacementCache(policy, size)
    rng = random.Random(712)
    for i in range(4000):
        key = str(rng.randrange(64)) if i % 7 else f"cold-{i}"
        before = cache.resident
        hit, victims = cache.access(key)
        assert hit == (key in before)
        assert key in cache.resident
        assert set(victims) <= before
        cache.validate()


@pytest.mark.parametrize("policy", ["lru", "lfu", "arc"])
def test_inflight_entry_is_not_evicted(policy):
    cache = ReplacementCache(policy, 3)
    for key in "abc":
        cache.access(key)
    assert "a" not in cache.access("d", {"a"})[1]
    assert "a" in cache.resident
