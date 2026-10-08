"""Deterministic, equal-entry LRU/LFU/classic ARC replacement state machines.

ARC follows Megiddo and Modha, FAST 2003, Figure 4. Pinned entries may be
skipped during a worker step; this is explicitly counted, not hidden.
"""
from __future__ import annotations

from collections import OrderedDict
import sys


class ReplacementCache:
    def __init__(self, policy: str, capacity: int):
        if policy not in {"lru", "lfu", "arc"} or capacity < 1:
            raise ValueError("Expected lru/lfu/arc and a positive slot capacity")
        self.policy, self.capacity = policy, capacity
        self.t1: OrderedDict[str, None] = OrderedDict()
        self.t2: OrderedDict[str, None] = OrderedDict()
        self.b1: OrderedDict[str, None] = OrderedDict()
        self.b2: OrderedDict[str, None] = OrderedDict()
        self.freq: dict[str, int] = {}
        self.p = 0
        self.pin_skips = 0

    @property
    def resident(self) -> set[str]:
        return set(self.t1) | set(self.t2)

    def _pop(self, source, protected):
        for key in source:
            if key not in protected:
                del source[key]
                self.freq.pop(key, None)
                return key
            self.pin_skips += 1
        raise RuntimeError("All candidate entries are pinned; cannot reserve cache space")

    def _replace(self, incoming, protected):
        recent = bool(self.t1) and (len(self.t1) > self.p or
                                   (incoming in self.b2 and len(self.t1) == self.p))
        source, ghost = (self.t1, self.b1) if recent else (self.t2, self.b2)
        if not any(k not in protected for k in source):
            source, ghost = (self.t2, self.b2) if recent else (self.t1, self.b1)
        victim = self._pop(source, protected)
        ghost[victim] = None
        return victim

    def access(self, key: str, protected=frozenset()) -> tuple[bool, list[str]]:
        hit = key in self.resident
        evicted = []
        if self.policy != "arc":
            if hit:
                self.t1.move_to_end(key)
                self.freq[key] += 1
            else:
                if len(self.t1) == self.capacity:
                    if self.policy == "lfu":
                        candidates = [k for k in self.t1 if k not in protected]
                        if not candidates:
                            raise RuntimeError("All LFU entries pinned")
                        victim = min(candidates, key=lambda k: self.freq[k])
                        del self.t1[victim]
                        del self.freq[victim]
                    else:
                        victim = self._pop(self.t1, protected)
                    evicted.append(victim)
                self.t1[key] = None
                self.freq[key] = 1
        elif key in self.t1:
            del self.t1[key]
            self.t2[key] = None
        elif key in self.t2:
            self.t2.move_to_end(key)
        elif key in self.b1:
            self.p = min(self.capacity, self.p + max(1, len(self.b2) / len(self.b1)))
            evicted.append(self._replace(key, protected))
            del self.b1[key]
            self.t2[key] = None
        elif key in self.b2:
            self.p = max(0, self.p - max(1, len(self.b1) / len(self.b2)))
            evicted.append(self._replace(key, protected))
            del self.b2[key]
            self.t2[key] = None
        else:
            if len(self.t1) + len(self.b1) == self.capacity:
                if len(self.t1) < self.capacity:
                    self.b1.popitem(last=False)
                    evicted.append(self._replace(key, protected))
                else:
                    evicted.append(self._pop(self.t1, protected))
            elif len(self.t1) + len(self.b1) < self.capacity:
                total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
                if total >= self.capacity:
                    if total == 2 * self.capacity:
                        self.b2.popitem(last=False)
                    evicted.append(self._replace(key, protected))
            self.t1[key] = None
        self.validate()
        return hit, evicted

    def validate(self):
        lists = [set(x) for x in (self.t1, self.t2, self.b1, self.b2)]
        if sum(map(len, lists)) != len(set.union(*lists)):
            raise AssertionError("An entry appears in multiple ARC lists")
        if len(self.resident) > self.capacity or not 0 <= self.p <= self.capacity:
            raise AssertionError("Resident budget or ARC target exceeded")
        if self.policy == "arc":
            if len(self.t1) + len(self.b1) > self.capacity or sum(map(len, lists)) > 2 * self.capacity:
                raise AssertionError("ARC directory budget exceeded")

    def snapshot(self):
        return dict(policy=self.policy, slots=self.capacity, p=self.p,
                    t1=list(self.t1), t2=list(self.t2), b1=list(self.b1), b2=list(self.b2),
                    frequencies=dict(self.freq), pin_skips=self.pin_skips)

    def metadata_bytes(self):
        """CPython object sizes of replacement state, deduplicating shared keys."""
        seen = set()
        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if isinstance(value, dict):
                total += sum(size(k)+size(v) for k,v in value.items())
            return total
        return sys.getsizeof(self)+size(self.__dict__)
