"""Lossy / delayed / duplicating link model (seeded). With defaults it is a perfect link."""
from __future__ import annotations

import random

from .config import LinkCfg


class Channel:
    def __init__(self, name: str, link: LinkCfg, seed: int):
        self.name, self.link = name, link
        self.rng = random.Random(f"trafma-chan:{name}:{seed}")
        self._q: list = []       # (deliver_at, order, item, tag)
        self._n = 0

    def send(self, item, now: float) -> list:
        """Schedule delivery; returns outcome tags, e.g. ['queued'], ['lost'], ['queued','duplicated']."""
        l = self.link
        if l.loss_prob > 0 and self.rng.random() < l.loss_prob:
            return ["lost"]
        outcomes = ["queued"]
        copies = 1
        if l.duplicate_prob > 0 and self.rng.random() < l.duplicate_prob:
            copies = 2
            outcomes.append("duplicated")
        for c in range(copies):
            self._n += 1
            self._q.append((now + l.delay_s, self._n, item, "dup" if c else "orig"))
        return outcomes

    def deliver_due(self, now: float) -> list:
        due = [x for x in self._q if x[0] <= now + 1e-9]
        self._q = [x for x in self._q if x[0] > now + 1e-9]
        due.sort(key=lambda x: (x[0], x[1]))
        return [(x[2], x[3]) for x in due]

    def in_flight(self) -> int:
        return len(self._q)
