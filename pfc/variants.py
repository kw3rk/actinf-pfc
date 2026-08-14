"""Thompson-sampling bandit over prompt-template variants.

Each slot (solver / verifier / skeptic / prep) holds a population of
wordings. The bandit picks one per call; the episode runner credits every
call against ground truth at episode end. The actinf controller decides
*which* action to take; this decides *which words* implement it.
"""
from __future__ import annotations

import json

import numpy as np


class VariantBandit:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.slots: dict[str, list[str]] = {}
        self.wins: dict[str, np.ndarray] = {}
        self.losses: dict[str, np.ndarray] = {}

    def register(self, slot: str, variants: list[str]):
        self.slots[slot] = variants
        self.wins[slot] = np.ones(len(variants))
        self.losses[slot] = np.ones(len(variants))

    def choose(self, slot: str) -> tuple[int, str]:
        w, l = self.wins[slot], self.losses[slot]
        idx = int(np.argmax(self.rng.beta(w, l)))
        return idx, self.slots[slot][idx]

    def update(self, slot: str, idx: int, success: bool):
        if success:
            self.wins[slot][idx] += 1
        else:
            self.losses[slot][idx] += 1

    def posterior_means(self, slot: str) -> np.ndarray:
        w, l = self.wins[slot], self.losses[slot]
        return w / (w + l)

    def report(self) -> str:
        lines = []
        for slot in self.slots:
            means = self.posterior_means(slot)
            n = self.wins[slot] + self.losses[slot] - 2
            cells = "  ".join(f"v{i}={m:.2f}(n={int(k)})"
                              for i, (m, k) in enumerate(zip(means, n)))
            lines.append(f"{slot:10s}: {cells}")
        return "\n".join(lines)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({s: {"wins": self.wins[s].tolist(),
                           "losses": self.losses[s].tolist()}
                       for s in self.slots}, f)

    def load(self, path: str):
        with open(path) as f:
            data = json.load(f)
        for s, d in data.items():
            if s in self.slots:
                self.wins[s] = np.array(d["wins"])
                self.losses[s] = np.array(d["losses"])
