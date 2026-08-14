"""Synthetic multi-step arithmetic word problems with exact integer answers.

Difficulty is the number of chained operations. The observable cue (problem
length) correlates with difficulty but noisily — the controller has to learn
how much to trust it.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .engine import D_EASY, D_HARD, CUE_SHORT, CUE_LONG

NOUNS = ["crates", "widgets", "batteries", "sensors", "gears", "tiles",
         "capacitors", "spools", "brackets", "valves"]
PLACES = ["warehouse", "depot", "workshop", "factory", "lab", "storeroom"]


@dataclass
class Task:
    tid: int
    text: str
    answer: int
    difficulty: int      # D_EASY / D_HARD
    cue: int             # CUE_SHORT / CUE_LONG


def _gen_ops(rng: np.random.Generator, n_steps: int, lo: int, hi: int):
    """Chain of exact-arithmetic ops starting from a base value."""
    val = int(rng.integers(lo, hi))
    steps = [f"A {rng.choice(PLACES)} starts with {val} {rng.choice(NOUNS)}."]
    for _ in range(n_steps):
        op = rng.choice(["add", "sub", "mul", "div"])
        if op == "add":
            k = int(rng.integers(lo, hi))
            val += k
            steps.append(f"Then {k} more arrive.")
        elif op == "sub":
            k = int(rng.integers(1, max(2, val)))
            val -= k
            steps.append(f"Then {k} are shipped out.")
        elif op == "mul":
            k = int(rng.integers(2, 5))
            val *= k
            steps.append(f"Then the stock is multiplied {k}-fold by a new delivery.")
        else:
            divisors = [d for d in range(2, 7) if val % d == 0 and val // d > 0]
            if not divisors:
                k = int(rng.integers(lo, hi))
                val += k
                steps.append(f"Then {k} more arrive.")
            else:
                k = int(rng.choice(divisors))
                val //= k
                steps.append(f"Then the stock is split evenly into {k} groups; keep one group.")
    steps.append("How many items remain?")
    return " ".join(steps), val


def make_task(tid: int, rng: np.random.Generator, p_hard: float = 0.5,
              cue_noise: float = 0.15) -> Task:
    hard = rng.random() < p_hard
    if hard:
        text, ans = _gen_ops(rng, n_steps=int(rng.integers(5, 8)), lo=17, hi=97)
        diff = D_HARD
    else:
        text, ans = _gen_ops(rng, n_steps=int(rng.integers(1, 3)), lo=3, hi=20)
        diff = D_EASY
    true_cue = CUE_LONG if diff == D_HARD else CUE_SHORT
    cue = true_cue if rng.random() > cue_noise else 1 - true_cue
    return Task(tid=tid, text=text, answer=ans, difficulty=diff, cue=cue)
