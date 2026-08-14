"""Self-normalizing observation channels (domain adaptation, layer 1).

Absolute signal thresholds are model+domain artifacts: the entropy threshold
calibrated on GSM arithmetic saturated on humanities multiple choice (91% of
solves read "turbulent"). What transfers is the signal's *meaning relative to
its local baseline* — so track a running window of recent raw values and
bucket by percentile. The sensor re-centers itself when the task distribution
drifts; "domain" is never represented explicitly.

Cold start falls back to the absolute threshold until the window has enough
mass to trust.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class RunningNorm:
    def __init__(self, hi_quantile: float = 0.67, window: int = 200,
                 min_samples: int = 30, absolute_fallback: float | None = None):
        self.q = hi_quantile
        self.window: deque = deque(maxlen=window)
        self.min_samples = min_samples
        self.fallback = absolute_fallback

    def is_high(self, value: float) -> bool:
        """Bucket `value` against the current baseline, THEN absorb it.
        (Absorbing first would let each value dilute its own judgment.)"""
        if len(self.window) < self.min_samples:
            high = (value >= self.fallback) if self.fallback is not None \
                else False
        else:
            high = value >= float(np.quantile(self.window, self.q))
        self.window.append(value)
        return high

    def stats(self) -> str:
        if not self.window:
            return "empty"
        a = np.array(self.window)
        return (f"n={len(a)} median={np.median(a):.3f} "
                f"q{int(self.q*100)}={np.quantile(a, self.q):.3f}")
