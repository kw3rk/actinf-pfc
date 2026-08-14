import numpy as np

from pfc.normalize import RunningNorm


def test_cold_start_uses_absolute_fallback():
    n = RunningNorm(absolute_fallback=0.13, min_samples=30)
    assert n.is_high(0.20) is True
    assert n.is_high(0.05) is False


def test_recentering_after_domain_shift():
    rng = np.random.default_rng(0)
    n = RunningNorm(absolute_fallback=0.13, min_samples=30, window=100)
    # domain A: low-entropy regime (math-like, median ~0.08)
    for _ in range(150):
        n.is_high(float(rng.normal(0.08, 0.03)))
    # domain B: uniformly hotter regime (MC-like, median ~0.30)
    flags = [n.is_high(float(rng.normal(0.30, 0.05))) for _ in range(200)]
    # early in B: everything reads high vs A's baseline (saturation)
    assert sum(flags[:20]) >= 15
    # after re-centering: only ~the top tercile of B's own distribution
    late = flags[-100:]
    assert 0.15 < sum(late) / len(late) < 0.55


def test_within_domain_flags_track_quantile():
    rng = np.random.default_rng(1)
    n = RunningNorm(hi_quantile=0.67, min_samples=30, window=200)
    vals = rng.normal(0.1, 0.04, size=600)
    flags = [n.is_high(float(v)) for v in vals]
    rate = sum(flags[100:]) / len(flags[100:])
    assert 0.25 < rate < 0.42          # ~top third flagged
