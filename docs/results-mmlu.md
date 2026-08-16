# MMLU-Pro humanities transfer arc (2026-08-14/15)

Far-transfer stress test: GSM-calibrated controller on MMLU-Pro non-STEM
subjects (law, psychology, philosophy, history, health, economics, business),
n=500, Qwen3.6-35B-A3B, λ=0.0002 throughout.

| run | acc | tok/ep | ledger (fixed/broke) |
|---|---|---|---|
| baseline (single shot) | 0.776 | 1277 | — |
| frozen, absolute threshold | 0.802 | 3634 | 25 / 15 |
| frozen, adaptive threshold | 0.778 | 2488 | 11 / 15 |
| learning, absolute threshold | 0.782 | 2734 | 11 / 5 |
| learning, adaptive threshold | 0.794 | 2865 | 7 / 5 |

n=500 ⇒ ±1.8 pts SE per cell; treat accuracy deltas within ~2.5 pts as noise.
The robust findings are mechanistic:

1. **Saturation**: the GSM-calibrated entropy threshold reads 91% of
   humanities generations as turbulent (vs ~46% on GSM) — the sensor's
   operating point is domain-specific even though the signal's meaning
   transfers.
2. **Fragility selection**: on judgment domains, high local entropy predicts
   *debatable*, which predicts both being wrong and caving under pressure
   when right. Entropy-targeted skeptic pressure therefore concentrates fire
   where pressure is most dangerous (break-rate per press ~3× the blanket
   rate). Seen independently in two runs.
3. **Price discovery**: blanket pressure buys +2.6 pts at ~90k tokens per
   marginal correct answer (GSM: ~900/pt). Both learning runs measured this
   in-flight and mostly declined to pay at the declared λ — the
   accuracy-maximal and EV-maximal policies diverge, and the learner follows
   the declared economics. Math errors are variance (harvestable);
   knowledge errors are gaps (near-unbuyable by re-asking).
4. **Sensor × learner complementarity (directional)**: only the
   adaptive+learning cell retained a small (~12%) positive-ledger pressure
   set instead of abandoning intervention; best second-half like-for-like.
   Within noise at this n — trajectory evidence, not a significance claim.
