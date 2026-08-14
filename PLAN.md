# Roadmap

Three stages, ordered by cost. Each stage's claim is measurable in the
existing harness before it graduates.

## Stage 1 — Template-variant bandit (prompt populations)

**Claim**: prompt wording is a measurable quantity, not a craft exercise.
The adversarial→neutral verifier rewrite (found by reading the A-matrix)
moved P(ok|correct,easy) from 0.74 to 0.94 — done by hand once; automate it.

**Mechanism**: each prompt slot (solver / verifier / skeptic / prep) holds a
*population* of variants. A Thompson-sampling bandit (Beta per variant)
picks the wording each call; at episode end every call is credited against
ground truth (solve-like: candidate graded correct; verify: verdict matched
truth). Two-level structure: the actinf controller chooses the abstract
action, the bandit chooses the concrete words. Losers retire by measurement.

**Files**: `pfc/variants.py` (bandit), wiring in `pfc/llm.py`,
`pfc/mock_llm.py`, `pfc/episodes.py`.
**Validation**: mock agent with hidden per-variant efficacy deltas —
the bandit must converge to the planted best variant.

## Stage 2 — Preparation stage with certainty gate (pipeline actions)

**Claim**: a cheap typed intermediate product (extract givens/constraints/
question), certainty-gated before the expensive call, beats raw solving on
hard tasks — and factors doubt (misread-the-question vs botched-the-math).

**Mechanism v1**: composite action `solve_prepped` — prep call (cheap,
entropy-measured; one re-prep if turbulent) then solve with the extraction
embedded. To the controller it is just another action column: its transition
counts answer "does preparation pay, per difficulty" empirically, and its
learned token cost prices the pipeline honestly.

**v2 (after v1 measures positive)**: `prepared` becomes a controller-known
state dimension so *all* downstream actions condition on it, and extraction
entropy becomes a cheap difficulty probe at episode start.

**Validation**: mock with a planted prep-boost on hard tasks; then GSM/MATH
A/B against Stage-0 counts.

## Stage 3 — Node-graph controller (the v2 system)

**Claim**: generalize the linear episode to a graph. LLM outputs are nodes;
prompt operators are edges; each node carries its own correctness posterior
(entropy at creation, agreement with siblings, verdicts against it); the
controller runs best-first search with expected free energy as the priority,
at a declared token price. Tree-of-Thoughts with the ingredient that lineage
lacks: a *calibrated* value function (measured: LLM self-evaluation AUC
0.734, logit entropy 0.876).

Properties: backtracking from any node; per-operator transition counts as
the persistent "what works" model (small), per-episode graph transient
(grows, discarded); termination when no edge has positive EFE — inquiry
ends when nothing more is worth asking.

**Design**: `docs/graph-controller.md`. Skeleton: `pfc/graph.py`.
**Validation**: mock first; must beat the linear controller's
frontier on tasks where backtracking matters (multi-step with poisoned
intermediate states).

## Standing constraints

- Every new observation channel or action costs calibration cells;
  additions must earn their counts (the framework audits its own features).
- The λ price stays the single declared knob; no hidden budgets.
- The mock harness validates machinery before any GPU tokens are spent.

## Known issues

- **Cold-start under-exploration (Stage 2, observed in mock)**: a new action
  whose true value is masked by an already-good policy can get ~20 trials of
  bad luck, a pessimistic count cell, and premature abandonment — which would
  then freeze into transferred models. Planned fix: an exploration floor
  (minimum trial count per (action, difficulty) cell before its estimate is
  eligible to be frozen), or optimistic initialization for newly added actions.

## Domain adaptation (added after the MMLU saturation finding)

Domains are never represented explicitly; what shifts across them is the
baseline statistics of the channels.

- **Layer 1 (built)**: self-normalizing sensors — `pfc/normalize.py`, opt-in
  via `run_llm.py --adaptive-entropy`. Entropy bucketed against a running
  local quantile window (absolute threshold as cold-start fallback); the
  sensor re-centers within ~30 episodes of a domain shift. Found because the
  GSM-calibrated absolute threshold saturated on MMLU humanities (91% of
  solves read turbulent → blanket rescue at 2.5x tokens for the same +7).
- **Layer 2 (planned)**: structural context read off the task, not inferred —
  condition agreement likelihoods on answer-space cardinality (open-ended vs
  10-choice vs 4-choice chance-match rates).
- **Layer 3 (planned)**: nonparametric domain discovery — embed tasks,
  cluster online, per-cluster count tables with shrinkage toward the global
  prior; sustained surprise in channel statistics triggers local novelty
  raise + count decay (the harness notices its sensors are miscalibrated and
  re-explores).
- **Layer 1.5 (design, after the in-flight-adjustment gotcha)**: always-on
  sliding normalization has an asymmetric transition tax (hotter-domain entry
  overspends visibly; cooler-domain entry under-verifies silently), drifts
  the count semantics mid-flight, and collapses into a domain detector under
  interleaved traffic. Fix: surprise-gated re-centering (CUSUM-style change
  detection on the raw stream) + temporarily flattened channel likelihood
  (precision drop) so the planner knowingly buys information from unaffected
  channels during recalibration + locally raised novelty; windows keyed by
  layer-2 structural features to survive interleaving; absolute channel kept
  in parallel to catch uniform degradation. (Yu & Dayan's expected/unexpected
  uncertainty, implemented.)
