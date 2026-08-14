# actinf-pfc

**A learned orchestration layer for local LLMs — active inference as the
"prefrontal cortex" that decides when your model needs to double-check.**

The real bottleneck for local AI isn't speed — it's reliability. Local models
are fast and cheap, but you can't tell *which* answers to trust, and the
standard fixes (always verify, always sample 8×, always think) burn your
compute budget indiscriminately.

`actinf-pfc` puts a small discrete active-inference controller on top of your
model. It watches the model work through cheap, domain-general signals —
**logit entropy**, **self-consistency between attempts**, verifier verdicts —
and learns from experience what each signal is actually worth for *your*
model. Then it plans: solve cheaply, verify, redo under pressure, escalate to
thinking mode, or submit — choosing whichever action has the best expected
value at the token price you declare. The controller itself is microseconds of
numpy on a 6-cell belief space; all the compute stays in the LLM calls it
decides to make.

## Measured results (Qwen3.6-35B-A3B, llama.cpp on one GPU)

Full GSM8K + GSM-Hard benchmark (n=2638), controller calibrated on just 120
episodes then frozen:

| | single-shot | actinf controller |
|---|---|---|
| overall | 0.787 | **0.857** |
| GSM8K | 0.891 | **0.957** |
| GSM-Hard | 0.684 | **0.757** |
| tokens/episode | 950 | 1808 |

**Blind transfer** — the same GSM-calibrated controller dropped onto MATH-500
(competition math it never saw, n=374): baseline 0.799 → **0.869** (+7.0),
with the gain concentrated on the hardest problems (levels 4–5: 0.747 →
0.830). The calibration is knowledge about *the model*, not the task — you
calibrate once per model and it ports across domains.

The token price λ is a knob, not a constant: at a frugal price the controller
converges to ~0.83 accuracy at ~1050 tokens/episode; at a generous price,
~0.87–0.93 at ~1700. Same engine, same sensors — the cost-accuracy point is
*declared*, not designed.

## What it learned (that you'd want to know anyway)

The controller's model is a set of Dirichlet count tables you can read
directly. On our test model it discovered, without being told:

- **Verbal self-confidence is nearly worthless** (P(says "high" | wrong) ≈
  0.85) — but **mean logit entropy is a strong truth signal** (AUC 0.876 vs
  0.734 for verbal confidence; P(wrong | turbulent) ≈ 0.5 vs 0.04 smooth).
- **An LLM "verifier" persona shares the solver's blind spots**: on problems
  the model can't compute, verify-verdicts carried ~zero information — and
  an adversarial "find the error" framing *false-alarms on half of correct
  answers*.
- **Agreement between a first attempt and a redo-under-pressure is the
  strongest signal available** (P(agree | correct) ≈ 0.8 vs ≈ 0.4 wrong).
- Thinking mode wasn't worth its price on this task family — measured, not
  assumed.

## How it works

- **Hidden state** (6 cells): is the current candidate answer correct ×
  is this task hard. Never directly observable — that's the point.
- **Observations** (discrete, cheap): entropy bucket of the generation,
  agree/disagree with the previous candidate, verifier verdict, a task cue.
- **Actions**: `solve` (thinking off) · `solve_think` · `verify` ·
  `rework` (with critique) · `rework_skeptic` (the controller's posterior is
  disclosed to the model: "an external monitor estimates an N% chance this is
  wrong — redo it; keep it if your redo confirms") · `submit`.
- **Planning**: exact expectimax over the belief tree, scoring expected
  utility (submit-correct reward − token costs) + information gain +
  Dirichlet novelty (the exploration drive; without it, unlearned actions
  predict zero info-gain and are never tried).
- **Learning**: at episode end, every intermediate candidate is graded and
  the likelihood/transition/cost tables update by counting. No gradients,
  no training runs — calibration is legible and accumulates online.

**Self-calibration without ground truth**: `probe.py` runs a cold bench —
K temperature-varied replicates per task — and uses answer *dispersion* as
the difficulty label, no answer key needed. That's how the entropy threshold
is set for a new model. (Audited on GSM: the method's blind spot is
consistent-but-wrong convergence, ~15% of hard tasks — measure it before
trusting agreement signals alone.)

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests/          # engine sanity (no GPU needed)
.venv/bin/python run_mock.py               # watch it learn against a simulated agent

# real model: any OpenAI-compatible server (llama.cpp shown; needs --jinja)
llama-server -m your-model.gguf --jinja -fa 1 -ngl 99 --port 8085 &
scripts/fetch_benchmarks.sh

# 1. cold-calibrate the entropy threshold for your model (no labels needed)
.venv/bin/python probe.py --base-url http://127.0.0.1:8085

# 2. calibrate the controller (~100 episodes)
.venv/bin/python run_llm.py --base-url http://127.0.0.1:8085 \
    --dataset gsm --episodes 120 --save-model runs/counts.npz

# 3. run it
.venv/bin/python run_llm.py --base-url http://127.0.0.1:8085 \
    --dataset gsm-full --frozen --load-model runs/counts.npz
```

Thinking-mode toggling uses `chat_template_kwargs.enable_thinking` (Qwen3
family; other hybrid models may need a different switch in `pfc/llm.py`).

## Adapting it

- `pfc/llm.py` — role templates; swap in your own task besides math by
  changing the solver/verifier prompts and the grading function.
- `H_TURBULENT` in `pfc/engine.py` — entropy threshold; re-derive with
  `probe.py` for a new model.
- `lam` — the token price. The single most important knob: it declares how
  many tokens a correct answer is worth, and the controller's entire
  personality follows from it.

## Honest limitations

- Calibration currently learns from graded episodes — you need checkable
  tasks (or the dispersion-probe fallback) to calibrate. Fully
  posterior-based learning is future work.
- Orchestration harvests *variance*: it cannot fix answers the model can't
  compute at any temperature. Tool actions (e.g. code execution) are the
  natural next action class — to the controller, a tool is just another
  column with (very good) learnable likelihoods.
- One task at a time; the multi-agent/cloud version (this controller as the
  routing brain over many workers) is the roadmap.

## Background

The design bet: the orchestration layer of an agent system is a small,
factored, slowly-changing state space — exactly where discrete active
inference works, while LLMs handle everything open-ended. "Double-check" is
an epistemic action, chosen when the posterior over "actually correct" is
diffuse and a verdict would sharpen it. Related lineages: adaptive
self-consistency (Aggarwal et al. 2023), semantic entropy (Farquhar et al.
2024), compute-optimal test-time scaling (Snell et al. 2024). The
combination here — learned observation calibration + expected-free-energy
planning over remediation actions + posterior disclosure back into prompts —
is, as far as we can tell, new.

MIT licensed. Built with Claude Code.
