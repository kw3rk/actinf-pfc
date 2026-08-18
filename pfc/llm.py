"""OpenAI-compatible client + role templates for a real llama.cpp endpoint.

Works against any /v1/chat/completions server: the local 4090 llama-server or
the ai-server's Qwen3.6-35B-A3B on 192.168.1.194:8153. Thinking is toggled with
Qwen's /think — /no_think soft switch (requires --jinja on the server).
"""
from __future__ import annotations

import json
import re

import numpy as np
import requests

from .mock_llm import AgentResult, VerifyResult
from .tasks import Task

SOLVER_SYS = (
    "You are a careful math solver. Work through the problem, then answer "
    'ONLY with a JSON object on the last line: {"answer": <number>, '
    '"confidence": "high" or "low"}. Use "high" only if you are quite sure.'
)
# neutral framing on purpose: "find the error" measurably false-alarms on
# correct answers (~50% on hard tasks in the first Qwen3.6 run)
VERIFIER_SYS = (
    "You are an independent checker. Recompute the problem yourself, step by "
    "step, WITHOUT looking at the proposed answer until you are done. Then "
    "compare your result to the proposed answer. "
    'Answer ONLY with JSON on the last line: {"verdict": "ok" or "issue", '
    '"critique": "<one sentence: what differs, or empty if they match>"}.'
)
REWORK_SYS = (
    "You are a careful math solver. A checker flagged a problem with a "
    "previous answer. Redo the computation from scratch, taking the critique into "
    'account. Answer ONLY with JSON on the last line: {"answer": <number>, '
    '"confidence": "high" or "low"}.'
)
SKEPTIC_SYS = (
    "You are a careful math solver. An independent statistical monitor — which "
    "has watched many of your past attempts on problems like this and grades "
    "them against ground truth — is skeptical of your previous answer. Its "
    "skepticism is based on measured failure rates, not on spotting a specific "
    "error, so do not simply defer to it: redo the computation from scratch, "
    "carefully, and if your independent redo confirms the previous answer, KEEP "
    "IT. Answer ONLY with JSON on the last line: "
    '{"answer": <number>, "confidence": "high" or "low"}.'
)

# Variant populations (Stage 1): the bandit picks the wording; the counts
# retire the losers. Index 0 of each slot is the incumbent.
SKEPTIC_VARIANTS = [
    SKEPTIC_SYS,
    # v1: method-switch framing — force a different solution path
    "You are a careful math solver. A statistical monitor doubts your previous "
    "answer based on historical failure rates for this problem type. Redo the "
    "problem using a DIFFERENT method or order of operations than you would "
    "naturally use, then compare. If your alternative route reaches the same "
    "answer, keep it; if not, trust the more careful derivation. Answer ONLY "
    'with JSON on the last line: {"answer": <number>, "confidence": "high" or "low"}.',
    # v2: checklist framing — enumerate then recompute
    "You are a careful math solver. An external monitor estimates a meaningful "
    "chance your previous answer is wrong. First list each quantity and "
    "operation the problem requires, one per line. Then recompute step by step "
    "from your list. If the previous answer survives this audit, keep it. "
    'Answer ONLY with JSON on the last line: {"answer": <number>, '
    '"confidence": "high" or "low"}.',
]
VERIFIER_VARIANTS = [
    VERIFIER_SYS,
    # v1: numeric-substitution framing
    "You are an independent checker. Solve the problem yourself from scratch "
    "without looking at the proposed answer. Only after stating your own "
    "result, compare it to the proposed answer. "
    'Answer ONLY with JSON on the last line: {"verdict": "ok" or "issue", '
    '"critique": "<one sentence: what differs, or empty if they match>"}.',
]
PREP_SYS = (
    "You are a careful reader. Do NOT solve the problem. Extract only: "
    "(1) each given quantity or fact, one per line; (2) each constraint or "
    "operation the problem describes, in order; (3) exactly what is being "
    "asked. Be terse and complete."
)
CODE_SYS = (
    "You are a careful programmer. Write a complete Python 3 program that "
    "computes the answer to the problem. The program must print ONLY the "
    "final numeric answer. Use exact integer arithmetic where possible. "
    "Respond with a single ```python code block and nothing else."
)


def _num(x) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", "").replace("$", "").strip())
        except ValueError:
            return None
    return None


def _last_json(text: str) -> dict:
    matches = re.findall(r"\{[^{}]*\}", text, re.DOTALL)
    for m in reversed(matches):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue
    return {}


class LlmAgent:
    def __init__(self, base_url: str, model: str = "default",
                 timeout: float = 300.0, max_tokens: int = 4096,
                 variants=None):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        if variants is None:
            from .variants import VariantBandit
            variants = VariantBandit()
        self.variants = variants
        if "skeptic" not in self.variants.slots:
            self.variants.register("skeptic", SKEPTIC_VARIANTS)
            self.variants.register("verifier", VERIFIER_VARIANTS)

    def _chat(self, system: str, user: str, think: bool, logprobs: bool = False):
        # /think — /no_think soft switches are gone in Qwen3.6; the template
        # kwarg works on both Qwen3.6-35B-A3B (box) and classic Qwen3-4B (local)
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.7,
            # thinking chains need headroom: a truncated chain drops the
            # answer JSON and gets graded flawed, poisoning solve_think's
            # transition counts (observed on Qwen3.8-27B at 4096)
            "max_tokens": self.max_tokens * 3 if think else self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": think},
        }
        if logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = 10
        r = requests.post(self.url, timeout=self.timeout, json=payload)
        r.raise_for_status()
        data = r.json()
        ch = data["choices"][0]
        text = ch["message"]["content"] or ""
        tokens = data.get("usage", {}).get("total_tokens", 0)
        h_mean = None
        content = (ch.get("logprobs") or {}).get("content") if logprobs else None
        if content:
            ents = []
            for tok in content:
                tops = tok.get("top_logprobs") or []
                if not tops:
                    continue
                lps = np.array([t["logprob"] for t in tops], dtype=float)
                p = np.exp(lps - lps.max())
                p /= p.sum()
                ents.append(float(-(p * np.log(p + 1e-12)).sum()))
            h_mean = float(np.mean(ents)) if ents else None
        return text, tokens, h_mean

    def _solve_like(self, system: str, user: str, think: bool) -> AgentResult:
        text, tokens, h_mean = self._chat(system, user, think, logprobs=True)
        j = _last_json(text)
        conf = j.get("confidence", "low")
        return AgentResult(answer=_num(j.get("answer")),
                           confidence="high" if conf == "high" else "low",
                           tokens=tokens, raw=text, entropy=h_mean)

    def solve(self, task: Task, think: bool) -> AgentResult:
        return self._solve_like(SOLVER_SYS, task.text, think)

    def solve_code(self, task: Task) -> AgentResult:
        """Write-and-execute: the model writes a program; the sandbox runs it.
        Execution doesn't hallucinate — a wrong answer means a wrong program,
        not a bad sample, which is why this action's likelihoods should
        calibrate sharp."""
        from .sandbox import extract_code, run_python
        text, tokens, h_mean = self._chat(CODE_SYS, task.text, think=False,
                                          logprobs=True)
        ans = run_python(extract_code(text))
        return AgentResult(answer=ans,
                           confidence="high" if ans is not None else "low",
                           tokens=tokens, raw=text, entropy=h_mean)

    def solve_prepped(self, task: Task) -> AgentResult:
        """Preparation pipeline: extract the problem structure (cheap), gate on
        the extraction's entropy (one re-prep if turbulent), then solve with
        the extraction in context."""
        from .engine import H_TURBULENT
        prep_tokens = 0
        extraction, h = "", None
        for _ in range(2):
            text, tok, h = self._chat(PREP_SYS, task.text, think=False,
                                      logprobs=True)
            prep_tokens += tok
            extraction = text.strip()
            if h is None or h < H_TURBULENT:
                break
        user = (f"{task.text}\n\nA careful reading of the problem:\n"
                f"{extraction}\n\nSolve using this breakdown.")
        res = self._solve_like(SOLVER_SYS, user, think=False)
        res.tokens += prep_tokens
        return res

    def verify(self, task: Task, answer: float | None) -> VerifyResult:
        idx, sys = self.variants.choose("verifier")
        user = f"{task.text}\n\nProposed answer: {answer}"
        text, tokens, _ = self._chat(sys, user, think=False)
        j = _last_json(text)
        ok = j.get("verdict", "issue") == "ok"
        return VerifyResult(ok=ok, critique=str(j.get("critique", "")),
                            tokens=tokens, raw=text,
                            variant=("verifier", idx))

    def rework(self, task: Task, answer: float | None, critique: str) -> AgentResult:
        user = (f"{task.text}\n\nPrevious answer: {answer}\n"
                f"Checker critique: {critique or 'the answer is suspected wrong'}")
        return self._solve_like(REWORK_SYS, user, think=False)

    def rework_skeptic(self, task: Task, answer: float | None,
                       p_flawed: float) -> AgentResult:
        idx, sys = self.variants.choose("skeptic")
        user = (f"{task.text}\n\nPrevious answer: {answer}\n"
                f"Monitor's estimate: {round(100 * p_flawed)}% chance this "
                f"answer is wrong, based on historical failure rates for "
                f"problems of this type (arithmetic slips on large numbers are "
                f"the most common cause).")
        res = self._solve_like(sys, user, think=False)
        res.variant = ("skeptic", idx)
        return res
