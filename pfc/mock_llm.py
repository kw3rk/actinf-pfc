"""Simulated agent with configurable competence and (mis)calibration.

Lets the harness run thousands of episodes in seconds with known ground-truth
behavior, so we can check the controller learns the right calibration table
before pointing it at a real model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .engine import D_EASY, D_HARD
from .tasks import Task


@dataclass
class AgentResult:
    answer: float | None
    confidence: str          # "low" | "high"
    tokens: int
    raw: str = ""
    entropy: float | None = None      # mean logit entropy; None if unavailable


@dataclass
class VerifyResult:
    ok: bool
    critique: str
    tokens: int
    raw: str = ""


@dataclass
class MockConfig:
    # P(candidate correct) by (difficulty, think)
    p_solve: dict = field(default_factory=lambda: {
        (D_EASY, False): 0.85, (D_EASY, True): 0.95,
        (D_HARD, False): 0.45, (D_HARD, True): 0.75,
    })
    # weakly calibrated self-confidence
    p_conf_high_given_correct: float = 0.80
    p_conf_high_given_flawed: float = 0.60
    # imperfect verifier persona (shares blind spots with solver)
    p_ok_given_correct: float = 0.90
    p_ok_given_flawed: float = 0.35
    # rework fixes a flawed candidate with these odds; may break a correct one
    p_fix: dict = field(default_factory=lambda: {D_EASY: 0.80, D_HARD: 0.55})
    p_break_correct: float = 0.05
    # skeptical rework: better fix rate (focused redo) but a real sycophancy
    # cost — pressure sometimes breaks a correct answer
    p_fix_skeptic: dict = field(default_factory=lambda: {D_EASY: 0.85, D_HARD: 0.60})
    p_break_skeptic: float = 0.12
    tokens: dict = field(default_factory=lambda: {
        "solve": 300, "solve_think": 1500, "verify": 250, "rework": 400,
        "rework_skeptic": 450})


class MockAgent:
    def __init__(self, cfg: MockConfig | None = None, seed: int = 0):
        self.cfg = cfg or MockConfig()
        self.rng = np.random.default_rng(seed)

    def _emit(self, task: Task, correct: bool, tokens: int) -> AgentResult:
        c = self.cfg
        ans = task.answer if correct else task.answer + int(self.rng.integers(1, 30))
        p_hi = c.p_conf_high_given_correct if correct else c.p_conf_high_given_flawed
        conf = "high" if self.rng.random() < p_hi else "low"
        return AgentResult(answer=ans, confidence=conf, tokens=tokens)

    def solve(self, task: Task, think: bool) -> AgentResult:
        c = self.cfg
        correct = self.rng.random() < c.p_solve[(task.difficulty, think)]
        return self._emit(task, correct, c.tokens["solve_think" if think else "solve"])

    def verify(self, task: Task, answer: int | None) -> VerifyResult:
        c = self.cfg
        correct = answer == task.answer
        p_ok = c.p_ok_given_correct if correct else c.p_ok_given_flawed
        ok = self.rng.random() < p_ok
        return VerifyResult(ok=ok, critique="" if ok else "step arithmetic looks off",
                            tokens=c.tokens["verify"])

    def rework(self, task: Task, answer: int | None, critique: str) -> AgentResult:
        c = self.cfg
        was_correct = answer == task.answer
        if was_correct:
            correct = self.rng.random() > c.p_break_correct
        else:
            correct = self.rng.random() < c.p_fix[task.difficulty]
        return self._emit(task, correct, c.tokens["rework"])

    def rework_skeptic(self, task: Task, answer: float | None,
                       p_flawed: float) -> AgentResult:
        c = self.cfg
        was_correct = answer == task.answer
        if was_correct:
            correct = self.rng.random() > c.p_break_skeptic
        else:
            correct = self.rng.random() < c.p_fix_skeptic[task.difficulty]
        return self._emit(task, correct, c.tokens["rework_skeptic"])
