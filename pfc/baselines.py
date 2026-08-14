"""Policy protocol wrappers: the actinf controller, fixed policies, and a
contextual Thompson-sampling bandit — all runnable through episodes.run_episode.
"""
from __future__ import annotations

import numpy as np

from .engine import (ActInfController, SOLVE, SOLVE_THINK, VERIFY, REWORK,
                     SUBMIT, S_CORRECT, S_FLAWED,
                     OB_CONF_HIGH, OB_VERDICT_ISSUE, TOKEN_COST)


class ActInfPolicy:
    """Thin adapter around ActInfController for the runner protocol."""
    name = "actinf"

    def __init__(self, **kw):
        self.ctrl = ActInfController(**kw)
        self.can_rework = False

    def reset(self, cue):
        self.ctrl.reset_episode(cue)
        self.can_rework = False

    def choose(self, steps_left):
        return self.ctrl.choose_action(steps_left, self.can_rework)

    def observe(self, action, obs):
        self.ctrl.update_belief(action, obs)
        if action == VERIFY:
            self.can_rework = obs == OB_VERDICT_ISSUE
        else:                       # fresh candidate makes the critique stale
            self.can_rework = False

    def learn(self, trace, difficulty, cue):
        self.ctrl.learn(trace, difficulty, cue)

    def skepticism(self) -> float:
        """P(flawed | has a candidate) — what the oracle discloses to the agent."""
        b = self.ctrl.belief
        pc, pf = b[:, S_CORRECT].sum(), b[:, S_FLAWED].sum()
        return pf / (pc + pf) if (pc + pf) > 1e-9 else 0.5


class FixedPolicy:
    """Scripted action sequences reacting minimally to observations."""

    def __init__(self, name: str):
        self.name = name
        self._script = None

    def reset(self, cue):
        self._t = 0
        self._issue = False

    def choose(self, steps_left):
        t = self._t
        self._t += 1
        if self.name == "never_verify":
            return SOLVE if t == 0 else SUBMIT
        if self.name == "think_always":
            return SOLVE_THINK if t == 0 else SUBMIT
        if self.name == "always_verify":
            # solve -> verify -> (rework -> verify)? -> submit
            seq = [SOLVE, VERIFY]
            if t < len(seq):
                return seq[t]
            if self._issue and t < 4:
                self._issue = False
                return REWORK if self._last != REWORK else VERIFY
            return SUBMIT
        raise ValueError(self.name)

    def observe(self, action, obs):
        self._last = action
        if action == VERIFY:
            self._issue = obs == OB_VERDICT_ISSUE

    def learn(self, trace, difficulty, cue):
        pass

    def skepticism(self) -> float:
        return 0.5


class BanditPolicy:
    """Contextual Thompson sampling over macro-strategies.

    Context = (cue, confidence-of-first-solve). Arms are complete strategies:
      0: solve, submit
      1: solve, verify -> (rework if issue), submit
      2: solve_think, submit
    Reward = (correct ? r+ : r-) - lam * tokens. Normal-Thompson per (ctx, arm).
    """
    name = "bandit"

    def __init__(self, r_correct=1.0, r_wrong=-1.0, lam=0.0007, seed=0):
        self.r_correct, self.r_wrong, self.lam = r_correct, r_wrong, lam
        self.rng = np.random.default_rng(seed)
        self.n = np.ones((2, 2, 3))          # (cue, conf, arm)
        self.mean = np.zeros((2, 2, 3))

    def reset(self, cue):
        self.cue = cue
        self.t = 0
        self.arm = None
        self.conf_hi = 0
        self.issue = False
        self.tokens = 0
        self.done_verify = False
        self.done_rework = False

    def choose(self, steps_left):
        t = self.t
        self.t += 1
        if t == 0:
            # arm 2 must be chosen before seeing confidence; sample using both
            # confidence rows averaged for the pre-solve decision
            samp = self.mean + self.rng.standard_normal(self.mean.shape) / np.sqrt(self.n)
            pre = samp[self.cue].mean(axis=0)          # (3,)
            self.arm = int(np.argmax(pre))
            return SOLVE_THINK if self.arm == 2 else SOLVE
        if self.arm == 1:
            # re-sample arm 0 vs 1 now that confidence is known (arm 2 sunk)
            if t == 1:
                samp = self.mean + self.rng.standard_normal(self.mean.shape) / np.sqrt(self.n)
                row = samp[self.cue, self.conf_hi]
                self.arm = 1 if row[1] >= row[0] else 0
            if self.arm == 1:
                if not self.done_verify:
                    self.done_verify = True
                    return VERIFY
                if self.issue and not self.done_rework:
                    self.done_rework = True
                    self.issue = False
                    return REWORK
        return SUBMIT

    def observe(self, action, obs):
        if action in (SOLVE, SOLVE_THINK, REWORK):
            self.conf_hi = int(obs == OB_CONF_HIGH)
        if action == VERIFY:
            self.issue = obs == OB_VERDICT_ISSUE

    def learn(self, trace, difficulty, cue):
        correct = trace[-1]["status_after"] == 1          # S_CORRECT
        tokens = sum(TOKEN_COST[s["action"]] for s in trace)
        reward = (self.r_correct if correct else self.r_wrong) - self.lam * tokens
        cell = (self.cue, self.conf_hi, self.arm)
        self.n[cell] += 1
        self.mean[cell] += (reward - self.mean[cell]) / self.n[cell]

    def skepticism(self) -> float:
        return 0.5


def make_policy(name: str, seed: int = 0, **kw):
    if name == "actinf":
        return ActInfPolicy(rng=np.random.default_rng(seed), **kw)
    if name == "bandit":
        return BanditPolicy(seed=seed, **kw)
    return FixedPolicy(name)
