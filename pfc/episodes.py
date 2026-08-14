"""Episode runner: controller x agent -> graded, logged episodes.

Every intermediate candidate is graded against ground truth at episode end,
which is what makes supervised Dirichlet calibration (and honest backtesting)
possible. All policies implement the small Protocol below so the actinf
controller and the baselines run through the identical loop.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass

from .engine import (SOLVE, SOLVE_THINK, VERIFY, REWORK, SUBMIT,
                     REWORK_SKEPTIC, S_NONE, S_CORRECT, S_FLAWED,
                     OB_NONE, OB_CONF_LOW, OB_CONF_HIGH,
                     OB_VERDICT_OK, OB_VERDICT_ISSUE, OB_AGREE, OB_DISAGREE,
                     OB_ENT_SMOOTH, OB_ENT_TURBULENT, H_TURBULENT,
                     ACTION_NAMES, TOKEN_COST)
from .tasks import Task

MAX_STEPS = 8

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy TEXT, tid INTEGER, difficulty INTEGER, cue INTEGER,
    actions TEXT, observations TEXT, statuses TEXT,
    final_answer INTEGER, truth INTEGER, correct INTEGER,
    tokens INTEGER, steps INTEGER
);
"""


@dataclass
class EpisodeResult:
    correct: bool
    tokens: int
    steps: int
    actions: list
    trace: list          # per-step dicts used for controller learning


def open_log(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    return conn


def log_episode(conn, policy: str, task: Task, res: EpisodeResult,
                final_answer, observations, statuses):
    conn.execute(
        "INSERT INTO episodes (policy, tid, difficulty, cue, actions, observations,"
        " statuses, final_answer, truth, correct, tokens, steps)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (policy, task.tid, task.difficulty, task.cue,
         json.dumps(res.actions), json.dumps(observations), json.dumps(statuses),
         final_answer, task.answer, int(res.correct), res.tokens, res.steps))


def run_episode(policy, agent, task: Task) -> EpisodeResult:
    """policy: reset(cue) / choose(steps_left) -> action /
    observe(action, obs) / learn(trace, difficulty, cue)."""
    policy.reset(task.cue)
    candidate = None          # current AgentResult
    critique = ""
    tokens = 0
    actions, observations, statuses, trace = [], [], [], []

    def status_of(ans):
        if ans is None:
            return S_NONE
        ok = math.isclose(ans, task.answer, rel_tol=1e-4, abs_tol=1e-6)
        return S_CORRECT if ok else S_FLAWED

    import math as _math

    def fresh_obs(prev_ans, new_cand):
        """Re-solves report self-consistency (agree/disagree with the previous
        candidate); first attempts report logit entropy when available (real
        LLM), falling back to verbal self-confidence (mock)."""
        if prev_ans is not None and new_cand.answer is not None:
            same = _math.isclose(new_cand.answer, prev_ans,
                                 rel_tol=1e-4, abs_tol=1e-6)
            return OB_AGREE if same else OB_DISAGREE
        if new_cand.entropy is not None:
            return (OB_ENT_TURBULENT if new_cand.entropy >= H_TURBULENT
                    else OB_ENT_SMOOTH)
        return OB_CONF_HIGH if new_cand.confidence == "high" else OB_CONF_LOW

    for step in range(MAX_STEPS):
        a = policy.choose(MAX_STEPS - step)
        prev_ans = candidate.answer if candidate else None
        s_before = status_of(prev_ans)
        step_tokens = TOKEN_COST[SUBMIT]

        if a in (SOLVE, SOLVE_THINK):
            candidate = agent.solve(task, think=(a == SOLVE_THINK))
            obs = fresh_obs(prev_ans, candidate)
            step_tokens = candidate.tokens
        elif a == VERIFY:
            v = agent.verify(task, candidate.answer)
            critique = v.critique
            obs = OB_VERDICT_OK if v.ok else OB_VERDICT_ISSUE
            step_tokens = v.tokens
        elif a == REWORK:
            candidate = agent.rework(task, candidate.answer, critique)
            obs = fresh_obs(prev_ans, candidate)
            step_tokens = candidate.tokens
        elif a == REWORK_SKEPTIC:
            candidate = agent.rework_skeptic(task, candidate.answer,
                                             policy.skepticism())
            obs = fresh_obs(prev_ans, candidate)
            step_tokens = candidate.tokens
        else:  # SUBMIT
            obs = OB_NONE

        tokens += step_tokens
        s_after = status_of(candidate.answer if candidate else None)
        actions.append(ACTION_NAMES[a])
        observations.append(obs)
        statuses.append(s_after)
        trace.append({"action": a, "obs": obs, "tokens": step_tokens,
                      "status_before": s_before, "status_after": s_after})

        if a == SUBMIT:
            break
        policy.observe(a, obs)

    final = candidate.answer if candidate else None
    correct = final is not None and math.isclose(
        final, task.answer, rel_tol=1e-4, abs_tol=1e-6)
    res = EpisodeResult(correct=correct, tokens=tokens, steps=len(actions),
                        actions=actions, trace=trace)
    policy.learn(trace, task.difficulty, task.cue)
    res.observations, res.statuses, res.final = observations, statuses, final
    return res
