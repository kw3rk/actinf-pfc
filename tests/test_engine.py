import numpy as np
import pytest

from pfc.engine import (ActInfController, SOLVE, SOLVE_THINK, VERIFY, REWORK,
                        SUBMIT, REWORK_SKEPTIC, SOLVE_PREPPED, SOLVE_LIKE,
                        S_NONE, S_CORRECT, S_FLAWED, D_EASY, D_HARD,
                        OB_CONF_HIGH, OB_CONF_LOW, OB_VERDICT_OK,
                        OB_VERDICT_ISSUE, OB_AGREE, OB_DISAGREE,
                        CUE_SHORT, CUE_LONG)


def trained_controller(**kw) -> ActInfController:
    """Controller with hand-seeded counts mimicking a well-calibrated model."""
    c = ActInfController(**kw)
    N = 500
    for d, p_ok in [(D_EASY, 0.85), (D_HARD, 0.45)]:
        # solves: land correct with prob p_ok, confidence informative
        # a solve is a fresh attempt whatever the previous status: seed all rows,
        # otherwise unlearned re-solve cells attract a large novelty bonus
        for s_prev in (S_NONE, S_CORRECT, S_FLAWED):
            c.B_counts[SOLVE, d, s_prev, S_CORRECT] += N * p_ok
            c.B_counts[SOLVE, d, s_prev, S_FLAWED] += N * (1 - p_ok)
            p_thk = min(0.95, p_ok + 0.3)
            c.B_counts[SOLVE_THINK, d, s_prev, S_CORRECT] += N * p_thk
            c.B_counts[SOLVE_THINK, d, s_prev, S_FLAWED] += N * (1 - p_thk)
            c.B_counts[REWORK_SKEPTIC, d, s_prev, S_CORRECT] += N * 0.7
            c.B_counts[REWORK_SKEPTIC, d, s_prev, S_FLAWED] += N * 0.3
            p_prep = min(0.95, p_ok + 0.1)
            c.B_counts[SOLVE_PREPPED, d, s_prev, S_CORRECT] += N * p_prep
            c.B_counts[SOLVE_PREPPED, d, s_prev, S_FLAWED] += N * (1 - p_prep)
        for a in SOLVE_LIKE:
            # ratios preserve the pre-agreement-channel posteriors:
            # P(hi|corr)/P(hi|flawed)=4/3, P(lo|corr)/P(lo|flawed)=1/2
            c.A_counts[a, d, S_CORRECT, OB_CONF_HIGH] += N * 0.40
            c.A_counts[a, d, S_CORRECT, OB_CONF_LOW] += N * 0.10
            c.A_counts[a, d, S_CORRECT, OB_AGREE] += N * 0.40
            c.A_counts[a, d, S_CORRECT, OB_DISAGREE] += N * 0.10
            c.A_counts[a, d, S_FLAWED, OB_CONF_HIGH] += N * 0.30
            c.A_counts[a, d, S_FLAWED, OB_CONF_LOW] += N * 0.20
            c.A_counts[a, d, S_FLAWED, OB_AGREE] += N * 0.05
            c.A_counts[a, d, S_FLAWED, OB_DISAGREE] += N * 0.45
        # verifier is informative
        c.A_counts[VERIFY, d, S_CORRECT, OB_VERDICT_OK] += N * 0.9
        c.A_counts[VERIFY, d, S_CORRECT, OB_VERDICT_ISSUE] += N * 0.1
        c.A_counts[VERIFY, d, S_FLAWED, OB_VERDICT_OK] += N * 0.35
        c.A_counts[VERIFY, d, S_FLAWED, OB_VERDICT_ISSUE] += N * 0.65
        # rework mostly fixes
        c.B_counts[REWORK, d, S_FLAWED, S_CORRECT] += N * 0.7
        c.B_counts[REWORK, d, S_FLAWED, S_FLAWED] += N * 0.3
        c.B_counts[REWORK, d, S_CORRECT, S_CORRECT] += N * 0.95
        c.B_counts[REWORK, d, S_CORRECT, S_FLAWED] += N * 0.05
    # cue is informative of difficulty
    c.Acue_counts[D_EASY, CUE_SHORT] += N * 0.85
    c.Acue_counts[D_EASY, CUE_LONG] += N * 0.15
    c.Acue_counts[D_HARD, CUE_LONG] += N * 0.85
    c.Acue_counts[D_HARD, CUE_SHORT] += N * 0.15
    return c


def test_cue_shifts_difficulty_belief():
    c = trained_controller()
    c.reset_episode(cue=CUE_LONG)
    assert c.belief[D_HARD].sum() > 0.7
    c.reset_episode(cue=CUE_SHORT)
    assert c.belief[D_EASY].sum() > 0.7


def test_verdict_issue_shifts_belief_to_flawed():
    c = trained_controller()
    c.reset_episode(cue=CUE_SHORT)
    c.update_belief(SOLVE, OB_CONF_HIGH)
    p_correct_before = c.belief[:, S_CORRECT].sum()
    c.update_belief(VERIFY, OB_VERDICT_ISSUE)
    assert c.belief[:, S_CORRECT].sum() < p_correct_before


def test_first_action_is_a_solve():
    c = trained_controller()
    c.reset_episode(cue=CUE_SHORT)
    assert c.choose_action(steps_left=8) in (SOLVE, SOLVE_THINK)


def test_easy_high_conf_submits_hard_low_conf_does_not():
    c = trained_controller()
    c.reset_episode(cue=CUE_SHORT)
    c.update_belief(SOLVE, OB_CONF_HIGH)
    assert c.choose_action(steps_left=6) == SUBMIT
    c.reset_episode(cue=CUE_LONG)
    c.update_belief(SOLVE, OB_CONF_LOW)
    # belief(correct) is low -> should not submit yet
    assert c.choose_action(steps_left=6) != SUBMIT


def test_info_gain_drives_verification_when_uncertain():
    c = trained_controller(epistemic_weight=0.3)
    c.reset_episode(cue=CUE_LONG)
    c.update_belief(SOLVE, OB_CONF_LOW)
    p_correct = c.belief[:, S_CORRECT].sum()
    assert 0.2 < p_correct < 0.7          # genuinely uncertain
    a = c.choose_action(steps_left=6)
    assert a in (VERIFY, SOLVE_THINK, REWORK, REWORK_SKEPTIC)


def test_forced_submit_at_budget_end():
    c = trained_controller()
    c.reset_episode(cue=CUE_LONG)
    c.update_belief(SOLVE, OB_CONF_LOW)
    assert c.choose_action(steps_left=1) == SUBMIT


def test_learning_updates_counts():
    c = ActInfController()
    a0 = c.A_counts.copy()
    trace = [
        {"action": SOLVE, "obs": OB_CONF_HIGH,
         "status_before": S_NONE, "status_after": S_CORRECT},
        {"action": SUBMIT, "obs": 0,
         "status_before": S_CORRECT, "status_after": S_CORRECT},
    ]
    c.learn(trace, true_difficulty=D_EASY, cue=CUE_SHORT)
    assert c.A_counts[SOLVE, D_EASY, S_CORRECT, OB_CONF_HIGH] == \
        a0[SOLVE, D_EASY, S_CORRECT, OB_CONF_HIGH] + 1
    assert c.D_counts[D_EASY] == 2.0      # prior 1 + 1


def test_agreement_shifts_belief():
    c = trained_controller()
    c.reset_episode(cue=CUE_LONG)
    c.update_belief(SOLVE, OB_CONF_LOW)
    p_before = c.belief[:, S_CORRECT].sum()
    c.update_belief(SOLVE, OB_AGREE)
    p_agree = c.belief[:, S_CORRECT].sum()
    assert p_agree > p_before
    c.reset_episode(cue=CUE_LONG)
    c.update_belief(SOLVE, OB_CONF_LOW)
    c.update_belief(SOLVE, OB_DISAGREE)
    assert c.belief[:, S_CORRECT].sum() < p_before


def test_beliefs_stay_normalized():
    c = trained_controller()
    c.reset_episode(cue=CUE_LONG)
    for a, o in [(SOLVE, OB_CONF_LOW), (VERIFY, OB_VERDICT_ISSUE),
                 (REWORK, OB_CONF_HIGH), (VERIFY, OB_VERDICT_OK)]:
        c.update_belief(a, o)
        assert np.isclose(c.belief.sum(), 1.0)
