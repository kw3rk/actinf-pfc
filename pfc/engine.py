"""Discrete active-inference controller for agent orchestration.

Hidden state (joint, 6 cells):
    status     in {NONE, CORRECT, FLAWED}   -- correctness of current candidate answer
    difficulty in {EASY, HARD}              -- persistent per episode

Actions:
    SOLVE        fresh attempt, thinking off (cheap)
    SOLVE_THINK  fresh attempt, thinking on  (expensive)
    VERIFY       LLM-verifier persona critiques the candidate
    REWORK       revise candidate using last critique
    SUBMIT       commit the answer (terminal)

Observations (feedback modality, action-conditioned):
    OB_NONE, OB_CONF_LOW, OB_CONF_HIGH, OB_VERDICT_OK, OB_VERDICT_ISSUE
plus a one-shot difficulty cue at episode start: CUE_SHORT / CUE_LONG.

Generative model is action-conditioned:  P(o_t | s_t, a_t)  and  P(s_t | s_{t-1}, a_t).
Likelihood (A) and transition (B) tables are Dirichlet-count matrices updated at
episode end from ground truth (the harness grades every intermediate candidate),
i.e. learning is supervised calibration; *control* is expected-free-energy planning
over the current beliefs: EFE = expected utility + epistemic (information-gain) bonus.
"""
from __future__ import annotations

import numpy as np

# state factor indices
S_NONE, S_CORRECT, S_FLAWED = 0, 1, 2
N_STATUS = 3
D_EASY, D_HARD = 0, 1
N_DIFF = 2

# actions
(SOLVE, SOLVE_THINK, VERIFY, REWORK, SUBMIT, REWORK_SKEPTIC,
 SOLVE_PREPPED) = range(7)
N_ACTIONS = 7
ACTION_NAMES = ["solve", "solve_think", "verify", "rework", "submit",
                "rework_skeptic", "solve_prepped"]
SOLVE_LIKE = (SOLVE, SOLVE_THINK, REWORK, REWORK_SKEPTIC, SOLVE_PREPPED)

# feedback observations; agree/disagree = does a fresh candidate match the
# previous one (self-consistency), emitted instead of confidence on re-solves;
# ent_smooth/turbulent = mean logit entropy of the generation, which replaces
# verbal confidence on first attempts when logprobs are available (probe:
# P(wrong|turbulent)=0.50 vs 0.04 smooth; AUC 0.876 vs confidence's 0.734)
(OB_NONE, OB_CONF_LOW, OB_CONF_HIGH, OB_VERDICT_OK, OB_VERDICT_ISSUE,
 OB_AGREE, OB_DISAGREE, OB_ENT_SMOOTH, OB_ENT_TURBULENT) = range(9)
N_OBS = 9
OBS_NAMES = ["none", "conf_low", "conf_high", "verdict_ok", "verdict_issue",
             "agree", "disagree", "ent_smooth", "ent_turbulent"]

# H_mean threshold between smooth and turbulent, from the cold probe's
# top-tercile boundary (runs/probe_cold.jsonl, Qwen3.6-35B-A3B no-think)
H_TURBULENT = 0.1294

CUE_SHORT, CUE_LONG = 0, 1

# prior token costs per action; actual costs are learned online per
# (action, difficulty) and dominate these priors after a few episodes
TOKEN_COST = {SOLVE: 300, SOLVE_THINK: 1500, VERIFY: 250, REWORK: 400,
              SUBMIT: 20, REWORK_SKEPTIC: 450, SOLVE_PREPPED: 550}


class ActInfController:
    def __init__(
        self,
        r_correct: float = 1.0,
        r_wrong: float = -1.0,
        lam: float = 0.0007,        # utility per token: 1 correct ~ 1400 tokens
        epistemic_weight: float = 0.1,
        novelty_weight: float = 0.5,
        horizon: int = 3,
        prior_count: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        self.r_correct = r_correct
        self.r_wrong = r_wrong
        self.lam = lam
        self.epi_w = epistemic_weight
        self.nov_w = novelty_weight
        self.horizon = horizon
        self.rng = rng or np.random.default_rng(0)

        # Dirichlet counts.
        # A[a]: (N_DIFF, N_STATUS, N_OBS) feedback likelihood given action taken
        self.A_counts = np.full((N_ACTIONS, N_DIFF, N_STATUS, N_OBS), prior_count)
        # cue likelihood: (N_DIFF, 2)
        self.Acue_counts = np.full((N_DIFF, 2), prior_count)
        # B[a]: (N_DIFF, N_STATUS_prev, N_STATUS_next) status transition under action
        self.B_counts = np.full((N_ACTIONS, N_DIFF, N_STATUS, N_STATUS), prior_count)
        # difficulty prior counts
        self.D_counts = np.full(N_DIFF, prior_count)

        # learned expected tokens per (action, difficulty); priors from TOKEN_COST
        self.cost_sum = np.array(
            [[TOKEN_COST[a]] * N_DIFF for a in range(N_ACTIONS)], dtype=float)
        self.cost_n = np.ones((N_ACTIONS, N_DIFF))

        self._structural_priors()
        self.reset_episode(cue=None)

    # -- model structure ------------------------------------------------------

    def _structural_priors(self):
        """Hard structural facts the controller shouldn't have to learn:
        which observations are *possible* under which actions, and that
        verify/submit don't change the candidate. Everything quantitative
        (how reliable confidence/verdicts are, how often solves succeed)
        stays learnable."""
        # solve-type actions can only emit confidence obs; verify only verdicts;
        # submit/none emit OB_NONE. Implemented as near-zero counts elsewhere.
        mask = np.zeros((N_ACTIONS, N_OBS))
        for a in SOLVE_LIKE:
            mask[a, [OB_CONF_LOW, OB_CONF_HIGH, OB_AGREE, OB_DISAGREE,
                     OB_ENT_SMOOTH, OB_ENT_TURBULENT]] = 1
        mask[VERIFY, [OB_VERDICT_OK, OB_VERDICT_ISSUE]] = 1
        mask[SUBMIT, OB_NONE] = 1
        self.obs_mask = mask
        self.A_counts *= mask[:, None, None, :]
        self.A_counts += 1e-6

        # verify & submit are status-preserving (structural identity)
        for a in (VERIFY, SUBMIT):
            self.B_counts[a] = np.tile(np.eye(N_STATUS) * 1e6, (N_DIFF, 1, 1))
        # solve-type actions always leave a candidate: staying in NONE is impossible
        for a in SOLVE_LIKE:
            self.B_counts[a, :, :, S_NONE] = 1e-6

    # -- normalized model tensors --------------------------------------------

    def _A(self):
        A = self.A_counts / self.A_counts.sum(axis=-1, keepdims=True)
        return A

    def _Acue(self):
        return self.Acue_counts / self.Acue_counts.sum(axis=-1, keepdims=True)

    def _B(self):
        return self.B_counts / self.B_counts.sum(axis=-1, keepdims=True)

    # -- episode-level belief state ------------------------------------------

    def reset_episode(self, cue: int | None):
        """Start a new episode. Belief is joint over (difficulty, status)."""
        d_prior = self.D_counts / self.D_counts.sum()
        b = np.zeros((N_DIFF, N_STATUS))
        b[:, S_NONE] = d_prior          # no candidate yet
        if cue is not None:
            b *= self._Acue()[:, cue][:, None]
            b /= b.sum()
        self.belief = b

    def update_belief(self, action: int, obs: int):
        """Forward filter: propagate through B[action], condition on obs."""
        B, A = self._B(), self._A()
        b = np.einsum("ds,dst->dt", self.belief, B[action])
        b *= A[action][:, :, obs]
        tot = b.sum()
        if tot < 1e-12:  # impossible obs under model; fall back to propagate-only
            b = np.einsum("ds,dst->dt", self.belief, B[action])
            tot = b.sum()
        self.belief = b / tot

    # -- expected free energy planning ---------------------------------------

    def _exp_cost(self, b: np.ndarray, action: int) -> float:
        """Expected tokens for `action` under the difficulty marginal (learned)."""
        d_marg = b.sum(axis=1)
        means = self.cost_sum[action] / self.cost_n[action]
        return float(d_marg @ means)

    def _submit_value(self, b: np.ndarray) -> float:
        p_correct = b[:, S_CORRECT].sum()
        p_none = b[:, S_NONE].sum()
        # submitting with no/flawed candidate counts as wrong; extra penalty if empty
        util = p_correct * self.r_correct + (1 - p_correct) * self.r_wrong
        return util - p_none * 0.5 - self.lam * self._exp_cost(b, SUBMIT)

    def _predict(self, b: np.ndarray, action: int):
        """Predictive posteriors per observation under `action`.
        Returns (posts: {obs: belief}, p_obs, info_gain)."""
        B, A = self._B(), self._A()
        b_pred = np.einsum("ds,dst->dt", b, B[action])
        joint = b_pred[:, :, None] * A[action]                  # (d, s, o)
        p_obs = joint.sum(axis=(0, 1))
        posts, info = {}, 0.0
        H_prior = -np.sum(b_pred * np.log(b_pred + 1e-12))
        for o in range(N_OBS):
            if p_obs[o] < 1e-9:
                continue
            post = joint[:, :, o] / p_obs[o]
            posts[o] = post
            info += p_obs[o] * (H_prior + np.sum(post * np.log(post + 1e-12)))
        return posts, p_obs, info

    def _novelty(self, b: np.ndarray, action: int) -> float:
        """Expected information gain about the model parameters themselves
        (Dirichlet novelty, ~1/sqrt(counts)): the drive to try under-sampled
        actions. Without it, an action with a flat likelihood prior predicts
        zero state-information gain and is never explored — the calibration
        table can't bootstrap."""
        B = self._B()
        b_pred = np.einsum("ds,dst->dt", b, B[action])
        n_A = self.A_counts[action].sum(axis=-1)          # (d, s)
        n_B = self.B_counts[action].sum(axis=-1)          # (d, s)
        return float(np.sum(b_pred / np.sqrt(n_A)) + np.sum(b / np.sqrt(n_B)))

    def _available(self, b: np.ndarray, can_rework: bool) -> list[int]:
        if b[:, S_NONE].sum() >= 0.5:                 # no candidate yet
            return [SOLVE, SOLVE_THINK, SOLVE_PREPPED]
        acts = [SOLVE, SOLVE_THINK, SOLVE_PREPPED, VERIFY, REWORK_SKEPTIC,
                SUBMIT]
        if can_rework:
            acts.insert(3, REWORK)
        return acts

    def _plan(self, b: np.ndarray, steps_left: int, can_rework: bool,
              depth: int) -> tuple[float, int]:
        """Exact expectimax over the belief tree (spaces are tiny).
        Beyond `depth`, positions are valued as submit-now. Returns (value, action)."""
        has_candidate = b[:, S_NONE].sum() < 0.5
        if steps_left <= 1 or depth <= 0:
            if has_candidate:
                return self._submit_value(b), SUBMIT
            # last step with nothing in hand: solve, graded unsubmitted
            posts, p_obs, _ = self._predict(b, SOLVE)
            v = -self.lam * self._exp_cost(b, SOLVE) + sum(
                p_obs[o] * self._submit_value(post) for o, post in posts.items())
            return v, SOLVE

        best_v, best_a = -np.inf, SUBMIT
        for a in self._available(b, can_rework):
            if a == SUBMIT:
                v = self._submit_value(b)
            else:
                posts, p_obs, info = self._predict(b, a)
                v = (-self.lam * self._exp_cost(b, a) + self.epi_w * info
                     + self.nov_w * self._novelty(b, a))
                # branch only on the mass-dominant observations; value the
                # improbable tail as submit-now (keeps the tree tractable
                # as observation channels are added)
                ranked = sorted(posts, key=lambda o: -p_obs[o])
                kept, mass = [], 0.0
                for o in ranked:
                    kept.append(o)
                    mass += p_obs[o]
                    if mass >= 0.95 or len(kept) >= 3:
                        break
                for o, post in posts.items():
                    if o in kept:
                        cr = (a == VERIFY and o == OB_VERDICT_ISSUE)
                        v += p_obs[o] * self._plan(post, steps_left - 1, cr,
                                                   depth - 1)[0]
                    else:
                        v += p_obs[o] * self._submit_value(post)
            if v > best_v:
                best_v, best_a = v, a
        return best_v, best_a

    def choose_action(self, steps_left: int, can_rework: bool = False) -> int:
        return self._plan(self.belief, steps_left, can_rework, self.horizon)[1]

    # -- learning -------------------------------------------------------------

    def learn(self, episode: list[dict], true_difficulty: int, cue: int):
        """Supervised Dirichlet updates from a graded episode.

        episode: list of {action, obs, status_before, status_after} where the
        status_* fields are ground-truth grades of the candidate at that step.
        """
        self.D_counts[true_difficulty] += 1
        self.Acue_counts[true_difficulty, cue] += 1
        for step in episode:
            a, o = step["action"], step["obs"]
            s0, s1 = step["status_before"], step["status_after"]
            if a in (VERIFY, SUBMIT):
                pass  # structural identity, don't touch B
            else:
                self.B_counts[a, true_difficulty, s0, s1] += 1
            if self.obs_mask[a, o]:
                self.A_counts[a, true_difficulty, s1, o] += 1
            if "tokens" in step:
                self.cost_sum[a, true_difficulty] += step["tokens"]
                self.cost_n[a, true_difficulty] += 1

    # -- introspection --------------------------------------------------------

    def calibration_table(self) -> str:
        """Human-readable read-out of what the A-matrix has learned."""
        A = self._A()
        rows = []
        for a in (SOLVE, SOLVE_THINK, REWORK, REWORK_SKEPTIC, SOLVE_PREPPED):
            for d, dn in [(D_EASY, "easy"), (D_HARD, "hard")]:
                pc_hi = A[a, d, S_CORRECT, OB_CONF_HIGH]
                pf_hi = A[a, d, S_FLAWED, OB_CONF_HIGH]
                pc_ag = A[a, d, S_CORRECT, OB_AGREE]
                pf_ag = A[a, d, S_FLAWED, OB_AGREE]
                pc_tb = A[a, d, S_CORRECT, OB_ENT_TURBULENT]
                pf_tb = A[a, d, S_FLAWED, OB_ENT_TURBULENT]
                rows.append(f"{ACTION_NAMES[a]:14s} {dn}: "
                            f"P(conf_hi) c={pc_hi:.2f} f={pf_hi:.2f} | "
                            f"P(agree) c={pc_ag:.2f} f={pf_ag:.2f} | "
                            f"P(turb) c={pc_tb:.2f} f={pf_tb:.2f}")
        for d, dn in [(D_EASY, "easy"), (D_HARD, "hard")]:
            ok_c = A[VERIFY, d, S_CORRECT, OB_VERDICT_OK]
            ok_f = A[VERIFY, d, S_FLAWED, OB_VERDICT_OK]
            rows.append(f"P(verdict_ok | verify      {dn}) : "
                        f"correct={ok_c:.2f}  flawed={ok_f:.2f}")
        B = self._B()
        for a in (SOLVE, SOLVE_THINK):
            for d, dn in [(D_EASY, "easy"), (D_HARD, "hard")]:
                rows.append(f"P(correct after {ACTION_NAMES[a]:11s} {dn}) : "
                            f"{B[a, d, S_NONE, S_CORRECT]:.2f}")
        return "\n".join(rows)

    def save(self, path: str):
        np.savez(path, A=self.A_counts, Acue=self.Acue_counts,
                 B=self.B_counts, D=self.D_counts,
                 cost_sum=self.cost_sum, cost_n=self.cost_n)

    def load(self, path: str):
        z = np.load(path)
        self.A_counts, self.Acue_counts = z["A"], z["Acue"]
        self.B_counts, self.D_counts = z["B"], z["D"]
        if "cost_sum" in z:
            self.cost_sum, self.cost_n = z["cost_sum"], z["cost_n"]
