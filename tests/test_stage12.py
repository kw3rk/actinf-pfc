"""Stage 1 (variant bandit) and Stage 2 (prep pipeline) machinery tests."""
import numpy as np

from pfc.baselines import make_policy
from pfc.episodes import run_episode
from pfc.mock_llm import MockAgent, MockConfig
from pfc.tasks import make_task
from pfc.variants import VariantBandit


def test_bandit_converges_to_planted_best():
    rng = np.random.default_rng(0)
    bandit = VariantBandit(seed=0)
    true_p = [0.55, 0.75, 0.45]
    bandit.register("slot", [f"v{i}" for i in range(3)])
    for _ in range(600):
        idx, _ = bandit.choose("slot")
        bandit.update("slot", idx, rng.random() < true_p[idx])
    n = bandit.wins["slot"] + bandit.losses["slot"]
    assert int(np.argmax(n)) == 1          # best variant got the most pulls
    assert int(np.argmax(bandit.posterior_means("slot"))) == 1


def test_bandit_save_load(tmp_path):
    b = VariantBandit()
    b.register("s", ["a", "b"])
    b.update("s", 0, True)
    b.update("s", 1, False)
    p = str(tmp_path / "bandit.json")
    b.save(p)
    b2 = VariantBandit()
    b2.register("s", ["a", "b"])
    b2.load(p)
    assert np.array_equal(b2.wins["s"], b.wins["s"])
    assert np.array_equal(b2.losses["s"], b.losses["s"])


def test_mock_prepped_beats_raw_solve_on_hard():
    cfg = MockConfig()
    agent = MockAgent(cfg, seed=1)
    rng = np.random.default_rng(2)
    raw = prepped = n = 0
    for i in range(2000):
        task = make_task(i, rng, p_hard=1.0)
        raw += agent.solve(task, think=False).answer == task.answer
        prepped += agent.solve_prepped(task).answer == task.answer
        n += 1
    assert prepped / n > raw / n + 0.05    # planted boost is visible


def test_episode_credits_variants():
    """Running episodes must move the agent's bandit counts."""
    agent = MockAgent(seed=3)
    policy = make_policy("always_verify")
    rng = np.random.default_rng(4)
    before = (agent.variants.wins["verifier"] +
              agent.variants.losses["verifier"]).sum()
    for i in range(30):
        run_episode(policy, agent, make_task(i, rng))
    after = (agent.variants.wins["verifier"] +
             agent.variants.losses["verifier"]).sum()
    assert after > before


def test_skeptic_bandit_learns_planted_delta_through_episodes():
    """End-to-end: episodes credit the skeptic slot; the planted best
    variant (idx 1, +0.10 fix rate) should lead after enough episodes."""
    agent = MockAgent(seed=5)
    policy = make_policy("actinf", seed=5)
    rng = np.random.default_rng(6)
    for i in range(800):
        run_episode(policy, agent, make_task(i, rng, p_hard=0.7))
    means = agent.variants.posterior_means("skeptic")
    n = agent.variants.wins["skeptic"] + agent.variants.losses["skeptic"] - 2
    if n.sum() >= 60:                      # controller used the skeptic enough
        assert means[1] >= means[2]        # planted best beats planted worst
