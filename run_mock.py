#!/usr/bin/env python3
"""Mock-mode experiment: all policies over the same task sequence.

Usage: python run_mock.py [--episodes 3000] [--db runs/mock.db] [--seed 7]
"""
import argparse
import pathlib

import numpy as np

from pfc.baselines import make_policy
from pfc.episodes import open_log, log_episode, run_episode
from pfc.mock_llm import MockAgent
from pfc.report import summarize, verify_rate_by_context
from pfc.tasks import make_task

POLICIES = ["actinf", "bandit", "never_verify", "always_verify", "think_always"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--db", default="runs/mock.db")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    pathlib.Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.db).unlink(missing_ok=True)
    conn = open_log(args.db)

    actinf_pol = None
    for pol_name in POLICIES:
        policy = make_policy(pol_name, seed=args.seed)
        if pol_name == "actinf":
            actinf_pol = policy
        task_rng = np.random.default_rng(args.seed)       # identical task stream
        agent = MockAgent(seed=args.seed + 1)             # identical agent noise stream
        for i in range(args.episodes):
            task = make_task(i, task_rng)
            res = run_episode(policy, agent, task)
            log_episode(conn, pol_name, task, res,
                        res.final, res.observations, res.statuses)
        conn.commit()

    print(summarize(conn, window=max(250, args.episodes // 6)))
    print()
    print(verify_rate_by_context(conn, "actinf", last_n=args.episodes // 3))
    print("\nlearned calibration table (actinf):")
    print(actinf_pol.ctrl.calibration_table())


if __name__ == "__main__":
    main()
