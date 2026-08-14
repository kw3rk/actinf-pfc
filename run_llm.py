#!/usr/bin/env python3
"""Real-LLM experiment against any OpenAI-compatible llama.cpp endpoint.

Local 4090:   scripts/serve_local.sh, then
              python run_llm.py --base-url http://127.0.0.1:8085
ai-server:    python run_llm.py --base-url http://192.168.1.194:8153
"""
import argparse
import pathlib

import numpy as np

from pfc.baselines import make_policy
from pfc.episodes import open_log, log_episode, run_episode
from pfc.llm import LlmAgent
from pfc.report import summarize, verify_rate_by_context
from pfc.tasks import make_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8085")
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--policy", default="actinf",
                    choices=["actinf", "bandit", "never_verify",
                             "always_verify", "think_always"])
    ap.add_argument("--db", default="runs/llm.db")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "gsm", "gsm-full", "math500", "mmlu-pro"])
    ap.add_argument("--lam", type=float, default=None,
                    help="token price (utility/token); actinf only")
    ap.add_argument("--frozen", action="store_true",
                    help="eval mode: no learning, no exploration bonus")
    ap.add_argument("--adaptive-entropy", action="store_true",
                    help="bucket entropy vs a running local baseline instead "
                         "of the absolute calibrated threshold")
    ap.add_argument("--load-model", help="npz of pre-trained Dirichlet counts")
    ap.add_argument("--save-model", help="save actinf counts here at the end")
    args = ap.parse_args()

    pathlib.Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    conn = open_log(args.db)
    kw = {}
    if args.policy == "actinf":
        if args.lam:
            kw["lam"] = args.lam
        if args.frozen:
            kw["novelty_weight"] = 0.0
    policy = make_policy(args.policy, seed=args.seed, **kw)
    if args.policy == "actinf" and args.load_model:
        policy.ctrl.load(args.load_model)
    if args.frozen:
        policy.learn = lambda *a, **k: None

    agent = LlmAgent(args.base_url)
    task_rng = np.random.default_rng(args.seed)
    if args.dataset == "gsm":
        from pfc.bench import GsmTaskSource
        source = GsmTaskSource(task_rng)
        get_task = source.make
        n_eps = args.episodes
    elif args.dataset == "gsm-full":
        from pfc.bench import FullBench
        source = FullBench(task_rng)
        get_task = source.make
        n_eps = len(source)
    elif args.dataset == "math500":
        from pfc.bench import Math500Bench
        source = Math500Bench(task_rng)
        get_task = source.make
        n_eps = len(source)
    elif args.dataset == "mmlu-pro":
        from pfc.bench import MmluProBench
        source = MmluProBench(task_rng)
        get_task = source.make
        n_eps = len(source)
    else:
        get_task = lambda i: make_task(i, task_rng)
        n_eps = args.episodes
    norm = None
    if args.adaptive_entropy:
        from pfc.engine import H_TURBULENT
        from pfc.normalize import RunningNorm
        norm = RunningNorm(absolute_fallback=H_TURBULENT)
    for i in range(n_eps):
        task = get_task(i)
        res = run_episode(policy, agent, task, norm=norm)
        log_episode(conn, args.policy, task, res,
                    res.final, res.observations, res.statuses)
        conn.commit()
        marker = "+" if res.correct else "-"
        print(f"[{i:4d}] {marker} diff={task.difficulty} tok={res.tokens:5d} "
              f"acts={'/'.join(res.actions)}", flush=True)

    if args.policy == "actinf" and args.save_model:
        policy.ctrl.save(args.save_model)
        print(f"saved counts -> {args.save_model}")
    print(summarize(conn, window=max(20, args.episodes // 4)))
    if args.policy == "actinf":
        print(verify_rate_by_context(conn, "actinf", last_n=args.episodes))
        print("\nlearned calibration table:")
        print(policy.ctrl.calibration_table())


if __name__ == "__main__":
    main()
