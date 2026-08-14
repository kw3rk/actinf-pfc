#!/usr/bin/env python3
"""Cold-bench difficulty probe — fully self-supervised, no ground truth used.

For each task, fire K temperature-varied no-think replicates with logprobs on.
Task-level answer dispersion (majority agreement across replicates) becomes the
behavioral difficulty label; per-response logit-entropy stats are the candidate
single-shot observation features.

Dataset ground truth is loaded but used ONLY for the audit columns (how often
is the majority consistent-but-wrong) — never for the labels themselves.

Usage: python probe.py --base-url http://192.168.1.194:8153 \
           --n-easy 60 --n-hard 60 --k 8 --out runs/probe_cold.jsonl
"""
import argparse
import json
import math
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from pfc.bench import load_pool
from pfc.llm import SOLVER_SYS, _last_json, _num

TEMPS = [0.5, 0.7, 0.9, 1.1]


def entropy_stats(logprob_content) -> dict:
    """Per-token entropy over renormalized top-k logprobs, summarized."""
    ents = []
    for tok in logprob_content or []:
        tops = tok.get("top_logprobs") or []
        if not tops:
            continue
        lps = np.array([t["logprob"] for t in tops], dtype=float)
        p = np.exp(lps - lps.max())
        p /= p.sum()
        ents.append(float(-(p * np.log(p + 1e-12)).sum()))
    if not ents:
        return {"n_tok": 0}
    e = np.array(ents)
    return {
        "n_tok": len(e),
        "H_mean": float(e.mean()),
        "H_p90": float(np.quantile(e, 0.9)),
        "spike1": float((e > 1.0).mean()),     # share of tokens with H > 1 nat
        "spike2": float((e > 2.0).mean()),
        "varentropy": float(e.var()),
    }


def solve_once(url, text, temp, max_tokens=3072):
    r = requests.post(url, timeout=600, json={
        "messages": [{"role": "system", "content": SOLVER_SYS},
                     {"role": "user", "content": text}],
        "temperature": temp,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 10,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    r.raise_for_status()
    d = r.json()
    ch = d["choices"][0]
    j = _last_json(ch["message"]["content"] or "")
    stats = entropy_stats((ch.get("logprobs") or {}).get("content"))
    return {"answer": _num(j.get("answer")),
            "conf": j.get("confidence", "low"),
            "temp": temp,
            "tokens": d.get("usage", {}).get("completion_tokens", 0),
            **stats}


def probe_task(url, text, true_ans, k):
    temps = (TEMPS * ((k + len(TEMPS) - 1) // len(TEMPS)))[:k]
    resps = [solve_once(url, text, t) for t in temps]
    # group answers by numeric identity (None = its own bucket)
    groups: dict = {}
    for r in resps:
        key = "none" if r["answer"] is None else round(r["answer"], 4)
        groups[key] = groups.get(key, 0) + 1
    n = len(resps)
    maj_key, maj_n = max(groups.items(), key=lambda kv: kv[1])
    is_corr = lambda a: a is not None and math.isclose(
        a, true_ans, rel_tol=1e-4, abs_tol=1e-6)
    for r in resps:
        r["correct"] = is_corr(r["answer"])          # audit column
    return {
        "majority_frac": maj_n / n,
        "n_distinct": len(groups),
        "majority_correct": (maj_key != "none"
                             and is_corr(float(maj_key))),   # audit
        "correct_frac": sum(r["correct"] for r in resps) / n,  # audit
        "responses": resps,
    }


def _auc(pos, neg):
    """Rank AUC: P(feature(pos) > feature(neg))."""
    if not pos or not neg:
        return float("nan")
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def report(rows):
    out = []
    for stratum in ("easy", "hard"):
        rs = [r for r in rows if r["stratum"] == stratum]
        if not rs:
            continue
        mf = np.array([r["majority_frac"] for r in rs])
        cf = np.array([r["correct_frac"] for r in rs])
        cw = sum(1 for r in rs
                 if r["majority_frac"] >= 0.75 and not r["majority_correct"])
        out.append(f"[{stratum}] n={len(rs)}  majority_frac={mf.mean():.2f}  "
                   f"correct_frac={cf.mean():.2f}  "
                   f"consistent-but-wrong: {cw}/{len(rs)}")
    # single-response entropy features vs correctness (audit) and vs
    # task dispersion (the label we'd actually train against)
    resp = [(r["stratum"], x) for r in rows for x in r["responses"]
            if x.get("n_tok", 0) > 0]
    feats = ["H_mean", "H_p90", "spike1", "spike2", "varentropy", "tokens"]
    out.append("\nsingle-response feature -> AUC vs INCORRECT (audit) | "
               "AUC vs task-dispersion>median (label)")
    disp_med = np.median([r["majority_frac"] for r in rows])
    for f in feats:
        wrong = [x[f] for _, x in resp if not x["correct"]]
        right = [x[f] for _, x in resp if x["correct"]]
        task_of = {id(x): r for r in rows for x in r["responses"]}
        hi = [x[f] for _, x in resp
              if task_of[id(x)]["majority_frac"] < disp_med]
        lo = [x[f] for _, x in resp
              if task_of[id(x)]["majority_frac"] >= disp_med]
        out.append(f"  {f:10s}: {_auc(wrong, right):.3f} | {_auc(hi, lo):.3f}")
    conf_wrong = [1.0 if x["conf"] == "high" else 0.0
                  for _, x in resp if not x["correct"]]
    conf_right = [1.0 if x["conf"] == "high" else 0.0
                  for _, x in resp if x["correct"]]
    out.append(f"  (baseline) self-conf AUC vs INCORRECT: "
               f"{1 - _auc(conf_wrong, conf_right):.3f} (higher=conf predicts correct)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://192.168.1.194:8153")
    ap.add_argument("--n-easy", type=int, default=60)
    ap.add_argument("--n-hard", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out", default="runs/probe_cold.jsonl")
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    rng = np.random.default_rng(args.seed)
    easy, hard = load_pool()
    jobs = ([("easy", easy[i]) for i in
             rng.choice(len(easy), args.n_easy, replace=False)] +
            [("hard", hard[i]) for i in
             rng.choice(len(hard), args.n_hard, replace=False)])
    rng.shuffle(jobs)

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out, "w") as f, \
            ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(probe_task, url, text, ans, args.k): (s, text, ans)
                for s, (text, ans) in jobs}
        from concurrent.futures import as_completed
        for i, fut in enumerate(as_completed(futs)):
            s, text, ans = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                print(f"[{i}] {s} FAILED: {e}", flush=True)
                continue
            row.update(stratum=s, true_answer=ans, question=text[:200])
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"[{i:3d}] {s:4s} maj={row['majority_frac']:.2f} "
                  f"distinct={row['n_distinct']} "
                  f"corr={row['correct_frac']:.2f}", flush=True)

    print("\n" + report(rows))


if __name__ == "__main__":
    main()
