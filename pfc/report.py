"""Windowed metrics tables from the episode log."""
from __future__ import annotations

import json
import sqlite3


def summarize(conn: sqlite3.Connection, window: int = 500) -> str:
    out = []
    policies = [r[0] for r in conn.execute(
        "SELECT DISTINCT policy FROM episodes ORDER BY policy")]
    for pol in policies:
        rows = conn.execute(
            "SELECT correct, tokens, actions, difficulty FROM episodes "
            "WHERE policy=? ORDER BY id", (pol,)).fetchall()
        out.append(f"\n=== {pol} ({len(rows)} episodes) ===")
        out.append(f"{'window':>12s} {'acc':>6s} {'tok/ep':>8s} {'tok/corr':>9s} "
                   f"{'verify%':>8s} {'think%':>7s} {'acc-hard':>8s}")
        for w0 in range(0, len(rows), window):
            chunk = rows[w0:w0 + window]
            n = len(chunk)
            acc = sum(r[0] for r in chunk) / n
            tok = sum(r[1] for r in chunk) / n
            tokc = sum(r[1] for r in chunk) / max(1, sum(r[0] for r in chunk))
            ver = sum("verify" in json.loads(r[2]) for r in chunk) / n
            thk = sum("solve_think" in json.loads(r[2]) for r in chunk) / n
            hard = [r for r in chunk if r[3] == 1]
            acc_h = (sum(r[0] for r in hard) / len(hard)) if hard else float("nan")
            out.append(f"{w0:>5d}-{w0+n-1:<6d} {acc:>6.3f} {tok:>8.0f} {tokc:>9.0f} "
                       f"{ver:>8.2f} {thk:>7.2f} {acc_h:>8.3f}")
    return "\n".join(out)


def verify_rate_by_context(conn: sqlite3.Connection, policy: str,
                           last_n: int = 1000) -> str:
    """Where does the policy spend verification, once trained?"""
    rows = conn.execute(
        "SELECT cue, observations, actions FROM episodes WHERE policy=? "
        "ORDER BY id DESC LIMIT ?", (policy, last_n)).fetchall()
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0])
    for cue, obs_s, act_s in rows:
        obs, acts = json.loads(obs_s), json.loads(act_s)
        conf = "high" if 2 in obs[:1] else "low"       # first solve's confidence
        agg[(cue, conf)][0] += "verify" in acts
        agg[(cue, conf)][1] += 1
    lines = [f"verification rate by context — {policy}, last {last_n} episodes:"]
    for (cue, conf), (v, n) in sorted(agg.items()):
        cue_s = "short" if cue == 0 else "long "
        lines.append(f"  cue={cue_s} conf={conf:4s}: {v/max(1,n):.2f}  (n={n})")
    return "\n".join(lines)
