"""GSM8K (easy stratum) + GSM-Hard (hard stratum) task source.

GSM-Hard is GSM8K with numbers inflated until arithmetic breaks — same
reasoning, hostile arithmetic — giving a ground-truth difficulty label per
episode. The observable cue is number magnitude in the text (the analogue of
the synthetic generator's noisy length cue).
"""
from __future__ import annotations

import json
import re

import numpy as np

from .engine import D_EASY, D_HARD, CUE_SHORT, CUE_LONG
from .tasks import Task

BIG_NUMBER = 100_000


def _gsm8k_answer(ans_text: str) -> float:
    tail = ans_text.rsplit("####", 1)[-1].strip().replace(",", "")
    return float(tail)


def _cue(text: str) -> int:
    nums = [float(n.replace(",", "")) for n in
            re.findall(r"\d[\d,]*\.?\d*", text)]
    return CUE_LONG if any(abs(n) >= BIG_NUMBER for n in nums) else CUE_SHORT


def load_pool(easy_path="data/gsm8k_test.jsonl", hard_path="data/gsmhard.jsonl"):
    easy, hard = [], []
    with open(easy_path) as f:
        for line in f:
            j = json.loads(line)
            easy.append((j["question"], _gsm8k_answer(j["answer"])))
    with open(hard_path) as f:
        for line in f:
            j = json.loads(line)
            hard.append((j["input"], float(j["target"])))
    return easy, hard


class GsmTaskSource:
    """Draws episodes from shuffled GSM8K/GSM-Hard pools, 50/50 by default."""

    def __init__(self, rng: np.random.Generator, p_hard: float = 0.5,
                 easy_path="data/gsm8k_test.jsonl",
                 hard_path="data/gsmhard.jsonl"):
        self.rng = rng
        self.p_hard = p_hard
        self.easy, self.hard = load_pool(easy_path, hard_path)
        self._easy_order = rng.permutation(len(self.easy)).tolist()
        self._hard_order = rng.permutation(len(self.hard)).tolist()

    def make(self, tid: int) -> Task:
        if self.rng.random() < self.p_hard:
            text, ans = self.hard[self._hard_order.pop()]
            diff = D_HARD
        else:
            text, ans = self.easy[self._easy_order.pop()]
            diff = D_EASY
        return Task(tid=tid, text=text, answer=ans, difficulty=diff,
                    cue=_cue(text))


def _parse_math_answer(a: str) -> float | None:
    a = a.strip().replace(",", "").replace("\\!", "").replace("$", "")
    try:
        return float(a)
    except ValueError:
        pass
    m = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(\d+)\}", a)
    if m:
        v = float(m.group(1)) / float(m.group(2))
        return -abs(v) if a.startswith("-") else v
    return None


class Math500Bench:
    """MATH-500 filtered to numerically-gradable answers. A *transfer* bench:
    difficulty here (level>=4) is an audit label only — the controller runs
    blind on whatever cues it has."""

    def __init__(self, rng: np.random.Generator, path="data/math500.jsonl"):
        items = []
        with open(path) as f:
            for line in f:
                j = json.loads(line)
                ans = _parse_math_answer(j["answer"])
                if ans is None:
                    continue
                diff = D_HARD if j["level"] >= 4 else D_EASY
                items.append((j["problem"], ans, diff))
        self.items = [items[i] for i in rng.permutation(len(items))]

    def __len__(self):
        return len(self.items)

    def make(self, tid: int) -> Task:
        text, ans, diff = self.items[tid]
        return Task(tid=tid, text=text, answer=ans, difficulty=diff,
                    cue=_cue(text))


MMLU_SUBJECTS = ("law", "psychology", "philosophy", "history", "health",
                 "economics", "business")


class MmluProBench:
    """MMLU-Pro filtered to non-STEM subjects — a maximally-foreign transfer
    bench (knowledge/judgment, not computation). 10-option multiple choice;
    the model answers with the option number so numeric grading is unchanged.
    No difficulty labels; audit by category post-hoc via question_id."""

    def __init__(self, rng: np.random.Generator, n: int = 500,
                 path="data/mmlu_pro.parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        df = df[df["category"].isin(MMLU_SUBJECTS)]
        df = df.sample(n=min(n, len(df)), random_state=int(rng.integers(1 << 31)))
        self.items = []
        for _, row in df.iterrows():
            opts = "\n".join(f"{i+1}. {o}" for i, o in enumerate(row["options"]))
            text = (f"{row['question']}\n\n{opts}\n\n"
                    f"Answer with the number (1-{len(row['options'])}) "
                    f"of the correct option.")
            self.items.append((text, float(row["answer_index"] + 1),
                               row["category"]))

    def __len__(self):
        return len(self.items)

    def make(self, tid: int) -> Task:
        text, ans, _cat = self.items[tid]
        return Task(tid=tid, text=text, answer=ans, difficulty=0,
                    cue=_cue(text))


class FullBench:
    """Every GSM8K + GSM-Hard problem exactly once, deterministically shuffled."""

    def __init__(self, rng: np.random.Generator,
                 easy_path="data/gsm8k_test.jsonl",
                 hard_path="data/gsmhard.jsonl"):
        easy, hard = load_pool(easy_path, hard_path)
        items = ([(t, a, D_EASY) for t, a in easy] +
                 [(t, a, D_HARD) for t, a in hard])
        self.items = [items[i] for i in rng.permutation(len(items))]

    def __len__(self):
        return len(self.items)

    def make(self, tid: int) -> Task:
        text, ans, diff = self.items[tid]
        return Task(tid=tid, text=text, answer=ans, difficulty=diff,
                    cue=_cue(text))
