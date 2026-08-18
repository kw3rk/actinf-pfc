"""Execute model-written Python in a restricted subprocess.

The tool action's whole value is that execution doesn't hallucinate: the
program either prints a number or it doesn't. Isolation: python -I (no site,
no user paths), hard timeout, CPU and memory rlimits, no shell.
"""
from __future__ import annotations

import re
import resource
import subprocess
import sys


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m[-1] if m else text


def run_python(code: str, timeout: float = 10.0) -> float | None:
    """Run code, parse the last number printed. None on any failure."""
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=timeout,
            preexec_fn=_limits)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if p.returncode != 0:
        return None
    nums = re.findall(r"-?\d[\d,]*\.?\d*(?:e[+-]?\d+)?",
                      p.stdout.strip(), re.IGNORECASE)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None
