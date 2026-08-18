from pfc.sandbox import extract_code, run_python


def test_runs_and_parses_number():
    assert run_python("print((17+6)*3)") == 69.0


def test_extracts_fenced_block():
    text = "Here you go:\n```python\nprint(2**10)\n```\n"
    assert run_python(extract_code(text)) == 1024.0


def test_failure_modes_return_none():
    assert run_python("raise ValueError('boom')") is None
    assert run_python("print('no numbers here')") is None
    assert run_python("while True: pass", timeout=2) is None


def test_big_int_within_grading_tolerance():
    # answers flow through the pipeline as floats and are graded with
    # rel_tol=1e-4; exact integer identity is not required (or preserved
    # for >15-digit results)
    import math
    got = run_python("print(123456789 * 987654321)")
    assert math.isclose(got, 123456789 * 987654321, rel_tol=1e-9)
