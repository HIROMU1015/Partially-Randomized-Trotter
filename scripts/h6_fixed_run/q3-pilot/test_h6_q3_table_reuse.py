"""Run with the H6 risk-source on PYTHONPATH."""

from contextlib import nullcontext

import run_q3


def test_complete_q3_result_matches_uncached_evaluation(monkeypatch):
    rows = [
        {"m": m, "signal": {"raw": {"real": 0.5 if m < 2 else 1.0, "imag": 0.25 if m < 2 else 1.0}}}
        for m in range(13)
    ]
    job = ("Full PF", rows, [1] * 13, {"kind": "synthetic"}, 1, 0.5, 0.0)
    actual = run_q3.compute(job)
    monkeypatch.setattr(run_q3, "reuse_round_tables", lambda _: nullcontext(None))
    expected = run_q3.compute(job)
    assert actual.pop("round_tables") == {"built": 13, "reused": 13}
    expected.pop("round_tables")
    for report in (actual, expected):
        report.pop("wall_seconds")
    assert actual == expected
    assert actual["MSE_lower"] is not None and actual["MSE_upper"] is not None
