"""Persist strict C8-1 postcheck diagnostics, including the failing observable.

The historical checker, numerical backend, circuits and acceptance rules are
reused. Hooks are worker-local and restored even when the checker raises.
"""
from pathlib import Path
import json
import numpy as np


def residual(actual, expected, tolerance, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        return dict(label=label, status="FAIL", reason="shape/nonfinite", tolerance=tolerance, rtol=0)
    error = actual - expected
    maximum = float(np.max(np.abs(error)))
    return dict(label=label, status="PASS" if maximum <= tolerance else "FAIL",
                max_entry=maximum, frobenius=float(np.linalg.norm(error, "fro")),
                tolerance=tolerance, rtol=0)


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def diagnose(checker, compiled, arm, axis, arrays, implementation, threads, output):
    """One historical postcheck; FAIL stays FAIL, with partial checks retained."""
    from . import c1_stage3_molecular as molecular
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    m = implementation
    original_check = molecular.component_check
    original_observable = m.observable_from_columns
    original_small = m.small_operator
    records, observable_checks = [], []

    def same(actual, expected, tolerance, *, label):
        # Capture before the historical check can raise. Never change its result.
        row = residual(actual, expected, tolerance, label)
        records.append(row)
        save_json(output / "partial_checks.json", records)
        return original_check(actual, expected, tolerance, label=label)

    def small(circuit):
        value = original_small(circuit)
        # <=9-qubit matrix only; never a full Reference-register operator.
        np.savez(output / "small_matrix.npz", matrix=value)
        return value

    def observable(columns, ancilla_dimension):
        value = original_observable(columns, ancilla_dimension)
        targets = {"CAR": m.artificial.expected_observable(arrays["V"], axis),
                   "raw": arrays["raw_" + arm + "_" + axis]}
        np.savez(output / "observable.npz", observable=value, **targets)
        for label, target in targets.items():
            row = residual(value, target, m.ATOL, label)
            row["signal_tolerance"] = m.SIGNAL_ATOL
            row["signal_status"] = "PASS" if row.get("frobenius", float("inf")) <= m.SIGNAL_ATOL else "FAIL"
            observable_checks.append(row)
        save_json(output / "observable_residuals.json", observable_checks)
        return value

    molecular.component_check, m.observable_from_columns, m.small_operator = same, observable, small
    try:
        _, report = checker.check_compiled(compiled, arm, axis, arrays, m, threads)
    except Exception as error:
        report = dict(status="FAIL", error_type=type(error).__name__, error=str(error))
    finally:
        molecular.component_check, m.observable_from_columns, m.small_operator = original_check, original_observable, original_small
    report.update(arm=arm, axis=axis, checks=records, observable_residuals=observable_checks,
                  component_atol=m.ATOL, signal_atol=m.SIGNAL_ATOL, rtol=0,
                  global_phase_fit=False, renormalized=False, input_basis_columns=256,
                  acceptance_changed=False, sampling_calls=0, compile_calls=0)
    save_json(output / "check.json", report)
    return report
