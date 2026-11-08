"""Prepared V1/role-budget diagnostic; saved-input execution needs permission.

Only synthetic inputs are exercised during preparation. The production provider
is unchanged. No word, quantum object, guard, RPE calculation or T price is made.
"""
from __future__ import annotations

import ast
from fractions import Fraction as F
from functools import lru_cache
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import tarfile

BASE = "460726569c60ea4cf8f9839f2aff16f84157babc"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/pr_reference_post_v0_v1_budget/2026-11-08"
POLICY = "src/trottertracks/pr_reference_dyadic_dictionary.py"
B3 = "artifacts/pr_reference_n5b2_numeric/2026-11-08/"
M17 = "artifacts/pr_reference_n5b0_static/2026-11-08/M17_INPUTS.json"
V0 = "artifacts/pr_reference_post_n5b3_v0/2026-11-08/"
A4 = "artifacts/pr_reference_n4a_coverage/2026-11-08/ROLE_COVERAGE.json"
B4 = "artifacts/pr_reference_n4b_consumer/2026-11-08/ROLE_WORD_CERTIFICATES.json"
C0 = "artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz"
FILES = {}


@lru_cache(None)
def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    FILES[path] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return raw


def read(path):
    return json.loads(blob(path))


@lru_cache(None)
def rounder(b):
    """Reuse the unchanged round_index body with a different isolated B global."""
    if not isinstance(b, int) or b < 0:
        raise ValueError("Nonnegative integer b required")
    parsed = ast.parse(blob(POLICY))
    node = next(n for n in parsed.body if isinstance(n, ast.FunctionDef)
                and n.name == "round_index")
    assert not node.decorator_list
    ns = {"math": math, "B": b}
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 BASE + ":" + POLICY + ":round_index_only", "exec"), ns)
    return ns["round_index"]


def new_precisions(literals):
    """Fixed prior V1 rule; b36 is inherited, never evaluated again."""
    exponents = []
    for key in literals:
        theta = float.fromhex(key)
        if not math.isfinite(theta) or theta <= 0 or theta.hex() != key:
            raise ValueError("Canonical positive binary64 literal required")
        _, d = theta.as_integer_ratio()
        assert d > 0 and d & (d - 1) == 0
        exponents.append(d.bit_length() - 1)
    if not exponents:
        raise ValueError("No targets")
    return tuple(range(37, max(36, max(exponents)) + 1))


def components(theta, b, guards, sqrt2_upper):
    """Exact additive components; missing guard is NOT an error of zero."""
    if not isinstance(sqrt2_upper, F) or sqrt2_upper <= 0:
        raise ValueError("Positive saved rational sqrt2 upper required")
    if any(not isinstance(g, F) or g < 0 for g in guards.values()):
        raise ValueError("Nonnegative rational guards required")
    n = rounder(b)(theta)
    assert rounder(b)(-theta) == -n
    indices = tuple(k - b for k in range(abs(n).bit_length()) if abs(n) & (1 << k))
    assert sum(1 << (j + b) for j in indices) == abs(n)
    delta = abs(F.from_float(theta) - F(n, 2**b))
    assert delta <= F(1, 2**(b + 1))
    assert delta != F(1, 2**(b + 1)) or n % 2 == 0
    q = sqrt2_upper * delta / 2
    missing = tuple(j for j in indices if j not in guards)
    known = sum((guards[j] for j in indices if j in guards), F(0))
    return dict(b=b, rounded_index=n, selected_indices=indices,
                selected_chunk_count=len(indices), delta=delta, Q=q,
                G_known=known, missing_indices=missing,
                logical_upper=None if missing else q + known,
                negative_rule="complete positive-product actual adjoint")


def local_probe(row, cap):
    r = cap - row["Q"] - row["G_known"]
    if r <= 0:
        label = "KNOWN_ADDITIVE_CERTIFICATE_EXHAUSTS_LOCAL_ALLOCATION"
    elif row["missing_indices"]:
        label = "POSITIVE_RESIDUAL_FOR_MISSING_GUARDS_ONLY"
    else:
        label = "LOGICAL_UPPER_WITHIN_LOCAL_ALLOCATION_NOT_RPE_PASS"
    return dict(residual=r, relation=label, emitter_and_shell="UNQUALIFIED_NOT_ZERO")


def central_probe(rows, arm_condition, axis):
    parts = arm_condition["axis"][axis]["parts"]
    budget = F.from_float(arm_condition[
        "joint_synthesis_error_retaining_saved_N_inclusive_upper"])
    central_known = parts["rotation"] * max(r["Q"] + r["G_known"] for r in rows)
    missing = sorted({j for r in rows for j in r["missing_indices"]})
    return dict(axis=axis, central_known_certificate_load=central_known,
                complete_central_upper=None if missing else central_known,
                missing_indices=missing, saved_joint_budget=budget,
                residual_for_missing_other_emitter_shell=budget - central_known,
                remaining_role_count_bounds={k: v for k, v in parts.items() if k != "rotation"},
                remaining_errors="UNQUALIFIED_NOT_ZERO", m17_RPE="UNQUALIFIED",
                known_load_is_not_actual_error_lower_bound=True)


def role_envelope(keys, logical_guards):
    """An empty set is unqualified, unless SOURCE separately proves absence."""
    missing = sorted(set(keys) - logical_guards.keys())
    return dict(missing_keys=missing,
                logical_upper=None if missing or not keys else max(logical_guards[k] for k in keys))


def saved_noncentral_envelopes(condition):
    """Future reanalysis of saved guards. No T/sequence/matrix evaluation."""
    coverage, certificates = read(A4), read(B4)
    logical = {}
    raw = blob(C0)
    inventory = read("artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/INPUTS.json")["record_inventory"]
    expected_hash = next(r["sha256"] for r in inventory if r["path"] == C0)
    assert hashlib.sha256(raw).hexdigest() == expected_hash
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:xz") as t:
        for member in t:
            if not member.isfile():
                continue
            r = json.load(t.extractfile(member))
            assert r["status"] == "PASS" and r["guard_method"] == (
                "exact cyclotomic word matrix + outward interval strict Frobenius")
            assert r["precision_exponent"] == -11
            logical[r["theta_hex"]] = F(r["strict_error_upper"])
    assert len(logical) == 299
    assert certificates["count"] == len(certificates["rows"]) == 85
    for r in certificates["rows"]:
        assert r["strict_qualified"] and r["negative"]["negative_whole_adjoint"]
        assert r["positive_literal_hex"] not in logical
        p = r["positive"]
        logical[r["positive_literal_hex"]] = F(p["quantization_upper"]) + F(p["word_error_upper"])
    groups = dict(D=("D_unfused", "D_boundary_fusion_difference", "D_actual_SOURCE_inverse_supplement"),
                  frames=("RTE_basis_frames",), products=("RTE_products",), CSWAP=("CSWAP",))
    output = []
    for arm in ("Bprime", "reference"):
        for group, roles in groups.items():
            keys = set()
            for r in coverage["roles"]:
                if r["arm"] == arm and r["role"] in roles:
                    keys.update(r["positive_saved_keys"])
            envelope = role_envelope(keys, logical)
            absent = all(condition["arms"][arm]["axis"][ax]["parts"][group] == 0 for ax in "XY")
            if absent:
                assert not keys
                envelope["logical_upper"] = F(0)
            output.append(dict(arm=arm, count_group=group, source_roles=roles,
                saved_positive_keys=sorted(keys), **envelope,
                zero_only_for_SOURCE_absent_role=absent,
                m17_key_transfer="CONDITIONAL_ON_FIXED_COMPILER_C_AND_SAVED_SOURCE_CLOSURE",
                signed_D_bookkeeping_not_subtracted=True, emitter="RESERVED_SEPARATELY"))
    return output


def noncentral_load(envelopes, arm, parts):
    if any(not isinstance(v, int) or v < 0 for v in parts.values()):
        raise ValueError("Nonnegative physical count bounds required")
    selected = [r for r in envelopes if r["arm"] == arm]
    known = sum((parts[r["count_group"]] * r["logical_upper"]
                 for r in selected if r["logical_upper"] is not None), F(0))
    return dict(known_noncentral_certificate_load=known,
                missing_role_envelopes=[r["count_group"] for r in selected if r["logical_upper"] is None],
                unassessed_role_groups=["event_phase", "scalar_and_wrapper", "emitter", "shell"],
                full_joint_error_upper=None)


def json_value(x):
    if isinstance(x, F):
        return str(x)
    raise TypeError(type(x).__name__)


def main():
    """Future scientific entry point. Not run during static preparation."""
    assert not OUT.exists(), "Fresh output required; preserve prior results/STOP"
    old = read(B3 + "INPUTS.json")
    previous = read(B3 + "RESULTS.json")
    condition = read(M17)["condition"]
    old_v0 = read(V0 + "RESULTS.json")
    assert read(B3 + "STOP.json")["status"] == "MANDATORY_STOP"
    assert read(V0 + "STOP.json")["status"] == "MANDATORY_STOP"
    assert (condition["m"], condition["q"], condition["r"], condition["bits"]) == (17, 131072, 32, 88)
    assert {a: x["saved_shots_per_axis"] for a, x in condition["arms"].items()} == {
        "Bprime": 626, "reference": 630}
    expected = {"Bprime": ["0x1.7e4088e024c5fp-12", "0x1.fdab627b8d37ap-14"],
                "reference": ["0x1.7e4088e024c5fp-11", "0x1.fdab627b8d37ap-13"]}
    assert {a: sorted(r["positive_literal"] for r in previous["rows"] if r["arm"] == a)
            for a in expected} == {a: sorted(v) for a, v in expected.items()}
    assert old["saved_sqrt2"]["exact_square_enclosure_passed"]
    sqrt2 = F(old["saved_sqrt2"]["upper"])
    guards = {r["j"]: F(r["saved_guard"]) for r in old["basic_records"]}
    assert len(old["basic_records"]) == len(guards) == 43
    assert set(guards) == set(range(-36, 7))
    assert all(0 <= g <= F(1, 10**13) for g in guards.values())
    precisions = new_precisions([k for keys in expected.values() for k in keys])
    envelopes = saved_noncentral_envelopes(condition)
    noncentral = [dict(arm=arm, axis=ax, **noncentral_load(envelopes, arm,
                      condition["arms"][arm]["axis"][ax]["parts"]))
                  for arm in expected for ax in "XY"]
    rows, local, joint = [], [], []
    for b in precisions:
        for arm, keys in expected.items():
            pair = []
            for key in keys:
                row = dict(arm=arm, positive_literal=key,
                           **components(float.fromhex(key), b, guards, sqrt2))
                pair.append(row)
                rows.append(row)
                for axis in "XY":
                    cap = F.from_float(condition["arms"][arm]["axis"][axis][
                        "uniform_gate_error_strict_upper"])
                    local.append(dict(b=b, arm=arm, positive_literal=key, axis=axis,
                                      saved_cap=cap, **local_probe(row, cap)))
            for axis in "XY":
                probe = central_probe(pair, condition["arms"][arm], axis)
                rest = next(r for r in noncentral if (r["arm"], r["axis"]) == (arm, axis))
                probe["residual_after_known_noncentral"] = (
                    probe["residual_for_missing_other_emitter_shell"] - rest["known_noncentral_certificate_load"])
                joint.append(dict(b=b, arm=arm, **probe))
    assert len(rows) == 4 * len(precisions)
    assert len(local) == 8 * len(precisions) and len(joint) == 4 * len(precisions)
    if precisions:
        assert all(r["Q"] == 0 for r in rows if r["b"] == precisions[-1])
    results = dict(status="FINITE_V1_AND_SYMBOLIC_ROLE_DIAGNOSTIC_COMPLETE",
        new_b=precisions, rows=rows, local_cap_probes=local, central_role_probes=joint,
        b36_inherited_rows=previous["rows"], b36_inherited_V0_probes=old_v0["central_role_probes"],
        saved_noncentral_role_envelopes=envelopes, fixed_noncentral_loads=noncentral,
        T_prices="NOT_EVALUATED", domain="UNCHANGED_UNQUALIFIED", m17_RPE="UNQUALIFIED",
        no_precision_or_policy_adopted=True)
    provenance = dict(base_commit=BASE, inputs=FILES,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reuse="unchanged round_index body, saved 43 guards/sqrt2, saved role counts and budgets",
        archive_guards_not_recomputed=True)
    OUT.mkdir(parents=True)
    for name, value in (("RESULTS.json", results), ("SOURCE_INPUTS.json", provenance),
                        ("STOP.json", dict(status="MANDATORY_STOP", next_owner="GPT/user",
                          new_guard_synthesis_word_compile_RPE_T_calls=0, m17_RPE="UNQUALIFIED"))):
        (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False, default=json_value) + "\n")
    print(json.dumps(dict(status=results["status"], new_component_rows=len(rows),
                         local_probes=len(local), central_probes=len(joint))))


if __name__ == "__main__":
    main()
