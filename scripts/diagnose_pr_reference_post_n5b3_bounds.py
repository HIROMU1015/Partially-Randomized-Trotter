"""Prepared V0 only: NOT executed as part of the static policy study.

Four saved delta/G brackets and two central-role budget probes. Running this
file is new scientific reanalysis and requires the separate fixed-V0 permission
described in NEXT_VALIDATION_PLAN.md. No approval-proof machinery is involved.
No word generation, strict guard, provider, compiler, RPE or price calls.
"""
import ast
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import time

import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
BASE = "2886ac07c1ec706a1180d107e0111745e74aec1f"
OUT = ROOT / "artifacts/pr_reference_post_n5b3_v0/2026-11-08"
B3 = "artifacts/pr_reference_n5b2_numeric/2026-11-08/"
B0 = "artifacts/pr_reference_n5b0_static/2026-11-08/"
FILES = {}


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE+":"+path], cwd=ROOT)
    FILES[path] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw


def read(path):
    return json.loads(blob(path))


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False)+"\n")


def reuse(path, name, ns):
    raw = blob(path)
    node = next(n for n in ast.parse(raw).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    assert not node.decorator_list
    exec(compile(ast.Module(body=[node], type_ignores=[]), BASE+":"+path, "exec"), ns)
    return ns[name]


def main():
    started = time.monotonic()
    assert not OUT.exists(), "Fresh V0 output required; preserve any prior STOP"
    assert mp.__version__ == "1.3.0", "Use the saved fixed mpmath environment"
    old = read(B3+"RESULTS.json")
    inputs = read(B3+"INPUTS.json")
    old_checks = read(B3+"CHECKS.json")
    assert read(B3+"STOP.json")["status"] == "MANDATORY_STOP"
    assert old["status"] == "FIXED_CENTRAL_LOGICAL_COMPONENTS_EVALUATED"
    assert old_checks["positive_component_cases"] == 4 and old_checks["reused_round_index_calls"] == 8
    expected = {
        ("Bprime", "0x1.7e4088e024c5fp-12"), ("Bprime", "0x1.fdab627b8d37ap-14"),
        ("reference", "0x1.7e4088e024c5fp-11"), ("reference", "0x1.fdab627b8d37ap-13")}
    assert len(old["rows"]) == 4 and {(r["arm"],r["positive_literal"]) for r in old["rows"]} == expected
    m17 = read(B0+"M17_INPUTS.json")["condition"]
    assert m17 == inputs["fixed_scope"]["fixed_m17_condition"]
    assert (m17["m"],m17["q"],m17["r"],m17["bits"]) == (17,131072,32,88)
    assert [m17["arms"][a]["saved_shots_per_axis"] for a in ("Bprime","reference")] == [626,630]
    ns = dict(mp=mp, F=F)
    I = reuse("src/trottertracks/track_c_compiler_a_18round.py", "I", ns)
    endpoints = reuse("scripts/account_pr_reference_n4c.py", "endpoints", ns)
    mp.iv.prec = 192

    def rational_interval(x):
        return I(x.numerator)/I(x.denominator)

    root = mp.iv.sqrt(I(2))
    root_lo, root_hi = endpoints(root)
    assert root_lo > 0
    brackets, comparisons = [], []
    for row in old["rows"]:
        delta, G = F(row["delta"]["exact_fraction"]), F(row["G"]["exact_fraction"])
        assert 0 <= delta <= F(1,2**37) and G >= 0
        distance = I(2)*root*mp.iv.sin(rational_interval(delta)/I(4))
        d_lo,d_hi = endpoints(distance)
        LF,UF = max(F(0),d_lo-G),d_hi+G
        LS = LF/root_hi  # Generic 2x2 norm conversion, no SU(2) assumption.
        assert 0 <= LF <= UF and 0 <= LS <= UF
        brackets.append(dict(arm=row["arm"], positive_literal=row["positive_literal"],
            delta=str(delta), saved_G=str(G), grid_distance_F_lower=str(d_lo), grid_distance_F_upper=str(d_hi),
            logical_F_lower=str(LF), logical_F_upper=str(UF),
            logical_spectral_lower=str(LS), logical_spectral_upper=str(UF),
            negative_bound="same by saved complete-adjoint and signed-index symmetry"))
        for axis in ("X","Y"):
            cap = F.from_float(m17["arms"][row["arm"]]["axis"][axis]["uniform_gate_error_strict_upper"])
            for norm,lower in (("Frobenius",LF),("spectral",LS)):
                relation = ("LOCAL_ALLOCATION_EXCLUDED_FOR_THIS_WORD" if lower > cap else
                    "LOCAL_LOGICAL_UPPER_STRICTLY_BELOW_CAP" if UF < cap else "UNRESOLVED")
                comparisons.append(dict(arm=row["arm"],axis=axis,norm=norm,
                    positive_literal=row["positive_literal"], cap=str(cap),relation=relation,
                    does_not_decide_joint_RPE=True))
    probes = []
    for arm in ("Bprime","reference"):
        condition = m17["arms"][arm]
        count = condition["axis"]["X"]["parts"]["rotation"]
        assert count == condition["axis"]["Y"]["parts"]["rotation"]
        bound = max(F(r["logical_component_upper"]["exact_fraction"]) for r in old["rows"] if r["arm"] == arm)
        central = count*bound
        budget = F.from_float(condition["joint_synthesis_error_retaining_saved_N_inclusive_upper"])
        probes.append(dict(arm=arm,rotation_count_upper=count,central_error_upper=str(central),
            saved_joint_budget=str(budget), residual_for_UNEVALUATED_other_terms=str(budget-central),
            certificate_probe="CENTRAL_SLICE_LEAVES_ROOM" if central < budget else "THIS_CENTRAL_UPPER_EXHAUSTS_BUDGET",
            other_roles_emitter_shell="UNQUALIFIED_NOT_ZERO", full_RPE_result="UNQUALIFIED"))
    assert len(brackets) == 4 and len(comparisons) == 16 and len(probes) == 2
    save("RESULTS.json",dict(status="V0_SAVED_INPUT_DIAGNOSTIC_COMPLETE",brackets=brackets,
        local_allocation_comparisons=comparisons,central_role_probes=probes,
        no_new_precision_or_policy=True,m17_RPE="UNQUALIFIED"))
    save("SOURCE_INPUTS.json",dict(base_commit=BASE,files=FILES,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interval_bits=192,only_new_transcendental_targets="four saved delta/4 sine intervals and shared sqrt(2)",
        old_result_not_reexecuted=True))
    save("STOP.json",dict(status="MANDATORY_STOP",reason="Fixed V0 complete",m17_RPE="UNQUALIFIED",
        new_guard_synthesis_word_provider_compile_RPE_price_calls=0,next_owner="GPT/user"))
    print(json.dumps(dict(status="V0_SAVED_INPUT_DIAGNOSTIC_COMPLETE",positive_brackets=4,
        norm_axis_comparisons=16,central_probes=2,elapsed_seconds=time.monotonic()-started)))


if __name__ == "__main__":
    fresh = not OUT.exists()
    try:
        main()
    except Exception as exc:
        if fresh and not (OUT/"STOP.json").exists():
            save("STOP.json",dict(status="MANDATORY_STOP",reason=type(exc).__name__+": "+str(exc),
                m17_RPE="UNQUALIFIED",next_owner="GPT/user"))
        raise
