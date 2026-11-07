"""N4-C: four saved m7 error/shot checks, no circuit or synthesis imports.

Reuse unchanged CT-0 pure functions and the numeric assignment block from
CT-4.validate. Only shell/RZ Fraction bookkeeping and evidence output are new.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
from importlib.metadata import version
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time
from fractions import Fraction as F
import mpmath as mp

ROOT = Path(__file__).resolve().parents[1]
BASE = "112d2f7eeaa6a0adc8bea2c14347900266d3c7ea"
OUT = ROOT / "artifacts/pr_reference_n4c_accounting/2026-11-08"
B4 = "artifacts/pr_reference_n4b_consumer/2026-11-08/"
A4 = "artifacts/pr_reference_n4a_coverage/2026-11-08/"
CT0 = "artifacts/pr_reference_ct0/2026-11-07/"
CT4 = "artifacts/pr_reference_ct4/2026-11-07/"
D3 = "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/"
BP = "artifacts/pr_reference_bprime/2026-11-07/"
TR = "artifacts/pr_reference_transfer_qualification/2026-11-06/"
VR = "artifacts/pr_reference_depth_v5_reinforcement/2026-11-07/"
INPUTS, REUSED = {}, []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE+":"+path], cwd=ROOT)
    INPUTS[path] = dict(sha256=sha(raw), bytes=len(raw))
    return raw


def read(path):
    return json.loads(blob(path))


def archive(path):
    with tarfile.open(fileobj=io.BytesIO(blob(path)), mode="r:xz") as t:
        return [json.load(t.extractfile(m)) for m in t if m.isfile()]


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(data, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False)+"\n")


def tree(path):
    raw = blob(path)
    assert (ROOT/path).read_bytes() == raw, path
    return raw, ast.parse(raw)


def reuse(path, names, ns):
    raw, parsed = tree(path)
    nodes = [n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    assert all(not n.decorator_list for n in nodes)
    for n in nodes:
        REUSED.append(dict(path=path, function=n.name,
            SOURCE_sha256=sha(raw), function_sha256=sha(ast.get_source_segment(raw.decode(), n).encode())))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), ns)


def numerical_namespace():
    ns = dict(mp=mp, math=math, F=F, defaultdict=defaultdict, archive=archive, BP=BP, TR=TR, VR=VR)
    reuse("src/trottertracks/track_c_compiler_a_18round.py", ["I", "upper", "lower"], ns)
    reuse("src/trottertracks/pr_reference_ct0.py", ["rzcount", "count_assets", "worst_counts", "geometry"], ns)
    path = "src/trottertracks/pr_reference_ct4.py"
    raw, parsed = tree(path)
    parent = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "validate")
    loop = next(n for n in ast.walk(parent) if isinstance(n, ast.For)
        and isinstance(n.target, ast.Tuple) and [a.id for a in n.target.elts] == ["kind", "e"])
    # Extract the ORIGINAL six assignments, not a rewritten RPE formula. The
    # enclosing validate runner also acquires prices/writes old files: never run it.
    assignments = []
    for n in loop.body:
        if not isinstance(n, ast.Assign):
            break
        assignments.append(n)
    assert [n.targets[0].id for n in assignments] == ["ds", "d", "rho", "ratio", "beta", "need"]
    fields = ["ds", "d", "rho", "ratio", "beta", "need"]
    wrapper = ast.FunctionDef(name="saved_CT4_numeric", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg="e"), ast.arg(arg="old")], kwonlyargs=[],
        kw_defaults=[], defaults=[]), body=assignments+[ast.Return(value=ast.Dict(
            keys=[ast.Constant(x) for x in fields], values=[ast.Name(id=x, ctx=ast.Load()) for x in fields]))],
        decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    exec(compile(module, path+":validate:numeric_assignments_only", "exec"), ns)
    REUSED.append(dict(path=path, function="validate: unchanged six numeric assignments only",
        SOURCE_sha256=sha(raw), assignment_source=[ast.get_source_segment(raw.decode(), n) for n in assignments]))
    # CT-1 already closes the complex X/Y signal with the larger per-axis
    # uniform joint bound. Reuse ONLY that unchanged assignment, not its q1 run.
    path = "src/trottertracks/pr_reference_ct1.py"
    raw, parsed = tree(path)
    parent = next(n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == "error_and_shots")
    select = next(n for n in ast.walk(parent) if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "uniform")
    wrapper = ast.FunctionDef(name="saved_CT1_pair_uniform", args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg="old"), ast.arg(arg="pre")], kwonlyargs=[],
        kw_defaults=[], defaults=[]), body=[select, ast.Return(value=ast.Name(id="uniform", ctx=ast.Load()))],
        decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])),
        path+":error_and_shots:pair_uniform_only", "exec"), ns)
    REUSED.append(dict(path=path, function="error_and_shots: unchanged max-axis uniform assignment only",
        SOURCE_sha256=sha(raw), assignment_source=ast.get_source_segment(raw.decode(), select)))
    return ns


def endpoints(value):
    # Existing mpmath exact mpf->rational conversion, no float comparison.
    return tuple(F(*mp.libmp.to_rational(t)) for t in value._mpi_)


def enclosure(value, ns):
    lo, hi = endpoints(value)
    return dict(lower_fraction=str(lo), upper_fraction=str(hi),
        lower_outward_binary64=ns["lower"](value), upper_outward_binary64=ns["upper"](value))


def saved_input_checks():
    checks = read(B4+"CHECKS.json")
    assert checks["status"] == "PASS" and checks["role_certificates"] == 85
    assert checks["C0_unchanged"] == 299 and checks["basic_words_unchanged"] == 43
    assert checks["common_policy"] and checks["local_correctness"]
    assert read(B4+"STOP.json")["status"] == "MANDATORY_STOP"
    provenance = read(B4+"SOURCE_INPUTS.json")
    for p, r in provenance["SOURCE"].items():
        assert sha(blob(p)) == r["sha256"], p
    # Only relevant retained input evidence; no acquisition, provider or guard calls.
    selected = [CT0+"PRECISION_FEASIBILITY.json", CT4+"ERROR_AND_SHOTS.json",
        D3+"basic_records.tar.xz", D3+"BASIC_WORDS_MANIFEST.json", CT4+"strict_records.tar.xz"]
    for p in selected:
        assert sha(blob(p)) == provenance["files"][p]["sha256"]
    roles = read(B4+"ROLE_WORD_CERTIFICATES.json")
    static = read(A4+"ROLE_COVERAGE.json")
    assert {r["positive_literal_hex"] for r in roles["rows"]} == set(static["non_C0_domain_candidate_keys"])
    assert roles["count"] == len(roles["rows"]) == 85
    for r in roles["rows"]:
        assert r["strict_qualified"] and sha(r["positive_sequence"].encode()) == r["positive"]["positive_sequence_sha256"]
        assert F(r["positive"]["strict_operator_Frobenius_upper"]) < F(1, 10**11)
        assert F(r["negative"]["strict_operator_Frobenius_upper"]) == F(r["positive"]["strict_operator_Frobenius_upper"])
        assert r["negative"]["negative_whole_adjoint"]
    policy = read(B4+"COMMON_POLICY_VERIFICATION.json")
    assert policy["status"] == "COMMON_POLICY_IMPLEMENTED" and not policy["arm_or_role_selection_argument"]
    assert len(policy["fixed_C0_priority"]) == 299 and all(r["preserved"] for r in policy["fixed_C0_priority"])
    assert len(policy["paired_arm_checks"]) == 85 and all(r["same_certificate"] for r in policy["paired_arm_checks"])
    local, whole = read(B4+"LOCAL_CORRECTNESS.json"), read(B4+"SAVED_WHOLE_INTEGRATION.json")
    assert len(local["cases"]) == 34 and all(r["passed"] for r in local["cases"])
    assert len(whole["rows"]) == 8 and all(r["passed"] and r["all_saved_literal_counts_match"] for r in whole["rows"])
    assert all(r["audit"]["extra_segment_joins"] == 1 for r in whole["rows"])
    budget = read(D3+"STRICT_ERROR_BUDGET.json")
    assert budget["status"] == "PASS" and budget["C0_guard_plus_emitter_qualified"] == 299
    assert F(budget["allocated_uniform_upper"]) < F(budget["epsilon"]) == F(1,10**11)
    domain = next(r for r in read(CT0+"PHASE_SUPPORT.json")["vacuum_phase"] if r["m"] == 7)
    assert domain["absolute_actual_binary_angle_upper"].hex() == static["vacuum"]["domain_key"]
    assert provenance["fixed_conditions"]["b"] == 36
    return dict(status="ROLE_WORD_POLICY_QUALIFIED", fixed_epsilon="1/100000000000",
        C0=299, basic_words=43, role_keys=85, local_cases=34, saved_whole_cases=8,
        vacuum_domain_key=static["vacuum"]["domain_key"], new_provider_or_guard_calls=0)


def shell_SOURCE_checks():
    specs = {
        "src/trottertracks/pr_reference_interference_pilot.py": ["wrapper", "reference_core", "symmetric_fused"],
        "src/trottertracks/pr_reference_bprime.py": ["build_saved", "rewrite_rte"],
        "src/trottertracks/pr_reference_transfer_qualification.py": ["assemble"],
        "src/trottertracks/pr_reference_ct4.py": ["prepare_one"],
        "src/trottertracks/track_c_compiler_a.py": ["compile_reference"],
        "src/trottertracks/pr_reference_common_word_consumer.py": ["consume_native"],
    }
    rows = []
    for p, names in specs.items():
        raw, parsed = tree(p)
        for n in parsed.body:
            if not isinstance(n, ast.FunctionDef) or n.name not in names: continue
            text = ast.get_source_segment(raw.decode(), n)
            calls = [c.func.attr for c in ast.walk(n) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)]
            if n.name in ("reference_core", "symmetric_fused", "build_saved", "rewrite_rte", "assemble"):
                assert not set(calls) & {"reset", "measure", "barrier"}
            if n.name == "wrapper":
                assert calls.count("measure") == 1 and not set(calls) & {"reset", "barrier"}
                assert isinstance(n.body[-2], ast.Expr) and n.body[-2].value.func.attr == "measure"
            rows.append(dict(path=p, function=n.name, function_sha256=sha(text.encode()),
                reset_measure_barrier_calls=[c for c in calls if c in ("reset", "measure", "barrier")]))
    for p in ("src/trotterlib/df_rte_qiskit.py", "src/trotterlib/df_partial_s2.py"):
        raw, parsed = tree(p)
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ("reset", "measure", "barrier") for n in ast.walk(parsed))
    # The proof is SOURCE structure + unitary template closure under condition C,
    # not an inference from the eight observed native circuits.
    return dict(status="CONSUMER_SHELL_BOUND_INCLUDED", S_upper=1,
        SOURCE_basis="prefix reference resets (Bprime none), contiguous unitary body, one final measurement; no body barriers",
        compiler_dependency="fixed C: unitary gate templates, no new fence-generating pass/routing",
        nonempty_required_for_bound=False,
        finite_observations_are_not_proof=True, SOURCE_functions=rows,
        outer_initial_phase_settings=1, per_initial_setting_units_u=16,
        per_segment_join_units_u=64, u="1/9007199254740992",
        epsilon_shell_spectral_upper=str(F(80, 2**53)),
        norm_transfer="local 1q scalar Frobenius bound used conservatively as spectral; NOT joint Frobenius",
        RZ_emitter_terms_in_shell=False, additional_QPY_copy_to_instruction=0)


def main():
    started = time.monotonic()
    assert not OUT.exists(), "Use fresh N4-C output; preserve completed results"
    assert version("mpmath") == "1.3.0" and version("qiskit") == "1.3.0"
    environment = dict(python=sys.version, executable=sys.executable, host=platform.node(),
        mpmath=version("mpmath"), qiskit_metadata_only=version("qiskit"),
        affinity=len(os.sched_getaffinity(0)), loadavg=os.getloadavg(),
        memory_available=next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')),
        workers=1, reason="four tiny interval expressions; process startup/I/O would dominate",
        interval_precision_bits=192, shared_settings_changed=False)
    source_policy = saved_input_checks()
    shell = shell_SOURCE_checks()
    ns = numerical_namespace()
    mp.iv.prec = 192
    I = ns["I"]
    e_shell = F(shell["epsilon_shell_spectral_upper"])
    epsilon = F(1, 10**11)
    counts, inv = ns["count_assets"]()
    ct0 = next(r for r in read(CT0+"PRECISION_FEASIBILITY.json")["rows"] if r["m"] == 7)
    assert (ct0["m"], ct0["q"], ct0["r"], ct0["bits"]) == (7, 128, 1, 73)
    previous = read(CT4+"ERROR_AND_SHOTS.json")
    assert previous["alpha"] == "1/720" and previous["phase_cap"] == ".4"
    assert previous["total_failure"] == ".05" and previous["compiler_and_phase_disks_added_once"]
    assert previous["energy_accuracy_Ha"] == 1.5936e-4
    old_b = next(r for r in read(BP+"ERROR_RPE.json")["rows"] if r["m"] == 7)
    old_r = next(r for r in read(TR+"STAGE3_ALL_SUPPORT.json")["rows"] if r["m"] == 7)
    expected = {("Bprime","X"):1187357, ("Bprime","Y"):1187358,
                ("reference","X"):1184885, ("reference","Y"):1184886}
    joints, disks, shots, crosschecks = [], [], [], []
    for arm in ("Bprime", "reference"):
        old = ct0["arms"][arm]
        assert old["saved_shots_per_axis"] == 199
        assert old["rho0"] == old_r["nominal_raw_radius_lower"]
        assert F(old["B_lower"])+F(old["PF_phase_upper"]) <= F(2,5)
        retained = old_b if arm == "Bprime" else old_r["arms"]["reference"]
        assert old["d_old"] == retained["total_raw_disk_upper"]
        for axis in "XY":
            parts = ns["worst_counts"](128, 1, axis, counts, inv)[arm]
            L = sum(parts.values())
            assert parts == old["axis"][axis]["parts"]
            assert L == old["axis"][axis]["worst_case_RZ_upper"] == expected[arm, axis]
            e_rz = L*epsilon
            e_syn = e_rz+e_shell
            # A single complex signal uses BOTH coordinates. Keep each axis's
            # physical bound, but use the common larger bound in the RPE disk.
            e_pair = ns["saved_CT1_pair_uniform"](old, dict(epsilon=str(epsilon)))+e_shell
            assert e_pair >= e_syn
            calc = ns["saved_CT4_numeric"](e_pair, old)
            rho_check, beta_check = ns["geometry"](old["rho0"], old["d_old"], old["B_lower"], calc["ds"])
            assert endpoints(rho_check) == endpoints(calc["rho"])
            assert endpoints(beta_check) == endpoints(calc["beta"])
            no_shell = ns["saved_CT4_numeric"](e_rz, old)
            # Reproduce exactly the saved conditional-uniform rows; finite
            # trajectory rows have different logical guards and are not mixed in.
            saved = [r for r in previous["rows"] if r["kind"] == "conditional_all_support_uniform_precision"
                     and r["name"].endswith("_"+arm+"_"+axis)]
            assert len(saved) == 2
            for r in saved:
                assert r["saved_N"] == 199 and r["saved_N_sufficient"]
                for field, name in [("synthesis_raw_disk_upper","ds"), ("d_total_upper","d"),
                    ("required_continuous_N_upper","need")]:
                    assert r[field] == ns["upper"](no_shell[name]), (arm, axis, field)
                assert r["rho_lower"] == ns["lower"](no_shell["rho"])
                assert r["beta_stat_lower"] == ns["lower"](no_shell["beta"])
            lower_need, upper_need = endpoints(calc["need"])
            ceiling = -(-upper_need.numerator // upper_need.denominator)
            checks = dict(positive_radius=endpoints(calc["rho"])[0]>0,
                disk_smaller_than_nominal=endpoints(calc["ratio"])[1]<1,
                positive_statistical_phase=endpoints(calc["beta"])[0]>0,
                PF_plus_disk_phase_within_cap=endpoints(I(old["PF_phase_upper"])
                    + I(old["B_lower"])-calc["beta"])[1] < F(2,5),
                continuous_upper_at_most_saved_N=upper_need <= F(199),
                integer_upper_at_most_saved_N=ceiling<=199)
            passed = all(checks.values())
            tag = dict(arm=arm, axis=axis, m=7, q=128, r=1)
            joints.append(dict(**tag, worst_case_RZ_upper=L, physical_count_parts=parts,
                epsilon_per_RZ_strict_upper=str(epsilon), epsilon_RZ_spectral_upper=str(e_rz),
                epsilon_shell_spectral_upper=str(e_shell), epsilon_syn_spectral_upper=str(e_syn),
                epsilon_syn_used_for_XY_RPE=str(e_pair), common_XY_max_used=True,
                norm_rule="each local Frobenius -> spectral embedding -> unitary telescoping; no price/error cancellation"))
            phase = I(old["B_lower"])-calc["beta"]
            disks.append(dict(**tag, d_old=old["d_old"], d_old_exact_binary64=str(F(old["d_old"])),
                d_syn=enclosure(calc["ds"],ns), d_total=enclosure(calc["d"],ns),
                rho0=old["rho0"], rho_lower_expression=enclosure(calc["rho"],ns),
                additional_disk_phase_bound=enclosure(phase,ns),
                statistical_phase_lower_expression=enclosure(calc["beta"],ns),
                PF_phase_upper=old["PF_phase_upper"], B_lower=old["B_lower"],
                PF_plus_disk_phase_bound=enclosure(I(old["PF_phase_upper"])+phase,ns),
                raw_units="same complex raw X/Y signal disk as CT-0; no further normalization multiplier",
                epsilon_syn_used_for_XY_RPE=str(e_pair),
                addition_terms=["saved d_old ONCE", "2sqrt(2)*(max_XY(L)*1e-11 + e_shell) ONCE"],
                old_disk_components_quoted_only={k:retained[k] for k in retained
                    if k in ("native_raw_disk_upper", "inherited_source_phase_proposal_raw_disk_upper",
                        "phase_generation_extra_raw_disk_upper")},
                old_compiler_or_vacuum_error_readded=False))
            shots.append(dict(**tag, required_continuous_N=enclosure(calc["need"],ns),
                integer_ceiling_of_required_upper=ceiling,
                saved_shots_per_axis=199, alpha="1/720", decision_checks=checks,
                saved_N_margin=enclosure(I(199)-calc["need"],ns),
                saved_199_shots_sufficient=passed,
                status="SAVED_199_SHOTS_SUFFICIENT" if passed else "SAVED_199_SHOTS_NOT_CERTIFIED"))
            assert endpoints(no_shell["need"])[1] < endpoints(calc["need"])[0]
            assert endpoints(calc["d"] - (I(old["d_old"])+calc["ds"]))[0] <= 0 <= endpoints(
                calc["d"] - (I(old["d_old"])+calc["ds"]))[1]
            crosschecks.append(dict(**tag, CT0_pure_geometry_identical=True,
                saved_CT4_uniform_rows_reproduced=len(saved), shell_makes_bound_stricter=True,
                complex_XY_bound_uses_max_axis=True,
                decision_uses_exact_interval_Fraction_endpoints=True))
    shell["raw_shell_disk_contribution"] = enclosure(2*mp.iv.sqrt(I(2))*(I(e_shell.numerator)/e_shell.denominator),ns)
    assert F(36)*F(1,720) == F(1,20) and 2/F(1,720) == 1440
    passed = all(r["saved_199_shots_sufficient"] for r in shots)
    status = "SAVED_199_SHOTS_SUFFICIENT" if passed else "SAVED_199_SHOTS_NOT_CERTIFIED"
    save("JOINT_SYNTHESIS_ERROR.json", dict(status="JOINT_SYNTHESIS_DISK_EVALUATED", rows=joints,
        scope="all canonical support under fixed SOURCE/compiler C, m7 only", new_precision_selection=False))
    save("SHELL_ERROR_ACCOUNTING.json", shell)
    save("RAW_DISK_ACCOUNTING.json", dict(rows=disks,
        intervals_enclose="expressions for conservative bounds, NOT lower/upper values of the unknown physical error",
        d_old_preserved=True, source_vacuum_and_compiler_added_once=True))
    save("RPE_199_SHOT_CHECKS.json", dict(status=status, rows=shots, all_four_passed=passed,
        conditional_on="fixed SOURCE/compiler C, role closure, common policy, C_phi/runtime, original raw signal model",
        total_failure_budget="36/720=0.05 per arm protocol; unchanged, not a new all18 CT certification",
        energy_accuracy_Ha=1.5936e-4, phase_cap="0.4", new_N_schedule=False, new_alpha=False,
        all18_CT_RPE="FULL_SUPPORT_CT_RPE_UNQUALIFIED outside this fixed m7 result"))
    save("CHECKS.json", dict(status="PASS" if passed else "STOP_INSUFFICIENT_BOUND", rows=crosschecks,
        role_policy=source_policy, shell_SOURCE_upper=1,
        source_and_compiler_conditions="SOURCE_AND_COMPILER_CONDITIONS_PRESERVED; CONDITIONAL_ON_COMPILER_CONTEXT",
        guard_synthesis_provider_circuit_compile_sampling_calls=0,
        T_aggregations=0, other_round_reanalyses=0,
        shell_RZ_separation=True, local_not_joint_Frobenius=True,
        raw_conversion="2sqrt(2)", disk_addition_once=True,
        unchanged_N_alpha_failure_precision=True, elapsed_seconds=time.monotonic()-started))
    source = "scripts/account_pr_reference_n4c.py"
    save("SOURCE_INPUTS.json", dict(base_commit=BASE, runner=source,
        runner_sha256=sha((ROOT/source).read_bytes()), files=INPUTS, reused=REUSED,
        new_code="Fraction L*epsilon+shell; saved-input checks; evidence wrappers; original numeric assignments unchanged",
        scientific_conditions=read(B4+"SOURCE_INPUTS.json")["fixed_conditions"]))
    save("ENVIRONMENT.json", environment)
    save("STOP.json", dict(status="MANDATORY_STOP", reason="Fixed four-condition N4-C completed" if passed else "199-shot sufficient bound not certified",
        result=status, next_owner="GPT/user", automatic_expansion=False,
        new_T_or_all18_or_depth_execution=False))
    print(json.dumps(dict(status=status, rows=[dict(arm=r["arm"], axis=r["axis"],
        N_upper=r["required_continuous_N"]["upper_outward_binary64"],
        integer_upper=r["integer_ceiling_of_required_upper"]) for r in shots])))


if __name__ == "__main__":
    main()
