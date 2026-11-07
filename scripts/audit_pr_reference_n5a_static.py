"""N5-A metadata extraction only: no scientific/numerical library imports.

Copy saved round conditions; compare literal/domain/certificate metadata.
Never evaluate an error sum, RPE geometry, shot condition, price or provider.
"""
from __future__ import annotations
import ast
from collections import Counter, defaultdict
from fractions import Fraction as F
import hashlib
import json
import lzma
import math
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "0af90ca199361810f4efbf0ae86422738f4d7f12"
OUT = ROOT/"artifacts/pr_reference_n5a_static/2026-11-08"
CT0 = "artifacts/pr_reference_ct0/2026-11-07/"
B4 = "artifacts/pr_reference_n4b_consumer/2026-11-08/"
C4 = "artifacts/pr_reference_n4c_accounting/2026-11-08/"
D3 = "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/"
REFERENCES = {}
PINNED = {
    "112d2f7eeaa6a0adc8bea2c14347900266d3c7ea": [
        "src/trottertracks/pr_reference_common_word_consumer.py", B4+"ROLE_WORD_CERTIFICATES.json"],
    "002ba1141fcdbfb9bd59b8173fb594da9a89388b": [
        "src/trottertracks/pr_reference_dyadic_dictionary.py", D3+"basic_records.tar.xz", D3+"STRICT_ERROR_BUDGET.json"],
    "fbe7ac8b53acd5c795e94939c5c5e400ced5679a": ["scripts/verify_pr_reference_n3b1b0_runtime.py"],
    "9c60c43135666a9ab5497a43a4b1695ab371650f": ["artifacts/pr_reference_n3b0_numeric/2026-11-08/FIXED_PARAMETERS.json"],
    "0a6ba582d246b3b724d42c8b56475c16ebef1dd9": ["artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json"],
    "0dbf1fe265176677e026ea982c0bd435987f4338": ["docs/research/pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md"],
    "8012c3e9547690f0d722edac9d0b9a9391ba2f8e": ["artifacts/pr_reference_ct4/2026-11-07/strict_records.tar.xz"],
}


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE+":"+path], cwd=ROOT)
    REFERENCES[path] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw


def read(path):
    raw = blob(path)
    return json.loads(lzma.decompress(raw) if path.endswith(".json.xz") else raw)


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
        allow_nan=False)+"\n")


def main():
    pinned_matches = []
    for commit, paths in PINNED.items():
        for path in paths:
            old = subprocess.check_output(["git", "show", commit+":"+path], cwd=ROOT)
            assert old == blob(path), (commit, path)
            pinned_matches.append(dict(commit=commit, path=path, matches_BASE=True))
    # Same normalization as N4-A, not a reimplementation of word selection.
    path = "src/trottertracks/pr_reference_ct0.py"
    raw = blob(path)
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == "positive")
    ns = dict(math=math)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path+":positive only", "exec"), ns)
    positive = ns["positive"]
    conditions = read(CT0+"PRECISION_FEASIBILITY.json")
    phase = read(CT0+"PHASE_SUPPORT.json")
    coverage = read(CT0+"KEY_COVERAGE.json.xz")
    retained = read(D3+"C0_STRICT_HEADROOM.json")
    cache = {r["key"]:r for r in retained["rows"]}
    certificates = read(B4+"ROLE_WORD_CERTIFICATES.json")
    certs = {r["positive_literal_hex"]:r for r in certificates["rows"]}
    parameters = read(D3+"INPUTS.json")["fixed_parameters"]
    budget = read(D3+"STRICT_ERROR_BUDGET.json")
    limit = float.fromhex(parameters["L_hex"])
    assert len(cache) == 299 and len(certs) == 85
    assert len(conditions["rows"]) == 18 and [r["m"] for r in conditions["rows"]] == list(range(18))
    assert all(r["passed"] and r["in_vacuum_domain"] for r in cache.values())
    adopted = read(C4+"RPE_199_SHOT_CHECKS.json")
    assert adopted["all_four_passed"] and adopted["status"] == "SAVED_199_SHOTS_SUFFICIENT"
    assert all(r["m"]==7 and r["q"]==128 and r["r"]==1 for r in adopted["rows"])
    for p in ["JOINT_SYNTHESIS_ERROR.json", "SHELL_ERROR_ACCOUNTING.json", "RAW_DISK_ACCOUNTING.json"]:
        read(C4+p)
    for p in ["COMMON_POLICY_VERIFICATION.json", "LOCAL_CORRECTNESS.json", "SOURCE_INPUTS.json", "STOP.json"]:
        read(B4+p)
    shell = read(C4+"SHELL_ERROR_ACCOUNTING.json")
    # All statements below are static field comparisons, not a new precision
    # choice or a numerical success/shot evaluation.
    rows, per_role = [], []
    for saved in conditions["rows"]:
        m = saved["m"]
        vac = next(r for r in phase["vacuum_phase"] if r["m"] == m)
        event = next(r for r in phase["event_phase"] if r["r"] == saved["r"])
        result = dict(saved_condition=saved, saved_vacuum=vac,
            vacuum_saved_enclosure_within_fixed_domain=vac["absolute_actual_binary_angle_upper"] <= limit,
            vacuum_note="Enclosure containment only; failure is not a reachable outside-angle witness",
            common_all_support_policy_precision_adopted="1e-11 in N4-C" if m==7 else None,
            RPE_status="SAVED_M7_199_SUFFICIENT_CONDITIONAL" if m==7 else "COMMON_POLICY_CT_RPE_ACCOUNTING_NOT_PERFORMED",
            T_status="SAVED_M7_POINTWISE_SUFFICIENT_CONDITION_CLOSED_CONDITIONAL" if m==7 else "ROUND_T_COMPARISON_NOT_EVALUATED",
            shell_structure="S<=1 conditional on the same SOURCE/body/terminal wrapper and compiler C",
            shell_bound_reference=C4+"SHELL_ERROR_ACCOUNTING.json",
            arms={})
        for arm in ("Bprime", "reference"):
            groups = defaultdict(set)
            for r in coverage["role_rows"]:
                if r["m"] == m and r["arm"] == arm:
                    groups[r["role"]].update(positive(k) for k in r["numeric_keys"])
            if arm == "reference":
                groups["actual_SOURCE_inverse_D_superset"].update(positive(k) for k in coverage["inverse_D_supplement"]["mapping"])
            else:
                groups["RTE_event_sign_phase"].update(positive(k) for k in event["keys"])
            union = set().union(*groups.values())
            classifications, known = {}, {}
            for key in union:
                if abs(float.fromhex(key)) > limit:
                    kind = "OUTSIDE_FIXED_PROVIDER_DOMAIN_SAVED_SUPERSET"
                elif key in cache:
                    kind = "C0_FIXED_CERTIFICATE"
                    known[key] = cache[key]["total_upper"]
                elif key in certs:
                    kind = "N4B_SAVED_ROLE_CERTIFICATE"
                    known[key] = certs[key]["positive"]["strict_operator_Frobenius_upper"]
                else:
                    kind = "DICTIONARY_DOMAIN_ONLY_NOT_NEWLY_CALLED"
                classifications[key] = kind
            for role, keys in sorted(groups.items()):
                per_role.append(dict(m=m, arm=arm, role=role,
                    note="saved numeric key superset, NOT physical gate multiplicities or reachability proof",
                    signed_boundary_bookkeeping=(role=="D_boundary_fusion_difference"),
                    positive_key_count=len(keys), classifications=dict(Counter(classifications[k] for k in keys)),
                    domain_only_keys=sorted(k for k in keys if classifications[k]=="DICTIONARY_DOMAIN_ONLY_NOT_NEWLY_CALLED"),
                    outside_saved_superset_keys=sorted(k for k in keys if classifications[k]=="OUTSIDE_FIXED_PROVIDER_DOMAIN_SAVED_SUPERSET")))
            old = saved["arms"][arm]
            axes = {}
            for axis in "XY":
                cap = F(old["axis"][axis]["uniform_gate_error_strict_upper"])
                # Stored logical guards from CT-0 are candidates only, since
                # their emitter terms and C0-priority compatibility are separate.
                logical_candidates = set()
                for rr in coverage["role_rows"]:
                    if rr["m"] != m or rr["arm"] != arm: continue
                    for key in rr["numeric_keys"]:
                        for record in coverage["mapping"][arm].get(key, {}).get("records", []):
                            if record["qualification"] == "STRICT_REUSE_CANDIDATE" and F(record["strict_error_upper"]) <= cap:
                                logical_candidates.add(positive(key))
                axes[axis] = dict(saved_uniform_sufficient_cap=old["axis"][axis]["uniform_gate_error_strict_upper"],
                    fixed_1e11_uniform_guarantee_fits_cap=F(1,10**11)<=cap,
                    saved_dictionary_allocated_bound_fits_cap=F(budget["allocated_uniform_upper"])<=cap,
                    saved_C0_emitter_term_alone_fits_cap=F(budget["C0_emitter_upper"])<=cap,
                    saved_dictionary_emitter_term_alone_fits_cap=F(budget["emitter_upper"])<=cap,
                    known_numeric_certificate_keys_fitting_cap=sum(F(v)<=cap for v in known.values()),
                    known_numeric_certificate_keys_not_fitting_cap=sum(F(v)>cap for v in known.values()),
                    logical_guard_only_candidate_key_count=len(logical_candidates),
                    logical_guard_candidates_not_consumer_or_policy_qualified=True,
                    cap_is_sufficient_not_necessary=True, shell_not_reaccounted_in_N5A=True)
            result["arms"][arm] = dict(saved_numeric_positive_keys=len(union),
                route_counts=dict(Counter(classifications.values())),
                known_numeric_certificate_keys=len(known), axes=axes,
                role_rows_reference="ROLE_METADATA.json#same m/arm",
                total_word_coverage="CONDITIONAL_ON_VACUUM_DOMAIN_AND_COMPILER_CONTEXT; precision separate")
        rows.append(result)
    # Static SOURCE shape: no circuit constructors are executed.
    shape = []
    for path, names in {
        "src/trottertracks/pr_reference_interference_pilot.py": ["wrapper", "reference_core", "symmetric_fused", "fused_steps"],
        "src/trottertracks/pr_reference_bprime.py": ["build_saved", "rewrite_rte"],
        "src/trottertracks/pr_reference_transfer_qualification.py": ["assemble"],
        "src/trottertracks/pr_reference_common_word_consumer.py": ["consume_native"],
    }.items():
        raw = blob(path)
        for n in ast.parse(raw).body:
            if isinstance(n, ast.FunctionDef) and n.name in names:
                text = ast.get_source_segment(raw.decode(), n)
                calls = [x.func.attr for x in ast.walk(n) if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute)]
                if n.name in ("reference_core", "symmetric_fused", "fused_steps", "build_saved", "rewrite_rte", "assemble"):
                    assert not set(calls) & {"measure", "reset", "barrier"}
                shape.append(dict(path=path, function=n.name, sha256=hashlib.sha256(text.encode()).hexdigest()))
    for path in ["src/trottertracks/pr_reference_dyadic_dictionary.py", "src/trottertracks/track_c2_b_verify.py",
        "src/trottertracks/track_c_compiler_a.py", "src/trotterlib/df_rte_qiskit.py", "src/trotterlib/df_partial_s2.py",
        "scripts/account_pr_reference_n4c.py", "src/trottertracks/pr_reference_rotation_v3.py",
        "src/trottertracks/pr_reference_depth_v5.py", "docs/research/pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md",
        "docs/research/pr_reference_ct5b_20261107/CONDITIONAL_T_ADVANTAGE.md",
        "artifacts/pr_reference_n1_numeric/2026-11-08/RESULTS.json", D3+"T_PRICE_BOUND.json"]:
        blob(path)
    for path in ["docs/research/pr_reference_n4c_accounting_20261108/REPORT.md",
        "docs/research/pr_reference_n4b_consumer_20261108/PHASE_ACCOUNTING.md",
        "docs/research/pr_reference_n4b_consumer_20261108/CONSUMER_INTEGRATION.md",
        "docs/research/pr_reference_n3b1b1_dictionary_20261108/DICTIONARY_PROVIDER_SPEC.md",
        "docs/research/pr_reference_n3b1b0_runtime_20261108/EMITTER_BOUND_ASSESSMENT.md"]:
        blob(path)
    save("ROUND_METADATA.json", dict(status="STATIC_SAVED_METADATA_ONLY", base_commit=BASE,
        common_uniform_sufficient_cap_quoted=conditions["common_uniform_gate_error_sufficient_inclusive_upper"],
        rows=rows, precision_selection=False, computed_new_scientific_bounds=False))
    save("ROLE_METADATA.json", dict(rows=per_role, provider_calls=0,
        superset_key_counts_are_not_gate_counts=True, new_reachable_keys_generated=0))
    save("CHECKS.json", dict(status="PASS_STATIC_AUDIT", round_labels_complete=True,
        saved_m7_qualification_preserved=True, shell_SOURCE_structure=shape,
        shell_bound_copied_not_recalculated=shell["epsilon_shell_spectral_upper"],
        classification_only=True, scientific_numerical_reanalyses=0,
        synthesis_guard_provider_build_compile_simulation_sampling_price_calls=0,
        precision_or_policy_adoptions=0, native_resource_study_or_13round_adapter_reimplemented=False,
        host=platform.node(), affinity=len(os.sched_getaffinity(0)), artificial_resource_caps=False))
    save("SOURCE_REFERENCES.json", dict(base_commit=BASE, files=REFERENCES,
        original_commit_alignment=pinned_matches,
        helper="scripts/audit_pr_reference_n5a_static.py",
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reused_function="CT-0 positive only, unchanged AST; following N4-A metadata helper pattern",
        methods="saved-field extraction, schema/hash, set membership, domain/guard/cap metadata comparisons; no science function calls"))
    save("STOP.json", dict(status="MANDATORY_STOP", reason="18-round static audit complete; remaining science execution not authorized",
        next_owner="GPT/user", automatic_science_execution=False, all18_CT_RPE_and_total_T="UNQUALIFIED"))
    print(json.dumps(dict(status="STATIC_AUDIT_COMPLETE", rounds=len(rows), role_groups=len(per_role))))


if __name__ == "__main__":
    main()
