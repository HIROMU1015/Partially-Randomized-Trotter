"""N5-B2 saved metadata/SOURCE extraction, stdlib only; no scientific calls.

Reuse the N5-A/B0 fixed-Git extraction pattern and their saved classifications.
Do not execute extracted SOURCE, generate angles/words or recompute any bound.
"""
import ast
import hashlib
import json
import lzma
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "f250bdb99a15804adfe2cfebc0ce16d5d8be6c1e"
OUT = ROOT / "artifacts/pr_reference_n5b2_static/2026-11-08"
B1 = "artifacts/pr_reference_n5b1_scalar/2026-11-08/"
B0 = "artifacts/pr_reference_n5b0_static/2026-11-08/"
N5 = "artifacts/pr_reference_n5a_static/2026-11-08/"
D3 = "artifacts/pr_reference_n3b1b1_dictionary/2026-11-08/"
REFERENCES = {}


def blob(path):
    raw = subprocess.check_output(["git", "show", BASE + ":" + path], cwd=ROOT)
    REFERENCES[path] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return raw


def read(path):
    raw = blob(path)
    return json.loads(lzma.decompress(raw) if path.endswith(".xz") else raw)


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(data, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False) + "\n")


def main():
    assert not OUT.exists(), "Fresh output only"
    m17 = read(B0 + "M17_INPUTS.json")
    evidence = read(B0 + "WORD_EVIDENCE.json")
    roles = [r for r in read(N5 + "ROLE_METADATA.json")["rows"] if r["m"] == 17]
    checks = read(B1 + "CHECKS.json")
    runtime = read(B1 + "LOCAL_RUNTIME_CHECKS.json")
    old_source = read(B1 + "SOURCE_INPUTS.json")
    old_stop = read(B1 + "STOP.json")
    assert checks["joint_fixture_cases"] == runtime["cases"] == len(runtime["rows"]) == 160
    assert runtime["all_passed"] and old_stop["status"] == "MANDATORY_STOP"
    assert hashlib.sha256(blob(old_source["helper"])).hexdigest() == old_source["helper_sha256"]
    coverage = read("artifacts/pr_reference_ct0/2026-11-07/KEY_COVERAGE.json.xz")
    finite = []
    for r in coverage["role_rows"]:
        if r["m"] != 17 or r["arm"] not in ("Bprime", "reference") or r["axis"] != "Y":
            continue
        if r["role"] not in ("RTE_central_rotation", "scalar_or_vacuum_correction"):
            continue
        maps = coverage["mapping"][r["arm"]]
        finite.append(dict(arm=r["arm"], role=r["role"],
            symbolic_keys=r["symbolic_keys"], saved_literal_metadata=[dict(
                signed_key=k, positive_key=maps[k]["positive_key"],
                saved_status=maps[k]["status"], actual_saved_record_keys=[v["key"] for v in maps[k]["records"]])
                for k in r["numeric_keys"]]))
    assert all(not x["actual_saved_record_keys"] for r in finite for x in r["saved_literal_metadata"])
    central = [dict(arm=r["arm"], axes=["X", "Y"],
        signed_literals=[x["signed_key"] for x in r["saved_literal_metadata"]],
        positive_literals=sorted({x["positive_key"] for x in r["saved_literal_metadata"]}))
        for r in finite if r["role"] == "RTE_central_rotation"]
    assert len(central) == 2 and all(len(r["signed_literals"]) == 4 and len(r["positive_literals"]) == 2 for r in central)
    path_functions = []
    wanted = {
        "src/trottertracks/pr_reference_interference_pilot.py": ["reference_core", "wrapper", "vacuum_phase", "directional_right", "symmetric_fused"],
        "src/trottertracks/pr_reference_transfer_qualification.py": ["assemble"],
        "src/trottertracks/pr_reference_bprime.py": ["rewrite_rte", "build_saved"],
        "src/trotterlib/df_rte_qiskit.py": ["QiskitDFRTEEventCircuitBuilder._build", "QiskitDFRTEEventCircuitBuilder._append_basis", "QiskitDFRTEEventCircuitBuilder._append_central_application"],
        "src/trottertracks/track_c_compiler_a.py": ["compile_reference"],
        "src/trottertracks/pr_reference_ct4.py": ["prepare_one"],
        "src/trottertracks/pr_reference_dyadic_dictionary.py": ["round_index", "actual_adjoint", "DictionaryProvider.__init__", "DictionaryProvider.provide", "DictionaryProvider.emit"],
        "src/trottertracks/track_c2_b_verify.py": ["lower_native"],
        "src/trottertracks/pr_reference_common_word_consumer.py": ["consume_native"],
        "src/trottertracks/track_c_h4_synthesis.py": ["exact_word_matrix", "strict_guard"],
        "src/trottertracks/pr_reference_ct0.py": ["key_map", "worst_counts", "geometry"],
    }
    for path, names in wanted.items():
        raw = blob(path)
        parsed = ast.parse(raw)
        found = {}
        for n in parsed.body:
            if isinstance(n, ast.FunctionDef): found[n.name] = n
            elif isinstance(n, ast.ClassDef):
                found.update({n.name + "." + v.name:v for v in n.body if isinstance(v, ast.FunctionDef)})
        for name in names:
            node = found[name]
            path_functions.append(dict(path=path, function=name, line=node.lineno,
                function_source_sha256=hashlib.sha256(ast.get_source_segment(raw.decode(), node).encode()).hexdigest(),
                selected_call_sites=sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)
                    and any(k in ast.unparse(n.func).lower() for k in ("phase", "control", "inverse", "compose", "measure", "reset", "provide", "lower", "compile"))})))
    for path in ["scripts/account_pr_reference_n4c.py", "src/trottertracks/pr_reference_ct1.py",
        "docs/research/pr_reference_ct5b_20261107/STRUCTURAL_CERTIFICATES.md",
        "docs/research/pr_reference_n5b1_scalar_20261108/JOINT_SCALAR_FACTORIZATION.md",
        "docs/research/pr_reference_n5b1_scalar_20261108/SOURCE_PATH_VERIFICATION.md",
        "docs/research/pr_reference_n5b1_scalar_20261108/SIGNAL_INVARIANCE_ASSESSMENT.md",
        "docs/research/pr_reference_n5b0_static_20261108/M17_ERROR_BUDGET_BLOCKERS.md",
        "docs/research/pr_reference_n5b0_static_20261108/ANGLE_DOMAIN_AND_WORD_REUSE.md"]:
        blob(path)
    parameters = read(D3 + "INPUTS.json")["fixed_parameters"]
    manifest = read(D3 + "BASIC_WORDS_MANIFEST.json")
    assert len(manifest["rows"]) == 43
    save("SAVED_METADATA.json", dict(status="STATIC_SAVED_FIELDS_ONLY", m17_condition=m17["condition"],
        vacuum=m17["saved_vacuum"], existing_dictionary=parameters,
        copied_N5A_m17_role_classifications=roles,
        copied_N5B0_guard_metadata_summary=evidence["summary"],
        central_scalar_saved_mapping=finite,
        caveat="A candidate status with an empty records list is NOT a saved word. No angle, guard, sum, price or reachability calculation.",
        N5B1_reused=dict(status=runtime["status"], cases=runtime["cases"], STOP=old_stop)))
    save("NEXT_NUMERIC_INPUTS.json", dict(status="PROPOSED_NOT_EXECUTED_NOT_A_NEW_POLICY",
        base_commit=BASE, fixed_m17_condition=m17["condition"], central_slice=central,
        dictionary_parameters_source=D3 + "INPUTS.json#/fixed_parameters",
        basic_record_manifest=D3 + "BASIC_WORDS_MANIFEST.json", basic_archive=D3 + "basic_records.tar.xz",
        fixed_b=36, fixed_basic_exponents=[-36,6],
        future_calls="Four existing positive literals: reused round_index; selected saved guard sum and exact quantization term only; signed symmetry checks. No provider or word generation.",
        interpretation="Sufficient-bound diagnostic for a finite SOURCE superset slice; not all-support, not an impossibility test or m17 RPE evaluation",
        required_explicit_numeric_permission=True, precision_or_domain_search=False))
    save("SOURCE_INPUTS.json", dict(base_commit=BASE, files=REFERENCES,
        SOURCE_path_functions=path_functions,
        helper="scripts/audit_pr_reference_n5b2_static.py",
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reuse="N5-A/B0 saved classifications and fixed Git extraction pattern; N5-B1 results cited only",
        scientific_modules_imported_or_called=False))
    save("CHECKS.json", dict(status="STATIC_CONNECTION_AUDIT_COMPLETE", reused_N5B1_cases=160,
        new_runtime_fixture_cases=0, old_SOURCE_results_STOP_changed=False,
        new_guard_synthesis_provider_emitter_compile_simulation_RPE_price_calls=0,
        new_error_bounds_or_allocations=0, new_precision_domain_policy_adopted=False,
        all_consumer_path="SOURCE_CONDITIONAL_ON_INPUT_POLICY_AND_TERMINAL_SINK",
        m17_RPE="UNQUALIFIED"))
    save("STOP.json", dict(status="MANDATORY_STOP", reason="Fixed N5-B2 static scope complete",
        m17_RPE="UNQUALIFIED", policy_changed=False, next_owner="GPT/user",
        automatic_numerical_execution=False))
    print(json.dumps(dict(status="STATIC_CONNECTION_AUDIT_COMPLETE", source_function_blocks=len(path_functions),
        saved_m17_roles=len(roles), new_scientific_calls=0)))


if __name__ == "__main__":
    main()
