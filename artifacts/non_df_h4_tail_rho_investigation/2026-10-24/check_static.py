#!/usr/bin/env python3
"""Saved bytes, SOURCE and joins only. No real-H4 scientific arithmetic.

Reuses the existing C/L identity checker and the original price-join function.
Does not import the trotterlib package initializer or execute a scientific runner.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = "5600f0e4b28c140a6a50b8aba47425860972a932"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def extract(path, name, namespace):
    nodes = [n for n in ast.parse((ROOT / path).read_bytes()).body
             if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, (path, name)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


def audit():
    # This is the same direct-file import used by verify_h4_cl_transfer.py.
    spec = importlib.util.spec_from_file_location(
        "tail_rho_saved_cl", ROOT / "src/trotterlib/h4_cl_binding.py")
    cl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cl)
    store = cl.TransferStore(ROOT, cl.T.read_json_bytes(
        (ROOT / "artifacts/non_df_h4_cl_transfer/2026-10-23/evidence_manifest.json").read_bytes()))
    local = cl.verify_L_joins(store)  # Exact token equality; no bound arithmetic.

    refs = dict(store.base.manifest["files"])
    refs.update(store.manifest["files"])
    checked = {r["path"]: r["sha256"] for r in refs.values()}
    for path, expected in checked.items():
        assert sha(ROOT / path) == expected, path

    digest = extract("src/trotterlib/isolated_qiskit_execution.py", "digest",
                     dict(json=json, hashlib=hashlib))
    join = extract("src/trotterlib/rte_event_audit.py", "verify_join", dict(digest=digest))
    case_base = ROOT / "artifacts/non_df_h4_pilot/2026-10-23/run01/run"
    join_counts = {}
    old_probability_tokens = {}
    for case in ("pauli_jw_PF", "pauli_jw_PR", "df_native_PF", "df_native_PR"):
        folder = case_base / case
        result_path = folder / "result.json"
        result = read(result_path)
        checked[str(result_path.relative_to(ROOT))] = sha(result_path)
        seen = set()
        for sample in result["circuit_cost"]["samples"]:
            tid = sample["trajectory_id"]
            assert tid not in seen
            seen.add(tid)
            path = folder / "trajectories" / tid / "price_join.json"
            saved = read(path)
            assert join(saved, expected_case=case, expected_binding=result["provenance"]["source"])
            assert saved["compiled_RZ"] == sample["RZ"]
            assert saved["trajectory_record_hash"] == sample["trajectory_record_hash"]
            assert saved["seed"] == sample["seed"]
            assert saved["event_costs_additive"] is False
            checked[str(path.relative_to(ROOT))] = sha(path)
        join_counts[case] = len(seen)
        old_probability_tokens[case] = dict(
            saved_max_probability_difference=result["circuit_cost"]["compiled_signal_max_error"],
            newly_evaluated=False, population_or_uniform_guarantee=False)
    assert join_counts == dict(pauli_jw_PF=1, pauli_jw_PR=256, df_native_PF=1, df_native_PR=256)

    # CNG results and old contracts are only hashed/identified, never re-analysed.
    cng = ROOT / "artifacts/non_df_h4_conditional_cng_results/2026-10-24"
    for rel in ("run01/rare_cost_sensitivity.json", "run01/conditional_functions.json",
                "run01/guard_thresholds.csv", "run01/comparison_regions.json",
                "run01/cost_summary.json", "run01/STOP.json", "verification.json"):
        path = cng / rel
        checked[str(path.relative_to(ROOT))] = sha(path)
    remote = read(OUT / "github_current_verification.json")
    assert len(remote["files"]) == 15
    for row in remote["files"]:
        path = row["path"]
        assert row["matches_fixed_git_blob"] and row["matches_worktree_bytes"]
        assert sha(ROOT / path) == row["sha256"]
        assert git("rev-parse", row["ref"] + ":" + path).decode().strip() == row["git_blob_sha"]
        checked[path] = row["sha256"]

    source_paths = [
        "src/trotterlib/df_rte_qiskit.py", "src/trotterlib/df_partial_s2.py",
        "src/trotterlib/df_partial_s2_repeated.py", "src/trotterlib/df_trotter/ops.py",
        "src/trotterlib/rte.py", "src/trotterlib/rte_compiled_cost.py",
        "src/trotterlib/rte_event_audit.py", "src/trotterlib/fixed_signal_evaluation.py",
        "src/trotterlib/h4_pilot_science.py", "src/trotterlib/h4_legacy_export.py",
        "src/trotterlib/h4_ball_evidence.py", "src/trotterlib/h4_local_enclosure.py",
        "src/trotterlib/h4_transfer_rules.py", "src/trotterlib/h4_conditional_cng.py",
        "src/trotterlib/native_df_gate_model.py"]
    sources = []
    for name in source_paths:
        raw = (ROOT / name).read_bytes()
        assert raw == git("show", BASE + ":" + name), name
        tree = ast.parse(raw)
        sources.append(dict(path=name, sha256=sha(ROOT / name),
            base_git_blob=git("rev-parse", BASE + ":" + name).decode().strip(),
            functions=[dict(name=n.name, line=n.lineno, end_line=n.end_lineno)
                       for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]))
        checked[name] = sha(ROOT / name)

    seal = store.base.document("C_seal")
    suffixes = ["qiskit/transpiler/preset_passmanagers/level1.py",
        "qiskit/transpiler/preset_passmanagers/builtin_plugins.py",
        "qiskit/transpiler/preset_passmanagers/common.py",
        "qiskit/transpiler/passes/optimization/optimize_1q_decomposition.py",
        "qiskit/transpiler/passes/basis/basis_translator.py",
        "qiskit/transpiler/passes/synthesis/high_level_synthesis.py",
        "qiskit/circuit/add_control.py",
        "qiskit/circuit/library/standard_gates/rz.py",
        "qiskit/circuit/library/standard_gates/rzz.py"]
    runtime = []
    for suffix in suffixes:
        paths = [p for p in seal["legacy_runtime_files"] if p.endswith("/" + suffix)]
        assert len(paths) == 1, suffix
        path = paths[0]
        assert sha(path) == seal["legacy_runtime_files"][path], path
        runtime.append(dict(path=path, sha256=sha(path), matches_original_runtime_seal=True))

    # Preserve all three files present before this session's edits.
    inherited = {name: sha(OUT / name) for name in
                 ("approval.txt", "github_verification.json", "next_contracts_DRAFT.json")}
    contract = read(OUT / "next_contracts_DRAFT.json")
    for key in ("execution_authorized", "grant_valid", "dispatch_enabled",
                "N_G_authorized", "formal_certificate_accepted"):
        assert contract[key] is False
    assert contract["grant_id"] is None and contract["SOURCE"] is None
    assert contract["not_a_production_contract"] is True

    for name, expected in checked.items():
        assert sha(ROOT / name) == expected, name
        assert (ROOT / name).read_bytes() == git("show", BASE + ":" + name), name
    forbidden = ("numpy", "scipy", "qiskit", "flint", "pyscf", "openfermion", "trotterlib")
    assert not any(p in sys.modules for p in forbidden)
    unified = ROOT.parent / "research_project_instructions_unified_2026-10-23.md"
    return dict(kind="H4_TAIL_RHO_STATIC_INVESTIGATION_AUDIT", status="PASS_STATIC_ONLY",
        base_commit=BASE, branch=git("branch", "--show-current").decode().strip(),
        L_identity_check=local, old_paired_join_counts=join_counts,
        old_compiled_probability_tokens=old_probability_tokens,
        fixed_evidence_and_source_hashes=checked, source_map=sources,
        runtime_files_actually_read=runtime, preexisting_task_files=inherited,
        github_current_full_byte_checks=len(remote["files"]),
        unified_instructions=dict(path=str(unified), sha256=sha(unified),
            requested_section_10_3_present="### 10.3" in unified.read_text(),
            applied_sections=["8.4", "10", "10.1", "10.2"],
            missing_section_handling="Record discrepancy; follow explicit current user scope and existing-code reuse policy"),
        imported_scientific_packages=[], real_H4_numeric_reanalysis=False,
        compiled_cost_upper=None, actual_rho=None, formal_certificate_accepted=False,
        forbidden_operations=dict(new_scientific_computation=0, sampling=0, RNG_replay=0,
            compile=0, new_probability_or_mean_or_sensitivity=0, N_G_reaccounting=0,
            grants_issued=0, grants_consumed=0, subagents=0),
        unchanged_status=dict(C_L="COMBINATION_BLOCKED", compiler="MISSING_EVIDENCE",
            original_N_G=None, original_native="UNKNOWN"),
        mandatory_STOP=True)


if __name__ == "__main__":
    output = audit()
    (OUT / "static_audit.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status=output["status"], old_joins=output["old_paired_join_counts"],
        protected_files=len(output["fixed_evidence_and_source_hashes"]),
        github_files=output["github_current_full_byte_checks"], real_H4_numeric_reanalysis=False)))
