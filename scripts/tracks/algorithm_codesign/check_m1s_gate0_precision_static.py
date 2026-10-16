#!/usr/bin/env python3
"""Additional scope/precision checks only; no numerical correctness tests.

Do not import the numerical module, load NPZs, fabricate execution authority or
rerun the previous 30 source/79 document checks. JSON values below test schema
rules, never mathematical outputs from scientific functions.
"""
import ast
import importlib.abc
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
BASE = "83fbac28bca8e308b3039d8532492e6510492f89"
BLOCKED = {"numpy", "scipy", "openfermion", "openfermionpyscf", "pyscf", "qiskit", "numba"}


class BlockScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED or fullname.endswith(".science"):
            raise AssertionError("Scientific import forbidden in this static checker: " + fullname)


def audit(event, args):
    if event == "open" and str(args[0]).endswith(".npz"):
        raise AssertionError("NPZ access forbidden in this static checker")


def main():
    sys.meta_path.insert(0, BlockScience())
    sys.addaudithook(audit)
    from trottertracks.algorithm_codesign.m1s_gate0_v2 import control
    checks = []

    def check(name, condition):
        assert condition, name
        checks.append({"check": name, "status": "PASS"})

    for file, count in (("static_checks.json", 30), ("document_validation.json", 79)):
        path = "artifacts/track_b_m1s_gate0_source_preparation/2026-10-16/v1/" + file
        raw = (ROOT / path).read_bytes()
        check("previous_evidence_retained:" + file,
              json.loads(raw)["passed"] == count and raw == subprocess.check_output(["git", "-C", str(ROOT), "show", BASE + ":" + path]))
    for path in (*control.SOURCE_PATHS, str(Path(__file__).relative_to(ROOT))):
        tree = ast.parse((ROOT / path).read_text(), path)
        compile(tree, path, "exec")  # Syntax only; no execution of code object.
        check("syntax:" + path, True)
    cfg = control.candidate()
    check("limited_input", cfg["input"]["sha256"] == control.INPUT_SHA and cfg["input"]["n_qubits"] == 8 and cfg["input"]["df_rank"] == 12)
    check("three_fixed_splits", cfg["split_ids"] == ["G3", "G6", "G9"])
    check("law_signed_before_abs", cfg["law"]["aggregation"] == "signed_before_abs_lexicographic_IXYZ")
    check("explicit_identity_to_HD", cfg["law"]["identity_policy"] == "move_to_hd")
    check("no_approximate_cutoff", cfg["law"]["coefficient_cutoff"] == 0.0)
    check("oracle_domain_label", cfg["state"]["label"] == "NUMERICAL_SECTOR_GROUND_ORACLE_NOT_THEOREM_CERTIFIED")
    check("no_scientific_GO_STOP", cfg["decisions"]["automatic_GO_STOP"] is False and cfg["decisions"]["rho_threshold"] is None)
    check("all_17_precision_keys", len(control.EPS_KEYS) == 17 and set(cfg["precision"]) == {"dtype", *control.EPS_KEYS})
    for key in control.EPS_KEYS:
        result = control.validate_config(cfg)
        check("null_blocks_execution:" + key, key in result["pending"] and not result["execution_ready"])
        value = cfg["precision"].pop(key)
        check("missing_key_rejected:" + key, not control.validate_config(cfg)["valid_static_shape"])
        cfg["precision"][key] = value
    for key in ("eps_basis", "eps_hermiticity_factors"):
        cfg["precision"][key] = 0
        check("exact_zero_tolerance_supported:" + key, control.validate_config(cfg)["valid_static_shape"])
        cfg["precision"][key] = None
    for key in cfg["limits"]:
        if key not in ("wall_time_seconds", "max_pauli_terms"):
            check("cap_required:" + key, "limits." + key in control.validate_config(cfg)["pending"])
    check("scope_adoption_not_contract_adoption", "scientific_scope_record" in control.validate_config(cfg)["pending"] and "formal_execution_contract_adoption" in control.validate_config(cfg)["pending"])
    source = (ROOT / control.SOURCE_PATHS[2]).read_text()
    tree = ast.parse(source)
    entry = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_authorized")
    check("entry_guard_precedes_numerics", entry.body[1].value.func.id == "require_permit")
    check("raw_intervals_preserved", "raw_W_interval=" in source and "raw_v_interval=" in source)
    check("rho_precision_class_separate", '"RHO_PRECISION_UNRESOLVED"' in source and 'row["raw_rho_in_theoretical_range"]' in source)
    check("exact_checks_avoid_norm_underflow", "state_sector_embedding_exact" in source and "_input_hermiticity_exact" in source and "basis_ladder_exact_" in source)
    check("native_log_and_preimport_context", "os.dup2(log_fd, 2)" in source and '"execution_context.json"' in source)
    control_source = (ROOT / control.SOURCE_PATHS[1]).read_text()
    check("old_authority_schema_rejected", 'auth["schema"] != "m1s_gate0_separate_execution_authorization_v2"' in control_source)
    check("scope_kind_not_execution_authority", "SCOPE_ADOPTION_NOT_EXECUTION_AUTHORIZATION" in control_source)
    for name in ("lambda_R=sumabs",):
        check(name, "lam = math.fsum(abs(coeff) for coeff in rreal.values())" in source)
    check("no_numerical_modules_imported", not any(x.split(".")[0] in BLOCKED or x.endswith(".science") for x in sys.modules))
    print(json.dumps({"schema": "m1s_gate0_precision_additional_static_validation_v1",
                      "passed": len(checks), "checks": checks,
                      "previous_checks_rerun": False, "scientific_numerical_tests": 0,
                      "NPZ_reads": 0, "scientific_module_imports": 0,
                      "numerical_correctness": "UNTESTED_BY_EXPLICIT_SCOPE",
                      "mandatory_STOP": True}, indent=2))


if __name__ == "__main__":
    main()
