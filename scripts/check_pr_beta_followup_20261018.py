"""B03 artificial regressions + B08 fixed-metadata tracing; no real science.

Reuse prior audit outcomes; do not rerun its 79-blob/38-check audit or evaluate
saved shots, costs, signals, overlaps, norms, or budget sums.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = "3fd8fca966e6b5a6a972aa71de68b31a0129f5ab"
SCIENCE = "f29eafbf4412193e8e8660d9f051889c2cba684b"
SOURCE_COMMIT = "d87d217a2c4044fbe62e5aa4d3bc44bfb864ff00"
OUT = ROOT / "artifacts/resource_applicability/pr_beta_contract_followup/2026-10-18"
PRIOR = "artifacts/resource_applicability/pr_beta_contract_audit/2026-10-18/"
GRID = "artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json"
WP11 = "artifacts/research_direction_full_opt2/"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def keyword_default(source, function, argument):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == function)
    defaults = {a.arg:d for a,d in zip(node.args.kwonlyargs, node.args.kw_defaults)}
    return ast.literal_eval(defaults[argument])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    cache = {}
    def read(commit, path):
        if path.endswith(".npz"):
            raise ValueError("REAL_MOLECULAR_INPUT_FORBIDDEN")
        key = (commit, path)
        if key not in cache:
            raw = git("show", commit+":"+path)
            cache[key] = raw
            inventory.append(dict(commit=commit, path=path, bytes=len(raw), sha256=sha(raw),
                                  git_blob=git("rev-parse", commit+":"+path).decode().strip()))
        return cache[key]
    def obj(commit, path):
        return json.loads(read(commit, path))

    prior = obj(BASE, PRIOR+"technical_validation.json")
    prior_checks = obj(BASE, PRIOR+"repository_checks.json")
    prior_stop = obj(BASE, PRIOR+"STOP.json")
    # Read/hash only: do not repeat old scenario checks or old synthetic tests.
    read(BASE, PRIOR+"beta_contracts.json")
    prior_index = {(e["commit"], e["path"]):e for e in prior["inventory"]}
    grid = obj(SCIENCE, GRID)
    source_names = [
        "src/trotterlib/df_partial_s2.py", "src/trotterlib/df_rte_tail.py",
        "src/trotterlib/df_partial_randomized_pf.py", "src/trotterlib/finite_rte_signal_validation.py",
        "scripts/run_research_direction_ablation.py", "scripts/run_research_direction_prevalidation.py",
        "src/trotterlib/research_direction_full_opt2.py",
        "src/trotterlib/research_direction_full_opt2_extension_analysis.py",
        "src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py",
        "src/trotterlib/rpe_delta_round_schedule_validation.py",
    ]
    sources = {p:read(SCIENCE,p).decode() for p in source_names}
    for commit,path in list(cache):
        if (commit,path) in prior_index:
            assert sha(cache[(commit,path)]) == prior_index[(commit,path)]["sha256"]
    defaults = dict(
        physical_grid_coefficient_atol=keyword_default(sources["src/trotterlib/finite_rte_signal_validation.py"], "validate_finite_rte_signals", "coefficient_atol"),
        accounting_coefficient_atol=keyword_default(sources["src/trotterlib/df_partial_s2.py"], "prepare_df_partial_s2", "coefficient_atol"),
    )
    assert defaults == dict(physical_grid_coefficient_atol=1e-12, accounting_coefficient_atol=0.)
    providers = []
    for meta in prior["wp04_provider_metadata"]:
        if meta["ld"] != "3":
            continue
        ref = meta["files"]["dataset"]
        raw = read(SCIENCE, ref["path"])
        assert sha(raw) == ref["sha256"]
        d = json.loads(raw)
        providers.append(dict(pair=meta["pair"], path=ref["path"],
                              preparation_fingerprint=d["preparation_fingerprint"],
                              hamiltonian_fingerprint=d["hamiltonian_fingerprint"],
                              partition_fingerprint=d["partition_fingerprint"],
                              compiler_settings=d["compiler_settings"],
                              circuit_scope=d["circuit_scope"]))
    final_path = WP11+"2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json"
    final = obj(SCIENCE, final_path)
    provenance_records = []
    historical_defaults = []
    for label,folder in [
        ("initial", "wp11_all_r_opt2_initial_20260924_023012"),
        ("extension", "wp11_all_r_opt2_fresh32_20260924_232418"),
    ]:
        path = WP11+"2026-09-24/"+folder
        p = obj(SCIENCE,path+"/run_provenance.json")
        a = obj(SCIENCE,path+"/aggregate.json")
        # Public aggregate projections contain result IDs, not worker/prep payloads.
        prep_fields = sorted({k for record in a["records"] for k in record if "preparation" in k})
        checkpoints = git("ls-tree", "-r", "--name-only", SCIENCE, path+"/checkpoints").decode().splitlines()
        provenance_records.append(dict(label=label, git_commit=p["git_commit"],
                                       input_sha256=p["input_sha256"],
                                       source_sha256=p["source_sha256"],
                                       public_task_count=a["task_count"],
                                       public_record_fields=sorted(a["records"][0]),
                                       public_preparation_fields=prep_fields,
                                       tracked_checkpoint_files=checkpoints,
                                       closure_status="RAW_WORKER_PREPARATION_STAMP_MISSING_IN_CHECKED_PUBLIC_PATHS"))
        source_path="src/trotterlib/df_partial_s2.py"
        old=read(p["git_commit"],source_path).decode()
        historical_defaults.append(dict(label=label, git_commit=p["git_commit"], path=source_path,
                                        coefficient_atol_default=keyword_default(old,"prepare_df_partial_s2","coefficient_atol"),
                                        source_sha_recorded_for_this_dependency=source_path in p["source_sha256"]))
    grid_origin=grid["provenance"]["git_commit"]
    for path,func in [("src/trotterlib/finite_rte_signal_validation.py","validate_finite_rte_signals"),
                      ("src/trotterlib/df_partial_s2.py","prepare_df_partial_s2")]:
        old=read(grid_origin,path).decode()
        historical_defaults.append(dict(label="grid_origin",git_commit=grid_origin,path=path,
                                        coefficient_atol_default=keyword_default(old,func,"coefficient_atol"),
                                        source_sha_recorded_for_this_dependency=path in grid["provenance"].get("source_sha256",{})))

    # Exactly the known physical-validation directory and the known long schedule.
    candidate_paths=git("ls-tree","-r","--name-only",SCIENCE,"artifacts/finite_rte_signal_validation").decode().splitlines()
    candidates=[]
    for path in candidate_paths:
        if not path.endswith(".json"):
            continue
        d=obj(SCIENCE,path)
        candidates.append(dict(path=path,request=d.get("request"),
                               preparation_hash=d.get("hamiltonian",{}).get("preparation_hash"),
                               status="OTHER_SAVED_GRID_NOT_ESTABLISHED_AS_WP11_PREPARATION"))
    long_path="artifacts/rpe_delta_round_schedule_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_executed_delta_ca_over_10_round_schedule_v1.json"
    long=obj(SCIENCE,long_path)
    long_projection=[dict(delta_time=v["delta_time"],finite_rte_signal_validation_fingerprint=v["finite_rte_signal_validation_fingerprint"],
                          saved_fields=sorted(v),preparation_hash_saved="preparation_hash" in v) for v in long["selected_schedule_matrix_validations"]]
    joined=[dict(q_m=v["key"][0],r_m=v["key"][1],K_m=v["key"][2],delta_time=.02,
                 status="REUSED_PRIOR_PARAMETER_MATCH_NOT_PREPARATION_IDENTITY") for v in prior["wp11_pr_existing_physical_grid_join"]["rows"]]
    assert len(joined)==18
    investigation=dict(
        schema_version="pr_beta_b08_followup_v1",status="MODEL_MISMATCH",technical_trace_completed=True,
        science_snapshot=SCIENCE,prior_audit=BASE,source_defaults=defaults,
        historical_source_defaults=historical_defaults,
        physical_grid=dict(path=GRID,request=grid["request"],hamiltonian=grid["hamiltonian"],partial_s2=grid["partial_s2"],states=grid["states"],
                           proof_kind="SAVED_NUMERICAL_OBSERVATIONS_NOT_TRANSFER_CERTIFICATE"),
        wp04_cost_datasets=providers,wp11_public_provenance=provenance_records,
        wp11_accounting_preparation=dict(coefficient_atol_source_default=0.,actual_runtime_preparation_hash=None,
                                         evidence_kind="STATIC_SOURCE_RECIPE_WITH_INCOMPLETE_HISTORICAL_CLOSURE"),
        corresponding_rounds=joined,other_known_physical_grids=candidates,long_schedule_projection=long_projection,
        no_matching_physical_grid_established_in_checked_scope=True,
        missing_evidence=["WP11 actual canonical preparation/tail/basis payload and hash tied to its runtime and input",
                          "Physical signal/state/operator evidence for the exact WP11 preparation or a justified transfer bound",
                          "Hash-bound reusable state/operator arrays; saved fingerprints/overlap/norm numbers alone are insufficient"],
        minimal_followup_proposals=[
            dict(kind="METADATA_RECOVERY",scope="Locate already-generated exact-preparation checkpoint/state/operator evidence and verify input/runtime/hash bindings before considering new computation",execution_grant_issued=False),
            dict(kind="NEW_SCIENCE_IF_REQUIRED",scope="After separate approval, validate only the exact WP11 H4 preparation and its 18 selected (q,r,K,delta) physical signal rows; reuse a verified state only if full hash/basis/sector bindings exist; no automatic reoptimization, costs or measurement simulation",execution_grant_issued=False),
        ],
        impact="No saved cost/shot changes or ranking recalculation. Conditional WP11 costs retain their source assumptions; exact-preparation and final RPE success guarantees remain unsupported by this mismatched grid.",
    )

    # Only synthetic tests are executed; all real results above were metadata reads.
    test_path=ROOT/"tests/test_rpe_shot_underflow_20261018.py"
    source_path=ROOT/"src/trotterlib/rpe_resource_accounting.py"
    for path in (test_path,source_path):
        rel=str(path.relative_to(ROOT))
        assert path.read_bytes()==read(SOURCE_COMMIT,rel)
    test_module=load_module(test_path,"_beta_followup_regression")
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(test_module.ShotUnderflowRegression)
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    (OUT/"tests.log").write_text(log.getvalue())
    if not result.wasSuccessful():
        raise AssertionError("B03_REGRESSION_FAILED")
    tests=dict(status="PASS",test_methods=result.testsRun,failures=len(result.failures),errors=len(result.errors),
               skipped=len(result.skipped),existing_certification_assertion_cases=10,
               normal_integer_shot_fixtures=18,ceil_boundary_fixtures=9,normal_candidate_cost_fixtures=16,
               dependency_mode="actual accounting definitions; artificial preparation/cost objects; no scipy/qiskit/pytest imports; existing assertion bodies reused with scalar approx shim",
               before_source=BASE,after_source=SOURCE_COMMIT,source_sha256=sha(source_path.read_bytes()),test_sha256=sha(test_path.read_bytes()))
    counts=prior["saved_contract_summary"]
    boundary=dict(status="DOCUMENTATION_COMPLETED_NO_RECALCULATION",prior_audit_commit=BASE,
                  beta_strict_sum_overshoots=counts["beta_sum_within_represented_budget_without_tolerance:False"],
                  alpha_strict_sum_overshoots=counts["alpha_exact_sum_within_represented_total:False"],
                  saved_scenarios=50,counts_operation="QUOTE_PRIOR_AUDIT_ONLY_NO_BUDGET_SUM_REEVALUATION",
                  guarantee_projection="all_rounds_feasible is not a stored guarantee_status; describe empirical source conditions without adding retrospective certificate fields",
                  wp11_saved_scope=final["scope"])
    stop=dict(status="MANDATORY_STOP",completed_technical_scope=["B03","B08_TRACE","B02_B06_DOCUMENTATION"],
              source_commit=SOURCE_COMMIT,b08_scientifically_resolved=False,new_scientific_execution_authorized=False,
              real_molecular_analysis=0,saved_shot_recalculation=0,cost_reaccounting=0,signal_generation=0,
              sampling=0,native_build_compile=0,measurement_simulation=0,old_audit_stop_preserved=prior_stop)
    provenance=dict(base_commit=BASE,science_snapshot=SCIENCE,new_source_commit=SOURCE_COMMIT,
                    fixed_inputs=inventory,prior_audit_reused=True,prior_79_blob_audit_rerun=False,
                    no_npz_private_runtime_or_grant_originals_opened=True,
                    prior_global_manifest_failure_preserved=prior_checks["base"]["error"]==prior_checks["current"]["error"],
                    publication="LOCAL_COMMITS_ONLY_PENDING_CONFIRMED_EXTERNAL_PUBLICATION_AUTHORIZATION")
    for filename,data in [("b08_investigation.json",investigation),("test_results.json",tests),
                          ("b02_b06_citation_metadata.json",boundary),("STOP.json",stop),("provenance.json",provenance)]:
        (OUT/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(B03=tests["status"],test_methods=tests["test_methods"],B08=investigation["status"],
                          corresponding_parameter_rows=len(joined),fixed_input_blobs=len(inventory),
                          B02_B06=boundary["status"],new_science_calls=0,mandatory_stop=True),indent=2))


if __name__=="__main__":
    main()
