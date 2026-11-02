"""Assemble preparation-only artifacts from new inventory and synthetic results."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main():
    root, output, temporary = map(Path, sys.argv[1:])
    env = json.loads((output / "environment_inventory_v0.json").read_text())
    static = json.loads((output / "static_audit_v0.json").read_text())
    bench = json.loads((temporary / "benchmark_result_v0.json").read_text())
    fixture = json.loads((temporary / "task_definitions_v0.json").read_text())
    tests = json.loads((temporary / "synthetic_tests_v0.json").read_text())
    operators = json.loads((temporary / "compiled_operator_checks_v0.json").read_text())
    assert bench["status"] == "SYNTHETIC_BENCHMARK_COMPLETE"
    assert tests["status"] == "PURE_SYNTHETIC_WRAPPER_TESTS_PASS"
    assert operators["status"] == "SYNTHETIC_FULL_OPERATOR_CHECKS_PASS"
    assert bench["actual_transpile_calls"] + tests["actual_transpiles"] + operators["actual_transpile_calls"] <= 128
    assert bench["wall_s"] < 1800
    complete = [x for x in bench["conditions"] if x["status"] == "COMPLETE"]
    assert all(x["same_gate_metrics_as_one_worker"] for x in complete)
    fastest = max(complete, key=lambda x: x["tasks_per_min"])
    # Operational proposal: conserve shared CPU while retaining >=90% throughput.
    # This is not a scientific acceptance threshold or production authorization.
    recommended = min(x["workers"] for x in complete if x["tasks_per_min"] >= 0.90 * fastest["tasks_per_min"])
    library = output / "synthetic_fixture"
    library.mkdir(mode=0o700)
    for name in ["synthetic_fixture.py", "inventory.py", "static_audit.py", "build_preparation.py", "validate_preparation.py", "verify_compiled_operator.py", "qiskit_settings.conf", "task_definitions_v0.json", "synthetic_tests_v0.json", "benchmark_result_v0.json", "operator_check_definition_v0.json", "compiled_operator_checks_v0.json"]:
        shutil.copyfile(temporary / name, library / name)
    inventory = json.loads((root / "artifacts/resource_applicability/pr2_pm2_precision_preparation/2026-10-05/candidate_inventory_v1.json").read_text())
    templates = [{"template_id": row["candidate_id"], "method": row["parameters"]["method"],
                  "L_D": row["parameters"]["rank"], "q": row["parameters"]["q"],
                  "r": row["parameters"]["r"], "K": row["parameters"]["K"],
                  "T": 0.8, "delta": 0.8 / row["parameters"]["q"],
                  "new_candidate_fingerprint": None} for row in inventory["development"]]
    compiler = {
        "qiskit_version": env["dependencies"]["qiskit"]["version"],
        "rustworkx_version": env["dependencies"]["rustworkx"]["version"],
        "explicit_options": fixture["compiler"],
        "inherited_defaults": env["qiskit_transpile_defaults"],
        "available_plugins_metadata": env["qiskit_plugin_metadata"],
        "custom_passes": [], "custom_plugin_selected": False,
        "user_settings_policy": "private per-process config; parallel=false, num_processes=1",
        "qiskit_settings_sha256": hashlib.sha256((temporary / "qiskit_settings.conf").read_bytes()).hexdigest(),
        "process_only_environment": fixture["thread_environment"],
        "qiskit_num_processes": 1, "rust_rayon_threads": 1,
        "official_reference": "https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/compiler",
        "same_complete_compiler_as_old_results_established": False,
    }
    compiler["fingerprint"] = fingerprint(compiler)
    environment_identity = {
        "python_version": env["python"]["version"], "python_executable": env["python"]["executable"],
        "dependencies": env["dependencies"], "blas": env["numpy_blas_configuration"],
        "compiler": compiler,
        "original_wheel_archives_verified": False,
        "future_science_runtime_compatibility_validated": False,
    }
    environment_identity["fingerprint"] = fingerprint(environment_identity)
    cache_fields = ["geometry", "hamiltonian_sha256", "df_sha256", "state_sha256", "candidate_fingerprint",
                    "axis", "trajectory_seed", "trajectory_index", "compiler_fingerprint",
                    "environment_fingerprint", "source_commit", "wrapper_semantics"]
    plan = {
        "schema_version": "track_a_h4_geometry_server_zero_compute_plan_v0",
        "status": "SERVER_PREPARATION_COMPLETE_AWAITING_CONTRACT_REVIEW",
        "handoff_commit": "48f9ac3b756bcf572aeb7c594a0c88791a458725",
        "evidence_base_commit": "4c23453c541700c6a41ba71fc5ec9323b53858d6",
        "science_execution_authorized": False, "geometry_frozen": False, "sealed": False,
        "production_source_commit": None, "production_source_hashes": None,
        "production_authorization": None, "production_output_root": None,
        "master_seed": None, "new_snapshot_identities": None,
        "next_stage_authorized": False, "automatic_research_decision_authorized": False,
        "research_decision": None, "mandatory_stop": True,
        "worktree": str(root), "preparation_output": str(output),
        "manuscript_workflow": "PAUSED_PRESERVE_V01_V02_AND_FIGURES",
        "model": {"molecule": "linear_H4", "basis": "STO-3G", "system_qubits": 8,
                  "ancilla_qubit": 8, "wrapper_total_qubits": 9, "T": 0.8,
                  "DF_rank_requested": 12, "DF_rank_actual_required": 12,
                  "charge_proposal": 0, "spin_proposal": 0,
                  "reference": "lowest eigenstate of each geometry's rank-12 DF Hamiltonian in same physical sector as S0",
                  "sector_proposal": {"electrons": 4, "alpha": 2, "beta": 2, "Sz": 0},
                  "target_signal": "exp(-i E_DF T); not exact untruncated chemistry energy",
                  "PF": "second_order_DF_prefix_partial_S2", "RTE": "canonical_finite_RTE"},
        "geometry": {"fixed_distances_angstrom": None, "draft_distances_angstrom": [0.70, 0.80, 0.90, 1.10, 1.40, 1.60],
                     "draft_count": 6, "fresh_blind": False, "existing_1p00_templates": 218,
                     "existing_1p30_templates": 5, "existing_1p30_is_full_grid": False,
                     "auto_geometry_addition": False},
        "candidate_templates": templates, "template_set_fingerprint": fingerprint(templates),
        "candidate_counts": {"total": 218, "random": 194, "deterministic_discard": 24,
                             "by_method": static["template_method_counts"]},
        "sampling": {"trajectories_per_random_cell": 32, "same_trajectory_across_axes": True,
                     "axes": ["cosine", "sine"], "old_r64_selector_reused": False,
                     "fixed_r64_templates": ["B2-rank3-q1-r64-K2", "B3-rank0-q8-r64-K2"],
                     "old_sixteen_cell_selector_reused": False, "adaptive_add_remove": False,
                     "ineligible_retained_for_compile_audit": True, "ineligible_matched_work": None},
        "proposed_resource_caps": {"wrapper_records_per_geometry": 12464,
                                  "six_geometry_wrapper_records": 74784,
                                  "eight_geometry_wrapper_records": 99712,
                                  "signal_evaluations_per_geometry": 218,
                                  "six_geometry_signal_evaluations": 1308,
                                  "actual_transpile_upper_cap_if_no_reuse_six": 74784,
                                  "quantum_shots": 0, "geometry_and_caps_formally_frozen": False,
                                  "additional_96_trajectories": 0},
        "environment_identity": environment_identity,
        "resource_proposal": {"recommended_workers": recommended, "formal_worker_cap": None,
                              "recommendation_rule": "operational proposal: smallest tested pool >=90% of measured maximum synthetic throughput; not a scientific gate",
                              "recommendation_is_not_production_authorization": True,
                              "blas_threads": 1, "qiskit_processes": 1, "rayon_threads": 1,
                              "own_process_nice_proposal": 19, "own_process_AS_cap_GiB_proposal": 4,
                              "observed_CPU_affinity_count": env["available_logical_cpu_count"],
                              "observed_physical_core_count": env["available_physical_core_count"],
                              "shared_resource_recheck_before_any_future_launch": True,
                              "science_worker_memory_profile_unknown": True,
                              "synthetic_AS_cap_is_not_automatically_science_memory_cap": True},
        "snapshot_generation_contract_proposal": {
            "allowed_now": False, "generator_source_commit": None,
            "fixed_geometry_units_and_coordinates_required": True, "SCF_integrals_DF_provenance_required": True,
            "SCF_convergence_mandatory": True, "requested_actual_rank_exactly_12": True,
            "insufficient_rank_or_nonconvergence_rescue": False,
            "DF_fragment_order": "freeze lambda_frobenius_squared policy, tie rule and fragment hashes before prefix split",
            "canonical_state_phase": "largest sector amplitude real positive; freeze tie convention before source implementation",
            "state_normalization_threshold_proposal": 1e-12,
            "DF_ground_state_residual_threshold_proposal": 1e-9,
            "sector_and_Hermiticity_checks_required": True,
            "thresholds_and_solver_details_frozen": False,
            "signal_cost_same_geometry_H_DF_state_source_compiler_match_required": True,
            "old_snapshot_hash_reproduction_assumed": False,
        },
        "seed_key_contract_proposal": {
            "master_seed": None, "seed_policy": "SHA256 of canonical JSON domain-separated campaign/geometry/H/DF/state/template/index/master-seed; use first 64 bits",
            "canonical_distance_representation": "contract-frozen decimal Angstrom string; not yet fixed",
            "axis_in_trajectory_seed": False, "epsilon_in_trajectory_seed": False,
            "axis_in_wrapper_key": True, "duplicate_trajectory_seed_gate": "fail; no silent replacement/retry",
            "candidate_fingerprint_recipe": "geometry/H/DF/state/template/source/compiler/environment/wrapper semantics",
            "old_candidate_fingerprints_copied": False, "execution_task_keys_materialized": False,
        },
        "checkpoint_and_cache_contract_proposal": {
            "wrapper_record_identity_fields": cache_fields,
            "completion_policy": "atomic write+fsync+rename; unique completed wrapper identity; completion ledger binds record digest",
            "ledger_states": ["REGISTERED", "RESERVED", "COMPLETE", "AMBIGUOUS_AWAITING_REVIEW", "FAILED_STOP"],
            "before_actual_transpile": "atomic reservation of a compile invocation budget slot; reservation is not completion",
            "after_actual_transpile": "commit measured metrics/circuit fingerprint/invocation identity then atomically mark complete",
            "uncertain_inflight_invocation": "count as unresolved consumed reservation; stop for budget/identity review; no implicit retry",
            "compile_cache_identity": "same geometry and candidate and axis + exact numerical full-wrapper circuit fingerprint including global phase + compiler/environment/source semantics",
            "different_trajectory_exact_circuit_reuse": "only completed cache entries; retain separate seed/index sample records and cache owner links",
            "cross_geometry_or_cross_cell_reuse": False, "cross_axis_reuse": False,
            "counters": ["logical_wrapper_records", "actual_transpile_invocations", "completed_transpiles", "exact_cache_reuse", "ambiguous_reservations"],
            "old_runtime_or_checkpoints_migrated": False, "implicit_resume_or_retry": False,
        },
        "old_evidence_comparison": {
            "saved_old_records_unchanged": True, "old_1p00_and_1p30_cost_layer": "legacy source/compiler/dependency layer",
            "mix_as_same_condition_points": False,
            "direct_cross_environment_cost_comparison_requires": "complete identity matching established, or separately authorized new-environment anchor",
            "anchor_proposal": "recommended only if direct geometry comparison to historical 1.00 A is required; not needed for comparing the six new points within one new campaign",
            "anchor_automatically_authorized": False, "anchor_wrapper_addition": 12464,
            "six_new_plus_anchor_wrapper_cap": 87248,
            "1p30_full_grid_completion_included": False,
            "synthetic_local_server_speedup_claim": False, "science_campaign_ETA_from_synthetic": None,
        },
        "precision_postprocessing_proposal": {
            "saved_values_only": True, "epsilon_range": [0.005, 0.1], "epsilon_points": 302,
            "display_grid_definition": "0.005*(0.1/0.005)**(i/300), i=0..300; exact endpoints; insert exactly 0.05, sort and deduplicate",
            "alpha_axis": 0.025, "axis_allowance": "epsilon/sqrt(2)-axis_bias",
            "strict_eligibility": "all axis allowances >0; boundary equality is ineligible",
            "axis_shots": "ceil(2*B**2/(epsilon/sqrt(2)-bias_axis)**2*log(2/alpha_axis))",
            "primary": "N_real*E[C_cosine,RZ]+N_imag*E[C_sine,RZ]",
            "metrics": ["rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size"],
            "P_sensitivity": "common P>=0; primary work plus (N_real+N_imag)*P; analytic affine envelope; retain ties",
            "SE": "sqrt((N_real**2*scc+N_imag**2*sss+2*N_real*N_imag*scs)/32); paired axis sample covariance, denominator n-1",
            "engineering_interval": "point +/- 2SE; not formal/familywise CI",
            "ineligible_shots_and_work": None, "missing_values": None, "missing_as_zero": False,
            "epsilon_points_independent_experiments": False, "new_signal_sampling_compile_per_epsilon": False,
            "strict_winner_guarantee": False, "point_pareto": "eligible complete candidates only; all metrics <= and at least one <; preserve ties",
            "geometry_interpolation_as_rigorous_boundary": False, "research_GO_threshold": None,
            "presentation_materiality_and_uncertainty_rule_freeze_required": True,
        },
        "source_before_launch": ["review distance/model/rank/sector/state/seed/budget/reporting rules",
                                 "implement new Track A science module/runner/synthetic tests; keep old source and guards unchanged",
                                 "freeze actual source commit", "generate and review source-bound sealed plan",
                                 "create separate result-prior authorization", "final independent review",
                                 "user explicit launch of fixed command/root/output", "generate new authorized snapshots"],
        "terminal_statuses_proposal": ["GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEW", "IMPLEMENTATION_GATE_FAILED"],
        "zero_science_counters": {key: 0 for key in ["molecular_snapshot_resolve_stat_hash_load", "old_runtime_checkpoint_access", "SCF_DF_reference_solve", "new_science_signals", "science_trajectory_sampling", "science_circuit_build", "science_transpile", "science_wrapper_records", "quantum_shots", "GPU_query_allocation_kernel", "shared_environment_mutations", "other_user_job_mutations", "science_module_runner_execution", "commit_push"]},
    }
    plan["draft_fingerprint"] = fingerprint(plan)
    write_json(output / "zero_compute_plan_draft_v0.json", plan)
    # Bind exact draft contents, including false/null gates; this is not a sealed execution schema.
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "Preparation-only exact draft schema; not an executable sealed plan",
              "type": "object", "additionalProperties": False, "required": list(plan),
              "properties": {key: {"const": value} for key, value in plan.items()}}
    write_json(output / "zero_compute_plan_draft_schema_v0.json", schema)
    checkpoint_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Proposed future wrapper record; schema review only",
        "type": "object", "additionalProperties": False,
        "required": cache_fields + ["status", "metrics", "cache_reuse", "actual_transpile_invocation_id", "mandatory_stop"],
        "properties": {key: {"type": "string", "minLength": 1} for key in cache_fields},
    }
    checkpoint_schema["properties"].update({
        "trajectory_seed": {"type": ["integer", "null"], "minimum": 0},
        "trajectory_index": {"type": "integer", "minimum": 0},
        "status": {"enum": ["REGISTERED", "RESERVED", "COMPLETE", "AMBIGUOUS_AWAITING_REVIEW", "FAILED_STOP"]},
        "metrics": {"type": ["object", "null"]}, "cache_reuse": {"type": "boolean"},
        "actual_transpile_invocation_id": {"type": ["string", "null"]}, "mandatory_stop": {"const": True},
    })
    checkpoint_schema["allOf"] = [
        {"if": {"properties": {"status": {"const": "COMPLETE"}}}, "then": {"properties": {"metrics": {"type": "object"}}}},
        {"if": {"properties": {"status": {"const": "COMPLETE"}, "cache_reuse": {"const": False}}}, "then": {"properties": {"actual_transpile_invocation_id": {"type": "string", "minLength": 1}}}},
    ]
    for key in ["hamiltonian_sha256", "df_sha256", "state_sha256", "candidate_fingerprint", "compiler_fingerprint", "environment_fingerprint"]:
        checkpoint_schema["properties"][key] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    checkpoint_schema["properties"]["source_commit"] = {"type": "string", "pattern": "^[0-9a-f]{40}$"}
    checkpoint_schema["properties"]["axis"] = {"enum": ["cosine", "sine"]}
    metrics = ["rz_count", "rz_depth", "cx_count", "cx_depth", "total_depth", "circuit_size"]
    checkpoint_schema["properties"]["metrics"] = {"type": ["object", "null"], "additionalProperties": False, "required": metrics, "properties": {key: {"type": "integer", "minimum": 0} for key in metrics}}
    checkpoint_schema["properties"]["cache_owner_wrapper_key"] = {"type": ["string", "null"]}
    checkpoint_schema["required"].append("cache_owner_wrapper_key")
    checkpoint_schema["allOf"].append({"if": {"properties": {"status": {"const": "COMPLETE"}, "cache_reuse": {"const": True}}}, "then": {"properties": {"cache_owner_wrapper_key": {"type": "string", "minLength": 1}}}})
    write_json(output / "future_checkpoint_schema_draft_v0.json", checkpoint_schema)
    rows = []
    for condition in complete:
        by_scale = {}
        for scale in ["small", "medium", "large"]:
            selected = [row for row in condition["records"] if row["scale"] == scale]
            by_scale[scale] = {"tasks": len(selected), "mean_build_wall_s": sum(x["build_wall_s"] for x in selected) / len(selected),
                               "mean_transpile_wall_s": sum(x["transpile_wall_s"] for x in selected) / len(selected),
                               "compiled_size_min": min(x["compiled_size"] for x in selected), "compiled_size_max": max(x["compiled_size"] for x in selected),
                               "max_worker_peak_rss_kib": max(x["peak_worker_rss_kib"] for x in selected)}
        rows.append({"workers": condition["workers"], "wall_s": condition["wall_s"], "tasks_per_min": condition["tasks_per_min"],
                     "speedup": condition["speedup_vs_one_worker"], "parallel_efficiency": condition["parallel_efficiency"],
                     "peak_owned_pool_rss_kib": condition["peak_owned_pool_rss_kib"], "scales": by_scale,
                     "thread_count_max": max(x["os_thread_count"] for x in condition["records"]), "failures": len(condition["failures"])})
    summary = {"final_status": "SERVER_NATIVE_ENV_REQUIRES_SOURCE_PORT", "draft_status": plan["status"],
               "recommended_workers": recommended, "worker_cap_frozen": False, "benchmark_conditions": rows,
               "synthetic_actual_transpiles": bench["actual_transpile_calls"] + tests["actual_transpiles"] + operators["actual_transpile_calls"],
               "synthetic_wrapper_records": bench["synthetic_wrapper_records"] + tests["synthetic_wrapper_records"] + len(operators["records"]),
               "compiled_operator_max_residual": max(row["exact_operator_max_absolute_residual"] for row in operators["records"]),
               "science_wrapper_records": 0, "benchmark_wall_s": bench["wall_s"], "fixture_fingerprint": fixture["tasks_fingerprint"],
               "compiler_fingerprint": compiler["fingerprint"], "environment_fingerprint": environment_identity["fingerprint"],
               "shared_environment_changes": 0, "other_user_job_changes": 0, "science_execution_authorized": False,
               "geometry_frozen": False, "mandatory_stop": True, "next_stage_authorized": False}
    write_json(output / "preparation_summary_v0.json", summary)
    table = "\n".join(f"| {x['workers']} | {x['wall_s']:.2f} | {x['tasks_per_min']:.2f} | {x['speedup']:.2f} | {x['parallel_efficiency']:.3f} | {x['peak_owned_pool_rss_kib']/1024:.1f} |" for x in rows)
    sizes = rows[0]["scales"]
    cpu = json.loads(env["cpu_lscpu"]["stdout"])["lscpu"]
    cpu_values = {x["field"].rstrip(":"): x["data"] for x in cpu}
    contract = f"""# Track A H4 geometry server preparation / contract draft v0

`SERVER_NATIVE_ENV_REQUIRES_SOURCE_PORT`

2026-10-06 JST完了（2026-10-05 handoff準備directoryを継続使用）。handoff `{plan['handoff_commit']}` から独立branch/worktreeを作成し、環境記録・静的監査・純synthetic benchmark・契約/schema/zero-compute草案を作成した。準備statusは `{plan['status']}`。science_execution_authorized=false、geometry_frozen=false、sealed=false。本計算・science source実装・commit/pushは行わず、ここでmandatory STOP。

## 既存worktreeと共有環境

作業root：`{root}`。branch：`track-a-h4-geometry-resource-server-prep-20261005`。最初のstatusはclean。
Git tree内にNPZが4件あるため、`git worktree add --no-checkout`後にworktree専用sparse patternsとcommand-scoped `-c core.sparseCheckout=true -c core.sparseCheckoutCone=false`でcheckoutした。NPZ/NPY/pickle/runtime/checkpointは除外し、内容・分子hashへアクセスしていない。共有Git configは変更していない。Git操作はfetch、新branch/worktree登録のみ。

既存mainと既存未追跡 `docs/gpu_execution_environment.md` は保持。venv、package、OS、CUDA/driver、shell/Qiskit共有設定、他ユーザーのprocess・priority・affinityは変更0。専用一時directory内のQiskit設定を自分のprocessにだけ渡し、自分のdriver/workerをnice19にした。

## 環境と比較scope

Python：`{env['python']['executable']}`、3.12.3。NumPy1.26.4、SciPy1.14.1、Qiskit1.3.0、rustworkx0.17.1、OpenFermion1.6.1、OpenFermion-PySCF0.5、PySCF2.7.0。CPU：{cpu_values.get('Model name')}、{cpu_values.get('Socket(s)')} sockets、{env['available_physical_core_count']} physical/allowed logical cores、NUMA {cpu_values.get('NUMA node(s)')}。
RAM total {int(env['memory']['MemTotal'].split()[0])/1024**2:.2f} GiB、available {int(env['memory']['MemAvailable'].split()[0])/1024**2:.2f} GiB、swap {int(env['memory']['SwapTotal'].split()[0])/1024**2:.2f} GiB。開始load1/5/15は {env['loadavg']}。自身のcgroup CPU/memory quotaは観測時max、scheduler環境のjob割当指定はなし。これは観測値であり共有資源の予約ではない。詳細は[environment inventory](environment_inventory_v0.json)。

metadataを優先して依存version・installer・WHEEL・RECORD digestを保存した。元wheel archiveや全installed binaryの独立照合は未実施で、wheel/archive identityはnull。GPU packageはimportしていない。NumPy/QiskitだけCPU importし、BLAS実装・全transpile defaults・plugin entrypointsを記録した。

旧PM-1契約のPython3.11.0rc1 guardと現況3.12.3は一致しない。旧guardを削除せず、次段で新Track A source/契約に分離する。247 sourceをAST parseして構文failure0。legacy Almost_optimal_grouping.pyにはPython3.12のinvalid-escape SyntaxWarningが5件あるが、旧fileを修正していない。構文検査はPySCF/DFのruntimeや数学的同値性の検証ではない。

Qiskitは旧記録と同じversion1.3.0だが、完全なcompiler/dependency/source identityの同一性は未確立。新campaign compiler案はbasis rz/sx/x/cx、opt1、seed17、backend/coupling/layout/routing/targetなし、approximation_degree=1.0、default synthesis、custom pass/pluginなし、num_processes=1。全default/plugin情報とprivate config digestを[plan](zero_compute_plan_draft_v0.json)へ結合した。
Qiskitのmultiprocessing条件とnum_processesは[公式1.3 compiler文書](https://quantum.cloud.ibm.com/docs/en/api/qiskit/1.3/compiler)およびinstalled utils/parallel.py・user_config.pyから確認した。BLAS/OMP/MKL/NumExpr各1、QISKIT_PARALLEL=false、QISKIT_NUM_PROCS=1、RAYON_NUM_THREADS=1。process限定の設定で共有環境は変更しない。

## 保存証拠identity

指定保存4 JSONの内容hashとbase/handoff blob、候補inventory/input identity二件のbyte identityを照合し、全PASS。[static audit](static_audit_v0.json)にexact path/hashを保存した。218 template集合は保存inventoryと完全一致し、B0=20、B1=4、B2=145、B3=49。r64二件は予め含み、旧selectorを再実行しない。旧candidate fingerprintは新geometryへコピーしない。

## 科学データを使わないCPU benchmark

研究DF係数・candidate/trajectory seedを使わない9 qubit/1 classical-bit fixture。ancillaは8、systemは0..7。固定synthetic seedからcontrolled XX+YY Givens network、対角回転、inverse networkを作り、cosine/sine Hadamard wrapperでcompileした。小/中/大は16/64/128 layers、各10 task。全worker条件で同じ30 task、結果を見た規模変更・選抜なし。

| workers | wall s | tasks/min | speedup vs 1 | efficiency | peak own pool RSS MiB |
|---:|---:|---:|---:|---:|---:|
{table}

固定compiled sizeの実測範囲：small {sizes['small']['compiled_size_min']}–{sizes['small']['compiled_size_max']}、medium {sizes['medium']['compiled_size_min']}–{sizes['medium']['compiled_size_max']}、large {sizes['large']['compiled_size_min']}–{sizes['large']['compiled_size_max']}。旧約12k/45k/90kは参考で、規模を合わせる追加transpileはしていない。build/transpile wallはtask別に別記録し、出力gate数・depth・worker RSS/thread数を[benchmark result](synthetic_fixture/benchmark_result_v0.json)、scale別集計を[summary](preparation_summary_v0.json)へ保存した。

benchmark120＋小型意味論検査4＋別に事前定義した全operator照合4＝{summary['synthetic_actual_transpiles']} actual synthetic transpile、synthetic wrapper record {summary['synthetic_wrapper_records']}、science wrapper0。benchmark全wall {bench['wall_s']:.2f}秒、failure/OOM0、全worker条件でgate metrics一致。初期fixtureの124-call capは遵守し、最後の4件を[別定義](synthetic_fixture/operator_check_definition_v0.json)で宣言して利用者の総128-call cap内に収めた。小型3-qubit wrapperは非零global phase・diag(I,U)・bit0→+1/bit1→−1・cosine/sineのaxis意味論を確認し、最大axis残差は1e-12未満。追加のcompile前後全operator最大差は{summary['compiled_operator_max_residual']:.3g}でglobal phase込みで一致した。既存科学tests・全repository testsは実行0。

自分のpool以外の並列を抑制し、1 process AS4 GiB/CPU600秒、総wall30分、共有load32/RAM64 GiB/CPU-pressure5%の停止guardを置いた。guard変更やfailureなら自分のworkerだけ停止しretryなし。実測はサーバー内部scalingで、同じfixtureをローカルでは実行していないためhost間速度倍率や実H4 ETAは示さない。

推奨workerは **{recommended}**。最大throughputの90%以上を満たす最小の測定worker数という共有CPU節約の運用提案であり、科学的受理gateではない。productionのworker capはnullのままで、自動認可しない。science memory profileも未測定のため、syntheticの4 GiB上限を科学計算へそのまま流用しない。future launch直前に共有負荷と利用者の許容資源を再確認する。

fixture/task fingerprint：`{fixture['tasks_fingerprint']}`。source SHA-256は[task definitions](synthetic_fixture/task_definitions_v0.json)に保存。再利用可能source/config/definitions/resultsを新準備directoryだけへ保存した。

## 結果前契約の草案

H4 linear、STO-3G、8 system qubits、requested/actual DF rank12、T=0.8、二次DF-prefix PF、canonical finite-RTE。L_DはB0=3/4/5/6/9、B1=12、B2=3/6/9、B3=0。q=1/2/4/8、delta=0.8/0.4/0.2/0.1、B2/B3 r=1/2/4/8/16/32・K=2/4に固定r64二件。各geometryで同じ218 template。

新規6距離0.70/0.80/0.90/1.10/1.40/1.60 Aは未承認案。geometry_frozen=falseを保つ。1 geometry random194×32×2=12,416＋baseline24×2=48＝12,464 wrapper。6点74,784、8点99,712は上限案であり正式capではない。保存1.30 Aは固定5構成で完全gridに含めない。8点、1.30 A全候補化、追加96、高次PF、H6/H8/H12、energy/RPE、Track Bは自動追加しない。

新snapshotはgeometry座標・単位、SCF収束、rank実数、fragment order/ties、物理sector、phase規約、残差・正規化、生成source/dependencyを新provenanceで固定する。参照はそのDF Hamiltonianの同じsector内の最低固有状態で、exact untruncated chemistry ground energyとは区別する。rank不足/不収束をpadding・別rank・別geometry/状態で救済しない。旧snapshot hash再現を仮定せず、旧状態loadによる環境比較も行わない。

seed/master seed、生成source、solver/threshold/DF ordering detailsは未固定。master seed・新snapshot/source hashはnull。checkpoint keyはgeometry/H/DF/state/candidate/axis/seed/index/compiler/environment/source/wrapper semanticsを全て結合する。logical wrapper ledgerと実transpile invocation/cache hitを分け、atomic complete recordだけ再利用する。同じcandidate・axisの厳密numerical full-wrapper一致だけcache reuseを許し、geometry/cell間reuse・角度差無視は不可。未解決予約は停止し、残予算を確認せず自動retry/resumeしない。

precisionは保存bias/normalization/軸別costだけを用い、PM-2同じ302表示grid epsilon0.005–0.1・alpha_axis0.025。軸headroom=epsilon/sqrt(2)−bias_axis、N_axis=ceil(2 B^2/headroom^2 log(2/alpha_axis))。不適格はshots/work=null、欠測を0にしない。primaryはN_real E[C_cosine,RZ]+N_imag E[C_sine,RZ]、secondary6指標と共通P>=0。paired-axis covarianceから±2SE engineering intervalを保存し、formal CI・独立表示実験・一般的winnerは主張しない。表示materiality/uncertainty・境界規則は新source実装前にreviewする。

## 旧証拠layerとanchor

旧1.00 A/1.30 Aは旧source/compiler/dependency layerで保存し、新6点同士は一つの新identityで比較する。完全identity差がある旧costを同条件のmap点へ混ぜない。新環境1.00 A anchorは新6点間比較には必須でなく、旧1.00 Aとの直接geometry比較が必要なら別予算で提案する。218候補anchorは12,464 wrapper追加、6新点＋anchor87,248。今回は未認可・未実行。1.30 A全候補化も別認可。

## source実装前にreviewすべき未固定条件

1. 利用者が確定する距離list・formal wrapper/worker/memory cap・固定output root。
2. snapshot生成規約：SCF/DF algorithm・rank policy・fragment order/ties・sector/phase・solver/残差threshold。
3. 新campaign master seedとcanonical distance/seed/cache key規則。
4. サーバー3.12.3への新source環境bindingとsemantic gate、全default/pluginを含むcompiler identity。
5. precision表示/materiality/uncertaintyと旧証拠layer、anchorを追加するか。
6. 曖昧reservationの停止、resume/retryの別review、atomic ledgerと実transpile予算管理。

順序は契約review→新science module/runner/synthetic tests→actual source commit→source-bound sealed plan→別authorization→最終review→利用者の明示launch。現状は準備草案であり、production source/authorizationは未作成。

成功terminal案GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEW、failure IMPLEMENTATION_GATE_FAILED、どちらもmandatory STOP、next-stage=false、research_decision=null。準備もここでSTOPする。

## 新規資料と差分

[zero-compute plan](zero_compute_plan_draft_v0.json)、[exact draft schema](zero_compute_plan_draft_schema_v0.json)、[future checkpoint schema draft](future_checkpoint_schema_draft_v0.json)、[static audit](static_audit_v0.json)、[environment inventory](environment_inventory_v0.json)、[summary](preparation_summary_v0.json)、[synthetic source](synthetic_fixture/synthetic_fixture.py)を新規追加した。旧source/test/result/manifest/authorization/原稿/図/研究概要/研究ノートは変更しない。研究判断・科学結果の変更ではなくserver準備記録として別directoryに保存する。

明示science access countersは[plan](zero_compute_plan_draft_v0.json)を参照。分子resolve/stat/hash/load、runtime/checkpoint、科学runner実行、GPU query/allocation/kernel、共有環境変更、他job変更、commit/pushは0。本計算へ自動移行しない。
"""
    (output / "SERVER_PREPARATION_REPORT.md").write_text(contract)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
