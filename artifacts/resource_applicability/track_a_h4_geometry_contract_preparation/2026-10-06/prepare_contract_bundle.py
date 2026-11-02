"""Generate contract-only artifacts; reads fixed text/JSON and installed metadata."""
import hashlib
import importlib.metadata as md
import json
import pathlib
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
from contract_validator_v1 import fingerprint

HERE = pathlib.Path(__file__).parent
REPO = pathlib.Path.cwd()
PREP = REPO / "artifacts/resource_applicability/track_a_h4_geometry_server_preparation/2026-10-05"
HANDOFF = "2a80f1d5d5e5734e51d970b2b6822cd2543fd596"
BASE = "c2ab34fed49bb1fb104d39fe83b36858a2c92c2a"
STATUS = "H4_GEOMETRY_CONTRACT_REQUIRES_DECISION_SCIENCE_NOT_AUTHORIZED"
DISTANCES = ["0.70", "0.80", "0.90", "1.10", "1.40", "1.60"]
SEMANTICS = "ordinary_controlled_diag(I,U);T=0.8;DF-prefix-S2;canonical-finite-RTE;no-state-prep;bit0=+1;bit1=-1;real=cosine;imag=sine"
DRAFT2020 = "https://json-schema.org/draft/2020-12/schema"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(name, data):
    path = HERE / name
    with path.open("x") as handle:
        handle.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def exact_schema(payload):
    return {"$schema": DRAFT2020, "type": "object", "additionalProperties": False,
            "required": list(payload), "properties": {k: {"const": v} for k, v in payload.items()}}


def main():
    manifest = read(PREP / "preparation_artifact_manifest_v0.json")
    assert digest(PREP / "preparation_artifact_manifest_v0.json") == "33d8a7a5986bd4a29cbe7d678874d54d859614bfab8643b88de2e9f613651741"
    for item in manifest["files"]:
        assert digest(PREP / item["path"]) == item["sha256"]
    old = read(PREP / "zero_compute_plan_draft_v0.json")
    content = {k: v for k, v in old.items() if k != "draft_fingerprint"}
    assert plan_digest(content) == old["draft_fingerprint"]
    assert old["template_set_fingerprint"] == "a939f5602a7840d92d93dfa9be40db40af52dede4483091c649413f480ef03c4"
    for name, target in [
        ("docs/research/gpu_server_track_a_h4_geometry_contract_preparation_prompt.md", "handoff_prompt.md"),
        ("docs/research/track_a_h4_geometry_contract_preparation_scope_v1.json", "scope_v1.json")]:
        data = subprocess.check_output(["git", "show", HANDOFF + ":" + name])
        with (HERE / target).open("xb") as handle:
            handle.write(data)
    scope = read(HERE / "scope_v1.json")
    write("scope_schema_v1.json", exact_schema(scope))
    audit = read(PREP / "static_audit_v0.json")
    sources = audit["source_hashes"]
    evidence = audit["allowed_json_identity"]
    for item in sources + evidence:
        assert digest(REPO / item["path"]) == item["sha256"]
    closure = read(PREP / "dependency_metadata_closure_v0.json")
    differences = []
    observations = {}
    for name, saved in closure["records"].items():
        distribution = md.distribution(name)
        record = distribution.read_text("RECORD")
        current = {"version": distribution.version,
                   "installed_record_sha256": hashlib.sha256(record.encode()).hexdigest() if record else None}
        observations[name] = current
        if any(current[k] != saved[k] for k in current):
            differences.append({"name": name, "saved": saved, "current": current})
    env = read(PREP / "environment_inventory_v0.json")
    plugins = sorted(
        ({"group": ep.group, "name": ep.name, "value": ep.value}
         for ep in md.entry_points() if ep.group.startswith("qiskit.")),
        key=lambda e: (e["group"], e["name"], e["value"]))
    assert sorted(env["qiskit_plugin_metadata"], key=lambda e: (e["group"], e["name"], e["value"])) == plugins
    assert not differences and sys.version == env["python"]["version"]
    write("environment_binding_audit_v1.json", {
        "status": "READ_ONLY_METADATA_BINDING_MATCH", "observed_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        "python_executable": sys.executable, "python_version": sys.version,
        "dependency_metadata_count": len(observations), "dependency_observations": observations,
        "metadata_differences": differences, "qiskit_plugins_match": True,
        "transpile_defaults_reference": env["qiskit_transpile_defaults"],
        "transpile_defaults_reimported": False, "compiler_fingerprint_reference": old["environment_identity"]["compiler"]["fingerprint"],
        "environment_fingerprint_reference": old["environment_identity"]["fingerprint"],
        "science_package_imports": 0, "package_mutations": 0,
        "full_wheel_binary_independent_equivalence_established": False})
    findings = [
        {"source": "src/trotterlib/chemistry_hamiltonian.py", "line": 116,
         "finding": "centered z_i=(i-1.5)*d Angstrom; charge=0, multiplicity=1 for H4"},
        {"source": "src/trotterlib/df_hamiltonian.py", "line": 513,
         "finding": "run_pyscf(run_scf=1,run_fci=0) leaves SCF controls implicit"},
        {"source": "src/trotterlib/pr2_s0_s1_validation.py", "line": 267,
         "finding": "generation_ranked_fragments preserves OpenFermion generated order"},
        {"source": "src/trotterlib/df_partial_randomized_pf.py", "line": 217,
         "finding": "generic ranking sorts |lambda|*||G||_F^2 descending, tie original_index; different from generated order"},
        {"source": "src/trotterlib/pr2_new_series_validation.py", "line": 75,
         "finding": "static frozen layout has 36 sector amplitudes; consistent with alpha=beta=2"},
        {"source": "src/trotterlib/pr2_s0_s1_validation.py", "line": 159,
         "finding": "first maximum-absolute sector amplitude phase pivot"},
        {"source": "src/trotterlib/pr2_matched_accuracy_m1_execution.py", "line": 208,
         "finding": "dense Hermitian eigh, block residual 1e-9; target Rayleigh residual 1e-9 and imaginary energy 1e-11"},
        {"source": "docs/research/pr2_pm2_precision_resource_contract_v1.md",
         "finding": "strict headroom, 302 display points, paired covariance, exact point ties, null ineligible; no winner GO"},
    ]
    write("static_source_findings_v1.json", {"science_source_imports": 0, "findings": findings,
        "dependency_source_findings": {
            "openfermionpyscf/_run_pyscf.py:58-60": "RHF for spin0; ROHF otherwise",
            "openfermion/circuits/low_rank.py:137-153": "generation order uses |lambda|*(sum|G|)^2 with numpy.argsort reversed; rank overrides tolerance",
            "pyscf/scf/hf.py:1664-1667": "config-sensitive defaults conv_tol=1e-9,max_cycle=50"},
        "old_guard_preserved": True, "new_server_native_science_source_required": True})
    decisions = [
        {"id": "D1_SCF_DF", "fixed": False,
         "recommendation": "RHF charge0 spin0, real canonical MO integrals, explicit conv_tol=1e-9/max_cycle=50, no DIIS/newton/restart rescue; OpenFermion1.6.1 spin_basis=True final_rank=12",
         "decision_needed": "approve explicit SCF/initial-guess/DIIS/integral conventions and convergence gates; implicit legacy defaults are not a closed generation contract"},
        {"id": "D2_ORDER_SOLVER_GATES", "fixed": False,
         "recommendation": "preserve generated DF prefix order for old M1 scope; no Frobenius reranking; make l1-weight ties stable by original eigenpair index, freeze degeneracy conventions; Nalpha=Nbeta=2, sector36; dense eigh; first largest-amplitude phase pivot",
         "numeric_gate_proposal": {"state_norm_abs": "1e-12", "reference_residual_L2_Ha": "1e-9",
             "relative_Hermiticity": "1e-12", "minimum_sector_gap_Ha": "1e-10", "imaginary_energy_Ha": "1e-11"},
         "decision_needed": "approve deterministic DF/eigenvector tie and state degeneracy policy (STOP for gap<=threshold) and numerical gates; no degenerate state selection after results"},
        {"id": "D3_MASTER_SEED", "fixed": False, "recommended_master_seed": 20261006,
         "decision_needed": "approve campaign master seed before science source; actual master seed remains null, source/input hashes remain null"},
        {"id": "D4_MEMORY_WALL_OUTPUT", "fixed": False,
         "recommendation": {"per_worker_address_space_GiB": 8, "driver_address_space_GiB": 8,
             "owned_process_memory_GiB": 104, "minimum_host_available_GiB": 64,
             "wall_stop_hours": 72, "output_cap_GiB": 10,
             "output_root_template": "artifacts/resource_applicability/track_a_h4_geometry_execution/<approved-run-id>"},
         "decision_needed": "approve stop ceilings and fixed run ID/root; circuit/compiler working set is not bounded by dense-array estimates, no science ETA or reservation inferred"},
    ]
    write("review_decisions_v1.json", {"status": STATUS, "decisions": decisions, "next_stage_authorized": False})
    caps = scope["future_resource_caps_for_contract"]
    plan = {
        "schema_version": "track-a-h4-contract-zero-compute-v1", "status": STATUS,
        "repository": scope["repository"], "handoff_commit": HANDOFF, "preparation_commit": BASE,
        "distance_scope_frozen": True, "geometry_coordinates_rule": "H_i=(0,0,(i-1.5)*d) Angstrom, i=0..3; decimal d string converted once to binary64; preserve ordered atom IDs",
        "distances_angstrom": DISTANCES, "fresh_blind_claim": False,
        "model": scope["model_for_contract"], "reference_sector_proposal": {"electrons": 4, "alpha": 2, "beta": 2, "dimension": 36},
        "templates": old["candidate_templates"], "template_set_fingerprint": old["template_set_fingerprint"],
        "candidate_counts": old["candidate_counts"], "future_resource_caps": caps,
        "thread_caps": {"BLAS": 1, "OMP": 1, "MKL": 1, "NumExpr": 1, "Rayon": 1, "Qiskit_internal_processes": 1},
        "parallel_policy": "spawned outer pool <=12; stages including snapshot/solver use no nested parallelism; lower worker count or STOP if shared-resource guard fails; no reservation implied",
        "symbolic_slots": [{"distance_angstrom": d, "signal_slots": 218, "random_wrapper_slots": 12416,
            "baseline_wrapper_slots": 48, "logical_wrapper_slots": 12464,
            "actual_source_commit": None, "hamiltonian_sha256": None, "df_sha256": None, "state_sha256": None,
            "slot_state": "SYMBOLIC_NOT_SCIENCE_RECORD"} for d in DISTANCES],
        "compiler_environment_reference": {
            "environment_fingerprint": old["environment_identity"]["fingerprint"],
            "compiler_fingerprint": old["environment_identity"]["compiler"]["fingerprint"],
            "compiler": old["environment_identity"]["compiler"], "metadata_binding_audit": "environment_binding_audit_v1.json"},
        "seed_key_rules": {
            "encoding": "UTF-8 sorted-key compact JSON; integer literals; floats as canonical float.hex tagged real64_hex; complex as [real_hex,imag_hex]; signed zero retained; NaN/infinity/symbolic rejected",
            "trajectory": "SHA256(domain=h4-trajectory-v1,master_seed,geometry,H,DF,state,template,source,compiler,environment,wrapper_semantics,index); first8 bytes unsigned big-endian; no axis/epsilon",
            "step_occurrence": "future sampler domain h4-step-occurrence-v1, parent trajectory seed, outer_step=0..q-1, short_step=0..r-1, occurrence=0..K-1, draw_kind; independent derived RNG stream per occurrence; no axis/epsilon",
            "duplicate_seed": "STOP; no replacement seed, extension or implicit retry",
            "wrapper_key": "SHA256(domain=h4-wrapper-v1,all identity fields including axis,trajectory_seed,index); baseline seed/index both null",
            "sample_weight": "random each 1/32; baseline 1; reuse never deduplicates logical samples"},
        "numerical_circuit_rules": {
            "format": "ordered-numerical-full-circuit-v1", "before_compiler": True,
            "identity": "all exact parameters and global_phase, ordered bit identities/operands, axis and measurement, instruction identities, conditions/control state and full custom definitions; no skeleton/rounded keys",
            "serializer_note": "declarative JSON contract only; production Qiskit serializer and operation round-trip tests require later science source review",
            "record_digest": "SHA256(domain=h4-completion-record-v1,whole record) stored only in external completion ledger"},
        "checkpoint_schema": "checkpoint_schema_v1.json", "ledger_schema": "completion_ledger_schema_v1.json",
        "cache_policy": "COMPLETE noncached owner only; same geometry/candidate/axis/H/DF/state/source/compiler/environment/semantics and exact numerical fingerprint; owner link and each sample weight retained",
        "atomic_policy": "future exclusive one-run registration; reserve invocation before compile using atomic replace+fsync and lock; record fsync+rename then ledger commit+directory fsync; interruption between writes => AMBIGUOUS/STOP, never implicit resume/retry",
        "future_accounting": ["logical_slots", "reserved_compile_invocations", "completed_actual_transpiles", "cache_reuse_records", "ambiguous_consumed_reservations", "failed_consumed_reservations"],
        "precision_rules": old["precision_postprocessing_proposal"],
        "reporting_rules": {
            "ties": "exact point ties retained; sorted template IDs for display only; no tolerance winner threshold",
            "uncertainty": "paired n=32, ddof=1, covariance retained; +/-2SE engineering only; overlap labelled unresolved; deterministic variance0 only with actual deterministic evidence; missing random variance null",
            "precision_boundary": "epsilon_min=sqrt(2)*max(axis_bias); strict epsilon>epsilon_min and each headroom>0; display-grid brackets only for cost switches",
            "geometry_boundary": "six sampled distances only; no interpolated/extrapolated certified boundary",
            "missing": "ineligible/incomplete metrics, shots and matched work null; not zero; all candidates retained for audit; point Pareto only eligible complete coverage",
            "P": "common P>=0 added to primary RZ only; analytic affine envelope; exact intersection ties; infinite upper endpoint null",
            "GO_winner_threshold": None},
        "freeze_barriers": ["contract decisions closed", "new source implemented and actual commit frozen under separate instruction",
            "new H/DF/state/sector/ordering arrays frozen after authorized generation; no signal/cost before freeze",
            "signal/cost identity match and semantic gates", "source/input-bound sealed plan",
            "separate result-prior authorization and independent final review", "explicit one-run launch"],
        "old_evidence": {"one_A_templates": 218, "one_point_three_A_templates": 5, "separate_identity_layer": True,
            "anchor_authorized": False, "eight_geometries_authorized": False},
        "unresolved_decisions": [d["id"] for d in decisions],
        "science_execution_authorized": False, "execution_plan_sealed": False,
        "contract_conditions_fully_closed": False, "actual_science_source_commit": None,
        "actual_science_source_hashes": None, "new_snapshot_identities": None,
        "master_seed": None, "execution_authorization": None,
        "science_memory_cap": None, "science_wall_cap": None, "science_output_root": None,
        "runtime_cache_registry_created": False, "next_stage_authorized": False,
        "automatic_research_decision_authorized": False, "research_decision": None, "mandatory_stop": True,
        "future_terminal_statuses": ["GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEW", "IMPLEMENTATION_GATE_FAILED"],
        "current_execution_counts": dict.fromkeys(["molecular_access", "molecular_generation", "SCF_DF_solve",
            "signal", "sampling", "science_circuit_build", "science_transpile", "additional_synthetic_transpile",
            "runtime_checkpoint_cache_access", "GPU", "shared_environment_changes", "other_job_changes", "commit", "push"], 0)}
    # Compiler definition is stored under the environment identity in the published draft.
    plan["plan_fingerprint"] = plan_digest(plan)
    write("zero_compute_plan_v1.json", plan)
    write("plan_schema_v1.json", exact_schema(plan))
    h64 = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    null = {"type": "null"}
    number = {"type": "object", "additionalProperties": False, "oneOf": [
        {"required": ["real64_hex"], "properties": {"real64_hex": {"type": "string"}}},
        {"required": ["complex128_hex"], "properties": {"complex128_hex": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}}}],
        "properties": {"real64_hex": {"type": "string"}, "complex128_hex": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}}}
    circuit = {"$schema": DRAFT2020, "type": "object", "additionalProperties": False,
        "required": ["format", "axis", "qubits", "clbits", "global_phase", "instructions"],
        "properties": {"format": {"const": "ordered-numerical-full-circuit-v1"}, "axis": {"enum": ["cosine", "sine"]},
            "qubits": {"type": "array", "items": {"type": "string"}, "minItems": 9, "maxItems": 9, "uniqueItems": True},
            "clbits": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 1},
            "global_phase": number, "instructions": {"type": "array", "minItems": 1, "items": {"type": "object"}}}}
    write("numerical_circuit_schema_v1.json", circuit)
    from contract_validator_v1 import IDENTITY_FIELDS, METRICS
    props = {k: h64 for k in ["hamiltonian_sha256", "df_sha256", "state_sha256", "candidate_fingerprint", "compiler_fingerprint", "environment_fingerprint", "wrapper_key"]}
    props.update({
        "schema_version": {"const": "h4-wrapper-record-v1"},
        "artifact_scope": {"enum": ["SYNTHETIC_CONTRACT_TEST_ONLY", "FUTURE_SCIENCE_RECORD"]},
        "geometry": {"enum": DISTANCES},
        "candidate_template": {"type": "object", "additionalProperties": False,
            "required": ["template_id", "method", "L_D", "q", "r", "K"],
            "properties": {"template_id": {"type": "string"}, "method": {"enum": ["B0", "B1", "B2", "B3"]},
                **{k: {"type": "integer", "minimum": 0} for k in ["L_D", "q", "r", "K"]}}},
        "axis": {"enum": ["cosine", "sine"]},
        "trajectory_seed": {"type": ["integer", "null"], "minimum": 0, "maximum": 2**64 - 1},
        "trajectory_index": {"type": ["integer", "null"], "minimum": 0, "maximum": 31},
        "source_commit": {"type": "string", "pattern": "^[0-9a-f]{40}$"},
        "wrapper_semantics": {"const": SEMANTICS},
        "numerical_circuit_fingerprint": {"anyOf": [h64, null]},
        "status": {"enum": ["REGISTERED", "RESERVED", "COMPLETE", "AMBIGUOUS_AWAITING_REVIEW", "FAILED_STOP"]},
        "metrics": {"anyOf": [null, {"type": "object", "additionalProperties": False, "required": list(METRICS),
            "properties": {k: {"type": "integer", "minimum": 0} for k in METRICS}}]},
        "cache_reuse": {"type": "boolean"}, "cache_owner_wrapper_key": {"anyOf": [h64, null]},
        "actual_transpile_invocation_id": {"type": ["string", "null"], "minLength": 1},
        "sample_weight": {"type": "object", "additionalProperties": False, "required": ["numerator", "denominator"],
            "properties": {"numerator": {"const": 1}, "denominator": {"enum": [1, 32]}}},
        "mandatory_stop": {"const": True}})
    checkpoint = {"$schema": DRAFT2020, "type": "object", "additionalProperties": False,
        "required": list(props), "properties": props, "allOf": [
            {"if": {"properties": {"status": {"const": "COMPLETE"}}},
             "then": {"properties": {"numerical_circuit_fingerprint": h64, "metrics": {"type": "object"}}}},
            {"if": {"properties": {"status": {"enum": ["REGISTERED", "RESERVED", "AMBIGUOUS_AWAITING_REVIEW", "FAILED_STOP"]}}},
             "then": {"properties": {"metrics": null, "cache_reuse": {"const": False}, "cache_owner_wrapper_key": null}}},
            {"if": {"properties": {"cache_reuse": {"const": True}}},
             "then": {"properties": {"status": {"const": "COMPLETE"}, "cache_owner_wrapper_key": h64, "actual_transpile_invocation_id": null}}},
            {"if": {"properties": {"cache_reuse": {"const": False}, "status": {"const": "COMPLETE"}}},
             "then": {"properties": {"cache_owner_wrapper_key": null, "actual_transpile_invocation_id": {"type": "string", "minLength": 1}}}},
            {"if": {"properties": {"status": {"const": "RESERVED"}}},
             "then": {"properties": {"actual_transpile_invocation_id": {"type": "string", "minLength": 1}}}}]}
    write("checkpoint_schema_v1.json", checkpoint)
    ledger = {"$schema": DRAFT2020, "type": "object", "additionalProperties": False,
        "required": ["schema_version", "entries", "reservations", "mandatory_stop"],
        "properties": {"schema_version": {"const": "h4-external-completion-ledger-v1"}, "mandatory_stop": {"const": True},
            "entries": {"type": "object", "maxProperties": 74784, "propertyNames": h64,
                "additionalProperties": {"type": "object", "additionalProperties": False,
                    "required": ["status", "record_sha256"], "properties": {"status": {"const": "COMPLETE"}, "record_sha256": h64}}},
            "reservations": {"type": "object", "maxProperties": 74784,
                "additionalProperties": {"type": "object", "additionalProperties": False,
                    "required": ["status", "wrapper_key"], "properties": {"status": {"enum": ["RESERVED", "COMPLETE", "AMBIGUOUS_AWAITING_REVIEW", "FAILED_STOP"]}, "wrapper_key": h64}}}}}
    write("completion_ledger_schema_v1.json", ledger)
    result = {"$schema": DRAFT2020, "type": "object", "additionalProperties": False,
        "required": ["schema_version", "status", "source_commit", "plan_fingerprint", "completion_ledger_sha256", "signal_records", "logical_wrapper_records", "actual_transpile_invocations", "research_decision", "next_stage_authorized", "automatic_research_decision_authorized", "mandatory_stop"],
        "properties": {"schema_version": {"const": "h4-geometry-result-v1"},
            "status": {"enum": plan["future_terminal_statuses"]}, "source_commit": props["source_commit"],
            "plan_fingerprint": h64, "completion_ledger_sha256": h64,
            "signal_records": {"type": "integer", "minimum": 0, "maximum": 1308},
            "logical_wrapper_records": {"type": "integer", "minimum": 0, "maximum": 74784},
            "actual_transpile_invocations": {"type": "integer", "minimum": 0, "maximum": 74784},
            "research_decision": null, "next_stage_authorized": {"const": False},
            "automatic_research_decision_authorized": {"const": False}, "mandatory_stop": {"const": True}},
        "allOf": [{"if": {"properties": {"status": {"const": "GEOMETRY_PRECISION_MAP_COMPLETE_AWAITING_REVIEW"}}},
                   "then": {"properties": {"signal_records": {"const": 1308}, "logical_wrapper_records": {"const": 74784}}}}]}
    write("result_schema_v1.json", result)
    write("identity_access_audit_v1.json", {
        "handoff_commit": HANDOFF, "preparation_commit": BASE,
        "preparation_manifest_sha256": digest(PREP / "preparation_artifact_manifest_v0.json"),
        "published_preparation_file_count_unchanged": 25,
        "old_source_count_unchanged": len(sources), "allowed_evidence_json_count_unchanged": len(evidence),
        "handoff_prompt_sha256": digest(HERE / "handoff_prompt.md"),
        "handoff_scope_sha256": digest(HERE / "scope_v1.json"),
        "current_execution_counts": plan["current_execution_counts"],
        "access_scope": "fixed source text, published lightweight JSON, installed metadata only; excluded input paths enumerated by Git metadata only",
        "science_inputs_materialized": False, "private_sparse_worktree_only": True,
        "shared_git_config_changes": 0, "contract_local_only": True})
    print(json.dumps({"status": STATUS, "schemas": 6, "metadata_dependencies": 45, "science_counts": plan["current_execution_counts"]}))


if __name__ == "__main__":
    main()
