"""Prepare review-only JSON/text; never import or execute molecular source."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
OLD = HERE.parent / "2026-10-06"
BASE = "7c1a3d43f61c5501a9e79206b7c60933f94b1077"
STATUS = "H4_GEOMETRY_CONTRACT_V2_PREPARED_AWAITING_REVIEW_SCIENCE_NOT_AUTHORIZED"
PROJECT = "/home/AbeHiromu/projects/partially-randomized-trotter"
RUN_ID = "track-a-h4-geometry-v2-20261006-run01"
GIB = 2**30
STAGES = [
    "CONTRACT_V2_AWAITING_REVIEW", "CONTRACT_CONDITIONS_APPROVED",
    "SCIENCE_SOURCE_IMPLEMENTED", "SCIENCE_SOURCE_FROZEN",
    "INPUT_GENERATION_PLAN_READY", "INPUT_GENERATION_AUTHORIZATION_REVIEWED",
    "INPUT_GENERATION_LAUNCH_APPROVED", "INPUTS_FROZEN_STOP",
    "INPUT_BOUND_SIGNAL_PLAN_SEALED", "SIGNAL_AUTHORIZATION_REVIEWED",
    "SIGNAL_LAUNCH_APPROVED", "MAP_COMPLETE_STOP",
]
ACTIONS = ["approve_contract", "implement_source", "freeze_source",
           "prepare_generation_plan", "review_generation_authorization",
           "explicit_generation_launch", "generate_freeze_inputs_then_stop",
           "seal_input_bound_signal_plan", "review_signal_authorization",
           "explicit_signal_launch", "signal_compile_map_then_stop"]


def write(name, data):
    with (HERE / name).open("x") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def exact_schema(data, title):
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": title,
            "type": "object", "additionalProperties": False, "required": list(data),
            "properties": {k: {"const": v} for k, v in data.items()}}


def main():
    old = json.loads((OLD / "zero_compute_plan_v1.json").read_text())
    assert hashlib.sha256((OLD / "zero_compute_plan_v1.json").read_bytes()).hexdigest() == "805874dcbe1466adbd92f2300b90a88a2e111e8d43543759fae4b28a9eb1d758"
    assert old["plan_fingerprint"] == "09ce44081a5a20f2e1f3dba2fced3bf8d8732251ff88393e8a190d0fa820bb88"
    assert hashlib.sha256((OLD / "artifact_manifest_v1.json").read_bytes()).hexdigest() == "68cc9c28faaf6ab1b43799d5ffc18ece633d244634f5146239dbc93770c530f7"
    d1 = {
        "status": "REVIEW_PROPOSAL_NOT_APPROVED", "method": "RHF", "charge": 0,
        "multiplicity": 1, "spin": 0, "basis": "STO-3G", "unit": "Angstrom",
        "symmetry": False, "initial_guess": "minao", "dm0": None,
        "conv_tol_Ha": 1e-9, "max_cycle": 50,
        "conv_tol_grad": 10**(-4.5), "tight_gradient_norm": True,
        "check_convergence": None, "conv_check": True,
        "ordinary_DIIS": {"enabled": True, "class": "pyscf.scf.diis.CDIIS",
                          "space": 8, "start_cycle_zero_based": 1, "damp": 0,
                          "rollback": 0, "file": None},
        "damp": 0, "level_shift": 0, "direct_scf": True, "direct_scf_tol": 1e-13,
        "pyscf_advisory_max_memory_MB": 4000, "cart": False, "incore_anyway": False,
        "checkpoint": None,
        "convergence_acceptance": "mf.converged AND a main cycle has abs(delta_E)<1e-9 and raw orbital-gradient L2<sqrt(1e-9); after normal extra confirmation, final raw gradient<sqrt(1e-9), finite energies/integrals/MOs. Built-in relaxed extra-cycle OR is not sufficient by itself.",
        "guess_fallback": "minao internal atom fallback, unavailable projection/basis, missing source verification => STOP; no alternative guess after results",
        "rescue": {"Newton": False, "restart": False, "different_guess": False,
                   "change_DIIS_after_failure": False, "cycle_extension": False},
        "coordinates": old["geometry_coordinates_rule"],
        "integrals": "real nonrelativistic all-electron RHF; AO hcore=T+V_nuc, overlap S, Coulomb ERI; C.T@hcore@C; ao2mo.kernel then restore(1), transpose(0,2,3,1) to OpenFermion (ps|qr); include nuclear repulsion; all 4 spatial orbitals, no frozen core/active-space reduction",
        "MO_order_phase": "ascending canonical orbital energies; PySCF generalized scipy.linalg.eigh(h,S), first maximum-absolute real AO component positive; no localization or geometry tracking. MO degeneracy with relative gap<=1e-12 => STOP, no eigenspace rotation",
        "spin_orbitals": "2p=alpha,2p+1=beta; OpenFermion spinorb_from_spatial incl EQ_TOLERANCE=1e-8 zeroing, InteractionOperator two-body factor 1/2; no new screening",
        "DF": {"package": "openfermion", "version": "1.6.1", "spin_basis": True,
               "final_rank": 12, "truncation_threshold": 1e-8,
               "tolerance_overridden_by_final_rank": True,
               "source_tensor_symmetry_absolute_L1": 1e-8,
               "rank_acceptance": "returned lambda/G count exactly12, finite arrays; actual rank means returned fragment count, not number of eigenvalues above a numerical threshold; retain small/zero lambda unchanged; no padding, rank/threshold/geometry rescue",
               "rank_shortfall": "STOP"},
        "untested": "proposed explicit controls are source-audited, not molecularly exercised; existing run_pyscf saves data and hides controls, so later separate source must expose controls and disable implicit checkpoint/save",
    }
    d2 = {
        "status": "REVIEW_PROPOSAL_NOT_APPROVED",
        "fragment_order": "preserve OpenFermion generated order exactly; S0/M1 generation_ranked_fragments enumerates it; no generic Frobenius reranking",
        "generation_weight": "abs(lambda)*(sum(abs(G_spin)))**2",
        "weight_tie": "retain installed numpy.argsort(weights)[::-1] emitted order and full returned permutation; do not claim stable equal-weight tie sorting. Exact equal nonzero-significant weights crossing any planned prefix 3/4/5/6/9/12 => STOP before science signal/cost; null ties and other ties retain emitted index, frozen once with no rerun",
        "DF_sign": "first maximum-absolute C-flattened real spatial eigenvector element positive; same sign applied to G; record raw eigenpair index and returned permutation; G squared unchanged, new bytes not claimed identical to legacy",
        "DF_degeneracy": "using full 16 eigenpairs, relative gap |lambda_i-lambda_j|/max(1,max|lambda|)<=1e-12, at least one pair retained and at least one |lambda|/max(1,max|lambda|)>1e-12 => STOP. Null eigenspaces retain the pinned solver single invocation basis/permutation, sign-canonicalized and frozen; never rotate/rerun/substitute, and retain exact small coefficients. No cross-host basis reproducibility claimed",
        "degeneracy_relative_threshold": 1e-12,
        "sector": {"electrons": 4, "Nalpha": 2, "Nbeta": 2, "dimension": 36,
                   "basis": "OpenFermion JW spin-orbital bit p at integer bit (7-p); choose alpha2/beta2 occupations, ascending 8-bit integer indices; Qiskit state conversion is explicit 8-bit reversal, qubit p little-endian; no reordered sector after results"},
        "solver": {"name": "numpy.linalg.eigh", "UPLO": "L", "dtype": "complex128",
                   "matrix": "dense 36x36 sector DF H; check original Hermiticity then use (H+H.conj().T)/2; ascending eigenvalues; choose index0 only"},
        "phase": "first maximum-absolute component in ordered sector basis; multiply by exp(-i*angle(pivot)) and ensure positive real pivot; apply same phase to full state; no selection among degenerate states",
        "gates": {"state_norm_absolute_max": 1e-12,
                  "reference_residual_L2_Ha_max": 1e-9,
                  "relative_Hermiticity_max": 1e-12,
                  "imaginary_energy_Ha_max": 1e-11,
                  "minimum_sector_gap_Ha_STOP_if_le": 1e-10},
        "norm_definitions": "abs(norm2(psi)-1); residual against original H and Rayleigh real E; Hermiticity spectral-norm(H-H†)/max(1,spectral-norm(H)); abs(Im(psi†Hpsi)); gap=E1-E0. Outside-sector amplitude and full/sector consistency absolute max<=1e-12",
        "gate_failure": "STOP; no state/prefix/rank substitution, threshold relaxation, retry or resume",
    }
    d3 = {"status": "REVIEW_PROPOSAL_NOT_APPROVED", "proposed_master_seed": 20261006,
          "actual_master_seed": None, "seed_rules": copy.deepcopy(old["seed_key_rules"]),
          "actual_trajectory_seeds_created": 0,
          "rule": "derive science seeds only after actual source/input identities fixed and separate science launch; no placeholder hash seed fixation; artificial tests reuse seed7 records only"}
    d4 = {
        "status": "REVIEW_PROPOSAL_NOT_APPROVED", "unit_bytes_per_GiB": GIB,
        "worker_address_space_budget_bytes": 8*GIB, "driver_address_space_budget_bytes": 8*GIB,
        "worker_RSS_stop_bytes": 8*GIB, "driver_RSS_stop_bytes": 8*GIB,
        "fixed_host_headroom_bytes": 16*GIB, "worker_max": 12, "worker_min": 1,
        "owned_address_space_sum_at_max_bytes": 104*GIB,
        "required_available_at_max_bytes": 120*GIB,
        "required_available_formula": "driver_budget + w * worker_budget + fixed_host_headroom",
        "selection": "largest integer w in [1,min(requested,12,allowed_cpu_count)] satisfying effective_available_bytes>=required_available(w); reduce before launch only; if w=1 cannot fit STOP",
        "admission_observation": "fresh<=5s host /proc/meminfo MemAvailable (kB*1024); effective available=min(host,finite current cgroup memory.max-memory.current). Missing/malformed/negative/stale values STOP. /proc/self/status Cpus_allowed_list intersects explicitly permitted CPU set; permission count null now, cannot infer permission from nproc",
        "RSS_vs_AS": "future own-process RLIMIT_AS hard/soft8GiB for each worker and driver, not host/cgroup change; RSS separately from own /proc/<pid>/status VmRSS and tree, monitor every<=5s; sum budget104GiB is neither RSS guarantee nor reserved physical memory; advisory PySCF max_memory4000MB is separate",
        "pressure_stop": "host/effective available<16GiB, own RSS>per-role budget or total admitted own budget, allocation/AS limit failure, memory PSI full avg10>0, or cgroup OOM event delta>0 => stop own run, preserve audit and STOP for review. No automatic worker restart/retry/resume or host swap/cgroup/job change",
        "wall_stop_seconds": 72*3600, "wall_clock": "time.monotonic from first stage launch; total generation+map<=72h, persisted consumed seconds, downtime not reset; wall is stop ceiling not ETA",
        "output_cap_bytes": 10*GIB,
        "output_accounting": "all own-run snapshots/logs/records/manifests/temp files included; reserve bytes before write, stop if next write exceeds10GiB; symlink escapes rejected; no output eviction to free budget or overwritten evidence",
        "proposed_run_id": RUN_ID, "proposed_absolute_project_root": PROJECT,
        "proposed_absolute_output_root": PROJECT+"/artifacts/resource_applicability/track_a_h4_geometry_execution/"+RUN_ID,
        "run_registration": "future exclusive fixed run ID, no existing output overwrite; generation closes at frozen inputs STOP, separate map launch references same frozen inputs with different authorization; no automatic resume",
        "output_directory_created": False, "registry_created": False,
        "resource_reservation_claimed": False, "shared_environment_or_other_jobs_changed": False,
    }
    decisions = {"schema_version": "h4-contract-review-decisions-v2", "status": STATUS,
                 "all_values_are_review_proposals": True,
                 "decisions": {"D1_SCF_DF": d1, "D2_ORDER_SOLVER_GATES": d2,
                               "D3_MASTER_SEED": d3, "D4_MEMORY_WALL_OUTPUT": d4},
                 "pending_reviews": ["D1 controls/final SCF gate", "D2 weight ties/degeneracy STOP thresholds/solver gates", "D3 seed proposal", "D4 headroom/admission/stop budgets/run roots"],
                 "implementation_gates_deferred": ["explicit SCF controls and no implicit save", "full DF permutation and eigenspace gate", "source synthetic serializer/operator tests", "owned memory/output/atomic accounting"],
                 "contract_final_approval": False, "next_stage_authorized": False}
    write("review_decisions_v2.json", decisions)
    stage = {"schema_version": "h4-two-stage-authorization-contract-v2",
             "artifact_scope": "REVIEW_CONTRACT_ONLY_NOT_AUTHORIZATION", "stages": STAGES,
             "transitions": [{"from": STAGES[i], "action": a, "to": STAGES[i+1]} for i,a in enumerate(ACTIONS)],
             "input_generation": {"separate_source_bound_plan": True, "new_input_hashes_required_before_generation": False, "separate_authorization": True, "review_and_explicit_launch_required": True, "only_six_distances": True, "freeze_bytes_then_STOP": True, "signal_sampling_build_compile_permitted": False},
             "signal_compile": {"input_hashes_required_for_seal": True, "all_six_frozen_inputs_required": True, "separate_result_prior_authorization": True, "review_and_explicit_launch_required": True, "must_not_reuse_generation_authorization": True, "mandatory_STOP_after_map": True},
             "semantic_gates": {"source_stage": "synthetic semantic tests only, no molecular import or actual signal/cost", "generation_stage": "SCF/DF/state/sector/order/coordinate gates only under generation authorization; no signal/cost", "science_stage": "actual input-bound signal/cost gates only after signal authorization and explicit launch; never before authorization"},
             "authorization_issued_now": 0, "current_stage": STAGES[0],
             "synthetic_validator_is_an_execution_controller": False}
    write("stage_contract_v2.json",stage)
    plan=copy.deepcopy(old)
    plan.update(schema_version="h4-zero-compute-contract-plan-v2",status=STATUS,
                base_contract_commit=BASE, v1_plan_sha256="805874dcbe1466adbd92f2300b90a88a2e111e8d43543759fae4b28a9eb1d758",
                v1_plan_fingerprint=old["plan_fingerprint"],
                v1_published_manifest_sha256="68cc9c28faaf6ab1b43799d5ffc18ece633d244634f5146239dbc93770c530f7",
                review_proposals=decisions["decisions"],authorization_order=stage,
                source_port_authorized=False,input_generation_authorized=False,
                input_generation_authorization=None,signal_compile_authorization=None,
                input_generation_plan_sealed=False,science_output_directory_created=False,
                actual_run_id=None,source_implementation_instruction=None,
                actual_allowed_cpu_count=None,observed_memory_reserved=False,
                stage=STAGES[0],checkpoint_schema="checkpoint_schema_v2.json",
                ledger_schema="completion_ledger_schema_v2.json")
    plan["model"].update(ancilla_count=1,ancilla_index=8,total_qubits=9)
    plan["future_resource_caps"].update(total_qubits=9,ancilla_count=1)
    plan["freeze_barriers"]=STAGES
    plan["parallel_policy"]="spawn outer workers<=12; allowed CPU and D4 required_available admission; internal threads/process1 including generation/solver; observations not reservations; pre-launch reduction only, pressure STOP without retry/resume"
    plan["unresolved_decisions"]=["REVIEW_APPROVAL_D1", "REVIEW_APPROVAL_D2", "REVIEW_APPROVAL_D3", "REVIEW_APPROVAL_D4"]
    plan["current_execution_counts"].pop("commit",None);plan["current_execution_counts"].pop("push",None)
    plan["current_execution_counts"].update(source_port=0,authorization_issued=0)
    del plan["plan_fingerprint"]
    plan["plan_fingerprint"]=digest(plan)
    write("zero_compute_plan_v2.json",plan)
    ps=exact_schema(plan,"Review-only zero-compute plan v2: exact proposed conditions and null actual identities")
    ps["properties"]["plan_fingerprint"]={"type":"string","pattern":"^[0-9a-f]{64}$"}
    write("plan_schema_v2.json",ps)
    scope={"schema_version":"h4-geometry-scope-v2","model":plan["model"],
           "distances_angstrom":plan["distances_angstrom"],"candidate_counts":plan["candidate_counts"],
           "templates":plan["templates"],"template_set_fingerprint":plan["template_set_fingerprint"],
           "logical_wrappers_per_geometry":12464,"logical_wrappers_total":74784,
           "actual_science_transpile_invocation_cap":74784,"worker_max":12,
           "trajectory_count":32,"paired_axes_shared_trajectory":True,
           "retain_accuracy_ineligible":True,"old_evidence":plan["old_evidence"],
           "science_execution_authorized":False,"mandatory_stop":True}
    write("scope_v2.json",scope);write("scope_schema_v2.json",exact_schema(scope,"Frozen scope v2"))
    write("stage_contract_schema_v2.json",exact_schema(stage,"Two-stage review contract, never authorization"))
    for name in ["checkpoint","completion_ledger","numerical_circuit","result"]:
        data=(OLD/(name+"_schema_v1.json")).read_bytes()
        with (HERE/(name+"_schema_v2.json")).open("xb") as handle: handle.write(data)
    # v1 record wire formats, identity and digest domains are deliberately retained.
    write("memory_observation_schema_v2.json",{
        "$schema":"https://json-schema.org/draft/2020-12/schema","type":"object",
        "additionalProperties":False,
        "required":["artifact_scope","available_bytes","allowed_cpu_count","requested_workers","observation_age_seconds","memory_pressure"],
        "properties":{"artifact_scope":{"const":"SYNTHETIC_CONTRACT_TEST_ONLY"},
            "available_bytes":{"type":"integer","minimum":0,"maximum":2**63-1},
            "allowed_cpu_count":{"type":"integer","minimum":0,"maximum":2**16},
            "requested_workers":{"type":"integer","minimum":1,"maximum":12},
            "observation_age_seconds":{"type":"integer","minimum":0,"maximum":5},
            "memory_pressure":{"type":"boolean"}}})
    write("synthetic_stage_fixtures_v2.json",{
        "artifact_scope":"SYNTHETIC_CONTRACT_TEST_ONLY","actual_authorizations":[],
        "source_commit":"a"*40,"expected_source_commit":"a"*40,
        "source_hashes":{"ARTIFICIAL_SOURCE_ONLY.py":"b"*64},
        "generation_plan_fingerprint":"c"*64,"expected_generation_plan_fingerprint":"c"*64,
        "signal_plan_fingerprint":"d"*64,"expected_signal_plan_fingerprint":"d"*64,
        "frozen_inputs":[{"distance_angstrom":g,**{k:"e"*64 for k in ["hamiltonian_sha256","df_sha256","state_sha256","sector_sha256","order_sha256","coordinate_sha256"]}} for g in plan["distances_angstrom"]],
        "flags": {k:True for k in ["contract_conditions_approved","separate_implementation_instruction","synthetic_source_gates_passed","generation_plan_source_bound","generation_authorization_reviewed","generation_explicit_launch","generation_numerical_gates_passed","generation_complete_stop","input_bound_signal_plan_sealed","signal_authorization_reviewed","signal_explicit_launch","resources_admitted"]},
        "generation_authorization_example":{"marker":"ARTIFICIAL_NOT_AUTHORIZATION","scope":"INPUT_GENERATION_ONLY","result_prior":True,"source_commit":"a"*40,"plan_fingerprint":"c"*64},
        "signal_authorization_example":{"marker":"ARTIFICIAL_NOT_AUTHORIZATION","scope":"SIGNAL_COMPILE_ONLY","result_prior":True,"source_commit":"a"*40,"plan_fingerprint":"d"*64},
        "actual_signal_or_cost_before_authorization":False})
    print(json.dumps({"status":STATUS,"plan_fingerprint":plan["plan_fingerprint"],"science_executions":0}))


if __name__ == "__main__":
    main()
