"""One-shot fixed B probe, after scoped scientific execution permission.

No RNG, transpilation, price calculation, or saved-signal reanalysis. Six
explicit conditional trajectories, not samples from the RTE distribution.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PREPARATION = ROOT / "docs/research/rpe_18round_measurement_connection_20261103"
CARRIER = "4591b8d75a2a6a55c74a51225f0c61648023c8ee"
PLAN_COMMIT = "8722b5a461c617b8846c2754275895f152df78f3"
SNAPSHOT = "artifacts/rte_connected_cluster_cost_validation/h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz"
SNAPSHOT_SHA = "13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b"
H_SHA = "56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5"
OLD_STATE_SHA = "a07d6c7fdbbb23a035f9fc54b9956ee79e557a2185e29b1870748b771958be90"
SAVED_ENERGY = -2.1663874486347616
TRAINING_SHA = "9dcdc4a28866dc223afb0ea212df645a3ed4a28e5c6dad6ddfc0ec1fcb28ba9e"
ATOL = 1e-10
SPECS = ((3, 1, "all_order0"), (3, 1, "first_order2"),
         (3, 8, "all_order0"), (3, 8, "first_order2"),
         (12, 1, "deterministic"), (12, 8, "deterministic"))
SOURCE_PATHS = (
    "src/trotterlib/df_hamiltonian.py", "src/trotterlib/df_partial_randomized_pf.py",
    "src/trotterlib/df_partial_s2.py", "src/trotterlib/df_partial_s2_cost.py",
    "src/trotterlib/df_partial_s2_repeated.py", "src/trotterlib/df_rte_circuit.py",
    "src/trotterlib/df_rte_tail.py", "src/trotterlib/df_rte_qiskit.py",
    "src/trotterlib/rte.py", "src/trotterlib/rte_connected_cluster_cost_validation.py",
    "src/trotterlib/finite_rte_signal_validation.py",
    "src/trotterlib/rpe_hadamard_failure_validation.py",
    "src/trotterlib/rpe_hadamard_interrogation.py",
    "src/trotterlib/rpe_hadamard_compiled_cost_benchmark.py",
    "src/trotterlib/research_direction_sequence_policy.py",
    "src/trotterlib/research_direction_full_scope.py",
)


def component_indices(components):
    ordered = sorted(enumerate(components), key=lambda p: p[1].component_id)
    i0, c0 = ordered[0]
    matches = [(i, c) for i, c in ordered if c.basis_id != c0.basis_id]
    if not matches:
        raise ValueError("UNDETERMINED: no c1 with a different basis_id")
    return i0, matches[0][0]


def event_profile(q, profile, zero, two):
    if q not in (1, 8) or profile not in ("all_order0", "first_order2"):
        raise ValueError("Outside the fixed conditional profiles")
    return (two if profile == "first_order2" else zero,) + (zero,) * (q - 1)


def embed_sector(state, indices, permutation, n_qubits):
    import numpy as np
    full = np.zeros(1 << n_qubits, dtype=np.complex128)
    full[np.asarray(indices)[permutation]] = state
    return full


def trajectory_state(state, forward, reverse, events, operators, q, r, apply_event):
    """Vector-returning companion to legacy _trajectory_signal; same order.

    The legacy helper discards the output vector, which is needed for control
    branch comparisons. Event application itself is reused without changes.
    """
    if len(events) != q * r:
        raise ValueError("Event count mismatch")
    value = state
    for step in range(q):
        value = forward @ value
        for event in events[step*r:(step+1)*r]:
            value = apply_event(event, operators, value)
        value = reverse @ value
    return value


def evaluate_job(job):
    """Pure ideal actions on supplied circuits/vectors; no H or sampler calls."""
    import numpy as np
    from qiskit.quantum_info import Statevector
    started = time.monotonic()
    psi, target = job["system_state"], job["target_system_state"]
    dim = psi.size
    zero = np.zeros_like(psi)
    initial0, initial1 = np.concatenate((psi, zero)), np.concatenate((zero, psi))
    output0 = np.asarray(Statevector(initial0).evolve(job["evolution"]).data)
    output1 = np.asarray(Statevector(initial1).evolve(job["evolution"]).data)
    expected0, expected1 = initial0, np.concatenate((zero, target))
    keep = np.zeros(dim, dtype=bool)
    keep[job["sector_indices"]] = True
    residuals = {
        "control_zero_branch_max_abs": float(np.max(np.abs(output0-expected0))),
        "control_one_branch_max_abs": float(np.max(np.abs(output1-expected1))),
        "control_zero_sector_leakage_l2": float(np.linalg.norm(output0.reshape(2,dim)[:,~keep])),
        "control_one_sector_leakage_l2": float(np.linalg.norm(output1.reshape(2,dim)[:,~keep])),
    }
    axes = {}
    for axis, circuit in job["wrappers"].items():
        # Only terminal ancilla measurement is removed, on a copy.
        unitary = circuit.remove_final_measurements(inplace=False)
        vector = np.asarray(Statevector(initial0).evolve(unitary).data)
        actual = float(np.sum(np.abs(vector[:dim])**2)-np.sum(np.abs(vector[dim:])**2))
        predicted = float(job["signal"].real if axis == "cosine" else job["signal"].imag)
        delta = abs(actual-predicted)
        axes[axis] = {"actual_Z_mean":actual, "theoretical_mean":predicted,
                      "absolute_residual":delta, "pass":delta <= ATOL}
    return {"id":job["id"], "ld":job["ld"], "q":job["q"],
            "profile":job["profile"], "axes":axes, "residuals":residuals,
            "pass":all(v<=ATOL for v in residuals.values()) and all(a["pass"] for a in axes.values()),
            "elapsed_seconds":time.monotonic()-started}


def load_api():
    # Import definitions only; none of the legacy whole scientific runners run.
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(PREPARATION))
    import numpy as np
    import measurement_adapter as adapter
    from trotterlib.df_hamiltonian import PhysicalSector
    from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash, split_df_hamiltonian_by_ld
    from trotterlib.df_partial_s2 import prepare_df_partial_s2, QiskitDFPartialS2CircuitBuilder
    from trotterlib.df_partial_s2_cost import _request_for_events
    from trotterlib.df_partial_s2_repeated import DFPartialS2RepeatedRequest, QiskitDFPartialS2RepeatedCircuitBuilder
    from trotterlib.df_rte_tail import basis_change_unitary
    from trotterlib.rte import _make_event
    from trotterlib.rte_connected_cluster_cost_validation import load_connected_cluster_hamiltonian_snapshot
    from trotterlib.finite_rte_signal_validation import (
        dense_df_operator_in_sector, _array_fingerprint,
        _qiskit_to_openfermion_sector_permutation, _circuit_operator_in_openfermion_sector,
        _matrix_in_openfermion_sector)
    from trotterlib.rpe_hadamard_failure_validation import _apply_event_to_state
    from trotterlib.research_direction_full_scope import _finite_inputs, apply_run_threshold_policy
    from trotterlib.research_direction_sequence_policy import register_support_restricted_bases
    from trotterlib.rpe_hadamard_interrogation import RPEHadamardInterrogationRequest
    from trotterlib.rpe_hadamard_compiled_cost_benchmark import QiskitRPEHadamardBenchmarkCircuitBuilder
    return locals()


def require_small(values, label):
    if not all(math.isfinite(v) and v <= ATOL for v in values):
        raise ValueError("INCONSISTENT: " + label)


def run(record, output):
    api = load_api()
    np = api["np"]
    versions = {n:importlib.metadata.version(n) for n in ("qiskit", "numpy", "scipy")}
    record["versions"] = versions
    if versions["qiskit"] != "1.3.0":
        raise ValueError("UNDETERMINED: Qiskit1.3.0 required")
    snapshot = ROOT / SNAPSHOT
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != SNAPSHOT_SHA:
        raise ValueError("UNDETERMINED: snapshot identity mismatch")
    for path in SOURCE_PATHS:
        raw = (ROOT/path).read_bytes()
        old = subprocess.check_output(["git","-C",str(ROOT),"show",f"{CARRIER}:{path}"])
        if raw != old:
            raise ValueError("UNDETERMINED: legacy SOURCE differs at " + path)
    record["inputs"] = {"snapshot_path":SNAPSHOT,"snapshot_sha256":SNAPSHOT_SHA,
                        "hamiltonian_hash":H_SHA,"source_carrier_commit":CARRIER,
                        "legacy_source_paths_matched":list(SOURCE_PATHS)}
    hamiltonian = api["load_connected_cluster_hamiltonian_snapshot"](snapshot)
    if api["df_hamiltonian_hash"](hamiltonian) != H_SHA or hamiltonian.n_qubits != 8 or hamiltonian.n_blocks != 12:
        raise ValueError("UNDETERMINED: Hamiltonian identity mismatch")
    sector = api["PhysicalSector"].number_sector(n_qubits=8,n_electrons=4)
    dense = api["dense_df_operator_in_sector"](hamiltonian,sector)
    record["counts"]["ground_state_solves_started"] = 1
    energies, vectors = np.linalg.eigh(dense)
    psi, energy = np.asarray(vectors[:,0],dtype=np.complex128), float(energies[0])
    state_residuals = {"normalization":float(abs(np.linalg.norm(psi)-1)),
                       "eigen_residual_l2":float(np.linalg.norm(dense@psi-energy*psi)),
                       "saved_energy_difference":abs(energy-SAVED_ENERGY),
                       "hermiticity_max_abs":float(np.max(np.abs(dense-dense.conj().T)))}
    fingerprint = api["_array_fingerprint"](psi)
    record["physical_state"] = {
        "fingerprint":fingerprint,"saved_fingerprint":OLD_STATE_SHA,
        "matches_saved_fingerprint":fingerprint==OLD_STATE_SHA,
        "identity_status":"MATCHED_RECONSTRUCTED_STATE_FINGERPRINT" if fingerprint==OLD_STATE_SHA else
                          "SOURCE_CONSISTENT_PHYSICAL_RECONSTRUCTION; SAVED_STATE_BYTES_UNVERIFIED",
        "energy":energy,"saved_energy":SAVED_ENERGY,"sector_dimension":sector.dimension,
        "residuals":state_residuals,"shared_by_both_ld":True,
        "old_Full_diagnostic_fingerprint_still_missing":True}
    np.savez(output/"physical_state.npz",state=psi,basis_indices=sector.basis_indices,
             n_qubits=8,n_electrons=4,hamiltonian_hash=H_SHA,fingerprint=fingerprint)
    require_small(state_residuals.values(),"physical state")
    permutation = api["_qiskit_to_openfermion_sector_permutation"](sector)
    full_psi = embed_sector(psi,sector.basis_indices,permutation,8)
    preparations = {ld:api["prepare_df_partial_s2"](hamiltonian,api["split_df_hamiltonian_by_ld"](hamiltonian,ld),
                    identity_policy="extract_identity_phase") for ld in (3,12)}
    partial = preparations[3]
    config, distribution = api["_finite_inputs"](partial,delta_time=.02,rte_steps=1,finite_taylor_order=2)
    components = partial.rte_preparation.symbolic_tail.components
    i0,i1 = component_indices(components)
    zero = api["_make_event"]((i0,),components,distribution,distribution.orders.index(0))
    two = api["_make_event"]((i0,i1,i0),components,distribution,distribution.orders.index(2))
    record["conditional_events"] = {
        "selector":"component_id order: c0 first, c1 first different basis_id",
        "components":[{"id":components[i].component_id,"basis_id":components[i].basis_id,
                       "coefficient_sign":components[i].coefficient_sign,
                       "support":list(components[i].diagonal_pauli_support)} for i in (i0,i1)],
        "order0":{"rotation":zero.rotation_component_id,"angle":zero.rotation_angle,"phase":_z(zero.phase)},
        "order2":{"rotation":two.rotation_component_id,"products":list(two.product_component_ids),
                  "angle":two.rotation_angle,"phase":_z(two.phase)},
        "random_draws":0,"not_a_distribution_sample":True}
    support_defs, proofs = api["register_support_restricted_bases"](hamiltonian,partial)
    support_error = max((p["preserved_columns_max_abs_residual"] for p in proofs),default=0.)
    require_small([support_error],"support basis completion")
    record["support_policy"]={"policy":"support_run_le_1","training_fingerprint":TRAINING_SHA,
                              "maximum_preserved_column_residual":support_error}
    operators, basis_cache = {}, {}
    for i in (i0,i1):
        component=components[i]
        if component.basis_id not in basis_cache:
            basis_cache[component.basis_id]=api["basis_change_unitary"](
                partial.tail_extraction.basis_definition(component.basis_id),max_dense_qubits=8)
        basis=basis_cache[component.basis_id]
        states=np.arange(256,dtype=np.uint64);parity=np.zeros(256,dtype=np.uint64)
        for qubit in component.diagonal_pauli_support:parity ^= (states>>int(qubit)) & 1
        signs=1.-2.*parity.astype(np.float64)
        restricted=api["_matrix_in_openfermion_sector"](
            component.coefficient_sign*((basis*signs[np.newaxis,:])@basis.conj().T),sector,permutation)
        require_small([restricted.leakage_frobenius_norm],"component leakage")
        operators[component.component_id]=restricted.matrix
    halves={};record["half_checks"]={}
    for ld,prep in preparations.items():
        reference=api["_request_for_events"](prep,step_time=.02,config=config if ld==3 else None,
                distribution=distribution if ld==3 else None,events=(zero,) if ld==3 else (),
                seed=0 if ld==3 else None,controlled=False,ancilla_qubit=None,cancel_adjacent_equal_bases=True)
        parts=api["QiskitDFPartialS2CircuitBuilder"]().build_additive_circuits(reference)
        record["counts"]["uncontrolled_reference_half_builds"] += 1
        fs=api["_circuit_operator_in_openfermion_sector"](parts.forward_deterministic_half,sector,permutation)
        rs=api["_circuit_operator_in_openfermion_sector"](parts.reverse_deterministic_half,sector,permutation)
        record["half_checks"][str(ld)]={"forward_leakage_frobenius":fs.leakage_frobenius_norm,
                                        "reverse_leakage_frobenius":rs.leakage_frobenius_norm}
        require_small(record["half_checks"][str(ld)].values(),"deterministic half leakage")
        halves[ld]=(fs.matrix,rs.matrix)
    settings={ld:{s.q:s for s in api["adapter"].load_fixed_rounds(ld) if s.q in (1,8)} for ld in (3,12)}
    jobs=[];record["requests"]=[]
    wrapper_builder=api["QiskitRPEHadamardBenchmarkCircuitBuilder"](maximum_repetition_count=8)
    for ld,q,profile in SPECS:
        prep=preparations[ld]
        events=event_profile(q,profile,zero,two) if ld==3 else ()
        target=trajectory_state(psi,*halves[ld],events,operators,q,1 if ld==3 else 0,api["_apply_event_to_state"])
        signal=complex(np.vdot(psi,target));record["counts"]["theoretical_trajectory_signals"] += 1
        steps=[api["_request_for_events"](prep,step_time=.02,config=config if ld==3 else None,
                distribution=distribution if ld==3 else None,events=(events[j],) if ld==3 else (),
                seed=0 if ld==3 else None,controlled=True,ancilla_qubit=8,cancel_adjacent_equal_bases=True)
               for j in range(q)]
        request=api["DFPartialS2RepeatedRequest"].from_step_requests(steps,
            sampling_policy="fixed_conditional_profile_no_rng",construction_policy="boundary_optimized")
        if ld==3:request=api["apply_run_threshold_policy"](request,support_defs,training_fingerprint=TRAINING_SHA)
        evolution=api["QiskitDFPartialS2RepeatedCircuitBuilder"]().build(request,construction_policy="boundary_optimized")
        record["counts"]["controlled_evolution_builds"] += 1
        wrappers={}
        for axis in ("cosine","sine"):
            wrapper=wrapper_builder.build(api["RPEHadamardInterrogationRequest"](evolution,axis,include_measurement=True))
            record["counts"]["measurement_wrapper_builds"] += 1
            wrappers[axis]=wrapper.circuit
        identity=f"ld{ld}_q{q}_{profile}"
        record["requests"].append({"id":identity,"settings":asdict(settings[ld][q]),
            "theoretical_signal":_z(signal),"trajectory_fingerprint":evolution.trajectory_fingerprint,
            "circuit_semantics_fingerprint":evolution.circuit_semantics_fingerprint,
            "constant_phase":evolution.constant_phase,"extracted_identity_phase":evolution.extracted_identity_phase,
            "rte_relative_phase":evolution.rte_relative_phase,
            "support_application_count":sum(s.rte_support_restricted_application_count for s in evolution.step_results)})
        jobs.append({"id":identity,"ld":ld,"q":q,"profile":profile,"system_state":full_psi,
            "target_system_state":embed_sector(target,sector.basis_indices,permutation,8),
            "sector_indices":sector.basis_indices,"signal":signal,"evolution":evolution.circuit,"wrappers":wrappers})
    available=len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else os.cpu_count() or 1
    workers=min(len(jobs),available)
    record["execution"]={"available_cpu_count":available,"workers":workers,"start_method":"spawn",
        "BLAS_threads_per_process":1,"reason":"Six independent fixed requests; small 2xN local-gate arrays do not benefit from BLAS threading; immutable basis cache reused during setup",
        "gpu_used":False,"gpu_reason":"No working NVIDIA driver; 512-amplitude actions use existing CPU Qiskit Statevector",
        "compiled_price_profile_metadata_only":{"qiskit":"1.3.0","optimization_level":2,"basis":["rz","sx","x","cx"],"seed":17}}
    record["measurements"]=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        futures=[pool.submit(evaluate_job,job) for job in jobs]
        record["counts"]["request_action_jobs_submitted"] = len(futures)
        for future in as_completed(futures):
            item=future.result();record["measurements"].append(item)
            record["counts"]["control_branch_state_actions"] += 2
            record["counts"]["Hadamard_ideal_mean_actions"] += 2
            if not item["pass"]:
                for f in futures:f.cancel()
                raise ValueError("INCONSISTENT: "+item["id"])
    record["measurements"].sort(key=lambda x:x["id"])
    record["status"]="LOCAL_FIXED_REQUEST_MEANS_CONSISTENT_MANDATORY_STOP"


def _z(value):
    return {"real":float(value.real),"imag":float(value.imag)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    # Current job only; small matrix kernels plus process parallelism avoid
    # oversubscription. No shared environment or hardware settings are changed.
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="1"
    args.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    record={"status":"STARTED","source_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"]).decode().strip(),
        "plan_commit":PLAN_COMMIT,"atol":ATOL,"fixed_specs":list(SPECS),
        "counts":{k:0 for k in ("ground_state_solves_started","uncontrolled_reference_half_builds",
            "theoretical_trajectory_signals","controlled_evolution_builds","measurement_wrapper_builds",
            "control_branch_state_actions","Hadamard_ideal_mean_actions","RNG_draws","quantum_shots",
            "transpiles_or_compiles","price_evaluations","saved_signal_reanalyses")},
        "scope":"Fixed conditional requests; not fresh-IID or marginal validation",
        "action_count_convention":"Completed action counts cover harvested worker results; after a worker error or early STOP other submitted jobs may have run within the six-request scope",
        "mandatory_STOP":True}
    try:
        run(record,args.output_dir)
    except Exception as exc:
        record["status"]="INCONSISTENT_MANDATORY_STOP" if str(exc).startswith("INCONSISTENT:") else "UNDETERMINED_MANDATORY_STOP"
        record["error"]={"type":type(exc).__name__,"message":str(exc)}
        raise
    finally:
        record["wall_seconds"]=time.monotonic()-started
        (args.output_dir/"RESULT.json").write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
