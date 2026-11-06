"""Fixed B1 operator transport and C1 saved q=2 correctness; no compiler."""
from __future__ import annotations

import io
import json
import math
from pathlib import Path
import tarfile

PAIR = (0, 1)  # first two saved sample IDs, append in ascending order


def transport_round(reference, *, m, df_error, base_time, historical_target,
                    finite):
    """State-free Hermitian Duhamel transport, in the saved invariant sector.

    d_H(t) <= min(2, |t| d_H); eps = d_H(t)+eps_PF+eps_RTE.
    The raw reference is A exp(-it E0) for an exact ORIGINAL-H ground.
    This does not call the own-DF assess_round/energy or compute N.
    Saved numerical norms/guards are premises, not rigorous enclosures.
    """
    if type(m) is not int or m < 0:
        raise ValueError("ROUND_INDEX")
    A = finite(reference["attenuation"], "saved attenuation", positive=True)
    if A > 1:
        raise ValueError("ATTENUATION_ABOVE_ONE")
    pf = finite(reference["pf_operator_error"], "saved PF operator error")
    rte = finite(reference["rte_corrected_operator_error"], "saved RTE operator error")
    df = finite(df_error, "same-sector DF error")
    time = finite(base_time, "base time", positive=True) * (1 << m)
    evolution = min(2., time * df)
    epsilon = math.fsum((pf, rte, evolution))
    raw_error = A * epsilon
    radius = A - raw_error
    phase = math.asin(epsilon) if epsilon < 1 else None
    total, stat = historical_target["beta_total"], historical_target["beta_stat"]
    if not 0 < stat < total < math.pi / 3:
        raise ValueError("HISTORICAL_PHASE_CONTEXT")
    room = None if phase is None else total - phase
    slack = None if room is None else room - stat
    raw_room = A * (math.sin(total - stat) - epsilon)
    sufficient = radius > 0 and slack is not None and slack >= 0
    return dict(m=m, q=1 << m, time=time, attenuation=A,
        saved_pf_operator_error=pf, saved_rte_corrected_operator_error=rte,
        df_hamiltonian_error_premise=df, df_evolution_error=evolution,
        original_H_corrected_operator_error=epsilon, raw_signal_error=raw_error,
        observed_radius_lower=radius, deterministic_phase_upper=phase,
        phase_room_using_historical_beta_total=room,
        historical_statistical_phase=stat, historical_phase_slack=slack,
        extra_raw_implementation_error_room_historical=raw_room,
        conditional_sufficient=sufficient, state_vector_used=False,
        saved_own_DF_signal_used=False, N=None)


def b1(root, output):
    from trottertracks.c1_stage3_accuracy import _finite
    root, output = Path(root), Path(output)
    old = root / "artifacts/research_c_near_practical_r32/2026-11-03/run01/PR"
    identity = json.loads((old / "A/identity.json").read_text())
    packet = json.loads((old / "B/references.json").read_text())
    contract = json.loads((root / "docs/research/research_c_near_practical_r32_downstream_contract_20261103.json").read_text())
    from trottertracks.track_d_h4_paired_reference import ENDPOINT
    df = identity["saved_DF"]
    if (identity["candidate"] != ENDPOINT or packet["candidate"] != ENDPOINT
            or df["candidate_sha256"] != ENDPOINT or df["actual_rank"] != 8
            or df["kappa"] != "original_g_corrected" or df["scalar"] != "nuclear_once"
            or df["evidence_kind"] != "numerical_pilot" or df["rigorous_enclosure"] is not None
            or [x["m"] for x in packet["rounds"]] != list(range(13))
            or contract["policy"]["oracle"] != "own_DF_ground_state_oracle"):
        raise ValueError("B1_SAVED_IDENTITY_OR_SCOPE")
    bound = math.fsum((_finite(df["sector_error"], "saved DF norm"),
                       _finite(df["numerical_guard"], "saved DF guard")))
    rows = [transport_round(x["reference"], m=x["m"], df_error=bound,
                           base_time=contract["target"]["base_time"],
                           historical_target=contract["target"], finite=_finite)
            for x in packet["rounds"]]
    return dict(status="CONDITIONAL_OPERATOR_SUFFICIENT" if all(x["conditional_sufficient"] for x in rows)
                else "OPERATOR_SUFFICIENT_CONDITION_NOT_MET", rows=rows,
        model="exact_original_H_ground_in_invariant_Nalpha2_Nbeta2_sector36",
        law="ideal_finite_RTE_IID_exact_native", state_preparation="EXCLUDED",
        df_counting="DF enters evolution/signal error once; no extra Weyl energy addition",
        norm_qualification="saved binary64 sector operator norms plus DF engineering guard",
        rigorous_enclosure=False, historical_phase_context=contract["target"],
        new_target_adopted=False, own_DF_signal_or_N_reused=False,
        state_vector_used=False, N_evaluated=False, actual_RPE_success_qualified=False,
        missing=["all-law compiled/synthesis error", "implemented joint-law error and IID bridge",
                 "original-H alias/horizon qualification for a selected target",
                 "selected common energy precision and failure allocation"],
        B2_started=False, ground_solves=0, DF_refits=0)


def unwrap_raw_x(wrapper):
    """Remove only the saved H/core/H/measure shell; retain declared phase."""
    from qiskit import QuantumCircuit
    n = wrapper.num_qubits - 1
    if wrapper.num_clbits != 1 or wrapper.parameters or len(wrapper.data) < 3:
        raise ValueError("SAVED_RAW_X_WRAPPER_SCHEMA")
    for item, name in ((wrapper.data[0], "h"), (wrapper.data[-2], "h"),
                       (wrapper.data[-1], "measure")):
        if item.operation.name != name or [wrapper.find_bit(q).index for q in item.qubits] != [n]:
            raise ValueError("SAVED_RAW_X_WRAPPER_SHELL")
    out = QuantumCircuit(n + 1); out.global_phase = wrapper.global_phase
    for item in wrapper.data[1:-2]:
        if item.clbits or item.operation.name in ("measure", "reset", "initialize"):
            raise ValueError("SAVED_UNITARY_CORE_REQUIRED")
        out.append(item.operation, [wrapper.find_bit(q).index for q in item.qubits])
    return out


def archive_member(root, sample, leaf):
    """Read existing lossless archive bytes; no extraction or old-file writes."""
    path = Path(root) / f"payloads/sample{sample:02d}.tar.gz"
    with tarfile.open(path, "r:gz") as archive:
        return archive.extractfile(f"sample{sample:02d}/" + leaf).read()


def c1(root, saved, output, base, checks, threads):
    import numpy as np
    from threadpoolctl import threadpool_limits
    from trottertracks import track_d_h4_paired_reference as paired
    root, saved, output = Path(root), Path(saved), Path(output)
    stage_a = root / "artifacts/track_d_h4_paired32_reference/2026-11-06/run01"
    ordinary, strong, targets, phases, events, fusion = [], [], [], [], [], []
    for i in PAIR:
        label = f"sample{i:02d}"
        record = json.loads((saved / "samples" / (label + "_events.json")).read_text())
        path_events = paired.saved_record(record, i)
        info = json.loads(archive_member(stage_a, i, "conditions.json"))
        phase = paired.path_phase(path_events, info["scalar_phase"])
        if phase != info["phase_chi"] or info["cut_after_event"] != 16 or info["h"] != "1/10":
            raise ValueError("C1_SAVED_PHASE_OR_SCOPE")
        raw = base.load_qpy(saved / "circuits" / (label + "_raw.qpy"))
        wrapper, = base.qpy.load(io.BytesIO(archive_member(stage_a, i, "circuits/symmetric_directional_X_raw.qpy")))
        ordinary.append(raw); strong.append(unwrap_raw_x(wrapper))
        with np.load(saved / "references" / (label + ".npz"), allow_pickle=False) as z:
            targets.append(z["system_reference"].copy())
        phases.append(phase); events.extend(path_events)
        fusion.append(dict(step=i, saved_fusion=json.loads(archive_member(stage_a, i, "fusion.json"))))
    earlier = base.control_one(ordinary[0]); later = base.control_one(ordinary[1])
    total_phase = math.fsum(phases)
    symmetric = strong[0].compose(strong[1])
    reference = base.reference_gadget(later, earlier, total_phase)
    cores = dict(symmetric_directional=symmetric, reference=reference)
    wrappers = {(arm, axis): base.artificial.hadamard_wrapper(core, axis)
                for arm, core in cores.items() for axis in base.AXES}
    with threadpool_limits(limits=1):
        target = targets[1] @ targets[0]
    identity = dict(fixture="saved_sample00_then_sample01", q=2, h="1/10", time="1/5",
        endpoint=paired.ENDPOINT, cohort=paired.COHORT, ordered_samples=list(PAIR),
        event_count=64, split="after complete first h-step, including its reverse head and scalar",
        A="saved sample01 physical unitary", B="saved sample00 physical unitary",
        phase_chi=total_phase, per_step_chi=phases, scalar_identity_occurrences=2,
        initial_state="exact original-H ground assumed; all256 system inputs checked",
        state_preparation="EXCLUDED", normalization_applied_to_unitary=False,
        baseline="saved strong symmetric cores, preserving each step's head/tail sharing/fusion",
        cross_step_optimization="raw coherent concatenation; no new price claim",
        reference_gadgets=1, short_gadget_repetition=False, compiler_calls=0)
    bundle = base.Bundle(None, tuple(events), cores, wrappers, earlier, later, later.inverse(),
        target, ordinary[0].compose(ordinary[1]), total_phase, identity, fusion)
    (output / "circuits").mkdir()
    for name, circuit in (("A", later), ("B", earlier), ("A_actual_adjoint", bundle.later_adjoint),
                          ("ordinary_q2_raw", bundle.saved_raw), *[(k + "_core", v) for k,v in cores.items()]):
        base.save_qpy(output / "circuits" / (name + ".qpy"), circuit)
    for (arm, axis), circuit in wrappers.items():
        base.save_qpy(output / "circuits" / f"{arm}_{axis}_raw.qpy", circuit)
    records = []
    try:
        with threadpool_limits(limits=1):
            report = paired.run_prechecks(bundle, checks, output, records, threads)
    except BaseException:
        (output / "partial_checks.json").write_text(json.dumps(dict(status="FAIL", checks=records), indent=2) + "\n")
        raise
    report.update(conditions=identity, fusion=fusion, whole_law_qualified=False,
                  compiled_correctness=None, compiler_calls=0, price_evaluated=False,
                  state_vector_ground_used=False, sampling_calls=0, C2_started=False)
    return report
