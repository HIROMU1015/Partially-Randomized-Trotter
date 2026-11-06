"""Saved-cohort extension of Track D's fixed-H4 bridge; no sampler or solver."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qiskit import QuantumCircuit

from . import track_d_h4_reference as base

ARMS = ("symmetric_directional", "reference")
AXES = base.AXES
COHORT = "near_practical_PR_r32_fixed32_v1"
ENDPOINT = "a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a"
COHORT_SOURCE = "103c8ba7ecda6e5da48cc0e73f267befcf104c4ad3c18c32de875289fa532739"


def saved_record(record, sample):
    """Only deserialize this saved K0 cohort; never sample replacements."""
    policy = dict(master=2026110304, cohort=COHORT, axis="NEAR_PRACTICAL_PR_PRICE_ONLY",
                  samples=32, events_per_sample=32, PRNG="numpy.random.PCG64",
                  paired=False, adaptive=False)
    if (record["sample"] != sample or record["sampling"] != policy
            or record["JU"] != ENDPOINT or record["SOURCE"] != COHORT_SOURCE):
        raise ValueError("SAVED_COHORT_BINDING")
    events = base.stored_events(record["events"])
    if len(events) != 32 or any(e.taylor_order or e.product_component_ids or e.phase != 1 for e in events):
        raise ValueError("FIXED_32_K0_EVENTS")
    return events


def path_phase(events, scalar_phase):
    # unsigned_rotation_angle in the original RTE API applies the saved
    # coefficient sign. K0 has no product factors; keep event.phase explicit.
    return math.fsum([scalar_phase, *[math.atan2(e.phase.imag, e.phase.real)
                                     - e.unsigned_rotation_angle for e in events]])


@dataclass
class Context:
    arm: object
    config: object
    tail: object
    builder: object
    provider: object
    forward: QuantumCircuit
    backward: QuantumCircuit
    directional: QuantumCircuit
    scalar_phase: float


def prepare_common(origin, contract):
    """Same prepare_arm/head/provider/phase policy as the fixed-word bridge."""
    origin = Path(origin)
    with np.load(origin / contract["input"], allow_pickle=False) as z:
        packet = {k: z[k] for k in ("h", "g", "constant")}
    with np.load(origin / contract["candidate"], allow_pickle=False) as z:
        endpoint = {k: z[k] for k in ("J", "U")}
    with np.load(origin / contract["sector"], allow_pickle=False) as z:
        if not all(np.array_equal(endpoint[k], z[k]) for k in ("J", "U")):
            raise ValueError("SAVED_JU_REFERENCE_BINDING")
    if endpoint["J"].shape != (8, 4, 4) or endpoint["U"].shape != (8, 4, 4):
        raise ValueError("FIXED_CDF8_INPUT")
    arm = base.original.prepare_arm(**packet, **endpoint, role="proposed",
        source_identity=contract["saved_inputs"][contract["candidate"]],
        input_identity=contract["saved_inputs"][contract["input"]], ld=2)
    if arm.model.modes != base.N:
        raise ValueError("FIXED_EIGHT_MODES")
    rows = json.loads((origin / contract["atoms"]).read_text())["atoms"]
    actual = {a.atom_id: a for a in arm.plan.atoms}
    for row in rows:
        atom = actual[row["atom_id"]]
        if (atom.frame_id != row["frame_id"] or list(atom.support) != row["support"]
                or base.Fraction(atom.coefficient) != base.Fraction(row["coefficient"])):
            raise ValueError("SAVED_SIGNED_ATOM")
    if len(actual) != len(rows):
        raise ValueError("SAVED_ATOM_COUNT")
    tail, config, _ = base.original.configure(arm, base.H, 32, 0)
    law = json.loads((origin / "artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json").read_text())
    if (arm.plan.extracted_identity != law["identity"]
            or not math.isclose(tail.lambda_r, law["lambda_R"], rel_tol=1e-12, abs_tol=1e-14)):
        raise ValueError("SAVED_IDENTITY_OR_LAMBDA")
    builder = base.StrongBuilder(relative_fusion=True)
    forward_schedule = tuple((i, base.H / 2) for i in range(3))
    backward_schedule = tuple(reversed(forward_schedule))
    forward = builder.sequence(arm.model, forward_schedule, optimized=True, scalar_time=0)
    backward = builder.sequence(arm.model, backward_schedule, optimized=True, scalar_time=0)
    directional = base.directional_head(builder, arm.model, backward_schedule)

    class Provider(base.BasisProvider):
        def basis(self, atom, singleton=False):
            for block in arm.model.blocks:
                if block.rotation == arm.plan.frames[atom.frame_id]:
                    return builder._basis(arm.model, block).copy()
            raise ValueError("UNKNOWN_ARM_BASIS")

    provider = Provider(base.N, arm.plan.frames)
    return Context(arm, config, tail, builder, provider, base.control_one(forward),
                   base.control_one(backward), directional,
                   -(arm.model.constant + arm.plan.extracted_identity) * float(base.H))


def prepare_path(context, saved, sample):
    """Keep the strong head fusion and original full-frame tail sharing."""
    c = context; arm = c.arm; label = f"sample{sample:02d}"; saved = Path(saved)
    record = json.loads((saved / "samples" / (label + "_events.json")).read_text())
    events = saved_record(record, sample)
    atoms = {a.atom_id: a for a in arm.plan.atoms}
    for event in events:
        atom = atoms[event.rotation_component_id]; app = event.application_sequence[0]
        if (app.coefficient_sign != (1 if atom.coefficient > 0 else -1)
                or app.coefficient_abs != abs(float(atom.coefficient))
                or app.basis_id != atom.basis_key or app.diagonal_pauli_support != atom.support
                or not math.isclose(event.rotation_angle, math.atan(c.config.dimensionless_step_time),
                                    rel_tol=1e-14, abs_tol=1e-15)):
            raise ValueError("EVENT_SIGN_FRAME_OR_ANGLE")
    tail = base.build_event_sequence(events, arm.plan.atoms, c.provider, optimized=True)
    symmetric = QuantumCircuit(base.N + 1)
    symmetric.compose(c.forward, range(base.N), inplace=True)
    symmetric.compose(tail, inplace=True)
    symmetric.compose(c.directional, inplace=True)
    symmetric.p(c.scalar_phase, base.N)
    first = base.control_one(base.build_event_sequence(events[:base.CUT], arm.plan.atoms, c.provider, optimized=True))
    last = base.control_one(base.build_event_sequence(events[base.CUT:], arm.plan.atoms, c.provider, optimized=True))
    earlier = c.forward.compose(first)
    later = last.compose(c.backward); later.global_phase += c.scalar_phase
    phase = path_phase(events, c.scalar_phase)
    reference = base.reference_gadget(later, earlier, phase)
    cores = dict(symmetric_directional=symmetric, reference=reference)
    wrappers = {(name, axis): base.artificial.hadamard_wrapper(core, axis)
                for name, core in cores.items() for axis in AXES}
    with np.load(saved / "references" / (label + ".npz"), allow_pickle=False) as z:
        target = z["system_reference"]
    if target.shape != (256, 256):
        raise ValueError("SAVED_FULL_CAR_REFERENCE")
    info = dict(sample=sample, cohort=COHORT, endpoint=ENDPOINT,
        model="linear H4/1.00A/STO-3G/number CDF8", Nalpha=2, Nbeta=2,
        ld=2, h=str(base.H), r=32, K=0, events=32, cut_after_event=base.CUT,
        scalar=float(arm.model.constant), extracted_identity=arm.plan.extracted_identity,
        scalar_phase=c.scalar_phase, phase_chi=phase,
        signed_rotation_angles=[e.unsigned_rotation_angle for e in events],
        event_normalization=[e.event_normalization for e in events], lambda_R=c.tail.lambda_r,
        normalization_applied_to_unitary_or_observable=False,
        system_initial_state="exact original-H ground assumed; no ground vector used",
        state_preparation_cost="EXCLUDED", cross_head_tail_relative_fusion=False,
        head_relative_fusion=True, strong_head_relative_transitions=len(c.builder.fusion_records),
        cut_basis_keys_equal=events[15].application_sequence[0].basis_id == events[16].application_sequence[0].basis_id,
        cut_policy="full optimized tail for symmetric; two optimized half tails for Reference; no extra cross-cut fusion",
        cut_fusion_cost_causal_ablation=None)
    return base.Bundle(arm, events, cores, wrappers, earlier, later, later.inverse(), target,
        base.load_qpy(saved / "circuits" / (label + "_raw.qpy")), phase, info,
        list(c.builder.fusion_records))


def run_prechecks(bundle, checks, output, records, threads):
    # Existing harness checks its first two ARMS as direct controlled operators.
    # Expose exactly the one direct arm here; the Reference state-action branch
    # below that loop remains unchanged. No ordinary wrapper is rebuilt.
    implementation = SimpleNamespace(**{**vars(base), "ARMS": (ARMS[0],)})
    result = checks.run_checks(bundle, implementation, output, records, threads)
    from trottertracks.c1_stage3_molecular import component_check
    with np.load(output / "references.npz", allow_pickle=False) as z:
        result["paired_observables"] = {}
        for axis in AXES:
            row = component_check(z["raw_reference_" + axis], z["raw_symmetric_directional_" + axis],
                                  base.ATOL, label="raw_paired_" + axis)
            if row["frobenius"] > base.SIGNAL_ATOL:
                raise ValueError("RAW_PAIRED_OBSERVABLE_SIGNAL_ALLOWANCE")
            result["paired_observables"][axis] = row
    return result
