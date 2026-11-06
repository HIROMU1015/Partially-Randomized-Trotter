"""Fixed C3 entry plus opt-in ordered saved paths; reuse emitters/CAR/checks."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from . import track_d_h4_paired_reference as paired
from . import track_d_h4_reference as base
from trotterlib.df_trotter.ops import append_diagonal_primitives

FIXTURES = {4: tuple(range(4)), 8: tuple(range(8))}
CUTS = {4: 2, 8: 4}


def head_recipe(kind):
    """Chronological blocks: preserve the repeated builder's half-sweep rule.

    D0(h/2) then F0(h/2) is ordinary controlled U0(h), including scalar.
    A physical segment uses the same legal merge without directional control.
    """
    half = base.H / 2
    if kind == "first":
        return tuple((i, half, "uncontrolled") for i in range(3))
    if kind == "last":
        return tuple((i, half, "directional") for i in (2, 1, 0))
    if kind == "boundary":
        return ((2, half, "directional"), (1, half, "directional"),
                (0, base.H, "ordinary"), (1, half, "uncontrolled"),
                (2, half, "uncontrolled"))
    raise ValueError("FIXED_HEAD_KIND")


def mixed_head(context, recipe):
    """StrongBuilder transitions with existing central emitters of mixed modes.

    Existing sequence()/directional_head() each accepts only one control mode;
    neither alone represents the legal directional/uncontrolled boundary merge.
    """
    builder, model = context.builder, context.arm.model
    if not builder.relative_fusion or not recipe:
        raise ValueError("STRONG_HEAD_REQUIRED")
    groups = []
    for index, time, mode in recipe:
        if mode not in ("uncontrolled", "ordinary", "directional"):
            raise ValueError("HEAD_CONTROL_MODE")
        block = model.blocks[index]
        if groups and mode == groups[-1][0] and block.rotation == model.blocks[groups[-1][1][-1][0]].rotation:
            groups[-1][1].append((index, time))
        else:
            groups.append((mode, [(index, time)]))
    qc = QuantumCircuit(model.modes + 1)
    qc.compose(builder._basis(model, model.blocks[groups[0][1][0][0]]).inverse(), range(model.modes), inplace=True)
    for position, (mode, schedule) in enumerate(groups):
        block = model.blocks[schedule[0][0]]
        diagonal = base.combined_diagonal(model, schedule)
        if mode == "directional":
            base.append_directional_diagonal(qc, diagonal, model.modes)
        else:
            append_diagonal_primitives(qc, diagonal, controlled=mode == "ordinary",
                                      ancilla_qubit=model.modes if mode == "ordinary" else None)
        if position + 1 == len(groups):
            qc.compose(builder._basis(model, block), range(model.modes), inplace=True)
        else:
            following = model.blocks[groups[position + 1][1][0][0]]
            if block.rotation != following.rotation:
                qc.compose(builder.transition(model, block, following), range(model.modes), inplace=True)
    return qc


def physical_segment(context, tails):
    """One closed A/B, merging internal step boundaries before control removal."""
    n = base.N
    qc = QuantumCircuit(n)
    qc.compose(context.forward, inplace=True)
    boundary = tuple((i, t) for i, t, _ in head_recipe("boundary"))
    bridge = base.control_one(context.builder.sequence(context.arm.model, boundary,
                                                      optimized=True, scalar_time=0))
    for i, tail in enumerate(tails):
        qc.compose(base.control_one(tail), inplace=True)
        qc.compose(context.backward if i == len(tails) - 1 else bridge, inplace=True)
    qc.global_phase += math.fsum([context.scalar_phase] * len(tails))
    return qc


def prepare(context, saved, q):
    if q not in FIXTURES:
        raise ValueError("ONLY_FIXED_Q4_Q8")
    return prepare_ordered(context, saved, FIXTURES[q], CUTS[q],
        fixture_selection="ascending saved IDs, fixed before results; not fresh-IID law sample")


def prepare_ordered(context, saved, ordered_samples, cut, *, fixture_selection):
    """Opt-in ordered saved request; the fixed C3 entry remains unchanged."""
    ordered_samples = tuple(ordered_samples)
    q = len(ordered_samples)
    if q < 2 or type(cut) is not int or not 0 < cut < q:
        raise ValueError("COMPLETE_INTERNAL_STEP_CUT")
    if any(type(i) is not int or not 0 <= i < 32 for i in ordered_samples):
        raise ValueError("SAVED_SAMPLE_ID")
    saved = Path(saved); events, phases, tails, target_rows, raw_rows, bindings = [], [], [], [], [], []
    c = context
    atoms = {a.atom_id: a for a in c.arm.plan.atoms}
    for sample in ordered_samples:
        label = f"sample{sample:02d}"
        record = json.loads((saved / "samples" / (label + "_events.json")).read_text())
        step = paired.saved_record(record, sample)
        for event in step:
            atom = atoms[event.rotation_component_id]; app = event.application_sequence[0]
            if (app.coefficient_sign != (1 if atom.coefficient > 0 else -1)
                    or app.coefficient_abs != abs(float(atom.coefficient))
                    or app.basis_id != atom.basis_key or app.diagonal_pauli_support != atom.support
                    or not math.isclose(event.rotation_angle, math.atan(c.config.dimensionless_step_time), rel_tol=1e-14, abs_tol=1e-15)):
                raise ValueError("EVENT_SIGN_FRAME_OR_ANGLE")
        events.extend(step); phases.append(paired.path_phase(step, c.scalar_phase))
        tails.append(base.build_event_sequence(step, c.arm.plan.atoms, c.provider, optimized=True))
        raw_rows.append(base.load_qpy(saved / "circuits" / (label + "_raw.qpy")))
        with np.load(saved / "references" / (label + ".npz"), allow_pickle=False) as z:
            target_rows.append(z["system_reference"].copy())
        bindings.append(dict(sample=sample, signed_angles=[e.unsigned_rotation_angle for e in step],
                             event_normalization=[e.event_normalization for e in step]))
    symmetric = QuantumCircuit(base.N + 1)
    symmetric.compose(mixed_head(c, head_recipe("first")), inplace=True)
    bridge = mixed_head(c, head_recipe("boundary"))
    for i, tail in enumerate(tails):
        symmetric.compose(tail, inplace=True)
        symmetric.compose(mixed_head(c, head_recipe("last")) if i == q - 1 else bridge, inplace=True)
    scalar_phase = math.fsum([c.scalar_phase] * q)
    symmetric.p(scalar_phase, base.N)
    earlier, later = physical_segment(c, tails[:cut]), physical_segment(c, tails[cut:])
    phase = math.fsum(phases)
    cores = dict(symmetric_directional=symmetric, reference=base.reference_gadget(later, earlier, phase))
    wrappers = {(arm, axis): base.artificial.hadamard_wrapper(core, axis)
                for arm, core in cores.items() for axis in base.AXES}
    target = np.eye(1 << base.N, dtype=complex); raw = QuantumCircuit(base.N + 1)
    for matrix, circuit in zip(target_rows, raw_rows):
        target = matrix @ target; raw.compose(circuit, inplace=True)
    info = dict(q=q, h="1/10", time=str(q * base.H), ordered_samples=list(ordered_samples),
        cohort=paired.COHORT, endpoint=paired.ENDPOINT, event_count=len(events),
        model="linear H4/1.00A/STO-3G/number-CDF8/Nalpha=Nbeta=2/LD2/S2/s1/r32/K0",
        cut_after_complete_step=cut, AB_order="A=last half, B=first half; V=U_last ... U_first=A B",
        initial_state="exact original-H ground assumed; all256 input columns checked", state_preparation="EXCLUDED",
        scalar=float(c.arm.model.constant), extracted_identity=c.arm.plan.extracted_identity,
        scalar_identity_occurrences=q, per_step_scalar_phase=c.scalar_phase, aggregated_scalar_phase=scalar_phase,
        per_step_chi=phases, phase_chi=phase, lambda_R=c.tail.lambda_r, per_step_events=bindings,
        normalization_applied_to_unitary=False, attenuation_applied_to_unitary=False,
        reference_gadgets=1, short_gadget_repetition=False,
        strong_symmetric=dict(uncontrolled_gaussian=True, directional_centers=True,
            head_relative_fusion=True, tail_full_frame_same_basis_sharing=True,
            step_boundaries_fused=q-1, boundary_rule="directional half + uncontrolled half -> ordinary controlled full",
            scalar_identity_aggregated=True, head_tail_cross_fusion=False),
        reference_optimization=dict(internal_step_boundaries_fused=q-2, global_cut_breaks_boundary_fusion=1,
            head_relative_fusion=True, tail_same_basis_sharing=True, scalar_aggregated_per_A_B=True),
        C2_difference="C2 concatenated saved strong cores without new manual cross-step fusion",
        fixture_selection=fixture_selection)
    return base.Bundle(c.arm, tuple(events), cores, wrappers, earlier, later, later.inverse(),
                       target, raw, phase, info, list(c.builder.fusion_records))


def raw_checks(origin, saved, contract, output, q, checks, threads):
    from threadpoolctl import threadpool_limits
    output = Path(output)
    with threadpool_limits(limits=1):
        context = paired.prepare_common(origin, contract)
        bundle = prepare(context, saved, q)
        return save_and_check(bundle, output, checks, threads)


def save_and_check(bundle, output, checks, threads):
    """Shared raw persistence/harness; never compiles or samples."""
    output = Path(output); records = []
    (output / "circuits").mkdir()
    (output / "conditions.json").write_text(json.dumps(bundle.identity, indent=2) + "\n")
    (output / "fusion.json").write_text(json.dumps(bundle.fusion, indent=2) + "\n")
    for name, circuit in (("A", bundle.later), ("B", bundle.earlier), ("A_actual_adjoint", bundle.later_adjoint),
                          ("ordinary_raw", bundle.saved_raw), *[(k+"_core", v) for k,v in bundle.cores.items()]):
        base.save_qpy(output / "circuits" / (name + ".qpy"), circuit)
    for (arm, axis), circuit in bundle.wrappers.items():
        base.save_qpy(output / "circuits" / f"{arm}_{axis}_raw.qpy", circuit)
    try:
        return paired.run_prechecks(bundle, checks, output, records, threads)
    except BaseException:
        (output / "partial_checks.json").write_text(json.dumps(dict(status="FAIL", checks=records), indent=2) + "\n")
        raise
