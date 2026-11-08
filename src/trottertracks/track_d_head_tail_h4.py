"""Opt-in C1 transfer to the single saved C6a H4/q4 path.

The old two-mode experiment and every C6a default remain unchanged. Only
register/span binding is extended; Gaussian synthesis and phase correction
are StrongBuilder.transition's existing implementation.
"""
from dataclasses import replace
import math
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit

from . import track_d_head_tail_artificial as old
from .c1_h4_strong_baseline import vacuum_phase

c3 = old.c3


def certified_basis(context, index, actual=None, *, rotation=None, wires=tuple(range(8))):
    model = context.arm.model
    if model.modes != 8 or tuple(wires) != tuple(range(8)):
        raise ValueError("FIXED_H4_SYSTEM_WIRES")
    block = model.blocks[index]
    if rotation is not None and rotation != block.rotation:
        raise ValueError("FRAME_ROTATION_BINDING")
    expected = context.builder._basis(model, block)
    if expected.num_qubits != 8 or expected.num_clbits or expected.parameters:
        raise ValueError("BOUND_EIGHT_MODE_BASIS")
    vacuum_phase(expected)  # Refuses non-Gaussian/unknown analytic gauge gates.
    if actual is not None and not old.same_ir(expected, actual):
        raise ValueError("ACTUAL_FRAME_OR_GAUGE_BINDING")
    return expected.copy()


def splice_join(core, tail_start, context, tail_index, *, mode, head_index=2):
    if mode not in ("ordinary", "physical"):
        raise ValueError("TAIL_CONTROL_MODE_CHANGED")
    if (core.num_qubits != (9 if mode == "ordinary" else 8)
            or core.num_clbits or core.parameters):
        raise ValueError("CORE_REGISTER")
    gh = certified_basis(context, head_index)
    gt = certified_basis(context, tail_index)
    expected = QuantumCircuit(core.num_qubits)
    expected.compose(gh, range(8), inplace=True)
    expected.compose(gt.inverse(), range(8), inplace=True)
    start, stop = tail_start-len(gh.data), tail_start+len(gt.data)
    if start < 0 or stop > len(core.data) or old.signature(core)[start:stop] != old.signature(expected):
        raise ValueError("NONADJACENT_OR_UNBOUND_HEAD_TAIL_INTERFACE")
    model = context.arm.model
    if old.same_ir(gh, gt):
        bridge = QuantumCircuit(8)
        method = "exact_actual_basis_inverse_pair"
    else:
        bridge = context.builder.transition(model, model.blocks[head_index], model.blocks[tail_index]).copy()
        method = "existing_phase_corrected_relative_transition"
    vacuum_phase(bridge)
    result = QuantumCircuit(core.num_qubits)
    result.global_phase = math.fsum([float(core.global_phase), -float(gh.global_phase), float(gt.global_phase)])
    for item in core.data[:start]:
        result.append(item.operation, [core.find_bit(b).index for b in item.qubits])
    result.compose(bridge, range(8), inplace=True)
    for item in core.data[stop:]:
        result.append(item.operation, [core.find_bit(b).index for b in item.qubits])
    return result, dict(head_block=head_index, tail_block=tail_index, mode=mode,
        wires=list(range(8)), ancilla=8 if mode == "ordinary" else None,
        method=method, original_instruction_span=[start, stop],
        removed_instructions=stop-start, inserted_instructions=len(bridge.data),
        head_rotation=dict(real=np.asarray(model.blocks[head_index].rotation).real.tolist(),
                           imag=np.asarray(model.blocks[head_index].rotation).imag.tolist()),
        tail_rotation=dict(real=np.asarray(model.blocks[tail_index].rotation).real.tolist(),
                           imag=np.asarray(model.blocks[tail_index].rotation).imag.tolist()),
        head_declared_phase=float(gh.global_phase), tail_declared_phase=float(gt.global_phase),
        bridge_declared_phase=float(bridge.global_phase),
        head_vacuum_phase=vacuum_phase(gh), tail_vacuum_phase=vacuum_phase(gt),
        bridge_vacuum_phase=vacuum_phase(bridge), matrix_order="G_tail.adjoint @ G_head")


def prepare(context, record, saved, *, on_call=lambda kind: None):
    """Reconstruct OFF IR, reuse saved wrappers/targets, emit ON exactly once.

    prepare_steps owns signed-event validation and unchanged head/tail assembly.
    Its wrapper callback returns the exact saved OFF wrappers after comparing
    the core. Thus no new OFF full wrapper, sampler or CAR target is generated.
    """
    base = c3.base
    cores = {m: base.load_qpy(saved/"circuits"/(m+"_core.qpy"))
             for m in ("symmetric_directional", "reference")}
    wrappers = {(m, a): base.load_qpy(saved/"circuits"/(m+"_"+a+"_raw.qpy"))
                for m in cores for a in base.AXES}
    halves = {k: base.load_qpy(saved/"circuits"/(v+".qpy"))
              for k, v in (("earlier", "B"), ("later", "A"), ("later_adjoint", "A_actual_adjoint"))}
    with np.load(saved/"references.npz", allow_pickle=False) as z:
        target = z["V"].copy()
    raw = base.load_qpy(saved/"circuits/ordinary_raw.qpy")
    if target.shape != (256, 256):
        raise ValueError("SAVED_FULL_CAR_TARGET")
    steps = [dict(sample=f"q4_sample00_step{i:02d}", events=base.stored_events(rows),
                  reference=np.eye(256, dtype=complex), raw=QuantumCircuit(9))
             for i, rows in enumerate(record["ordered_events"])]
    tails = []
    emitter = base.build_event_sequence
    shell = base.artificial.hadamard_wrapper
    off_wrapper_reuse = []

    def capture(*args, **kwargs):
        tail = emitter(*args, **kwargs)
        tails.append(tail.copy())
        return tail

    def reuse_wrapper(core, axis):
        matches = [m for m in cores if old.same_ir(core, cores[m])]
        if len(matches) != 1:
            raise ValueError("OFF_CANONICAL_IR_OR_DECLARED_PHASE")
        key = (matches[0], axis)
        if key in off_wrapper_reuse:
            raise ValueError("DUPLICATE_OFF_WRAPPER")
        off_wrapper_reuse.append(key)
        return wrappers[key]

    with patch.object(base, "build_event_sequence", capture), patch.object(base.artificial, "hadamard_wrapper", reuse_wrapper):
        baseline = c3.prepare_steps(context, steps, 2,
            ordered_samples=tuple(s["sample"] for s in steps), cohort=record["cohort"],
            fixture_selection="fixed saved C6a q4_sample00; no sampling")
    if len(tails) != 4 or len(off_wrapper_reuse) != 4:
        raise ValueError("FIXED_FOUR_SAVED_STEPS")
    for name, value in halves.items():
        if not old.same_ir(getattr(baseline, name), value):
            raise ValueError("OFF_HALF_IR_OR_PHASE:"+name)
    baseline = replace(baseline, cores=cores, wrappers=wrappers, reference=target, saved_raw=raw)
    atoms = {a.atom_id: a for a in context.arm.plan.atoms}
    indices = []
    for step in steps:
        atom = atoms[step["events"][0].rotation_component_id]
        rotation = context.arm.plan.frames[atom.frame_id]
        candidates = [i for i, b in enumerate(context.arm.model.blocks) if b.rotation == rotation]
        if not candidates:
            raise ValueError("TAIL_ACTUAL_FRAME_BINDING")
        index = candidates[0]
        certified_basis(context, index, context.provider.basis(atom, singleton=False), rotation=rotation)
        indices.append(index)
    first = c3.mixed_head(context, c3.head_recipe("first"))
    boundary = c3.mixed_head(context, c3.head_recipe("boundary"))
    offsets = [len(first.data)]
    for tail in tails[:-1]:
        offsets.append(offsets[-1]+len(tail.data)+len(boundary.data))
    symmetric = cores["symmetric_directional"]
    records = []
    on_call("fused_S1_core")
    for i in reversed(range(4)):
        symmetric, row = splice_join(symmetric, offsets[i], context, indices[i], mode="ordinary")
        records.append(dict(configuration="S1", step=i, status="FUSED", **row))
    fused_halves = []
    on_call("fused_R1_core")
    physical_boundary = base.control_one(context.builder.sequence(context.arm.model,
        tuple((i, t) for i, t, _ in c3.head_recipe("boundary")), optimized=True, scalar_time=0))
    for half_index, original in enumerate((baseline.earlier, baseline.later)):
        part = original
        local_offsets = [len(context.forward.data),
                         len(context.forward.data)+len(base.control_one(tails[2*half_index]).data)+len(physical_boundary.data)]
        for j in (1, 0):
            i = 2*half_index+j
            part, row = splice_join(part, local_offsets[j], context, indices[i], mode="physical")
            records.append(dict(configuration="R1", half="B" if half_index == 0 else "A", step=i, status="FUSED", **row))
        fused_halves.append(part)
    B, A = fused_halves
    fused_cores = dict(symmetric_directional=symmetric, reference=base.reference_gadget(A, B, baseline.phase))
    fused_wrappers = {}
    for method, core in fused_cores.items():
        for axis in base.AXES:
            on_call("new_full_wrappers")
            fused_wrappers[method, axis] = shell(core, axis)
    fused = replace(baseline, cores=fused_cores, wrappers=fused_wrappers, earlier=B, later=A,
                    later_adjoint=A.inverse(), identity=dict(baseline.identity, head_tail_opt_in=True))
    return baseline, fused, records, indices
