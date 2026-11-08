"""Opt-in, fixed two-mode head-exit/tail-entry experiment. No molecular entry."""
from dataclasses import replace
import math
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from . import track_d_cut_artificial as checks

toy = checks.toy
c3 = toy.c3


def signature(qc):
    """Exact native instructions/wires, separately from declared global phase."""
    return tuple((x.operation, tuple(qc.find_bit(b).index for b in x.qubits),
                  tuple(qc.find_bit(b).index for b in x.clbits)) for x in qc.data)


def same_ir(a, b):
    return (a.num_qubits == b.num_qubits and a.num_clbits == b.num_clbits
            and float(a.global_phase) == float(b.global_phase)
            and signature(a) == signature(b))


def certified_basis(context, index, actual=None, *, rotation=None, wires=(0, 1)):
    model = context.arm.model
    if model.modes != 2 or tuple(wires) != (0, 1):
        raise ValueError("FIXED_SYSTEM_WIRES")
    block = model.blocks[index]
    if rotation is not None and rotation != block.rotation:
        raise ValueError("FRAME_ROTATION_BINDING")
    expected = context.builder._basis(model, block)
    if actual is not None and not same_ir(expected, actual):
        raise ValueError("ACTUAL_FRAME_OR_GAUGE_BINDING")
    return expected.copy()


def splice_join(core, tail_start, context, tail_index, *, mode, head_index=2):
    """Replace a certified adjacent G_H, G_T†; do not move other operations."""
    if mode not in ("ordinary", "physical"):
        raise ValueError("TAIL_CONTROL_MODE_CHANGED")
    if core.num_qubits != (3 if mode == "ordinary" else 2):
        raise ValueError("CORE_REGISTER")
    gh = certified_basis(context, head_index)
    gt = certified_basis(context, tail_index)
    expected = QuantumCircuit(core.num_qubits)
    expected.compose(gh, [0, 1], inplace=True)
    expected.compose(gt.inverse(), [0, 1], inplace=True)
    start, stop = tail_start - len(gh.data), tail_start + len(gt.data)
    if start < 0 or stop > len(core.data) or signature(core)[start:stop] != signature(expected):
        raise ValueError("NONADJACENT_OR_UNBOUND_HEAD_TAIL_INTERFACE")
    model = context.arm.model
    if same_ir(gh, gt):
        bridge = QuantumCircuit(2)
        method = "exact_actual_basis_inverse_pair"
    else:
        bridge = context.builder.transition(model, model.blocks[head_index], model.blocks[tail_index]).copy()
        method = "existing_phase_corrected_relative_transition"
    result = QuantumCircuit(core.num_qubits)
    # The phases belonging to the removed circuit bodies were already composed
    # into the enclosing circuit. Subtract them, then compose the new bridge.
    result.global_phase = float(core.global_phase) - float(gh.global_phase) + float(gt.global_phase)
    for item in core.data[:start]:
        result.append(item.operation, [core.find_bit(b).index for b in item.qubits])
    result.compose(bridge, [0, 1], inplace=True)
    for item in core.data[stop:]:
        result.append(item.operation, [core.find_bit(b).index for b in item.qubits])
    return result, dict(head_block=head_index, tail_block=tail_index, mode=mode,
        head_rotation=[list(x) for x in model.blocks[head_index].rotation],
        tail_rotation=[list(x) for x in model.blocks[tail_index].rotation],
        wires=[0, 1], method=method, original_instruction_span=[start, stop],
        removed_instructions=stop-start, inserted_instructions=len(bridge.data),
        head_declared_phase=float(gh.global_phase), tail_declared_phase=float(gt.global_phase),
        bridge_declared_phase=float(bridge.global_phase),
        matrix_order="G_tail.adjoint @ G_head", centers_and_other_interfaces_unchanged=True)


def fixture(spec, lift):
    """Narrow subset entry: old make_cases would also build 53 other cases."""
    context, lifts, head, reverse, basis_checks = toy.context(spec, lift)
    item = next(x for x in spec["whole_paths"] if x["id"] == "q2_mixed")
    if item != dict(id="q2_mixed", cut=1, steps=[["plus", "zz"], ["minus", "plus"]]):
        raise ValueError("FIXED_Q2_MIXED")
    atoms = {a.atom_id: a for a in context.arm.plan.atoms}
    steps = []
    for i, names in enumerate(item["steps"]):
        events = tuple(toy.event(atoms[name], math.atan(spec["tau"])) for name in names)
        matrix = np.eye(4, dtype=complex)
        for event in events:
            atom = atoms[event.rotation_component_id]
            z = np.array([np.prod([1-2*((state >> bit) & 1) for bit in atom.support]) for state in range(4)])
            u = lifts[int(atom.frame_id)]
            matrix = (u @ np.diag(np.exp(-1j*event.unsigned_rotation_angle*z)) @ u.conj().T) @ matrix
        target = np.exp(1j*context.scalar_phase) * reverse @ matrix @ head
        steps.append(dict(sample=i, events=events, reference=target, raw=QuantumCircuit(3)))
    tails = [c3.base.build_event_sequence(s["events"], context.arm.plan.atoms,
        context.provider, optimized=True) for s in steps]
    local = local_checks(context, tails, lift)
    tail_calls = []
    def capture(*args, **kwargs):
        i = len(tail_calls)
        if i >= 2 or tuple(args[0]) != steps[i]["events"] or args[2] is not context.provider or kwargs != {"optimized": True}:
            raise ValueError("FIXED_TAIL_CACHE_BINDING")
        tail_calls.append(i)
        return tails[i].copy()
    # Process-local observation only, restored before returning. All original
    # emitters/defaults stay intact; no second tail emission is required.
    with patch.object(c3.base, "N", 2), patch.object(c3.base, "build_event_sequence", capture):
        baseline = c3.prepare_steps(context, steps, 1, ordered_samples=(0, 1),
            fixture_selection="fixed C7-2 q2_mixed", cohort="C7-2_ARTIFICIAL")
    if len(tail_calls) != 2:
        raise ValueError("EXACTLY_TWO_TAIL_EMISSIONS")
    baseline.identity = dict(artificial=True, modes=2, q=2, cut=1,
        strong=baseline.identity["strong_symmetric"],
        reference=baseline.identity["reference_optimization"], phase_chi=baseline.phase)
    first = c3.mixed_head(context, c3.head_recipe("first"))
    boundary = c3.mixed_head(context, c3.head_recipe("boundary"))
    offsets = [len(first.data), len(first.data)+len(tails[0].data)+len(boundary.data)]
    symmetric = baseline.cores["symmetric_directional"]
    records = []
    for i in (1, 0):
        index = int(atoms[steps[i]["events"][0].rotation_component_id].frame_id)
        symmetric, record = splice_join(symmetric, offsets[i], context, index, mode="ordinary")
        records.append(dict(configuration="S1", step=i, **record))
    halves = []
    for i, half in enumerate((baseline.earlier, baseline.later)):
        index = int(atoms[steps[i]["events"][0].rotation_component_id].frame_id)
        half, record = splice_join(half, len(context.forward.data), context, index, mode="physical")
        halves.append(half); records.append(dict(configuration="R1", step=i, **record))
    B, A = halves
    cores = dict(symmetric_directional=symmetric, reference=c3.base.reference_gadget(A, B, baseline.phase))
    wrappers = {(method, axis): c3.base.artificial.hadamard_wrapper(core, axis)
        for method, core in cores.items() for axis in ("X", "Y")}
    fused = replace(baseline, cores=cores, wrappers=wrappers, earlier=B, later=A,
        later_adjoint=A.inverse(), identity=dict(baseline.identity, head_tail_opt_in=True))
    return context, steps, baseline, fused, records, basis_checks, local


def local_checks(context, tails, lift):
    from .c1_h4_strong_baseline import StrongBuilder, vacuum_phase
    model = context.arm.model
    records, negatives = [], []
    for tail_index in (1, 2):
        gh, gt = certified_basis(context, 2), certified_basis(context, tail_index)
        bridge = context.builder.transition(model, model.blocks[2], model.blocks[tail_index])
        actual = Operator(bridge).data
        target = Operator(gt).data.conj().T @ Operator(gh).data
        records.append(toy.check(actual, target, f"bridge_{tail_index}:full_Fock_unitary_phase"))
        rotation = np.asarray(model.blocks[tail_index].rotation).conj().T @ np.asarray(model.blocks[2].rotation)
        records.append(toy.check(actual, np.exp(1j*vacuum_phase(bridge))*lift(rotation),
                                 f"bridge_{tail_index}:independent_exterior_lift"))
        records.append(toy.check(actual[:, 0], target[:, 0], f"bridge_{tail_index}:vacuum_phase"))
    def different(a, b, label):
        error = float(np.max(np.abs(a-b)))
        if not math.isfinite(error) or error <= toy.ATOL:
            raise ValueError("NEGATIVE_NOT_DETECTED:"+label)
        negatives.append(dict(label=label, detected=True, max_entry_difference=error))
    gh, gt = certified_basis(context, 2), certified_basis(context, 1)
    target = Operator(gt).data.conj().T @ Operator(gh).data
    wrong = context.builder.transition(model, model.blocks[1], model.blocks[2])
    different(Operator(wrong).data, target, "N1_inverse_direction")
    witness = StrongBuilder(relative_fusion=True)
    witness._basis(model, model.blocks[1]); witness._basis(model, model.blocks[2])
    key = (model.source_identity, model.input_identity, model.blocks[2].rotation)
    shifted = witness.cache[key].copy(); shifted.global_phase += math.pi/7
    witness.cache[key] = shifted  # Local witness builder only; main cache intact.
    good = witness.transition(model, model.blocks[2], model.blocks[1]).copy()
    expected = Operator(witness._basis(model, model.blocks[1])).data.conj().T @ Operator(shifted).data
    records.append(toy.check(Operator(good).data, expected, "witness_eta_pi_over_7:correct_phase"))
    bad = good.copy(); bad.global_phase -= witness.fusion_records[-1]["phase_correction"]
    different(Operator(bad).data, expected, "N2_missing_analytic_gauge")
    tail = Operator(tails[0]).data
    physical = Operator(c3.base.control_one(tails[0])).data
    wrong_mode = np.zeros_like(tail); wrong_mode[:4, :4] = physical.conj().T; wrong_mode[4:, 4:] = physical
    different(wrong_mode, tail, "N3_ordinary_to_directional")
    def rejected(label, fn):
        try:
            fn()
        except ValueError as error:
            negatives.append(dict(label=label, detected=True, rejection=str(error)))
        else:
            raise ValueError("NEGATIVE_NOT_REJECTED:"+label)
    rejected("N4_same_ID_wrong_rotation", lambda: certified_basis(context, 2, rotation=model.blocks[1].rotation))
    rejected("N4_same_ID_wrong_wires", lambda: certified_basis(context, 2, wires=(1, 0)))
    wrong_gauge = gh.copy(); wrong_gauge.global_phase += math.pi/7
    rejected("N4_same_ID_wrong_gauge", lambda: certified_basis(context, 2, wrong_gauge))
    intervening = QuantumCircuit(2); intervening.compose(gh, inplace=True)
    intervening.x(0); intervening.compose(gt.inverse(), inplace=True)
    rejected("N5_intervening_noncommuting_X", lambda: splice_join(intervening, len(gh.data)+1,
        context, 1, mode="physical"))
    return dict(status="PASS",positive=records,negative=negatives,witness_eta=math.pi/7,
                witness_in_whole_path=False,phase_fit=False)


def raw_checks(baseline, fused, steps):
    records = {name: checks.raw_checks(bundle, steps, 1) for name, bundle in (("baseline", baseline), ("fused", fused))}
    equality = []
    for name in ("earlier", "later", "later_adjoint"):
        equality.append(toy.check(Operator(getattr(fused, name)).data, Operator(getattr(baseline, name)).data,
                                  "unchanged_"+name+":full_phase"))
    for arm in baseline.cores:
        equality.append(toy.check(Operator(fused.cores[arm]).data, Operator(baseline.cores[arm]).data,
                                  "unchanged_"+arm+":full_unitary_phase"))
    S = Operator(fused.cores["symmetric_directional"]).data
    equality.append(toy.check(S, toy.controlled(fused.reference), "Strong_both_direct_control_branches"))
    return dict(status="PASS",bundles=records,phase_equality=equality)
