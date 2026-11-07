"""N4-B adapter: fixed provider and unchanged lower_native, after compiler A.

No synthesis, compilation, phase fitting, new control, or price aggregation.
The caller supplies the qualified DictionaryProvider and original lower_native.
Certificates cover RZ replacement; additional shell phase operations are exposed
separately. This module does not certify RPE success.
"""
from __future__ import annotations

import math
from qiskit import QuantumCircuit

UNITARY = frozenset(("rz", "sx", "x", "cx"))
FENCES = frozenset(("reset", "measure", "barrier"))


def consume_native(native, provider, lower_native):
    """Replace numeric native RZs, preserving wires and nonunitary fences.

    Word choice has no arm/role argument. Each segment starts at phase zero;
    only its unitary operations enter lower_native. Joining is always inplace.
    Nothing is recompiled or converted to an Instruction/QPY. Unsupported
    instructions/conditions fail before any materialization.
    """
    if native.parameters:
        raise ValueError("Unbound native parameters")
    phase = float(native.global_phase)
    if not math.isfinite(phase) or not 0 <= phase <= 2 * math.pi:
        raise ValueError("Unqualified native phase")
    certificates = {}
    positives = {}
    for inst in native.data:
        op = inst.operation
        if op.name not in UNITARY | FENCES or getattr(op, "condition", None) is not None:
            raise ValueError("Unsupported native operation/condition: " + op.name)
        if op.name == "rz":
            # Actual post-compiler literal, never ideal SOURCE reconstruction.
            theta = float(op.params[0])
            key = theta.hex()
            if key not in certificates:
                certificates[key] = provider.provide(theta)
                positives[abs(theta).hex()] = certificates[key].positive_sequence

    out = QuantumCircuit(native.qubits, native.clbits, name=native.name + "_fixed_CT")
    for register in (*native.qregs, *native.cregs):
        out.add_register(register)
    out.global_phase = phase
    audit = dict(native_global_phase_hex=phase.hex(),
        initial_outer_phase_hex=float(out.global_phase).hex(),
        outer_phase_settings=1, extra_segment_joins=0,
        phase_extra_local_1q_Frobenius_bound="(16 + 64 * nonempty_unitary_segments) * 2^-53",
        RZ_positions=[], segments=[], fences=[],
        certificates={k: v.summary() for k, v in sorted(certificates.items())},
        copying_compose=0, new_QPY_roundtrips=0, new_controls=0)
    segment = QuantumCircuit(native.num_qubits)
    start = None

    def flush(end):
        nonlocal segment, start
        if not segment.data:
            return
        used = {abs(float(x.operation.params[0])).hex()
                for x in segment.data if x.operation.name == "rz"}
        # Sequence-only carriers are NOT backend records or new strict guards.
        carriers = {(k, -11): dict(sequence=positives[k]) for k in used}
        allocation = [dict(role="common_native", angle=k, exponent=-11) for k in used]
        lowered = lower_native(segment, allocation, carriers, "common_native")
        before = float(out.global_phase).hex()
        out.compose(lowered, qubits=list(range(native.num_qubits)), inplace=True)
        audit["segments"].append(dict(native_start=start, native_end_exclusive=end,
            native_operations=len(segment.data), lowered_operations=len(lowered.data),
            segment_input_phase_hex=float(segment.global_phase).hex(),
            segment_emitted_phase_hex=float(lowered.global_phase).hex(),
            outer_before_hex=before, outer_after_hex=float(out.global_phase).hex()))
        audit["extra_segment_joins"] += 1
        segment = QuantumCircuit(native.num_qubits)
        start = None

    for position, inst in enumerate(native.data):
        qs = [native.find_bit(q).index for q in inst.qubits]
        cs = [native.find_bit(c).index for c in inst.clbits]
        if inst.operation.name in UNITARY:
            if start is None:
                start = position
            segment.append(inst.operation, qs, copy=False)
            if inst.operation.name == "rz":
                audit["RZ_positions"].append(dict(native_position=position,
                    qubits=qs, literal_hex=float(inst.operation.params[0]).hex()))
        else:
            flush(position)
            out.append(inst.operation, qs, cs, copy=False)
            audit["fences"].append(dict(native_position=position,
                output_position=len(out.data)-1, name=inst.operation.name,
                qubits=qs, clbits=cs))
    flush(len(native.data))
    audit["output_global_phase_hex"] = float(out.global_phase).hex()
    assert out.qubits == native.qubits and out.clbits == native.clbits
    return out, audit
