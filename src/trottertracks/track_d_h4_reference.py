"""Limited CDF8/word02 bridge. No runner, sampler, fit, or ground solve.

The original C source is imported from its separate frozen checkout. Existing
StrongBuilder, signed-event emission, directional diagonal and Track D shells
are reused. Only the head mode, fixed-control restriction, and register size
are adapted here.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import Operator

from trotterlib.rte import RTEEvent, RTEEventApplication
from trotterlib import track_d_minimal_reference as artificial
from . import c_pr_aware_df_stage_b as original
from .c1_h4_strong_baseline import StrongBuilder
from .c1_stage2_native import runs, combined_diagonal
from .pr_dictionary_validation.circuits import BasisProvider, build_event_sequence
from .resource_applicability.ax2b_native_df_v5 import append_directional_diagonal

N, CUT, H = 8, 16, Fraction(1, 10)
ATOL, SIGNAL_ATOL = 1e-10, 2e-10
ARMS = ("ordinary", "symmetric_directional", "reference")
AXES = ("X", "Y")


def small_operator(circuit):
    """Never materialize the 17-qubit circuit operator."""
    if circuit.num_qubits > 9:
        raise ValueError("FULL_REFERENCE_OPERATOR_FORBIDDEN")
    return Operator(circuit).data


def load_qpy(path):
    with Path(path).open("rb") as stream:
        values = qpy.load(stream)
    if len(values) != 1:
        raise ValueError("ONE_SAVED_CIRCUIT_REQUIRED")
    return values[0]


def save_qpy(path, circuit):
    with Path(path).open("xb") as stream:
        qpy.dump(circuit, stream)


def stored_events(rows):
    """Deserialize the saved events; never enumerate or sample replacements."""
    events = []
    for row in rows:
        data = dict(row)
        data["phase"] = complex(**dict(real=row["phase"]["real"], imag=row["phase"]["imag"]))
        for field in ("product_component_ids", "selected_component_ids", "df_fragment_ids"):
            data[field] = tuple(data[field])
        data["basis_reuse_intervals"] = tuple(tuple(x) for x in data["basis_reuse_intervals"])
        applications = []
        for value in data["application_sequence"]:
            value = dict(value)
            value["basis_change_operations"] = tuple(value["basis_change_operations"])
            if value["diagonal_pauli_support"] is not None:
                value["diagonal_pauli_support"] = tuple(value["diagonal_pauli_support"])
            applications.append(RTEEventApplication(**value))
        data["application_sequence"] = tuple(applications)
        events.append(RTEEvent(**data))
    if json.dumps([e.to_dict() for e in events], sort_keys=True) != json.dumps(rows, sort_keys=True):
        raise ValueError("SAVED_EVENT_ROUNDTRIP")
    return tuple(events)


def control_one(circuit):
    """Restrict an existing diagonal-only control to ancilla=1, gate by gate.

    System gates and declared phases survive. A controlled central gate is
    replaced with its existing base gate. No dense synthesized gate is emitted.
    The RZ/CX central recipe becomes RZ/X and the same P0 optimizer simplifies
    it. Unexpected ancilla targets or operations fail rather than guessing.
    """
    if circuit.num_clbits or circuit.parameters or circuit.num_qubits < 2:
        raise ValueError("BOUND_CONTROLLED_KERNEL_REQUIRED")
    anc = circuit.num_qubits - 1
    out = QuantumCircuit(anc)
    out.global_phase = circuit.global_phase
    for item in circuit.data:
        gate = item.operation
        q = [circuit.find_bit(bit).index for bit in item.qubits]
        if anc not in q:
            out.append(gate, q)
        elif q == [anc] and gate.name in ("p", "rz", "z", "s", "sdg"):
            angle = {"z": math.pi, "s": math.pi / 2, "sdg": -math.pi / 2}.get(gate.name)
            if gate.name == "p": angle = float(gate.params[0])
            if gate.name == "rz": angle = float(gate.params[0]) / 2
            out.global_phase += angle
        elif isinstance(gate, ControlledGate) and gate.num_ctrl_qubits == 1 and gate.ctrl_state == 1 and q[0] == anc:
            out.append(gate.base_gate, q[1:])
        else:
            raise ValueError("UNSUPPORTED_CONTROL_ONE_GATE:" + gate.name + str(q))
    return out


def directional_head(builder, model, schedule):
    """StrongBuilder's same runs/relative transitions with directional centers."""
    if not builder.relative_fusion:
        raise ValueError("STRONG_RELATIVE_BASELINE_REQUIRED")
    groups = runs(model, schedule, optimized=True)
    if not groups:
        raise ValueError("EMPTY_HEAD")
    qc = QuantumCircuit(model.modes + 1)
    first = model.blocks[groups[0][0][0]]
    qc.compose(builder._basis(model, first).inverse(), range(model.modes), inplace=True)
    for index, group in enumerate(groups):
        block = model.blocks[group[0][0]]
        append_directional_diagonal(qc, combined_diagonal(model, group), model.modes)
        if index + 1 < len(groups):
            following = model.blocks[groups[index + 1][0][0]]
            qc.compose(builder.transition(model, block, following), range(model.modes), inplace=True)
        else:
            qc.compose(builder._basis(model, block), range(model.modes), inplace=True)
    return qc


def reference_gadget(later, earlier, phase):
    """Same two-cascade Track D recipe, with explicit n-mode registers."""
    n = later.num_qubits
    if earlier.num_qubits != n or later.num_clbits or earlier.num_clbits:
        raise ValueError("SAME_UNITARY_SYSTEM_REGISTERS_REQUIRED")
    qc = QuantumCircuit(2 * n + 1)
    anc = 2 * n
    for system in range(n): qc.cswap(anc, system, n + system)
    qc.compose(later.inverse(), range(n), inplace=True)
    qc.compose(earlier, range(n, 2 * n), inplace=True)
    for system in range(n): qc.cswap(anc, system, n + system)
    qc.p(phase, anc)
    return qc


@dataclass
class Bundle:
    arm: object
    events: tuple
    cores: dict
    wrappers: dict
    earlier: QuantumCircuit
    later: QuantumCircuit
    later_adjoint: QuantumCircuit
    reference: np.ndarray
    saved_raw: QuantumCircuit
    phase: float
    identity: dict
    fusion: list


def prepare(origin, saved, contract):
    """Only the permitted fixed input, model preparation, and saved word."""
    origin, saved = Path(origin), Path(saved)
    with np.load(origin / contract["input"], allow_pickle=False) as z:
        packet = {k: z[k] for k in ("h", "g", "constant")}
    with np.load(origin / contract["candidate"], allow_pickle=False) as z:
        endpoint = {k: z[k] for k in ("J", "U")}
    with np.load(origin / contract["sector"], allow_pickle=False) as z:
        if not all(np.array_equal(endpoint[k], z[k]) for k in ("J", "U")):
            raise ValueError("SAVED_JU_REFERENCE_BINDING")
    if endpoint["J"].shape != (8, 4, 4) or endpoint["U"].shape != (8, 4, 4):
        raise ValueError("FIXED_CDF8_INPUT")
    arm = original.prepare_arm(**packet, **endpoint, role="proposed",
        source_identity=contract["saved_inputs"][contract["candidate"]],
        input_identity=contract["saved_inputs"][contract["input"]], ld=2)
    if arm.model.modes != N:
        raise ValueError("FIXED_EIGHT_MODES")
    atoms = json.loads((origin / contract["atoms"]).read_text())["atoms"]
    actual = {a.atom_id: a for a in arm.plan.atoms}
    for row in atoms:
        atom = actual[row["atom_id"]]
        if atom.frame_id != row["frame_id"] or list(atom.support) != row["support"] or Fraction(atom.coefficient) != Fraction(row["coefficient"]):
            raise ValueError("SAVED_SIGNED_ATOM")
    if len(actual) != len(atoms): raise ValueError("SAVED_ATOM_COUNT")
    tail, config, distribution = original.configure(arm, H, 32, 0)
    record = json.loads((saved / "samples/word02_events.json").read_text())
    events = stored_events(record["events"])
    wanted = contract["words"][2]
    if wanted["id"] != "alternating_frames" or [e.rotation_component_id for e in events] != wanted["atom_ids"]:
        raise ValueError("FIXED_WORD02")
    if len(events) != 32 or any(e.taylor_order or e.product_component_ids or e.phase != 1 for e in events):
        raise ValueError("FIXED_K0_EVENTS")
    for e in events:
        a = actual[e.rotation_component_id]
        app = e.application_sequence[0]
        if (app.coefficient_sign != (1 if a.coefficient > 0 else -1)
                or app.coefficient_abs != abs(float(a.coefficient))
                or app.basis_id != a.basis_key or app.diagonal_pauli_support != a.support
                or not math.isclose(e.rotation_angle, math.atan(config.dimensionless_step_time), rel_tol=1e-14, abs_tol=1e-15)):
            raise ValueError("EVENT_SIGN_FRAME_OR_ANGLE")
    old_law = json.loads((origin / "artifacts/research_c_near_practical_r32/2026-11-03/run01/PR/A/law.json").read_text())
    if arm.plan.extracted_identity != old_law["identity"]:
        raise ValueError("EXTRACTED_IDENTITY_CHANGED")
    if not math.isclose(tail.lambda_r, old_law["lambda_R"], rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("TAIL_NORMALIZATION_CHANGED")
    builder = StrongBuilder(relative_fusion=True)
    ordinary = original.build_kernel(arm, H, events, config, builder=builder)
    forward_schedule = tuple((i, H / 2) for i in range(3))
    backward_schedule = tuple(reversed(forward_schedule))
    forward = builder.sequence(arm.model, forward_schedule, optimized=True, scalar_time=0)
    backward = builder.sequence(arm.model, backward_schedule, optimized=True, scalar_time=0)
    class Provider(BasisProvider):
        def basis(self, atom, singleton=False):
            for block in arm.model.blocks:
                if block.rotation == arm.plan.frames[atom.frame_id]:
                    return builder._basis(arm.model, block).copy()
            raise ValueError("UNKNOWN_ARM_BASIS")
    provider = Provider(N, arm.plan.frames)
    tail_c = build_event_sequence(events, arm.plan.atoms, provider, optimized=True)
    phase_scalar = -(arm.model.constant + arm.plan.extracted_identity) * float(H)
    phase = math.fsum([phase_scalar, *[math.atan2(e.phase.imag, e.phase.real) - e.unsigned_rotation_angle for e in events]])
    forward_u, backward_u = control_one(forward), control_one(backward)
    symmetric = QuantumCircuit(N + 1)
    symmetric.compose(forward_u, range(N), inplace=True)
    symmetric.compose(tail_c, inplace=True)
    symmetric.compose(directional_head(builder, arm.model, backward_schedule), inplace=True)
    symmetric.p(phase_scalar, N)
    first = control_one(build_event_sequence(events[:CUT], arm.plan.atoms, provider, optimized=True))
    last = control_one(build_event_sequence(events[CUT:], arm.plan.atoms, provider, optimized=True))
    earlier = forward_u.compose(first)
    later = last.compose(backward_u)
    later.global_phase += phase_scalar
    reference = reference_gadget(later, earlier, phase)
    cores = dict(ordinary=ordinary, symmetric_directional=symmetric, reference=reference)
    wrappers = {(name, axis): artificial.hadamard_wrapper(core, axis) for name, core in cores.items() for axis in AXES}
    with np.load(saved / "references/word02.npz", allow_pickle=False) as z:
        target = z["system_reference"]
    if target.shape != (256, 256): raise ValueError("SAVED_FULL_CAR_REFERENCE")
    info = dict(model="linear H4/1.00A/STO-3G/number CDF8", Nalpha=2, Nbeta=2,
        ld=2, h=str(H), r=32, K=0, word="word02/alternating_frames", events=32, cut_after_event=CUT,
        scalar=float(arm.model.constant), extracted_identity=arm.plan.extracted_identity,
        scalar_phase=phase_scalar, phase_chi=phase, signed_rotation_angles=[e.unsigned_rotation_angle for e in events],
        event_normalization=[e.event_normalization for e in events], lambda_R=tail.lambda_r,
        normalization_applied_to_unitary_or_observable=False, system_initial_state="exact original-H ground assumed; no ground vector used",
        state_preparation_cost="EXCLUDED", sampling_calls=0, DF_fit_calls=0, ground_solves=0,
        cross_head_tail_relative_fusion=False, head_relative_fusion=True,
        gaussian_reports=builder.basis_reports, relative_transitions=len(builder.fusion_records))
    return Bundle(arm, events, cores, wrappers, earlier, later, later.inverse(), target,
        load_qpy(saved / "circuits/word02_raw.qpy"), phase, info, builder.fusion_records)


def observable_from_columns(columns, ancilla_dimension):
    """Columns are row-stored complete input-basis state actions."""
    zero, one = columns[:, :ancilla_dimension], columns[:, ancilla_dimension:]
    return zero.conj() @ zero.T - one.conj() @ one.T


def raw_reference_columns(bundle, threads):
    from .c1_h6_state_action import StateActions
    circuit = QuantumCircuit(2 * N + 1)
    circuit.h(2 * N)
    circuit.compose(bundle.cores["reference"], inplace=True)
    inputs = np.zeros((1 << N, 1 << (2 * N + 1)), complex)
    inputs[np.arange(1 << N), np.arange(1 << N)] = 1
    return StateActions(threads).apply(circuit, inputs)


def reference_column_check(columns, u0, u1, same, label):
    """All 256 system basis inputs, full outputs including reference leakage."""
    d = len(u0); view = columns.reshape(d, 2, d, d)
    same(view[:, 0, 0, :].T, u0, label + ":branch0")
    same(view[:, 1, 0, :].T, u1, label + ":branch1")
    leakage = view[:, :, 1:, :]
    maximum = float(np.max(np.abs(leakage)))
    if not np.isfinite(maximum) or maximum > ATOL:
        raise ValueError(label + ":reference_leakage=" + str(maximum))
    return dict(max_reference_leakage_component=maximum,
        max_reference_leakage_L2=float(np.max(np.linalg.norm(leakage.reshape(d, -1), axis=1))))
