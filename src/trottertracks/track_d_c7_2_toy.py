"""C7-2 artificial-only structural probes; no molecular I/O or sampling.

Use the actual StrongBuilder, full-frame tail sharing and C3 prepare_steps.
Only the two-mode register constant and the explicitly artificial metadata are
adapted. This module never relaxes a molecular request/price guard.
"""
from fractions import Fraction
import math
from types import SimpleNamespace as S
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from . import track_d_c3_whole_path as c3
from .c1_stage2_native import NativeBlock, NativeModel
from .pr_dictionary_validation.prototype import MixedAtom
from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives as D, append_diagonal_primitives

ATOL, OBS_ATOL = 1e-10, 2e-10
METRICS = ("rz_count", "cx_count", "rz_depth", "cx_depth", "total_depth", "circuit_size", "qubit_count")


def check(actual, expected, label, *, observable=False):
    error = np.asarray(actual) - np.asarray(expected)
    maximum, frobenius = float(np.max(abs(error))), float(np.linalg.norm(error))
    passed = frobenius <= OBS_ATOL if observable else maximum <= ATOL
    record = dict(label=label, max_entry=maximum, frobenius=frobenius,
                  tolerance=OBS_ATOL if observable else ATOL, criterion="frobenius" if observable else "max_entry",
                  rtol=0, phase_fit=False, status="PASS" if passed else "FAIL")
    if not passed:
        raise ValueError(record)
    return record


def diagonal_matrix(primitives):
    z = np.array([[1-2*((state >> bit) & 1) for bit in range(2)] for state in range(4)])
    phase = np.full(4, primitives.global_phase, dtype=float)
    for bit, angle in primitives.rz:
        phase -= angle*z[:, bit]/2
    for left, right, angle in primitives.rzz:
        phase -= angle*z[:, left]*z[:, right]/2
    return np.diag(np.exp(1j*phase))


def controlled(matrix):
    out = np.eye(2*len(matrix), dtype=complex)
    out[len(matrix):, len(matrix):] = matrix
    return out


def weighted_depth(circuit, gate=None):
    """Qiskit 1.3 depth recurrence; excluded gates still synchronize wires.

    Narrow artificial domain: no conditions, variables, directives or resets.
    This is a feature extractor, not a circuit simulator or compiler.
    """
    finish = {bit: 0 for bit in (*circuit.qubits, *circuit.clbits)}
    for item in circuit.data:
        if getattr(item.operation, "condition", None) is not None or getattr(item.operation, "_directive", False):
            raise ValueError("TOY_DEPTH_DOMAIN")
        bits = (*item.qubits, *item.clbits)
        value = max((finish[bit] for bit in bits), default=0) + int(gate is None or item.operation.name == gate)
        for bit in bits:
            finish[bit] = value
    return dict(depth=max(finish.values(), default=0),
                qubit_finish=[finish[bit] for bit in circuit.qubits],
                classical_finish=[finish[bit] for bit in circuit.clbits])


def structure(circuit):
    from qiskit.converters import circuit_to_dag
    value = c3.base.artificial.metrics(circuit)
    depths = {key: weighted_depth(circuit, gate) for key, gate in
              (("total_depth", None), ("rz_depth", "rz"), ("cx_depth", "cx"))}
    if any(depths[key]["depth"] != value[key] for key in depths):
        raise ValueError("DEPTH_RECURRENCE_DISAGREEMENT")
    path = [node for node in circuit_to_dag(circuit).longest_path() if hasattr(node, "op")]
    return dict(metrics=value, counts=dict(circuit.count_ops()), weighted_wire_finish=depths,
                longest_path_operations=[node.op.name for node in path],
                operations=[dict(name=x.operation.name,
                    qubits=[circuit.find_bit(b).index for b in x.qubits],
                    parameters=[float(p) for p in x.operation.params] if x.operation.name in {"rz", "p", "crz", "rzz"} else [])
                    for x in circuit.data])


def event(atom, theta):
    sign = 1 if atom.coefficient > 0 else -1
    app = S(component_id=atom.atom_id, coefficient_sign=sign,
            coefficient_abs=abs(float(atom.coefficient)), basis_id=atom.basis_key,
            is_identity=False, role="rotation", diagonal_pauli_support=atom.support)
    return S(rotation_component_id=atom.atom_id, application_sequence=(app,),
             selected_component_ids=(atom.atom_id,), rotation_angle=theta,
             unsigned_rotation_angle=sign*theta, event_normalization=math.sqrt(1+.2**2), phase=1+0j)


def context(spec, lift):
    blocks = []
    for i, angle in enumerate(spec["frame_angles"]):
        u = ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
        rz = ((0, .21+.1*i), (1, -.17))
        rzz = () if i == 0 else ((0, 1, (-1)**i*.19),)
        # Every artificial deterministic block has vacuum energy zero.
        primitive = D(math.fsum([v for _, v in rz]+[v for _, _, v in rzz])/2, rz, rzz)
        blocks.append(NativeBlock(str(i), u, primitive))
    model = NativeModel("C7-2_ARTIFICIAL_TWO_MODE", tuple(blocks), spec["scalar"],
                        "artificial-C7-2-source", "artificial-C7-2-input")
    builder = c3.base.StrongBuilder(relative_fusion=True)
    atoms = tuple(MixedAtom(name, "FRAME_Z", Fraction(coefficient), ("artificial",), tuple(word), str(frame), str(frame))
                  for name, coefficient, word, frame in spec["atoms"])
    class FullFrame(c3.base.BasisProvider):
        def basis(self, atom, singleton):
            return builder._basis(model, model.blocks[int(atom.frame_id)]).copy()
    provider = FullFrame(2, {str(i): b.rotation for i, b in enumerate(blocks)})
    half = tuple((i, Fraction(1, 20)) for i in range(3))
    forward = c3.base.control_one(builder.sequence(model, half, optimized=True, scalar_time=0))
    backward = c3.base.control_one(builder.sequence(model, tuple(reversed(half)), optimized=True, scalar_time=0))
    scalar = -(spec["scalar"]+spec["identity"])*float(c3.base.H)
    c = S(builder=builder, arm=S(model=model, plan=S(atoms=atoms, extracted_identity=spec["identity"])),
          forward=forward, backward=backward, scalar_phase=scalar,
          config=S(dimensionless_step_time=spec["tau"]), provider=provider, tail=S(lambda_r=2.))
    lifts = [lift(np.asarray(b.rotation, complex)) for b in blocks]
    checks = []
    for b, exterior in zip(blocks, lifts):
        # Analytic RZ vacuum phase, not a fitted phase from a matrix entry.
        basis = builder._basis(model, b)
        from .c1_h4_strong_baseline import vacuum_phase
        checks.append(check(Operator(basis).data, np.exp(1j*vacuum_phase(basis))*exterior, "exterior_basis_"+b.block_id))
    ds = [u@diagonal_matrix(c3.base.combined_diagonal(model, ((i, Fraction(1,20)),)))@u.conj().T
          for i, u in enumerate(lifts)]
    head = ds[2]@ds[1]@ds[0]
    reverse = ds[0]@ds[1]@ds[2]
    return c, lifts, head, reverse, checks


def make_cases(spec, lift):
    """Construct only the predeclared 54 artificial circuits; no result selection."""
    cases, bundles, fixture_checks = [], {}, {}
    def add(name, circuit, expected=None, fixture=None, role="local"):
        cases.append(dict(name=name, raw=circuit, expected=expected, fixture=fixture, role=role))
    for item in spec["diagonals"]:
        d = D(item["phase"], tuple(tuple(x) for x in item["rz"]), tuple(tuple(x) for x in item["rzz"]))
        matrix = diagonal_matrix(d)
        for mode in ("uncontrolled", "ordinary", "directional"):
            qc = QuantumCircuit(2 if mode == "uncontrolled" else 3)
            if mode == "directional":
                c3.base.append_directional_diagonal(qc, d, 2)
                target = np.zeros((8, 8), complex); target[:4, :4] = matrix.conj().T; target[4:, 4:] = matrix
            else:
                append_diagonal_primitives(qc, d, controlled=mode == "ordinary", ancilla_qubit=2 if mode == "ordinary" else None)
                target = controlled(matrix) if mode == "ordinary" else matrix
            add("diagonal_"+item["id"]+"_"+mode, qc, target)
    for item in spec["centers"]:
        qc = QuantumCircuit(3); physical = np.eye(4, dtype=complex)
        for support, angle in item["sequence"]:
            from .pr_dictionary_validation.circuits import append_central
            append_central(qc, tuple(support), 2, angle)
            z = np.array([np.prod([1-2*((state>>bit)&1) for bit in support]) for state in range(4)])
            physical = np.diag(np.exp(-1j*angle*z))@physical
        add("center_"+item["id"]+"_controlled", qc, controlled(physical))
        add("center_"+item["id"]+"_physical", c3.base.control_one(qc), physical)
    for item in spec["cx_boundaries"]:
        qc = QuantumCircuit(2); qc.cx(0,1)
        if item["rz_wire"] is not None:
            qc.rz(.23, item["rz_wire"])
        qc.cx(0,1)
        expected = QuantumCircuit(2)
        if item["rz_wire"] == 0:
            expected.rz(.23, 0)
        elif item["rz_wire"] == 1:
            expected.rzz(.23, 0, 1)
        add("cx_"+item["id"], qc, Operator(expected).data)
    qc = QuantumCircuit(3); qc.cswap(2, 0, 1)
    # Exact controlled-swap permutation in little-endian wire order.
    permutation = np.zeros((8,8), complex)
    for x in range(8):
        y = x if not (x&4) else ((x&4)|((x&1)<<1)|((x&2)>>1))
        permutation[y,x] = 1
    add("isolated_cswap", qc, permutation)
    with patch.object(c3.base, "N", 2):
        for item in spec["whole_paths"]:
            c, lifts, head, reverse, checks = context(spec, lift)
            by_id = {a.atom_id: a for a in c.arm.plan.atoms}
            steps = []
            for i, ids in enumerate(item["steps"]):
                events = tuple(event(by_id[name], math.atan(spec["tau"])) for name in ids)
                tail_matrix = np.eye(4, dtype=complex)
                for e in events:
                    atom = by_id[e.rotation_component_id]; u = lifts[int(atom.frame_id)]
                    z = np.array([np.prod([1-2*((state>>bit)&1) for bit in atom.support]) for state in range(4)])
                    tail_matrix = (u@np.diag(np.exp(-1j*e.unsigned_rotation_angle*z))@u.conj().T)@tail_matrix
                target = np.exp(1j*c.scalar_phase)*reverse@tail_matrix@head
                steps.append(dict(sample=i, events=events, reference=target, raw=QuantumCircuit(3)))
            bundle = c3.prepare_steps(c, steps, item["cut"], ordered_samples=tuple(range(len(steps))),
                fixture_selection="predeclared artificial word; no sampling", cohort="C7-2_ARTIFICIAL")
            # Never export prepare_steps' molecular description for a toy.
            bundle.identity = dict(artificial=True, modes=2, q=len(steps), cut=item["cut"],
                model="artificial two-mode, three native blocks; not H4", scalar=spec["scalar"],
                extracted_identity=spec["identity"], phase_chi=bundle.phase,
                strong_symmetric=bundle.identity["strong_symmetric"],
                reference_optimization=bundle.identity["reference_optimization"])
            a,b,adj = [Operator(x).data for x in (bundle.later, bundle.earlier, bundle.later_adjoint)]
            checks += [check(a@b, bundle.reference, "AB=V"), check(adj,a.conj().T,"actual_adjoint"),
                       check(bundle.reference[:,0],np.exp(1j*bundle.phase)*np.eye(4)[:,0],"vacuum_chi")]
            checks += [check(Operator(bundle.cores["symmetric_directional"]).data.conj().T@Operator(bundle.cores["symmetric_directional"]).data,np.eye(8),"strong_unitarity")]
            strong=Operator(bundle.cores["symmetric_directional"]).data
            checks += [check(strong[:4,:4].conj().T@strong[4:,4:],bundle.reference,"strong_direct_V_coherence"),
                       check(strong[:4,4:],np.zeros((4,4)),"strong_control_block")]
            for method in ("symmetric_directional", "reference"):
                for axis in ("X", "Y"):
                    wrapper=bundle.wrappers[method,axis]
                    checks.append(check(c3.base.artificial.induced_observable(wrapper),
                        c3.base.artificial.expected_observable(bundle.reference,axis),method+"_raw_"+axis,observable=True))
                    if method == "reference":
                        matrix=Operator(wrapper.remove_final_measurements(inplace=False)).data
                        rows=[i for i in range(32) if i & 12]
                        checks.append(check(matrix[rows,:4],np.zeros((len(rows),4)),"raw_reference_leakage"))
                    add(item["id"]+"_"+method+"_"+axis,wrapper,fixture=item["id"],role="full_wrapper")
            add(item["id"]+"_A_inverse",bundle.later_adjoint,adj,fixture=item["id"],role="lane")
            add(item["id"]+"_B",bundle.earlier,b,fixture=item["id"],role="lane")
            shell_a=QuantumCircuit(2);shell_a.global_phase=bundle.phase
            shell_b=QuantumCircuit(2)
            shell=c3.base.reference_gadget(shell_a,shell_b,bundle.phase)
            for axis in ("X","Y"):
                wrapper=c3.base.artificial.hadamard_wrapper(shell,axis)
                checks.append(check(c3.base.artificial.induced_observable(wrapper),
                    c3.base.artificial.expected_observable(np.exp(1j*bundle.phase)*np.eye(4),axis),"valid_shell_"+axis,observable=True))
                add(item["id"]+"_shell_"+axis,wrapper,fixture=item["id"],role="shell")
            if item["id"] == "q2_same":
                inverse_pair=bundle.later.compose(bundle.later_adjoint)
                add("actual_inverse_pair",inverse_pair,np.eye(4))
            bundles[item["id"]]=bundle;fixture_checks[item["id"]]=checks
    if len(cases) != spec["compile_budget"]:
        raise ValueError("FIXED_CASE_COUNT")
    return cases, bundles, fixture_checks
