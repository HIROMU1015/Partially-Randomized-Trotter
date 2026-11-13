"""P0: controlled-ZZ lowering in the compiled path versus the structural S word.

Compiled path (provider c31ef424 df_rte_qiskit._append_central_application):
``RZZGate(2u).control(1)`` on [ancilla, s0, s1], with the saved basis change
around it (inverse first, forward after). Structural S (C1 Accounting, 8457c443):
``'crzz'`` = cx(s0,s1) crz(2u)(ancilla,s1) cx(s0,s1). The design below is fixed
from published C metadata (support/basis/law angles) before any compile. The
module builds circuits, counts the structural word, checks operators and calls
the existing compiled-cost function; it never samples, fits or prices.
"""
from copy import deepcopy
import json
import math

from . import pf_pr_rpe_pipeline as p
from .h4_df10_price_v1 import Accounting

C_RELEASE = '83d219cc68c25540507a5539ebd2da94c29d06b3'
C_RESULT = 'docs/research/h4_pr_physical_closure_20261113/run02/RESULT.json'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
COUNTER = '8457c443554af5fadaa3ba36f97f6316ca3c5050'
ANCILLA = 8
N_QUBITS = 9
LAW_ROUNDS = (0, 6)            # R2 (m0) and R128 (m6) laws of the fixed S0 configuration
COMPILE_BUDGET = 8
COMPILER_ARGS = (('rz', 'sx', 'x', 'cx'), None, None, 1, None, None, 17, '1.3.0')   # as C1 Backend / pf_common cost_one
FORMS = ('GENERIC_CONTROL', 'ANALYTIC_CX_CRZ_CX')
CONTEXTS = ('ALONE', 'SAVED_BASIS_PAIR')
OPERATOR_ATOL = 1e-10


def inputs(store):
    """Published C closure metadata: 324 components, 9 registered bases, R2/R128 laws."""
    result = store.read(dict(commit=C_RELEASE, path=C_RESULT))
    laws = {law['m']: law for law in result['laws']}
    return dict(components=result['preparation']['components'], bases=result['preparation']['basis_registry'],
                laws=laws, preparation=result['preparation']['hash'], H=result['preparation']['H'])


def design(data):
    """Fixed selection rule (recorded before execution): highest-probability ZZ component,
    its registered full basis, order-0 rotation angle of the R2 and R128 laws, eight circuits."""
    zz = [c for c in data['components'] if len(c['support']) == 2]
    component = sorted(zz, key=lambda c: (-c['probability'], c['id']))[0]
    basis = next(b for b in data['bases'] if b['id'] == component['basis_id'] and b['hash'] == component['basis_hash'])
    cases = []
    for m in LAW_ROUNDS:
        law = data['laws'][m]
        angle = law['rotation_angles']['0']
        unsigned = component['sign'] * angle             # Accounting: coefficient_sign * rotation_angle
        for context in CONTEXTS:
            for form in FORMS:
                cases.append(dict(id='%s|%s|m%d' % (form, context, m), form=form, context=context, m=m, R_outer=law['R_outer'],
                                  rotation_angle=angle, rotation_angle_hex=float(angle).hex(), unsigned=unsigned,
                                  gate_angle=2 * unsigned, gate_angle_hex=float(2 * unsigned).hex()))
    return dict(component={k: component[k] for k in ('id', 'basis_id', 'basis_hash', 'support', 'sign', 'probability')},
                basis=dict(id=basis['id'], hash=basis['hash'], operations=len(basis['operations']),
                           names=sorted({o['name'] for o in basis['operations']})),
                ancilla=ANCILLA, n_qubits=N_QUBITS, cases=cases, compile_budget=COMPILE_BUDGET,
                compiler=dict(args=list(COMPILER_ARGS[:1][0]), optimization_level=1, seed_transpiler=17, qiskit='1.3.0',
                              settings_hash=p.PROFILE['compiler']),
                selection_rule='max probability among two-wire Z supports; registered full basis of that component; '
                               'order-0 rotation angle of the m0 (R2) and m6 (R128) laws; sign from the component',
                not_tested='support-restricted basis choices, order-2 products/CZ, tail relative phase, heads, wrapper')


def _basis_gates(operations, inverse):
    from qiskit.circuit.library import PhaseGate, XXPlusYYGate
    out = []
    for op in operations:
        params = [float(v) for v in op['parameters']]
        gate = PhaseGate(*params) if op['name'] == 'p' else XXPlusYYGate(*params) if op['name'] == 'xx_plus_yy' else None
        if gate is None:
            raise ValueError('Unsupported saved basis primitive: ' + op['name'])
        out.append((gate, tuple(op['qubits'])))
    return [(g.inverse(), q) for g, q in reversed(out)] if inverse else out


def circuit(case, design_, data):
    """Provider order: basis inverse -> central controlled rotation -> basis forward."""
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RZZGate
    s0, s1 = design_['component']['support']
    qc = QuantumCircuit(N_QUBITS, name=case['id'])
    ops = next(b for b in data['bases'] if b['id'] == design_['basis']['id'])['operations']
    if case['context'] == 'SAVED_BASIS_PAIR':
        for gate, qubits in _basis_gates(ops, True):
            qc.append(gate, list(qubits))
    if case['form'] == 'GENERIC_CONTROL':
        qc.append(RZZGate(case['gate_angle']).control(1), [ANCILLA, s0, s1])
    else:
        qc.cx(s0, s1); qc.crz(case['gate_angle'], ANCILLA, s1); qc.cx(s0, s1)
    if case['context'] == 'SAVED_BASIS_PAIR':
        for gate, qubits in _basis_gates(ops, False):
            qc.append(gate, list(qubits))
    return qc


def structural(case, design_, data):
    """C1 Accounting word for the same application (crzz is always counted as cx-crz-cx)."""
    s0, s1 = design_['component']['support']
    ops = next(b for b in data['bases'] if b['id'] == design_['basis']['id'])['operations']
    a = Accounting()
    if case['context'] == 'SAVED_BASIS_PAIR':
        a.basis(ops, True, 'tail_basis')
    a.gate('crzz', (ANCILLA, s0, s1), (case['gate_angle'],), 'tail_rotation')
    if case['context'] == 'SAVED_BASIS_PAIR':
        a.basis(ops, False, 'tail_basis')
    r = a.finish()
    return dict(structural_rz=r['structural_rz'], expanded_rz_after_basis_cancellation=r['expanded_rz_after_basis_cancellation'],
                native_rz_by_role=r['native_rz_by_role'])


def decomposition(qc):
    """Library lowering before compile: the gates of each instruction's definition (no transpile)."""
    out = []
    for inst in qc.data:
        op = inst.operation
        if op.name in ('cx', 'crz', 'p', 'xx_plus_yy'):
            continue
        definition = op.definition
        out.append(dict(name=op.name, definition=[(d.operation.name, [definition.find_bit(q).index for q in d.qubits])
                                                  for d in definition.data],
                        nested_counts={d.operation.name: dict(d.operation.definition.count_ops()) for d in definition.data
                                       if d.operation.definition is not None}))
    return out


def operators(qc_generic, qc_analytic, *, s0, s1):
    """Exact operator comparison including global and controlled phases."""
    import numpy as np
    from qiskit.quantum_info import Operator
    g, a = Operator(qc_generic).data, Operator(qc_analytic).data
    dim = 2 ** (N_QUBITS - 1)
    # Qiskit little-endian: the ancilla (qubit 8) is the most significant index bit.
    off_block = max(np.abs(g[:dim, dim:]).max(), np.abs(g[dim:, :dim]).max())
    zero_block = np.abs(g[:dim, :dim] - np.eye(dim)).max()
    out = {}
    if not any(i.operation.name in ('p', 'xx_plus_yy') for i in qc_generic.data):    # ALONE: independent matrix
        theta = next(i.operation.base_gate.params[0] for i in qc_generic.data if i.operation.name == 'crzz')
        ref = independent_controlled_rzz(float(theta), s0, s1)
        out = dict(independent_matrix_error_generic=float(np.abs(g - ref).max()), independent_matrix_error_analytic=float(np.abs(a - ref).max()))
    return dict(out, max_abs_difference=float(np.abs(g - a).max()), equal_including_global_phase=bool(np.allclose(g, a, atol=OPERATOR_ATOL, rtol=0)),
                generic_ancilla0_block_identity_error=float(zero_block), generic_offdiagonal_block_max=float(off_block),
                controlled_structure=bool(zero_block < OPERATOR_ATOL and off_block < OPERATOR_ATOL), atol=OPERATOR_ATOL)


def independent_controlled_rzz(theta, s0, s1):
    """Diagonal |0><0| x I + |1><1| x exp(-i theta/2 Z_s0 Z_s1), built without Qiskit definitions."""
    import numpy as np
    index = np.arange(2 ** N_QUBITS)
    anc = (index >> ANCILLA) & 1; z0 = 1 - 2 * ((index >> s0) & 1); z1 = 1 - 2 * ((index >> s1) & 1)
    return np.diag(np.where(anc == 1, np.exp(-0.5j * theta * z0 * z1), 1.0 + 0j))


def compiled_equivalence(original, compiled):
    import numpy as np
    from qiskit.quantum_info import Operator
    diff = np.abs(Operator(original).data - Operator(compiled).data).max()
    return dict(max_abs_difference=float(diff), equal_including_global_phase=bool(diff < OPERATOR_ATOL), atol=OPERATOR_ATOL)


def gate_counts(qc):
    by_wire = {}
    for inst in qc.data:
        for q in inst.qubits:
            w = qc.find_bit(q).index
            by_wire.setdefault(w, {}).setdefault(inst.operation.name, 0)
            by_wire[w][inst.operation.name] += 1
    return dict(total=dict(qc.count_ops()), by_wire={str(k): v for k, v in sorted(by_wire.items())})


def ops_list(qc):
    return [[inst.operation.name, [qc.find_bit(q).index for q in inst.qubits], [float(x) for x in inst.operation.params]]
            for inst in qc.data]


def execute(store, compile_fn, settings):
    """All eight fixed cases. compile_fn(circuit, settings) -> TranspiledCircuitCost; called exactly 8 times."""
    data = inputs(store); d = design(data)
    s0, s1 = d['component']['support']
    built = {c['id']: circuit(c, d, data) for c in d['cases']}
    calls = []
    def counted(qc):
        if len(calls) >= COMPILE_BUDGET:
            raise RuntimeError('P0 compile budget exceeded')
        calls.append(qc.name)
        return compile_fn(qc, settings)
    rows = []
    for c in d['cases']:
        qc = built[c['id']]
        cost = counted(qc)
        if cost.compiler_settings_hash != p.PROFILE['compiler']:
            raise ValueError('Compiler settings hash differs from the fixed profile')
        rows.append(dict(case=deepcopy(c), structural=structural(c, d, data), precompile=gate_counts(qc),
            library_lowering=decomposition(qc),
            compiled=dict(rz_count=cost.rz_count, cx_count=cost.cx_count, rz_depth=cost.rz_depth, cx_depth=cost.cx_depth,
                          total_depth=cost.total_depth, circuit_size=cost.circuit_size, posttranspile_counts=dict(cost.posttranspile_gate_counts),
                          counts=gate_counts(cost.transpiled_circuit), ops=ops_list(cost.transpiled_circuit),
                          global_phase=float(cost.transpiled_circuit.global_phase)),
            compiled_vs_original=compiled_equivalence(qc, cost.transpiled_circuit)))
    pairs = []
    for m in LAW_ROUNDS:
        for context in CONTEXTS:
            g = next(r for r in rows if r['case']['form'] == 'GENERIC_CONTROL' and r['case']['context'] == context and r['case']['m'] == m)
            a = next(r for r in rows if r['case']['form'] == 'ANALYTIC_CX_CRZ_CX' and r['case']['context'] == context and r['case']['m'] == m)
            pairs.append(dict(m=m, context=context, operator=operators(built[g['case']['id']], built[a['case']['id']], s0=s0, s1=s1),
                delta_compiled=dict(rz=g['compiled']['rz_count'] - a['compiled']['rz_count'], cx=g['compiled']['cx_count'] - a['compiled']['cx_count']),
                delta_compiled_by_wire={w: {k: g['compiled']['counts']['by_wire'].get(w, {}).get(k, 0) - a['compiled']['counts']['by_wire'].get(w, {}).get(k, 0)
                                            for k in ('rz', 'sx', 'x', 'cx')}
                                        for w in sorted(set(g['compiled']['counts']['by_wire']) | set(a['compiled']['counts']['by_wire']))},
                structural_S=g['structural']['structural_rz'], analytic_C_minus_S=a['compiled']['rz_count'] - a['structural']['structural_rz'],
                generic_C_minus_S=g['compiled']['rz_count'] - g['structural']['structural_rz']))
    return dict(design=d, rows=rows, pairs=pairs, compile_calls=calls, sampling=0, fit=0, prices=0,
                inputs=dict(source=dict(commit=C_RELEASE, path=C_RESULT), preparation=data['preparation'], H=data['H'],
                            counter=dict(commit=COUNTER, path='src/trottertracks/h4_df10_price_v1.py'),
                            provider=dict(commit=PROVIDER, functions=['df_rte_qiskit._append_central_application',
                                          'df_rte_qiskit._append_basis', 'rte_compiled_cost.transpile_and_measure_cost'])))
