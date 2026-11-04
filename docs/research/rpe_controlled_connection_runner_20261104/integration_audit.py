"""Finite descriptor/parameter audit. No scientific work at import."""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'src'),
               str(ROOT/'docs/research/rpe_categorical_circuit_connection_20261104')]
import connection_adapter as transfer


class Unknown(ValueError):
    """Missing/unsupported scientific evidence; never replace it with zero."""


def require(condition, message):
    if not condition:
        raise Unknown(message)


def finite_hex(value):
    value = float(value)
    require(math.isfinite(value), 'nonfinite parameter')
    return value.hex()


def descriptor(definition, saved=None):
    """Read existing metadata/runtime parameters; no Operator/to_matrix call."""
    from qiskit.circuit.library import RZGate, XXPlusYYGate
    from trotterlib.df_rte_tail import canonical_basis_hash
    meta, runtime = definition.metadata, definition.runtime_operations
    require(canonical_basis_hash(meta.num_system_qubits, meta.operations) == meta.basis_hash,
            'basis canonical hash mismatch')
    require(len(runtime) == len(meta.operations), 'descriptor length mismatch')
    operations = []
    for description, (gate, wires) in zip(meta.operations, runtime, strict=True):
        require(type(gate) in (RZGate, XXPlusYYGate) and gate.base_class is type(gate),
                'unsupported basis gate class')
        require(tuple(wires) == tuple(description.qubits)
                and gate.name == description.name
                and tuple(str(p) for p in gate.params) == tuple(description.parameters),
                'runtime gate/wire/parameter descriptor mismatch')
        operations.append(dict(name=gate.name, qubits=list(wires),
                               params=[finite_hex(p) for p in gate.params],
                               legacy_descriptor=json.loads(json.dumps(asdict(description)))))
    transfer.gaussian_descriptor_check(operations, meta.num_system_qubits)
    result = dict(basis_id=meta.basis_id, basis_hash=meta.basis_hash,
                  n_qubits=meta.num_system_qubits, operations=operations)
    if saved is not None:
        require(all(result[k] == saved[k] for k in ('basis_id','basis_hash','n_qubits'))
                and len(operations)==len(saved['operations'])
                and all(all(op[k]==old[k] for k in op)
                        for op,old in zip(operations,saved['operations'],strict=True)),
                'saved full basis differs')
    return result


def export_context(prep, support, saved, saved_proofs, counts):
    """Full and selected coverage, using already-registered definitions."""
    require(prep.preparation_hash == saved['preparation_hash']
            and prep.tail_extraction.tail_hash == saved['tail_hash']
            and prep.identity_policy == saved['identity_policy']
            and prep.coefficient_atol == saved['coefficient_atol'], 'preparation identity')
    registry = prep.rte_preparation.basis_registry
    from trotterlib.df_rte_tail import DFBasisDefinition, BasisChangeMetadata
    deterministic = {block.basis_id: DFBasisDefinition(
        BasisChangeMetadata(block.basis_id,block.basis_hash,block.num_system_qubits,
                            block.basis_change_operations),block.runtime_basis_operations)
        for block in prep.deterministic_blocks}
    bases = {}
    for name, spec in sorted(saved['bases'].items()):
        definition = deterministic[name] if name in deterministic else registry.definition(name)
        bases[name] = descriptor(definition, spec)
    counts['full_descriptors'] = len(bases)
    require(len(saved_proofs) == len(support), 'support proof coverage')
    proof_map = {}
    for p in saved_proofs:
        key = (p['source_basis_id'], p['source_basis_hash'], tuple(p['support']))
        require(key not in proof_map, 'duplicate support proof')
        proof_map[key] = p
    require(set(proof_map) == set(support), 'support keys differ from run02')
    for key, definition in sorted(support.items()):
        p = proof_map[key]
        source = bases.get(key[0])
        require(source is not None and source['basis_hash'] == key[1], 'support source identity')
        require((definition.basis_id, definition.basis_hash) ==
                (p['selected_basis_id'], p['selected_basis_hash']), 'selected basis hash differs from run02')
        current = descriptor(definition)
        require(len(source['operations']) == p['source_basis_operation_count']
                and len(current['operations']) == p['support_basis_operation_count'],
                'support descriptor operation count')
        if current['basis_id'] in bases:
            require(bases[current['basis_id']] == current, 'conflicting selected basis')
        bases[current['basis_id']] = current
    counts['selected_descriptors'] = len(support)
    components = prep.rte_preparation.symbolic_tail.components
    require(len(components) == len(saved['components']), 'component count mismatch')
    output, seen = [], set()
    for c, old in zip(components, saved['components'], strict=True):
        require(c.component_id not in seen, 'duplicate component')
        seen.add(c.component_id)
        require(c.component_id == old['id'] and c.coefficient_sign == old['sign']
                and c.basis_id == old['basis_id']
                and c.basis_hash == saved['bases'][old['basis_id']]['basis_hash']
                and list(c.diagonal_pauli_support) == old['support']
                and finite_hex(c.coefficient_abs) == finite_hex(old['coefficient_abs'])
                and finite_hex(c.probability) == finite_hex(old['probability64']),
                'component sign/basis/coefficient/law mismatch')
        key = (c.basis_id, c.basis_hash, tuple(c.diagonal_pauli_support))
        require(key in support, 'component support uncovered')
        output.append(dict(id=c.component_id, sign=c.coefficient_sign,
                           full=c.basis_id, selected=support[key].basis_id,
                           support=list(c.diagonal_pauli_support)))
    return bases, output


def primitive(block, time):
    import numpy as np
    from trotterlib.df_trotter.ops import one_body_diagonal_primitives, df_squared_diagonal_primitives
    if block.block_kind == 'one_body':
        return one_body_diagonal_primitives(np.asarray(block.diagonal_eigenvalues), time)
    require(block.block_kind == 'df_fragment', 'unsupported deterministic block')
    return df_squared_diagonal_primitives(np.asarray(block.diagonal_eta), block.lam, time)


def primitive_payload(value):
    return dict(global_phase=finite_hex(value.global_phase),
                rz=[[int(q), finite_hex(a)] for q,a in value.rz],
                rzz=[[int(q),int(k),finite_hex(a)] for q,k,a in value.rzz])


def boundary_context(prep, saved, delta, counts):
    require(len(prep.deterministic_blocks) == len(saved['deterministic_blocks'])
            and bool(prep.deterministic_blocks), 'deterministic block coverage')
    halves = []
    for block, old in zip(prep.deterministic_blocks, saved['deterministic_blocks'], strict=True):
        counts['half_primitives'] = counts.get('half_primitives',0)+1
        half = primitive(block, delta/2.0)
        require(block.block_id == old['block_id'] and block.basis_id == old['basis_id']
                and primitive_payload(half) == old['primitives'], 'saved half primitive mismatch')
        halves.append(half)
    counts['full_primitives'] = counts.get('full_primitives',0)+1
    full = primitive(prep.deterministic_blocks[0],delta)
    # Same evaluation order as _append_step_phases / _append_aggregated_phases.
    a = -delta*prep.constant_coefficient
    b = -delta*prep.extracted_identity_coefficient
    require([finite_hex(a),finite_hex(b)] == saved['step_phase_angles'], 'saved raw scalar mismatch')
    return halves, full, (a,b), transfer.block_gap(halves[0],full)


def controlled_template(kind, angle):
    """Finite Qiskit 1.3 definitions; no dense Operator or transpilation."""
    from qiskit.circuit.library import RZGate, RZZGate, CRZGate, CXGate, CCXGate
    value = float(angle)
    require(math.isfinite(value) and transfer.half_angle_exact(value),
            'RZ/CRZ halving loses parameter')
    gate = {'rz': RZGate, 'rzz': RZZGate}[kind](value).control(1)
    require(gate.ctrl_state == 1 and gate.num_ctrl_qubits == 1, 'control state/arity')
    definition = gate.definition
    require(gate.num_qubits==(2 if kind=='rz' else 3)
            and definition.num_qubits==gate.num_qubits,'controlled gate dimension')
    require(float(definition.global_phase) == 0., 'controlled global phase')
    def wires(circuit, instruction):
        return tuple(circuit.find_bit(q).index for q in instruction.qubits)
    def check_crz(crz):
        require(crz.base_class is CRZGate and list(crz.params) == [value]
                and crz.ctrl_state == 1 and crz.num_ctrl_qubits == 1, 'CRZ class/angle/control')
        d = crz.definition
        require(float(d.global_phase) == 0. and len(d.data) == 4, 'CRZ definition phase/length')
        expected = [(RZGate,(1,),[value/2]), (CXGate,(0,1),[]),
                    (RZGate,(1,),[-value/2]), (CXGate,(0,1),[])]
        for ins,(cls,qs,ps) in zip(d.data,expected,strict=True):
            require(ins.operation.base_class is cls and wires(d,ins)==qs
                    and list(ins.operation.params)==ps, 'CRZ definition mismatch')
            if cls is CXGate:
                require(ins.operation.ctrl_state==1,'open CX in CRZ definition')
    if kind == 'rz':
        check_crz(gate)
    else:
        require(len(definition.data)==3, 'controlled RZZ definition length')
        for i,ins in enumerate(definition.data):
            cls,qs = (CCXGate,(0,1,2)) if i != 1 else (CRZGate,(0,2))
            require(ins.operation.base_class is cls and wires(definition,ins)==qs,
                    'controlled RZZ class/wire mismatch')
            if i == 1:
                check_crz(ins.operation)
            else:
                require(not ins.operation.params and ins.operation.ctrl_state==3,
                        'CCX unexpected parameters/control')
    return dict(kind=kind,angle=value.hex(),halving_exact=True,template_confirmed=True)


def audit_lowering(halves, full, theta_by_r, counts, library_evidence):
    require(library_evidence is True, 'missing CZ/Phase/runtime SOURCE evidence')
    candidates = []
    for p in [*halves,full]:
        candidates.extend(('rz',float(a)) for _,a in p.rz)
        candidates.extend(('rzz',float(a)) for _,_,a in p.rzz if float(a) != 0.)
    for r,angles in sorted(theta_by_r.items()):
        require(type(r) is int and r > 0 and len(angles)==2, 'theta/order coverage')
        for theta in angles:
            theta = float.fromhex(theta)
            require(math.isfinite(theta) and math.isfinite(2*theta)
                    and transfer.exact(2*theta)==2*transfer.exact(theta), 'event doubling loses parameter')
            for sign in (-1,1):
                for kind in ('rz','rzz'):
                    candidates.append((kind,2.0*(sign*theta)))
    require(len(candidates)<=172, 'local constructor domain exceeds fixed scope')
    result, cache = [], set()
    for kind,angle in candidates:
        key = kind,finite_hex(angle)
        if key not in cache:
            counts['local_control_constructors'] = counts.get('local_control_constructors',0)+1
            result.append(controlled_template(kind,angle))
            cache.add(key)
    require(bool(result), 'no lowering evidence')
    return dict(bound='0',candidate_slots=len(candidates),templates=result,
                phase_CZ='exact standard gates; source/version bound',endian='local control bit0; system wire bit j')


def classify_axis(saved, epsilon, extra):
    """Only called in science/synthetic mode, never by metadata/default mode."""
    required = ('U_cat_upper','shot_half_width','interval')
    require(all(k in saved for k in required), 'missing saved axis evidence')
    u = Q(saved['U_cat_upper'])
    sl = Q(saved['shot_half_width']['lower'])
    s = Q(saved['shot_half_width']['upper'])
    lo,hi = Q(saved['interval']['lower']),Q(saved['interval']['upper'])
    eps = Q(epsilon)
    require(u>=0 and 0<=sl<=s and lo<=hi and eps>0, 'invalid saved interval/budget')
    if extra is None:
        return dict(decision='UNKNOWN',reason='missing transfer evidence',extra_upper=None)
    extra = transfer.exact(extra)
    require(extra>=0, 'negative extra bound')
    strict = u+extra+s < eps
    return dict(decision='SUFFICIENT_CONDITION_CONFIRMED' if strict else 'UNDETERMINED',
                U_cat_upper=str(u),extra_upper=str(extra),shot_half_width_upper=str(s),epsilon=str(eps),
                total_upper=str(u+extra+s),interval=dict(lower=str(lo-extra),upper=str(hi+extra)),
                pi64_added_again=False)
