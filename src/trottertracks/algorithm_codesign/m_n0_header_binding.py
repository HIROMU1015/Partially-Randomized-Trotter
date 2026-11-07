"""Saved N1 n=0 headers -> existing M eligibility; no circuit/matrix/state/RNG.

Only explicitly supplied records are inspected. The real five-file input is
loaded separately by the one-pass execution entry, never during import/tests.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import re

import numpy as np

from . import m_label_average as m
from .a1_new_target_n1 import canonical, digest
from .shared_cpu import module

SLOT = 'm0/q1/r1/order0/application0/rotation'


class BindingError(ValueError):
    def __init__(self, code, detail, *, label=None, partial=()):
        self.code, self.detail, self.label, self.partial = code, str(detail), label, tuple(partial)
        super().__init__(f'{code}: {detail}')


def require(condition, detail, *, mismatch=False):
    if not condition:
        raise BindingError('INPUT_OR_POLICY_MISMATCH' if mismatch else 'BINDING_INCOMPLETE', detail)


def finite(value):
    require(type(value) in (float, int, str), 'explicit real numeric metadata required')
    result = float(value)
    require(math.isfinite(result), 'nonfinite operation parameter')
    return result


def _basis(rows, n, expected_hash):
    """Reuse the saved canonical basis hash, NOT matrix fingerprints or restore_operations."""
    operations, described = [], []
    for row in rows:
        name, wires, parameters = row['name'], tuple(row['qubits']), tuple(row['parameters'])
        require(name in ('rz', 'xx_plus_yy'), 'unsupported saved basis operation: '+str(name))
        require(isinstance(row.get('matrix_sha256'), str) and
                re.fullmatch('[0-9a-f]{64}', row['matrix_sha256']), 'missing stored matrix provenance')
        cls = module('rte').BasisChangeOperation
        described.append(cls(name=name, qubits=wires, parameters=parameters,
                             matrix_sha256=row['matrix_sha256']))
        operations.append((name, tuple(finite(x).hex() for x in parameters), wires))
    observed = module('df_rte_tail').canonical_basis_hash(n, described)
    require(observed == expected_hash, 'saved basis operation/hash mismatch', mismatch=True)
    return m.LiteralCircuit(n, tuple(operations))


def _half_blocks(preparation, n, dt):
    """Same descriptor-only primitives as df_partial_s2._append_block.

    Scalar phases remain separate ordered terms. Only existing stored diagonal
    arrays are read; no diagonalization, Hamiltonian, basis or circuit is built.
    """
    blocks = []
    ops = module('df_trotter.ops')
    for index, block in enumerate(preparation['deterministic_blocks']):
        require(block['num_system_qubits'] == n and block['order_index'] == index,
                'deterministic block width/order mismatch', mismatch=True)
        rows = preparation['basis_operations'][block['basis_hash']]
        require(canonical(rows) == canonical(block['basis_change_operations']),
                'block and catalog basis descriptions disagree', mismatch=True)
        basis = _basis(rows, n, block['basis_hash'])
        if block['block_kind'] == 'one_body':
            values = tuple(finite(x) for x in block['diagonal_eigenvalues'])
            require(len(values) == n, 'one-body diagonal width')
            primitive = ops.one_body_diagonal_primitives(np.asarray(values), dt/2.)
        elif block['block_kind'] == 'df_fragment':
            values = tuple(finite(x) for x in block['diagonal_eta'])
            require(len(values) == n, 'fragment diagonal width')
            primitive = ops.df_squared_diagonal_primitives(np.asarray(values), finite(block['lam']), dt/2.)
        else:
            raise BindingError('BINDING_INCOMPLETE', 'unsupported deterministic block kind')
        # append_diagonal_primitives emits RZZ first, then RZ, then scalar phase.
        diagonal = tuple(('rzz', (finite(a).hex(),), (i,j)) for i,j,a in primitive.rzz if a != 0.)
        diagonal += tuple(('rz', (finite(a).hex(),), (i,)) for i,a in primitive.rz)
        phase = finite(primitive.global_phase)
        blocks.append(m.LiteralCircuit(n, basis.adjoint().operations + diagonal + basis.operations,
                                       (phase.hex(),) if phase != 0. else ()))
    require(bool(blocks), 'no saved deterministic context')
    def chain(sequence):
        return m.LiteralCircuit(n, tuple(op for b in sequence for op in b.operations),
                               tuple(p for b in sequence for p in b.global_phase_terms))
    # Application order is forward blocks for B; reversed blocks for A.
    return chain(tuple(reversed(blocks))), chain(tuple(blocks))


@dataclass(frozen=True)
class BoundN0:
    rows: tuple
    law: m.CategoricalLaw
    outer_mass: F
    records: tuple
    identities: dict


def bind_n0_headers(records):
    """One finite supplied catalog -> SlotLabel rows, including explicit zero rows.

    This function is a new scientific metadata evaluation when given real N1.
    Tests use only invented saved-schema records. It does not open any file.
    """
    partial = []
    label = None
    try:
        source, config, law_record, policy, prep = (records[k] for k in
            ('SOURCE.json', 'CONFIG.json', 'LAW.json', 'POLICY.json', 'PREPARATION.json'))
        scope = source['scope']; n = scope['system_qubits']; cfg = config['config']
        require(type(n) is int and n >= 1, 'system width')
        require((scope['m'],scope['q'],scope['r'],scope['K']) == (0,1,1,2), 'fixed slot context', mismatch=True)
        require(scope['coefficient_atol'] == 0. and prep['coefficient_atol'] == 0., 'threshold', mismatch=True)
        require(cfg['rte_steps'] == 1 and cfg['finite_taylor_order'] == 2, 'r/K mismatch', mismatch=True)
        dt = finite(cfg['evolution_time'])
        require(dt == scope['delta'] and cfg['tail_hash'] == prep['tail_hash'], 'time/tail mismatch', mismatch=True)
        dist = config['distribution']
        require(tuple(dist['orders']) == (0,2) and
                finite(dist['dimensionless_step_time']) == finite(cfg['dimensionless_step_time']),
                'saved RTE distribution mismatch', mismatch=True)
        phi = m.Angle('rad', math.atan(finite(dist['dimensionless_step_time'])/1))
        require(law_record['uniform_grid_size'] == 2**53 and
                law_record['application_label_order'] == 'products_then_rotation' and
                law_record['Q_definition'] == 'Q(order)*product(Q(label_position)); independent uniform-word model',
                'actual law/application convention', mismatch=True)
        require(tuple(law_record['orders']) == (0,2), 'actual order law', mismatch=True)
        orders = m.CategoricalLaw((0,2),tuple(F(x['exact']) for x in law_record['order_masses']), 'orders')
        require(orders.mass(0) > 0 and all((x*2**53).denominator == 1 for x in orders.masses),
                'actual order support/grid', mismatch=True)
        law_id = source['source_commit']+':'+prep['preparation_hash']
        law = m.CategoricalLaw(tuple(r['label'] for r in law_record['label_masses']),
                               tuple(F(r['exact']) for r in law_record['label_masses']), law_id)
        m._law_check(law)
        catalog = prep['components']
        require(len({c['component_id'] for c in catalog}) == len(catalog) and
                set(law.labels) == {c['component_id'] for c in catalog}, 'law/catalog coverage', mismatch=True)
        declaration = policy['declaration']
        require(declaration['schema'] == 'a1_new_target_selected_policy_v1' and
                declaration['source_commit'] == source['source_commit'] and
                declaration['preparation_hash'] == prep['preparation_hash'] and
                declaration['tail_hash'] == prep['tail_hash'], 'selected policy identity', mismatch=True)
        require((declaration['rule'],declaration['maximum_support_run_length'],declaration['boundary'],
                 declaration['controlled']) == ('support_run_le_1',1,'boundary_optimized','ordinary_diag_I_U'),
                'unsupported selected policy rule', mismatch=True)
        require(digest(canonical(declaration).encode()) == policy['policy_fingerprint'] and
                digest(canonical(policy['definitions']).encode()) == declaration['definition_sha256'],
                'selected policy content fingerprint', mismatch=True)
        definitions = {}
        for row in policy['definitions']:
            key = row['source_basis_id'],row['source_basis_hash'],tuple(row['support'])
            require(key not in definitions, 'duplicate selected definition', mismatch=True)
            md = row['definition']
            require(md['num_system_qubits'] == n, 'selected basis width', mismatch=True)
            definitions[key] = (md, _basis(md['operations'], n, md['basis_hash']))
        expected = {(c['basis_id'],c['basis_hash'],tuple(c['diagonal_pauli_support']))
                    for c in catalog if not c['is_identity']}
        require(set(definitions) == expected, 'selected policy coverage', mismatch=True)
        A, B = _half_blocks(prep, n, dt)
        constant = finite(float.fromhex(prep['constant_coefficient_hex']))
        identity = finite(float.fromhex(prep['identity_coefficient_hex']))
        # Exact source expression in df_partial_s2_repeated._append_aggregated_phases.
        scalar = m.Angle('rad', -dt*1*(constant+identity))
        identities = {k:prep[k] for k in ('hamiltonian_hash','partition_hash','preparation_hash','tail_hash')}
        identities.update(selected_policy_fingerprint=policy['policy_fingerprint'], law_id=law_id)
        context = digest(canonical(dict(identities=identities,slot=SLOT,system_qubits=n)).encode())
        bound = []
        original_bases = {}
        for c in catalog:
            label = c['component_id']; support = tuple(c['diagonal_pauli_support'])
            require(type(c['is_identity']) is bool and type(c['coefficient_sign']) is int
                    and c['coefficient_sign'] in (-1,1), 'identity/sign description')
            require(finite(c['coefficient_abs']) > 0 and
                    finite(c['coefficient_abs']) == float.fromhex(c['coefficient_abs_hex']),
                    'positive coefficient metadata mismatch')
            require(len(set(support)) == len(support) and
                    all(type(q) is int and 0 <= q < n for q in support), 'diagonal wire mapping')
            if c['is_identity']:
                require(not support, 'identity has nonempty support')
                selected_id = 'identity/no-selected-basis'
                V = m.LiteralCircuit(n, ())
                selected_hash = None
            else:
                require(len(support) in (1,2), 'nonidentity diagonal support')
                if c['basis_hash'] not in original_bases:
                    original_bases[c['basis_hash']] = _basis(prep['basis_operations'][c['basis_hash']], n, c['basis_hash'])
                md, selected = definitions[(c['basis_id'],c['basis_hash'],support)]
                selected_id, selected_hash = md['basis_id'], md['basis_hash']
                # Source applies selected.inverse BEFORE R and selected AFTER R.
                # Thus the V in M's A V† R V B is selected†, not selected.
                V = selected.adjoint()
            row = m.SlotLabel(label,context,SLOT,policy['policy_fingerprint'],law.source,law.mass(label),
                A,B,V,tuple(f'qiskit-system-{i}' for i in range(n)),selected_id,support,c['coefficient_sign'],
                'rotation',phi,scalar,m.ZERO,original_basis_id=str(c.get('basis_id')))
            # Existing eligibility validates the same literal metadata even at width 8.
            if row.actual_mass > 0:
                m.qualify_group((row,),law)
            status = 'ELIGIBLE_HEADER' if row.actual_mass > 0 else 'ZERO_GRID_NOT_GROUPED'
            partial.append(dict(label=label,status=status,actual_mass=str(row.actual_mass),slot_id=SLOT,
                selected_basis_id=selected_id,selected_basis_hash=selected_hash,
                original_basis_id=c.get('basis_id'),original_basis_hash=c.get('basis_hash'),
                kind='rotation',phi_hex=phi.numeric.hex(),signed_support=[row.sign,list(row.support)],
                scalar_phase_hex=scalar.numeric.hex(),event_phase_pi='0',
                V_orientation='adjoint_of_selected_runtime_operations',
                original_target_application_order='aggregated_phase; forward_D; selected_inverse; R; selected_forward; reversed_D',
                identity_source_rule='df_rte_qiskit identity phase' if c['is_identity'] else 'SavedEngine.selected_circuit'))
            bound.append(row)
        return BoundN0(tuple(bound),law,orders.mass(0),tuple(partial),identities)
    except BindingError as exc:
        raise BindingError(exc.code,exc.detail,label=label,partial=partial) from exc
    except (KeyError,TypeError,ValueError,OverflowError) as exc:
        raise BindingError('BINDING_INCOMPLETE',str(exc),label=label,partial=partial) from exc


def summarize_binding(binding):
    """No r(z) enumeration: only finite supplied label headers and exact masses."""
    positive = tuple(row for row in binding.rows if row.actual_mass > 0)
    groups = m.partition_supplied_rows(positive,binding.law)
    records = []
    for index,g in enumerate(groups):
        functions = sorted({(row.sign,tuple(sorted(row.support))) for row in g.rows})
        nontrivial = len(g.rows) >= 2 and len(functions) >= 2
        row = g.rows[0]
        explicit_adjoint = m.circuit_key(row.A) == m.circuit_key(row.B.adjoint())
        records.append(dict(group=index,labels=[r.label for r in g.rows],
            conditional_mass=str(g.mass),full_context_mass=str(binding.outer_mass*g.mass),
            actual_conditional_q=[str(p) for p in g.conditional],signature_sha256=digest(canonical(json_metadata(g.signature)).encode()),
            signed_diagonal_functions=[[s,list(w)] for s,w in functions],nontrivial=nontrivial,
            direct_measurement_structure='ONLY_KNOWN_DIRECT_MEASUREMENT_STRUCTURE' if explicit_adjoint else 'UNKNOWN_NO_LITERAL_ADJOINT_MATCH'))
    require(sum((g.mass for g in groups),F(0)) == 1, 'positive support mass lost', mismatch=True)
    return dict(status='NONTRIVIAL_GROUP_FOUND' if any(g['nontrivial'] for g in records)
                else 'NO_NONTRIVIAL_GROUP_IN_SELECTED_SLOT',
                slot=SLOT,outer_order_mass=str(binding.outer_mass),groups=records,
                zero_grid_labels=[r.label for r in binding.rows if r.actual_mass == 0],
                UNKNOWN=['real-state variance','instrument numerical error','full compiled cost','quantum resource improvement'])


def json_metadata(value):
    """Lossless representation of the exact fractions in this adapter's keys."""
    if isinstance(value,F):
        return {'exact_fraction':str(value)}
    if isinstance(value,(tuple,list)):
        return [json_metadata(v) for v in value]
    if isinstance(value,dict):
        return {k:json_metadata(v) for k,v in value.items()}
    return value
