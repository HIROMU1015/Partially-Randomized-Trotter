"""Fixed A attribution and physical B pair study; canonical RTE is unchanged.

Only conditional *classical circuit-cost* sampling uses the existing event
factory. It does not replace quantum trajectory sampling or its return law.
"""
from __future__ import annotations
from dataclasses import dataclass
import itertools
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from trotterlib.rte import _make_event, event_unitary, finite_rte_distribution
from . import construction_comparison as cc

CONDITIONAL_DRAWS = 96
COMPILE_CAP = 2048
UNIFORM_SEED = 20264110
STRUCTURAL = ('basis_calls', 'basis_operations', 'reflection_z_actions')


def full_square_terms(factors):
    terms = {}
    for g in factors:
        p = cc.orbital_paulis(g)
        terms = cc.padd(terms, cc.pmul(p, p))
    return cc.real_terms(terms)


def whole_pauli(factors):
    n = len(factors[0]); scalar, tail = cc.split_identity(full_square_terms(factors), n)
    basis = QuantumCircuit(n)
    return cc.PRConstruction('collected_whole_pauli', n, [],
        [cc.Primitive(p, c, p, basis) for p, c in tail.items()], scalar, basis, 0,
        {'constructor': 'coefficient-collected JW whole H, no Fock fitting'})


def complete_core(factors):
    """Occupation diagonal derived from g entries, including off-diagonal squares."""
    n = len(factors[0]); core = {}; number = []
    for i in range(n):
        g = np.zeros((n, n)); g[i, i] = 1
        number.append(cc.orbital_paulis(g))
    for g in factors:
        diag = cc.orbital_paulis(np.diag(np.diag(g)))
        core = cc.padd(core, cc.pmul(diag, diag))
        for i, j in itertools.combinations(range(n), 2):
            correction = cc.padd(number[i], number[j], cc.pscale(cc.pmul(number[i], number[j]), -2))
            core = cc.padd(core, cc.pscale(correction, abs(g[i, j])**2))
    core = cc.real_terms(core)
    scalar, tail = cc.split_identity(cc.real_terms(cc.padd(full_square_terms(factors), cc.pscale(core, -1))), n)
    basis = QuantumCircuit(n)
    return cc.PRConstruction('complete_occupation_core', n, [cc.DiagonalBlock('occupation_diagonal', core, basis)],
        [cc.Primitive(p, c, p, basis) for p, c in tail.items()], scalar, basis, None,
        {'constructor': 'entrywise occupation-diagonal formula, no dense diagonal extraction',
         'core_terms': core, 'residual_terms': tail})


def pair_frame(x, y):
    """Stable Gram area and conjugated QR orbital completion for c_x=sum x_i a_i.

    Exact zero wedge is skipped; near-collinear nonzero terms are retained.
    Binary64 QR conditioning is recorded, not certified for arbitrary inputs.
    """
    x = np.asarray(x, complex); y = np.asarray(y, complex)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2 or not np.isfinite([x, y]).all():
        raise ValueError('Finite matching vectors of at least two modes required')
    delta = math.fsum(abs(x[i]*y[j]-x[j]*y[i])**2 for i, j in itertools.combinations(range(len(x)), 2))
    if delta == 0: return delta, None
    q, _ = np.linalg.qr(np.column_stack((x, y)), mode='complete')
    return delta, q.conj()


def quartic_terms(x, y):
    n = len(x); ax = {}; ay = {}
    for i in range(n):
        a = cc.annihilator_paulis(n, i)
        ax = cc.padd(ax, cc.pscale(a, x[i])); ay = cc.padd(ay, cc.pscale(a, y[i]))
    dagger = lambda d: {p: complex(c).conjugate() for p, c in d.items()}
    return cc.pmul(cc.pmul(cc.pmul(dagger(ax), dagger(ay)), ay), ax)


@dataclass
class PairInvolution:
    component_id: str
    coefficient: float
    basis: QuantumCircuit
    label: str = 'pair_Q'
    reflected_aux: int | None = None

    def dense(self):
        v = Operator(self.basis).data
        z = np.ones(2**self.basis.num_qubits); z[(np.arange(len(z)) & 3) == 3] = -1
        return v @ np.diag(z) @ v.conj().T

    def apply(self, qc, ancilla, *, angle=None):
        ids = list(range(self.basis.num_qubits))
        qc.compose(self.basis.inverse(), qubits=ids, inplace=True)
        if angle is None:
            if self.coefficient < 0: qc.p(math.pi, ancilla)
            qc.ccz(ancilla, 0, 1)
        else:
            theta = math.copysign(1, self.coefficient)*angle
            qc.p(-theta, ancilla)
            qc.mcp(2*theta, [ancilla, 0], 1)
        qc.compose(self.basis, qubits=ids, inplace=True)


def physical_pairs(u, edges, *, involution):
    n = len(u); primitives = []; scalar = 0.; pairs = []
    for k, (i, j, weight) in enumerate(edges):
        x, y = u[:, i].conj(), u[:, j].conj()
        delta, v = pair_frame(x, y); omega = weight*delta
        info = {'edge': [i, j, weight], 'x': cc.matrix(x), 'y': cc.matrix(y),
                'delta': delta, 'omega': omega, 'skipped_exact_zero': v is None}
        if v is None:
            pairs.append(info); continue
        basis = cc.gaussian_circuit(v); info['orbital_frame'] = cc.matrix(v)
        info['singular_values'] = np.linalg.svd(np.column_stack((x, y)), compute_uv=False).tolist()
        if involution:
            scalar += omega/2
            primitives.append(PairInvolution(f'pair{k}:Q', -omega/2, basis))
        else:
            scalar += omega/4
            for support, coef in (((0,), -omega/4), ((1,), -omega/4), ((0, 1), omega/4)):
                label = ''.join('Z' if a in support else 'I' for a in reversed(range(n)))
                primitives.append(cc.Primitive(f'pair{k}:{label}', coef, label, basis))
        pair = (np.eye(2**n)-PairInvolution('check', 1, basis).dense())/2
        info['quartic_identity_residual'] = cc.norm(cc.pdense(quartic_terms(x, y), n)-delta*pair)
        if info['quartic_identity_residual'] > cc.ATOL: raise AssertionError('Physical pair identity')
        pairs.append(info)
    return cc.PRConstruction('physical_pair_Q' if involution else 'physical_pair_Z_ZZ', n, [], primitives,
                            scalar, QuantumCircuit(n), 0, {'pairs': pairs, 'unmerged_dictionary': True})


def coefficient_direct(u, edges):
    n = len(u); terms = {}
    for i, j, weight in edges:
        terms = cc.padd(terms, cc.pscale(quartic_terms(u[:, i].conj(), u[:, j].conj()), weight))
    scalar, tail = cc.split_identity(cc.real_terms(terms), n); basis = QuantumCircuit(n)
    return cc.PRConstruction('coefficient_physical_pauli', n, [],
        [cc.Primitive(p, c, p, basis) for p, c in tail.items()], scalar, basis, 0,
        {'constructor': 'normal-ordered coefficient JW algebra; no dense Pauli fitting', 'terms': tail})


def serialize(con):
    tail = con.tail()
    # Recover single-particle matrices from already input-derived Gaussian gates.
    def orbital(b):
        f = Operator(b).data; indices = [1 << i for i in range(con.n)]
        return cc.matrix(f[np.ix_(indices, indices)])
    return {'name': con.name, 'n': con.n, 'ld': con.ld, 'metadata': con.metadata,
            'identity': con.identity, 'tail_lambda': 0. if tail is None else tail.lambda_r,
            'outer_frame': orbital(con.outer_basis), 'whole_hamiltonian': cc.matrix(con.dense()),
            'tail_matrix': cc.matrix(np.zeros((2**con.n, 2**con.n)) if tail is None else tail.dense_hamiltonian),
            'deterministic_blocks': [{'id': b.block_id, 'terms': b.terms, 'orbital_frame': orbital(b.basis),
                                      'matrix': cc.matrix(b.dense())} for b in con.blocks],
            'primitives': [{'id': p.component_id, 'coefficient': p.coefficient, 'label': p.label,
                            'orbital_frame': orbital(p.basis), 'reflected_aux': p.reflected_aux,
                            'kind': 'pair_Q' if isinstance(p, PairInvolution) else 'Pauli'} for p in con.primitives]}


def conditional_design(tail, dist, uniforms):
    """Full order0; exact order2 up to96 events, otherwise96 coupled iid draws."""
    n = len(tail.components); probs = np.array([c.probability for c in tail.components]); probs /= probs.sum()
    designs = []
    for index, order in enumerate(dist.orders):
        exact = n**(order+1) <= CONDITIONAL_DRAWS
        if exact: indices = list(itertools.product(range(n), repeat=order+1))
        else:
            if order != 2: raise ValueError('This study fixes K2')
            indices = np.searchsorted(np.cumsum(probs), uniforms, side='right').tolist()
        rows = []
        for draw, selected in enumerate(indices):
            event = _make_event(selected, tail.components, dist, index)
            weight = math.prod(probs[j] for j in selected) if exact else 1/len(indices)
            rows.append({'draw': draw, 'indices_rotation_first': list(selected), 'weight': weight,
                         'event': event.to_dict()})
        designs.append({'order': order, 'probability': dist.order_probabilities[index], 'exact': exact, 'draws': rows})
    return designs


def circuit_for_event(con, time, event, operators):
    n = con.n; qc = QuantumCircuit(n+1); u = np.eye(2**n, dtype=complex); ids = list(range(n))
    qc.compose(con.outer_basis.inverse(), qubits=ids, inplace=True)
    primitives = {p.component_id: p for p in con.primitives}
    structural = dict.fromkeys(STRUCTURAL, 0)
    def basis_count(basis):
        structural['basis_calls'] += 2 if len(basis.data) else 0
        structural['basis_operations'] += 2*len(basis.data)
    for kind, i, duration in cc.schedule(con, time, 1, None if event is None else [event]):
        if kind == 'd':
            block = con.blocks[i]; block.apply(qc, n, duration); basis_count(block.basis)
            u = expm(-1j*duration*block.dense()) @ u
        else:
            for cid in event.product_component_ids: primitives[cid].apply(qc, n)
            primitives[event.rotation_component_id].apply(qc, n, angle=event.rotation_angle)
            qc.p(float(np.angle(event.phase)), n); u = event_unitary(event, operators) @ u
            for cid in event.selected_component_ids:
                p = primitives[cid]; basis_count(p.basis)
                structural['reflection_z_actions'] += 2 if p.reflected_aux is not None else 0
    basis_count(con.outer_basis); qc.p(-time*con.identity, n)
    qc.compose(con.outer_basis, qubits=ids, inplace=True)
    v = Operator(con.outer_basis).data
    return qc, np.exp(-1j*time*con.identity)*v@u@v.conj().T, structural


def mean_se(strata, axis, metric):
    mean = 0.; variance = 0.
    for s in strata:
        vals = [sum(d['costs'][a][metric] for a in ('X', 'Y')) if axis == 'XY' else d['costs'][axis][metric]
                for d in s['draws']]
        mean += s['probability']*sum(d['weight']*v for d, v in zip(s['draws'], vals, strict=True))
        if not s['exact']: variance += s['probability']**2*float(np.var(vals, ddof=1))/len(vals)
    return {'mean': mean, 'se': math.sqrt(variance)}


def coupled_vector(row, metric, epsilon):
    shots = next(x['shots_per_axis'] for x in row['precision_resources'] if x['epsilon_complex'] == epsilon)
    if shots is None: return None
    out = np.zeros(CONDITIONAL_DRAWS)
    for s in row['strata']:
        vals = [sum(d['costs'][a][metric] for a in ('X', 'Y')) for d in s['draws']]
        if s['exact']: out += s['probability']*sum(d['weight']*v for d, v in zip(s['draws'], vals, strict=True))
        else: out += s['probability']*np.array(vals)
    return shots*out


def study(constructions, h, time, prepared, audit, uniforms, group):
    candidates = []; rows = []
    for con in constructions:
        c = serialize(con); candidates.append(c)
        physical_h = con.dense()[:len(h), :len(h)]
        c['physical_reconstruction_residual'] = cc.norm(physical_h-h)
        if c['physical_reconstruction_residual'] > cc.ATOL: raise AssertionError('Hamiltonian reconstruction')
        tail = con.tail(); diagnostics = []
        for q in (1, 2, 4):
            corr, gamma = cc.corrected_pr_mean(con, time, q); corr = corr[:len(h), :len(h)]
            bias = cc.norm(corr-expm(-1j*time*h))
            diagnostics.append({'q': q, 'delta': time/q, 'normalization': gamma, 'operator_bias': bias,
                                'corrected_mean_physical_block': cc.matrix(corr),
                                'shots_per_axis': [cc.shot_budget(gamma, bias, e) for e in cc.EPSILONS]})
        row = {'candidate': con.name, 'q': 1, 'all_q_diagnostics': diagnostics, 'strata': []}
        if tail:
            dist = finite_rte_distribution(tail.lambda_r*time, 2)
            designs = conditional_design(tail, dist, uniforms)
            operators = {comp.component_id: op for comp, op in zip(tail.components, tail.operators)}
        else:
            dist = None; operators = {}; designs = [{'order': None, 'exact': True, 'probability': 1.,
                'draws': [{'draw': 0, 'indices_rotation_first': [], 'weight': 1., 'event': None}]}]
        for index, s in enumerate(designs):
            cache = {}
            for d in s['draws']:
                selected = tuple(d['indices_rotation_first'])
                if selected not in cache:
                    event = None if not tail else _make_event(selected, tail.components, dist, index)
                    qc, unitary, structure = circuit_for_event(con, time, event, operators)
                    costs = {}
                    for axis in ('X', 'Y'):
                        circ, ref = cc.wrapper(qc, unitary, axis, prepared)
                        cost = audit.compile(circ, ref, f'{group}/{con.name}/order{s["order"]}/{selected}/{axis}')
                        costs[axis] = {**cost, **structure}
                    cache[selected] = costs
                d['costs'] = cache[selected]
            row['strata'].append(s)
        stats = {a: {m: mean_se(row['strata'], a, m) for m in (*cc.METRICS, *STRUCTURAL)} for a in ('X', 'Y', 'XY')}
        row['stratified_cost'] = stats
        gamma = diagnostics[0]['normalization']; bias = diagnostics[0]['operator_bias']
        row['precision_resources'] = [{'epsilon_complex': e, 'shots_per_axis': shots,
            'normalization': gamma, 'bias': bias,
            'expected_work': None if shots is None else {m: shots*stats['XY'][m]['mean'] for m in cc.METRICS},
            'expected_work_se': None if shots is None else {m: shots*stats['XY'][m]['se'] for m in cc.METRICS}}
            for e in cc.EPSILONS for shots in [cc.shot_budget(gamma, bias, e)]]
        rows.append(row)
    differences = []
    for left, right in itertools.combinations(rows, 2):
        for e in cc.EPSILONS:
            for m in cc.METRICS:
                a, b = coupled_vector(left, m, e), coupled_vector(right, m, e)
                difference = None if a is None or b is None else a-b
                differences.append({'left': left['candidate'], 'right': right['candidate'], 'epsilon_complex': e,
                    'metric': m, 'mean_left_minus_right': None if difference is None else float(np.mean(difference)),
                    'paired_se': None if difference is None else float(np.std(difference, ddof=1)/math.sqrt(CONDITIONAL_DRAWS))})
    return {'hamiltonian': cc.matrix(h), 'time': time, 'q_grid': [1, 2, 4], 'cost_q': 1,
            'prepared_gates': prepared, 'candidates': candidates, 'rows': rows, 'paired_differences': differences}


def run_all():
    audit = cc.NativeAudit(cap=COMPILE_CAP)
    uniforms = np.random.default_rng(UNIFORM_SEED).random((CONDITIONAL_DRAWS, 3))
    factors = [np.array([[.8, .09, 0], [.09, -.35, 0], [0, 0, .15]]),
               np.array([[-.2, 0, 0], [0, .6, .07], [0, .07, -.45]])]
    h = sum(cc.second_quantize_one_body(g)@cc.second_quantize_one_body(g) for g in factors)
    acons = [cc.native_df_construct(factors, k) for k in (0, 1, 2)]
    acons += [whole_pauli(factors), cc.frame_construct(factors), complete_core(factors)]
    a = study(acons, h, .4, [(0, 'x')], audit, uniforms, 'A')
    a.update(input_factors=[cc.matrix(g) for g in factors], modes=3, geometry=None, chemical_basis=None,
             df_rank=2, rank_policy='exact synthetic positive-square rank2; no onebody/constant correction',
             bias_scope='full Fock dimension8; oneparticle prepared signal is sector-Gaussian, no correlated chemistry claim')
    reflected, old_direct, metadata = cc.enlarged_connection()
    u = np.array(metadata['orbital_isometry']['real'])+1j*np.array(metadata['orbital_isometry']['imag'])
    edges = metadata['diagonal_density_edges']; hp = old_direct.dense()
    bcons = [reflected, coefficient_direct(u, edges), physical_pairs(u, edges, involution=False),
             physical_pairs(u, edges, involution=True)]
    b = study(bcons, hp, .2, [(0, 'x'), (1, 'x')], audit, uniforms, 'B')
    b['metadata'] = metadata
    b['direct_coefficient_vs_old_dense_residual'] = cc.norm(bcons[1].dense()-old_direct.dense())
    b['task'] = 'physical full-Fock first-moment precision; aux vacuum; LD0 r1 K2; no reset/channel protection claim'
    return {'schema_version': 1, 'status': 'ATTRIBUTION_PAIR_COMPLETE_AWAITING_GPT_REVIEW', 'A': a, 'B': b,
            'C': {'new_runs': 0, 'status': 'prior evidence retained; no scan'},
            'common_uniforms': uniforms.tolist(), 'uniform_seed': UNIFORM_SEED, 'conditional_draws': CONDITIONAL_DRAWS,
            'uncertainty': 'order0 exact; small order2 exact, else96 iid conditional draws; shared inverse-CDF uniforms across candidates; paired XY and candidate difference SE; engineering SE only',
            'native_ir': audit.records, 'compiled_circuits': len(audit.records),
            'quantum_shots_executed': 0, 'molecular_loads': 0, 'gpu_calls': 0, 'ground_state_solves': 0,
            'central_hypothesis_adopted': None, 'next_stage_authorized': False, 'mandatory_stop': True}
