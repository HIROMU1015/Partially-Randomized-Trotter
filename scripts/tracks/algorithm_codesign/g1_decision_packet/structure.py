"""Deferred symbolic audit. No work at import; CLI requires a consumed launch marker."""
import ast
import hashlib
import itertools
import json
from fractions import Fraction as F
from pathlib import Path
import sys
import time

from .rational_symbolic import RF, determinant, evaluate_ast, parse, rref, sign_proof, solve_many


class Counterexample(Exception):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__('independent identity differs from frozen proposal')


def require_zero(value, label, variable, points):
    if value == 0: return
    for point in points:
        if value.evaluate(F(point)) != 0:
            raise Counterexample({'claim': label, 'residual': value.json(),
                                  'off_domain_witness': {variable: point, 'residual': str(value.evaluate(F(point)))}})
    raise ArithmeticError('nonzero identity without a witness in the fixed supplemental set')


def source_vectors(path, order):
    """Restricted AST interpreter reads just coefficient expressions, never source imports."""
    tree = ast.parse(Path(path).read_text())
    assignments = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('rho', 'formulas'):
                if name in assignments: raise ValueError('ambiguous coefficient assignment')
                assignments[name] = node.value
    x = RF.variable()
    rho = evaluate_ast(assignments['rho'], {'x': x})
    node = assignments['formulas']
    if not isinstance(node, ast.Dict): raise ValueError('literal coefficient dictionary required')
    found = {}
    for key, value in zip(node.keys, node.values):
        if not isinstance(key, ast.Constant) or not isinstance(value, ast.Tuple) or len(value.elts) != 3:
            raise ValueError('prototype shape')
        k, a, b = [evaluate_ast(v, {'x': x, 'rho': rho}) for v in value.elts]
        if len(k.num) != 1 or k.den != (F(1),) or k.num[0].denominator != 1:
            raise ValueError('degree must be a literal integer')
        degree = int(k.num[0])
        column = [RF(0)]*4
        column[degree] = a
        if b != 0:
            if degree+1 >= 4: raise ValueError('degree overflow')
            column[degree+1] = b
        found[key.value] = column
    if set(found) != set(order): raise ValueError('prototype dictionary differs')
    return x, rho, [found[name] for name in order]


def phase_convention_present(path):
    expected = ast.dump(ast.parse('e["phase_i_power"] != (-degree) % 4', mode='eval').body)
    return any(isinstance(node, ast.Compare) and ast.dump(node) == expected
               for node in ast.walk(ast.parse(Path(path).read_text())))


def determinant3(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
            -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
            +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def enumerate_intersections(faces, rhs):
    records, vertices = [], set()
    for indices in itertools.combinations(range(len(faces)), 3):
        a, b = [faces[i] for i in indices], [[rhs[i]] for i in indices]
        det = determinant3(a)
        row = {'faces': list(indices), 'determinant': det.json()}
        if det == 0:
            reduced, pivots = rref([r+[v[0]] for r, v in zip(a, b)], 3)
            inconsistent = any(all(v == 0 for v in r[:3]) and r[3] != 0 for r in reduced)
            row.update(kind='SINGULAR_NO_UNIQUE_INTERSECTION', generic_rank=len(pivots),
                       generic_consistency='INCONSISTENT' if inconsistent else 'DEPENDENT',
                       justification='identically zero determinant excludes a unique triple intersection throughout the domain')
        else:
            det_sign = sign_proof(det, 'unit')
            if det_sign['sign'] not in (-1, 1): raise ArithmeticError('unproved determinant sign')
            point = tuple(r[0] for r in solve_many(a, b))
            slacks = [right-sum((c*v for c, v in zip(left, point)), RF(0))
                      for left, right in zip(faces, rhs)]
            signs = [sign_proof(v, 'unit') for v in slacks]
            if any(v['sign'] is None for v in signs): raise ArithmeticError('unproved feasibility sign')
            feasible = all(v['sign'] >= 0 for v in signs)
            row.update(kind='FEASIBLE_VERTEX' if feasible else 'INFEASIBLE_INTERSECTION',
                       point=[v.json() for v in point], determinant_sign=det_sign,
                       slacks=[v.json() for v in slacks], slack_sign_certificates=signs)
            if feasible: vertices.add(point)
        records.append(row)
    return records, vertices


def linear(*values):
    return list(map(RF, values))


def linear_sum(forms):
    return [sum((a[j] for a in forms), RF(0)) for j in range(4)]


def decomposition_cases(mu):
    """Explicit coefficient convex decomposition; column order (1,s,r,b)."""
    zero = linear(0, 0, 0, 0)
    return [
        {'condition': 's+r>=1', 'weights': {
            'ordinary': linear(0, 0, 0, 1), 'PTSC_K0': linear(1, 0, -1, -1),
            'A': linear(1, -1, 0, 0), 'J3': linear(-1, 1, 1, 0), 'J1': zero, 'J2': zero},
         'nonnegative_reason': 'b>=0; 1-r-b>=0; 1-s>=0; s+r-1>=0'},
        {'condition': 's+r<=1 and b<=s', 'weights': {
            'ordinary': linear(0, 0, 0, 1), 'PTSC_K0': linear(0, 1, 0, -1),
            'A': linear(0, 0, 1, 0), 'J1': linear(1, -1, -1, 0), 'J2': zero, 'J3': zero},
         'nonnegative_reason': 'b>=0; s-b>=0; r>=0; 1-s-r>=0'},
        {'condition': 's+r<=1 and b>=s', 'weights': {
            'ordinary': linear(0, 1, 0, 0), 'PTSC_K0': zero, 'A': linear(0, 0, 1, 0),
            'J2': linear(0, -1/mu, 0, 1/mu),
            'J1': linear(1, (1-mu)/mu, -1, -1/mu), 'J3': zero},
         'nonnegative_reason': 's>=0; r>=0; (b-s)/mu>=0; J1=(mu+(1-mu)s-mu*r-b)/mu>=0'}]


def audit(root, contract):
    """Only called by the separately authorized one-shot subprocess, never by tests."""
    began = time.perf_counter()
    xs = contract['supplemental_off_domain_points']['x']
    mus = contract['supplemental_off_domain_points']['mu']
    order = contract['prototype_order']
    x, rho, columns = source_vectors(root/contract['source_prototype_file'], order)
    if not phase_convention_present(root/contract['source_prototype_file']):
        raise ValueError('fixed source phase/degree convention not identified')
    target = [parse(v.replace('^', '**'), x=x) for v in contract['degree_target']]
    mu_x = (x*x+2)/(x*x+6)
    positivity = {name: sign_proof(v, 'positive') for name, v in {
        'rho': rho, 'x_minus_rho': x-rho, 'mu_minus_third': mu_x-F(1, 3), 'one_minus_mu': 1-mu_x}.items()}
    if any(p['sign'] != 1 for p in positivity.values()): raise ArithmeticError('domain sign not proved')
    # Free coefficients are source O0, A2, O2. Solve the other four, not the GPT formula.
    free, dependent = [order.index(v) for v in ('O0', 'A2', 'O2')], [order.index(v) for v in ('P2', 'P3', 'A0', 'A1')]
    matrix = [[columns[g][k] for g in dependent] for k in range(4)]
    minor = determinant(matrix)
    minor_sign = sign_proof(minor, 'positive')
    if minor_sign['sign'] not in (-1, 1): raise ArithmeticError('rank witness nonzero domain sign not proved')
    rhs = [[target[k]]+[-columns[g][k] for g in free] for k in range(4)]
    solved = solve_many(matrix, rhs)
    coefficients = [None]*7
    for j, g in enumerate(free):
        coefficients[g] = [RF(0)]*4
        coefficients[g][j+1] = RF(1)
    for g, values in zip(dependent, solved): coefficients[g] = values
    proposed = [linear(0, 1, 0, 0), linear(0, 0, 0, 1),
                linear(mu_x, 1-mu_x, -mu_x, -1), linear(1, 0, -1, -1),
                linear(1, -1, 0, 0), linear(1, -1, 0, 0), linear(0, 0, 1, 0)]
    for g in range(7):
        for j in range(4): require_zero(coefficients[g][j]-proposed[g][j], f'parameterization:{order[g]}:{j}', 'x', xs)
    for k in range(4):
        for j in range(4):
            require_zero(sum((columns[g][k]*coefficients[g][j] for g in range(7)), RF(0))
                         -(target[k] if j == 0 else 0), f'general_mean:{k}:{j}', 'x', xs)
    mu = RF.variable()
    # These facets are the seven independently derived nonnegative gamma conditions (one duplicate removed).
    faces = [linear(-1, 0, 0), linear(1, 0, 0), linear(0, -1, 0),
             linear(0, 0, -1), linear(0, 1, 1), linear(-(1-mu), mu, 1)]
    right = list(map(RF, [0, 1, 0, 0, 1]))+[mu]
    interior = list(map(RF, [F(1, 2), F(1, 4), F(1, 8)]))
    interior_slacks = [sign_proof(v-sum((a*z for a, z in zip(row, interior)), RF(0)), 'unit')
                       for row, v in zip(faces, right)]
    if any(s['sign'] != 1 for s in interior_slacks): raise ArithmeticError('not proved three dimensional')
    records, vertices = enumerate_intersections(faces, right)
    expected = {name: tuple(parse(v, mu=mu) for v in point)
                for name, point in contract['proposed_vertices'].items()}
    if vertices != set(expected.values()):
        raise Counterexample({'claim': 'vertex_completeness', 'independently_obtained': [[v.json() for v in p] for p in vertices],
                              'twenty_triple_sign_certificates': records})
    named = {name: point for name, point in expected.items() if point in vertices}
    for name, point in named.items():
        sx, rx, bx = [v.compose(mu_x) for v in point]
        gamma = [sum((c*v for c, v in zip(form, [RF(1), sx, rx, bx])), RF(0)) for form in coefficients]
        for k in range(4):
            require_zero(sum((columns[g][k]*gamma[g] for g in range(7)), RF(0))-target[k], f'vertex_mean:{name}:{k}', 'x', xs)
    cases = decomposition_cases(mu)
    for case in cases:
        total = linear_sum(case['weights'].values())
        for j, value in enumerate(total): require_zero(value-(1 if j == 0 else 0), 'convex_weight_sum', 'mu', mus)
        for axis in range(3):
            reconstructed = [sum((case['weights'][name][j]*named[name][axis] for name in named), RF(0)) for j in range(4)]
            for j in range(4): require_zero(reconstructed[j]-(1 if j == axis+1 else 0), 'convex_coordinate', 'mu', mus)
    obligations = [t['id'] for t in contract['audit_tasks']]
    output = {
        'classification': 'G1_STRUCTURE_PASS_WITH_DECLARED_LIMITS',
        'obligations': {key: 'RESOLVED_FOR_STATED_IDEAL_CLASS' for key in obligations},
        'prototype_vectors': {name: [v.json() for v in column] for name, column in zip(order, columns)},
        'signed_phase_convention': 'Each degree k has fixed (-i*sigma)^k; column entries are pre-normalization positive degree coefficients, not sampling probabilities.',
        'domain_positivity': positivity, 'derived_parameterization': [[v.json() for v in form] for form in coefficients],
        'dimension_proof': {'rank': 4, 'dependent_minor': minor.json(), 'minor_sign_certificate': minor_sign,
                            'free_parameters': ['s', 'r', 'b'], 'strict_interior_slack_certificates': interior_slacks,
                            'boundedness': '0<=s<=1; r,b>=0 and r+b<=1 -> r,b<=1'},
        'boundary_triples': records, 'independently_enumerated_vertex_count': len(vertices),
        'completeness_reason': 'A bounded full-dimensional polytope has only vertices incident to three independent facets; all 20 triples were classified with exact domain signs.',
        'B2_embedding': 'theta_O=b, theta_A=r, theta_P=s-b; sum(theta)=1 iff r=1-s; nonnegative iff 0<=b<=s.',
        'convex_decomposition_cases': [{**case, 'weights': {k: [v.json() for v in f] for k, f in case['weights'].items()}} for case in cases],
        'precision_restoration': {'positive_group': 'pi_gp=gamma_gp/gamma_g; gamma_gp=sum_l theta_l vertex_lg pi_gp.',
                                 'zero_group': 'Nonnegativity and sum(theta_l vertex_lg)=0 imply every positive-weight contribution is zero. All gamma_gp=0; no share division is performed.',
                                 'no_precision_vertex_optimization_claim': True},
        'v4_scope': 'For y>0, U_g=sum_p u_gp=y gamma_g. Ideal c_g gamma_g is a weight; cbar is an enclosure midpoint, not the exact c_g. Exact matching does not cover all numerical K3.',
        'sampler_proof': {'component_probability': 'theta_l B_l / B; B=sum_l theta_l B_l',
                          'joint_cancellation': '(theta_l B_l/B)*(c_g vertex_lg pi_gp/B_l) gives the canonical aggregate weight.',
                          'different_estimator': 'Choosing by theta_l and weighting each draw B_l has second moment sum_l theta_l B_l^2, whereas canonical constant weight has B^2; difference=sum_{l<m} theta_l theta_m (B_l-B_m)^2.'},
        'limits': ['fixed seven prototypes/P3/x>0 only', 'ideal coefficient decomposition only', 'numerical K3 equivalence not proved',
                   'separate LRM decode images need not agree', 'resource caps may favor mixtures', 'no cost scoring, new B2 baseline, or novelty claim'],
        'registered_LP_quantum_matrix_synthesis_cost_scoring': 0,
        'CPU_or_wall_seconds_diagnostic': time.perf_counter()-began,
    }
    return output


def main():
    root, contract_path, permit_path = map(Path, sys.argv[1:4])
    actual_root = Path(__file__).resolve().parents[4]
    if root.resolve() != actual_root: raise PermissionError('wrong audit source root')
    from .controller import APPROVAL_SENTENCE, PREP, SOURCE_MANIFEST
    packet_path = root/(PREP+'/decision_packet_contract_v1.json')
    packet = json.loads(packet_path.read_text())
    if contract_path != root/packet['structure_contract']:
        raise PermissionError('wrong structure contract')
    permit = json.loads(permit_path.read_text())
    marker_path = Path(permit['marker'])
    if marker_path != Path(packet['state']['one_shot_marker']) or permit_path != marker_path.parent/'structure_permit.json':
        raise PermissionError('wrong one-shot state root')
    marker = marker_path.read_bytes()
    if permit.get('stage') != 'A_STRUCTURE_AUDIT' or hashlib.sha256(marker).hexdigest() != permit['marker_sha256']:
        raise PermissionError('source-bound one-shot permit required')
    consumed = json.loads(marker)
    if consumed.get('explicit_one_shot_instruction_bound') is not True:
        raise PermissionError('no explicit execution instruction')
    if hashlib.sha256(contract_path.read_bytes()).hexdigest() != permit['structure_contract_sha256']:
        raise PermissionError('structure contract mismatch')
    if hashlib.sha256(packet_path.read_bytes()).hexdigest() != consumed['contract_sha256']:
        raise PermissionError('packet contract mismatch')
    if hashlib.sha256((root/SOURCE_MANIFEST).read_bytes()).hexdigest() != consumed['source_manifest_sha256']:
        raise PermissionError('source manifest mismatch')
    if APPROVAL_SENTENCE.format(source=consumed['source_commit']) not in consumed['instruction']:
        raise PermissionError('explicit scope not recorded')
    import os
    parent_command = Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0')
    expected_guard = str(root/'scripts/tracks/algorithm_codesign/exact_backend_pilot_v2/guard.py').encode()
    expected_spec = str(marker_path.parent/'specs'/'A_STRUCTURE_AUDIT.json').encode()
    if expected_guard not in parent_command or expected_spec not in parent_command:
        raise PermissionError('audit must be an active guarded controller subprocess')
    started = json.loads((marker_path.parent/'ledger'/'A_STRUCTURE_AUDIT.started.json').read_text())
    if started['id'] != 'A_STRUCTURE_AUDIT' or started['command'][-1] != str(permit_path):
        raise PermissionError('exclusive audit stage ledger missing/mismatched')
    try:
        result = audit(root, json.loads(contract_path.read_text()))
    except Counterexample as exc:
        result = {'classification': 'G1_STRUCTURE_COUNTEREXAMPLE', 'counterexample': exc.evidence}
    except Exception as exc:
        result = {'classification': 'G1_STRUCTURE_TECHNICAL_INCONCLUSIVE', 'reason': f'{type(exc).__name__}: {exc}'}
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
