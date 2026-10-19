"""F1 P0--P4 and the existing half-angle/alternating-atan enclosure.

All precision is supplied by the caller. No selection, refinement, retry or RNG.
"""
from fractions import Fraction as F
from math import factorial

from .g6_f3_contract import (Evidence, F3Failure, Request, fingerprint, exact_integer,
                             exact_rational, invalid, make_request, SPEC_COMMIT,
                             PROOF_COMMIT, POLICY)
from .return_aggregation import root_interval


def comparison(name, left, operator, right):
    operations = {'<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
                  '==': lambda a, b: a == b}
    return {'name': name, 'left': left, 'operator': operator, 'right': right,
            'passed': operations[operator](left, right)}


def rounding_predicate(zeta, q_min, eta, h):
    zeta, q_min, eta, h = (exact_rational(v, n) for v, n in
                          ((zeta, 'zeta'), (q_min, 'q_min'), (eta, 'eta'), (h, 'h')))
    if zeta < 0 or q_min <= 0 or h <= 0 or not 0 < eta < 1:
        raise invalid('INVALID_ROUNDING_PREDICATE_DOMAIN', 'preflight.R')
    checks = [comparison('zeta_less_than_one', zeta, '<', F(1))]
    if zeta < 1:
        lower = (1-zeta)/(1+zeta) - (1-eta)*(1+zeta)
        upper = (1+eta)*(1-zeta) - (1+zeta)/(1-zeta)
        checks += [comparison('lower_relative_guard', h/q_min, '<=', lower),
                   comparison('upper_relative_guard', h/q_min, '<=', upper)]
    return checks


def coefficient_predicate(delta, d_min, rho):
    delta, d_min, rho = (exact_rational(v, n) for v, n in
                         ((delta, 'delta'), (d_min, 'd_min'), (rho, 'rho')))
    if delta <= 0 or d_min <= 0 or rho <= 0:
        raise invalid('INVALID_COEFFICIENT_PREDICATE_DOMAIN', 'preflight.P3')
    return [comparison('positive_root_lower_bound', delta, '<', d_min),
            comparison('relative_midpoint_guard', delta/2, '<=', rho*(d_min-delta))]


def acceptance_predicate(delta, h, eta):
    delta, h, eta = (exact_rational(v, n) for v, n in
                     ((delta, 'delta'), (h, 'h'), (eta, 'eta')))
    if delta <= 0 or h <= 0 or not 0 < eta < 1:
        raise invalid('INVALID_ACCEPTANCE_PREDICATE_DOMAIN', 'preflight.P4')
    return [comparison('positive_acceptance_lower_endpoint', delta, '<', F(1, 4)),
            comparison('relative_downward_threshold', (2-eta)*delta+h, '<=', eta/4)]


def _predicate(identifier, checks, bounds=None, reason=''):
    passed = all(c['passed'] for c in checks)
    return {'id': identifier, 'status': 'PASS' if passed else 'FAIL', 'passed': passed,
            'checks': checks, 'bounds': bounds or {},
            'reason': '' if passed else reason,
            'basis': {'spec_commit': SPEC_COMMIT, 'section': identifier,
                      'proof_commit': PROOF_COMMIT, 'existing_lemma': 'finite-bit audit section 5'}}


def assess_precision(request, source):
    """P0--P4 only: P5 cannot pass until the actual generator law is bound."""
    if type(request) is not Request:
        raise invalid('VALIDATED_REQUEST_REQUIRED', 'preflight.request')
    request = make_request(**request.payload())
    if request.x == 0:
        raise invalid('IDENTITY_BYPASSES_POSITIVE_PREFLIGHT', 'preflight.domain')
    if request.arm != 'full_return':
        raise invalid('CANONICAL_PREFLIGHT_NOT_IMPLEMENTED', 'preflight.arm')
    if type(source) is not Evidence:
        raise invalid('SOURCE_EVIDENCE_REQUIRED', 'preflight.source')
    source.read()
    p_min = min(request.p)
    t_min = request.x**(request.m-1) / factorial(request.m-1)
    d_min = t_min * p_min**(request.m-1) / 2
    q_min = t_min / 3
    delta, h = F(1, 1 << request.root_bits), F(1, 1 << request.probability_bits)
    n_o = (request.m+1)//2
    e, a = n_o*delta, delta/t_min
    order_checks = [comparison('total_root_relative_error', e, '<', F(1))]
    zeta = None
    if e < 1:
        zeta = (a+e)/(1-e)
        order_checks += rounding_predicate(zeta, q_min, request.eta, h)
    predicates = [
        _predicate('P0', [comparison('validated_exact_domain', True, '==', True)],
                   {'fresh_uniform_independent_bits': 'caller assumption, not verified',
                    'controlled_Q': 'conditional exact query model'}),
        _predicate('P1', order_checks,
                   {'n_o': n_o, 'e': e, 'a': a, 'zeta_o': zeta, 'q_min': q_min},
                   'ORDER_NORMALIZED_INTERVAL_PRECISION_INSUFFICIENT'),
        _predicate('P2', [comparison('labels', h, '<=', request.eta*p_min),
                          comparison('children', h, '<=', request.eta*p_min/2)],
                   {'label_min': p_min, 'child_min': p_min/2},
                   'DYADIC_SUPPORT_PRECISION_INSUFFICIENT'),
        _predicate('P3', coefficient_predicate(delta, d_min, request.rho),
                   {'d_min': d_min}, 'COEFFICIENT_ROOT_PRECISION_INSUFFICIENT'),
        _predicate('P4', acceptance_predicate(delta, h, request.eta),
                   {'conditional_A_min': F(1, 4)}, 'ACCEPTANCE_PRECISION_INSUFFICIENT'),
        {'id': 'P5', 'status': 'NOT_BOUND', 'passed': False, 'checks': [],
         'reason': 'ACTUAL_LAW_BINDING_PENDING'},
    ]
    return Evidence.create({
        'schema': 'g6_f3_preflight_v1', 'mode': 'FULL_RETURN', 'input_id': request.input_id,
        'policy_id': request.policy_id, 'source_id': source.digest, 'policy': POLICY,
        'request': request.payload(), 'input_dimensions': request.dimensions(), 'delta': delta, 'h': h,
        'p_min': p_min, 't_min': t_min, 'predicates': predicates,
        'precision_conditions_satisfied': all(p['passed'] for p in predicates[:5]),
        'ready': False, 'all_parent_enumeration_used': False,
        'fixed_supplied_precision': True, 'precision_values_selected': False,
        'fresh_IID_verified': False, 'physical_provider_certified': False,
        'run_completion_or_confidence_certified': False,
    })


def bind_law(assessment, actual_state, expected_laws, source):
    data = assessment.read()
    if not data['precision_conditions_satisfied']:
        raise F3Failure('TECHNICAL_PRECISION_FAILURE', 'PREFLIGHT_CONDITIONS_FAILED',
                        'preflight.bind', input_id=data['input_id'], details=data)
    req = data['request']
    checks = [comparison('source_identity', source.digest, '==', data['source_id'])]
    for key in ('p', 'x', 'm', 'arm', 'eta', 'rho', 'root_bits', 'probability_bits'):
        checks.append(comparison('generator_' + key, actual_state['parameters'][key], '==', req[key]))
    for key in ('p', 'x', 'm'):
        checks.append(comparison('kernel_' + key, actual_state['kernel'][key], '==', req[key]))
    for key in ('order', 'labels'):
        checks.append(comparison('actual_' + key + '_law', actual_state[key], '==', expected_laws[key]))
    if not all(c['passed'] for c in checks):
        raise F3Failure('CONTRACT_INCONSISTENCY', 'ACTUAL_LAW_BINDING_FAILED',
                        'preflight.P5', input_id=data['input_id'], details={'checks': checks})
    law_id = fingerprint(actual_state)
    data['predicates'][-1] = {
        'id': 'P5', 'status': 'PASS', 'passed': True,
        'checks': checks,
        'actual_law_state_id': law_id,
        'fresh_bits_requirement': 'external fresh independent uniform bits; not empirically verified',
    }
    data['actual_law_state_id'] = law_id
    data['ready'] = True
    return Evidence.create(data)


def _atan_endpoint(z, terms):
    partial = sum(((-1)**j * z**(2*j+1) / (2*j+1) for j in range(terms)), F(0))
    next_term = (-1)**terms * z**(2*terms+1)/(2*terms+1)
    return (partial+min(F(0), next_term), partial+max(F(0), next_term),
            partial, abs(next_term))


def enclose_atan(ratio, *, tau, root_bits, terms, sigma=1, context=None):
    """Existing half-angle identity and alternating-series remainder, exactly.

    One supplied root precision and one finite series length, never refined.
    This returns mathematical angle evidence, not native rotation evidence.
    """
    ratio, tau = exact_rational(ratio, 'ratio'), exact_rational(tau, 'tau')
    root_bits = exact_integer(root_bits, 'angle_root_bits', positive=True)
    terms = exact_integer(terms, 'angle_terms', positive=True)
    sigma = exact_integer(sigma, 'sigma')
    if not 0 <= ratio <= 2 or tau <= 0 or sigma not in (-1, 1):
        raise invalid('INVALID_ANGLE_DOMAIN', 'angle.entrance')
    if context is not None and type(context) is not dict:
        raise invalid('ANGLE_CONTEXT_MAPPING_REQUIRED', 'angle.context')
    binding = context or {'scope': 'standalone mathematical angle; no parent binding'}
    try:
        Evidence.create(binding)
    except (TypeError, ValueError) as exc:
        raise invalid('EXACT_ANGLE_CONTEXT_REQUIRED', 'angle.context') from exc
    root_data, z_data, endpoints = None, None, None
    if ratio == 0:
        lo, hi = F(0), F(0)
    else:
        root = root_interval(1+ratio*ratio, root_bits)
        if not (1 <= root.lo <= root.hi and root.lo**2 <= 1+ratio*ratio <= root.hi**2):
            raise F3Failure('CONTRACT_INCONSISTENCY', 'INVALID_ROOT_ENCLOSURE', 'angle.root')
        z_lo, z_hi = ratio/(1+root.hi), ratio/(1+root.lo)
        if not 0 <= z_lo <= z_hi < 1:
            raise F3Failure('TECHNICAL_PRECISION_FAILURE', 'HALF_ANGLE_INTERVAL_INSUFFICIENT',
                            'angle.half_angle', details={'z_interval': (z_lo, z_hi)})
        left, right = _atan_endpoint(z_lo, terms), _atan_endpoint(z_hi, terms)
        lo, hi = 2*left[0], 2*right[1]
        root_data, z_data = (root.lo, root.hi), (z_lo, z_hi)
        endpoints = {'at_lo': left, 'at_hi': right}
    radius, midpoint = (hi-lo)/2, (hi+lo)/2
    data = {'schema': 'g6_f3_angle_enclosure_v1', 'ratio': ratio, 'sigma': sigma,
            'root_bits': root_bits, 'terms': terms, 'tau': tau, 'context': binding,
            'sqrt_interval': root_data, 'half_argument_interval': z_data,
            'endpoint_series_evidence': endpoints, 'phi_interval': (lo, hi),
            'representative': midpoint, 'representative_error_upper': radius,
            'signed_phi_interval': (lo, hi) if sigma == 1 else (-hi, -lo),
            'signed_representative': sigma*midpoint,
            'precision_check': comparison('representative_radius', radius, '<=', tau),
            'native_rotation_acquired': False, 'physical_provider_certified': False,
            'basis': {'proof_commit': PROOF_COMMIT, 'section': 'finite-bit audit 6'}}
    if not data['precision_check']['passed']:
        raise F3Failure('TECHNICAL_PRECISION_FAILURE', 'ANGLE_PRECISION_INSUFFICIENT',
                        'angle.enclosure', input_id=binding.get('input_id'), details=data)
    return Evidence.create(data)
