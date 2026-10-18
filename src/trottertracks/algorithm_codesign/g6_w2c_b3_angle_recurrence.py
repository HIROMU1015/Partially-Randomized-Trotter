"""Non-adopted artificial-only angle prototype. Never replaces F3 globals/session.

The enclosure body follows the fixed g6_f3_precision API, retaining its root,
validation, Evidence codec, failure detail and context. Only the two endpoint
calls are selected explicitly. A future production integration needs a separately
reviewed/pinned F3 backend connection; no monkeypatch or live adapter is provided.
"""
from fractions import Fraction as F


def atan_endpoint_recurrence(z, terms):
    if type(z) is not F or type(terms) is not int or terms < 0:
        raise ValueError('EXACT_FRACTION_AND_NONNEGATIVE_TERMS_REQUIRED')
    power, square, partial = z, -z*z, F(0)
    for j in range(terms):
        partial += power/(2*j+1)
        power *= square
    next_term = power/(2*terms+1)
    return (partial+min(F(0), next_term), partial+max(F(0), next_term),
            partial, abs(next_term))


def enclose_atan_artificial(ratio, *, tau, root_bits, terms, sigma=1, context=None, series='recurrence'):
    """Existing half-angle identity and alternating-series remainder, exactly.

    One supplied root precision and one finite series length, never refined.
    This returns mathematical angle evidence, not native rotation evidence.
    """
    if type(context) is not dict or context.get('ARTIFICIAL_ONLY') is not True:
        raise ValueError('ARTIFICIAL_INPUT_CONTEXT_REQUIRED')
    if series not in ('reference', 'recurrence'):
        raise ValueError('UNKNOWN_SERIES')
    from .g6_f3_precision import (exact_rational, exact_integer, invalid, Evidence,
        F3Failure, root_interval, comparison, PROOF_COMMIT, _atan_endpoint)
    endpoint = _atan_endpoint if series == 'reference' else atan_endpoint_recurrence
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
        left, right = endpoint(z_lo, terms), endpoint(z_hi, terms)
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
