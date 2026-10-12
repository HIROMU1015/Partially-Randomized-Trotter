"""Saved-witness implications only; no generator, optimizer or numerical imports.

The uniform OLD exclusion lemma is sufficient. Failure of this lemma is not
OLD feasibility, OCC failure, or evidence against physical reduction.
"""
from fractions import Fraction as F

from .n3_old_family_contract_v1 import decode, encode, fingerprint, matrix, rational
from .n3_old_family_check_v1 import verify
from .n3_occupation_certificate_audit import Audit


def uniform_exclusion(instance, raw_W):
    h, v, nu, U, c, _ = decode(instance)
    W = matrix(raw_W)
    if len(W) != len(h):
        raise ValueError('W_DIMENSION')
    a = Audit()
    a.psd(W)
    if a.total(W[i][i] for i in range(len(W))) != 1:
        raise ValueError('NORMALIZED_W_REQUIRED')
    beta2 = a.total(a.q(x*x) for x in v)
    r = a.total(a.q(W[i][j] * (h[j][i] - (U/nu if i == j else 0)))
                for i in range(len(h)) for j in range(len(h)))
    b = a.total(a.q(v[i]*W[i][j]*v[j])
                for i in range(len(h)) for j in range(len(h)))
    margin = a.q(b-beta2/nu)
    sufficient = beta2 > 0 and r <= 0 and margin > 0
    return encode({'status': 'UNIFORM_OLD_EXCLUSION_PROVED' if sufficient
                   else 'SUFFICIENT_LEMMA_NOT_ESTABLISHED',
                   'r': r, 'b': b, 'beta_squared': beta2,
                   'strict_intercept_margin': margin, 'g': c-U,
                   'instance_sha256': fingerprint(instance),
                   'PSD_checks': a.psd_checks, 'checker_conditions': a.checks,
                   'max_rational_bits': a.max_bits,
                   'conclusion': 'OLD infeasible for all kappa>=0 and 0<delta<=delta_N; '
                   'infimum of NORM OR FIVE OR complete OLD is delta_N'
                   if sufficient else 'No family feasibility or impossibility verdict'})


def norm_enclosure(g, beta2, raw_low, raw_high):
    """Check an existing root enclosure by polynomials, without creating a root."""
    low, high = rational(raw_low), rational(raw_high)
    if not (g > 0 and beta2 > 0 and 0 < low <= high
            and low*(g+low) <= beta2 <= high*(g+high)):
        raise ValueError('NORM_ROOT_ENCLOSURE')
    return low, high


def saved_implication(instance, old_record, summary):
    old_audit = verify(instance, old_record)
    if old_record['certificate']['kind'] != 'dual':
        raise ValueError('SAVED_DUAL_REQUIRED')
    ans = uniform_exclusion(instance, old_record['certificate']['W'])
    if summary['input_id'] != instance['id']:
        raise ValueError('SAVED_SUMMARY_ID')
    if summary['common_U'] != instance['U'] or summary['common_c_star'] != instance['c']:
        raise ValueError('SAME_U_C_REQUIRED')
    h, v, nu, U, c, delta = decode(instance)
    beta2 = sum((x*x for x in v), F(0))
    norm = summary['widths']['NORM']
    low, high = norm_enclosure(c-U, beta2, norm['low'], norm['high'])
    width = summary['widths']['OCC_RAW_ALL_SAVED_SPECTRA']
    if width['high'] != instance['delta'] or width['positive_proof_delta'] != instance['delta']:
        raise ValueError('SAVED_OCC_WIDTH_BINDING')
    w = summary['winning_witnesses']['OCC_RAW_ALL_SAVED_SPECTRA']
    if width['upper_witness_id'] != w['id'] or w['kind'] != 'OCC':
        raise ValueError('SAVED_WINNER_ID')
    k, s = rational(w['kappa']), rational(w['s'])
    q = k*U-s
    if rational(w['q']) != q or not (0 <= k <= c-U+delta and q <= delta*(c-U+delta)):
        raise ValueError('SAVED_OCC_SCALAR_CONDITIONS')
    forced = (beta2-q)/k if k > 0 else None
    ans.update(encode({'old_target_audit': old_audit, 'saved_winner_id': w['id'],
        'saved_positive_delta': delta, 'norm_root_enclosure': [low, high],
        'achieved_relative_improvement_enclosure': [1-delta/low, 1-delta/high],
        'saved_q': q, 'forced_energy_minus_U_lower': forced,
        'spectral_validity': 'Inherited from immutable source-bound saved OCC/U,c audit; '
        'not replaced by this scalar implication check',
        'OCC_family_optimum_claimed': False, 'physical_error_reduction_claimed': False,
        'new_certificate_proposals': 0, 'new_eigensolver_calls': 0,
        'mandatory_stop': True, 'next_stage_authorized': False, 'central_theme': None}))
    return ans


def fixed_witness_accepts(kappa, s, U, c, delta):
    """Pure rational contract used by synthetic monotonicity tests."""
    if any(type(x) is not F for x in (kappa, s, U, c, delta)):
        raise ValueError('EXACT_FRACTIONS_REQUIRED')
    if kappa < 0 or c <= U or delta <= 0:
        raise ValueError('WITNESS_DOMAIN')
    D = c-U+delta
    return kappa <= D and kappa*U-s <= delta*D
