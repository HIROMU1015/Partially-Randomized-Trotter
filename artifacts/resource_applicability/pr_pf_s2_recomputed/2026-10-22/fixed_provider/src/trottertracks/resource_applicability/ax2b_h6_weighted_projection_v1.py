"""Saved-DF projection policy. Array operations run only on explicit caller input.

Engineering evaluation of a mathematical bound; finite precision norms are not
interval certificates. No molecule, decomposer, solver, signal or circuit API.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import math
import numpy as np

from .ax2b_h6_df_diagnostic_port_v1 import array_record

POLICY = 'WEIGHTED_HERMITIAN_PROJECTION_FROM_SAVED_RAW_V1'


def policy():
    return {'name': POLICY, 'df_tol': 1e-8, 'cutoff': 0., 'all_returned_fragments': True,
            'projection_budget_hartree': 1e-10, 'decision_limit_hartree': 9.9e-11,
            'decision_margin_fraction': .01, 'primary_N': 12, 'also_record_N': 6,
            'norm_evaluation': 'binary64 Frobenius; Decimal.from_float, 60-digit sum',
            'summary_atol': 1e-15, 'summary_rtol': 1e-8,
            'coefficient_atol': 1e-13, 'coefficient_rtol': 1e-10,
            'post_hermiticity_atol': 1e-12, 'cross_spin': 'exact zero',
            'alpha_beta': 'exact equality', 'representation_error_certified': False,
            'label': 'PASS_ENGINEERING', 'old_unweighted_tolerance': 1e-10}


def norm(a):
    value = float(np.linalg.norm(np.asarray(a).ravel()))
    if not math.isfinite(value):
        raise ValueError('PROJECTION_NONFINITE_NORM')
    return value


def structure(a, *, post=False):
    if not np.isfinite(a).all():
        raise ValueError('PROJECTION_NONFINITE')
    if np.any(a[::2, 1::2] != 0) or np.any(a[1::2, ::2] != 0):
        raise ValueError('PROJECTION_CROSS_SPIN')
    if not np.array_equal(a[::2, ::2], a[1::2, 1::2]):
        raise ValueError('PROJECTION_ALPHA_BETA')
    if post and norm(a-a.conj().T) > policy()['post_hermiticity_atol']:
        raise ValueError('PROJECTION_POST_HERMITICITY')


def canonical_antisym(t):
    """Independent pair-index construction, not the diagnostic swapaxis formula."""
    n = t.shape[0]
    out = np.zeros_like(t)
    for p in range(n):
        for q in range(p+1, n):
            for r in range(n):
                for s in range(r+1, n):
                    v = (t[p,q,r,s]-t[q,p,r,s]-t[p,q,s,r]+t[q,p,s,r])/4
                    out[p,q,r,s], out[q,p,r,s] = v, -v
                    out[p,q,s,r], out[q,p,s,r] = -v, v
    return out


def reconstruct(corrected, weights, blocks):
    """Normal-order coefficients of corrected + sum(lambda*dGamma(g)**2).

    Plain products, including signed real lambda. No conjugation. Constant is
    unchanged and is accounted outside these coefficients.
    """
    n = corrected.shape[0]
    one = np.array(corrected, dtype='<c16', copy=True)
    two = np.zeros((n,)*4, dtype='<c16')
    for weight, g in zip(weights, blocks, strict=True):
        for p in range(n):
            for q in range(n):
                one[p,q] += weight*sum(g[p,x]*g[x,q] for x in range(n))
        # outer has [p,r,q,s], then explicitly reorder to [p,q,r,s].
        two -= weight*np.outer(g.ravel(), g.ravel()).reshape((n,)*4).transpose(0,2,1,3)
    return one, canonical_antisym(two)


def close_scalar(value, expected, *, summary=True):
    if isinstance(expected, bool) or not isinstance(expected, (int,float)) or not math.isfinite(expected):
        raise ValueError('PROJECTION_COMPARISON_SCALAR')
    prefix = 'summary' if summary else 'coefficient'
    if abs(value-expected) > policy()[prefix+'_atol']+policy()[prefix+'_rtol']*max(abs(value),abs(expected)):
        raise ValueError('PROJECTION_INDEPENDENT_MISMATCH')


def accept(*, constant, one_body, two_body, lambdas, g_matrices, correction,
           truncation, summary=None, diagnostic_coefficients=None):
    """No file IO or implicit real data. Return new target arrays and receipt.

    Real use MUST supply the frozen summary and independently saved diagnostic
    coefficients, validated by the bound importer. Tests use explicit fixtures.
    """
    one, two = np.asarray(one_body), np.asarray(two_body)
    weights, gs, k = np.asarray(lambdas), np.asarray(g_matrices), np.asarray(correction)
    n = one.shape[0] if one.ndim == 2 else 0
    if (n not in (4,12) or one.shape != (n,n) or two.shape != (n,)*4
            or weights.ndim != 1 or not 1 <= len(weights) <= 19
            or weights.dtype.str != '<f8' or gs.shape != (len(weights),n,n)
            or k.shape != one.shape):
        raise ValueError('PROJECTION_LAYOUT_OR_REAL_LAMBDA')
    if (any(not np.isfinite(a).all() for a in (one,two,weights,gs,k))
            or isinstance(constant,bool) or not math.isfinite(float(constant))
            or np.asarray(truncation).shape != () or not math.isfinite(float(truncation))
            or not 0 <= float(truncation) <= policy()['df_tol']):
        raise ValueError('PROJECTION_INPUT_OR_TRUNCATION')
    corrected = one+k
    for a in (one, k, corrected, *gs):
        structure(a)
    post_one = (corrected+corrected.conj().T)/2
    posts = (gs+gs.conj().transpose(0,2,1))/2
    for a in (post_one, *posts):
        structure(a, post=True)
    dh = norm(post_one-corrected)
    rows = []
    with localcontext() as ctx:
        ctx.prec = 60
        D = Decimal.from_float
        total = Decimal(0)
        for i, (weight, g, post) in enumerate(zip(weights, gs, posts, strict=True)):
            s, d = norm(g), norm(post-g)
            term = D(abs(float(weight)))*D(d)*(2*D(s)+D(d))
            total += term
            rows.append({'index':i, 'lambda':float(weight), 'abs_lambda':abs(float(weight)),
                'raw':array_record(g), 'after':array_record(post), 'raw_frobenius_norm':s,
                'hermiticity_defect_frobenius':norm(g-g.conj().T),
                'difference_frobenius':d, 'relative_change':d/s if s else None,
                'old_gate_exceeded':d > 1e-10, 'bound_contribution_before_N_squared':str(term)})
        bounds = {str(N):str(N*D(dh)+N*N*total) for N in (6,12)}
        if Decimal(bounds['12']) > Decimal('9.9e-11'):
            raise ValueError('PROJECTION_WEIGHTED_BUDGET')
    raw_one, raw_two = reconstruct(corrected,weights,gs)
    post_c, post_t = reconstruct(post_one,weights,posts)
    g_post_c, _ = reconstruct(corrected,weights,posts)
    raw_residual = raw_one-one
    target_two = canonical_antisym(two)
    stats = {'raw_normal_order_one_body_residual_frobenius':norm(raw_residual),
             'raw_normal_order_antisym_two_body_residual_frobenius':norm(target_two-raw_two),
             'hypothetical_projection_one_body_change_frobenius':norm(raw_one-g_post_c),
             'hypothetical_projection_antisym_two_body_change_frobenius':norm(raw_two-post_t)}
    # Check the independent reconstruction against saved arrays, not just norms.
    if diagnostic_coefficients is not None:
        for actual, expected in zip((raw_residual,raw_two,target_two),diagnostic_coefficients,strict=True):
            if actual.shape != expected.shape or not np.isfinite(expected).all():
                raise ValueError('PROJECTION_DIAGNOSTIC_COEFFICIENT_LAYOUT')
            if norm(actual-expected) > policy()['coefficient_atol']+policy()['coefficient_rtol']*max(norm(actual),norm(expected)):
                raise ValueError('PROJECTION_INDEPENDENT_COEFFICIENT_MISMATCH')
    if summary is not None:
        if summary['actual_rank'] != len(weights) or summary['kwargs'] != {'truncation_threshold':1e-8}:
            raise ValueError('PROJECTION_SUMMARY_SCOPE')
        close_scalar(float(truncation),summary['truncation_value'])
        close_scalar(dh,summary['corrected_one_body_hermitization_change_frobenius'])
        if len(summary['fragments']) != len(rows):
            raise ValueError('PROJECTION_SUMMARY_RANK')
        for a,b in zip(rows,summary['fragments'],strict=True):
            if a['index'] != b['index'] or a['lambda'] != b['lambda'] or a['raw'] != b['raw'] or a['after'] != b['hypothetical_post']:
                raise ValueError('PROJECTION_SUMMARY_FRAGMENT_IDENTITY')
            close_scalar(a['raw_frobenius_norm'],b['raw_frobenius_norm'])
            close_scalar(a['difference_frobenius'],b['hermitization_change_frobenius'])
        for key,value in stats.items():
            close_scalar(value,summary['representation_diagnostics'][key])
    receipt = {'schema':'track_a_h6_weighted_projection_receipt_v1','policy':policy(),
        'status':'PASS_ENGINEERING','actual_rank':len(weights),'lambdas_raw':array_record(weights),
        'corrected_one_body_raw':array_record(corrected),'corrected_one_body_after':array_record(post_one),
        'corrected_one_body_change_frobenius':dh,'correction_raw':array_record(k),
        'constant':float(constant),'fragments':rows,'eta_N_hartree_decimal':bounds,
        'maximum_raw_change_index':max(rows,key=lambda r:r['difference_frobenius'])['index'],
        'maximum_weighted_contribution_index':max(rows,key=lambda r:Decimal(r['bound_contribution_before_N_squared']))['index'],
        'old_failed_fragment_indices':[r['index'] for r in rows if r['old_gate_exceeded']],
        'error_ledger':{'provider_truncation_value':float(truncation),'raw_df_coefficients':stats,
            'projection_fragment_triangle_bound':bounds,
            'projection_accumulated_coefficient_bound_evaluation':n*norm(post_c-raw_one)+n*n*norm(post_t-raw_two),
            'raw_to_integrals_coefficient_bound_evaluation':n*norm(raw_residual)+n*n*norm(target_two-raw_two),
            'PF_RTE':None,'roundoff_certificate':None,'measurement':None},
        'independent_coefficients_checked':diagnostic_coefficients is not None,
        'saved_summary_checked':summary is not None,'representation_error_certified':False,
        'old_stop_reclassified':False,'H6_status':'H6_NOT_AUTHORIZED',
        'contract_status':'DRAFT_NOT_AUTHORIZATION','mandatory_stop':True,'next_stage_authorized':False}
    arrays = {'constant':np.array(constant,dtype='<f8'),'one_body':post_one,
              'lambdas':weights.copy(),'g_matrices':posts}
    coefficients = {'raw_one_body_residual':raw_residual,'raw_two_body_antisym':raw_two,
                    'target_two_body_antisym':target_two,'accepted_one_body':post_c,'accepted_two_body_antisym':post_t}
    return arrays,receipt,coefficients
