"""H8 bias *evidence* inventory. Never subtract, sum or rank real bias values.

Independent forward endpoints are evidence for a future numerical cross-check;
their presence does not certify independently verified signed components.
"""
from collections import Counter
import hashlib
import json
import math

AXES = ('real', 'imag')
SAVED = 'SAVED_COMPONENTS_PENDING_NUMERICAL_VALIDATION'
RECOVER = 'SAVED_ENDPOINTS_RECONSTRUCTION_REQUIRES_SEPARATE_APPROVAL'
MISSING = 'INSUFFICIENT_SAVED_EVIDENCE'
FORMULAS = dict(discard='exact_truncated - full_H_reference',
                PF='corrected - exact_truncated', full_PF='corrected - full_H_reference')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def has_axis(value, axis):
    return type(value) is dict and finite_number(value.get(axis))


def inspect_record(record, origin):
    cell = record['cell']; method = cell['method']
    require(method in ('B0', 'B0I', 'B1'), 'DETERMINISTIC_METHOD_REQUIRED')
    parts_expected = ('PF',) if method == 'B1' else ('discard', 'PF')
    decomposition = record.get('error_decomposition', {})
    parts = decomposition.get('signed_parts')
    signals = record.get('signals', {})
    independent = record.get('independent_forward_signals', {})
    out = []
    for axis in AXES:
        total_present = has_axis(decomposition.get('total_signed'), axis)
        components_present = (type(parts) is dict and set(parts) == set(parts_expected) and
                              all(has_axis(parts[k], axis) for k in parts_expected))
        endpoints_present = (has_axis(signals.get('corrected'), axis) and
                             has_axis(record.get('reference'), axis) and
                             (method == 'B1' or has_axis(signals.get('exact_truncated'), axis)))
        independent_endpoints = (has_axis(independent.get('corrected'), axis) and
                                 (method == 'B1' or has_axis(independent.get('exact_truncated'), axis)))
        classification = (SAVED if total_present and components_present and endpoints_present else
                          RECOVER if total_present and endpoints_present and parts is None else MISSING)
        missing = []
        if not total_present: missing.append('total_signed_axis')
        if not endpoints_present: missing.append('corrected/truncated/full_H_endpoint_axis')
        if not components_present: missing.append('signed_components_axis')
        if not independent_endpoints: missing.append('independent_forward_endpoint_axis')
        out.append(dict(cell_id=cell['id'], method=method, prefix=cell['prefix'], q=cell['q'],
            formula=cell['formula'], axis=axis, classification=classification,
            component_names=list(parts_expected), saved_signed_components_present=components_present,
            saved_total_present=total_present, saved_endpoints_present=endpoints_present,
            independent_forward_endpoints_present=independent_endpoints,
            separate_independent_signed_components_saved=False,
            independent_component_numerical_agreement_verified_this_audit=False,
            independent_path_relation=('ROTATED_PARENT_FORWARD_ENDPOINTS_SHARED_SAVED_SCALAR' if
                method == 'B0I' and independent_endpoints else 'SAVED_FORWARD_ENDPOINTS' if
                independent_endpoints else 'NOT_IN_THIS_RECORD'),
            closure_discrepancy_field_present=finite_number(decomposition.get('complex_closure_discrepancy')),
            closure_recomputed_this_audit=False, signed_bias_recomputed_this_audit=False,
            noncancellation_bias_evaluated_this_audit=False,
            rigorous_bias_certificate_present=False,
            saved_decomposition_certified_flag=decomposition.get('certified'),
            allowance_kind=record.get('allowance', {}).get('kind'),
            allowance_certified=record.get('allowance', {}).get('certified'),
            missing_or_pending_fields=missing, origin=origin,
            exact_truncated_means='SAVED_NUMERICAL_TRUNCATED_REFERENCE_NOT_CERTIFIED_EXACT',
            full_PF_discard_convention='NO_TRUNCATION_COMPONENT_BY_FIXED_METHOD' if method == 'B1' else None))
    return out


def inventory(records, registry):
    expected = {row['id']: row for row in registry['old'] + registry['additional']}
    require(len(expected) == 184 and len(registry['old']) == 32 and len(registry['additional']) == 152,
            'FIXED_184_DETERMINISTIC_REGISTRY')
    require(len(records) == 184, 'EXACT_184_RECORDS')
    seen = set(); axes = []; by_candidate = {}
    for record, origin in records:
        cell = record['cell']; cid = cell['id']
        require(cid in expected and cid not in seen, 'MISSING_DUPLICATE_OR_EXTRA_CANDIDATE')
        seen.add(cid)
        require(all(cell[k] == expected[cid][k] for k in ('id', 'method', 'prefix', 'q', 'formula')),
                'CELL_DEFINITION_MISMATCH')
        pair = inspect_record(record, origin)
        by_candidate[cid] = pair
        axes.extend(pair)
    require(seen == set(expected), 'MISSING_CANDIDATE')
    rows = []
    for cid in sorted(expected):
        pair = by_candidate[cid]
        state = (SAVED if all(r['classification'] == SAVED for r in pair) else
                 RECOVER if all(r['classification'] in (SAVED, RECOVER) for r in pair) else MISSING)
        rows.append(dict(cell=expected[cid], classification=state, axes=pair))
    return dict(schema='H8_BIAS_SAVED_EVIDENCE_AUDIT_V1', status='METADATA_AUDIT_COMPLETE_NO_NUMERICAL_ANALYSIS',
        candidates=184, axis_records=368,
        candidate_counts=dict(Counter(row['classification'] for row in rows)),
        axis_counts=dict(Counter(row['classification'] for row in axes)),
        by_method={m: dict(Counter(row['classification'] for row in rows if row['cell']['method'] == m))
                   for m in ('B0', 'B0I', 'B1')},
        rows=rows, real_signed_component_recalculations=0, real_noncancellation_evaluations=0,
        real_shot_recalculations=0, real_cost_reaccount=0, real_rankings=0,
        certified_bias_bound=None, full_184_empirical_minimum=None, new_scientific_calls=0,
        runtime_authorization=False, mandatory_STOP=True)
