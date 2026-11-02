"""Saved scalar/endpoint consistency only; no signal generator or cost model."""
import cmath
from fractions import Fraction
import math

AXES = ('real', 'imag')
SCOPE = 'H8_SAVED_SIGNED_COMPONENT_RECONSTRUCTION_AND_VALIDATION_V1'
TOLERANCES = {'rel_tol': 1e-12, 'abs_tol': 1e-12}
CELL_FIELDS = ('id', 'method', 'prefix', 'q', 'formula')


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def finite(value):
    require(type(value) in (float, int), 'FINITE_REAL_NUMBER_REQUIRED')
    try:
        require(math.isfinite(value), 'NONFINITE_NUMBER')
    except OverflowError as exc:
        raise ValueError('NONFINITE_NUMBER') from exc
    return value


def vector(value):
    require(type(value) is dict, 'COMPLEX_ENDPOINT_REQUIRED')
    return complex(finite(value['real']), finite(value['imag']))


def packed(value):
    return dict(real=finite(value.real), imag=finite(value.imag))


def scalar_value(receipt, *, old=False):
    """Check the saved exact-coefficient receipt; never recompute DF scalars."""
    require(type(receipt) is dict, 'SAVED_SCALAR_RECEIPT_REQUIRED')
    if not old:
        policy = receipt.get('policy')
        if policy == 'REUSED_SAVED_P21_FULL_FOCK_SCALAR':
            return scalar_value(receipt['receipt'], old=True)
        require(policy == 'EXACT_STORED_BINARY64_TAIL_IDENTITY' and
                receipt.get('input_operator_error_certificate') is False,
                'SCALAR_POLICY')
        numerator, denominator = receipt['numerator'], receipt['denominator']
    else:
        require(receipt.get('full_Fock_identity') is True and
                receipt.get('new_scientific_certificate') is False,
                'FULL_FOCK_SCALAR_CONVENTION')
        numerator = receipt['exact_stored_scalar_numerator']
        denominator = receipt['exact_stored_scalar_denominator']
    require(type(numerator) is str and type(denominator) is str and
            len(numerator) <= 4096 and len(denominator) <= 4096, 'SCALAR_RATIONAL_ENCODING')
    require(int(denominator) > 0, 'SCALAR_RATIONAL_DENOMINATOR')
    exact = Fraction(int(numerator), int(denominator))
    C, error = finite(receipt['C']), finite(receipt['scalar_error_upper'])
    require(error >= 0 and abs(Fraction(C) - exact) <= Fraction(error),
            'SAVED_SCALAR_ARITHMETIC')
    # Agreement with a saved arithmetic receipt is not an independent proof of
    # the coefficient extraction or of the underlying physical Hamiltonian.
    return C


def validate_domain(dataset, registry, permit, guard=lambda: None):
    """Produce exactly 368 axis rows, retaining failures and the first reason.

    Only the approved executor issues a saved-data permit. Tests use explicitly
    ARTIFICIAL data. No candidate is dropped when an endpoint is missing.
    """
    permit.require(dataset.get('kind'))
    require(len(registry['old']) == 32 and len(registry['additional']) == 152,
            'FIXED_REGISTRY_COUNTS')
    cells = registry['old'] + registry['additional']
    expected = {c['id']: c for c in cells}
    require(len(expected) == 184, 'FIXED_REGISTRY_UNIQUE')
    records, wrappers = {}, {}
    input_errors = []
    for item in dataset['records']:
        cid = item['cell']['id']
        if cid not in expected or cid in records:
            input_errors.append('DUPLICATE_OR_EXTRA_CANDIDATE:' + cid)
        else:
            records[cid] = item
    for item in dataset['old_B0I_wrappers']:
        cid = item['signal']['cell']['id']
        if cid in wrappers:
            input_errors.append('DUPLICATE_OLD_OVERLAY:' + cid)
        wrappers[cid] = item
    old_overlay_ids = {c['id'] for c in registry['old'] if c['method'] == 'B0I'}
    if set(wrappers) != old_overlay_ids:
        input_errors.append('EXACT_12_OLD_OVERLAY_RECEIPTS_REQUIRED')
    if set(records) != set(expected):
        input_errors.append('EXACT_184_CANDIDATES_REQUIRED')
    reference = dataset['full_reference']
    results = []
    for cell in cells:
        cid = cell['id']
        guard()
        row = records.get(cid)
        for axis in AXES:
            checks, failure = [], []

            def check(name, observed, wanted):
                observed, wanted = finite(observed), finite(wanted)
                error = finite(observed - wanted)
                ok = math.isclose(observed, wanted, **TOLERANCES)
                checks.append(dict(name=name, observed=observed, expected=wanted,
                                   signed_difference=error, passed=ok))
                if not ok:
                    failure.append(name)

            parts = None
            provenance = dict(origin=dataset.get('origins', {}).get(cid, {'kind': 'ARTIFICIAL'}))
            try:
                require(not input_errors, input_errors[0] if input_errors else '')
                require(row is not None, 'MISSING_CANDIDATE')
                require(dataset['kind'] == 'ARTIFICIAL' or cid in dataset.get('origins', {}), 'PER_CANDIDATE_LINEAGE')
                require(all(row['cell'][k] == cell[k] for k in CELL_FIELDS), 'CELL_BINDING')
                require('T' not in row['cell'] or row['cell']['T'] == .8, 'SAVED_TIME_BINDING')
                signal = row['signals']
                corrected = finite(signal['corrected'][axis])
                full = finite(row['reference'][axis])
                check('full_H_reference_unrotated', full, reference[axis])
                full_pf = cell['method'] == 'B1'
                total = finite(corrected - full)
                if full_pf:
                    parts = {'PF': total}
                else:
                    truncated = finite(signal['exact_truncated'][axis])
                    parts = {'discard': finite(truncated - full),
                             'PF': finite(corrected - truncated)}
                saved = row['error_decomposition']
                check('saved_total_matches_endpoints', saved['total_signed'][axis], total)
                check('discard_PF_total_closure', sum(parts.values()), total)
                reconstruct = cid in old_overlay_ids
                if reconstruct:
                    require('signed_parts' not in saved, 'OLD_OVERLAY_COMPONENTS_ALREADY_PRESENT')
                else:
                    require(set(saved['signed_parts']) == set(parts), 'SIGNED_COMPONENT_SET')
                    for name, value in parts.items():
                        check('saved_' + name + '_matches_endpoints', saved['signed_parts'][name][axis], value)
                    check('saved_component_closure',
                          sum(finite(saved['signed_parts'][k][axis]) for k in parts),
                          saved['total_signed'][axis])
                forward = row.get('independent_forward_signals')
                if cell['method'] == 'B0I':
                    parent_id = cid.replace('B0I_', 'B0_', 1)
                    parent = records[parent_id]
                    require(all(parent['cell'][k] == cell[k] for k in ('prefix', 'q', 'formula')),
                            'PARENT_B0_BINDING')
                    if reconstruct:
                        wrapper = wrappers[cid]
                        require(wrapper['signal'] == row and wrapper['prefix'] == cell['prefix'] and
                                wrapper['q'] == cell['q'], 'OLD_OVERLAY_RECEIPT_BINDING')
                        C = scalar_value(wrapper['scalar'], old=True)
                    else:
                        C = scalar_value(row['scalar_receipt'])
                    phase = cmath.exp(complex(0., finite(-.8*C)))
                    for name in ('corrected', 'exact_truncated'):
                        rotated = packed(vector(parent['signals'][name])*phase)
                        check('overlay_' + name + '_phase', signal[name][axis], rotated[axis])
                    check('overlay_reference_equals_parent_unrotated', full, parent['reference'][axis])
                    parent_forward = parent['independent_forward_signals']
                    # Old overlays saved no separate forward field. Inspect the
                    # saved parent path, without inventing a saved rotated path.
                    check('parent_forward_corrected', parent_forward['corrected'][axis],
                          parent['signals']['corrected'][axis])
                    check('parent_forward_truncated', parent_forward['exact_truncated'][axis],
                          parent['signals']['exact_truncated'][axis])
                    if reconstruct:
                        require(forward is None, 'UNEXPECTED_OLD_OVERLAY_FORWARD')
                        provenance['forward_relation'] = 'SAVED_PARENT_ONLY_SHARED_SCALAR_NO_SEPARATE_OVERLAY_FORWARD'
                    else:
                        require(type(forward) is dict, 'SAVED_FORWARD_REQUIRED')
                        for name in ('corrected', 'exact_truncated'):
                            rotated = packed(vector(parent_forward[name])*phase)
                            check('overlay_forward_' + name + '_phase', forward[name][axis], rotated[axis])
                        provenance['forward_relation'] = 'SAVED_ROTATED_PARENT_SHARED_SCALAR_AND_REFERENCE'
                else:
                    require(type(forward) is dict, 'SAVED_FORWARD_REQUIRED')
                    provenance['forward_relation'] = 'SAVED_FORWARD_SHARED_TRUNCATED_REFERENCE'
                if forward is not None:
                    check('independent_forward_corrected', forward['corrected'][axis], corrected)
                    if not full_pf:
                        check('independent_forward_truncated', forward['exact_truncated'][axis], truncated)
                provenance.update(components='RECONSTRUCTED_FROM_SAVED_ENDPOINTS' if reconstruct else 'SAVED_COMPONENTS_CHECKED',
                                  full_Fock_scalar_checked=cell['method'] == 'B0I',
                                  scalar_extraction_independently_recomputed=False,
                                  reference_or_scalar_independence_certified=False,
                                  separate_old_native_projection_available=reconstruct and
                                  (cell['prefix'], cell['q']) in dataset.get('old_native_projection_cases', []))
            except (KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
                failure.append(type(exc).__name__ + ':' + str(exc))
            results.append(dict(cell=cell, axis=axis,
                status='FAIL_OR_MISSING' if failure else 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY',
                first_failure=failure[0] if failure else None, failures=failure,
                checks=checks, signed_parts=parts,
                reconstructed=cid in old_overlay_ids, provenance=provenance,
                rigorous_bias_guarantee=None))
    failures = [r for r in results if r['first_failure']]
    return dict(schema='H8_SIGNED_COMPONENT_VALIDATION_RESULT_V1',
                status='STOP' if failures else 'PASS_EMPIRICAL_NUMERICAL_CONSISTENCY',
                first_stop=(failures[0]['cell']['id'] + ':' + failures[0]['axis'] + ':' +
                            failures[0]['first_failure']) if failures else None,
                candidate_count=184, axis_count=len(results), failed_axes=len(failures),
                tolerances=dict(TOLERANCES), rows=results,
                lineage_limitations=dataset['lineage_limitations'],
                rigorous_bias_guarantee=None, noncancellation_shots=0, PR_cost_comparisons=0,
                rankings=0, new_signal_compile_sampling_SCF_DF_state=0,
                next_stage_authorized=False, mandatory_STOP=True)
