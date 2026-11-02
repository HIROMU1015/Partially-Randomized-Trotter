"""Independent stdlib checker. Never imports the OLD family producer or NumPy."""
from fractions import Fraction as F

from .n3_old_family_contract_v1 import CAPS, decode, fingerprint, matrix, rational
from .n3_occupation_certificate_audit import Audit


def verify(instance, record):
    A, v, nu, U, c, delta = decode(instance)
    if set(record) != {'schema_version', 'input_id', 'instance_sha256', 'status', 'certificate',
                       'NORM_at_target', 'finite_failure_proves_infeasible', 'events', 'resources',
                       'new_OCC_proposals', 'new_U_c_bounds', 'ground_or_gap_calls', 'central_theme',
                       'mandatory_stop', 'next_stage_authorized'}:
        raise ValueError('RESULT_SCHEMA')
    if type(record['schema_version']) is not int or record['schema_version'] != 1 or record['input_id'] != instance['id'] or record['instance_sha256'] != fingerprint(instance):
        raise ValueError('CERTIFICATE_INPUT_IDENTITY')
    audit = Audit(); n = len(A); D = audit.q(c - U + delta)
    target = audit.q(delta * D); beta2 = audit.total(audit.q(x * x) for x in v)
    if type(record['NORM_at_target']) is not bool or record['NORM_at_target'] != (target >= beta2):
        raise ValueError('NORM_IDENTITY')
    if (any(type(record[key]) is not int or record[key] != 0 for key in ('new_OCC_proposals', 'new_U_c_bounds', 'ground_or_gap_calls'))
            or record['central_theme'] is not None or record['next_stage_authorized'] is not False
            or record['mandatory_stop'] is not True or record['finite_failure_proves_infeasible'] is not False):
        raise ValueError('SCOPE_OR_STOP')
    resources = record['resources']; counts = resources['counts']
    if set(counts) != {'numeric_eigh_calls', 'primal_PSD_probes', 'ray_probes', 'mixture_probes'}:
        raise ValueError('CALL_ACCOUNTING_SCHEMA')
    for key, value in counts.items():
        if type(value) is not int or not 0 <= value <= CAPS[key]: raise ValueError('CALL_ACCOUNTING_CAP')
    if (type(resources['arithmetic_operations']) is not int or type(resources['max_monitored_rational_bits']) is not int
            or not 0 <= resources['arithmetic_operations'] <= CAPS['arithmetic_operations']
            or not 0 <= resources['max_monitored_rational_bits'] <= CAPS['rational_bits']):
        raise ValueError('ARITHMETIC_ACCOUNTING_CAP')
    event_counts = {name: sum(e['stage'] == stage for e in record['events']) for name, stage in
                    [('numeric_eigh_calls', 'numeric'), ('primal_PSD_probes', 'primal'), ('ray_probes', 'ray')]}
    if any(counts[name] != value for name, value in event_counts.items()): raise ValueError('EVENT_COUNT_MISMATCH')
    cert, status = record['certificate'], record['status']
    if cert is None:
        if status != 'UNDETERMINED_FINITE_PROPOSALS': raise ValueError('MISSING_CERTIFICATE')
        return {'status': 'UNKNOWN_SCOPE_AND_ACCOUNTING_VALID_NO_INFEASIBILITY_PROOF',
                'input_id': instance['id'], 'PSD_checks': 0, 'mathematical_classification': 'UNDETERMINED'}
    if cert['kind'] == 'primal':
        if set(cert) != {'kind', 'kappa', 'mu', 'origin'}: raise ValueError('PRIMAL_SCHEMA')
        kappa, mu = rational(cert['kappa']), rational(cert['mu'])
        if not 0 <= kappa <= D: raise ValueError('PRIMAL_BOX')
        if mu != (kappa * U - target) / nu: raise ValueError('PRIMAL_MU_THRESHOLD')
        # Reconstruct from original h/v, not producer's affine matrix.
        M = [[audit.q(kappa * A[i][j] - v[i] * v[j] - (mu if i == j else 0)) for j in range(n)] for i in range(n)]
        audit.psd(M)
        if kappa * U - nu * mu > target or status != 'OLD_FAMILY_RECOVERS_SAVED_OCC_WIDTH':
            raise ValueError('PRIMAL_STATUS_OR_AFFINE')
        classification = 'OLD_FAMILY_RECOVERS_SAVED_OCC_WIDTH'
    elif cert['kind'] == 'dual':
        if set(cert) != {'kind', 'W', 'endpoint_traces', 'trace_W', 'origins'}: raise ValueError('DUAL_SCHEMA')
        W = matrix(cert['W'])
        if len(W) != n: raise ValueError('DUAL_DIMENSION')
        audit.psd(W)
        trace = audit.total(W[i][i] for i in range(n))
        if trace <= 0 or rational(cert['trace_W']) != trace: raise ValueError('ZERO_OR_BAD_DUAL_TRACE')
        endpoints = []
        for kappa in (F(0), D):
            mu = audit.q((kappa * U - target) / nu)
            M = [[audit.q(kappa * A[i][j] - v[i] * v[j] - (mu if i == j else 0)) for j in range(n)] for i in range(n)]
            endpoints.append(audit.total(audit.q(W[i][j] * M[j][i]) for i in range(n) for j in range(n)))
        if max(endpoints) >= 0 or [rational(x) for x in cert['endpoint_traces']] != endpoints:
            raise ValueError('NONSTRICT_OR_BAD_DUAL_ENDPOINTS')
        classification = 'CERTIFIED_NORM_AND_OLD_FAMILY_SEPARATION' if target < beta2 else 'OLD_FAMILY_SEPARATED_NORM_RECOVERS_TARGET'
        if status != classification: raise ValueError('DUAL_STATUS')
    else: raise ValueError('CERTIFICATE_KIND')
    return {'status': 'INDEPENDENT_EXACT_OLD_FAMILY_CERTIFICATE_PASS', 'input_id': instance['id'],
            'mathematical_classification': classification, 'PSD_checks': audit.psd_checks,
            'checker_conditions': audit.checks, 'max_checker_rational_bits': audit.max_bits,
            'new_eigensolver_calls': 0, 'new_certificate_proposals': 0}
