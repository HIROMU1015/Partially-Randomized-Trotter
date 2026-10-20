"""Receipt guards only. No real native provider exists in this namespace."""
from fractions import Fraction as F
from family import require
from trottertracks.algorithm_codesign.g6_f3_contract import Evidence, fingerprint

KEY_FIELDS = ('source', 'model', 'basis', 'labels', 'angle', 'precision', 'profile',
              'control_phase', 'actual_adjoint', 'scalar_policy', 'event')
PRICE_FIELDS = ('T', 'CX', '1Q', 'depth', 'workspace')

def ideal_event_id(event):
    return fingerprint({k:event[k] for k in ('word','child','ratio','phase_i_power','circuit_time_word')})

def key(spec):
    require(set(spec) == set(KEY_FIELDS) and all(spec[k] for k in KEY_FIELDS), 'COMPLETE_NATIVE_KEY_REQUIRED')
    return fingerprint(spec)  # method intentionally absent: same event/provider => same key.

def check(receipt, spec, *, require_full_support=False, real=False):
    require(not real, 'REAL_PROVIDER_UNSUPPORTED_CERTIFICATE_REQUIRED')
    d = receipt.read()
    require(d['scope'] == 'SYNTHETIC_ONLY' and d['key'] == key(spec), 'NATIVE_IDENTITY_MISMATCH')
    require(d['phase_metric'] == 'STRICT_OPERATOR_NORM' and d['actual_adjoint'] is True,
            'PROJECTIVE_OR_NOMINAL_INVERSE_UNSUPPORTED')
    require(d['coverage'] in ('OBSERVED_ONLY', 'FULL_SUPPORT') and
            (not require_full_support or d['coverage'] == 'FULL_SUPPORT'), 'COVERAGE_GAP')
    require(type(d['strict_error_upper']) is F and d['strict_error_upper'] >= 0, 'UNKNOWN_ERROR')
    require(set(d['price']) == set(PRICE_FIELDS) and all(type(v) is F and v >= 0
            for v in d['price'].values()), 'UNKNOWN_COST_IS_NOT_ZERO')
    require(d['cost_scope'] == 'SINGLE_EVENT_ADDITIVE' and
            set(d['classical']) == {'setup', 'online', 'storage', 'terminal'}, 'COST_LAYERS_REQUIRED')
    require(all(type(v) is F and v >= 0 for v in d['classical'].values()), 'UNKNOWN_CLASSICAL_COST')
    return d

def synthetic_receipt(spec, *, coverage='OBSERVED_ONLY'):
    return Evidence.create(dict(scope='SYNTHETIC_ONLY', key=key(spec), phase_metric='STRICT_OPERATOR_NORM',
        actual_adjoint=True, strict_error_upper=F(1, 10**9), coverage=coverage,
        price={k: F(i+1) for i,k in enumerate(PRICE_FIELDS)}, cost_scope='SINGLE_EVENT_ADDITIVE',
        classical={k: F(1) for k in ('setup', 'online', 'storage', 'terminal')},
        certificate='FAKE_PRICE_FOR_GUARD_TEST_ONLY_NO_NATIVE_ACQUISITION'))

def path_price(receipts, specs, *, prep_readout, full_support=False):
    require(len(receipts) == len(specs) and bool(receipts) and type(prep_readout) is dict,
            'PATH_PRICES_REQUIRED')
    rows = [check(r, s, require_full_support=full_support) for r, s in zip(receipts, specs)]
    require(set(prep_readout) == set(PRICE_FIELDS) and all(type(v) is F and v >= 0
            for v in prep_readout.values()), 'PREPARATION_READOUT_UNKNOWN')
    price = {k: prep_readout[k] + sum(r['price'][k] for r in rows) for k in ('T','CX','1Q','depth')}
    price['workspace'] = max(prep_readout['workspace'], *(r['price']['workspace'] for r in rows))
    return dict(scope='SYNTHETIC_ONLY', price=price, expected_cost_certified=False,
                policy='serial additive; no cancellation; workspace maximum; one prep/readout per path')
