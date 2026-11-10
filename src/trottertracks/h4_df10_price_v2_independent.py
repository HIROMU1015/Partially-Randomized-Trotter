"""Fixed independent-q extension: unchanged V1 accounting, frozen S1 b."""
import ast
import inspect
import json
from . import h4_df10_price_v1 as v1
from .h4_df10_price_v2 import summarize

COHORT = 'H4_DF10_V2_INDEPENDENT_Q_20261110'
S1 = 'cefc01622a3cd75dd16d30de7b9fb68274f3eddd'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
B = {'PF': 606.75, 'PR': 176.88888888888894}

# Only replace the two entry scope checks. Reuse every accounting statement.
_node = ast.parse(inspect.getsource(v1.predict)).body[0]
assert all(isinstance(n, ast.If) for n in _node.body[:2])
_node.body = _node.body[2:]
_scope = {'Accounting': v1.Accounting}
exec(compile(ast.fix_missing_locations(ast.Module(body=[_node], type_ignores=[])),
             'unchanged_v1_predict_body', 'exec'), _scope)


def predict(blocks, q, scalar, T, axis, event_batches=None):
    if (q, T) not in ((8, .8), (16, 1.6), (32, 3.2)) or axis not in ('cosine', 'sine'):
        raise ValueError('INDEPENDENT_Q_SCOPE')
    if event_batches is not None and (q not in (8, 16) or len(event_batches) != q
                                     or any(len(batch) != 4 for batch in event_batches)):
        raise ValueError('PR_FIXED_R4_SCOPE')
    return _scope['predict'](blocks, q, scalar, T, axis, event_batches)


def validate_cache(record, fingerprint):
    if (record.get('compiler') != COMPILER or record.get('fingerprint') != fingerprint
            or type(record.get('metrics', {}).get('rz_count')) is not int
            or record['metrics']['rz_count'] < 0):
        raise ValueError('CACHE_BINDING')
    return record


def residual_rows(predictions, trials):
    table = {(t['candidate']['id'], t['index']): t for t in trials}
    rows = []
    for p in predictions:
        t = table[p['candidate_id'], p['path_index']]
        actual = t['prices'][p['axis']]['rz_count']
        for model in ('M0', 'Mq'):
            pred = p['structural_rz'] + (B[p['method']]*p['q'] if model == 'Mq' else 0)
            e = actual-pred
            meta = {k: p[k] for k in ('method', 'candidate_id', 'q', 'T', 'path_index', 'axis')}
            rows.append(dict(**meta, cohort=COHORT, role='INDEPENDENT_Q', model=model,
                compiled_rz=actual, prediction_rz=pred, error_rz=e,
                absolute_error_rz=abs(e), relative_error=e/actual if actual else None,
                diagnostic='NEGATIVE_PRICE' if pred < 0 else 'FINITE'))
    return rows
