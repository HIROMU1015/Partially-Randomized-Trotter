"""Metadata-only fixed-delta panel. Importing this file never imports science."""
import cmath
import hashlib
import json
import math
from copy import deepcopy

KIND = 'H6_FIXED_DELTA_T_PANEL_V1'
NAMESPACE = 'H6_FIXED_DELTA_T_20261015_INDEPENDENT32_V1'
BASE = '34efc914b2f485ec694c7d6016025d5c76829be6'
PROVIDER = '87b6b9f46b7e8ba9abe132ee21478c114b6f46f7'
TIMES = ((.4, 2), (.8, 4), (1.6, 8))
EPSILONS = [.05, .01, .005, .001, .0005]
GIB = 2**30


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def cells(q):
    if type(q) is not int or q not in (2, 4, 8):
        raise ValueError('FIXED_Q')
    result = []
    for method, prefix, formula, r, K in (
            ('B0', 15, '2nd', None, None), ('B0I', 15, '2nd', None, None),
            ('B1', 19, '2nd', None, None), ('B1', 19, '4th', None, None),
            ('B2', 5, '2nd', 4, 2)):
        name = f'{method}_p{prefix}_q{q}_{formula}' + (f'_r{r}_K{K}' if r else '')
        result.append(dict(id=name, method=method, prefix=prefix, formula=formula,
            q=q, r=r, K=K, R=q*r if r else None))
    return result


def seed(T, cell, replica):
    if (T, cell['q']) not in TIMES or T == .8 or cell != cells(cell['q'])[-1]:
        raise ValueError('NEW_PR_SEED_SCOPE')
    if type(replica) is not int or not 0 <= replica < 32:
        raise ValueError('NEW_PR_REPLICA')
    return int.from_bytes(hashlib.sha256(
        f'{NAMESPACE}/T={T.hex()}/{cell["id"]}/confirmation/{replica}'.encode()).digest()[:8], 'big')


def registrations():
    rows = []
    # Heavy deterministic cases are isolated, followed by the bounded normal queue.
    for T, q in (TIMES[0], TIMES[2]):
        for c in cells(q):
            for replica in range(32 if c['method'] == 'B2' else 1):
                rows.append(dict(index=len(rows), task=dict(T=T, cell=c,
                    phase='confirmation' if c['method'] == 'B2' else 'deterministic',
                    replica=replica, seed=seed(T, c, replica) if c['method'] == 'B2' else None,
                    heavy=c['formula'] == '4th', validate=replica == 0)))
    return rows


def demand(rows):
    random = [x for x in rows if x['task']['cell']['method'] == 'B2']
    return dict(groups=len(rows), compile=2*len(rows), trajectory=len(random),
        sampling_calls=sum(x['task']['cell']['q'] for x in random),
        rte_events=sum(x['task']['cell']['R'] for x in random),
        control_probe=4*sum(x['task']['validate'] for x in rows))


def rotate_record(record, T, C, C_error=0., roundoff=1e-12):
    """Rotate only B0 paths; full-H reference and its uncertainty stay fixed.

    The inherited combined allowance is charged conservatively to each rotated
    path AND the unrotated reference. It remains empirical, never a bound.
    """
    if record['cell']['method'] != 'B0' or (T, record['cell']['q']) not in TIMES:
        raise ValueError('B0I_SCOPE')
    if any(not math.isfinite(x) for x in (T, C, C_error, roundoff)) or min(C_error, roundoff) < 0:
        raise ValueError('B0I_NONFINITE_OR_ERROR')
    out = deepcopy(record)
    out['cell'] = deepcopy(cells(record['cell']['q'])[1])
    phase = cmath.exp(-1j*T*C)
    angle = T*C
    c, s = abs(math.cos(angle)), abs(math.sin(angle))
    for key in ('corrected', 'exact_truncated'):
        z = record['signals'][key]
        z = complex(z['real'], z['imag'])
        z = phase*z
        out['signals'][key] = dict(real=z.real, imag=z.imag)
    u = record['allowance']
    for a in ('real', 'imag'):
        other = 'imag' if a == 'real' else 'real'
        out['allowance'][a] = c*u[a] + s*u[other] + u[a] + T*C_error + roundoff
    if out['reference'] != record['reference']:
        raise ValueError('REFERENCE_ROTATED')
    z = lambda v: complex(v['real'], v['imag'])
    corrected, truncated, reference = (z(out['signals']['corrected']),
        z(out['signals']['exact_truncated']), z(out['reference']))
    rec = lambda x: dict(real=x.real, imag=x.imag)
    parts = dict(discard=truncated-reference, PF=corrected-truncated)
    out['error_decomposition'] = dict(signed_parts={k:rec(v) for k,v in parts.items()},
        absolute_parts={k:abs(v) for k,v in parts.items()},
        total_signed=rec(corrected-reference), total_absolute=abs(corrected-reference),
        complex_closure_discrepancy=abs(sum(parts.values())-(corrected-reference)),
        absolute_additivity_claim=False, certified=False)
    # Old auxiliary oracle fields describe B0, never mislabel them as B0I.
    for key in ('precision_rows', 'independent_forward_signals', 'signal_discrepancies',
                'state_discrepancies', 'B_raw_corrected_difference'):
        out.pop(key, None)
    out['B0I_phase'] = dict(T=T, C=C, C_error=C_error, reference_rotated=False,
        full_fock_identity=True, local_agreement_is_bound=False)
    return out


def preparation():
    return dict(schema='h6_fixed_delta_preparation_v1', kind=KIND, base=BASE,
        provider_source=PROVIDER, times=[dict(T=T, q=q, delta=.2, local_rte_time=.05,
            R=q*4) for T,q in TIMES], cells=[dict(T=T, cell=c) for T,q in TIMES for c in cells(q)],
        epsilons=EPSILONS, sample_plan=dict(new_PR_independent_per_T=32,
            new_exploration=0, candidate_selection=False, old_exploration8_separate=True,
            no_population_guarantee=True), registrations=registrations(), demand=demand(registrations()),
        approved=False, runtime_authorization=False, allowed_cpus=[],
        absolute_launch_command=None, execution_plan_sealed=False,
        retry=False, resume=False, mandatory_stop=True, next_stage_authorized=False)
