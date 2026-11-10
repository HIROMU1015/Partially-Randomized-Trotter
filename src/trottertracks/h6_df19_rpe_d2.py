"""Bounded D2 action adapter; saved inputs are never read on import.

Uses matrix exponentials, existing finite Taylor/normalizer kernels and D1
scalar geometry. No eigenproblem, sampler, circuit or price evaluator.
"""
from copy import deepcopy
import math
from types import SimpleNamespace

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from trotterlib import h46_m5_plan as schedule
from trotterlib.rte import finite_taylor_operator, finite_rte_distribution
from . import h4_df10_rpe_connection as common
from . import h4_df10_rpe_d1 as scalar
from .h6_df19_rpe_connection import BINDING

PLAN_COMMIT = '47b3157ecd8c651c326b1bcfa3e8037534d289e1'
PLAN_PATH = 'docs/research/h6_df19_rpe_d2_preparation_20261110/PLAN.json'
IDS = ('dec32b474f3fc33ddfab', 'f2b156ebaeb00c2c033b',
       '7791db0dc2fab94ce734', '9fe878b5ed57560be583',
       'e7200ae9e16ba879a78d', '307350517bc4a1b4f9ff',
       '3f813a46e17461bfe484', '72255c43facdfb8b4bee')


def validate_plan(plan):
    if tuple(r['candidate']['id'] for r in plan['rows']) != IDS:
        raise ValueError('D2 requires the eight published candidates in order')
    if plan['scientific_binding'] != BINDING:
        raise ValueError('D2 scientific input binding differs')
    config = plan['configuration_reuse']
    if (config['base_time'], config['beta_total'], config['beta_stat'],
            config['alpha_total_per_method'], config['horizon']['maximum_round_index_M']) != (.1,.4,.2,.05,12):
        raise ValueError('D2 fixed policy differs')
    for index, row in enumerate(plan['rows']):
        m = (0,1,2,5)[index//2]; c = row['candidate']
        method = 'PR' if index%2 else 'FULL'
        q = 2 if m == 5 else 1
        expected = (19,None,None) if method == 'FULL' else ((7,16,2) if m == 5 else (5,32,2))
        if (row['m'] != m or row['q_RPE'] != 1<<m or row['q_PF'] != q
                or c['system'] != 'H6' or c['method'] != method
                or c['T'] != math.ldexp(.1,m) or c['q'] != q
                or c['delta'] != c['T']/q or c['formula'] != '4th(m5_best)'
                or tuple(c[k] for k in ('prefix','R_outer','K')) != expected):
            raise ValueError('D2 time/formula/partition differs from finite menu')


def make_context(ham, sector, state, saved_tails):
    from h46_new2_core import occupation_one_body
    matrices = [occupation_one_body(ham.one_body, sector)]
    for weight, g in zip(ham.lambdas, ham.g_matrices):
        one = occupation_one_body(g, sector)
        matrices.append(weight*(one@one))
    eye = np.eye(len(state), dtype=complex)
    tails = {}
    for key, value in saved_tails.items():
        p = int(key)
        tails[p] = dict(value, matrix=sum(matrices[p+1:], np.zeros_like(eye))-value['scalar']*eye)
    full = sum(matrices, np.zeros_like(eye))+ham.constant*eye
    return SimpleNamespace(ham=ham, state=state, matrices=matrices, tails=tails,
                           full=full, half_steps={})


def reference_pair(ctx, T):
    a = expm_multiply(-1j*T*ctx.full, ctx.state)
    b = expm(-1j*T*ctx.full)@ctx.state
    za, zb = complex(np.vdot(ctx.state,a)), complex(np.vdot(ctx.state,b))
    error = abs(za-zb)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or error > 1e-10:
        raise ValueError('REFERENCE_NUMERICAL_GATE')
    return dict(reference=zrecord(za), independent_reference=zrecord(zb),
                signal_difference=error, reference_state_actions=2,
                kind='NEW_D2_TWO_EXPONENTIAL_PATHS', certified=False)


def half_step_keys(rows):
    return sorted({(block, c['T']*w/c['q']/2)
        for row in rows for c in (row['candidate'],)
        for w in schedule.weights(c['formula']) for block in range(c['prefix']+1)})


def half_step(ctx, key):
    block, time = key
    value = expm(-1j*time*ctx.matrices[block])
    if not np.isfinite(value).all():
        raise ValueError('NONFINITE_HALF_STEP')
    return key, value


def zrecord(z):
    return dict(real=float(z.real), imag=float(z.imag))


def evolve(ctx, candidate, reference):
    """Same signed block sequence as evolve_batch; only its backend changes."""
    c = candidate; pr = c['method'] == 'PR'
    if c['method'] not in ('FULL','PR'):
        raise ValueError('D2 supports Full PF or PR only')
    weights = schedule.weights(c['formula'])
    counts = schedule.allocation(weights,c['R_outer']) if pr else (0,)*len(weights)
    v = np.repeat(ctx.state[:,None],3 if pr else 1,axis=1)
    maximum = 1.; log_B = 0.; stages = []
    for w, r in zip(weights,counts):
        t = c['T']*w/c['q']
        if pr:
            tail = ctx.tails[c['prefix']]; lam = tail['lambda_R']
            step = finite_taylor_operator(tail['matrix']/lam,lam*t/r,c['K'])
            b = finite_rte_distribution(lam*t/r,c['K']).exact_finite_distribution
            exact = expm(-1j*t*tail['matrix'])
            log_B += c['q']*r*math.log(b)
            stages.append((t,r,step,b,exact))
        else:
            stages.append((t,0,None,None,None))
    def checked():
        nonlocal maximum
        if not np.isfinite(v).all():
            raise ValueError('NONFINITE_PROPAGATION')
        maximum = max(maximum,float(np.max(np.linalg.norm(v,axis=0))))
    for _ in range(c['q']):
        for t,r,step,b,exact in stages:
            for block in range(c['prefix']+1):
                v = ctx.half_steps[(block,t/2)]@v; checked()
            if pr:
                v[:,0] = exact@v[:,0]
                for _ in range(r):
                    v[:,1:] = step@v[:,1:]
                    v[:,2] /= b
                    checked()
            for block in reversed(range(c['prefix']+1)):
                v = ctx.half_steps[(block,t/2)]@v; checked()
    scalar_phase = ctx.ham.constant+(ctx.tails[c['prefix']]['scalar'] if pr else 0.)
    v *= np.exp(-1j*c['T']*scalar_phase)
    checked()
    norm_error = abs(float(np.linalg.norm(v[:,0]))-1.)
    closure = float(np.linalg.norm(v[:,2]-math.exp(-log_B)*v[:,1])) if pr else 0.
    if norm_error > 1e-9 or closure > 1e-9:
        raise ValueError('EXACT_NORM_OR_NORMALIZATION_GATE')
    exact = complex(np.vdot(ctx.state,v[:,0]))
    corr = complex(np.vdot(ctx.state,v[:,1])) if pr else exact
    raw = complex(np.vdot(ctx.state,v[:,2])) if pr else exact
    ref = complex(reference['real'],reference['imag'])
    return dict(candidate=deepcopy(c), signal=dict(reference=deepcopy(reference),
        corrected=zrecord(corr),raw=zrecord(raw),exact_tail=zrecord(exact),
        total_signed_bias=zrecord(corr-ref),outer_PF_or_discard_bias=zrecord(exact-ref),
        finite_RTE_bias=zrecord(corr-exact),log_B=log_B,
        allowance_evidence=dict(maximum_intermediate_norm=maximum,
            raw_corrected_vector_closure=closure,exact_tail_norm_error=norm_error)),
        origin='NEW_D2_FINITE_MEAN',phase_alignment=False,
        signed_weights=list(weights),allocation=list(counts),
        event_count=c['q']*c['R_outer'] if pr else 0)


def analyze(record, planned_row, plan, *, state_energy, state_residual, reference_evidence):
    c = record['candidate']; sig = record['signal']; ev = sig['allowance_evidence']
    if c != planned_row['candidate']:
        raise ValueError('Signal candidate differs from the fixed planned row')
    stages = c['q']*2*(c['prefix']+1)*len(schedule.weights(c['formula']))
    roundoff = 64.*2**-52*(1+ev['maximum_intermediate_norm'])*(1+stages)
    old = plan['saved_reference_T04']['preserved_empirical_allowance'] if c['T']==.4 else {}
    pair_error = reference_evidence.get('signal_difference')
    if c['T'] != .4 and (pair_error is None or not math.isfinite(pair_error) or pair_error < 0):
        raise ValueError('New reference requires its numerical pair diagnostic')
    u = {a:max(1e-9,10*ev['raw_corrected_vector_closure']+roundoff,old.get(a,0.))
         +(pair_error if pair_error is not None else 0.) for a in ('real','imag')}
    margin = max(1e-12,64.*2**-52*(1+abs(sig['log_B']))*(1+stages),old.get('log_B_margin',0.))
    point = dict(candidate=deepcopy(c),binding=deepcopy(plan['saved_tail_context']['binding']),
        compiler=plan['compiler_expectation']['hash'],prices=[],saved_signal=sig,
        saved_allowance=dict(u=max(u.values()),log_B_margin=margin),
        saved_finite_time_endpoint=dict(axes={a:dict(allowance=u[a]) for a in u}),
        origin=record['origin'])
    row = common.bind_saved_round(point,target_id=plan['rpe_task_id'],m=planned_row['m'],
        base_time=plan['configuration_reuse']['base_time'],expected_system='H6',
        expected_rank=19,expected_binding=BINDING)
    result = scalar.analyze_bound_point(point,row,plan['configuration_reuse'],
        state_energy=state_energy,state_residual=state_residual)
    for lane in ('ideal_numerical_pilot','actual_implementation'):
        result[lane].update(stage='D2',energy_target='saved_DF_H_ground_energy',
            energy_ha=plan['configuration_reuse']['energy_precision_ha'],
            reference_provenance='D2 full-H reference action or preserved T=.4 reference; saved numerical state, conditional ground identification',
            error_provenance='D2 total signed bias + fixed empirical allowance + reference offset + T*saved residual')
    return dict(result,stage='D2',origin=record['origin'],signal=sig,
        allowance=dict(u=u,log_B_margin=margin,roundoff=roundoff,stages=stages,
            reference_pair_difference=pair_error,preserved_reference_allowance=old,
            certified=False),signed_weights=record['signed_weights'],allocation=record['allocation'],
        event_count=record['event_count'],compiled_price=None,G=None,
        price_coverage='NO_EXACT_CANDIDATE_IN_INSPECTED_BASELINE_TABLES')
