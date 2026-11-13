"""Finite P2 plan, replay reconstruction and pilot-only paired statistics.

No sampler/compiler is invoked by plan creation or summary. Replay reconstructs
the stored component sequence with the original event function, never an RNG.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

from . import p1_efficient_zz as P
from . import compiled_residual as D

BASE = '5a67cea7705e80ff295306606611d774724ae2e2'
P1_RESULT = '54b64ef72bf507be25970f2d3362d3be091321c3'
P1_DIR = 'docs/research/d_p1_efficient_czz_20261113/'
COHORT = 'D_P2_DIRECT_PILOT_V1'
COMPILE_BUDGET = 20


def make_plan(root):
    root = Path(root)
    plan = P.make_plan(root, BASE)
    config = json.loads(P._blob(root, *P.B_CONFIG))
    result = json.loads(P._blob(root, P1_RESULT, P1_DIR + 'run02/RESULT.json'))
    checks = json.loads(P._blob(root, P1_RESULT, P1_DIR + 'run02_precheck/POST_RUN_CHECK.json'))
    seed = P.trial_seed_function(root)
    for cell, row in zip(plan['cells'], plan['menus']['H4']['candidates']):
        c = row['candidate']
        cell.update(cohort=COHORT, n=4, seeds=[[seed(c, COHORT, i, outer, 0)
            for outer in range(c['q'])] for i in range(4)])
        row['law_binding'] = config['rounds'][cell['m']]['law_binding']
    references = []
    for cell in plan['cells']:
        old = next(r for r in result['rows'] if r['cell_id'] == cell['id'] and r['index'] == 0)
        check = next(r for r in checks['events_PRIVATE_run02_equals_run01'] if r['cell_id'] == cell['id'] and r['index'] == 0)
        references.append(dict(cell_id=cell['id'], m=cell['m'], index=0,
            cohort=P.COHORT, seeds=old['seeds'], event_digest=old['event_digest'],
            event_file=str(Path(old['private_result'])/'events_PRIVATE.json'), event_sha256=check['sha256'],
            C=old['C'], S=old['S'], fingerprints=old['fingerprints'], compiled=old['compiled'],
            compiler=old['compiler'], old_price_seconds=old['price_seconds'], old_compile_log=old['compile_log'],
            old_rss_mb_peak=old['rss_mb_peak']))
    plan.update(schema='D_P2_DIRECT_PILOT_PLAN_V1', replay=references, compile_budget=COMPILE_BUDGET,
        trajectory_paths=8, replay_paths=2, phase_order=['REPLAY_ALL_PASS', 'FRESH'],
        not_in_scope=['additional paths/rounds', 'retry', 'forced rare', 'fit/extrapolation',
                      'qualified all-law E[C]', 'N/RMSE/G', 'eighth-order PF'])
    return plan


def restore_events(saved, components, distribution):
    """Restore ordered IDs; reject any differing metadata (including numeric bits)."""
    from trotterlib.rte import _make_event
    lookup = {c.component_id: i for i, c in enumerate(components)}
    events = []
    for value in saved:
        indices = [lookup[value['rotation_component_id']], *[lookup[k] for k in value['product_component_ids']]]
        event = _make_event(indices, components, distribution, distribution.orders.index(value['taylor_order']))
        if json.dumps(event.to_dict(), sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError('REPLAY_EVENT_METADATA_DIFFERS')
        events.append(event)
    return tuple(events)


def restore_requests(ctx, candidate, saved):
    """Same config/occurrence flags as the carrier, using saved events and seeds."""
    from trotterlib.rte import make_rte_config
    from trotterlib.df_partial_s2 import DFPartialS2StepRequest
    from trotterlib.df_rte_circuit import DFRTEEventSequenceCircuitRequest
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    prep = ctx.prep
    steps = ctx.h['core'].schedule(candidate['formula'], candidate['q'], 'ABS_TIME_PER_OUTER',
                                   candidate['R_outer'], T=candidate['T'])
    if len(steps) != len(saved['events']) or len(steps) != len(saved['seeds']) or saved['conditional']:
        raise ValueError('REPLAY_SCOPE_DIFFERS')
    requests, laws = [], []
    for step, batch, seed in zip(steps, saved['events'], saved['seeds']):
        tau = ctx.h['canonical_tau'](prep, step['signed_time'], step['rte_steps'])
        config, dist = make_rte_config(prep.rte_preparation.symbolic_tail,
            evolution_time=float(step['signed_time']), rte_steps=step['rte_steps'],
            truncation_tolerance=_explicit_cutoff_tolerance(tau, candidate['K']),
            finite_taylor_order=candidate['K'], seed=seed)
        events = restore_events(batch, prep.rte_preparation.symbolic_tail.components, dist)
        for event in events:
            prep.rte_preparation.resolve_event_basis_definitions(event)
        occurrence = DFRTEEventSequenceCircuitRequest(events=events,
            component_specs=prep.rte_preparation.component_specs, controlled=True,
            ancilla_qubit=prep.num_system_qubits, tail_id=config.tail_id, tail_hash=config.tail_hash,
            occurrence_rte_steps=config.rte_steps)
        requests.append(DFPartialS2StepRequest(preparation=prep, step_time=float(step['signed_time']),
            rte_config=config, rte_distribution=dist, rte_occurrence=occurrence, controlled=True,
            ancilla_qubit=prep.num_system_qubits, seed=seed, pf_label='2nd'))
        laws.append(dict(steps=step['rte_steps'], orders=list(dist.orders), probabilities=list(dist.order_probabilities)))
    if laws != saved['laws']:
        raise ValueError('REPLAY_LAWS_DIFFER')
    return requests, saved['seeds'], laws


def compare_replay(record, reference):
    keys = ('event_digest', 'seeds', 'C', 'S', 'fingerprints', 'compiled', 'compiler')
    checks = {k: record[k] == reference[k] for k in keys}
    if not all(checks.values()):
        raise ValueError('REPLAY_MISMATCH:' + ','.join(k for k, v in checks.items() if not v))
    return checks


def fresh_summary(plan, rows, source):
    """Only fresh complete normal paths; exact descriptive moments, no population CI."""
    out = []
    for cell, row in zip(plan['cells'], plan['menus']['H4']['candidates']):
        values = sorted([r for r in rows if r['phase'] == 'fresh' and r['cell_id'] == cell['id']], key=lambda r:r['index'])
        if [r['index'] for r in values] != list(range(4)) or any(r['cohort'] != COHORT or r['conditional'] for r in values):
            raise ValueError('FRESH_COHORT_INCOMPLETE_OR_WRONG')
        context = dict(candidate_id=cell['candidate_id'], role='DEVELOPMENT', compiler=plan['compiler']['settings_hash'],
            S_scope=D.FULL, C_scope=D.FULL, law_binding=row['law_binding'],
            structural_source=dict(execution_source=source, implementation='compiled_price_fast'),
            compiled_source=dict(execution_source=source, lowering=plan['lowering']['id']))
        records = [record for v in values for record in D.paired_residuals(v, context=context)]
        summary = D.cohort_summary(records)
        summary['m'] = cell['m']
        for column in ('C', 'S'):
            means = [Fraction(sum(v[column][a] for v in values), 4) for a in range(2)]
            summary[column] = dict(mean=means, sample_variance=[sum((v[column][a]-means[a])**2
                for v in values)/3 for a in range(2)], XY_covariance=sum(
                (v[column][0]-means[0])*(v[column][1]-means[1]) for v in values)/3)
        out.append(summary)
    return out


def planning(summary, structural_means):
    """Future design numbers only, conditional normal approximation with pilot s."""
    z = NormalDist().inv_cdf(1 - .05/26/2)
    out = []
    for item in summary:
        entry = dict(m=item['m'], z=z, alpha_per_axis=.05/26, confidence_claim=False,
            assumptions='PILOT_VARIANCE_PLUG_IN_NORMAL_APPROXIMATION_ONLY; LAW/RARE UNRESOLVED', axes={})
        for i, axis in enumerate(('X', 'Y')):
            saved = next(v for v in structural_means if v['m'] == item['m'] and v['axis'] == ('cosine', 'sine')[i])
            radius = max(saved['mean_S']-saved['numerical_enclosure'][0], saved['numerical_enclosure'][1]-saved['mean_S'])
            b = .012*saved['mean_S']-radius
            raw = math.ceil(z*z*float(item[axis]['sample_variance_D'])/(b*b))
            entry['axes'][axis] = dict(saved_mean_S=saved['mean_S'], MC_budget= b,
                n_plugin=raw, proposed_confirmation_n=max(4, raw),
                floor_reason='Fresh confirmation variance needs multiple paths; 4 is a design floor, not a guarantee')
        entry['proposed_n_paths'] = max(a['proposed_confirmation_n'] for a in entry['axes'].values())
        out.append(entry)
    return out


def json_safe(value):
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value
