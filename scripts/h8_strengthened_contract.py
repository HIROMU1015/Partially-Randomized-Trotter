"""H8 Phase A only: frozen metadata, schedules and unapproved resource plan.

No scientific imports. The PF iterator is an existing stdlib implementation.
The saved proposal and all imported code must be bound by the execution SOURCE.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
SCOPE = 'H8_STRENGTHENED_PHASE_A_SIGNAL_V1'
PROPOSAL = 'artifacts/resource_applicability/track_a_h8_strengthened_readiness/2026-10-16/execution_plan_UNAPPROVED_v1.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def proposal():
    return json.loads((ROOT/PROPOSAL).read_text())


@lru_cache(maxsize=1)
def iterator():
    return runpy.run_path(str(ROOT/'src/trotterlib/pf_decomposition.py'))['iter_pf_steps']


def schedule(cell):
    """Identical ordinary/directional ordering to the fixed provider schedule."""
    if (cell['method'] != 'B0' or type(cell['prefix']) is not int or
            not 8 <= cell['prefix'] <= 21 or cell['q'] not in (1, 2, 4, 8) or
            cell['formula'] not in ('2nd', '4th')):
        raise ValueError('H8_FIXED_DETERMINISTIC_CELL')
    count, delta = cell['prefix']+1, .8/cell['q']
    w = 1/(2-2**(1/3))
    weights = [1.] if cell['formula'] == '2nd' else [1-2*w, w]
    ordinary = [(i, delta*f) for i, f in iterator()(count, weights)]
    pieces = [1.] if cell['formula'] == '2nd' else [w, weights[0], w]
    directional = [(i, delta*f/2, mode) for f in pieces
                   for mode, indices in (('UNCONTROLLED', range(count)),
                                         ('DIRECTIONAL', reversed(range(count))))
                   for i in indices]
    times = set(ordinary)
    for i, t, mode in directional:
        times.add((i, t))
        if mode == 'DIRECTIONAL':
            times.add((i, -t))
    return dict(ordinary_one_outer_step=ordinary,
                directional_one_outer_step=directional,
                unique_primitive_times=sorted(times), scalar_total_time=.8)


def cells():
    return [dict(c, r=None, R=None, K=None) for c in
            proposal()['registrations']['additional_deterministic'] if c['method'] == 'B0']


def tasks():
    cs = cells()
    times = {(i, t) for c in cs for i, t in schedule(c)['unique_primitive_times']}
    primitives = [dict(id='primitive_'+str(i), kind='primitive', block=i,
                       times=sorted(t for j, t in times if j == i),
                       counts=dict(primitive=3*sum(j == i for j, t in times),
                                   reference_matvec=1000000))
                  for i in sorted({i for i, t in times})]
    prefixes = [dict(id='prefix_'+str(p), kind='signals', prefix=p,
                     cells=[c for c in cs if c['prefix'] == p],
                     counts=dict(B0_signals=sum(c['prefix'] == p for c in cs),
                                 B0I_analytic_overlays=sum(c['prefix'] == p for c in cs),
                                 truncated_reference_pairs=1,reference_matvec=2000000,
                                 tail_scalar=0 if p == 21 else 1))
                for p in sorted({c['prefix'] for c in cs})]
    return primitives, prefixes


def plan():
    original = proposal()
    numerical = deepcopy(original['numerical_policy'])
    numerical['scope'] = ('binary64 independent occupation CSR/expm action, split-time/reference residual, '
                          'spin-sector spectral/native and Horner/forward agreement; empirical, not certified')
    primitives, prefixes = tasks()
    resource = deepcopy(original['phase_A']['resources'])
    resource.update(max_workers=16, threads=1, AS_bytes=resource['driver_AS_bytes'],
                    RSS_bytes=resource['driver_RSS_bytes'], CPU_seconds=48*3600,
                    minimum_free_inodes=40000, log_bytes=128*2**20,
                    observer_CPU_seconds=1800, max_processes=18)
    caps = dict(snapshot_load=1, full_preparation_decode=1,
                B0_signals=76, B0I_analytic_overlays=76, truncated_reference_pairs=13,
                tail_scalar=12, primitive=20000, reference_matvec=50000000,
                reference_matvec_per_action=100000,
                compile=0, native_full_wrapper_build=0, trajectory=0, occurrence=0,
                control_probe=0, SCF=0, DF=0, state_solver=0,
                law=0, cost_fit=0, cost_reaccount=0)
    required = sum(t['counts']['primitive'] for t in primitives)
    if required > caps['primitive']:
        raise ValueError('PRIMITIVE_COVERAGE_OVER_BUDGET')
    return dict(schema='H8_STRENGTHENED_PHASE_A_EXECUTION_PLAN_V1', scope=SCOPE,
        proposal_sha256=digest(original), registrations=original['registrations'],
        numerical=numerical, call_caps=caps, resources=resource,
        primitive_tasks=primitives, signal_tasks=prefixes,
        required_primitive_probe_actions=required,
        schedule_sha256=digest([{ 'id':c['id'], 'schedule':schedule(c)} for c in cells()]),
        budgets_are_irrevocable_reservations=True,
        target=original['target'], native_instruction_cap=1000000,
        no_new_full_H_reference=True, full_H_reference_rotated=False,
        scalar_phase_once=True, separate_cost_phases=True,
        approved=False, runtime_authorization=False, SOURCE=None, allowed_cpus=[],
        absolute_launch_command=None, one_shot=True, retry=False, resume=False,
        mandatory_stop=True, next_stage_authorized=False)
