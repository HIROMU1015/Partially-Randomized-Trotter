"""Integer seed metadata only. Never samples an RTE event or reads a cost value."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os

import h4_dp_contract as c


def geometry_fingerprint(domain, payload):
    # Geometry identity uses an exact encoder. All fields here are ASCII strings
    # or integers, for which its canonical bytes equal this ordinary JSON.
    value = dict(domain=domain, **payload)
    c.require(all(type(v) in (str, int) and (not isinstance(v, str) or v.isascii())
                  for v in value.values()), 'INTEGER_ASCII_GEOMETRY_SEED_METADATA')
    return c.digest(value)


def geometry_master(identity, replica):
    c.require(type(replica) is int and 0 <= replica < 32, 'OLD_GEOMETRY_REPLICA')
    return int(geometry_fingerprint('h4-trajectory-v1', dict(master_seed=20261006, **identity,
                                                          index=replica))[:16], 16)


def geometry_child(master, outer, short, occurrence, kind):
    return int(geometry_fingerprint('h4-step-occurrence-v1', dict(parent_trajectory_seed=master,
        outer_step=outer, short_step=short, occurrence=occurrence, draw_kind=kind))[:16], 16)


def conditional_hierarchy(master, q):
    return [master, int(c.digest(['H6_DBA_ORDER', master])[:16], 16)] + [
        int(c.digest(['H6_DBA_COMPONENT', master, i])[:16], 16) for i in range(q)]


def task_seed_values(tasks, conditional=False):
    values = set()
    for task in tasks:
        if task.get('seed') is None:
            continue
        fn = conditional_hierarchy if conditional else c.seed_hierarchy
        values.update(fn(int(task['seed']), task['cell']['q']))
    return values


def geometry_cases(plan, projection):
    templates = {x['template_id']: x for x in plan['templates']}
    cases = []
    c.require(len(projection['candidates']) == 1308 and len(projection['geometry_identities']) == 6,
              'SIX_GEOMETRIES_1308_REGISTERED')
    for row in projection['candidates']:
        template = templates[row['template_id']]
        identity = dict(projection['geometry_identities'][row['geometry']], template=row['template_fingerprint'])
        c.require(geometry_fingerprint('h4-candidate-v1', identity) == row['candidate_fingerprint'],
                  'OLD_GEOMETRY_CANDIDATE_IDENTITY')
        if template['method'] in ('B2', 'B3'):
            cases.append(dict(identity=identity, q=template['q'], r=template['r'], K=template['K'],
                              template_id=row['template_id']))
    c.require(len({(x['identity']['geometry'], x['template_id']) for x in cases}) == len(cases),
              'OLD_GEOMETRY_UNIQUE_CASES')
    return cases


def geometry_case_stream(case):
    """Conservative superset: includes every component slot up to frozen K.

    No Taylor orders or components are drawn. Unused keys are also excluded.
    """
    for replica in range(32):
        master = geometry_master(case['identity'], replica)
        yield master
        for outer in range(case['q']):
            for short in range(case['r']):
                yield geometry_child(master, outer, short, 0, 'order')
                for occurrence in range(case['K']+1):
                    yield geometry_child(master, outer, short, occurrence, 'component')


def _geometry_check(payload):
    name, cases, new_values = payload
    new_values = set(new_values)
    count, digest, collisions = 0, hashlib.sha256(), set()
    for case in cases:
        for seed in geometry_case_stream(case):
            count += 1
            digest.update(seed.to_bytes(8, 'big'))
            if seed in new_values:
                collisions.add(seed)
    return dict(geometry=name, candidates=len(cases), old_seed_slots_checked=count,
                ordered_seed_stream_sha256=digest.hexdigest(), collisions=sorted(collisions))


def check_geometry(cases, new_values):
    groups = {x['identity']['geometry'] for x in cases}
    payloads = [(name, [x for x in cases if x['identity']['geometry'] == name], tuple(sorted(new_values)))
                for name in sorted(groups)]
    workers = min(len(payloads), len(os.sched_getaffinity(0)))
    # Six natural independent geometry units; no artificial CPU/RAM/wall caps.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = list(pool.map(_geometry_check, payloads))
    return rows, workers


def confirmation_tasks(spec, signals, preparation_identities):
    old = set(c.read(spec['seed_exclusions'])['values'])
    occupied = set(old)
    tasks, new_values = [], set()
    for cell in spec['cells']:
        cid = cell['id']; signal = signals[cid]
        for replica in range(spec['n_fixed'][cid]):
            master = c.independent_seed('H4_PR_INDEPENDENT_CONFIRMATION_V1', spec['SOURCE'],
                                        spec['seed_plan_digest'], cid, replica)
            hierarchy = c.reserve_seed(occupied, master, cell['q'])
            new_values.update(hierarchy)
            tasks.append(dict(id=cid+'_'+str(replica), cell=cell, replica=replica, seed=master,
                step_seeds=hierarchy if cell['q'] == 1 else hierarchy[1:],
                preparation_hash=preparation_identities[str(cell['prefix'])]['preparation_hash'],
                saved_log_B=signal['log_B'], log_B_tolerance=signal['log_B_tolerance']))
    return tasks, new_values
