"""One authorized H8 s2 COST pilot; no runtime imports at contract load."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import h8_conditional_s2_sampling as sampling

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = 'references/h8_conditional_s2_evidence_v1.json'
SCOPE = 'H8_S2_CONDITIONAL_Q4_THEN_Q8_EIGHT_PAIRED_EACH_V1'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
PROFILE = '622ce4a8391002558b877240b48d831d9bb0a72aeabba2e4d73bfec64b9c9f12'
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'NUMBA_NUM_THREADS', 'QISKIT_NUM_PROCS', 'RAYON_NUM_THREADS')
SOURCES = ('scripts/h8_conditional_s2_sampling.py', 'scripts/h8_conditional_s2_contract.py',
           'scripts/h8_conditional_s2_adapter.py', 'scripts/run_h8_conditional_s2_pilot.py',
           'scripts/witness_h8_conditional_s2_pilot.py', 'scripts/h8_key_compile_parallel.py',
           'tests/test_h8_conditional_s2_pilot.py', EVIDENCE)
GIT = ['git', '--no-optional-locks', '-c', 'maintenance.auto=false', '-c', 'gc.auto=0']
CAPS = dict(compile=32, trajectory=16, occurrence=96, control_probe=8,
            primitive=0, reference_matvec=0, signal=0, normalization=0)
require = sampling.require


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(2**20), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    def pairs(rows):
        result = {}
        for k, v in rows:
            require(k not in result, 'DUPLICATE_JSON_KEY')
            result[k] = v
        return result
    def constant(value):
        raise ValueError('NONFINITE_JSON')
    def number(value):
        parsed = float(value)
        require(math.isfinite(parsed), 'NONFINITE_JSON')
        return parsed
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def save(path, value):
    raw = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode()+b'\n'
    with Path(path).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def evidence():
    return read(ROOT/EVIDENCE)


def tasks(q):
    require(q in (4, 8), 'CANDIDATE_Q')
    cid = f'B2_p7_q{q}_2nd_r4_K4'
    return [dict(id=cid+'_'+str(i), cell_id=cid, q=q, replica=i, seed=sampling.seed(cid, i), validate=i == 0)
            for i in range(8)]


def budget():
    e = evidence()
    prior = {k: e['old_ledger']['known'].get(k, 0)+e['old_ledger']['reserved'].get(k, 0) for k in CAPS}
    total = {k: prior[k]+CAPS[k] for k in CAPS}
    require(all(total[k] <= e['old_caps'][k] for k in CAPS), 'ORIGINAL_APPROVED_CAP_EXCEEDED')
    carry = e['old_ledger']['preserved_carry']
    return dict(prior_known=e['old_ledger']['known'], prior_held=e['old_ledger']['reserved'],
                carry=carry, new_maximum=CAPS, maximum_after=total,
                original_listed_lineage_after=dict(compile=carry['listed_known_plus_reserved_compile']+total['compile'],
                    trajectory=carry['listed_known_plus_reserved_trajectory']+total['trajectory']),
                approved_caps=e['old_caps'], refunds=0, reset=False, historical_unknown_actual=None,
                scope='original H8 resource-map lineage; later separately authorized stages remain separate')


def plan():
    e = evidence()
    all_tasks = tasks(4)+tasks(8)
    seeds = [t['seed'] for t in all_tasks]
    require(len(set(seeds)) == 16 and not set(seeds)&set(e['original_seeds']), 'SEED_COLLISION')
    streams = [v for task in all_tasks for v in
               (task['seed'], sampling.order_seed(task['seed']), *sampling.step_seeds(task['seed'], task['q']))]
    require(len(set(streams)) == len(streams) and not set(streams)&(
        set(e['original_seeds']) | set(e['original_component_step_seeds'])), 'RANDOM_STREAM_SEED_COLLISION')
    return dict(schema='H8_CONDITIONAL_S2_PLAN_V1', scope=SCOPE, approved=False, runtime_authorization=False,
        science=dict(system='linear H8 1.00 A STO-3G', saved_DF_rank=28, modes=16, sector=4900, T=.8,
                     prefix=7, r=4, K=4, q=[4, 8], stratum='s2', trajectories_each=8, axes=['cosine', 'sine']),
        tasks=all_tasks, unique_new_stream_seeds=len(streams),
        old_master_seeds_checked=len(e['original_seeds']), old_step_seeds_checked=len(e['original_component_step_seeds']),
        order='A_Q4_ALL_EIGHT_SAVED_VALIDATED_REAPED_THEN_B_Q8', compiler=COMPILER, profile=PROFILE,
        provider=PROVIDER, evidence_sha256=sha(ROOT/EVIDENCE), budget=budget(),
        sampling='positive Binomial(R,p2/(p0+p2)) then uniform subset; original _make_event and component law',
        native='pinned whole symmetric-directional controlled native wrapper; state preparation excluded',
        control='first trajectory per candidate: four logical probes; proved joint-branch fast path or original two-branch fallback; all use pinned native constructor',
        fingerprints='every whole wrapper original canonical hash; paired event digest checked before each axis',
        resources=dict(worker_policy='up to eight trajectories concurrently after fresh CPU/memory/I/O gate',
            threads=1, AS_RSS_CPU_wall_caps=None, inherited_OS_limits='unchanged',
            peak_RSS_estimate_per_worker=4*2**30, headroom_estimate=16*2**30,
            output_estimate_bytes=2*2**30, inodes_estimate=8000, pressure='observation only; actual inability remains STOP'),
        metrics=['RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size'], descriptive_n=8,
        old_n32_changed=False, s4_cost=None, population_mean_guarantee=False,
        one_shot=True, retry=False, resume=False, next_stage_authorized=False, mandatory_STOP=True)


def authorize(config, grant, *, worker_cpu=None):
    require(config.get('execution_sealed') is True and grant.get('approved') is True
        and grant.get('runtime_authorization') is True and grant.get('scope') == SCOPE
        and grant.get('SOURCE') == config.get('SOURCE') and len(config.get('SOURCE', '')) == 40
        and config.get('plan_sha256') == grant.get('plan_sha256') == digest(plan())
        and grant.get('config_sha256') == digest(config) and grant.get('one_shot') is True
        and grant.get('retry') is False and grant.get('resume') is False
        and grant.get('next_stage_authorized') is False
        and grant.get('approval_original_sha256') == config.get('approval_original_sha256'), 'SEALED_ONE_SHOT_USER_GRANT_REQUIRED')
    require(sha(config['approval_original_path']) == config['approval_original_sha256'], 'USER_APPROVAL_CHANGED')
    cpus = grant.get('worker_cpus', [])
    require(1 <= len(cpus) <= 8 and len(set(cpus)) == len(cpus)
            and all(type(x) is int and x >= 0 for x in cpus), 'OWN_WORKER_CPU_SCOPE')
    if worker_cpu is None:
        require(set(cpus).issubset(os.sched_getaffinity(0)), 'PARENT_ALLOWED_CPUS')
    else:
        require(worker_cpu in cpus and os.sched_getaffinity(0) == {worker_cpu}, 'OWN_WORKER_CPU')


def stat_record(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def verify(config, *, runtime=False, git=True):
    e = evidence()
    require(config['plan_sha256'] == digest(plan()) and config['provider_source'] == PROVIDER
            and config['expected_profile_digest'] == PROFILE, 'PLAN_PROFILE_CHANGED')
    require(digest(config['provider_hashes']) == e['provider_map_sha256'] and
            digest(config['runtime_file_hashes']) == e['runtime_map_sha256'], 'PINNED_PROVIDER_RUNTIME_MAPS')
    require(set(config['source_hashes']) == set(SOURCES), 'EXACT_SOURCE_CLOSURE')
    for name, h in config['source_hashes'].items():
        require(sha(ROOT/name) == h, 'SOURCE_CHANGED:'+name)
        if git:
            require(hashlib.sha256(subprocess.check_output(GIT+['show', config['SOURCE']+':'+name], cwd=ROOT)).hexdigest() == h,
                    'SOURCE_COMMIT_BYTES')
    if git:
        require(subprocess.check_output(GIT+['merge-base', '--is-ancestor', config['SOURCE'], 'HEAD'], cwd=ROOT) == b'', 'SOURCE_ANCESTRY')
    by = {r['logical_name']: r for r in config['files']}
    require(len(by) == len(config['files']) == len(e['inputs']), 'EXACT_INPUT_ROLES')
    for spec in e['inputs']:
        row = by[spec['logical_name']]
        require(all(row[k] == spec[k] for k in ('bytes', 'sha256'))
                and Path(row['path']).stat().st_size == spec['bytes'] and sha(row['path']) == spec['sha256'], 'INPUT_BYTE_SHA:'+spec['logical_name'])
    require(sha(config['python']) == config['python_sha256'], 'PYTHON_BYTES')
    if runtime:
        for name, h in config['provider_hashes'].items():
            require(sha(Path(config['provider_root'])/name) == h, 'PROVIDER_CHANGED:'+name)
        for name, h in config['runtime_file_hashes'].items():
            require(sha(name) == h, 'RUNTIME_CHANGED:'+name)
    elif 'verified_stats' in config:
        for name, s in config['verified_stats'].items():
            require(stat_record(name) == s, 'VERIFIED_RUNTIME_STAT_CHANGED:'+name)


class Ledger:
    def __init__(self, output):
        self.output = Path(output)
        self.used = dict.fromkeys(CAPS, 0)
        self.ids = set()

    def reserve(self, task):
        require(task in tasks(4)+tasks(8) and task['id'] not in self.ids, 'EXACT_NEW_UNIQUE_TRAJECTORY')
        demand = dict(compile=2, trajectory=1, occurrence=task['q'], control_probe=4 if task['validate'] else 0)
        require(all(self.used[k]+n <= CAPS[k] for k, n in demand.items()), 'CALL_CAP')
        new = {k: self.used[k]+demand.get(k, 0) for k in CAPS}
        save(self.output/('reservation_'+task['id']+'.json'), dict(task=task, demand=demand,
            cumulative_reserved=new, nonrefundable=True, old_carry_sha256=digest(budget())))
        self.used = new
        self.ids.add(task['id'])


def paired_result(task, result):
    rows = result['rows']
    sample = result['sampling']
    sampling.validate_orders(tuple(sample['orders']), task['q'], 4)
    require(len(rows) == 2 and [r['axis'] for r in rows] == ['cosine', 'sine']
            and sample['seed'] == task['seed'] and sample['original_finite_law'] == evidence()['laws'][task['cell_id']], 'PAIRED_SCOPE')
    for row in rows:
        require(row['cell_id'] == task['cell_id'] and row['seed'] == task['seed']
            and row['event_digest'] == result['event_digest'] and row['compiler_hash'] == COMPILER
            and row['control'] == 'symmetric_directional' and len(row['fingerprint']) == 64
            and all(type(row['metrics'][m]) is int and row['metrics'][m] >= 0 for m in plan()['metrics'])
            and row['metrics']['RZ'] > 0, 'PAIRED_METRICS_IDENTITY')
    expected = dict(compile=2, trajectory=1, occurrence=task['q'], control_probe=4 if task['validate'] else 0,
                    primitive=0, reference_matvec=0)
    require(result['actual_calls'] == expected, 'EXACT_ACTUAL_CALLS')


def summary(results):
    import statistics
    require(len(results) == 8, 'EIGHT_COMPLETE_TRAJECTORIES')
    out = {}
    for metric in plan()['metrics']:
        out[metric] = {}
        values = {axis: [r['rows'][i]['metrics'][metric] for r in results]
                  for i, axis in enumerate(('cosine', 'sine'))}
        values['paired_sum'] = [a+b for a, b in zip(values['cosine'], values['sine'])]
        for axis, x in values.items():
            out[metric][axis] = dict(n=8, mean=statistics.mean(x), sample_sd=statistics.stdev(x),
                                     SE=statistics.stdev(x)/math.sqrt(8), minimum=min(x), maximum=max(x))
    return dict(conditional_s2_descriptive=out, original_n32_mixed=False, s4_cost=None,
                population_mean_guarantee=False, selection_after_effect_unresolved=True)
