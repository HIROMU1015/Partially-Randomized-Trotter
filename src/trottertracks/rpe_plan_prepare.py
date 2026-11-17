"""One-command metadata preparation around the existing optimizer entry.

The scientific computation checkout and the previous remote delivery remain
read-only. Only this preparation's local cache and new output are written.
"""
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace
import zlib

from . import rpe_method_optimizer as O
from .rpe_plan_history_cache import HistoryCache
from .rpe_plan_readiness import build_readiness, run_server2_dry_checks, finish_readiness

REFERENCE = '33928bf80cca18ce3f1fa3ddde88d27d7e2b7e5b'
REFERENCE_PATH = 'docs/research/rpe_distributed_h4_plan_20261117'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(O.jsonable(value), ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root)


class Stages:
    def __init__(self, output):
        self.output = Path(output)
        self.start = time.monotonic()
        self.rows = []

    def event(self, stage, event, **fields):
        row = dict(stage=stage, event=event, epoch=time.time(),
                   since_command_seconds=time.monotonic() - self.start, **fields)
        self.rows.append(row)
        with (self.output / 'PREPARATION_STAGES.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)

    @contextmanager
    def phase(self, name):
        started = time.monotonic()
        self.event(name, 'START')
        try:
            yield
        except BaseException:
            self.event(name, 'END', status='FAILED', seconds=time.monotonic() - started)
            raise
        self.event(name, 'END', status='OK', seconds=time.monotonic() - started)


class ContextCache:
    """Local-only binary cache, tied to actual planning code and fixed inputs."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.memo_path = self.directory / 'FILE_FINGERPRINTS.json'
        self.memo = json.loads(self.memo_path.read_text()) if self.memo_path.exists() else {}
        self.hashed = 0

    def fingerprint(self, path):
        path = Path(path).resolve()
        try:
            stat = path.stat()
        except FileNotFoundError:
            return dict(path=str(path), missing=True)
        signature = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
        key = str(path)
        prior = self.memo.get(key, {})
        if prior.get('signature') != signature:
            prior = dict(signature=signature, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            self.memo[key] = prior
            self.hashed += 1
        return dict(path=key, sha256=prior['sha256'])

    def inputs(self, root, problem, spec, methods):
        files, views = {}, {}
        def visit(value):
            if isinstance(value, list):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if key in ('registry_file', 'configurations_file', 'kappa_points_file') and isinstance(item, str):
                        file = Path(item)
                        files[str(file)] = self.fingerprint(file if file.is_absolute() else root / file)
                    elif key in ('snapshot', 'receipt') and isinstance(item, dict) and 'path' in item and 'sha256' in item:
                        observed = self.fingerprint(item['path'])
                        if observed.get('sha256') != item['sha256']:
                            raise ValueError('Fixed scientific input changed: ' + item['path'])
                        files[item['path']] = observed
                    elif key == 'view' and isinstance(item, str):
                        view = Path(item).resolve()
                        views[str(view)] = git(view, 'rev-parse', 'HEAD').decode().strip()
                        # These are the declared metadata providers, not an all-project search.
                        for area in ('scripts', 'src/trottertracks'):
                            for file in (view / area).rglob('*.py'):
                                files[str(file)] = self.fingerprint(file)
                    elif key == 'interpreter' and isinstance(item, str):
                        files[item] = self.fingerprint(item)
                    visit(item)
        visit(problem)
        visit({k: spec[k] for k in ('price_rules_overrides', 'routes_override') if k in spec})
        for name in ('rpe_method_optimizer.py', 'rpe_candidate_preparation.py', 'pf_optimizer_connection.py',
                     'saved_rpe_cost_catalog.py', 'pr_price_rules.py', 'rpe_plan_prepare.py'):
            path = Path(__file__).with_name(name)
            files[str(path)] = self.fingerprint(path)
        for name in ('run_rpe_optimizer_config_generator.py', 'run_rpe_optimizer_price_quotes.py'):
            path = root / 'scripts' / name
            files[str(path)] = self.fingerprint(path)
        source = dict(schema=1, problem=problem, methods=methods, files=files, views=views)
        digest = hashlib.sha256(json.dumps(source, sort_keys=True, default=str).encode()).hexdigest()
        write(self.memo_path, self.memo)
        return digest, source

    def prepare(self, root, problem, spec, methods):
        started = time.monotonic()
        key, provenance = self.inputs(root, problem, spec, methods)
        path = self.directory / (key + '.pickle.z')
        hit = path.exists()
        if hit:
            # Only read caches generated in this user's local preparation state.
            context = pickle.loads(zlib.decompress(path.read_bytes()))
        else:
            context = O.prepare(root, problem, methods=methods,
                preparation_dir=self.directory / 'metadata' / key,
                generate_configurations=not (spec.get('candidate_completion') or {}).get('bootstrap_saved_only', False))
            stable = {k: v for k, v in context.items() if k not in ('reader', 'saved', 'generate')}
            raw = zlib.compress(pickle.dumps(stable, protocol=5), 1)
            temporary = path.with_suffix('.tmp')
            temporary.write_bytes(raw)
            temporary.replace(path)
            write(self.directory / (key + '.source.json'), provenance)
        reader = O.Reader(root)
        context['reader'] = reader
        prices = {c['configuration_id']: c['prices'] for c in context['candidates'] if c['prices'] is not None}
        context['saved'] = O.SavedBackend(context['catalog']['candidates'], prices)
        context['generate'] = O.pinned_function(reader, problem['generator'])
        return context, dict(hit=hit, key=key, local_cache=str(path), newly_hashed_files=self.hashed,
                             seconds=time.monotonic() - started)


def _known_minima(context, methods):
    minimum = {}
    for method in methods:
        values = [O.frac(p['conditional_cost']) for p in context['catalog']['candidates']
                  if p['method'] == method and p['status'] == 'FEASIBLE' and p.get('conditional_cost') is not None]
        best = methods[method].get('best')
        if best:
            values.append(O.frac(best['cost']))
        minimum[method] = min(values) if values else None
    return minimum


def host_profile_inputs(prepared, output, code_root, descriptor=None):
    output, code_root = Path(output), Path(code_root)
    config = Path(descriptor) if descriptor else code_root / 'docs/research/rpe_q3_host_eligibility/EVIDENCE.json'
    config = config.resolve()
    data = O.load_json(config) if config.is_file() else {}
    profiles = {}
    for path in (Path(prepared['output']) / 'profiles').glob('*.json'):
        profile = O.load_json(path)
        profiles[profile['host_id']] = profile
    for host, value in data.get('additional_profiles', {}).items():
        if host in profiles:
            raise ValueError('Additional host profile would replace an existing host: ' + host)
        if isinstance(value, str):
            path = Path(value)
            profile = O.load_json(path if path.is_absolute() else config.parent / path)
        else:
            profile = deepcopy(value)
        if profile.get('host_id') != host:
            raise ValueError('Additional profile host identity differs: ' + host)
        profiles[host] = profile
    for host, rules in data.get('profile_overrides', {}).items():
        if host in profiles:
            profiles[host]['route_profiles'] = deepcopy(rules)
    for host, profile in profiles.items():
        write(output / 'worker_profiles' / (host + '.json'), profile)
    return config, data, profiles


def host_allocation_table(policy, admission):
    """Keep a request's proved capability separate from new-job admission.

    This is a configuration snapshot, not an assertion that a worker is online.
    The controller rechecks admission, capacity and fingerprints at each claim.
    """
    rows = []
    for job, record in policy['jobs'].items():
        hosts = {}
        for host, cell in record['hosts'].items():
            rule = admission.get('hosts', {}).get(host, {})
            accepting = rule.get('accepting', True)
            runtime = cell.get('current_host_fingerprint', {}).get('runtime', {})
            hosts[host] = dict(can_execute_this_request=cell['status'] == 'PASS',
                readiness_status=cell['status'], reasons=cell.get('reasons', []),
                accepting_new_jobs=accepting, optional=rule.get('optional', False),
                max_slots=rule.get('max_slots'),
                admission_observation='CONFIGURED_NOT_A_LIVE_WORKER_PROBE',
                executable=cell.get('execution_profile', {}).get('interpreter'),
                actual_runtime=runtime,
                dispatchable_when_connected=cell['status'] == 'PASS' and accepting)
        rows.append(dict(method=record['method'], job_id=job,
                         request_sha256=record['request_sha256'], hosts=hosts))
    return dict(schema='RPE_Q3_HOST_ALLOCATION_V1', rows=rows,
                scope='REQUEST_CAPABILITY_AND_CONFIGURED_ADMISSION_SEPARATE',
                scientific_evaluation_calls=0, queue_operations=0)


def merge_host_readiness(prepared, output, code_root, descriptor=None):
    """Join saved remote evidence with this run's local checks before READY."""
    from .rpe_host_eligibility import derive_host_eligibility
    output, code_root = Path(output), Path(code_root)
    config, data, profiles = host_profile_inputs(prepared, output, code_root, descriptor)
    evidence = deepcopy(data.get('evidence_by_host', {}))
    local_host = O.load_json(prepared['server2_profile'])['host_id']
    evidence[local_host] = dict(kind='local', root=str(output / 'host_checks'),
                              runtime_cache=str(output / 'host_checks/DRY_CACHE.json'))
    policy = derive_host_eligibility(prepared['assignments'], profiles, evidence,
        root=Path(profiles[local_host]['root']), base_directory=config.parent)
    policy_path = output / 'HOST_ELIGIBILITY.json'
    write(policy_path, policy)
    admission = deepcopy(data.get('host_admission', dict(schema='RPE_Q3_HOST_ADMISSION_V1', hosts={})))
    admission_path = output / 'HOST_ADMISSION.json'
    write(admission_path, admission)
    table_path = output / 'HOST_ALLOCATION.json'
    write(table_path, host_allocation_table(policy, admission))
    options_path = output / 'DISTRIBUTED_OPTIONS.json'
    write(options_path, dict(routes_override=prepared.get('routes_override', {}),
        executor=dict(scheduler='concurrent', dp_workers=3,
        prefix_checkpoint_rounds=[5, 11], prefix_cache_directory=None,
        distributed=dict(host_eligibility=str(policy_path), computation_root=profiles[local_host]['root'],
                         computation_source=prepared['computation_source'],
                         host_admission=str(admission_path),
                         required_hosts=[],
                         worker_script=str(code_root / 'scripts/run_rpe_q3_worker.py'))),
        note='Executor fragment only. It grants no scientific execution or budget; merge into a separately approved new round. Worker profiles keep the dry-checked interpreter/routes/DP/cache; work_dir and slots may be added.'))
    prepared['host_eligibility'] = dict(path=str(policy_path), schema=policy['schema'],
                                     evidence_descriptor=str(config), jobs=policy['jobs'],
                                     host_admission=str(admission_path),
                                     host_allocation=str(table_path),
                                     distributed_options=str(options_path),
                                     worker_profiles=str(output / 'worker_profiles'))
    for row in prepared['candidates']:
        decision = policy['jobs'].get(row.get('job_id'), {})
        allowed = decision.get('eligible_host_ids', [])
        accepting = [host for host in allowed
                     if admission.get('hosts', {}).get(host, {}).get('accepting', True)]
        row.update(eligible_host_ids=allowed, host_readiness=decision.get('hosts', {}),
                   accepting_eligible_host_ids=accepting,
                   dispatchable=bool(accepting) and row.get('queue_dispatchable', False))
        if row.get('queue_dispatchable'):
            row['dispatchability'] = ('DRY_CHECK_PASSED_HOSTS_ONLY' if accepting else
                                     'WAITING_FOR_HOST_ADMISSION' if allowed else 'HELD_NO_DRY_CHECK_PASSED_HOST')
    prepared['remote_dispatchability'] = 'ONLY_MATCHING_PASS_CELLS_ELIGIBLE'
    return prepared


def prepare(a, *, code_root, cli):
    """The existing schedule CLI's opt-in, zero-science readiness procedure."""
    if not a.plan_only:
        raise ValueError('--prepare-readiness requires --plan-only')
    output = a.run_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stages = Stages(output)
    raw_problem = O.load_json(a.problem)
    raw_spec = O.load_json(a.round)
    spec = deepcopy(raw_spec)
    spec.update(science_approved=False, max_new_q3_per_method={m: 0 for m in spec['methods']}, plan_output='compact')
    if spec.get('candidate_completion'):
        raise ValueError('Readiness preparation requires pre-existing evidence, not candidate acquisition')
    spec['executor'] = {k: v for k, v in spec.get('executor', {}).items() if k not in ('distributed', 'scheduler')}
    spec['executor']['scheduler'] = 'legacy'
    if spec.get('pf_completion'):
        spec['pf_completion']['mode'] = 'plan'
    round_path = output / 'PLAN_ROUND.json'
    write(round_path, spec)
    problem = cli.price_overrides(cli.scoped(raw_problem, spec.get('method_overrides')), spec)
    cache = Path(a.preparation_cache or output.parent / 'preparation_cache').resolve()
    cache.mkdir(parents=True, exist_ok=True)
    source_profile = json.loads(git(code_root, 'show', REFERENCE + ':' + REFERENCE_PATH + '/profiles/server2.json'))
    computation_root = Path(source_profile['root'])
    started_epoch = time.time()
    with (cache / 'PREPARE.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with stages.phase('CONTEXT_INPUTS_AND_CACHE'):
            context, context_stats = ContextCache(cache / 'context').prepare(code_root, problem, spec, spec['methods'])
        with HistoryCache(cache / 'HISTORY.sqlite3') as history:
            def setup(executor, current):
                executor.preparation_history_cache = history
                executor.launch_disabled = True

            def ready(result, current, routes):
                descriptor = getattr(a, 'host_readiness', None)
                configuration = Path(descriptor) if descriptor else code_root / 'docs/research/rpe_q3_host_eligibility/EVIDENCE.json'
                host_settings = O.load_json(configuration) if configuration.is_file() else {}
                with stages.phase('REQUEST_AND_REMOTE_DELIVERY'):
                    prepared = build_readiness(computation_root, dict(current, routes=routes), problem, spec,
                        dict(methods=result), output / 'readiness',
                        execution_route_updates=host_settings.get('execution_route_updates', ()))
                    _, _, profiles = host_profile_inputs(prepared, output, code_root, descriptor)
                    host = O.load_json(prepared['server2_profile'])['host_id']
                with stages.phase('SERVER2_DRY_CHECKS'):
                    dry = run_server2_dry_checks(computation_root, prepared, output / 'worker_profiles' / (host + '.json'),
                        output / 'host_checks', cache=cache / 'DRY_CACHE.json',
                        checker=code_root / 'scripts/check_rpe_q3_host_readiness.py')
                    write(cache / 'DRY_CACHE.json', dry)
                    prepared = finish_readiness(prepared, dry)
                with stages.phase('MERGE_HOST_READINESS'):
                    prepared = merge_host_readiness(prepared, output, code_root,
                                                    getattr(a, 'host_readiness', None))
                minimum = _known_minima(current, result)
                for row in prepared['candidates']:
                    method = row['method']
                    current_min = minimum[method]
                    if row.get('cost') is not None and current_min is not None:
                        cost = O.frac(row['cost'])
                        row.update(current_saved_minimum=O.enc(current_min), decrease_RZ_if_passes=O.enc(current_min - cost),
                                   decrease_percent_if_passes=O.enc(100 * (current_min - cost) / current_min))
                prepared['history'] = {m: {k: v for k, v in stats.items() if k != 'evidence'}
                                       for m, stats in history.stats_by_method.items()}
                prepared['history_cache'] = str(history.path)
                prepared['context_cache'] = context_stats
                prepared['command_to_ready_seconds'] = time.monotonic() - stages.start
                prepared['original_remote_reference'] = dict(commit=REFERENCE, path=REFERENCE_PATH + '/LOCAL0_DRY_CHECK.zip')
                prepared['all_current_scientific_states_refreshed'] = True
                write(output / 'readiness' / 'CANDIDATES.json', prepared['candidates'])
                write(output / 'readiness' / 'READY.json', prepared)
                write(output / 'READY.json', prepared)
                stages.event('CANDIDATES_AND_REMOTE_INPUTS_AVAILABLE', 'READY',
                             ready=str(output / 'READY.json'), seconds=prepared['command_to_ready_seconds'],
                             remote_delivery_unchanged=prepared['old_remote_delivery_unchanged'])

            command = SimpleNamespace(**vars(a))
            command.round = round_path
            command.run_dir = output / 'plan'
            command.plan_only = True
            stages.event('NORMAL_ENTRY', 'START')
            result = cli.round_mode(command, raw_problem, [], prepared_context=context,
                                    executor_setup=setup, result_callback=ready)
            stages.event('NORMAL_ENTRY', 'END', status='OK')
    with stages.phase('COMPACT_REPORT'):
        summary = json.loads((output / 'READY.json').read_text())
        lines = ['# H4 readiness preparation', '', 'Scientific calls and queue submissions: 0.', '',
                 '| Method | Candidate | Conditional RZ | Server2 dry-check | Eligible hosts |', '|---|---|---:|---|---|']
        for row in summary['candidates']:
            method = row['method']
            observed = summary['server2']['methods'].get(method, {})
            status = observed.get('original_status') or observed.get('status', 'NOT_CHECKED')
            cost = '' if row.get('cost') is None else O.show(O.frac(row['cost']))
            lines.append('| %s | %s | %s | %s | %s |' % (method, row.get('job_id', ''), cost, status,
                         ', '.join(row.get('eligible_host_ids', [])) or 'HELD: no matching PASS'))
        lines += ['', 'Exact saved outcomes remain in their original job paths and local history cache.',
                  'The previous remote requests, deployment commands and output paths are unchanged.']
        (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    metrics = dict(schema='RPE_PREPARATION_TIMING_V1', start_epoch=started_epoch,
        command_to_ready_seconds=summary['command_to_ready_seconds'],
        command_to_report_complete_seconds=time.monotonic() - stages.start,
        context_cache=context_stats, history=summary['history'], dry_checks=summary['server2'],
        scientific_calls=0, collection_calls=0, queue_operations=0,
        publication='Previous immutable remote delivery already public; new report Git publication measured separately')
    write(output / 'TIMING.json', metrics)
    stages.event('REPORT_COMPLETE', 'COMPLETE', seconds=metrics['command_to_report_complete_seconds'])
    return SimpleNamespace(run=output, state=dict(stop=dict(reason='READINESS_PLAN_COMPLETE')), result=result)
