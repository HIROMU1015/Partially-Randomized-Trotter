"""Partition a fixed search and reconcile the PBS allocation's local work."""
from copy import deepcopy
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from .rpe_q3_worker import atomic, digest, read


def partition_round(round_spec):
    widths = round_spec.get('search_overrides', {}).get('requested_widths', {})
    parts = []
    for method in round_spec['methods']:
        for width in widths.get(method) or [None]:
            boundary = dict(method=method, width=width)
            parts.append(dict(boundary, id=digest(boundary)[:20]))
    return parts


def prepare_plan(run, settings):
    scope, problem, round_spec = (read(run/name) for name in
                                 ('APPLIED_SCOPE.json', 'PROBLEM.json', 'ROUND.json'))
    if round_spec.get('science_approved') is not True:
        raise ValueError('PBS continuation requires the fixed normal search scope')
    for name in ('code_root', 'interpreter'):
        if not Path(settings[name]).is_absolute():
            raise ValueError('PBS continuation requires fixed absolute runtime paths')
    if any(not Path(path).is_absolute() for path in settings.get('input_repositories', ())):
        raise ValueError('PBS continuation requires absolute fixed input repository paths')
    setup = dict(settings)
    if not re_full_sha(setup['source']):
        raise ValueError('PBS continuation requires a fixed optimizer commit')
    budgets = round_spec['max_new_q3_per_method']
    prep = (round_spec['executor'].get('distributed') or {}).get('preparation', {}).get('budgets', {})
    if 'memory_bytes_per_task' in setup and (type(setup['memory_bytes_per_task']) is not int
                                           or setup['memory_bytes_per_task'] <= 0):
        raise ValueError('PBS continuation requires a positive task memory estimate')
    if (any(v not in ('UNLIMITED', 0) for v in (*budgets.values(), *prep.values()))
            or {'max_steps_per_method', 'max_cycles'}.intersection(scope.get('scope', {}).get('stop', {}))):
        return None  # Preserve ordinary execution when PBS has no separate budget.
    parts = partition_round(round_spec)
    return dict(schema='RPE_PBS_OFFLINE_PLAN_V1', run_id=scope['run_id'], scope=scope,
                settings=setup, problem=problem, round=round_spec, parts=parts)


def re_full_sha(value):
    return isinstance(value, str) and len(value) == 40 and all(c in '0123456789abcdef' for c in value)


def mapped_paths(tree, substitutes):
    if isinstance(tree, dict):
        return {k: mapped_paths(v, substitutes) for k, v in tree.items()}
    if isinstance(tree, list):
        return [mapped_paths(v, substitutes) for v in tree]
    if isinstance(tree, str):
        for source, target in substitutes:
            if tree == source or tree.startswith(source + '/'):
                return target + tree[len(source):]
    return tree


def input_environment(repositories):
    """Expose already-staged immutable Git objects to this subprocess."""
    environment = dict(os.environ)
    if repositories:
        object_dirs = []
        for path in repositories:
            relative = subprocess.check_output(['git', 'rev-parse', '--git-path', 'objects'], cwd=path,
                                               text=True, timeout=10).strip()
            directory = (Path(path)/relative).resolve()
            if not directory.is_dir():
                raise ValueError('Missing PBS offline input objects: '+str(directory))
            object_dirs.append(str(directory))
        if inherited := environment.get('GIT_ALTERNATE_OBJECT_DIRECTORIES'):
            object_dirs.append(inherited)
        environment['GIT_ALTERNATE_OBJECT_DIRECTORIES'] = os.pathsep.join(object_dirs)
    return environment


def part_documents(plan, part, profile, path):
    settings = plan['settings']
    substitutions = sorted(settings.get('path_map', {}).items(), key=lambda item: -len(item[0]))
    problem = mapped_paths(plan['problem'], substitutions)
    spec = mapped_paths(plan['round'], substitutions)
    method = part['method']
    spec['methods'] = [method]
    spec['round_id'] = str(spec['round_id']) + '-pbs-' + part['id']
    spec['max_new_q3_per_method'] = {method: spec['max_new_q3_per_method'][method]}
    spec.setdefault('search_overrides', {})['requested_widths'] = (
        {method: [part['width']]} if part['width'] is not None else {})
    selected = deepcopy(profile)
    selected.pop('pbs', None)
    selected.update(host_id='local', work_dir=str(path/'worker'),
                    prefix_cache_directory=str(path.parent/'dp-cache'),
                    artifact_cache_directory=str(path.parent/'artifacts'),
                    preparation_metadata_cache_directory=str(path.parent/'preparation-metadata'))
    distributed = spec['executor'].get('distributed', {})
    # The ordinary local dispatcher executes the preparation DAG and Q3
    # on the same allocation. There is no scheduler or remote relay here.
    for key in ('pbs', 'pbs_pinned_settings', 'relay', 'legacy_relay_command',
                'release_manifest', 'local_deployment_layout', 'execution_hosts',
                'required_hosts', 'required_release', 'host_admission',
                'host_eligibility', 'host_stage_capabilities', 'preparation_host_preference',
                'preparation_provider_hosts', 'local_pf_continuation', 'local_resource_worker'):
        distributed.pop(key, None)
    worker_script = Path(settings['code_root'])/'scripts/run_rpe_q3_worker.py'
    distributed.update(bind='127.0.0.1', port=0, local_worker=selected, preparation_profiles={'local': selected},
                       execution_hosts=['local'], live_q3_readiness=True, worker_script=str(worker_script))
    if 'computation_source' in distributed:
        distributed['computation_root'] = selected['root']
    spec['executor']['distributed'] = distributed
    spec['executor']['max_concurrent_jobs'] = selected['slots']
    spec['executor']['prior_run_dirs'] = [p for p in spec['executor'].get('prior_run_dirs', ())
                                          if Path(p).is_dir()]
    spec['price_validation'] = dict(spec.get('price_validation') or {}, compile_approved=False)
    return problem, spec


class OfflineSearch:
    """Run bounded slices of the normal optimizer inside the acquired node."""

    def __init__(self, plan, profile, directory, resources, *, popen=subprocess.Popen):
        from .rpe_pbs_worker import task_cpu_footprint

        self.plan, self.profile = plan, deepcopy(profile)
        cpu_slots = resources['cpu_count'] // task_cpu_footprint(profile)
        memory = max(0, resources['memory_bytes']-1024**3)
        task_memory = plan['settings'].get('memory_bytes_per_task') or memory
        self.slots = min(resources['maximum_slots'], cpu_slots, memory//task_memory if task_memory else 0)
        self.profile['slots'] = self.slots
        self.profile['slot_policy'] = dict(mode='fixed', initial_slots=self.slots, max_slots=self.slots,
            resource_limits={k: resources[k] for k in ('cpu_count', 'memory_bytes')})
        self.directory = Path(directory)/'offline-search'
        scratch = os.environ.get('PBS_JOBFS') or os.environ.get('TMPDIR')
        self.parts_dir = (Path(scratch)/('rpe-'+plan['run_id']+'-'+Path(directory).name)/'parts'
                          if scratch and Path(scratch).is_dir() else None)
        self.process = None
        self.part = None
        self.popen = popen
        self.cursor = 0
        self.validated_entry = None
        self.environment = None
        self.recheck_at = 0.
        self.next_export_at = 0.
        self.storage_ready = not bool(profile.get('storage_limit'))
        manifest = self.directory/'RESULTS.json'
        self.revision = len(read(manifest)) if manifest.exists() else 0
        coverage = self.directory/'COVERAGE.json'
        if coverage.exists():
            self.revision += read(coverage)['changes']

    def _entry(self):
        if self.validated_entry is None:
            settings = self.plan['settings']
            root = Path(settings['code_root'])
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                               text=True, timeout=10).strip()
            entry = root/'scripts/run_rpe_schedule_search.py'
            if revision != settings['source'] or not entry.is_file():
                raise ValueError('PBS continuation optimizer deployment differs')
            self.environment = input_environment(settings.get('input_repositories', ()))
            self.validated_entry = entry
        return self.validated_entry

    def request_stop(self):
        if self.part is not None and not (self.part/'STOP_REQUESTED.json').exists():
            atomic(self.part/'STOP_REQUESTED.json', {'reason': 'PBS_OFFLINE_HANDOFF'})

    def reserve_storage(self, storage):
        if self.storage_ready or self.parts_dir is None:
            return
        used, limit = storage.get('used_bytes'), storage.get('limit_bytes')
        if type(used) is not int or type(limit) is not int or limit <= used:
            return
        # The durable result is a second copy, so reserve up to the scratch
        # allowance for it within the last verified remaining host budget.
        remaining = (limit-used)//2
        if remaining:
            self.profile['storage_limit'] = dict(limit_bytes=remaining, roots=[str(self.parts_dir)],
                                                 sample_path=str(self.parts_dir/'STORAGE_USAGE.json'))
            self.storage_ready = True

    def advance(self, *, disconnected, runnable, batch):
        if self.process is not None:
            if not disconnected:
                self.request_stop()
            if self.process.poll() is None:
                if time.monotonic() >= self.next_export_at:
                    self.export(self.part)
                    self.next_export_at = time.monotonic()+20
                return True
            added = self.export(self.part)
            self.record_coverage(self.part)
            if added and not (self.part/'SEARCH_TERMINATION.json').exists():
                self.cursor = (self.cursor-1) % len(self.plan['parts'])
            self.process, self.part = None, None
        if (not disconnected or not runnable or self.parts_dir is None or not self.storage_ready
                or self.slots < 1
                or time.monotonic() < self.recheck_at):
            return False
        parts = self.plan['parts']
        for offset in range(len(parts)):
            index = (self.cursor + offset) % len(parts)
            part = parts[index]
            path = self.parts_dir/part['id']
            if (path/'SEARCH_TERMINATION.json').exists():
                continue
            self.cursor = (index+1) % len(parts)
            path.mkdir(parents=True, exist_ok=True)
            entry = self._entry()
            problem, spec = part_documents(self.plan, part, self.profile, path)
            for name, data in (('PROBLEM.json', problem), ('ROUND.json', spec)):
                file = path/name
                if file.exists() and read(file) != data:
                    raise ValueError('PBS continuation part inputs changed')
                if not file.exists():
                    atomic(file, data)
            (path/'STOP_REQUESTED.json').unlink(missing_ok=True)
            command = [self.plan['settings']['interpreter'], '-B', str(entry),
                       '--problem', str(path/'PROBLEM.json'), '--round', str(path/'ROUND.json'),
                       '--run-dir', str(path), '--batch', str(min(batch, self.slots)), '--max-cycles', '1']
            with (path/'OFFLINE.log').open('ab') as log:
                self.process = self.popen(command, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, env=self.environment)
            self.part = path
            self.next_export_at = time.monotonic()+20
            return True
        self.recheck_at = time.monotonic()+30
        return False

    def export(self, part):
        """Publish one bounded inventory of fixed completed Q3 results."""
        manifest = self.directory/'RESULTS.json'
        records = read(manifest) if manifest.exists() else {}
        new = 0
        for outcome in (part/'jobs').glob('*/*/OUTCOME.json'):
            job = outcome.parent
            if job.parent.name.startswith('PREPARATION:'):
                continue
            names = ('REQUEST.json', 'OUTCOME.json', 'PROCESS_RETURN.json')
            if any(not (job/name).is_file() for name in names):
                continue
            receipt = read(job/'PROCESS_RETURN.json')
            request = read(job/'REQUEST.json')
            if (receipt.get('returncode') != 0 or receipt.get('signal') is not None
                    or request.get('request_id') != job.name):
                continue
            key = f'{job.parent.name}/{job.name}'
            if key in records:
                continue
            destination = self.directory/'jobs'/key
            destination.mkdir(parents=True, exist_ok=True)
            for name in names:
                temporary = destination/(name+'.tmp')
                shutil.copyfile(job/name, temporary)
                temporary.replace(destination/name)
            records[key] = dict(part=part.name, files={name: (destination/name).stat().st_size
                                                       for name in names})
            new += 1
        if new:
            atomic(manifest, records)
            self.revision += new
        return new

    def record_coverage(self, part):
        path = self.directory/'COVERAGE.json'
        status = read(path) if path.exists() else dict(changes=0, parts={})
        marker = part/'SEARCH_TERMINATION.json'
        db = part/'distributed'/'QUEUE.sqlite3'
        counts = {}
        if db.is_file():
            from contextlib import closing

            with closing(sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True)) as connection:
                for method, state, total in connection.execute("SELECT method,state,count(*) FROM jobs GROUP BY 1,2"):
                    counts.setdefault(method, {})[state] = total
        observed = dict(counts=counts, termination=read(marker) if marker.exists() else None)
        if status['parts'].get(part.name) != observed:
            status['parts'][part.name] = observed
            status['changes'] += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic(path, status)
            self.revision += 1
