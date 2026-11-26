"""Partition a fixed search and reconcile the PBS allocation's local work."""
from collections import Counter
from copy import deepcopy
import os
import fcntl
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

from .rpe_q3_worker import atomic, digest, read


def sqlite_busy(error):
    # The PBS shared filesystem can briefly reject even a read-only snapshot.
    # Only retry contention; leave corruption and real SQL errors visible.
    return ('locked' in str(error).lower() or 'busy' in str(error).lower())


def partition_round(round_spec, problem=None, *, independent=False):
    if independent and round_spec.get('hchain'):
        from .rpe_hchain_scope import fixed_plan

        _, plan = fixed_plan(round_spec['hchain'])
        groups = {}
        for case in plan['cases']:
            if case['method'] not in round_spec['methods']:
                continue
            width = case['width_over_h']
            value = Fraction(width['numerator'], width['denominator']) if isinstance(width, dict) else Fraction(str(width))
            groups.setdefault((case['method'], value), []).append(case['id'])
        parts = []
        for (method, width), ids in groups.items():
            value = dict(numerator=width.numerator, denominator=width.denominator)
            boundary = dict(method=method, axis='width_over_h', value=value, width=value)
            parts.append(dict(boundary, id=digest(boundary)[:20], case_ids=ids))
        return parts
    widths = round_spec.get('search_overrides', {}).get('requested_widths') or (
        (problem or {}).get('search', {}).get('requested_widths') or {})
    parts = []
    for method in round_spec['methods']:
        values = widths.get(method) or [None]
        axis = 'width'
        base = dict((problem or {}).get('methods', {}).get(method, {}),
                    **round_spec.get('method_overrides', {}).get(method, {}))
        if independent and base.get('configuration_ids'):
            values = [None]
        if independent and len(values) == 1 and not base.get('configuration_ids'):
            # Split a single-width scope only along an already declared axis.
            # Never invent additional physical parameters or configuration IDs.
            for key in (('L_D_values',) if method.startswith('PR_')
                        else (() if base.get('configuration_ids') else ('delta_over_h_values', 'p_values'))):
                if isinstance(base.get(key), list) and len(base[key]) > 1:
                    axis, values = key, base[key]
                    break
        seen = set()
        for value in values:
            if independent:
                from .rpe_kappa_schedule import _frac
                canonical = str(_frac(value)) if axis == 'width' and value is not None else digest(value)
                if canonical in seen:
                    continue
                seen.add(canonical)
                boundary = dict(method=method, axis=axis, value=value,
                                width=value if axis == 'width' else None)
            else:
                boundary = dict(method=method, width=value)
            parts.append(dict(boundary, id=digest(boundary)[:20]))
    return parts


def divide_budget(value, blocks):
    if value == 'UNLIMITED':
        return {block['id']: value for block in blocks}
    if type(value) is not int or value < 0:
        raise ValueError('Invalid fixed search budget')
    count = len(blocks)
    return {block['id']: value//count + (i < value%count) for i, block in enumerate(blocks)}


def prepare_plan(run, settings, *, round_spec=None):
    scope, problem, saved = (read(run/name) for name in
                              ('APPLIED_SCOPE.json', 'PROBLEM.json', 'ROUND.json'))
    round_spec = deepcopy(round_spec if round_spec is not None else saved)
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
    if {'max_steps_per_method', 'max_cycles'}.intersection(scope.get('scope', {}).get('stop', {})):
        return None  # These whole-run limits cannot be silently multiplied.
    independent = setup.get('mode') == 'independent'
    if not independent and any(v not in ('UNLIMITED', 0) for v in (*budgets.values(), *prep.values())):
        return None
    parts = partition_round(round_spec, problem, independent=independent)
    if independent:
        hchain = bool(round_spec.get('hchain'))
        distributed = round_spec['executor'].get('distributed') or {}
        pbs_only = hchain and not distributed.get('local_worker') and not (
            (distributed.get('relay') or {}).get('workers'))
        if not parts or len(parts) < 2 and not pbs_only:
            return None  # No disjoint split; use the ordinary shared queue.
        pbs = distributed.get('pbs') or {}
        if hchain and not pbs.get('workers'):
            raise ValueError('Independent H-chain search requires PBS worker profiles')
        pbs_cpu = (pbs.get('resources') or {}).get('ncpus',0)*len(pbs.get('workers',[]))
        local = distributed.get('local_worker') or {}
        other_cpu = (local.get('slot_policy') or {}).get('resource_limits',{}).get('cpu_count',0)
        # If a complete fleet estimate is not available, keep a configurable
        # equal-share starting point; block ownership is still disjoint.
        auto_share = Fraction(pbs_cpu,pbs_cpu+other_cpu) if pbs_cpu and other_cpu else Fraction(1,2)
        share = Fraction(str(setup.get('pbs_share', auto_share)))
        if not (0 < share <= 1 if pbs_only else 0 < share < 1):
            raise ValueError('pbs_share must lie strictly between zero and one')
        # Interleave methods so an entire expensive method is not prioritized
        # merely because it is first in ROUND. All blocks remain whole.
        grouped = {m: [p for p in parts if p['method'] == m] for m in round_spec['methods']}
        interleaved = [group[i] for i in range(max(map(len, grouped.values())))
                       for group in grouped.values() if i < len(group)]
        number = len(parts) if pbs_only else max(1, min(len(parts)-1, round(len(parts)*share)))
        pbs_hosts = {worker['host_id'] for worker in pbs.get('workers',[])}
        blocked = set()
        if pbs_hosts:
            for provider, hosts in (distributed.get('preparation_provider_hosts') or {}).items():
                if pbs_hosts.intersection(hosts):
                    continue
                if provider.startswith('PF_'):
                    blocked.update(('FULL_PF','TRUNCATED_PF'))
                elif provider.startswith('DF_'):
                    blocked.add('PR_DF')
                elif provider.startswith('PAULI_'):
                    blocked.add('PR_PAULI')
                elif provider == 'HCHAIN_SCOPED':
                    blocked.update(round_spec['methods'])
        central_ids = set(setup.get('central_block_ids', ()))
        if central_ids - {p['id'] for p in parts}:
            raise ValueError('Central-only block is outside the fixed search')
        pbs_ids = {p['id'] for p in [p for p in interleaved if p['method'] not in blocked and p['id'] not in central_ids][:number]}
        if pbs_only and len(pbs_ids) != len(parts):
            raise ValueError('H-chain PBS-only plan has a block without an eligible PBS host')
        if not pbs_ids:
            return None
        charges = {}
        if hchain and 'legacy' in round_spec['hchain']:
            from .rpe_hchain_scope import fixed_plan
            from rpe_hchain_legacy import checked
            _, physical = fixed_plan(round_spec['hchain'])
            _, history = checked(round_spec['hchain'], physical)
            by_id = Counter(row['configuration_id'] for row in (*history['records'], *history['holds']))
            charges = {p['id']:sum(by_id[cid] for cid in p['case_ids']) for p in parts}
        remaining = {method: cap if cap == 'UNLIMITED' else max(
            0, cap-sum(charges.get(p['id'],0) for p in grouped[method]))
                     for method, cap in budgets.items()}
        for block in parts:
            quota = divide_budget(remaining[block['method']], grouped[block['method']])[block['id']]
            block.update(owner='pbs' if block['id'] in pbs_ids else 'central',
                         q3_budget=quota if quota == 'UNLIMITED' else quota+charges.get(block['id'],0),
                         preparation_budgets={stage: divide_budget(cap, parts)[block['id']]
                                              for stage, cap in prep.items()})
        # The local optimizer must never receive its parent's PBS plan back.
        (round_spec['executor'].get('distributed') or {}).pop('pbs_offline_plan', None)
    plan = dict(schema='RPE_PBS_OFFLINE_PLAN_V2' if independent else 'RPE_PBS_OFFLINE_PLAN_V1',
                run_id=scope['run_id'], scope=scope, settings=setup, problem=problem,
                round=round_spec, parts=parts)
    if independent and round_spec.get('hchain'):
        from .rpe_hchain_scope import fixed_plan
        from run_rpe_hchain_preparation import planned_inputs
        _, physical = fixed_plan(round_spec['hchain'])
        plan['hchain_preparation'] = dict(owner=physical['owner'], binding=physical['binding'],
            stages={stage: sorted(planned_inputs(physical, stage)) for stage in ('signal','compile','audit')})
    if independent:
        holds = round_spec.get('executor', {}).get('prior_q3_holds', [])
        preflight = run/'DUPLICATE_PREFLIGHT.json'
        if holds:
            if not preflight.is_file() or read(preflight).get('status') != 'PASS':
                raise ValueError('Independent search requires verified prior Q3 holds')
            plan['verified_prior_q3_holds'] = read(preflight)['prior_q3_holds']
        waiting = [p for p in interleaved if p['owner']=='pbs']
        nodes = max(1,len(pbs.get('workers',[])))
        waves = setup.get('packages_per_node',2)
        if type(waves) is not int or waves < 1:
            raise ValueError('packages_per_node must be positive')
        groups = min(len(waiting),nodes*waves)
        packages=[]
        for i in range(groups):
            members=waiting[i::groups]
            ids=[p['id'] for p in members]
            packages.append(dict(id=ids[0] if len(ids)==1 else digest(ids)[:20],
                members=ids,methods=list(dict.fromkeys(p['method'] for p in members))))
        plan['packages']=packages
    return plan


def independent_plan(plan):
    return bool(plan and plan.get('schema') == 'RPE_PBS_OFFLINE_PLAN_V2')


def restrict_round(plan, parts, *, source=None):
    """Restrict an existing exact scope along its one declared partition axis."""
    spec = deepcopy(source if source is not None else plan['round'])
    methods = [m for m in spec['methods'] if any(p['method'] == m for p in parts)]
    spec['methods'] = methods
    if spec.get('hchain'):
        spec['hchain']['case_ids'] = [cid for part in parts for cid in part['case_ids']]
    overrides = spec.setdefault('method_overrides', {})
    for method in list(overrides):
        if method not in methods:
            del overrides[method]
    search = spec.setdefault('search_overrides', {})
    widths = search.get('requested_widths') or plan['problem'].get('search', {}).get('requested_widths') or {}
    widths = {m: deepcopy(values) for m, values in widths.items() if m in methods}
    for method in methods:
        group = [p for p in parts if p['method'] == method]
        axis = group[0]['axis']
        values = [p['value'] for p in group]
        if axis == 'width_over_h' and spec.get('hchain'):
            pass  # The fixed H-chain plan selects exact case IDs, not H4 width overrides.
        elif axis == 'width':
            if values != [None]:
                widths[method] = values
        else:
            overrides.setdefault(method, {})[axis] = values
            if axis == 'delta_over_h_values':
                overrides[method]['delta_over_h_range'] = [values[0], values[-1]]
                overrides[method]['configuration_ids'] = None
        caps = [p['q3_budget'] for p in group]
        spec['max_new_q3_per_method'][method] = ('UNLIMITED' if 'UNLIMITED' in caps else sum(caps))
    spec['max_new_q3_per_method'] = {m: spec['max_new_q3_per_method'][m] for m in methods}
    search['requested_widths'] = widths or None
    executor = spec.get('executor', {})
    if 'prior_q3_holds' in executor:
        executor['prior_q3_holds'] = [row for row in executor['prior_q3_holds'] if row['method'] in methods]
    for key in ('width_families', 'q3_budget_parents'):
        if key in spec:
            spec[key] = [row for row in spec[key] if row['method'] in methods]
    distributed = spec['executor'].get('distributed') or {}
    if distributed.get('preparation'):
        stages = distributed['preparation'].get('budgets', {})
        for stage in stages:
            caps = [p['preparation_budgets'].get(stage, 0) for p in parts]
            stages[stage] = 'UNLIMITED' if 'UNLIMITED' in caps else sum(caps)
    return spec


def pbs_parts(plan):
    return [p for p in plan['parts'] if p.get('owner') == 'pbs'] if independent_plan(plan) else plan['parts']


def pbs_units(plan):
    return plan.get('packages',pbs_parts(plan))


def unit_members(plan, unit):
    if 'members' not in unit:
        return [unit]
    return [p for p in pbs_parts(plan) if p['id'] in unit['members']]


def all_parts_finished(plan, coverage):
    terminal = {'REQUESTED_WIDTH_RULE_EXHAUSTED', 'Q3_BUDGET_EXHAUSTED', 'EXPLICIT_STEP_LIMIT', 'HCHAIN_SCOPE_COMPLETE'}
    return all((coverage.get(p['id'], {}).get('termination') or {}).get('reason') in terminal
               for p in pbs_units(plan))


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
    original = restrict_round(plan, unit_members(plan,part)) if independent_plan(plan) else plan['round']
    spec = mapped_paths(original, substitutions)
    if (spec.get('candidate_completion') or {}).get('df_max_width_source'):
        spec['candidate_completion']['df_max_width_code_root'] = settings['code_root']
    spec['round_id'] = str(spec['round_id']) + '-pbs-' + part['id']
    if not independent_plan(plan):
        method=part['method']
        spec['methods']=[method]
        spec['max_new_q3_per_method']={method:spec['max_new_q3_per_method'][method]}
        spec.setdefault('search_overrides',{})['requested_widths']=(
            {method:[part['width']]} if part['width'] is not None else None)
    selected = deepcopy(profile)
    selected.pop('pbs', None)
    if independent_plan(plan):
        # The selected stage profiles are now local to the acquired node.
        for rule in selected.get('preparation_profiles', []):
            rule['profile']['host_id'] = 'local'
        for rule in selected.get('route_profiles', []):
            if isinstance(rule, dict) and isinstance(rule.get('profile'), dict):
                rule['profile']['host_id'] = 'local'
        spec.pop('release_worker_policy', None)
    selected.update(host_id='local', work_dir=str(path/'worker'),
                    prefix_cache_directory=str(path.parent/'dp-cache'),
                    artifact_cache_directory=str(path.parent/'artifacts'),
                    preparation_metadata_cache_directory=str(path.parent/'preparation-metadata'))
    distributed = spec['executor'].get('distributed', {})
    placement = {}
    if independent_plan(plan):
        original_host = profile['host_id']
        for key in ('preparation_provider_hosts','preparation_request_hosts'):
            if key in distributed:
                placement[key] = {name:['local'] if original_host in hosts else []
                                  for name,hosts in distributed[key].items()}
        if original_host in (distributed.get('host_stage_capabilities') or {}):
            placement['host_stage_capabilities']={'local':distributed['host_stage_capabilities'][original_host]}
        if hosts := distributed.get('pf_legacy_compile_hosts'):
            placement['pf_legacy_compile_hosts'] = ['local'] if original_host in hosts else hosts
    # The ordinary local dispatcher executes the preparation DAG and Q3
    # on the same allocation. There is no scheduler or remote relay here.
    for key in ('pbs', 'pbs_pinned_settings', 'relay', 'legacy_relay_command',
                'release_manifest', 'local_deployment_layout', 'execution_hosts',
                'required_hosts', 'required_release', 'host_admission',
                'host_eligibility', 'host_stage_capabilities', 'preparation_host_preference',
                'preparation_provider_hosts', 'local_pf_continuation', 'local_resource_worker',
                'pbs_offline_plan', 'pbs_offline_search', 'preparation_request_hosts', 'host_transport_paths'):
        distributed.pop(key, None)
    worker_script = Path(settings['code_root'])/'scripts/run_rpe_q3_worker.py'
    if independent_plan(plan):
        nested = profile.get('pbs', {})
        overlay = nested.get('offline_controller_overlay') or {}
        if adapter := overlay.get('q3_adapter'):
            if (not spec.get('hchain') or overlay.get('base_source') != settings['source']
                    or set(adapter) != {'adapter_root', 'adapter_manifest'}):
                raise ValueError('Q3 adapter selection does not match the fixed H-chain plan')
            # The existing adapter profile validates its own fixed checkout and
            # bytes at readiness and launch; the request SOURCE stays pinned.
            selected.update(deepcopy(adapter))
        if selected.get('storage_limit') and nested.get('independent_storage_sample') and overlay.get('worker_script'):
            selected['storage_limit']['parent_compute_sample'] = nested['independent_storage_sample']
            worker_script = Path(overlay['worker_script'])
    distributed.update(placement)
    if spec.get('hchain'):
        spec['hchain'].pop('pbs_offline_search', None)
        distributed.pop('auto_allocation', None)
        spec['hchain']['pbs_nodes'] = 0
        if selected.get('storage_limit'):
            # Host-specific sampler names cannot freeze a block to one node.
            # Keep the same quota roots and limit; only the cache path follows
            # the durable block as it changes allocation owner.
            selected['storage_limit']['sample_path'] = str(path/'STORAGE_SAMPLE.json')
    if independent_plan(plan):
        # Central saved decisions are tied to different host identities.
        # Start with an empty local policy so the normal per-request live
        # dry-check obtains fresh proof instead of treating the policy as absent.
        distributed['host_eligibility'] = {'schema':'RPE_Q3_HOST_ELIGIBILITY_V1','jobs':{}}
        if distributed.get('preparation'):
            distributed['preparation']['local_shared_git_inputs'] = True
    distributed.update(bind='127.0.0.1', port=0, local_worker=selected, preparation_profiles={'local': selected},
                       execution_hosts=['local'], live_q3_readiness=True, worker_script=str(worker_script))
    if 'computation_source' in distributed:
        distributed['computation_root'] = selected['root']
    spec['executor']['distributed'] = distributed
    spec['executor']['max_concurrent_jobs'] = selected['slots']
    if independent_plan(plan):
        if 'verified_prior_q3_holds' in plan:
            spec['executor'].pop('prior_q3_holds', None)
            spec['executor']['partition_prior_q3_holds'] = [deepcopy(row) for row in plan['verified_prior_q3_holds']
                                                               if row['method'] in spec['methods'] and
                                                               (not spec.get('hchain') or row['configuration_id'] in spec['hchain']['case_ids'])]
        # Every block can read previously exported Q3 receipts on the same
        # durable filesystem without a controller round trip.
        spec['executor'].setdefault('prior_run_dirs', []).append(str(path.parent.parent))
        spec['executor']['prior_run_dirs'] = list(dict.fromkeys(spec['executor']['prior_run_dirs']))
    spec['executor']['prior_run_dirs'] = [p for p in spec['executor'].get('prior_run_dirs', ())
                                          if Path(p).is_dir()]
    if not independent_plan(plan):
        spec['price_validation'] = dict(spec.get('price_validation') or {}, compile_approved=False)
    return problem, spec


class OfflineSearch:
    """Run bounded slices of the normal optimizer inside the acquired node."""

    def __init__(self, plan, profile, directory, resources, *, popen=subprocess.Popen):
        from .rpe_pbs_worker import task_cpu_footprint

        self.plan, self.profile = plan, deepcopy(profile)
        self.independent = independent_plan(plan)
        if self.independent and profile['pbs'].get('compute_storage_sampling'):
            self.profile['pbs']['independent_storage_sample'] = str(Path(directory)/'STORAGE_USAGE_GATEWAY.json')
        self.parts = pbs_units(plan)
        self.claim = None
        cpu_slots = resources['cpu_count'] // task_cpu_footprint(profile)
        memory = max(0, resources['memory_bytes']-1024**3)
        task_memory = plan['settings'].get('memory_bytes_per_task') or memory
        self.slots = min(resources['maximum_slots'], cpu_slots, memory//task_memory if task_memory else 0)
        self.profile['slots'] = self.slots
        self.profile['slot_policy'] = dict(mode='fixed', initial_slots=self.slots, max_slots=self.slots,
            resource_limits={k: resources[k] for k in ('cpu_count', 'memory_bytes')})
        if self.independent and self.slots:
            # Start with the known estimate, then let the ordinary local
            # worker's measured resource scheduler use the acquired node.
            maximum = min(resources['maximum_slots'], resources['cpu_count'])
            initial = self.slots
            self.slots = maximum
            self.profile['slots'] = maximum
            self.profile['slot_policy'].update(mode='auto', initial_slots=initial, max_slots=maximum,
                separate_task_resources=True, pack_measured_task_classes=True)
        self.directory = Path(directory)/'offline-search'
        if self.independent:
            durable = Path(profile['pbs']['independent_root']).resolve()
            policy = profile.get('storage_limit')
            if policy and not any(durable.is_relative_to(Path(root).resolve()) for root in policy['roots']):
                raise ValueError('PBS independent workspace is outside its storage quota')
            self.directory = durable
        scratch = os.environ.get('PBS_JOBFS') or os.environ.get('TMPDIR')
        self.parts_dir = (Path(scratch)/('rpe-'+plan['run_id']+'-'+Path(directory).name)/'parts'
                          if scratch and Path(scratch).is_dir() else None)
        if self.independent:
            # A new allocation opens the same durable checkpoints, not a
            # fresh scientific run. PBS nodes share this owned run directory.
            self.parts_dir = self.directory/'parts'
        self.process = None
        self.part = None
        self.popen = popen
        self.cursor = 0
        self.retry_after = {}
        self.validated_entry = None
        self.environment = None
        self.recheck_at = 0.
        self.next_export_at = 0.
        self.storage_ready = self.independent or not bool(profile.get('storage_limit'))
        if self.independent:
            pointer = Path(directory)/'offline-search'
            pointer.mkdir(parents=True, exist_ok=True)
            atomic(pointer/'INDEPENDENT.json', dict(root=str(self.directory)))
        manifest = self.directory/'RESULTS.json'
        self.revision = len(read(manifest)) if manifest.exists() else 0
        coverage = self.directory/'COVERAGE.json'
        if coverage.exists():
            self.revision += read(coverage)['changes']

    def shared_revision(self):
        if not self.independent:
            return self.revision
        manifest, coverage = self.directory/'RESULTS.json', self.directory/'COVERAGE.json'
        version = tuple(p.stat().st_mtime_ns if p.is_file() else None for p in (manifest,coverage))
        if getattr(self, '_revision_version', None) != version:
            self._shared_revision = (len(read(manifest)) if manifest.is_file() else 0) + (
                read(coverage).get('changes', 0) if coverage.is_file() else 0)
            self._revision_version = version
        return self._shared_revision

    def pending(self):
        return any(not (self.parts_dir/p['id']/'SEARCH_TERMINATION.json').is_file()
                   for p in self.parts) if self.parts_dir is not None else True

    def _entry(self):
        if self.validated_entry is None:
            settings = self.plan['settings']
            root = Path(settings['code_root'])
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                               text=True, timeout=10).strip()
            filename = 'run_rpe_method_optimizer.py' if self.plan['round'].get('hchain') else 'run_rpe_schedule_search.py'
            entry = root/'scripts'/filename
            if revision != settings['source'] or not entry.is_file():
                raise ValueError('PBS continuation optimizer deployment differs')
            # Preparation publishes relative Git locators. An independent
            # run must therefore have its parts inside a checkout, not beside
            # the fixed runtime. One sparse shared-object checkout per durable
            # run retains all block checkpoints and keeps evidence in quota.
            if self.independent and not self.directory.resolve().is_relative_to(root.resolve()):
                entry = self._durable_entry(root, settings['source'])
            self.environment = input_environment([root, entry.parents[1], *settings.get('input_repositories', ())])
            self.validated_entry = entry
        return self.validated_entry

    def _durable_entry(self, source_root, revision):
        root = self.directory.resolve()
        root.mkdir(parents=True, exist_ok=True)
        paths = ('scripts', 'src', 'tools', 'configs', 'schemas', 'pyproject.toml',
                 'requirements.txt', 'requirements-gpu.txt', 'setup.py', 'setup.cfg')
        def git(*args, **kwargs):
            return subprocess.run(['git', '-C', str(root), *args], check=True,
                                  capture_output=True, text=True, timeout=120, **kwargs).stdout.strip()
        with (root/'RUNTIME.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (root/'.git').exists():
                git('init', '--quiet')
            source_objects = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse',
                '--path-format=absolute', '--git-path', 'objects'], text=True, timeout=10).strip()
            alternate = root/'.git/objects/info/alternates'
            alternate.parent.mkdir(parents=True, exist_ok=True)
            if not alternate.exists():
                alternate.write_text(source_objects+'\n')
            head = subprocess.run(['git','-C',str(root),'rev-parse','--verify','HEAD'],
                                  capture_output=True, text=True, timeout=10)
            if head.returncode:
                # A portable runtime may lack unrelated historical blobs.
                # Keep SOURCE as parent but index only its exact code trees.
                tree = git('mktree', input=git('ls-tree', revision, '--', *paths)+'\n')
                base = git('-c','user.name=RPE worker','-c','user.email=rpe-worker@localhost',
                           'commit-tree',tree,'-p',revision,'-m','Portable evidence checkout')
                git('checkout','--detach',base)
                atomic(root/'PORTABLE_SOURCE.json', {'source':revision,'evidence_base':base})
            elif git('diff','--name-only',revision,'--',*paths):
                raise ValueError('Durable PBS optimizer code differs from fixed SOURCE')
            if git('ls-files','--others','--exclude-standard','--',*paths):
                raise ValueError('Unexpected durable PBS optimizer files')
        entry = root/'scripts'/('run_rpe_method_optimizer.py' if self.plan['round'].get('hchain') else 'run_rpe_schedule_search.py')
        if not entry.is_file():
            raise ValueError('Durable PBS optimizer entry missing')
        return entry

    def request_stop(self):
        if self.part is not None and not (self.part/'STOP_REQUESTED.json').exists():
            atomic(self.part/'STOP_REQUESTED.json', {'reason': 'PBS_BLOCK_HANDOFF' if self.independent else 'PBS_OFFLINE_HANDOFF'})

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

    def advance(self, *, disconnected, runnable, batch, stop=False):
        if self.process is not None:
            if stop or (not self.independent and not disconnected):
                self.request_stop()
            if self.process.poll() is None:
                if time.monotonic() >= self.next_export_at:
                    self.export(self.part)
                    if self.independent:
                        self.record_coverage(self.part)
                        self._rotate_waiting_part()
                    self.next_export_at = time.monotonic()+20
                return True
            added = self.export(self.part)
            self.record_coverage(self.part)
            if added and not self.independent and not (self.part/'SEARCH_TERMINATION.json').exists():
                self.cursor = (self.cursor-1) % len(self.parts)
            if self.claim is not None:
                self.claim.close()
                self.claim = None
            if self.independent and not (self.part/'SEARCH_TERMINATION.json').exists():
                self.retry_after[self.part.name] = time.monotonic()+30
            self.process, self.part = None, None
        if (stop or (not self.independent and not disconnected) or not runnable or self.parts_dir is None or not self.storage_ready
                or self.slots < 1
                or time.monotonic() < self.recheck_at):
            return False
        parts = self.parts
        first = self.cursor
        for offset in range(len(parts)):
            index = (first + offset) % len(parts)
            part = parts[index]
            if time.monotonic() < self.retry_after.get(part['id'],0):
                continue
            path = self.parts_dir/part['id']
            if (path/'SEARCH_TERMINATION.json').exists():
                if self.independent:
                    coverage = self.directory/'COVERAGE.json'
                    known = read(coverage).get('parts', {}) if coverage.is_file() else {}
                    if not known.get(part['id'], {}).get('termination'):
                        self.export(path)
                        self.record_coverage(path)
                continue
            self.cursor = (index+1) % len(parts)
            path.mkdir(parents=True, exist_ok=True)
            if self.independent:
                claim = (path/'OWNER.lock').open('a+b')
                try:
                    fcntl.flock(claim, fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    claim.close()
                    continue
                self.claim = claim
            try:
                if self._start_part(path, part, batch):
                    return True
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                if self.independent:
                    atomic(path/'BLOCK_STATUS.json',dict(status='WAITING_FOR_FIXED_INPUT',reason=str(error)))
                    self.record_coverage(path)
                self.retry_after[part['id']] = time.monotonic()+30
                if self.claim is not None:
                    self.claim.close()
                    self.claim = None
                if not self.independent:
                    raise
            except BaseException:
                if self.claim is not None:
                    self.claim.close()
                    self.claim = None
                raise
        self.recheck_at = time.monotonic()+30
        return False

    def _rotate_waiting_part(self):
        """A blocked block must not monopolize a whole node."""
        if self.part is None or not any(p['id'] != self.part.name and
                not (self.parts_dir/p['id']/'SEARCH_TERMINATION.json').exists() for p in self.parts):
            return
        state = self.part/'KAPPA_STATE.json'
        if not state.is_file() or not read(state).get('recovery_wait'):
            return
        database = self.part/'distributed'/'QUEUE.sqlite3'
        if database.is_file():
            from contextlib import closing
            try:
                with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True,timeout=1)) as connection:
                    live, = connection.execute("SELECT count(*) FROM jobs WHERE state IN ('ASSIGNED','STARTING','RUNNING')").fetchone()
            except sqlite3.OperationalError as error:
                if not sqlite_busy(error):
                    raise
                return  # Ownership was not established; leave this block running.
            if live:
                return
        self.request_stop()

    def _recover_fixed_input_failure(self, path, entry):
        """Archive a demonstrably pre-science block stop after its input is restored."""
        state = path/'KAPPA_STATE.json'
        if not self.independent or not state.is_file():
            return
        marker = path/'STOP_REQUESTED.json'
        if marker.is_file() and read(marker).get('reason') != 'PBS_BLOCK_HANDOFF':
            return
        stop = (read(state).get('stop') or {})
        message = str(stop.get('message',''))
        if stop.get('reason') != 'RUNTIME_EXCEPTION':
            return
        # A missing immutable Git commit is a technical input failure, not a
        # failed scientific evaluation. Only retry when the exact ref now resolves.
        import re
        match = re.search(r"'([0-9a-f]{40})\^\{commit\}'", message)
        if match:
            if subprocess.run(['git','-C',str(entry.parents[1]),'cat-file','-e',match.group(1)+'^{commit}'],
                              stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
                return
        elif 'not in the subpath' not in message or not path.resolve().is_relative_to(entry.parents[1].resolve()):
            return
        if any((path/'jobs').glob('*/*')) or any(path.rglob('SCIENTIFIC_STARTING.json')):
            return
        db = path/'distributed/QUEUE.sqlite3'
        if db.is_file():
            with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as connection:
                if connection.execute('SELECT 1 FROM jobs LIMIT 1').fetchone():
                    return
        archive = path/'technical-stop-history'/digest(stop)[:20]
        if archive.exists():
            return
        archive.mkdir(parents=True)
        for child in list(path.iterdir()):
            if child.name not in ('OWNER.lock','technical-stop-history'):
                child.rename(archive/child.name)
        atomic(path/'TECHNICAL_RECOVERY.json',dict(reason='FIXED_INPUT_RESTORED_BEFORE_SCIENCE',
                                                   archived=str(archive),original_stop=stop))

    def _controller_command(self, entry):
        """Apply a selected operational overlay without changing the fixed plan."""
        command = [self.plan['settings']['interpreter'], '-B']
        overlay = self.profile.get('pbs', {}).get('offline_controller_overlay')
        if not overlay:
            return command+[str(entry)]
        if overlay['base_source'] != self.plan['settings']['source']:
            raise ValueError('PBS controller overlay belongs to another fixed optimizer')
        files = overlay['modules']
        order = ('rpe_host_profiles', 'rpe_q3_resources', 'rpe_q3_store',
                 'rpe_preparation_bridge', 'rpe_optimizer_concurrent')
        if not {'rpe_q3_store','rpe_optimizer_concurrent'} <= set(files) <= set(order):
            raise ValueError('Unsupported PBS controller overlay modules')
        modules = []
        for name in order:
            if name not in files:
                continue
            item = files[name]
            path = Path(item['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('PBS controller overlay differs: '+name)
            modules.append((name,str(path)))
        bootstrap = (
            'import importlib.util,json,runpy,sys\n'
            'entry,root,modules=sys.argv[1:4]\n'
            'sys.argv=[entry,*sys.argv[4:]]\n'
            'sys.path.insert(0,root+"/src")\n'
            'import trottertracks\n'
            'for name,path in json.loads(modules):\n'
            '    spec=importlib.util.spec_from_file_location("trottertracks."+name,path)\n'
            '    module=importlib.util.module_from_spec(spec)\n'
            '    sys.modules[spec.name]=module\n'
            '    spec.loader.exec_module(module)\n'
            '    setattr(trottertracks,name,module)\n'
            'runpy.run_path(entry,run_name="__main__")')
        return command+['-c',bootstrap,str(entry),str(entry.parents[1]),json.dumps(modules)]

    def _start_part(self, path, part, batch):
        entry = self._entry()
        if self.plan['round'].get('hchain') and (path/'ROUND_RESULT.json').is_file():
            reasons = {r.get('stop_reason') for r in read(path/'ROUND_RESULT.json').get('methods',{}).values()}
            marker = path/'STOP_REQUESTED.json'
            handoff = marker.is_file() and read(marker).get('reason') == 'PBS_BLOCK_HANDOFF'
            blocked = {'UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER'}
            if not handoff:
                blocked.add('USER_REQUESTED_STOP')
            if reasons.intersection(blocked):
                atomic(path/'BLOCK_STATUS.json',dict(status='REQUIRES_RESUME_REVIEW', reasons=sorted(reasons)))
                self.record_coverage(path)
                if self.claim is not None:
                    self.claim.close()
                    self.claim = None
                return False
        self._recover_fixed_input_failure(path, entry)
        problem, spec = part_documents(self.plan, part, self.profile, path)
        for name, data in (('PROBLEM.json', problem), ('ROUND.json', spec)):
            file = path/name
            if file.exists() and read(file) != data:
                previous = read(file)
                # Existing blocks may have been created before live local
                # readiness was connected. This changes no physical inputs.
                if self.independent and name == 'ROUND.json':
                    migrated = deepcopy(previous)
                    settings = migrated['executor']['distributed']
                    if 'host_eligibility' not in settings:
                        settings['host_eligibility'] = {'schema':'RPE_Q3_HOST_ELIGIBILITY_V1','jobs':{}}
                    current = data['executor']['distributed']
                    if current.get('preparation', {}).get('local_shared_git_inputs') is True:
                        settings['preparation']['local_shared_git_inputs'] = True
                    # The new allocation owns a fresh sampler path and
                    # transport. Neither is part of the scientific contract.
                    if current.get('local_worker', {}).get('storage_limit', {}).get('parent_compute_sample'):
                        settings['worker_script'] = current['worker_script']
                        for field in ('local_worker',):
                            settings[field]['storage_limit']['parent_compute_sample'] = current[field]['storage_limit']['parent_compute_sample']
                        settings['preparation_profiles']['local']['storage_limit']['parent_compute_sample'] = current['preparation_profiles']['local']['storage_limit']['parent_compute_sample']
                    # The same fixed H-chain package may resume under an
                    # explicitly selected CPU execution profile. No science,
                    # preparation, budget or ownership fields may differ.
                    if spec.get('hchain') and current.get('local_worker',{}).get('q3_cpu_policy'):
                        for old_profile,new_profile in ((settings['local_worker'],current['local_worker']),
                                (settings['preparation_profiles']['local'],current['preparation_profiles']['local'])):
                            old_profile['q3_cpu_policy'] = deepcopy(new_profile['q3_cpu_policy'])
                    if spec.get('hchain') and (adapter := (self.profile.get('pbs', {}).get('offline_controller_overlay') or {}).get('q3_adapter')):
                        for old_profile,new_profile in ((settings['local_worker'],current['local_worker']),
                                (settings['preparation_profiles']['local'],current['preparation_profiles']['local'])):
                            if any(new_profile.get(key) != value for key,value in adapter.items()):
                                raise ValueError('Selected Q3 adapter differs from the resumed profile')
                            old_profile.update(deepcopy(adapter))
                    if migrated == data:
                        archive = path/'technical-stop-history'/('ROUND.before-transport-'+digest(previous)[:12]+'.json')
                        archive.parent.mkdir(exist_ok=True)
                        if not archive.exists():
                            atomic(archive, previous)
                        atomic(file, data)
                    else:
                        raise ValueError('PBS continuation part inputs changed')
                else:
                    raise ValueError('PBS continuation part inputs changed')
            if not file.exists():
                atomic(file, data)
        marker = path/'STOP_REQUESTED.json'
        if marker.exists():
            if self.independent and read(marker).get('reason') != 'PBS_BLOCK_HANDOFF':
                self.claim.close()
                self.claim = None
                self.recheck_at = time.monotonic()+30
                return False
            marker.unlink()
        command = self._controller_command(entry)+[
                   '--problem', str(path/'PROBLEM.json'), '--round', str(path/'ROUND.json'),
                   '--run-dir', str(path)]
        if not spec.get('hchain'):
            command += ['--batch', str(min(batch, self.slots))]
        if not self.independent:
            command += ['--max-cycles', '1']
        with (path/'OFFLINE.log').open('ab') as log:
            kwargs = {'pass_fds': (self.claim.fileno(),)} if self.claim is not None else {}
            self.process = self.popen(command, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, env=self.environment, **kwargs)
        self.part = path
        if self.independent:
            atomic(path/'BLOCK_STATUS.json',dict(status='RUNNING',host_id=self.profile['host_id']))
        self.next_export_at = time.monotonic()+20
        return True

    def export(self, part):
        """Publish fixed completed results without racing another allocation."""
        if self.independent:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory/'EXPORT.lock').open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return self._export(part)
        return self._export(part)

    def _export(self, part):
        manifest = self.directory/'RESULTS.json'
        records = read(manifest) if manifest.exists() else {}
        new = 0
        for outcome in (part/'jobs').glob('*/*/OUTCOME.json'):
            job = outcome.parent
            if job.parent.name.startswith('PREPARATION:'):
                if not self.independent:
                    continue
                # Preserve the original stage in the receipt; use an ordinary
                # filename component for the existing bulk gateway.
                method = job.parent.name.replace(':','-')
            else:
                method = job.parent.name
            names = ('REQUEST.json', 'OUTCOME.json', 'PROCESS_RETURN.json')
            if any(not (job/name).is_file() for name in names):
                continue
            receipt = read(job/'PROCESS_RETURN.json')
            request = read(job/'REQUEST.json')
            if (receipt.get('returncode') != 0 or receipt.get('signal') is not None
                    or request.get('request_id') != job.name):
                continue
            key = f'{method}/{job.name}'
            if key in records:
                continue
            origin = None
            if self.independent and method.startswith('PREPARATION-'):
                from contextlib import closing
                db = part/'distributed'/'QUEUE.sqlite3'
                if not db.is_file():
                    continue
                try:
                    with closing(sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True,timeout=1)) as connection:
                        row = connection.execute('SELECT request_sha256,contract,attempt_id,completion_sha256,state '
                            'FROM jobs WHERE job_id=?',(job.name,)).fetchone()
                except sqlite3.OperationalError as error:
                    if not sqlite_busy(error):
                        raise
                    continue  # The next export will recover the intact receipt.
                if not row or row[4] != 'ACCEPTED' or not row[2] or row[0] != digest(request):
                    continue
                origin = dict(request_sha256=row[0], contract=json.loads(row[1]), attempt_id=row[2],
                              completion_sha256=row[3], host_id=self.profile['host_id'], part=part.name)
            destination = self.directory/'jobs'/key
            destination.mkdir(parents=True, exist_ok=True)
            for name in names:
                temporary = destination/(name+'.tmp')
                try:
                    os.link(job/name, temporary)
                except OSError:
                    shutil.copyfile(job/name, temporary)
                temporary.replace(destination/name)
            records[key] = dict(part=part.name, files={name: (destination/name).stat().st_size
                                                       for name in names})
            if self.independent and method.startswith('PREPARATION-'):
                outcome_value = read(outcome)
                artifacts = outcome_value.get('artifacts', {})
                cache = part/'distributed'/'artifacts'
                if not all((cache/sha[:2]/sha).is_file() for sha in artifacts.values()):
                    del records[key]
                    continue
                records[key]['artifacts'] = {sha: (cache/sha[:2]/sha).stat().st_size
                                            for sha in set(artifacts.values())}
                records[key]['origin'] = origin
                atomic(destination/'ORIGIN.json',origin)
            new += 1
        if new:
            atomic(manifest, records)
            self.revision += new
        return new

    def record_coverage(self, part):
        if self.independent:
            with (self.directory/'EXPORT.lock').open('a+b') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                return self._record_coverage(part)
        return self._record_coverage(part)

    def _record_coverage(self, part):
        path = self.directory/'COVERAGE.json'
        status = read(path) if path.exists() else dict(changes=0, parts={})
        marker = part/'SEARCH_TERMINATION.json'
        db = part/'distributed'/'QUEUE.sqlite3'
        counts = {}
        if db.is_file():
            from contextlib import closing

            try:
                with closing(sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True,timeout=1)) as connection:
                    for method, state, total in connection.execute("SELECT method,state,count(*) FROM jobs GROUP BY 1,2"):
                        counts.setdefault(method, {})[state] = total
            except sqlite3.OperationalError as error:
                if not sqlite_busy(error):
                    raise
                return  # Keep the last valid coverage; do not end the worker.
        observed = dict(counts=counts, termination=read(marker) if marker.exists() else None)
        if self.independent:
            status_path=part/'BLOCK_STATUS.json'
            if status_path.is_file():
                observed['block_status']=read(status_path)
            reports = self.directory/'reports'/part.name
            reports.mkdir(parents=True, exist_ok=True)
            # A compact per-block map / summary is sufficient for fleet
            # status. Original results remain available by exact job identity.
            for name in ('SCOPE_MAP.json','ROUND_RESULT.json'):
                source = part/name
                if not source.is_file():
                    continue
                target = reports/name
                if target.exists() and target.stat().st_mtime_ns == source.stat().st_mtime_ns:
                    continue
                temporary = target.with_suffix('.tmp')
                if name == 'ROUND_RESULT.json':
                    source_mtime=source.stat().st_mtime_ns
                    atomic(temporary,compact_block_report(read(source)))
                    os.utime(temporary,ns=(source_mtime,source_mtime))
                else:
                    try:
                        temporary.unlink(missing_ok=True)
                        os.link(source,temporary)
                    except OSError:
                        shutil.copy2(source,temporary)
                temporary.replace(target)
            observed['reports'] = {name: (reports/name).stat().st_size
                                   for name in ('SCOPE_MAP.json','ROUND_RESULT.json') if (reports/name).is_file()}
            observed['report_revision'] = max(((reports/name).stat().st_mtime_ns
                                               for name in observed['reports']), default=0)
        if status['parts'].get(part.name) != observed:
            status['parts'][part.name] = observed
            status['changes'] += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic(path, status)
            self.revision += 1


def advance_independent(worker, control):
    """The node-local search is primary, not a fallback for a broken SSH."""
    search = worker.offline_search
    best = control.get('partition_best')
    if best is not None and best != getattr(search,'central_best',None):
        search.directory.mkdir(parents=True,exist_ok=True)
        atomic(search.directory/'CENTRAL_SHORTLIST.json',best)
        search.central_best=best
    marker = worker.directory/'INDEPENDENT_ACTIVE'
    active = False
    coverage = search.directory/'COVERAGE.json'
    if search.process is None and not search.pending() and coverage.is_file() and all_parts_finished(
            search.plan, read(coverage).get('parts',{})):
        worker.offline_active = False
        marker.unlink(missing_ok=True)
        return False
    if search.process is None:
        # No independent process overlaps previously assigned central work.
        for attempt, assignment in tuple(worker.records.items()):
            state = read(attempt/'WORKER_STATE.json')
            if state['state'] in ('STARTING', 'RUNNING', 'UNKNOWN'):
                state = worker._reconcile(attempt, assignment)
            active |= (state['state'] in ('ASSIGNED','STARTING','RUNNING')
                       or state['state'] == 'UNKNOWN' and not state.get('detail', {}).get('resource_released'))
    stopping = bool(control.get('stop') or not worker.accepting()
                    or worker.remaining() < worker.drain_seconds + 2*worker.minimum_task_seconds)
    report = getattr(worker, 'resource_report', None)
    storage = (report or {}).get('storage') or {}
    # Independent mode intentionally suppresses the ordinary poll loop.
    # Renew a blocked storage offer here, otherwise successful asynchronous
    # cleanup/sampling can never become visible to a waiting package.
    now = time.monotonic()
    if (search.process is None and not active and
            (report is None or not storage.get('accepting_new_jobs', True)) and
            now >= getattr(search, 'next_resource_refresh', 0.)):
        search.next_resource_refresh = now + 3.
        report = worker._resource_offer()
        storage = (report or {}).get('storage') or {}
    try:
        busy = search.advance(disconnected=False, runnable=not active and
                              storage.get('accepting_new_jobs', True),
                              batch=max(1, worker.slots), stop=stopping)
    except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
        from .rpe_q3_worker import event
        search.recheck_at = time.monotonic()+30
        event(worker.profile, 'PBS_INDEPENDENT_BLOCK_PENDING', reason=str(error))
        busy = search.process is not None
    worker.offline_active = busy
    if busy:
        marker.touch()
    else:
        marker.unlink(missing_ok=True)
    if search.process is not None:
        return True
    # Until the first usable storage reading, keep the independent backlog
    # primary. If no block is currently claimable, normal pull is assistance.
    return busy or (search.pending() and not active and not stopping
                    and not storage.get('accepting_new_jobs', True))


def compact_block_report(report):
    """Send the search view, not the allocation-by-allocation report repeatedly.

    Keep the allocation outcomes and shot choices. Omit repeated price
    evidence/quote bodies, which are transferred as exact original receipts.
    The full local report stays in the durable block.
    """
    row_fields=('configuration_id','source_configuration_id','method','configuration','M','best','comparison_best',
                'mse_feasible','cost_status','minimum_status','price_status','needs','required_rounds',
                'unresolved_competitors','allocations','configuration_source','signal_source','price_source',
                'comparison_delta_RZ','paper_comparison_pending','promising')
    method_fields=('method','best','stop_reason','new_calls','evaluated_count','budget','evaluation_summary')
    result={key:report[key] for key in ('schema','optimizer_source','scope_owner','ready','price_pruned','inherited') if key in report}
    result['methods']={}
    for method,entry in report.get('methods',{}).items():
        projected={key:entry[key] for key in method_fields if key in entry}
        projected['by_configuration']=[{key:row[key] for key in row_fields if key in row}
                                      for row in entry.get('by_configuration',[])]
        if 'scaled_shots' in entry:
            projected['scaled_shots']=entry['scaled_shots']
        projected['partition_report_status']='COMPACT_VIEW_EXACT_Q3_RECEIPTS_EXPORTED_SEPARATELY'
        result['methods'][method]=projected
    return result


def matches_block(part, method, coordinates):
    if part['method'] != method:
        return False
    if part['value'] is None:
        return True
    from .rpe_kappa_schedule import _frac
    key = {'width':'delta_pf','L_D_values':'L_D','p_values':'p',
           'delta_over_h_values':'delta_over_h'}[part['axis']]
    observed = coordinates.get(key)
    return observed is not None and _frac(observed) == _frac(part['value'])


def merge_partition_maps(plan, central, remote):
    """Join logical coordinates, retaining unsupported/unstarted cells."""
    from collections import Counter
    from .rpe_scope_map import build_scope_map
    from .rpe_scaled_shots import settings
    problem = deepcopy(plan['problem'])
    for method, change in plan['round'].get('method_overrides', {}).items():
        change = deepcopy(change)
        if 'generation' in change:
            problem['methods'][method].setdefault('generation',{}).update(change.pop('generation'))
        problem['methods'][method].update(change)
    from .rpe_schedule_search import requested_widths
    for method, widths in (requested_widths(problem,plan['round']) or {}).items():
        problem['methods'][method]['requested_delta_pf_values'] = [w['raw'] for w in widths]
    search = dict(problem.get('search',{}),**plan['round'].get('search_overrides',{}))
    horizons = settings(search.get('scaled_shots'))['M']
    methods={m:problem['methods'][m] for m in plan['round']['methods']}
    document = build_scope_map(methods, horizons, mapper_version={
        m:(spec.get('generation') or {}).get('kind') for m,spec in methods.items()})
    def index_map(source):
        return {(method,digest(row['coordinates'])):row
                for method,info in source.get('methods',{}).items() for row in info.get('rows',[])}
    indexes={key:index_map(value) for key,value in remote.items()}
    indexes['central']=index_map(central or {})
    package_for={p['id']:unit['id'] for unit in pbs_units(plan) for p in unit_members(plan,unit)}
    for method,info in document['methods'].items():
        for i,row in enumerate(info['rows']):
            parts = [part for part in plan['parts'] if matches_block(part,method,row['coordinates'])]
            if len(parts) != 1:
                raise ValueError('Logical coordinate has ambiguous independent ownership')
            part = parts[0]
            owner_key = 'central' if part['owner']=='central' else package_for[part['id']]
            observed = indexes.get(owner_key,{}).get((method,digest(row['coordinates'])))
            if observed is not None:
                row = deepcopy(observed)
            elif part['owner']=='pbs':
                row['status'] = 'PBS_BLOCK_PENDING'
            row.update(block_id=part['id'],owner=part['owner'])
            info['rows'][i]=row
        counts=Counter(row['status'] for row in info['rows'])
        resolved=counts['EVALUATED_FEASIBLE']+counts['EVALUATED_NO_PASS']
        info.update(counts=dict(counts),resolved=resolved,unresolved=info['requested']-resolved,
                    mse_pass_price_required=sum(bool(row.get('price_required')) for row in info['rows']),
                    mapper_reason_counts=dict(Counter(row['reason_kind'] for row in info['rows'] if row.get('reason_kind'))))
    return document


def merge_partition_reports(plan, central, remote, coverage, scope_map=None):
    """Aggregate reported feasible minima; do not infer a global proof."""
    from .rpe_kappa_schedule import _frac
    result = deepcopy(central or (next(iter(remote.values()), {})))
    result['schema'] = 'RPE_OPTIMIZER_ROUND_RESULT_V1'
    result['round'] = plan['round']
    result['partitioned_search'] = dict(parts=plan['parts'],packages=pbs_units(plan),coverage=coverage,
                                       pbs_complete=all_parts_finished(plan,coverage))
    methods = {}
    def best(rows):
        rows = [row for row in rows if isinstance(row,dict) and row.get('cost') is not None]
        return min(rows,key=lambda row:_frac(row['cost'])) if rows else None
    for method in plan['round']['methods']:
        items = []
        if central and method in central.get('methods',{}):
            items.append(central['methods'][method])
        parts = [unit for unit in pbs_units(plan)
                 if any(p['method']==method for p in unit_members(plan,unit))]
        items += [remote[p['id']]['methods'][method] for p in parts
                  if p['id'] in remote and method in remote[p['id']].get('methods',{})]
        if not items:
            methods[method] = dict(stop_reason='PBS_BLOCKS_PENDING',by_configuration=[],new_calls=0)
            continue
        combined = deepcopy(items[0])
        rows = {}
        for item in items:
            for row in item.get('by_configuration',[]):
                key=(row.get('configuration_id'),row.get('M'))
                if key not in rows or (row.get('best') and not rows[key].get('best')):
                    rows[key]=row
        combined['by_configuration']=list(rows.values())
        combined['new_calls']=sum(item.get('new_calls',0) for item in items)
        combined['partition_stop_reasons']=[item.get('stop_reason') for item in items]
        if 'budget' in combined:
            combined['budget']=dict(combined['budget'],max_new_q3=plan['round']['max_new_q3_per_method'][method],
                spent_before=sum((item.get('budget') or {}).get('spent_before',0) for item in items),
                new_calls=combined['new_calls'])
        combined['best']=best([item.get('best') for item in items])
        pending=any(not coverage.get(p['id'],{}).get('termination') for p in parts)
        if pending:
            combined['stop_reason']='PBS_BLOCKS_PENDING'
        scaled=[item['scaled_shots'] for item in items if item.get('scaled_shots')]
        if scaled:
            merged=deepcopy(scaled[0])
            for name in ('M12_best','all_M_best','comparison_only_best'):
                merged[name]=best([entry.get(name) for entry in scaled])
            for name in ('M12_minimum_status','all_M_minimum_status'):
                statuses={entry.get(name) for entry in scaled}
                # Partition-local proof is not a claim about unreturned blocks.
                if pending or len(statuses)!=1:
                    merged[name]='PROVISIONAL_EVALUATED_MINIMUM'
            compare, winner=merged['comparison_only_best'],merged['all_M_best']
            if compare and winner:
                delta=_frac(compare['cost'])-_frac(winner['cost'])
                merged['comparison_only_delta_RZ']=dict(numerator=delta.numerator,denominator=delta.denominator)
            else:
                merged['comparison_only_delta_RZ']=None
            if scope_map and method in scope_map['methods']:
                info=scope_map['methods'][method]
                merged['scope_coverage']=dict(merged.get('scope_coverage',{}),
                    requested=info['requested'],represented=sum(bool(r.get('configuration_id')) for r in info['rows']),
                    missing_count=sum(not bool(r.get('configuration_id')) for r in info['rows']),
                    logical_status_counts=info['counts'],scope_map='SCOPE_MAP.json')
            order = [row for entry in scaled for row in entry.get('estimated_order',[])]
            if order:
                order.sort(key=lambda row:(_frac(row['cost']),row['M'],row['configuration_id']))
                leading = {}
                for row in order:
                    row['leads_horizon'] = _frac(row['cost']) == leading.setdefault(row['M'],_frac(row['cost']))
                merged['estimated_order'] = order
            unpriced = {(row['configuration_id'],row['M']):row for entry in scaled
                        for row in entry.get('unpriced_feasible',[])}
            if unpriced:
                merged['unpriced_feasible']=list(unpriced.values())
            combined['scaled_shots']=merged
        methods[method]=combined
    result['methods']=methods
    if plan['round'].get('hchain'):
        reports = [central or {}, *remote.values()]
        result['ready'] = sorted({cid for report in reports for cid in report.get('ready',[])})
        result['price_pruned'] = sorted({cid for report in reports for cid in report.get('price_pruned',[])})
    return result


def publish_partition_view(run, *, locked=False):
    run=Path(run)
    if not locked:
        with (run/'ROUND_REPORT.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return publish_partition_view(run,locked=True)
    plan=read(run/'PBS_PARTITION_PLAN.json')
    history=run/'distributed'/'pbs'/'offline-history'
    def optional(path):
        return read(path) if path.is_file() else None
    remote_maps={}
    remote_reports={}
    coverage={}
    for marker in (history/'coverage').glob('*.json'):
        for key,row in read(marker).get('parts',{}).items():
            old=coverage.get(key,{})
            if row.get('report_revision',0)>=old.get('report_revision',0):
                coverage[key]=row
    report_paths = [(part['id'],history/'reports'/part['id']) for part in pbs_units(plan)]
    files=[run/'SCOPE_MAP_CENTRAL.json',run/'ROUND_RESULT_CENTRAL.json']
    files.extend(root/name for _,root in report_paths for name in ('SCOPE_MAP.json','ROUND_RESULT.json'))
    revision = [[str(path),path.stat().st_mtime_ns,path.stat().st_size] for path in files if path.is_file()]
    revision += [[key,row.get('termination')] for key,row in sorted(coverage.items()) if row.get('termination')]
    status=optional(run/'PBS_PARTITION_STATUS.json') or {}
    if status.get('publication_revision')==revision:
        atomic(run/'PBS_PARTITION_STATUS.json',dict(status,coverage=coverage,
                pbs_complete=all_parts_finished(plan,coverage)))
        return
    for part,path in report_paths:
        if value:=optional(path/'SCOPE_MAP.json'):
            remote_maps[part]=value
        if value:=optional(path/'ROUND_RESULT.json'):
            remote_reports[part]=value
    local_map=optional(run/'SCOPE_MAP_CENTRAL.json')
    scope_map=None
    if local_map or remote_maps:
        scope_map=merge_partition_maps(plan,local_map,remote_maps)
        atomic(run/'SCOPE_MAP.json',scope_map)
    local_report=optional(run/'ROUND_RESULT_CENTRAL.json')
    winners={}
    if local_report or remote_reports:
        result=merge_partition_reports(plan,local_report,remote_reports,coverage,scope_map)
        atomic(run/'ROUND_RESULT.json',result)
        write_partition_table(run/'PER_METHOD.csv',result['methods'])
        winners={method:(entry.get('scaled_shots') or {}).get('all_M_best')
                 for method,entry in result['methods'].items()}
        winners={method:{k:row[k] for k in ('configuration_id','M','cost') if k in row}
                 for method,row in winners.items() if row}
    atomic(run/'PBS_PARTITION_STATUS.json',dict(parts=plan['parts'],packages=pbs_units(plan),coverage=coverage,
            pbs_complete=all_parts_finished(plan,coverage),best_by_method=winners,publication_revision=revision))


def write_partition_table(path, methods):
    """Same user-facing method columns, with exact receipts kept in JSON."""
    import csv
    from .rpe_kappa_schedule import _frac
    def value(row,name):
        return row.get(name,'') if row else ''
    def bound(row,name):
        raw=value(row,name)
        if isinstance(raw,dict) and 'lower_30' in raw:
            return raw['lower_30' if name.endswith('lower') else 'upper_30']
        try:return '%.16e'%float(_frac(raw))
        except (ValueError,TypeError,KeyError):return raw
    temporary=Path(path).with_suffix('.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['method','best_candidate','configuration','N_X','N_Y','status','MSE_lower','MSE_upper',
                         'cost_RZ','stop_reason','next_Q3_candidate','next_Q3_N_X','next_Q3_cost_RZ',
                         'ready_candidates','not_ready_candidates','evaluated'])
        for method,entry in methods.items():
            b,p=entry.get('best'),entry.get('pending_request')
            writer.writerow([method,value(b,'candidate'),json.dumps(value(b,'configuration')),
                ' '.join(map(str,value(b,'N_X'))),' '.join(map(str,value(b,'N_Y'))),
                value(b,'status'),bound(b,'MSE_lower'),bound(b,'MSE_upper'),value(b,'cost_display'),
                entry.get('stop_reason'),value(p,'candidate'),' '.join(map(str,value(p,'N_X'))),
                value(p,'cost'),len(entry.get('ready_candidates',[])),len(entry.get('not_ready_candidates',[])),
                entry.get('evaluated_count',len(entry.get('by_configuration',[])))])
    temporary.replace(path)
