"""Wide PR kappa / split / requested PF width / shot search, resumable.

With search_overrides.requested_widths={PR_PAULI:[...], PR_DF:[...]}, widths
are absolute exact requested values. The existing method mapper supplies each
round's q and actual width. Without that declaration the historical Pauli-only
round entry and saved round05 behavior remain unchanged.

One cycle
  1. plan: rpe_schedule_search.plan_wide over every (L_D, width) family -- schedules (exact kappa intervals), not kappa values;
     saved / evaluated schedules are reused, so no schedule is evaluated twice;
  2. inputs: a family with a prepared base goes through A's preparation entry (configuration), D2 PriceRules MODEL and the
     C/A common calibration-request entry (round04 driver); a schedule whose proven price floor (one shot per round and
     axis) is not below the verified best is pruned without science; a family without a prepared base (new L_D) goes to
     D2's completion API (``Completion.complete`` with the selected L_D, width and absolute kappa); a saved READY record of
     the same design is reused, never re-requested;
  3. signals (A entry, missing rounds only) -> B intake -> route registry (round04 driver);
  4. V2 batch (coarse-first FINE class, HYBRID slots) on all ready schedules; completed Q3 jobs are reused by job key;
  5. the verified-all best (every identity-checked completed job, adopted or not) and the rule-adopted best are both kept.
Stops when the plan proposes nothing, nothing is pending and V2 has no non-small candidate (bases per family in the state).

    python run_rpe_schedule_search.py --problem P --round R --run-dir D [--batch 64] [--plan-only]
``--plan-only`` writes the first decision (proposals, configurations, prices, price floors, completion requests) and stops
before any signal, calibration or Q3 (metadata only).
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache, wraps
import hashlib
import json
import os
import subprocess
from pathlib import Path
import sys
import time

# The central resume launcher can explicitly keep the original run's source
# checkout for imports, Git locators and providers when selecting a new planner.
ROOT = Path(globals().get('FIXED_RUN_CODE_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402
from trottertracks import rpe_schedule_search as W  # noqa: E402
import run_rpe_kappa_neighbourhood as K  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


@lru_cache(maxsize=256)
def read_fixed_document(commit, path):
    return json.loads(subprocess.check_output(
        ['git', 'show', commit + ':' + path], cwd=ROOT, stderr=subprocess.PIPE))


def read_loc(loc):
    import re

    commit, path = loc['commit'], loc['path']
    if re.fullmatch('[a-f0-9]{40}|[a-f0-9]{64}', commit):
        x = deepcopy(read_fixed_document(commit, path))
    else:
        x = json.loads(subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT))
    for k in loc.get('selector', []):
        x = x[k]
    return x


def equivalent_df_descriptor(config, base):
    """Recognize the original H4 descriptor runner and its input-path port."""
    import re
    from trottertracks.pr_df_descriptor_extension import validate

    # The port moves the input paths and retains the fixed loader, hashed inputs
    # and matrix_only_tail_coefficients. Do not extend this to unknown runners.
    runners = {'194e00a8a058a7a009e186bcd03b38c7329ec2f7',
               'b0c04ae1915661af8de4f34957304e2cb0e7aed9'}
    compared = []
    for candidate in (config, base):
        split = candidate['spec']['split']
        loc = split.get('lambda_source')
        if (not isinstance(loc, dict) or not re.fullmatch('[0-9a-f]{40}', loc.get('commit', ''))
                or not loc.get('path') or set(loc) - {'commit', 'path', 'selector'}):
            return False
        try:
            document = read_loc(loc)
            descriptor = validate(document, loc, candidate)
            caller = document['execution_source']
            if (caller.get('path') != 'scripts/derive_pr_df_candidate_descriptor.py'
                    or caller.get('commit') not in runners
                    or KS._frac(split['lambda_R']) != Fraction(descriptor['lambda_R'])):
                return False
            compared.append({k: v for k, v in document.items() if k not in ('resources', 'execution_source')})
        except (ValueError, TypeError, KeyError, OSError, subprocess.CalledProcessError):
            return False
    return json.dumps(compared[0], sort_keys=True) == json.dumps(compared[1], sort_keys=True)


def write_width_report(run, cli, spec, result, optimizer_source, *, owner=None):
    import fcntl

    path = run/'ROUND_REPORT_BOUNDS_V1.json'
    try:
        document = O.load_json(path) if path.is_file() else {}
        cache = document.get('bounds', {}) if document.get('source') == optimizer_source else {}
    except (OSError, ValueError, AttributeError):
        cache = {}
    if not isinstance(cache, dict) or len(cache) > 16384:
        cache = {}
    with cli.round_phase(run, 'WIDTH_RESULT_BOUNDS'):
        summary = cli.slim(dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1',
            round=spec, methods=result, optimizer_source=optimizer_source),
            **({'bound_cache': cache} if 'bound_cache' in (cli.slim.__kwdefaults__ or {}) else {}))
    with cli.round_phase(run, 'WIDTH_RESULT_JSON'):
        # This report is a machine-readable checkpoint; keep every field and
        # exact value, without repeatedly writing indentation for each row.
        summary = json.dumps(O.jsonable(summary), ensure_ascii=False, separators=(',', ':'))
    with (run/'ROUND_REPORT.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if owner is not None and O.load_json(run/'NORMAL_DRIVER.json').get('pid') != owner:
            return
        with cli.round_phase(run, 'WIDTH_RESULT_WRITE'):
            path = run/'ROUND_RESULT.json'
            staging = path.with_suffix('.json.tmp')
            with staging.open('w') as stream:
                stream.write(summary)
                stream.write('\n')
            staging.replace(path)
        del summary
        if cache:
            from trottertracks.rpe_q3_coordinator import atomic_json
            atomic_json(run/'ROUND_REPORT_BOUNDS_V1.json', dict(source=optimizer_source, bounds=cache))
        with cli.round_phase(run, 'WIDTH_METHOD_TABLE'):
            path = run/'PER_METHOD.csv'
            staging = path.with_suffix('.csv.tmp')
            cli.table(result, staging)
            staging.replace(path)


def bind_pf_continuation_release(distributed, provider, root):
    """Bind the additional PF producer to the hosts with staged deployments."""
    from trottertracks.rpe_release_deployment import manifest_reference, read
    if not distributed.get('required_release'):
        return
    manifest = read(distributed['release_manifest'])
    parent = manifest_reference(manifest)
    if parent != distributed['required_release']:
        raise ValueError('PF continuation parent release differs')
    layout = read(distributed['local_deployment_layout'])
    host = distributed['local_worker']['host_id']
    if layout['host_id'] != host:
        raise ValueError('PF continuation local layout differs')
    [base] = [name for name, component in manifest['components'].items()
              if component['source'] == provider['base_source'] and
              name in manifest['stage_components']['compile']]
    component = 'pf_price_recheck_' + provider['source']
    remote = provider.get('host_deployments', {})
    relays = {worker['host_id']: worker for worker in distributed.get('relay', {}).get('workers', [])}
    pbs = distributed.get('pbs') or {}
    pbs_workers = {worker['host_id']: worker for worker in pbs.get('workers', [])}
    if host in remote or not set(remote) <= (relays.keys() | pbs_workers.keys()):
        raise ValueError('PF continuation deployment names an unconfigured remote host')
    manifest.update(parent_release=parent, release_id=parent['id'] + '-' + component)
    manifest['components'][component] = dict(manifest['components'][base],
                                             source=provider['source'], hosts=[host])
    layout['component_roots'][component] = str(root)
    for stage in ('signal', 'compile', 'audit'):
        allowed = manifest['stage_components'][stage]
        if base in allowed:
            allowed.append(component)
    for section in ('runtimes', 'input_files'):
        for rule in layout.get(section, {}).values():
            if base in rule.get('components', []):
                rule['components'].append(component)
    references = {host: manifest_reference(manifest)}
    for host_id, deployment in remote.items():
        successor = deepcopy(manifest)
        successor['release_id'] += '-' + host_id
        successor['components'][component]['hosts'] = [host_id]
        reference = manifest_reference(successor)
        if deployment['reference'] != reference:
            raise ValueError('PF continuation host release differs: '+host_id)
        references[host_id] = reference
        configured = O.load_json(Path(deployment['controller_profile']))
        if configured['host_id'] != host_id or not any(
                rule.get('source_commit') == provider['source'] and rule['provider'].startswith('PF_')
                for rule in configured.get('preparation_profiles', [])):
            raise ValueError('PF continuation host lacks its verified preparation profile: '+host_id)
        paths = {key: deployment[key] for key in ('manifest', 'layout')}
        if host_id in relays:
            worker = relays[host_id]
            if not worker.get('deployment'):
                raise ValueError('PF continuation requires the base host deployment: '+host_id)
            distributed['preparation_profiles'][host_id] = configured
            worker['profile'] = deployment['profile']
            worker.setdefault('task_deployments', {})[reference['id']] = dict(paths, entry=deployment['entry'])
        else:
            worker = pbs_workers[host_id]
            if not (worker.get('deployment') or pbs.get('deployment')):
                raise ValueError('PF continuation requires the base PBS deployment: '+host_id)
            distributed.setdefault('pbs_pinned_settings', deepcopy(pbs))
            chosen = [rule for rule in configured['preparation_profiles']
                      if rule.get('source_commit') == provider['source']]
            runtime = distributed['preparation_profiles'][host_id]
            if not isinstance(runtime, dict):
                runtime = O.load_json(Path(runtime))
                distributed['preparation_profiles'][host_id] = runtime
            for profile in (runtime, worker['profile']):
                profile['preparation_profiles'].extend(deepcopy(chosen))
            estimates = pbs['preparation_task_seconds']
            # Reuse the existing conservative bound and resource reservation
            # for the same validated producer stage and selected-axis subset.
            estimates.extend([dict(rule, id=rule['id']+'-price-recovery', source=provider['source'])
                              for rule in estimates if rule['source'] == provider['base_source']
                              and rule['provider'] in {choice['provider'] for choice in chosen}])
            worker.setdefault('task_deployments', {})[reference['id']] = paths
    distributed['local_pf_continuation'] = dict(source=provider['source'], manifest=manifest, layout=layout,
                                                references=references)


def selected_worker_policy(spec):
    selected = globals().get('FIXED_RUN_WORKER_TRIAL') or {}
    # Existing frozen runs predate the common-release policy API. Their
    # explicitly selected controller still uses the run's original policy.
    if not spec.get('release_worker_policy'):
        return dict(selected)
    from trottertracks.rpe_release_registry import common_worker_policy
    common = common_worker_policy(spec['release_worker_policy'])
    return dict(common, **selected)


def apply_worker_trial(spec, trial, *, run_dir=None):
    """Install the selected execution policy in memory after fixed-input checks."""
    import importlib.util

    distributed = spec['executor']['distributed']
    if binding := trial.get('pf_controller_runtime'):
        if run_dir is None:
            raise ValueError('PF controller runtime requires a run directory')
        run_path = Path(run_dir).resolve()
        if 'run_directory' in binding:
            matches = Path(binding['run_directory']).resolve() == run_path
        else:
            matches = bool(binding.get('run_root')) and run_path.parent == Path(binding['run_root']).resolve()
        if not matches:
            raise ValueError('PF controller runtime belongs to a different run or release root')
        for name, expected in binding['files'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                raise ValueError('Fixed PF controller runtime differs: '+name)
        settings = spec['pf_completion']
        if settings['interpreter'] != binding['replaces']:
            raise ValueError('PF controller runtime predecessor differs')
        settings['interpreter'] = binding['interpreter']
    if selection := trial.get('df_max_width'):
        code_root = Path(__file__).resolve().parents[1]
        source = selection['source']
        for name in ('src/trottertracks/pr_df_max_width.py', 'src/trottertracks/pr_rpe_rmse.py',
                     'src/trottertracks/rpe_scope_map.py', 'src/trottertracks/pr_rpe_signal_plan.py',
                     'scripts/run_pr_df_candidate_signal_jobs.py'):
            if subprocess.check_output(['git', 'show', source+':'+name], cwd=ROOT) != (code_root/name).read_bytes():
                raise ValueError('Selected max-width mapper differs: '+name)
        settings = spec['candidate_completion']
        settings['df_max_width_source'] = source
        settings['df_max_width_code_root'] = str(code_root)
        settings.setdefault('saved_df_descriptors', {})['0'] = selection['endpoint_descriptor']
        if price_source:=selection.get('price_helper_source'):
            name='src/trottertracks/pr_df_max_width_price.py'
            if subprocess.check_output(['git','show',price_source+':'+name],cwd=ROOT)!=(code_root/name).read_bytes():
                raise ValueError('Selected max-width price metadata helper differs')
            settings['df_max_width_price_helper']=price_source
        if selection.get('local_adapter'):
            adapter=selection['local_adapter']
            from trottertracks.rpe_host_profiles import validate_adapter_profile
            validate_adapter_profile(adapter)
            for rule in distributed['local_worker'].get('route_profiles', ()):
                if rule['route_kind'] in ('prdf_d2_split','prdf_d2_common_entry'):
                    rule['profile'].update(adapter)
    if trial.get('controller_parallel') is True:
        spec['search_overrides']['controller_parallel'] = True
        # Derive overlap from CPUs visible to this controller, rather than
        # carrying forward the old fixed readiness-pool width.
        distributed.pop('live_readiness_max_workers', None)
    if trial.get('exploratory_cost') is True:
        spec['search_overrides']['exploratory_cost'] = True
    if 'large_shot_adapter' in trial:
        selected = trial['large_shot_adapter']
        path = Path(__file__).resolve().with_name('run_rpe_method_optimizer.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != selected['optimizer_entry_sha256']:
            raise ValueError('Selected large-shot optimizer entry differs')
        adapter = dict(adapter_root=selected['root'], adapter_manifest=selected['manifest'])
        from trottertracks.rpe_host_profiles import validate_adapter_profile
        validate_adapter_profile(adapter)
        local = distributed['local_worker']
        local.update(adapter)
        for rule in local.get('route_profiles', ()):
            rule['profile'].update(adapter)
        entry = importlib.util.spec_from_file_location('_rpe_large_shot_optimizer_entry', path)
        cli = importlib.util.module_from_spec(entry)
        imports = sys.path[:]
        try:
            entry.loader.exec_module(cli)
        finally:
            sys.path[:] = imports
        cli.ROOT = ROOT
        K.CLI = cli
        spec['executor']['large_shot_bounds'] = True
    elif 'width_report_sha256' in trial:
        path = Path(__file__).resolve().with_name('run_rpe_method_optimizer.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != trial['width_report_sha256']:
            raise ValueError('Selected width report formatter differs')
    horizon_adapters = list(trial.get('local_q3_horizon_adapters', ()))
    if 'local_q3_horizon_adapter' in trial:
        horizon_adapters.insert(0, trial['local_q3_horizon_adapter'])
    if horizon_adapters:
        from trottertracks.rpe_host_profiles import route_matches, select_host_profile, validate_adapter_profile
    for selection in horizon_adapters:
        local = distributed['local_worker']
        adapter = dict(adapter_root=selection['root'], adapter_manifest=selection['manifest'])
        validate_adapter_profile(adapter)
        view = Path(selection['risk_view'])
        view_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=view, text=True).strip()
        for route in selection['routes']:
            risk = route['horizon_risk_source']
            if view_commit != risk['commit'] or hashlib.sha256((view/risk['path']).read_bytes()).hexdigest() != risk['sha256']:
                raise ValueError('Selected horizon risk view differs from the fixed Q3 route')
            base_route = {key: route[key] for key in ('kind', 'view_commit')}
            profile = select_host_profile(local, {'request': {'route': base_route}})
            profile.update(adapter)
            profile['route_views'][risk['commit']] = str(view)
            assignment = {'request': {'route': route}}
            local['route_profiles'] = [r for r in local.get('route_profiles', ()) if not route_matches(r, assignment)]
            local['route_profiles'].append(dict(route_kind=route['kind'], view_commit=route['view_commit'],
                horizon_risk_commit=risk['commit'], profile=profile))
    for field in ('ready_queue_waves', 'live_readiness_max_workers', 'live_readiness_refill_seconds'):
        if field in trial:
            distributed[field] = trial[field]
    if 'relay_sha256' in trial:
        relay = Path(__file__).resolve().with_name('run_rpe_q3_relay.py')
        if hashlib.sha256(relay.read_bytes()).hexdigest() != trial['relay_sha256']:
            raise ValueError('Selected relay implementation differs')
        distributed['relay_script'] = str(relay)
    if 'relay_batch_worker_sources' in trial:
        batch_sources = trial['relay_batch_worker_sources']
        configured_hosts = {worker['host_id'] for worker in distributed['relay']['workers']}
        if (not isinstance(batch_sources, dict) or not set(batch_sources) <= configured_hosts or
                not all(isinstance(sha, str) and len(sha) == 64 and
                        all(x in '0123456789abcdef' for x in sha) for sha in batch_sources.values())):
            raise ValueError('Expected fixed mailbox worker hashes for configured relay hosts')
        distributed['relay']['batch_worker_sources'] = batch_sources
    if 'local_resource_worker' in trial:
        selected = trial['local_resource_worker']
        root = Path(selected.get('root', Path(__file__).resolve().parents[1])).resolve()
        for file, expected in selected['files'].items():
            if hashlib.sha256((root/file).read_bytes()).hexdigest() != expected:
                raise ValueError('Selected local worker implementation differs: '+file)
        # The combined release owns transport and resource classification.
        # Legacy per-run selections still pin the scientific worker profile,
        # but must not accidentally keep an obsolete polling implementation.
        transport_root = Path(__file__).resolve().parents[1] if selected.get('transport_from_release', True) else root
        if transport_root != root and selected.get('transport_files'):
            for file, expected in selected['transport_files'].items():
                if hashlib.sha256((transport_root/file).read_bytes()).hexdigest() != expected:
                    raise ValueError('Selected release transport differs: '+file)
        distributed['worker_script'] = str(transport_root/'scripts/run_rpe_q3_worker.py')
        distributed['local_worker'].setdefault('slot_policy', {}).update(selected['slot_policy'])
        # The resumed controller starts with the frozen release on sys.path.
        name = 'trottertracks.rpe_q3_resources'
        path = transport_root/'src/trottertracks/rpe_q3_resources.py'
        if 'src/trottertracks/rpe_q3_resources.py' not in selected['files']:
            raise ValueError('Selected local worker requires fixed resource logic')
        helper = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(helper)
        helper.loader.exec_module(module)
        sys.modules[name] = module
        setattr(sys.modules['trottertracks'], 'rpe_q3_resources', module)
    if trial.get('pbs_hold_node') is True and distributed.get('pbs'):
        distributed['pbs_hold_node'] = True
    preparation_helpers = ('candidate_completion_sha256', 'preparation_bridge_sha256')
    if any(field in trial for field in preparation_helpers) and not all(field in trial for field in preparation_helpers):
        raise ValueError('Preparation recovery requires both fixed controller helpers')
    for module_name, field in (('rpe_q3_resources', 'q3_resources_sha256'),
                               ('pf_candidate_space', 'pf_candidate_space_sha256'),
                               ('rpe_live_q3_readiness', 'live_readiness_sha256'),
                               ('rpe_plan_readiness', 'plan_readiness_sha256'),
                               ('rpe_host_eligibility', 'host_eligibility_sha256'),
                               ('rpe_scope_map', 'scope_map_sha256'),
                               ('pf_saved_prices', 'pf_saved_prices_sha256'),
                               ('pf_distributed_preparation', 'pf_distributed_sha256'),
                               ('rpe_scaled_shots', 'scaled_shots_sha256'),
                               ('rpe_plan_history_cache', 'plan_history_cache_sha256'),
                               ('rpe_preparation_queue', 'preparation_queue_sha256'),
                               ('pf_candidate_selection', 'pf_selection_sha256'),
                               ('pr_candidate_completion', 'candidate_completion_sha256'),
                               ('rpe_preparation_bridge', 'preparation_bridge_sha256'),
                               ('rpe_method_optimizer', 'method_optimizer_sha256')):
        if field not in trial:
            continue
        # A resumed run imports other modules from its fixed base release.
        path = Path(__file__).resolve().parents[1]/'src/trottertracks'/(module_name+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != trial[field]:
            raise ValueError('Selected scheduler helper differs: '+module_name)
        name = 'trottertracks.'+module_name
        helper = importlib.util.spec_from_file_location(name, path)
        # The executor and CLI may already hold the base optimizer module.
        # Update that module before starting selectors so both use the handoff.
        if module_name == 'rpe_method_optimizer' and Path(O.__file__).resolve() == path.resolve():
            continue  # The pinned bootstrap installed this before importing the executor.
        module = O if module_name == 'rpe_method_optimizer' else importlib.util.module_from_spec(helper)
        if module is O:
            module.__file__ = str(path)
        if module_name == 'pr_candidate_completion':
            module.FIXED_RUN_CODE_ROOT = ROOT
            if trial.get('pr_price_service_sha256'):
                price_service = Path(__file__).resolve().with_name('pr_candidate_services.py')
                if hashlib.sha256(price_service.read_bytes()).hexdigest() != trial['pr_price_service_sha256']:
                    raise ValueError('Selected PR price metadata service differs')
                module.FIXED_RUN_PR_PRICE_SERVICE = str(price_service)
        helper.loader.exec_module(module)
        sys.modules[name] = module
        # The optimizer also imports from the package directly.
        setattr(sys.modules['trottertracks'], module_name, module)
    if assets := trial.get('scoped_horizon_sha256'):
        deployment = Path(__file__).resolve().parents[1]
        # Keep the run's provider checkout on sys.path. These entry points
        # belong to the selected scheduler, including its route metadata.
        for name, relative in (
            ('run_rpe_optimizer_q3_job', 'scripts/run_rpe_optimizer_q3_job.py'),
            ('run_rpe_pf_horizon_signal', 'scripts/run_rpe_pf_horizon_signal.py'),
            ('trottertracks.pf_horizon_preparation', 'src/trottertracks/pf_horizon_preparation.py'),
            ('trottertracks.pr_df_horizon_preparation', 'src/trottertracks/pr_df_horizon_preparation.py'),
            ('trottertracks.rpe_preparation_round', 'src/trottertracks/rpe_preparation_round.py'),
            ('trottertracks.rpe_price_validation', 'src/trottertracks/rpe_price_validation.py'),
        ):
            path = deployment / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != assets.get(relative):
                raise ValueError('Selected horizon preparation implementation differs: ' + relative)
            loader = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(loader)
            loader.loader.exec_module(module)
            sys.modules[name] = module
            if name.startswith('trottertracks.'):
                setattr(sys.modules['trottertracks'], name.split('.')[1], module)
        distributed['preparation']['horizon_transport_source'] = trial['horizon_transport_source']
    if trial.get('recover_saved_price_rounds'):
        for file, key in (('run_pf_preparation_stage.py', 'pf_stage_sha256'),
                          ('check_pf_preparation_owner.py', 'pf_owner_sha256')):
            path = Path(__file__).resolve().with_name(file)
            if hashlib.sha256(path.read_bytes()).hexdigest() != trial[key]:
                raise ValueError('Selected PF metadata entry differs: '+file)
        if isinstance(spec.get('pf_completion'), dict):
            spec['pf_completion']['recover_saved_price_rounds'] = True
    for host, workers in trial.get('calibration_workers', {}).items():
        profile = (distributed['local_worker'] if host == 'local' else
                   json.loads(Path(distributed['preparation_profiles'][host]).read_text()))
        for rule in profile['preparation_profiles']:
            if rule['provider'] == 'DF_CALIBRATION' and rule['profile'].get('task_cpu_count'):
                rule['profile']['task_cpu_count'] = workers
        if host == 'local':
            profile['slots'] = 2
            profile['slot_policy'] = dict(mode='fixed', initial_slots=2)
        distributed['preparation_profiles'][host] = profile
    if 'server2_profile' in trial:
        for worker in distributed['relay']['workers']:
            if worker['host_id'] == 'server2':
                worker['profile'] = trial['server2_profile']
    if 'server1_worker' in trial:
        worker = trial['server1_worker']
        workers = distributed['relay']['workers']
        previous = [entry for entry in workers if entry['host_id'] == 'server1']
        if (worker['host_id'] != 'server1' or len(previous) != 1
                or previous[0].get('enabled') is not False):
            raise ValueError('Expected one disabled server1 worker in the fixed relay configuration')
        workers[workers.index(previous[0])] = deepcopy(worker)
        distributed['preparation_profiles']['server1'] = trial['server1_preparation_profile']
        # The ordinary Start copies this mutable file into the run. The
        # release's admission template is covered by fixed_settings.
        admission = Path(distributed['host_admission']).resolve()
        if run_dir is None or admission != Path(run_dir).resolve() / 'HOST_ADMISSION.json':
            raise ValueError('Expected the run-owned HOST_ADMISSION.json')
        policy = O.load_json(admission)
        if policy['hosts']['server1'].get('reason') == 'USER_REPORTED_STOPPED':
            from trottertracks.rpe_q3_coordinator import atomic_json
            policy['hosts']['server1'] = dict(accepting=True, optional=True, reason='SERVER1_QG1_VALIDATED')
            atomic_json(admission, policy)
    for host, profile in trial.get('relay_q3_profiles', {}).items():
        workers = [worker for worker in distributed['relay']['workers'] if worker['host_id'] == host]
        if (len(workers) != 1 or not {'profile'} <= profile.keys() <= {'profile', 'root', 'deployment_entry'}
                or not all(Path(path).is_absolute() for path in profile.values())):
            raise ValueError('Expected one existing relay host and its fixed profile: '+host)
        worker = workers[0]
        worker.update({key: profile[key] for key in ('profile', 'root') if key in profile})
        if entry := profile.get('deployment_entry'):
            if not worker.get('deployment') or Path(entry) != Path(worker['root'])/'scripts/manage_rpe_release_deployment.py':
                raise ValueError('Expected the deployment entry from the selected relay SOURCE: '+host)
            # The preflight and mailbox must understand the same route fields.
            # Keep the original release manifest and worker layout.
            worker['deployment'] = dict(worker['deployment'], entry=entry)
    for worker in distributed.get('relay', {}).get('workers', ()):
        limit = trial.get('relay_cpu_limits', {}).get(worker['host_id'])
        if limit is not None:
            if type(limit) is not int or limit < 1 or worker.get('transport') in ('wsl', 'ssh-wsl'):
                raise ValueError('Invalid CPU limit for SSH worker: '+worker['host_id'])
            worker['cpu_limit'] = min(limit, worker.get('cpu_limit', limit))
    if 'local_memory_reserve_bytes' in trial:
        distributed['local_calibration_pair'] = dict(
            memory_reserve_bytes=trial['local_memory_reserve_bytes'],
            pf_preparation=trial.get('local_pf_preparation_pair', False))
    if 'preparation_host_preference' in trial:
        distributed['preparation_host_preference'] = trial['preparation_host_preference']
    if 'pf_compile_hosts' in trial:
        hosts = trial['pf_compile_hosts']
        configured = distributed['preparation_profiles']
        if not hosts or not set(hosts) <= configured.keys():
            raise ValueError('PF compile placement requires configured preparation profiles')
        placement = distributed.setdefault('preparation_provider_hosts', {})
        legacy = trial.get('pf_legacy_compile_hosts')
        if legacy is not None:
            if not set(legacy) <= set(hosts):
                raise ValueError('Legacy PF compile hosts must belong to the selected placement')
            distributed['pf_legacy_compile_hosts'] = list(legacy)
            helper = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_preparation_paths.py'
            if hashlib.sha256(helper.read_bytes()).hexdigest() != trial['pf_native_paths_sha256']:
                raise ValueError('Selected PF native path resolver differs')
        placement.update(PF_COMPILE=list(hosts), PF_HOLDOUT_COMPILE=list(hosts))
    if placements := trial.get('preparation_provider_hosts'):
        from trottertracks.rpe_preparation_bridge import STAGES
        configured = distributed['preparation_profiles']
        if any(provider not in STAGES or not isinstance(hosts, list) or not hosts
               or not set(hosts) <= configured.keys() for provider, hosts in placements.items()):
            raise ValueError('Preparation provider placement requires existing providers and hosts')
        distributed.setdefault('preparation_provider_hosts', {}).update(deepcopy(placements))
    if trial.get('pbs_storage_launcher'):
        distributed['pbs_gateway_storage_launcher'] = trial['pbs_storage_launcher']
    if 'pbs_collection_workers' in trial:
        distributed['pbs_collection_workers'] = trial['pbs_collection_workers']
    if 'pbs_gateway_completion_workers' in trial:
        count = trial['pbs_gateway_completion_workers']
        if type(count) is not int or count < 1:
            raise ValueError('PBS gateway completion workers must be positive')
        distributed['pbs_gateway_completion_workers'] = count
    if trial.get('pbs_ready_buffer') is True:
        distributed['pbs_ready_buffer'] = True
    if trial.get('pbs_q3_memory_backfill') is True:
        distributed['pbs_q3_memory_backfill'] = True
    if 'pbs_standby_waves' in trial:
        from trottertracks.rpe_pbs import standby_capacity
        waves = trial['pbs_standby_waves']
        standby_capacity(dict(cpu_count=1, memory_bytes=0), waves)
        if waves is None or not distributed.get('pbs_ready_buffer'):
            raise ValueError('PBS standby waves require the PBS ready buffer')
        distributed['pbs_standby_waves'] = waves
    if 'pbs_offline_search' in trial:
        if run_dir is None:
            raise ValueError('PBS offline search requires the fixed normal run directory')
        path = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_pbs_offline.py'
        source = importlib.util.spec_from_file_location('trottertracks.rpe_pbs_offline', path)
        offline = importlib.util.module_from_spec(source)
        source.loader.exec_module(offline)
        if plan := offline.prepare_plan(Path(run_dir), trial['pbs_offline_search']):
            distributed['pbs_offline_plan'] = plan
            history = str(Path(run_dir)/'distributed'/'pbs'/'offline-history')
            spec['executor'].setdefault('prior_run_dirs', []).append(history)
    if trial.get('pbs_dispatch_gateway'):
        distributed['pbs_dispatch_gateway'] = trial['pbs_dispatch_gateway']
    if trial.get('controller_queue_first'):
        distributed['controller_queue_first'] = True
        pf = spec.get('pf_completion') or {}
        if pf.get('mode') == 'complete' and not pf.get('selected'):
            # The approved preparation concurrency can feed distinct PF
            # families as well as the PR lanes. Keep explicit PF limits.
            pf.setdefault('max_candidates', spec.get('candidate_completion', {}).get('parallel', 1))
            # This controls the number of distinct PF families considered in a
            # wave. The shared queue still reserves actual resources per task.
            target = trial.get('pf_preparation_cpu_backlog')
            if target is not None:
                if type(target) is not int or target < 1:
                    raise ValueError('PF candidate preparation backlog must be a positive integer')
                pf['max_candidates'] = max(pf['max_candidates'], target)
                pf['refill_candidate_preparation'] = True
    if 'pf_recovered_price_receipts' in trial:
        evidence = trial['pf_recovered_price_receipts']
        raw = Path(evidence['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != evidence['sha256']:
            raise ValueError('Recovered PF receipt index changed')
        receipts = json.loads(raw)
        if not isinstance(receipts, dict):
            raise ValueError('Recovered PF receipts must map physical IDs to guarded locators')
        spec['pf_completion']['recovered_price_receipts_by_physical_id'] = receipts
    if 'pf_stage_audit_recoveries' in trial:
        selection = trial['pf_stage_audit_recoveries']
        raw = Path(selection['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != selection['sha256']:
            raise ValueError('Recovered PF audit package changed')
        recoveries = json.loads(raw)
        for job, record in recoveries.items():
            if record.get('kind') == 'PF_PORTABLE_INPUT_DELIVERY' and not all(record.get(key) for key in (
                    'configuration_id', 'audit_attempt', 'request_sha256', 'input_sha256')):
                raise ValueError('PF input redelivery lacks its fixed candidate or attempt: '+job)
        distributed['preparation']['pf_stage_audit_recoveries'] = recoveries
    if 'interrupted_calibrations' in trial:
        selection = trial['interrupted_calibrations']
        raw = Path(selection['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != selection['sha256']:
            raise ValueError('Interrupted calibration input plan changed')
        distributed['preparation']['interrupted_calibrations'] = json.loads(raw)
    if 'failed_preparation_recalculations' in trial:
        selection = trial['failed_preparation_recalculations']
        raw = Path(selection['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != selection['sha256']:
            raise ValueError('Fixed preparation recalculation selection changed')
        plans = json.loads(raw)
        if not isinstance(plans, dict) or any(not isinstance(plan, dict) or not all(
                isinstance(plan.get(key), str) and plan[key].strip()
                for key in ('attempt_id', 'request_sha256', 'reason')) for plan in plans.values()):
            raise ValueError('Preparation recalculation must identify the failed attempt and reason')
        distributed['preparation']['failed_preparation_recalculations'] = plans
        if core := (spec.get('candidate_completion') or {}).get('df_price_openblas_coretype'):
            distributed['preparation']['df_price_openblas_coretype'] = core
    if placements := trial.get('preparation_request_hosts'):
        if any(not job.startswith('prep-') or not hosts or any(
                host not in distributed['preparation_profiles'] for host in hosts)
                for job, hosts in placements.items()):
            raise ValueError('Preparation placement must name a fixed request and supported hosts')
        distributed['preparation_request_hosts'] = deepcopy(placements)
    if 'pf_terminal_price_leaves' in trial:
        terminal = trial['pf_terminal_price_leaves']
        raw = Path(terminal['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != terminal['sha256']:
            raise ValueError('Fixed PF terminal price plan changed')
        distributed['preparation']['pf_terminal_price_leaves'] = json.loads(raw)
    if 'horizon_preparation_catalog' in trial:
        catalog = trial['horizon_preparation_catalog']
        raw = Path(catalog['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != catalog['sha256']:
            raise ValueError('Fixed horizon preparation catalog changed')
        fixed = json.loads(raw)
        if set(fixed) != {'bindings', 'tasks'} or any(not isinstance(value, dict) for value in fixed.values()):
            raise ValueError('Horizon preparation catalog needs fixed bindings and tasks')
        distributed['preparation']['horizon_extensions'] = fixed
    if 'pf_price_recheck_provider' in trial:
        provider = trial['pf_price_recheck_provider']
        root = Path(provider['root']).resolve(strict=True)
        base = Path(spec['pf_completion']['entry']).resolve().parents[1]
        def revision(path):
            return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
        if revision(root) != provider['source'] or revision(base) != provider['base_source']:
            raise ValueError('PF price continuation requires its pinned producer and original source')
        subprocess.run(['git', 'merge-base', '--is-ancestor', provider['base_source'], provider['source']],
                       cwd=root, check=True)
        validation_sources = distributed['preparation'].get('bundle_validation_sources')
        if validation_sources:
            validation_sources.setdefault(provider['source'], str(root))
        # Remote mailboxes keep their deployed profiles. Only the verified
        # local profile advertises this additional producer on this Resume.
        local = distributed['local_worker']
        configured = distributed['preparation_profiles']['local']
        if not isinstance(configured, dict):
            configured = O.load_json(Path(configured))
        distributed['preparation_profiles']['local'] = configured
        for host_profile in (local, configured):
            rules = host_profile['preparation_profiles']
            for rule in list(rules):
                if rule.get('source_commit') != provider['base_source'] or not rule['provider'].startswith('PF_'):
                    continue
                if any(other.get('source_commit') == provider['source'] and other['provider'] == rule['provider']
                       and other['stage'] == rule['stage'] for other in rules):
                    continue
                copied = deepcopy(rule)
                copied['source_commit'] = provider['source']
                copied['profile']['preparation_sources'][provider['source']] = str(root)
                rules.append(copied)
        spec['pf_price_recheck_provider'] = deepcopy(provider)
    if trial.get('pbs_only'):
        distributed['execution_hosts'] = [worker['host_id'] for worker in distributed['pbs']['workers']]
    if 'preparation_resource_observations' in trial:
        observations = deepcopy(trial['preparation_resource_observations'])
        distributed['preparation_resource_observations'] = observations
        if distributed.get('pbs'):
            distributed.setdefault('pbs_pinned_settings', deepcopy(distributed['pbs']))
            distributed['pbs']['preparation_resource_observations'] = observations
    if trial.get('pbs_preparation_resources'):
        from trottertracks.rpe_pbs import apply_preparation_resources
        apply_preparation_resources(distributed, trial['pbs_preparation_resources'])
    if trial.get('pbs_q3_adapter'):
        from trottertracks.rpe_pbs import apply_q3_adapter
        apply_q3_adapter(distributed, trial['pbs_q3_adapter'])
    if trial.get('pbs_q3_horizon'):
        from trottertracks.rpe_pbs import apply_q3_horizon
        apply_q3_horizon(distributed, trial['pbs_q3_horizon'])
    if trial.get('pbs_q3_measured_memory'):
        from trottertracks.rpe_pbs import apply_q3_measured_memory
        apply_q3_measured_memory(distributed, trial['pbs_q3_measured_memory'])
    if 'pf_price_recheck_provider' in trial:
        bind_pf_continuation_release(distributed, provider, root)
    if selection := trial.get('df_max_width'):
        for host, path in selection.get('relay_profiles', {}).items():
            worker=next(w for w in distributed['relay']['workers'] if w['host_id']==host)
            worker['profile']=path
        if selection.get('pbs_adapter'):
            from trottertracks.rpe_pbs import apply_q3_adapter
            apply_q3_adapter(distributed,selection['pbs_adapter'])
    if trial.get('pbs_stage_artifacts_over_ssh'):
        pbs = distributed['pbs']
        if not pbs.get('gateway_python') or not pbs.get('ssh_alias'):
            raise ValueError('PBS artifact staging requires the fixed gateway interpreter and host')
        distributed.setdefault('pbs_pinned_settings', deepcopy(pbs))
        pbs['stage_artifacts_over_ssh'] = True

    # Relocating a fixed Git view does not change science, but the selected
    # execution profile contains its exact paths. Rebind only future host
    # probes; existing owner/attempt profiles continue to be replayed as saved.
    mappings=deepcopy(trial.get('relay_transport_paths') or {})
    if trial.get('pbs_dispatch_gateway') and distributed.get('pbs'):
        pbs_paths=(trial['pbs_dispatch_gateway'].get('transport') or {}).get('paths',{})
        for worker in distributed['pbs'].get('workers',()):
            mappings[worker['host_id']]=pbs_paths
    if mappings:
        from trottertracks.rpe_pbs import pbs_transport_paths
        profiles=distributed.get('preparation_profiles') or {}
        for host,paths in mappings.items():
            if (host not in profiles or not isinstance(paths,dict)
                    or any(not isinstance(old,str) or not isinstance(new,str)
                           or not Path(old).is_absolute() or not Path(new).is_absolute() for old,new in paths.items())):
                raise ValueError('Invalid relocated worker profile: '+host)
            if paths:
                profile=profiles[host]
                if not isinstance(profile,dict):profile=O.load_json(Path(profile))
                profiles[host]=pbs_transport_paths(profile,paths)
        distributed['host_transport_paths']=mappings


@contextmanager
def serialize_legacy_ast():
    """Keep parallel controller metadata reads safe on the pinned CPython."""
    if sys.implementation.name != 'cpython' or not (3, 11, 0) <= sys.version_info[:3] < (3, 11, 8):
        yield
        return
    import ast
    from threading import RLock

    # CPython gh-106905: these releases share AST-conversion recursion
    # state. Serialize just AST parsing, leaving all worker jobs parallel.
    parse = ast.parse
    lock = RLock()

    @wraps(parse)
    def parse_serially(*args, **kwargs):
        with lock:
            return parse(*args, **kwargs)

    ast.parse = parse_serially
    try:
        yield
    finally:
        ast.parse = parse


class WideDriver(K.Driver):
    def __init__(self, a):
        existing = (a.run_dir.resolve() / 'KAPPA_STATE.json').exists()
        super().__init__(a)
        if not existing and self.spec.get('continues'):
            previous = ROOT / self.spec['continues']['run_dir'] / 'KAPPA_STATE.json'
            carried = O.load_json(previous)
            for key in ('points', 'family_constants', 'saved_packets', 'splits'):
                if key in carried:self.state[key] = deepcopy(carried[key])
            self.state['continuation'] = dict(directory=str(previous.parent),
                previous_round=carried.get('round_id'), previous_stop=carried.get('stop'))
        # These requests were never executed; actual scientific HELD/STOPs stay held.
        for p in self.state['points']:
            if p['status'] == 'HELD' and p.get('reason') == 'COMPLETION_REQUIRED_NOT_RUN_IN_THIS_INVOCATION':
                p.update(status='COMPLETION', unrun_previous_reason=p.pop('reason'))
        self.wide = self.spec['search_overrides']['schedule_wide']
        if 'seeded' not in self.state:
            self.seed()

    # ------------------------------------------------------------ state
    def seed(self):
        """Points from the ledger (every configuration with verified saved Q3, incl. saved multiplier configurations) and the
        carried kappa state; family constants of prepared and new splits (exact, from canonical configurations)."""
        led = O.load_json(ROOT / self.spec['ledger_summary'])
        have = {(p['L_D'], Fraction(p['kappa'])) for p in self.state['points']}
        for cid, d in led['by_configuration'].items():
            if d.get('kappa') is None:
                continue
            k = O.frac(d['kappa']);ld = d['L_D']
            if (ld, k) not in have:
                self.state['points'].append(dict(L_D=ld, kappa=str(k), status='EVALUATED', configuration_id=cid, requested_cycle=0,
                                                 request_reason='SAVED_EVALUATED_SCHEDULE'));have.add((ld, k))
        for p in self.state['points']:
            d = led['by_configuration'].get(p.get('alias_of') or p.get('configuration_id') or '')
            if d:
                p['verified_all_best'] = d.get('verified_all');p['rule_adopted_best'] = d.get('rule_adopted')
        fc = self.state.setdefault('family_constants', {})
        for ld, loc in (self.spec.get('new_families') or {}).items():
            cfg = read_loc(loc['configuration']);cfg = cfg.get('configuration', cfg)
            rows = KS.constants(cfg)
            if KS.R_vector(rows, KS._frac(cfg['spec']['kappa']['value'])) != [r['R_outer'] for r in cfg['rounds']]:
                raise ValueError('MAPPER_DOES_NOT_REPRODUCE_NEW_SPLIT_' + str(ld))
            fc.setdefault(str(ld), KS.encode_rows(rows))
            self.state.setdefault('new_split_configurations', {})[str(ld)] = dict(loc['configuration'], configuration_id=cfg['id'])
        for p in self.state['points']:  # a new split is not 'waiting for P0' any more: D2's completion prepares it when selected
            if str(p['L_D']) in (self.spec.get('new_families') or {}) and p['status'] == 'HELD' and str(p.get('reason', '')).startswith('SPLIT_PREPARATION_REQUIRED'):
                p.update(status='COMPLETION', reopened=dict(previous=p.pop('reason'), by='D2_COMPLETION_API'))
        self.state['seeded'] = dict(UTC=K.now(), ledger=self.spec['ledger_summary'], new_families=sorted(self.spec.get('new_families') or {}))
        self.save()

    def families(self):
        return {int(k): W.Family(int(k), rows) for k, rows in self.state.get('family_constants', {}).items()}

    def wide_points(self):
        fams = self.families();out = []
        for p in self.state['points']:
            if p['L_D'] not in fams:
                continue
            best = [x for x in (p.get('best'), p.get('verified_all_best'), p.get('run_completed_best')) if x is not None]
            best = min(O.frac(x) for x in best) if best else None
            st = p['status']
            status = ('FEASIBLE' if best is not None else
                      'NO_FEASIBLE_FOUND' if st in ('EVALUATED', 'ALIAS') and not p.get('promising') else
                      'UNAVAILABLE' if st in ('HELD', 'PRUNED') else 'IN_PROGRESS' if st in ('EVALUATED', 'ALIAS') else 'PENDING')
            out.append(dict(family=p['L_D'], kappa=Fraction(p['kappa']), status=status, best=best, min_lower_over_tau=p.get('min_lower_over_tau')))
        return fams, out

    # ------------------------------------------------------------ plan
    def plan(self):
        fams, pts = self.wide_points()
        floors = sorted((Fraction(p['kappa']), p['L_D'], O.frac(p['price_floor'])) for p in self.state['points'] if p.get('price_floor') is not None)

        def price_floor(ld, R):  # prices grow with R: the floor of a quoted smaller-or-equal schedule bounds this one from below
            f = fams[ld];lb = [fl for k, l, fl in floors if l == ld and all(a <= b for a, b in zip(f.R(k), R))]
            return max(lb) if lb else None
        proposals, decisions, report = W.plan_wide(fams, pts, self.wide, price_floor)
        for q in proposals:
            if self.point(q['family'], q['kappa']) is None:
                new_split = str(q['family']) in (self.state.get('new_split_configurations') or {}) and not any(
                    p['L_D'] == q['family'] and p.get('configuration_id') for p in self.state['points'])
                self.state['points'].append(dict(L_D=q['family'], kappa=str(q['kappa']), status='COMPLETION' if new_split else 'PENDING',
                                                 requested_cycle=self.state['cycle'], request_reason=q['reason'],
                                                 request_evidence={k: v for k, v in q.items() if k not in ('family', 'kappa')}))
        self.state['history'].append(dict(cycle=self.state['cycle'], UTC=K.now(), rule=W.RULE, decisions=decisions, report=report,
                                          proposals=[dict(L_D=q['family'], kappa=str(q['kappa']), reason=q['reason']) for q in proposals]))
        return proposals, report

    # ------------------------------------------------------------ inputs
    def prune(self, ctx):
        """Proven price floor: an allocation of one shot per round and axis is the cheapest of the schedule."""
        fams, pts = self.wide_points();best = min((p['best'] for p in pts if p['status'] == 'FEASIBLE'), default=None)
        by = {c['configuration_id']: c for c in ctx['candidates']}
        for p in self.state['points']:
            c = by.get(p.get('configuration_id'))
            if c is None or c.get('prices') is None:
                continue
            p['price_floor'] = sum(c['prices'])
            if p['status'] == 'PREPARED' and best is not None and p['price_floor'] >= best:
                p.update(status='PRUNED', reason='PROVEN_COST_FLOOR:%s>=%s' % (O.show(p['price_floor']), O.show(best)))

    def completion_requests(self, on_wait=None):
        """New splits: D2 completion (READY -> search, HELD -> recorded).  --plan-only records the request only."""
        todo = [p for p in self.state['points'] if p['status'] == 'COMPLETION']
        if not todo:
            return
        cc = self.spec.get('candidate_completion');imported = {}
        if cc and cc.get('resume_preparation_from'):
            for r in read_loc(cc['resume_preparation_from'])['records'].values():
                if r['status'] == 'READY_FOR_SEARCH' and r.get('design') and r.get('method') == 'PR_PAULI':  # never a PR-DF record
                    d = r['design'];imported[(d['L_D'], O.frac(d['kappa']), O.frac(d['delta_pf']))] = r
        run = []
        for p in todo:
            design = dict(L_D=p['L_D'], formula='2nd', delta_over_h='1', kappa=str(p['kappa']), K=2)
            p['completion_request'] = dict(method='PR_PAULI', selection=dict(rule=W.RULE, reason=p.get('request_reason'), cycle=p['requested_cycle']),
                                           design=design)
            saved = next((r for (ld, k, dl), r in imported.items() if ld == p['L_D'] and k == Fraction(p['kappa'])
                          and dl == Fraction(0.1)), None)  # same split, absolute kappa and width (delta_PF = h = binary64 0.1)
            if saved is not None:
                p.update(status='REGISTERED', configuration_id=saved['configuration_id'], completion=dict(status='REUSED_SAVED_READY_RECORD',
                                                                                                         record=saved['key']));continue
            if self.a.plan_only or not cc or self.spec.get('science_approved') is not True:
                p.update(status='HELD', reason='COMPLETION_REQUIRED_NOT_RUN_IN_THIS_INVOCATION');continue
            run.append(p)
        if not run:
            return
        comp = self.completion_api(dict(cc, max_attempted_per_method=0))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, min(len(run), int(cc.get('parallel', 4))))) as pool:  # independent splits / schedules
            futures = [pool.submit(comp.complete, p['completion_request']) for p in run]
            self.log(event='COMPLETION_WAIT_START', requests=len(run))
            if on_wait is not None:on_wait()
            for p, future in zip(run, futures):
                rec = future.result()
                if rec['status'] == 'READY_FOR_SEARCH':
                    p.update(status='REGISTERED', configuration_id=rec['configuration_id'], completion=dict(status=rec['status'], key=rec['key']))
                else:  # HELD: the concrete reason is kept and the other candidates go on
                    p.update(status='HELD', reason='COMPLETION:%s:%s' % (rec['status'], str(rec.get('reason'))[:300]),
                             completion=dict(status=rec['status'], key=rec['key']))
            self.log(event='COMPLETION_WAIT_END', requests=len(run))

    def completion_api(self, settings):
        from trottertracks.pr_candidate_completion import Completion
        return Completion(self.run / 'completion', settings)

    def write_points(self):
        """A's preparation entry gets only schedules of splits with a prepared base (new splits go through D2 completion)."""
        rows = [dict(L_D=p['L_D'], kappa=p['kappa']) for p in self.state['points']
                if p['status'] not in ('HELD', 'PRUNED', 'COMPLETION') and not p.get('completion')]
        self.points_file.parent.mkdir(parents=True, exist_ok=True);self.points_file.write_text(json.dumps(rows, indent=1) + '\n')

    def absorb(self, res):
        """Rule-adopted bests (V2) plus the cheapest feasible completed job of this run per configuration, adopted or not
        (provisional for planning; the ledger re-verifies every job with the executor's identity checks before reporting)."""
        super().absorb(res)
        done = {}
        for q in (self.run / 'jobs' / 'PR_PAULI').glob('*/REQUEST.json') if (self.run / 'jobs' / 'PR_PAULI').exists() else []:
            job = q.parent
            if not (job / 'OUTCOME.json').exists():
                continue
            out, _ = O.job_outcome(job);r = json.loads(q.read_text())
            if out.get('status') == O.PASS:
                c = O.frac(r['cost']);cid = r['configuration_id']
                done[cid] = min(done.get(cid, c), c)
        for p in self.state['points']:
            cid = p.get('alias_of') if p['status'] == 'ALIAS' else p.get('configuration_id')
            if cid in done:
                p['run_completed_best'] = done[cid]

    # ------------------------------------------------------------ loop
    def batch(self, ready_only=False):
        spec = json.loads(json.dumps(self.spec))
        if ready_only:
            # Completion owns its state/registry while working. This interim batch
            # uses only the already published points and route registry snapshot.
            spec.pop('candidate_completion', None)
        if spec.get('candidate_completion'):
            spec['candidate_completion']['max_attempted_per_method'] = 0  # install READY records only; B2 chooses what to complete
        self.spec, keep = spec, self.spec
        try:
            return super().batch()
        finally:
            self.spec = keep

    def loop(self):
        previous_stop = self.state.get('stop', {}).get('reason')
        transient = {'USER_REQUESTED_STOP', 'PLAN_ONLY_FIRST_DECISION_WRITTEN', 'SCIENCE_NOT_APPROVED_FIRST_DECISION_WRITTEN',
                     'PLAN_ONLY_NOTHING_TO_PREPARE', 'MAX_CYCLES'}
        if previous_stop and previous_stop not in transient:
            self.log(event='PRESERVED_STOP', reason=previous_stop);return
        if previous_stop:self.state.pop('stop')
        while self.state['cycle'] < self.a.max_cycles:
            if (self.run/'STOP_REQUESTED.json').exists():
                self.state['stop']=dict(reason='USER_REQUESTED_STOP',UTC=K.now());self.save();return
            self.state['cycle'] += 1;self.save()
            proposals, report = self.plan();self.save()
            interim = []
            def ready_work():
                if self.spec.get('overlap_ready_batch') and any(p['status'] in ('REGISTERED', 'EVALUATED', 'ALIAS') for p in self.state['points']):
                    self.log(event='READY_BATCH_DURING_COMPLETION_START')
                    code, res = self.batch(ready_only=True)
                    if res is None:
                        interim.append(dict(stop_reason='INVOCATION_FAILED', returncode=code))
                        self.log(event='READY_BATCH_DURING_COMPLETION_END', returncode=code, stop='INVOCATION_FAILED')
                        return
                    interim.append(res)
                    self.absorb(res);self.save()
                    self.log(event='READY_BATCH_DURING_COMPLETION_END', stop=res['stop_reason'], new_calls=res['new_calls'])
            self.completion_requests(on_wait=ready_work);self.save()
            if interim and interim[-1]['stop_reason'] == 'INVOCATION_FAILED':
                self.state['stop'] = dict(reason='V2_INVOCATION_FAILED', returncode=interim[-1]['returncode'], UTC=K.now())
                self.save();return
            fatal = {'UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER', 'HOST_ELIGIBILITY_HOLD',
                     'Q3_BUDGET_EXHAUSTED', 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET'}
            if interim and interim[-1]['stop_reason'] in fatal:
                self.state['stop'] = dict(reason='V2_' + interim[-1]['stop_reason'], UTC=K.now())
                self.save();return
            stop_here = self.a.plan_only or self.spec.get('science_approved') is not True
            if any(p['status'] in ('PENDING', 'PREPARED') for p in self.state['points']):
                ctx = self.prepare();self.prune(ctx);self.save()
                if stop_here:
                    self.state['stop'] = dict(reason='PLAN_ONLY_FIRST_DECISION_WRITTEN' if self.a.plan_only else
                                              'SCIENCE_NOT_APPROVED_FIRST_DECISION_WRITTEN', UTC=K.now());self.save();return
                self.signals();self.save()
            elif stop_here:
                self.state['stop'] = dict(reason='PLAN_ONLY_NOTHING_TO_PREPARE', UTC=K.now());self.save();return
            code, res = self.batch()
            if res is None:
                self.state['stop'] = dict(reason='V2_INVOCATION_FAILED', returncode=code, UTC=K.now());self.save();return
            self.absorb(res);self.save()
            pending = [p for p in self.state['points'] if p['status'] in ('PENDING', 'PREPARED', 'REGISTERED', 'COMPLETION')]
            promising = sum(p.get('promising') or 0 for p in self.state['points'] if p['status'] == 'EVALUATED')
            self.log(event='CYCLE_END', v2_stop=res['stop_reason'], proposals=len(proposals), pending=len(pending), promising=promising,
                     plan_stop=report['stop'])
            if res['stop_reason'] in ('Q3_BUDGET_EXHAUSTED', 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET'):
                self.state['stop'] = dict(reason=res['stop_reason'], UTC=K.now());self.save();return
            if res['stop_reason'] in ('UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER', 'HOST_ELIGIBILITY_HOLD'):
                self.state['stop'] = dict(reason='V2_' + res['stop_reason'], UTC=K.now());self.save();return
            if not proposals and not pending and res['stop_reason'] in ('ONLY_SMALL_OR_DEFERRED_LEFT', 'NO_CANDIDATE_LEFT_IN_RANGE'):
                self.state['stop'] = dict(reason='SCHEDULE_WIDE_RULE_EXHAUSTED', UTC=K.now(), v2_stop=res['stop_reason'],
                                          families=report['families']);self.save();return
        self.state['stop'] = dict(reason='MAX_CYCLES', UTC=K.now());self.save()


def shortlist_price_ids(results):
    unresolved = {}
    for method, result in results.items():
        needed = {(row['configuration_id'], row['M'])
                  for row in (result.get('scaled_shots') or {}).get('unpriced_feasible', ())}
        unresolved[method] = sorted({row.get('source_configuration_id') or row['configuration_id']
                                     for row in result.get('by_configuration', ())
                                     if (row['configuration_id'], row.get('M')) in needed
                                     and row['configuration_id']})
    return unresolved


class LivePriceWork:
    """Persist selector-confirmed feasible points before the full batch ends."""
    def __init__(self, run, initial):
        from threading import RLock, Event
        self.path = run/'LIVE_SHORTLIST.json'
        self.lock = RLock()
        self.pending = Event()
        self.targets = {method: set(ids) for method, ids in initial.items()}
        self.rows = {}
        if self.path.is_file():
            for row in O.load_json(self.path).get('points', ()):
                key = (row['method'], row['configuration_id'], row['M'])
                self.rows[key] = row
                if row['estimated_total_RZ'] is None:
                    self.targets.setdefault(row['method'], set()).add(row['source_configuration_id'])

    def snapshot(self):
        with self.lock:
            return {method: sorted(ids) for method, ids in self.targets.items()}

    def observe(self, method, candidate, allocation):
        from trottertracks.rpe_q3_coordinator import atomic_json
        cid = candidate['configuration_id']
        parent = candidate.get('source_configuration_id') or cid
        key = (method, cid, allocation['M'])
        cost = allocation['cost']
        price_key = hashlib.sha256(str(candidate.get('prices')).encode()).hexdigest() if cost is not None else None
        with self.lock:
            previous = self.rows.get(key)
            if previous is not None and previous['estimated_total_RZ'] is not None:
                if cost is None or (previous.get('price_key') == price_key and KS._frac(previous['estimated_total_RZ']) <= cost):
                    return
            row = dict(method=method, configuration_id=cid, source_configuration_id=parent,
                       M=allocation['M'], estimated_total_RZ=O.enc(cost) if cost is not None else None, price_key=price_key)
            if previous == row:
                return
            self.rows[key] = row
            if cost is None and parent not in self.targets.setdefault(method, set()):
                self.targets[method].add(parent)
                self.pending.set()
            order = sorted((row for row in self.rows.values() if row['estimated_total_RZ'] is not None),
                           key=lambda row: (row['M'], KS._frac(row['estimated_total_RZ']),
                                            row['method'], row['configuration_id']))
            atomic_json(self.path, dict(schema='RPE_LIVE_SHORTLIST_V1',
                                        points=list(self.rows.values()), targets=self.snapshot(),
                                        estimated_order=[dict(method=row['method'], M=row['M'],
                                            configuration_id=row['configuration_id'],
                                            estimated_total_RZ=row['estimated_total_RZ']) for row in order],
                                        cost_stage='EXPLORATORY_SHORTLIST_NOT_FINAL'))


class WidthDriver:
    """Declared widths for both PR methods, reusing common completion and V2.

    The original round05 driver above remains the default without an explicit
    requested_widths declaration.  This path never invents q/R from a width:
    family constants come from the method's existing canonical generator.
    """
    work_conserving_search = False
    prior_history_inventory = None

    def __init__(self, a):
        self.a = a; self.run = a.run_dir.resolve(); self.run.mkdir(parents=True, exist_ok=True)
        self.spec = O.load_json(a.round)
        self.problem = K.CLI.price_overrides(K.CLI.scoped(O.load_json(a.problem), self.spec.get('method_overrides')), self.spec)
        self.rule_stop = self.spec.get('search_to_rule_stop',False)
        self.widths = W.requested_widths(self.problem, self.spec)
        if self.widths is None:
            if not self.rule_stop or any(m.startswith('PR_') for m in self.spec['methods']):
                raise ValueError('Explicit requested_widths required')
            self.widths={}
        self.scaled_shots = (self.spec.get('search_overrides', {}).get('rule', self.problem['search'].get('rule'))
                             == 'RPE_SCALED_SHOTS_V1')
        self.methods = list(self.spec['methods']) if self.rule_stop or self.scaled_shots else list(self.widths)
        self.live_ready_refill = self.spec.get('live_ready_refill') is True
        if self.live_ready_refill and (
            self.spec.get('overlap_ready_batch') is not True
            or self.spec.get('executor', {}).get('scheduler') != 'concurrent'
            or self.spec.get('search_overrides', {}).get('rule', self.problem['search'].get('rule'))
                not in ('METHOD_MINIMUM_PRIORITY_V2', 'RPE_SCALED_SHOTS_V1')
        ):
            raise ValueError('live_ready_refill requires overlapping preparation and a concurrent selector')
        self.work_conserving_search = self.spec.get('search_overrides', {}).get('work_conserving_search') is True
        if self.work_conserving_search and not self.live_ready_refill:
            raise ValueError('work_conserving_search requires live_ready_refill')
        for method,widths in self.widths.items():
            self.problem['methods'][method]['requested_delta_pf_values']=[O.enc(KS._frac(w['raw']))for w in widths]
        self.state_file = self.run/'KAPPA_STATE.json'
        self.state = O.load_json(self.state_file) if self.state_file.exists() else dict(
            schema='RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1', round_id=self.spec['round_id'],
            requested_widths=self.widths, cycle=0, points=[], families={}, history=[])
        if self.state.get('schema') != 'RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1':
            raise ValueError('Preserve the old single-width run; use an independent multi-width directory')
        if self.state['round_id'] != self.spec['round_id'] or self.state['requested_widths'] != self.widths:
            raise ValueError('Run belongs to different requested widths or round')
        # Resume overlays keep the old scientific checkout on sys.path.
        from importlib.util import module_from_spec, spec_from_file_location
        journal_path = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_width_checkpoint.py'
        loader = spec_from_file_location('trottertracks.rpe_width_checkpoint', journal_path)
        journal = module_from_spec(loader);loader.loader.exec_module(journal)
        self.point_journal = journal.PointJournal(self.run/'POINT_PROGRESS.jsonl', self.state)
        self.wide = self.spec.get('search_overrides', {}).get('schedule_wide', {})
        self.completion = None
        self.check_fixed_inputs()
        trial = selected_worker_policy(self.spec)
        if trial and not a.plan_only:
            apply_worker_trial(self.spec, trial, run_dir=self.run)
        if not self.state['families']:
            started=time.monotonic();self.log(event='SEED_START')
            self.seed();self.log(event='SEED_END',wall_seconds=time.monotonic()-started,families=len(self.state['families']))
        self.recheck_unsupported_families()
        if self.scaled_shots and not (self.run/'SCOPE_MAP.json').is_file():
            previous=self.run/'ROUND_RESULT.json'
            self.write_scope_map(O.load_json(previous)['methods'] if previous.is_file() else None)

    def log(self, **row):
        with (self.run/'DRIVER_PROGRESS.jsonl').open('a') as f:
            f.write(json.dumps(dict(UTC=K.now(),cycle=self.state['cycle'],**row),default=str)+'\n')

    def checkpoint_point(self, point):
        if journal := getattr(self, 'point_journal', None):
            journal.record(point)

    def save(self):
        def write():
            tmp=self.state_file.with_suffix('.tmp')
            tmp.write_text(json.dumps(O.jsonable(self.state),indent=2)+'\n')
            tmp.replace(self.state_file)
        journal = getattr(self, 'point_journal', None)
        if journal is not None:
            journal.checkpoint(self.state, write)
        else:
            write()

    def write_scope_map(self, results=None):
        """Materialize every declared logical coordinate, including rejected families."""
        from trottertracks.rpe_scope_map import build_scope_map
        from trottertracks.rpe_scaled_shots import settings
        from trottertracks.rpe_q3_coordinator import atomic_json
        search=dict(self.problem['search'],**self.spec.get('search_overrides',{}))
        horizons=settings(search.get('scaled_shots'))['M']
        methods={m:self.problem['methods'][m] for m in self.methods}
        document=build_scope_map(methods,horizons,families=self.state['families'],points=self.state['points'],
                                 results=results,mapper_version={m:(v.get('generation')or{}).get('kind') for m,v in methods.items()})
        if results is not None:
            for method, scope in document['methods'].items():
                coverage=results.get(method,{}).get('scaled_shots',{}).get('scope_coverage')
                if coverage is None:
                    continue
                represented=sum(bool(row.get('configuration_id')) for row in scope['rows'])
                coverage.update(requested=scope['requested'],represented=represented,
                    missing_count=scope['requested']-represented,logical_status_counts=scope['counts'],
                    scope_map='SCOPE_MAP.json')
        path=self.run/'SCOPE_MAP.json'
        self.scope_summary={m:{k:v for k,v in row.items() if k!='rows'} for m,row in document['methods'].items()}
        if not path.is_file() or O.load_json(path)!=document:
            atomic_json(path,document)

    def check_fixed_inputs(self):
        """Keep cached families tied to the same physical/evaluation inputs."""
        completion={k:v for k,v in(self.spec.get('candidate_completion')or{}).items()
                    if k not in('price_evidence_results','plan_only')}
        current=dict(schema='RPE_REQUESTED_WIDTH_FIXED_INPUTS_V1',problem=self.problem,
            requested_methods=self.methods,requested_widths=self.widths,
            round_scope={k:self.spec.get(k)for k in('method_overrides','search_overrides','width_families','new_families','kappa_families')},
            completion=completion)
        preparation=(self.spec.get('executor',{}).get('distributed')or{}).get('preparation')
        if preparation:
            current['preparation_budgets']={stage:preparation.get('budgets',{}).get(stage,0)
                for stage in ('signal','compile','audit')}
        if self.live_ready_refill:
            current['round_scope']['live_ready_refill'] = True
        pf_candidates=(self.spec.get('pf_completion') or self.problem.get('pf_completion') or {}).get('max_candidates',1)
        if pf_candidates!=1:
            current['round_scope']['pf_parallel_candidates']=pf_candidates
        path=self.run/'FIXED_INPUTS.json'
        if path.exists():
            if O.load_json(path)!=current:
                raise ValueError('FIXED_INPUTS_CHANGED: use a separate run for another physical/evaluation/search scope')
            return
        # The first technical multi-width prototype predates this small input
        # record.  Its metadata-only state can be adopted after source checks;
        # an executed state without its original inputs cannot be inferred safe.
        if self.state['families']:
            if (self.state.get('stop',{}).get('reason')!='REQUESTED_WIDTH_METADATA_ONLY'or
                    any(p['status']in('REGISTERED','EVALUATED')for p in self.state['points'])or
                    (self.run/'jobs').exists()):
                raise ValueError('FIXED_INPUTS_MISSING_FOR_EXECUTED_STATE: preserve this run and use a new one')
            self.verify_metadata_inputs()
        path.write_text(json.dumps(current,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

    def verify_metadata_inputs(self):
        reader,_=self.catalog();facts=getattr(self,'catalog_facts',{})
        expected_binding={k:facts.get(v)for k,v in(('hamiltonian','hamiltonian_hash'),('state','state_hash'),('snapshot','snapshot_sha256'))}
        for key,f in self.state['families'].items():
            method=f['method'];setting=self.problem['methods'].get(method)
            if setting is None or method not in self.methods:raise ValueError('METADATA_SCOPE_METHOD_CHANGED:'+key)
            if setting.get('L_D_values')and f['L_D']not in setting['L_D_values']:raise ValueError('METADATA_SCOPE_SPLIT_CHANGED:'+key)
            formulas=(setting.get('generation')or{}).get('formulas',['2nd'])
            if f['formula']not in formulas:raise ValueError('METADATA_SCOPE_FORMULA_CHANGED:'+key)
            if KS._frac(f['requested_delta_pf'])not in{KS._frac(v['raw'])for v in self.widths[method]}:
                raise ValueError('METADATA_SCOPE_WIDTH_CHANGED:'+key)
            if not f.get('configuration'):continue
            cfg=f['configuration'];s=cfg['spec']
            if s['M']!=self.problem['rpe']['M']:raise ValueError('METADATA_SCOPE_M_CHANGED:'+key)
            if self.problem['rpe'].get('h')is not None and KS._frac(s['h'])!=KS._frac(self.problem['rpe']['h']):
                raise ValueError('METADATA_SCOPE_H_CHANGED:'+key)
            for field,value in expected_binding.items():
                if value is not None and s.get('binding',{}).get(field)!=value:raise ValueError('METADATA_SCOPE_BINDING_CHANGED:'+key)
            source=f.get('base_source')or f.get('template_source')
            if source:
                base=reader.read(source);base=base.get('configuration',base)['spec']
                if KS._frac(s['h'])!=KS._frac(base['h'])or s['M']!=base['M']:
                    raise ValueError('METADATA_TIME_GRID_DIFFERS_FROM_FIXED_SOURCE:'+key)
            if f.get('rows')and KS.encode_rows(KS.constants(cfg))!=f['rows']:
                raise ValueError('METADATA_CONSTANTS_DIFFER_FROM_CONFIGURATION:'+key)

    def catalog(self):
        reader, inputs, catalog = self.saved_catalog(self.problem)
        self.catalog_facts=O.hamiltonian_facts(self.problem)
        O.check_comparison(self.problem,self.catalog_facts,inputs['comparison'])
        return reader,catalog

    def saved_catalog(self, problem):
        source=problem['evidence']['saved_catalog_inputs']
        if not hasattr(self,'_saved_catalog') or self._saved_catalog[0]!=source:
            self._saved_catalog=(deepcopy(source),O.load_catalog(ROOT,problem))
        return self._saved_catalog[1]

    def generate_families(self, method, items):
        return O.run_generator(ROOT,self.problem['methods'][method]['generation'],items)['items']

    @staticmethod
    def admit_generated_family(family,base,made):
        try:
            if made['status']!='GENERATED':raise ValueError(made.get('reason')or made['status'])
            cfg=made['configuration'];s=cfg['spec']
            if KS._frac(s['delta_pf'])!=KS._frac(family['requested_delta_pf']):
                raise ValueError('MAPPER_DOES_NOT_PRESERVE_EXACT_REQUESTED_WIDTH')
            if KS._frac(s['h'])!=KS._frac(base['h']) or s['M']!=base['M']:
                raise ValueError('MAPPER_CHANGED_FIXED_RPE_TIME_GRID')
            rows=KS.constants(cfg);k=KS._frac(s['kappa']['value'])
            if KS.R_vector(rows,k)!=[v['R_outer']for v in sorted(cfg['rounds'],key=lambda r:r['m'])]:
                raise ValueError('SCHEDULE_CONSTANTS_DO_NOT_REPRODUCE_CANONICAL_MAPPER')
            family.update(status='READY_METADATA',rows=KS.encode_rows(rows),configuration=cfg)
            family.pop('reason',None)
        except (ValueError,KeyError,TypeError) as exc:
            family.update(status='UNSUPPORTED_MAPPER',reason=str(exc))

    def admit_max_width_families(self):
        settings=self.spec.get('candidate_completion') or {}
        source=settings.get('df_max_width_source')
        if not source:return
        import importlib, types
        dirname=Path(settings['df_max_width_code_root'])/'src/trottertracks'
        package='_rpe_controller_max_width'
        if package not in sys.modules:
            module=types.ModuleType(package);module.__path__=[str(dirname)];sys.modules[package]=module
        maximum=importlib.import_module(package+'.pr_df_max_width')
        families=self.state['families']
        bases={f['L_D']:f for f in families.values() if f['method']=='PR_DF' and f['status']=='READY_METADATA'
               and not maximum.selected(f['configuration'])}
        if not bases:return
        descriptor=settings['saved_df_descriptors']['0']
        from importlib.util import spec_from_file_location, module_from_spec
        ext=spec_from_file_location('_rpe_max_width_descriptor',dirname/'pr_df_descriptor_extension.py')
        helper=module_from_spec(ext);ext.loader.exec_module(helper)
        count=0
        for key,f in families.items():
            if f.get('method')!='PR_DF' or f.get('status')!='UNSUPPORTED_MAPPER':continue
            ld=f['L_D'];base_family=bases.get(ld) if ld else next(iter(bases.values()))
            if base_family is None:continue
            base=base_family['configuration'];tail=base['spec']['A_input']['tail_norm'];loc=tail['source']
            if ld==0:
                provisional=deepcopy(base);provisional['spec']['L_D']=0
                tail=helper.validate(helper.read(ROOT,descriptor),descriptor,provisional);loc=descriptor
            design=dict(L_D=ld,formula=f['formula'],K=f['K'],delta_pf=f['requested_delta_pf'],
                        kappa=str(KS._frac(base['spec']['kappa']['value'])))
            cfg=maximum.build(ROOT,base,design,tail=tail,tail_source=loc,source=source)
            maximum.validate(ROOT,cfg)
            f.setdefault('mapper_rejections',[]).append(dict(revision=f.get('mapper_source'),reason=f.get('reason')))
            self.admit_generated_family(f,base['spec'],dict(status='GENERATED',configuration=cfg))
            if f['status']=='READY_METADATA':
                f.update(mapper_source=source)
                if base_loc:=base_family.get('base_source') or base_family.get('template_source'):f['template_source']=base_loc
                if 'commit' in loc and 'path' in loc:f['tail_descriptor']=loc
                count+=1
            self.log(event='MAPPER_FAMILY_RECHECK',family=key,status=f['status'],reason=f.get('reason'))
        if count:self.save()

    def recheck_unsupported_families(self):
        """A changed selected generator may repair a rejected logical family."""
        self.admit_max_width_families()
        revisions={method:hashlib.sha256(json.dumps(self.problem['methods'][method].get('generation',{}),
                                         sort_keys=True).encode()).hexdigest() for method in self.widths}
        prior=self.state.get('mapper_revisions')
        if prior is None:
            self.state['mapper_revisions']=revisions;self.save();return
        for method,revision in revisions.items():
            if prior.get(method)==revision:continue
            reader=O.Reader(ROOT);items=[];selected={}
            for key,f in self.state['families'].items():
                if f.get('method')!=method or f.get('status')!='UNSUPPORTED_MAPPER':continue
                f.setdefault('mapper_rejections',[]).append(dict(revision=prior.get(method),reason=f.get('reason')))
                if f.get('base_source'):
                    base=reader.read(f['base_source']);base=base.get('configuration',base)['spec']
                    ratio=KS._frac(f['requested_delta_pf'])/KS._frac(base['h'])
                    items.append(dict(item_id=key,base=f['base_source'],changes=dict(formula=f['formula'],delta_over_h=str(ratio))))
                    selected[key]=(f,base)
                elif f.get('preparation_request'):
                    f['status']='FAMILY_PREPARATION_PENDING'
            if items:
                results=self.generate_families(method,items)
                if len(results)!=len(selected) or {r['item_id'] for r in results}!=selected.keys():
                    raise ValueError('Width mapper recheck results differ from requested families')
                for result in results:
                    f,base=selected[result['item_id']];self.admit_generated_family(f,base,result)
                    self.log(event='MAPPER_FAMILY_RECHECK',family=result['item_id'],status=f['status'],reason=f.get('reason'))
        self.state['mapper_revisions']=revisions
        if prior!=revisions:self.save()

    def seed(self):
        reader,catalog=self.catalog();sources={}
        for point in catalog['candidates']:
            if point['method'] in self.methods and point.get('configuration_source'):
                sources.setdefault((point['method'],point['L_D']),point['configuration_source'])
        for ld,value in self.spec.get('new_families',{}).items():
            sources.setdefault(('PR_PAULI',int(ld)),value['configuration'])
        for value in self.spec.get('width_families',[]):
            sources[(value['method'],int(value['L_D']))]=value['configuration']
        for method, widths in self.widths.items():
            setting=self.problem['methods'][method]
            setting['requested_delta_pf_values']=[O.enc(KS._frac(w['raw'])) for w in widths]
            allowed=setting.get('L_D_values') or self.spec.get('kappa_families')
            if allowed is None and setting.get('L_D_range')and getattr(self,'catalog_facts',None):
                bound=setting['L_D_range']
                allowed=list(range(bound['min'],self.catalog_facts['fragment_count']+bound['max_offset_from_fragment_count']+1))
            bases=[(ld,loc)for (name,ld),loc in sources.items()if name==method and (allowed is None or ld in allowed)]
            families={}
            items=[]
            for ld,loc in bases:
                base=reader.read(loc);base=base.get('configuration',base)
                for formula in (setting.get('generation')or{}).get('formulas',['2nd']):
                    for width in widths:
                        key=W.width_family_key(method,ld,formula,width['raw'],base['spec']['finite_K'])
                        family=dict(method=method,L_D=ld,formula=formula,K=base['spec']['finite_K'],
                            requested_delta_pf=width['raw'],delta_pf=O.enc(KS._frac(width['raw'])),
                            base_source=loc,status='MAPPER_PENDING')
                        self.state['families'][key]=family
                        ratio=KS._frac(width['raw'])/KS._frac(base['spec']['h'])
                        items.append(dict(item_id=key,base=loc,changes=dict(formula=formula,delta_over_h=str(ratio))))
                        families[key]=(family,base['spec'])
            if items:
                started=time.monotonic()
                self.log(event='SEED_MAPPER_START',method=method,families=len(items))
                results=self.generate_families(method,items)
                by_id={r['item_id']:r for r in results}
                if len(by_id)!=len(results) or by_id.keys()!=families.keys():
                    raise ValueError('Width mapper results differ from the requested families')
                self.log(event='SEED_MAPPER_END',method=method,families=len(items),wall_seconds=time.monotonic()-started)
                for key,(family,base) in families.items():
                    self.admit_generated_family(family,base,by_id[key])
                    self.log(event='SEED_FAMILY_END',family=key,status=family['status'],reason=family.get('reason'))
            # Bootstrap a missing split through D2's ordinary preparation entry.
            # A template supplies only the fixed time grid / binding and a
            # metadata kappa.  The target split supplies its own descriptor and
            # lambda before W chooses any scientific schedule.
            templates=[loc for (name,_),loc in sources.items()if name==method]
            template=None if not templates else reader.read(templates[0])
            if template is not None:template=template.get('configuration',template)
            bootstrap_K=(setting.get('generation') or {}).get('finite_K') if template is None else template['spec']['finite_K']
            for ld in allowed or []:
                if not any(v['method']==method and v['L_D']==ld for v in self.state['families'].values()):
                    for formula in(setting.get('generation')or{}).get('formulas',['2nd']):
                        for width in widths:
                            key=W.width_family_key(method,ld,formula,width['raw'],bootstrap_K)
                            f=dict(method=method,L_D=ld,formula=formula,K=bootstrap_K,
                                requested_delta_pf=width['raw'],delta_pf=O.enc(KS._frac(width['raw'])),
                                status='NEEDS_FAMILY_CONFIGURATION_SOURCE',
                                reason='No same-method saved template: fixed H/state/snapshot/h/M and metadata kappa are required')
                            self.state['families'][key]=f
                            if method == 'PR_DF' and ld == 0:
                                f.update(status='UNSUPPORTED_MAPPER', reason='NONEMPTY_DF_HEAD_TAIL_REQUIRED')
                            elif templates:
                                source=templates[0]
                                f.update(status='FAMILY_PREPARATION_PENDING',template_source=source,template=template,
                                    preparation_request=dict(method=method,template_source=source,
                                        selection=dict(stage='FAMILY_CONSTANTS_METADATA_ONLY',family=key),
                                        design=dict(L_D=ld,formula=formula,K=template['spec']['finite_K'],delta_pf=width['raw'],
                                            kappa=str(KS._frac(template['spec']['kappa']['value'])))))
        # Only saved evaluations with this exact method/formula/requested width
        # seed a family; kappa boundaries and Q3 never cross widths implicitly.
        for p in catalog['candidates']:
            if p['method']not in self.methods or p.get('kappa')is None or p.get('delta_pf')is None:continue
            family=W.width_family_key(p['method'],p['L_D'],p['formula'],p['delta_pf'],p.get('finite_K')or 2)
            if self.state['families'].get(family,{}).get('status')!='READY_METADATA':continue
            k=KS._frac((p['kappa'].get('value')if isinstance(p['kappa'],dict)and'value'in p['kappa']else p['kappa']))
            if not W.kappa_allowed(self.wide_for(p['method']), k):
                continue
            same=next((v for v in self.state['points']if v['family']==family and Fraction(v['kappa'])==k),None)
            if same is None:
                same=dict(family=family,method=p['method'],L_D=p['L_D'],kappa=str(k),status='EVALUATED',
                    configuration_id=p['configuration_id'],best=None,saved_point_ids=[])
                self.state['points'].append(same)
            same['saved_point_ids'].append(p['point_id'])
            if p.get('status')=='FEASIBLE' and p.get('conditional_cost')is not None:
                cost=O.frac(p['conditional_cost'])
                if same['best']is None or cost<O.frac(same['best']):same['best']=O.enc(cost)
        if (self.spec.get('candidate_completion') or {}).get('resume_preparation_from'):
            self.seed_ready_continuation()
        self.state['seeded']=dict(UTC=K.now(),mapper='METHOD_SPECIFIC_EXISTING_GENERATOR',
            old_single_width_state_untouched=True)
        self.save()

    def seed_ready_continuation(self):
        comp = self.completion_api()
        for key, rec in comp.state['records'].items():
            if rec.get('status') != 'READY_FOR_SEARCH' or rec.get('method') not in self.methods:
                continue
            cfg = read_loc(rec['configuration'])
            method, spec = rec['method'], cfg['spec']
            kappa = KS._frac(spec['kappa']['value'])
            family = W.width_family_key(method, spec['L_D'], spec['formula'], spec['delta_pf'], spec['finite_K'])
            if not W.kappa_allowed(self.wide_for(method), kappa) or self.state['families'].get(family, {}).get('status') != 'READY_METADATA':
                continue
            point = dict(family=family, method=method, L_D=spec['L_D'], kappa=str(kappa),
                         status='REGISTERED', configuration_id=cfg['id'],
                         completion=dict(status='READY_FOR_SEARCH', key=key))
            if any(p['family'] == family and Fraction(p['kappa']) == kappa for p in self.state['points']):
                continue
            try:
                self.check_completed_configuration(point, rec)
                comp.verify_continuation_inputs(rec)
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.log(event='CONTINUATION_INPUT_HOLD', family=family, key=key, reason=str(error))
                continue
            self.state['points'].append(point)
            self.log(event='CONTINUATION_INPUT_READY', family=family, key=key, configuration_id=cfg['id'])

    def bootstrap_families(self):
        pending=[f for f in self.state['families'].values()if f['status']=='FAMILY_PREPARATION_PENDING']
        if not pending:return
        comp=self.completion_api();approved=self.spec.get('science_approved')is True and not self.a.plan_only
        def prepare_family(f):
            request=f['preparation_request'];started=time.monotonic()
            self.log(event='FAMILY_BOOTSTRAP_START',family=request['selection']['family'])
            if hasattr(comp,'prepare_family'):result=comp.prepare_family(request,allow_science=approved)
            else:result=comp.plan(request)
            f['preparation']=result;config=result.get('configuration')
            self.log(event='FAMILY_BOOTSTRAP_END',family=request['selection']['family'],status=result['status'],
                reason=result.get('reason'),wall_seconds=time.monotonic()-started)
            if config and 'commit'in config and 'path'in config:config=read_loc(config)
            if not config:
                f['reason']=result.get('reason',result['status'])
                if result['status'] == 'UNSUPPORTED_MAPPER':
                    f['status'] = 'UNSUPPORTED_MAPPER'
                return
            config=config.get('configuration',config);s=config['spec'];template=f['template']['spec']
            try:
                if (KS._frac(s['delta_pf'])!=KS._frac(f['requested_delta_pf'])or
                        KS._frac(s['h'])!=KS._frac(template['h'])or s['M']!=template['M']or
                        s['formula']!=f['formula']or s['finite_K']!=f['K']or s['split']['L_D']!=f['L_D']):
                    raise ValueError('FAMILY_PREPARATION_CHANGED_REQUESTED_DESIGN_OR_TIME_GRID')
                fields=('hamiltonian','state','snapshot','compiler','control')
                if any(s.get('binding',{}).get(k)!=template.get('binding',{}).get(k)for k in fields):
                    raise ValueError('FAMILY_PREPARATION_CHANGED_FIXED_BINDING')
                rows=KS.constants(config)
                if KS.R_vector(rows,KS._frac(s['kappa']['value']))!=[r['R_outer']for r in sorted(config['rounds'],key=lambda r:r['m'])]:
                    raise ValueError('FAMILY_PREPARATION_MAPPER_MISMATCH')
                f.update(status='READY_METADATA',configuration=config,rows=KS.encode_rows(rows),
                         source=result.get('configuration_source'),metadata_kappa_not_a_scientific_candidate=True)
                f.pop('reason',None)  # the fulfilled preparation need is no longer a blocking reason
            except(ValueError,KeyError,TypeError)as exc:
                f.update(status='HELD_FAMILY_MISMATCH',reason=str(exc))

        if self.spec.get('search_overrides', {}).get('controller_parallel') is True:
            from concurrent.futures import ThreadPoolExecutor
            from trottertracks.rpe_q3_resources import controller_cpu_count
            groups = {}
            for family in pending:
                groups.setdefault((family['method'], family['L_D']), []).append(family)

            def prepare_split(families):
                for family in families:
                    if (self.run/'STOP_REQUESTED.json').exists():
                        break
                    prepare_family(family)

            with ThreadPoolExecutor(max_workers=min(len(groups), controller_cpu_count()),
                                    thread_name_prefix='rpe-family-inputs') as pool:
                for _ in pool.map(prepare_split, groups.values()):
                    pass
        else:
            for family in pending:
                prepare_family(family)

    def families(self):
        return {key:W.Family(key,rows['rows'])for key,rows in self.state['families'].items()
                if rows['status']=='READY_METADATA'}

    def wide_for(self,method):
        overrides=getattr(self,'spec',{}).get('search_overrides',{}).get('schedule_wide_by_method',{})
        return overrides.get(method,self.wide)

    def point(self, family, kappa):
        return next((p for p in self.state['points']if p['family']==family and Fraction(p['kappa'])==Fraction(kappa)),None)

    def model_floor(self, families):
        """Use the same active price model to reproduce a saved feasible cost."""
        if getattr(self,'spec',{}).get('search_overrides',{}).get('common_pauli_price_floor') is not True:
            return None,None
        configs={key:self.state['families'][key]['configuration'] for key in families
                 if all(r['J']==1 for r in families[key].rows)}
        if not configs or not any(p.get('best_MSE_upper') is not None and p.get('best') is not None
                                  for p in self.state['points'] if p['family'] in configs):
            return None,None
        lookup=getattr(self.completion_api(),'common_pauli_model_terms',None)
        if lookup is None:
            return None,None
        try:
            terms=lookup(configs)
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
            return None,None

        def prices(key,R):
            return [one+slope*(r-1) for (one,slope),r in zip(terms[key],(r for r in R for _ in O.AXES))]

        feasible=[]
        tau=O.tau(self.problem)
        for p in self.state['points']:
            key=p['family']
            if (key not in terms or p.get('best_MSE_upper') is None or p.get('best') is None
                    or not 0<=O.frac(p['best_MSE_upper'])<=tau):
                continue
            nx,ny=p.get('best_N_X'),p.get('best_N_Y')
            if nx is None or ny is None or len(nx)!=13 or len(ny)!=13:
                continue
            vector=O.interleave(nx,ny)
            if any(type(n)is not int or n<1 for n in vector):
                continue
            own=prices(key,families[key].R(Fraction(p['kappa'])))
            cost=O.cost(vector,own)
            if cost==O.frac(p['best']):
                feasible.append(cost)
        if not feasible:
            return None,None
        return lambda key,R:sum(prices(key,R)) if key in terms else None,min(feasible)

    def plan(self):
        families=self.families();points=[]
        for p in self.state['points']:
            costs=[O.frac(v)for v in(p.get('best'),p.get('confirmed_all_best'))if v is not None]
            best=min(costs)if costs else None
            status=('FEASIBLE'if best is not None else 'NO_FEASIBLE_FOUND'if p['status']=='EVALUATED'
                    else 'UNAVAILABLE'if p['status']in('HELD','PRUNED')else'PENDING')
            points.append(dict(family=p['family'],kappa=Fraction(p['kappa']),status=status,
                best=best,min_lower_over_tau=p.get('min_lower_over_tau')))
        # The existing rule chooses schedules within each method. An incumbent
        # from another method must not close this method's refinement window.
        proposals=[];decisions=[];reports={};family_reports={}
        methods=list(dict.fromkeys(self.state['families'][key]['method']for key in families))
        for method in methods:
            scoped={key:f for key,f in families.items()if self.state['families'][key]['method']==method}
            scaled = getattr(self, 'scaled_shots', False)
            floor,verified_best=self.model_floor(scoped) if method=='PR_PAULI' and not scaled else (None,None)
            selection = self.wide_for(method)
            if scaled:
                selection = dict(selection, stop_by_observed_trend=False)
            grid = W.kappa_values(selection)
            if (grid is not None and not self.a.plan_only and self.spec.get('science_approved') is True
                    and self.work_conserving_search and self.live_ready_refill
                    and (self.spec.get('executor', {}).get('distributed') or {}).get('preparation')):
                # An explicit grid does not need a result to select its next
                # point. Keep the independent preparation lanes supplied even
                # when an earlier point (or its split) is still running.
                # complete_points still limits concurrent work and owns reuse.
                selection = dict(selection, per_decision=len(scoped) * len(grid))
            chosen,why,summary=W.plan_wide(scoped,[p for p in points if p['family']in scoped],selection,
                                          floor,price_floor_best=verified_best)
            proposals.extend(chosen);decisions.extend(why);reports[method]=summary;family_reports.update(summary['families'])
        report=dict(per_method=reports,global_best_by_method={m:r['global_best']for m,r in reports.items()},
            global_best=next(iter(reports.values()))['global_best']if len(reports)==1 else None,families=family_reports,
            scope='INDEPENDENT_METHOD_MINIMA',stop=None if proposals else
                'WAIT_FOR_PENDING'if any(r['stop']=='WAIT_FOR_PENDING'for r in reports.values())else'RULE_EXHAUSTED')
        for proposal in proposals:
            family=proposal['family']
            if self.point(family,proposal['kappa'])is not None:continue
            f=self.state['families'][family]
            request=dict(method=f['method'],selection=dict(rule=W.RULE,family=family,reason=proposal['reason']),
                design=dict(L_D=f['L_D'],formula=f['formula'],K=f['K'],delta_pf=f['requested_delta_pf'],
                            kappa=str(proposal['kappa'])))
            if f.get('mapper_source') and f.get('mapper_source')==(self.spec.get('candidate_completion') or {}).get('df_max_width_source'):
                request['df_max_width_source']=f['mapper_source']
                if f.get('template_source'):request['template_source']=f['template_source']
                if f.get('tail_descriptor'):request['tail_descriptor']=f['tail_descriptor']
            self.state['points'].append(dict(family=family,method=f['method'],L_D=f['L_D'],
                requested_delta_pf=f['requested_delta_pf'],kappa=str(proposal['kappa']),
                status='COMPLETION',completion_request=request,request_reason=proposal['reason']))
        self.state['history'].append(dict(cycle=self.state['cycle'],decisions=decisions,report=report,
            proposals=[dict(family=p['family'],kappa=str(p['kappa']),reason=p['reason'])for p in proposals]))
        return proposals,report

    def completion_api(self):
        if self.completion is None:
            from trottertracks.pr_candidate_completion import Completion
            settings=dict(self.spec.get('candidate_completion')or{})
            settings.setdefault('price_registry_file',self.problem.get('price_rules',{}).get('registry_file'))
            search = dict(self.problem['search'], **self.spec.get('search_overrides', {}))
            if search.get('exploratory_cost') is True:
                settings['exploratory_cost'] = True
            settings['plan_only']=self.a.plan_only or self.spec.get('science_approved')is not True
            self.completion=Completion(self.run/'completion',settings)
        return self.completion

    def complete_points(self, stop_requested=None, announce_ready=None, stop_on_error=None,
                        wait_for_preparation=None, *, continuous=False):
        comp=self.completion_api()
        if hasattr(comp,'ingest_price_results'):comp.ingest_price_results()
        if hasattr(comp,'reconcile_committed_df_holdout'):
            from trottertracks.pr_candidate_completion import Held
            for point in self.state['points']:
                completion = point.get('completion') or {}
                record = comp.state['records'].get(completion.get('key'))
                if (point['status'] == 'HELD' and point.get('reason') ==
                        'COMPLETED_CONFIGURATION_MISMATCH:DIFFERENT_SPLIT_LAMBDA_SOURCE'
                        and record and record['status'] == 'READY_FOR_SEARCH' and record['method'] == 'PR_DF'
                        and record['request'] == point.get('completion_request')):
                    try:
                        self.check_completed_configuration(point, record)
                        comp.verify_continuation_inputs(record)
                    except (Held, ValueError, KeyError, TypeError, OSError, subprocess.CalledProcessError) as error:
                        point['descriptor_recheck'] = str(error)
                        continue
                    point.update(status='REGISTERED', previous_hold=point.pop('reason'),
                                 configuration_id=record['configuration_id'])
                    self.checkpoint_point(point)
                    self.log(event='CONTINUATION_INPUT_READY', family=point['family'], key=record['key'],
                             configuration_id=record['configuration_id'])
                    if announce_ready is not None:
                        announce_ready(point['method'], record)
                    continue
                if point['status'] == 'HELD' and record and (
                        comp.reconcile_committed_df_holdout(record)
                        or getattr(comp, 'recoverable_transport', lambda _: False)(record)):
                    point.update(status='WAITING_INPUTS',reason=record['reason'])
        # INIT may expose a whole far-kappa ladder at once. Distributed
        # preparation refills bounded lanes as candidates finish; other callers
        # keep a single per-decision wave.
        pending=[p for p in self.state['points']if p['status']in(
            'COMPLETION','WAITING_INPUTS','PRICE_ACTION_UNAVAILABLE','PRICE_NOT_EVALUATED')]
        total_pending = len(pending)
        priced_rechecks = (
            getattr(comp, 'priced_pauli_recheck_keys', set)() if continuous else set())
        blocked = comp.price_deferred_keys() if hasattr(comp, 'price_deferred_keys') else set()
        owned = comp.deferred_stage_keys() if hasattr(comp, 'deferred_stage_keys') else set()
        excluded = blocked | owned
        if excluded:
            from trottertracks.pr_candidate_completion import key
            pending = [p for p in pending if key(p['completion_request']) not in excluded
                       and (p.get('completion') or {}).get('key') not in excluded]
        cycle=self.state['cycle'];limit=W.settings(self.wide)['per_decision']
        if continuous:
            pending = [p for p in pending if p.get('input_dispatch_cycle') != cycle]
        wave=self.state.get('preparation_wave')
        if wave is None or wave['cycle']!=cycle:
            wave=self.state['preparation_wave']=dict(cycle=cycle,dispatched={})
        selected=[];counts={}
        review_order=self.spec.get('search_overrides',{}).get('review_search_order') is True
        for method in self.methods:
            queues={}
            for p in pending:
                if p['method']==method:queues.setdefault(p['family'],[]).append(p)
            costs={};loads={};dear=set()
            incumbents=[O.frac(p['best']) for p in self.state['points']
                        if p.get('method')==method and p.get('best') is not None]
            incumbent=min(incumbents) if review_order and incumbents else None
            for family in queues:
                members=[p for p in self.state['points']if p.get('family')==family]
                saved=[O.frac(v)for p in members for v in(p.get('best'),p.get('confirmed_all_best'))if v is not None]
                costs[family]=min(saved)if saved else None
                loads[family]=sum(p.get('input_dispatches',0)for p in members)
                if incumbent is not None:
                    anchors=[Fraction(p['kappa']) for p in members if p.get('own_price_floor') is not None
                             and O.frac(p['own_price_floor'])>=incumbent]
                    if anchors:
                        dear.update(id(p) for p in queues[family] if Fraction(p['kappa'])>=min(anchors))
                queues[family].sort(key=lambda p:(id(p) in dear,p.get('input_dispatches',0)))
            known=sorted((f for f in queues if costs[f]is not None),key=lambda f:costs[f])
            unknown=[f for f in queues if costs[f]is None]
            order=[]
            for i in range(max(len(known),len(unknown))):
                if i<len(known):order.append(known[i])
                if i<len(unknown):order.append(unknown[i])
            # Fairness is only a dispatch order, not a price estimate or veto.
            order.sort(key=lambda f:loads[f])
            if self.work_conserving_search:
                # Diversify the next READY supply across requested widths, while
                # keeping the existing family and kappa order within each width.
                by_width = {}
                for family in order:
                    delta = KS._frac(self.state['families'][family]['requested_delta_pf'])
                    by_width.setdefault(delta, []).append(family)
                from itertools import zip_longest
                order = [family for level in zip_longest(*by_width.values()) for family in level if family is not None]
            available = sum(map(len, queues.values())) if continuous else max(0, limit-wave['dispatched'].get(method,0))
            chosen=[]
            while len(chosen)<available and any(queues.values()):
                ready=[family for family in order if queues[family] and id(queues[family][0]) not in dear]
                for family in ready or order:
                    if queues[family]and len(chosen)<available:chosen.append(queues[family].pop(0))
            selected.extend(chosen);counts[method]=len(chosen)
            if not continuous:
                wave['dispatched'][method]=wave['dispatched'].get(method,0)+len(chosen)
        if priced_rechecks:
            from trottertracks.pr_candidate_completion import key
            def priced_recheck(point):
                recorded = (point.get('completion') or {}).get('key')
                return (recorded or key(point['completion_request'])) in priced_rechecks

            priced_points = {id(point) for point in selected if priced_recheck(point)}
            selected.sort(key=lambda point: id(point) not in priced_points)
        else:
            priced_points = set()
        axes = self.spec.get('search_overrides', {}).get('coarse_axes', {})
        if axes:
            selected.sort(key=lambda p: (
                id(p) not in priced_points,
                W.coarse_priority(axes, p['method'], dict(L_D=p['L_D'],
                    delta_pf=self.state['families'][p['family']]['requested_delta_pf'],
                    kappa=p['kappa']))))
        # An explicitly selected predecessor is reconciled ahead of new grid
        # points. This changes only the order: normal ownership/delivery gates
        # still decide whether its existing result or a successor is usable.
        recovery = getattr(comp, 'selected_signal_recovery_requests', lambda: ())()
        if recovery:
            selected.sort(key=lambda p: p.get('completion_request') not in recovery)
        reason = ('CONTINUOUS_PER_METHOD_WIDTH_INTERLEAVED_FAMILY_ROUND_ROBIN' if continuous else
                  'PER_DECISION_PER_METHOD_WIDTH_INTERLEAVED_FAMILY_ROUND_ROBIN' if self.work_conserving_search else
                  'EXISTING_PER_DECISION_PER_METHOD_FAMILY_ROUND_ROBIN')
        self.log(event='PREPARATION_WAVE_START',dispatch_reason=reason,per_method_limit=limit,
            selected_by_method=counts,pending_before=total_pending,deferred=total_pending-len(selected),
            waiting_unchanged_price=len(blocked), waiting_original_attempt=len(owned),
            selected=[dict(method=p['method'],family=p['family'],kappa=p['kappa'])for p in selected])
        def prepare_points(points):
            for p in points:
                if ((wait_for_preparation is not None and not wait_for_preparation())
                    or (stop_requested is not None and stop_requested())):
                    self.log(event='PREPARATION_YIELD_AFTER_FATAL');break
                p['input_dispatches']=p.get('input_dispatches',0)+1
                if continuous:
                    p['input_dispatch_cycle'] = cycle
                started=time.monotonic();self.log(event='CANDIDATE_INPUTS_START',family=p['family'],kappa=p['kappa'])
                if self.a.plan_only or self.spec.get('science_approved')is not True:
                    record=comp.plan(p['completion_request'])
                else:record=comp.complete(p['completion_request'])
                self.log(event='CANDIDATE_INPUTS_END',family=p['family'],kappa=p['kappa'],status=record['status'],
                    reason=record.get('reason'),wall_seconds=time.monotonic()-started)
                p['completion']=dict(status=record['status'],key=record.get('key'),reason=record.get('reason'),
                                     configuration_id=record.get('configuration_id'),
                                     signal_plan=record.get('signal_plan'),price_plan=record.get('price_plan',record.get('price_quote')),
                                     calibration_requests=record.get('calibration_requests'),readiness=record.get('readiness'),
                                     rounds=record.get('rounds'))
                if record.get('conflicting_signal_rounds'):
                    p['completion']['conflicting_signal_rounds']=record['conflicting_signal_rounds']
                if record.get('price_actions'):
                    p['completion']['price_actions']=record['price_actions']
                if record.get('preparation_recovery'):
                    p['completion']['preparation_recovery']=record['preparation_recovery']
                for field in ('failure_class','retry_protected'):
                    if field in record:p['completion'][field]=record[field]
                if record['status']=='READY_FOR_SEARCH':
                    try:self.check_completed_configuration(p,record)
                    except (ValueError,KeyError,TypeError) as exc:
                        p.update(status='HELD',reason='COMPLETED_CONFIGURATION_MISMATCH:'+str(exc));self.checkpoint_point(p);continue
                    p.update(status='REGISTERED',configuration_id=record['configuration_id'])
                    self.checkpoint_point(p)
                    if announce_ready is not None:
                        announce_ready(p['method'], record)
                elif record['status']=='SELECTED'and str(record.get('reason','')).startswith('ORCHESTRATOR_YIELD_BEFORE_STAGE:'):
                    p.update(status='COMPLETION',reason=record['reason'])
                    self.checkpoint_point(p)
                    self.log(event='PREPARATION_YIELD_AFTER_FATAL',family=p['family'],stage=record.get('interrupted_before_stage'))
                    break
                elif record['status'] in ('PRICE_ACTION_UNAVAILABLE', 'PRICE_NOT_EVALUATED'):
                    p.update(status=record['status'],reason=record['reason'])
                elif (record.get('conflicting_signal_rounds') or record['status'].startswith('STOP')
                      or str(record.get('reason','')).startswith(('PREPARATION_REQUEST_MISMATCH:',
                                                                 'PREPARATION_RESULT_MISMATCH:'))
                      or 'NUMERICAL_GATE' in str(record.get('reason','')) or
                      getattr(self,'rule_stop',False) and (record.get('retry_protected',record.get('scientific_failure')))):
                    p.update(status='HELD',reason=record.get('reason',record['status']))
                else:
                    p.update(status='WAITING_INPUTS',reason=record.get('reason',record['status']))
                self.checkpoint_point(p)
        if announce_ready is None:
            prepare_points(selected)
        elif selected:
            # Keep one split in order to reuse its descriptor and signals.
            # Pauli also needs the distributed provider's isolated attempts.
            from collections import Counter, deque
            from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
            from contextlib import ExitStack
            parallel_methods={'PR_DF'}
            if getattr(comp,'parallel_pauli_preparation',lambda:False)():
                parallel_methods.add('PR_PAULI')
            groups = {}
            for point in selected:
                method = point['method']
                split = self.state['families'][point['family']]['L_D'] if method in parallel_methods else None
                if method in parallel_methods:
                    classify = getattr(comp, 'preparation_dependency_key', None)
                    dependency = classify(point['completion_request']) if classify is not None else split
                    # Keep only unresolved common prerequisites in one lane.
                    # Independent candidates share immutable descriptor inputs,
                    # while Completion owns the exact physical/price locks.
                    split = (split, id(point)) if dependency is None or id(point) in priced_points else dependency
                groups.setdefault(method, {}).setdefault(split, []).append(point)
            if self.spec.get('search_overrides', {}).get('controller_parallel') is True:
                from trottertracks.rpe_q3_resources import controller_cpu_count
                parallel = controller_cpu_count()
            else:
                parallel = int((self.spec.get('candidate_completion') or {}).get('parallel', 1))
            lane_limit = parallel if continuous else min(limit, parallel)
            dynamic_dependencies = getattr(comp, 'preparation_dependency_key', None)
            lanes = {m: max(1, min(lane_limit, sum(map(len,splits.values())) if dynamic_dependencies else len(splits))
                                 if m in parallel_methods else 1)
                     for m, splits in groups.items()}
            with ExitStack() as stack:
                pools = {method: stack.enter_context(ThreadPoolExecutor(max_workers=workers))
                         for method, workers in lanes.items()}
                tasks = []
                try:
                    if continuous:
                        # A preparation may be queued while Q3 owns every slot.
                        # Only these bounded, independent candidates can submit
                        # stages. The common executor prioritizes Q3 on dispatch.
                        by_point = {id(point): (method, split) for method, splits in groups.items()
                                    for split, points in splits.items() for point in points}
                        backlog = deque(selected)
                        active = {}
                        watching = {}
                        recovered_receipts = set()
                        bridge = getattr(comp, 'stage_executor', None)
                        store = getattr(getattr(bridge, 'executor', None), 'store', None)
                        revision = None
                        max_active = sum(lanes.values())
                        while backlog or active or watching:
                            if stop_requested is not None and stop_requested():
                                backlog.clear()
                                watching.clear()
                            if watching and store is not None:
                                # A result may arrive after its candidate yielded
                                # an operational wait. Revisit only changed queue
                                # evidence, without waiting for this whole wave.
                                import sqlite3
                                try:
                                    with store._connection() as db:
                                        changed = db.execute('SELECT max(sequence) FROM events').fetchone()[0]
                                except sqlite3.Error:
                                    changed = revision
                                if changed != revision:
                                    revision = changed
                                    for point_id, point in list(watching.items()):
                                        evidence = point.get('completion', {}).get('preparation_recovery') or {}
                                        job = evidence.get('job_id')
                                        row = store.row(job) if job else None
                                        if not row or row['state'] != 'ACCEPTED':
                                            continue
                                        receipt = (point_id, job, row['attempt_id'], row['completion_sha256'])
                                        if receipt in recovered_receipts:
                                            watching.pop(point_id, None)
                                            continue
                                        record = comp.state['records'].get(point['completion'].get('key'))
                                        if record and comp.recoverable_transport(record):
                                            recovered_receipts.add(receipt)
                                            watching.pop(point_id, None)
                                            point['status'] = 'WAITING_INPUTS'
                                            self.checkpoint_point(point)
                                            backlog.append(point)
                                            self.log(event='READY_RECEIPT_REQUEUED',job=job,family=point['family'])
                            busy = {by_point[id(p)] for p in active.values()}
                            used = Counter(method for method, _ in busy)
                            for _ in range(len(backlog)):
                                if len(active) >= max_active:
                                    break
                                point = backlog.popleft()
                                method, split = by_point[id(point)]
                                if dynamic_dependencies is not None and method in parallel_methods and continuous:
                                    dependency = dynamic_dependencies(point['completion_request'])
                                    resolved = (point['family'], id(point)) if dependency is None or id(point) in priced_points else dependency
                                    split = resolved
                                    by_point[id(point)] = (method, split)
                                if used[method] >= lanes[method] or (method, split) in busy:
                                    backlog.append(point)
                                    continue
                                active[pools[method].submit(prepare_points, [point])] = point
                                busy.add((method, split))
                                used[method] += 1
                                wave['dispatched'][method] = wave['dispatched'].get(method, 0) + 1
                            if not active and not backlog:
                                break
                            if active:
                                finished, _ = wait(active, return_when=FIRST_COMPLETED,
                                                   timeout=1 if watching else None)
                                for task in finished:
                                    point = active.pop(task)
                                    task.result()
                                    evidence = point.get('completion', {}).get('preparation_recovery') or {}
                                    if (store is not None and evidence.get('job_id')
                                            and point.get('status') in ('WAITING_INPUTS','HELD')):
                                        watching[id(point)] = point
                                        revision = None
                    else:
                        for method, splits in groups.items():
                            tasks.extend(pools[method].submit(prepare_points, points) for points in splits.values())
                        for task in as_completed(tasks):
                            task.result()
                except BaseException:
                    if stop_on_error is not None:
                        stop_on_error()
                    raise
        self.log(event='PREPARATION_WAVE_END',dispatch_reason=reason,selected_by_method=counts,
            remaining_pending=sum(p['status']in('COMPLETION','WAITING_INPUTS')for p in self.state['points']))

    def check_completed_configuration(self, point, record):
        """Metadata check against the mapper that set this family's kappa intervals."""
        config=record['configuration']
        if 'commit'in config and 'path'in config:config=read_loc(config)
        config=config.get('configuration',config);s=config['spec'];f=self.state['families'][point['family']]
        base=f['configuration']['spec'];rows=KS.constants(config)
        if KS.encode_rows(rows)!=f['rows']:raise ValueError('DIFFERENT_LAMBDA_TIME_GRID_OR_PF_SCHEDULE')
        if ([r['R_outer']for r in sorted(config['rounds'],key=lambda r:r['m'])]
                !=KS.R_vector(rows,Fraction(point['kappa']))):raise ValueError('DIFFERENT_R_SCHEDULE')
        if (KS._frac(s['delta_pf'])!=KS._frac(f['requested_delta_pf'])or
                KS._frac(s['h'])!=KS._frac(base['h'])or s['M']!=base['M']or
                s['formula']!=f['formula']or s['finite_K']!=f['K']or
                s['split']['L_D']!=f['L_D']):raise ValueError('DIFFERENT_REQUESTED_DESIGN')
        for section in('binding','law','profile'):
            if s.get(section)!=base.get(section):raise ValueError('DIFFERENT_'+section.upper())
        for field in('scalar','one_body_policy','tail_dictionary','lambda_source'):
            if s['split'].get(field) == base['split'].get(field):
                continue
            if field == 'lambda_source' and point['method'] == 'PR_DF' and equivalent_df_descriptor(config, f['configuration']):
                continue
            raise ValueError('DIFFERENT_SPLIT_'+field.upper())

    def context(self, problem):
        return O.prepare(ROOT,problem,methods=self.methods,generate_configurations=False,
                         preparation_dir=self.run/'preparation',saved_catalog=self.saved_catalog(problem))

    def confirm_completed(self, context, executor, result):
        """Existing ledger checks after finish, including unadopted speculation.

        No Q3/collect/price fit is run. Rule-adopted best stays separate; only
        source/identity-checked outcomes with consistent exact bounds and own
        request prices enter the confirmed minimum, regardless of gain size.
        """
        import hashlib
        tau=context['tau'];methods={}
        if getattr(self, 'scaled_shots', False):
            # Prior/free-vector results cannot update a multiplier-grid minimum.
            for method, r in result.items():
                r.setdefault('rule_adopted_best', r['best'])
                r['confirmed_all_best'] = (None if r['scaled_shots'].get('cost_stage') == 'EXPLORATORY_SHORTLIST'
                                           else r['best'])
                methods[method] = dict(rule_adopted_best=r['rule_adopted_best'], confirmed_all_best=r['confirmed_all_best'],
                                       scaled_shots=r['scaled_shots'])
            self.state['method_minima'] = methods
            K.CLI.write(self.run/'CONFIRMED_MINIMA.json', K.CLI.slim(dict(
                schema='RPE_WIDTH_CONFIRMED_MINIMA_V1', tau=tau, methods=methods,
                scope='RPE_SCALED_SHOTS_V1_CONDITIONAL_GRID_ONLY', new_scientific_calls=0)))
            return
        for method,r in result.items():
            candidates=[c for c in context['candidates']if c['method']==method];ready={c['configuration_id']:c for c in candidates if c['ready']}
            view=executor.for_method(method)
            view.prior_run_dirs=list(dict.fromkeys([*view.prior_run_dirs,self.run]))
            known,refused=view.prior_records(candidates)
            adopted={(t['configuration_id'],tuple(t['vector']))for t in r.get('trace',[])}
            r['rule_adopted_best']=deepcopy(r.get('best'));confirmed=deepcopy(r.get('best'));rows=[];by_config={}
            for b in r.get('by_configuration')or[]:
                b['rule_adopted_best']=deepcopy(b.get('best'));b['confirmed_all_best']=deepcopy(b.get('best'))
                by_config[b['configuration_id']]=b
            for (cid,vector),saved in known.items():
                job=Path(saved['job']);job=job if job.is_absolute()else view.root/job
                request=O.load_json(job/'REQUEST.json');c=ready[cid];price=O.cost(vector,c['prices'])
                requested=O.frac(request['cost'])if request.get('cost')is not None else None
                lo,up=saved.get('MSE_lower'),saved.get('MSE_upper');status=saved['status']
                consistent=((status==O.PASS and up is not None and up<=tau)or
                    (status==O.FAIL and lo is not None and lo>tau)or
                    (status=='INCONCLUSIVE'and lo is not None and up is not None and lo<=tau<up))
                outcome,path=O.job_outcome(job)
                row=dict(configuration_id=cid,N_X=O.split(vector)[0],N_Y=O.split(vector)[1],status=status,
                    cost=price,request_cost=requested,price_match=requested==price,status_consistent=consistent,
                    identity='EXECUTOR_PRIOR_RECORD_CHECK_PASSED',verified=requested==price and consistent,
                    adopted_by_rule=(cid,vector)in adopted,job=str(job),outcome_file=str(path),
                    outcome_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),artificial=outcome.get('artificial')is True)
                rows.append(row)
                if not row['verified']or status!=O.PASS:continue
                best=dict(candidate=c['name'],configuration_id=cid,configuration=c['key'],N_X=row['N_X'],N_Y=row['N_Y'],
                    status=status,MSE_lower=lo,MSE_upper=up,cost=price,cost_display=O.show(price),evidence='VERIFIED_COMPLETED_JOB',
                    job=str(job),outcome_source=dict(path=str(path),sha256=row['outcome_sha256']),adopted_by_rule=row['adopted_by_rule'])
                if confirmed is None or price<O.frac(confirmed['cost']):confirmed=best
                b=by_config.get(cid)
                if b is not None and(b['confirmed_all_best']is None or price<O.frac(b['confirmed_all_best']['cost'])):
                    b['confirmed_all_best']=dict(vector=list(vector),cost=price,cost_display=O.show(price),status=status,
                        MSE_lower=lo,MSE_upper=up,source='VERIFIED_COMPLETED_JOB',job=str(job),outcome_source=best['outcome_source'],
                        adopted_by_rule=row['adopted_by_rule'])
            r['confirmed_all_best']=confirmed
            r['completed_ledger']=dict(jobs_checked=len(rows),jobs_verified=sum(row['verified']for row in rows),
                refused=refused,rows=rows,scope='EXACT_TAU_AND_OWN_PRICE_CHECK_AFTER_EXECUTOR_FINISH')
            methods[method]=dict(rule_adopted_best=r['rule_adopted_best'],confirmed_all_best=confirmed,
                rule_adopted_min=None if r['best']is None else r['best']['cost'],
                verified_all_min=None if confirmed is None else confirmed['cost'],
                jobs_checked=len(rows),jobs_verified=r['completed_ledger']['jobs_verified'],jobs_refused=len(refused))
        self.state['method_minima']=methods
        K.CLI.write(self.run/'CONFIRMED_MINIMA.json',K.CLI.slim(dict(schema='RPE_WIDTH_CONFIRMED_MINIMA_V1',tau=tau,
            methods=methods,scope='PER_METHOD_EVALUATED_SET_CONDITIONAL_PRICE_NOT_FORMAL_G',new_scientific_calls=0)))

    def live_ready_source(self, executor, completion, problem, stop_preparation, *, continuous=False):
        """Publish validated registrations to each selector's private view."""
        from contextlib import nullcontext
        from types import SimpleNamespace
        announced = {method: set() for method in self.methods}
        observed = {method: set() for method in self.methods}
        from trottertracks.rpe_preparation_round import PF_METHODS, _merge_pf, snapshot_context
        pf_methods = PF_METHODS.intersection(self.methods)
        pf_snapshots = {}
        pf_revision = dict.fromkeys(pf_methods, 0)
        pf_observed = dict.fromkeys(pf_methods, 0)
        pf_ready_by_method = {}
        state = executor.state
        condition = state.cv

        def wait_for_preparation():
            # The distributed queue gives demand Q3 first access at dispatch.
            # A direct preparation still waits before acquiring a slot itself.
            with condition:
                while not (stop_preparation.is_set() or state.failure or executor._user_stop_requested()):
                    if continuous:
                        refill = getattr(executor, 'preparation_needs_refill', None)
                        if refill is None or refill():
                            return True
                    elif executor.has_spare_slot(allow_queued_speculation=self.work_conserving_search):
                        return True
                    condition.wait(.5)  # also observes STOP written outside the executor
                return False

        def announce(method, record):
            with condition:
                executor.for_method(method).log(event='CANDIDATE_READY', configuration=record['configuration_id'],
                                                completion_key=record['key'])
                announced[method].add(record['key'])
                condition.notify_all()

        def announce_pf(context, routes):
            ready = {
                method: ({row['configuration_id']: row for row in context['candidates']
                          if row['method'] == method and row.get('ready')}, routes.get(method, ()))
                for method in pf_methods
            }
            with condition:
                for method, snapshot in ready.items():
                    if snapshot == pf_ready_by_method.get(method):
                        continue
                    pf_ready_by_method[method] = snapshot
                    pf_snapshots[method] = (context, routes)
                    pf_revision[method] += 1
                    condition.notify_all()

        def has_new(method):
            with condition:
                return not stop_preparation.is_set() and (
                    bool(announced[method] - observed[method]) or
                    (method in pf_methods and pf_revision[method] > pf_observed[method]))

        def subscriber(method, context, candidates, view):
            seen = set()
            pf_seen = 0
            # A signal-ready candidate with a missing route or an active
            # execution hold has not yet reached a live selector.
            delivered = {c['configuration_id']: c.get('prices') for c in candidates
                         if c['ready'] and not c.get('execution_needs')}
            exploratory = problem['search'].get('exploratory_cost') is True

            def pending():
                with condition:
                    return not stop_preparation.is_set() and (
                        bool(announced[method] - seen) or
                        (method in pf_methods and pf_revision[method] > pf_seen))

            def take():
                nonlocal pf_seen
                with condition:
                    keys = announced[method] - seen
                    seen.update(keys)
                    observed[method].update(keys)
                    pf = pf_snapshots[method] if method in pf_methods and pf_revision[method] > pf_seen else None
                    if pf is not None:
                        pf_seen = pf_revision[method]
                        pf_observed[method] = pf_seen
                if (not keys and pf is None) or stop_preparation.is_set():
                    return [], None
                scratch = dict(context, candidates=deepcopy(candidates), expected=dict(view.expected))
                scoped = dict(problem, execution_routes=view.routes)
                if pf is not None:
                    scratch = snapshot_context(scratch)
                    scratch, routes = _merge_pf(scratch, pf[0], dict(view.routes), pf[1], [method])
                    registered = {c['configuration_id'] for c in scratch['candidates'] if c['method'] == method}
                elif hasattr(completion, 'install_snapshot'):
                    routes, records = completion.install_snapshot(scratch, scoped, keys)
                else:
                    with getattr(completion, 'lock', nullcontext()):
                        records = deepcopy([completion.state['records'][key] for key in keys])
                    routes = completion.install(scratch, scoped, records)
                if pf is None:
                    registered = {r.get('installed_configuration_id', r.get('configuration_id')) for r in records}
                O.scope_absolute_kappa(scratch, problem)
                O.apply_published(scratch['candidates'], routes, exploratory=exploratory)
                # The installer resolves aliases. Keep existing routes and
                # bindings, including those of jobs already running.
                trial = view.for_method(method)
                trial.routes, trial.expected = routes, scratch['expected']
                added = []
                new_routes = []
                expected = dict(view.expected)
                for candidate in scratch['candidates']:
                    cid = candidate['configuration_id']
                    if (candidate['method'] != method or not candidate['ready']
                        or not cid or cid not in registered
                        or (cid in delivered and (not exploratory or delivered[cid] == candidate.get('prices')))):
                        continue
                    route = trial.route(cid)
                    if route is None or trial.refresh_execution_hold(cid):
                        continue
                    candidate['execution_needs'] = []
                    candidate.pop('execution_hold', None)
                    added.append(candidate)
                    delivered[cid] = candidate.get('prices')
                    new_routes.append(dict(route, configuration_ids=[cid]))
                    expected[cid] = trial.expected_for(cid, route)
                view.routes = dict(view.routes, **{method: new_routes + view.routes.get(method, [])})
                view.expected = expected
                if added:
                    self.log(event='LIVE_READY_INSTALLED', method=method,
                             configurations=[c['configuration_id'] for c in added])
                return added, scratch.get('saved') if pf is not None else None
            return pending, take
        return announce, wait_for_preparation, SimpleNamespace(subscribe=subscriber, has_new=has_new,
                                                              announce_pf=announce_pf)

    def cycle_executor(self, routes, expected, problem, active_spec, budgets):
        """Reuse the same run-owned queue and services across planning cycles."""
        persistent = (self.work_conserving_search and bool(active_spec['executor'].get('distributed')))
        existing = getattr(self, 'waiting_executor', None)
        if existing is not None and getattr(existing.state, 'failure', None):
            self.finish_waiting_executor()
            raise RuntimeError('Persistent executor failure: '+str(existing.state.failure))
        if (persistent and existing is not None and not existing.state.closed
                and existing.state.failure is None and not existing._user_stop_requested()):
            executor, optimize = existing, self.waiting_optimize
            if advance := getattr(executor, 'advance_cycle', None):
                advance()
            with executor.state.cv:
                executor.routes, executor.expected = routes, expected
                for method, limit in executor.state.budgets.items():
                    if method in budgets and type(limit) is int and type(budgets[method]) is int:
                        budgets[method] = max(limit, budgets[method])
                executor.state.budgets = budgets
                executor.state.cv.notify_all()
            self.waiting_executor = None
            self.log(event='CONTROLLER_CYCLE_REUSED')
        else:
            self.finish_waiting_executor()
            executor,optimize=K.CLI.make_executor(ROOT,self.run,routes,expected,problem,active_spec)
        if persistent and hasattr(executor, 'state'):
            with executor.state.cv:
                executor.state.retain_between_cycles = True
                executor.state.cycle_metric_baseline = {m:dict(counts) for m,counts in getattr(executor.state, 'by_method', {}).items()}
        return executor,optimize

    def batch(self, prepare_during=False):
        # Same V2/Executor and risk contract as the old ordinary round entry.
        # Completion selects no new designs here: the width planner already did.
        problem=deepcopy(self.problem)
        for method, widths in self.widths.items():
            problem['methods'][method]['requested_delta_pf_values']=[O.enc(KS._frac(w['raw']))for w in widths]
        search = dict(problem['search'], **self.spec.get('search_overrides', {}))
        exploratory = search.get('exploratory_cost') is True
        if exploratory:
            problem['search'] = dict(problem['search'], exploratory_cost=True, rule=search['rule'])
        context=self.context(problem);comp=self.completion_api()
        routes=K.CLI.merged_routes(problem,self.spec)
        pf_settings=self.spec.get('pf_completion') or self.problem.get('pf_completion') or {}
        distributed_prep = (self.spec.get('executor', {}).get('distributed') or {}).get('preparation')
        pf_enabled = (getattr(self, 'rule_stop', False) or search.get('controller_parallel') is True and
                      bool(distributed_prep)) and ('TRUNCATED_PF' in self.methods or
                      'FULL_PF' in self.methods and bool(pf_settings))
        if pf_enabled:
            from trottertracks import pf_optimizer_connection as PF
            routes=PF.registry_routes(routes,context['pf_evidence_registry'])
        horizon_keys = set()
        if 'PR_DF' in self.methods and search.get('rule') == 'RPE_SCALED_SHOTS_V1':
            from trottertracks.pr_df_horizon_preparation import ready_keys as saved_horizon_keys
            from trottertracks.rpe_scaled_shots import settings as horizon_settings
            horizon_keys.update(saved_horizon_keys(context['candidates'], comp.state['records'].values(),
                                                  horizon_settings(search['scaled_shots'])['M']))
        def install_ready(current_routes, snapshot=False):
            accepted={p.get('completion',{}).get('key')for p in self.state['points']if p['status']in('REGISTERED','EVALUATED')}
            accepted.update(horizon_keys.copy())
            if snapshot and hasattr(comp,'install_snapshot'):
                current_routes,records=comp.install_snapshot(context,dict(problem,execution_routes=current_routes),accepted)
            else:
                records=[r for r in comp.state['records'].values()if r.get('key')in accepted]
                current_routes=comp.install(context,dict(problem,execution_routes=current_routes),records)
            installed={r['key']:r.get('installed_configuration_id',r.get('configuration_id'))for r in records}
            for p in self.state['points']:
                canonical=installed.get(p.get('completion',{}).get('key'))
                if canonical and p.get('evaluation_configuration_id') != canonical:
                    p['evaluation_configuration_id']=canonical
                    self.checkpoint_point(p)
            return current_routes
        if exploratory and hasattr(comp, 'refresh_exploratory_prices'):
            comp.refresh_exploratory_prices()
        routes=install_ready(routes)
        O.apply_published(context['candidates'],routes, exploratory=exploratory)
        options=self.spec.get('executor',{})
        launched={m:sum((job/'LAUNCH.json').exists()or(job/'STARTING.json').exists()
            for job in(self.run/'jobs'/m).glob('*'))for m in self.methods}
        budgets={m:launched[m]+self.a.batch for m in self.methods}
        for method,budget in budgets.items():
            maximum=self.spec.get('max_new_q3_per_method',{}).get(method)
            if maximum=='UNLIMITED' and getattr(self,'rule_stop',False):budgets[method]='UNLIMITED'
            elif maximum not in(None,'UNLIMITED'):budgets[method]=min(budget,maximum)
        if getattr(self,'rule_stop',False):
            search['max_steps_per_method']=self.spec.get('max_steps_per_method')
            search['work_slice_steps']=W.settings(self.wide)['per_decision']
        # Existing V2 switch: kappa schedules are chosen by this outer width
        # planner, not by a second inner multiplier-neighbour chooser.
        search.setdefault('kappa_neighbourhood',dict(outer_planner=W.RULE))
        # The width planner still chooses designs. Continuation may refresh and
        # install its existing waiting records, without a second design chooser.
        active_spec=deepcopy(self.spec);active_spec['max_new_q3_per_method']=budgets
        if exploratory and self.work_conserving_search:
            search.setdefault('feasibility_first',True)
        shortlist = {}
        live_price = None
        if exploratory:
            shortlist = self.state.get('shortlist_price_candidates')
            if shortlist is None:
                previous_report = self.run/'ROUND_RESULT.json'
                shortlist = shortlist_price_ids(O.load_json(previous_report)['methods']) if previous_report.is_file() else {}
                self.state['shortlist_price_candidates'] = shortlist
            live_price = LivePriceWork(self.run, shortlist)
            shortlist = live_price.snapshot()
            active_spec['exploratory_price_candidates'] = [
                cid for method in ('FULL_PF', 'TRUNCATED_PF') for cid in shortlist.get(method, ())]
        active_spec['executor']=dict(options,prior_run_dirs=list(options.get('prior_run_dirs',[]))+[str(self.run)])
        if self.live_ready_refill and active_spec['executor'].get('distributed'):
            if self.work_conserving_search and (active_spec['executor']['distributed'].get('ready_queue_waves', 2) > 1):
                search['prefetch_independent_references'] = True
                discrete = all(W.kappa_values(self.wide_for(method)) is not None for method in self.widths)
                search['ready_work_at_slice'] = 'refill' if discrete else 'drain'
            active_spec['executor']['distributed'] = dict(active_spec['executor']['distributed'], live_ready_refill=True,
                work_conserving_search=self.work_conserving_search)
        active_spec['candidate_completion']=dict(self.spec.get('candidate_completion')or{},max_attempted_per_method=0)
        if exploratory:
            active_spec['pf_completion'] = dict(pf_settings, exploratory_cost=True) if pf_settings else {}
        if search.get('controller_parallel') is True and pf_enabled and pf_settings.get('mode') == 'complete':
            from trottertracks.rpe_q3_resources import controller_cpu_count
            active_spec['pf_completion'] = dict(active_spec.get('pf_completion') or pf_settings,
                                                max_candidates=controller_cpu_count(),
                                                refill_candidate_preparation=True)
        executor,optimize = self.cycle_executor(routes, context['expected'], problem, active_spec, budgets)
        # Keep the bounded history inventory when a nonpersistent caller
        # still needs a new executor or a normal cycle reuses the existing one.
        inventory = getattr(executor, 'prior_history_inventory', None)
        if inventory is not None and getattr(executor, 'optimizer_history_directory', None) is not None:
            if self.prior_history_inventory is None:
                self.prior_history_inventory = inventory
            executor.prior_history_inventory = self.prior_history_inventory
        self.queued_preparation = 0
        self.preparation_queue_advanced = False
        manager = getattr(executor, 'pbs_manager', None)
        if manager is not None and hasattr(manager, 'bind_offline_points'):
            manager.bind_offline_points(self.state['points'])
        if manager is not None and manager.hold_node:
            manager.retain_between_batches = True
            self.run_pbs_manager = manager
        preparation_pool=None
        close_accepted_recovery=None
        try:
            runtime=K.CLI.preparation_runtime(executor,self.run,active_spec)
            if runtime is not None:
                comp.bind_stage_executor(runtime)
                if hasattr(runtime,'start_accepted_recovery') and hasattr(getattr(executor,'store',None),'unadopted_preparation_successes'):
                    close_accepted_recovery=runtime.start_accepted_recovery(comp)
            if runtime is not None and runtime.settings.get('interrupted_calibrations'):
                runtime.reconcile_interrupted_calibrations()
            if runtime is not None:
                runtime.reconcile_failed_preparations()
                runtime.reconcile_pf_input_redeliveries(
                    pf_settings, pf_settings.get('interpreter', sys.executable))
            if runtime is not None and search.get('rule') == 'RPE_SCALED_SHOTS_V1':
                from trottertracks.rpe_preparation_queue import save_horizon_preparation, submit_registered_horizons
                from trottertracks.rpe_scaled_shots import settings as scaled_settings
                early = submit_registered_horizons(runtime, context['candidates'],
                    dict(scaled_settings(search['scaled_shots']), exploratory_cost=exploratory), problem['rpe']['M'])
                if early:
                    save_horizon_preparation(self.run/'HORIZON_PREPARATION.json', early, runtime.executor.store)
            def user_stopped():
                return (self.run/'STOP_REQUESTED.json').exists()

            from threading import RLock
            df_price_lock = RLock()

            def df_enabled():
                return (runtime is not None and 'PR_DF' in self.methods and self.spec.get('science_approved') is True
                        and search.get('rule') == 'RPE_SCALED_SHOTS_V1' and not context.get('artificial')
                        and problem.get('problem',{}).get('system') == 'H4'
                        and not context.get('pf_evidence_registry',{}).get('artificial'))

            def prepare_df(scope, notify=None, stop_requested=user_stopped):
                if not df_enabled():
                    return
                def ready(record):
                    horizon_keys.add(record['key'])
                    if notify:
                        notify('PR_DF',record)
                if exploratory:
                    available = {cid for candidate in scope for cid in
                                 (candidate.get('configuration_id'), candidate.get('source_configuration_id')) if cid}
                    with df_price_lock:
                        comp.prepare_shortlist_prices(set(live_price.snapshot().get('PR_DF', ())) & available,
                                                      stop_requested=stop_requested, on_ready=ready)
                horizons = scaled_settings(search['scaled_shots'])['M']
                if 13 not in horizons:
                    return
                from trottertracks.pr_df_horizon_preparation import prepare as prepare_df_horizons
                prepare_df_horizons(ROOT,runtime,comp,scope,horizons,
                                    active_spec['executor'].get('prior_run_dirs', ()),
                                    stop_requested, on_ready=ready)
            if any(hold.get('replay_exact') is True for hold in getattr(executor, 'prior_q3_holds', ())):
                prepare_during = executor.replay_prior_q3() and prepare_during
            interim=None
            if prepare_during:
                # Snapshot/install happens before starting preparation. While
                # it owns Completion state, Q3 only reads this frozen context.
                from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
                import inspect,threading
                from trottertracks.rpe_preparation_round import snapshot_context, _merge_pf
                stop_preparation=threading.Event()

                def stop_preparing():
                    stop_preparation.set()
                    with executor.state.cv:
                        executor.state.cv.notify_all()

                executor.state.failure_callback=lambda error:stop_preparing()
                comp.stop_requested=stop_preparation.is_set
                supports_stop='stop_requested'in inspect.signature(self.complete_points).parameters
                prepare_options = {'stop_requested': stop_preparation.is_set} if supports_stop else {}
                price_options = (dict(on_feasible=live_price.observe,
                                      price_work_pending=live_price.pending.is_set) if live_price else {})
                search_options = dict(price_options)
                if self.live_ready_refill:
                    continuous = self.work_conserving_search and runtime is not None
                    announce, wait_for_preparation, subscriber = self.live_ready_source(
                        executor, comp, problem, stop_preparation, continuous=continuous)
                    prepare_options.update(announce_ready=announce, stop_on_error=stop_preparing,
                                           wait_for_preparation=wait_for_preparation, continuous=continuous)
                    search_options['ready_source'] = subscriber
                df_context = (snapshot_context(context)
                              if runtime is not None and 'PR_DF' in self.methods and
                              search.get('rule') == 'RPE_SCALED_SHOTS_V1' and
                              (13 in scaled_settings(search['scaled_shots'])['M']
                               or (exploratory and shortlist.get('PR_DF'))) else None)
                parallel_families = runtime is not None and search.get('controller_parallel') is True
                preparation_pool=ThreadPoolExecutor(max_workers=1 + bool(pf_enabled) + (df_context is not None)
                                                    + parallel_families)
                stopped = lambda: stop_preparation.is_set() or user_stopped()
                notify_df = prepare_options.get('announce_ready')
                df_future = None
                if df_context is not None:
                    # Older READY parents can advance while ordinary candidate
                    # preparation is waiting for its own signal or price.
                    df_future = preparation_pool.submit(prepare_df, df_context['candidates'], notify_df, stopped)
                    original_df_parents = {c.get('configuration_id') for c in df_context['candidates']}
                initial_keys = {p.get('completion', {}).get('key') for p in self.state['points']
                                if p['status'] in ('REGISTERED', 'EVALUATED')}
                def prepare():
                    if runtime is not None and not parallel_families and not stop_preparation.is_set():
                        self.bootstrap_families()
                    if not stop_preparation.is_set():
                        self.complete_points(**prepare_options)
                    if df_context is not None and not stopped() and (changed := ready_keys() - initial_keys - horizon_keys):
                        fresh = snapshot_context(df_context)
                        comp.install_snapshot(fresh, dict(problem, execution_routes=routes), changed)
                        prepare_df([c for c in fresh['candidates']
                                    if c.get('configuration_id') not in original_df_parents], notify_df, stopped)
                pf_future=None
                self.pf_more=False
                if pf_enabled:
                    from trottertracks import pf_candidate_selection
                    trial = selected_worker_policy(self.spec)
                    if trial.get('recover_saved_price_rounds') and not active_spec.get('pf_completion'):
                        active_spec = dict(active_spec, pf_completion=dict(problem['pf_completion'],
                                                                           recover_saved_price_rounds=True))
                    args = (ROOT, problem, snapshot_context(context), active_spec, self.run, deepcopy(routes), runtime)
                    def prepare_pf():
                        stopped = lambda: stop_preparation.is_set() or executor._user_stop_requested()
                        horizons = scaled_settings(search['scaled_shots'])['M'] if runtime is not None and search.get('rule') == 'RPE_SCALED_SHOTS_V1' else ()
                        extend_pf = (runtime is not None and 13 in horizons and
                                     problem.get('problem', {}).get('system') == 'H4' and
                                     not context['pf_evidence_registry'].get('artificial'))
                        initial_context, initial_routes = args[2], args[5]
                        latest = (initial_context, initial_routes)
                        terminal_entries = {}
                        lock = threading.RLock()
                        changed = threading.Condition(lock)
                        parent_revision = 0
                        regular_done = False
                        producer_failed = threading.Event()
                        registry_changed = threading.Event()
                        producers_done = threading.Event()
                        if extend_pf:
                            from trottertracks import pf_horizon_preparation as horizon_operations
                            from trottertracks import pf_optimizer_connection as pf_registry

                            def parent_evidence(pf_context):
                                return {row['configuration_id']: entry for row in pf_context['candidates']
                                    if row['method'] in ('FULL_PF', 'TRUNCATED_PF')
                                    and row.get('configuration_id') and row.get('M', 12) == 12
                                    and not row.get('source_configuration_id')
                                    and (entry := horizon_operations.parent_entry(row))}

                            parents = parent_evidence(initial_context)

                        def include_terminal(pf_context, pf_routes, entries):
                            if not entries:
                                return pf_context, pf_routes
                            updated = snapshot_context(pf_context)
                            registry = updated['pf_evidence_registry']
                            additions = {cid: entry for cid, entry in entries.items()
                                         if entry != registry['configurations'].get(cid)}
                            if not additions:
                                return pf_context, pf_routes
                            # Use the same PF validation as prepare(), just for transferred
                            # entries. Re-enumerating every method and re-quoting every
                            # unchanged candidate can leave the workers idle for minutes.
                            delta = dict(registry, configurations=additions)
                            expected = pf_registry.hydrate(ROOT, O.Reader(ROOT), dict(problem, search=search),
                                updated['candidates'], delta, updated['facts'])
                            for cid, binding in expected.items():
                                prior = updated['expected'].get(cid) or {}
                                for field in ('total_configuration_id', 'binding_id'):
                                    if prior.get(field) is not None and binding.get(field) is not None and prior[field] != binding[field]:
                                        raise ValueError('PF registry conflicts with saved Q3 binding: ' + field)
                                updated['expected'][cid] = dict(prior, **{k: v for k, v in binding.items() if v is not None})
                            if updated.get('saved') is not None:
                                prices = {c['configuration_id']: c['prices'] for c in updated['candidates']
                                          if c['configuration_id'] in expected and c['prices'] is not None}
                                restored = O.SavedBackend([p for p in updated['catalog']['candidates']
                                                           if p['configuration_id'] in expected], prices)
                                updated['saved'].table.update(restored.table)
                            registry['configurations'].update(additions)
                            return updated, pf_registry.registry_routes(pf_routes, delta)

                        def announce_pf(pf_context, pf_routes):
                            nonlocal latest, parents, parent_revision
                            with lock:
                                latest = (pf_context, pf_routes)
                                if extend_pf and (updated := parent_evidence(pf_context)) != parents:
                                    parents = updated
                                    parent_revision += 1
                                    changed.notify_all()
                                if self.live_ready_refill:
                                    if extend_pf:
                                        registry_changed.set()
                                    else:
                                        subscriber.announce_pf(*latest)

                        def terminal_ready(cid, entry):
                            with lock:
                                if terminal_entries.get(cid) == entry:
                                    return
                                terminal_entries[cid] = entry
                                if self.live_ready_refill:
                                    registry_changed.set()

                        def publish_ready_snapshots():
                            try:
                                while not stopped() and not producer_failed.is_set():
                                    if not registry_changed.wait(.5):
                                        continue
                                    with lock:
                                        registry_changed.clear()
                                        snapshot = latest
                                        entries = dict(terminal_entries)
                                        finished = producers_done.is_set()
                                    # Route/optimizer hydration can be expensive. Publish the
                                    # newest snapshot while both preparation producers keep working.
                                    subscriber.announce_pf(*include_terminal(*snapshot, entries))
                                    if finished:
                                        return
                            except BaseException:
                                producer_failed.set()
                                raise

                        def regular():
                            if (self.live_ready_refill and continuous and
                                    (search.get('controller_parallel') is True or search.get('ready_work_at_slice') == 'refill')):
                                return pf_candidate_selection.prepare_available(
                                    *args, stop_requested=stopped, announce=announce_pf,
                                    wait_for_work=wait_for_preparation,
                                    refill_requested=getattr(executor, 'preparation_needs_refill', None))
                            result = pf_candidate_selection.prepare_next(
                                *args, stop_requested=stopped,
                                refill_requested=getattr(executor, 'preparation_needs_refill', None))
                            announce_pf(*result[:2])
                            return result

                        if not extend_pf:
                            return regular()

                        def complete_extensions():
                            seen = -1
                            terminal_stopped = lambda: stopped() or producer_failed.is_set()
                            while not terminal_stopped():
                                with changed:
                                    while (parent_revision == seen and not regular_done
                                           and not terminal_stopped()):
                                        changed.wait(.5)
                                    if parent_revision == seen or terminal_stopped():
                                        return
                                    seen = parent_revision
                                    pf_context, pf_routes = latest
                                # Yield only the watcher on new parent evidence. Original queued
                                # jobs stay owned, and the next pass reuses the same requirements.
                                horizon_operations.complete_scoped(
                                    ROOT, runtime, pf_context['candidates'], self.methods, horizons,
                                    pf_routes, pf_settings.get('interpreter', sys.executable),
                                    active_spec['executor'].get('prior_run_dirs', ()),
                                    stop_requested=terminal_stopped, on_ready=terminal_ready, exploratory=exploratory,
                                    refresh_requested=lambda: parent_revision != seen)

                        with ThreadPoolExecutor(max_workers=2,thread_name_prefix='pf-terminal-producer') as pool:
                            work = pool.submit(complete_extensions)
                            publisher = pool.submit(publish_ready_snapshots) if self.live_ready_refill else None
                            try:
                                try:
                                    result = regular()
                                finally:
                                    with changed:
                                        # A signal receipt can arrive without a new
                                        # price-qualified candidate in the snapshot.
                                        parent_revision += 1
                                        regular_done = True
                                        changed.notify_all()
                                work.result()
                            except BaseException:
                                producer_failed.set()
                                raise
                            finally:
                                producers_done.set()
                                registry_changed.set()
                            if publisher is not None:
                                publisher.result()
                        return (*include_terminal(*result[:2], terminal_entries), result[2])
                    pf_future=preparation_pool.submit(prepare_pf)

                def install_pf():
                    nonlocal context, routes, pf_future
                    pf_context,pf_routes,pf=pf_future.result()
                    context,routes=_merge_pf(context,pf_context,routes,pf_routes,self.methods)
                    executor.routes=routes;executor.expected=context['expected']
                    O.apply_published(context['candidates'],routes, exploratory=exploratory)
                    self.pf_more=pf['status']=='PREPARATION_WAVE_PROCESSED'
                    K.CLI.write(self.run/'PF_PREPARATION_STATUS.json',pf)
                    pf_future=None
                def ready_keys():
                    return {p['completion']['key'] for p in self.state['points']
                            if p['status'] in ('REGISTERED', 'EVALUATED') and p.get('completion', {}).get('key')} | horizon_keys.copy()

                installed=ready_keys()
                pf_price_submitted = set(active_spec.get('exploratory_price_candidates', ()))
                df_price_submitted = set(shortlist.get('PR_DF', ())) if df_future is not None else set()
                df_result_revision = executor.store.preparation_completion_revision() if runtime is not None and hasattr(getattr(executor,'store',None),'preparation_completion_revision') else 0
                bootstrap = preparation_pool.submit(self.bootstrap_families) if parallel_families else None
                pending=preparation_pool.submit(prepare)
                can_search=any(c['ready']for c in context['candidates'])
                while True:
                    if bootstrap is not None and bootstrap.done():
                        bootstrap.result()
                        bootstrap = None
                    if df_future is not None and df_future.done():
                        df_future.result()
                        df_future = None
                    if pf_future is not None and pf_future.done():
                        install_pf()
                        can_search=any(c['ready']for c in context['candidates'])
                        self.log(event='PF_READY_INSTALLED_DURING_PREPARATION',pr_pending=not pending.done())
                    if can_search:
                        if live_price:
                            live_price.pending.clear()
                        with K.CLI.round_phase(self.run,'WIDTH_READY_DURING_PREPARATION',on_error=lambda error:stop_preparing()):
                            interim=optimize(ROOT,dict(problem,search=search),context=context,methods=self.methods,
                                initial=self.spec.get('initial','all'),max_new=budgets,executor=executor,science_approved=True,
                                **search_options)
                        if O.shared_stop(interim):
                            stop_preparing()
                            pending.result()
                            if df_future is not None:
                                df_future.result()
                            break
                    live_refill = bool(live_price and live_price.pending.is_set())
                    if live_price and not stopped():
                        # New points may arrive while a previous producer owns
                        # its workspace. Refill after that producer returns.
                        shortlist = live_price.snapshot()
                        pf_ids = {cid for method in ('FULL_PF', 'TRUNCATED_PF') for cid in shortlist.get(method, ())}
                        df_ids = set(shortlist.get('PR_DF', ()))
                        if pf_enabled and pf_future is None and pf_ids - pf_price_submitted:
                            active_spec['exploratory_price_candidates'] = sorted(pf_ids)
                            args = (ROOT, problem, snapshot_context(context), active_spec, self.run, deepcopy(routes), runtime)
                            pf_price_submitted = pf_ids
                            pf_future = preparation_pool.submit(prepare_pf)
                        current_revision=executor.store.preparation_completion_revision() if runtime is not None and hasattr(getattr(executor,'store',None),'preparation_completion_revision') else 0
                        if df_enabled() and df_future is None and (df_ids - df_price_submitted or df_ids and current_revision!=df_result_revision):
                            df_price_submitted = df_ids
                            df_result_revision=current_revision
                            def live_df():
                                with df_price_lock:
                                    comp.prepare_shortlist_prices(set(live_price.snapshot().get('PR_DF', ())),
                                        stop_requested=stopped,
                                        on_ready=lambda record: notify_df('PR_DF', record) if notify_df else None)
                            df_future = preparation_pool.submit(live_df)
                        live_price.pending.clear()
                    if pending.done():
                        pending.result()
                        if pf_future is None and df_future is None and bootstrap is None: break
                    ready=ready_keys()
                    changed=ready-installed
                    # Extend an internal batch only inside the same adopted
                    # overall per-method budget. Finite budgets stay hard caps.
                    replenished=False;remaining_ready_work=False
                    if interim:
                        for method,r in interim.items():
                            spent=executor.launched(method);maximum=self.spec.get('max_new_q3_per_method',{}).get(method)
                            can_continue = (r['stop_reason']=='STEP_LIMIT' or
                                            (r['stop_reason']=='WORK_SLICE_YIELDED' and r['new_calls']>0))
                            if can_continue and (budgets[method]=='UNLIMITED' or spent<budgets[method]):
                                remaining_ready_work=True
                            if isinstance(budgets[method],int) and spent>=budgets[method]and(changed or r['stop_reason']in('Q3_BUDGET_EXHAUSTED','STEP_LIMIT')):
                                target=spent+self.a.batch
                                if maximum not in(None,'UNLIMITED'):target=min(target,maximum)
                                if target>budgets[method]:budgets[method]=target;replenished=True
                    if changed:
                        routes=install_ready(routes,snapshot=True);executor.routes=routes;executor.expected=context['expected']
                        O.apply_published(context['candidates'],routes, exploratory=exploratory);installed=ready
                        self.log(event='LATE_READY_INSTALLED_DURING_PREPARATION',keys=sorted(changed))
                    pf_ready = self.live_ready_refill and any(subscriber.has_new(m) for m in self.methods)
                    can_search=bool(changed or replenished or remaining_ready_work or pf_ready or live_refill)
                    if not can_search:
                        producers=[f for f in (pending,pf_future,df_future,bootstrap) if f is not None and not f.done()]
                        wait(producers,timeout=1,return_when=FIRST_COMPLETED)
                # No other thread now changes Completion or its registry.
                if exploratory and hasattr(comp, 'refresh_exploratory_prices'):
                    comp.refresh_exploratory_prices()
                routes=install_ready(routes);executor.routes=routes;executor.expected=context['expected']
                if pf_future is not None:
                    install_pf()
                O.apply_published(context['candidates'],routes, exploratory=exploratory)
            if not prepare_during:
                prepare_df(context['candidates'])
                if horizon_keys:
                    routes=install_ready(routes)
                    executor.routes=routes;executor.expected=context['expected']
                    O.apply_published(context['candidates'],routes, exploratory=exploratory)
            if interim and O.shared_stop(interim):result=interim
            else:
                with K.CLI.round_phase(self.run,'WIDTH_SEARCH_AND_Q3'):
                    result=optimize(ROOT,dict(problem,search=search),context=context,methods=self.methods,
                        initial=self.spec.get('initial','all'),max_new=budgets,executor=executor,
                        science_approved=self.spec.get('science_approved')is True,
                        **(dict(on_feasible=live_price.observe, price_work_pending=live_price.pending.is_set)
                           if live_price else {}))
            actionable = any(req['kind'] != 'HORIZON_INPUTS_MISSING'
                             for r in result.values() for req in r.get('input_requests', []))
            if not O.shared_stop(result) and actionable:
                from trottertracks.pr_candidate_completion import continue_selected_preparation
                with K.CLI.round_phase(self.run,'WIDTH_PREPARATION_CONTINUATION'):
                    result=continue_selected_preparation(ROOT,dict(problem,search=search),context,
                        active_spec,self.run,comp,executor,result,optimizer=optimize)
            if runtime is not None and not O.shared_stop(result):
                from trottertracks.rpe_preparation_queue import save_horizon_preparation, submit_horizon_preparation
                requests = [req for r in result.values() for req in r.get('input_requests', [])
                            if req['kind'] == 'HORIZON_INPUTS_MISSING']
                if requests or (self.run/'HORIZON_PREPARATION.json').is_file():
                    reports = submit_horizon_preparation(runtime, requests, exploratory=exploratory)
                    save_horizon_preparation(self.run/'HORIZON_PREPARATION.json', reports, runtime.executor.store)
                settle = getattr(executor, 'await_dispatchable_preparation', None)
                if (settle is not None and self.work_conserving_search and not getattr(self, 'pf_more', False)
                        and not any(r['new_calls'] or r['stop_reason'] == 'WORK_SLICE_YIELDED' for r in result.values())
                        and not any(p['status'] in ('COMPLETION', 'REGISTERED') for p in self.state['points'])):
                    # Give eligible work a chance to return its receipt before a
                    # no-action planning cycle closes the worker connections.
                    self.preparation_queue_advanced = bool(settle())
        except BaseException as error:
            if preparation_pool is not None:stop_preparing()
            O.record_runtime_error(self.run,error,'WIDTH_BATCH')
            if hasattr(executor,'_stop_after_failure'):executor._stop_after_failure(error)
            self.state['stop']=dict(reason='RUNTIME_EXCEPTION',error_type=type(error).__name__,message=str(error),
                error_log='RUNTIME_ERRORS.jsonl',drain='WAIT_FOR_OWNED_RUNNING_STAGES_NO_NEW_PREPARATION',UTC=K.now())
            self.log(event='BATCH_EXCEPTION',error_type=type(error).__name__,message=str(error))
            raise
        finally:
            original_error=sys.exc_info()[1];cleanup_error=None
            preparation_count = getattr(executor, 'queued_preparation_count', None)
            if (original_error is None and preparation_count is not None and self.work_conserving_search
                    and not O.shared_stop(result)):
                # The next planning cycle uses the same durable queue.
                executor.state.retain_queued_preparation = True
            keep_connected = (original_error is None and self.work_conserving_search
                              and getattr(executor, 'rpc_server', None) is not None
                              and not executor._user_stop_requested() and not O.shared_stop(result)
                              and getattr(executor.state,'failure',None) is None)
            if keep_connected:
                self.waiting_executor = executor
                self.waiting_optimize = optimize
                self.waiting_revision = self.recovery_revision(executor)
            actions = [] if keep_connected else [('EXECUTOR_DRAIN', executor.finish)]
            if close_accepted_recovery is not None:
                actions.insert(0,('ACCEPTED_PREPARATION_RECOVERY_DRAIN',close_accepted_recovery))
            if preparation_pool is not None:
                interrupted = stop_preparation.is_set() or executor._user_stop_requested()
                if interrupted:
                    stop_preparing()
                preparation_drain = ('PREPARATION_DRAIN',
                                     lambda: preparation_pool.shutdown(wait=True, cancel_futures=interrupted))
                # Release queue waiters before joining producers on Stop.
                # Normal batches still let producers publish their final results first.
                actions.insert(len(actions) if interrupted else 0, preparation_drain)
            for phase,close in actions:
                try:close()
                except BaseException as error:
                    O.record_runtime_error(self.run,error,phase)
                    cleanup_error=cleanup_error or error
            if original_error is not None:
                try:self.save()  # the preparation producer has now drained
                except BaseException as error:O.record_runtime_error(self.run,error,'FAILED_STATE_SAVE')
            if original_error is None and cleanup_error is not None:raise cleanup_error
        if preparation_count is not None:
            self.queued_preparation = preparation_count()
        with K.CLI.round_phase(self.run, 'WIDTH_CONFIRM_COMPLETED'):
            self.confirm_completed(context,executor,result)
        launched_now={m:executor.launched(m)for m in self.methods}
        self.state['q3_launched']=launched_now
        with(self.run/'INVOCATIONS.jsonl').open('a')as f:
            f.write(json.dumps(O.jsonable(dict(time=time.time(),requested_widths=self.widths,budget=budgets,
                launched=launched_now,
                stop={m:r['stop_reason']for m,r in result.items()},new_calls={m:r['new_calls']for m,r in result.items()})))+'\n')
        with K.CLI.round_phase(self.run, 'WIDTH_UPDATE_CANDIDATE_STATE'):
            self.update_evaluated_points(context, search, result)
            if getattr(self, 'scaled_shots', False):
                self.write_scope_map(result)
            self.shortlist_price_changed = False
            if exploratory:
                current = shortlist_price_ids(result)
                if live_price:
                    for method, ids in live_price.snapshot().items():
                        current[method] = sorted(set(current.get(method, ())) | set(ids))
                self.shortlist_price_changed = any(set(ids) - set(self.state.get('shortlist_price_candidates', {}).get(method, ()))
                                                   for method, ids in current.items())
                self.state['shortlist_price_candidates'] = current
            self.save()
        self.report(result)
        if str(self.run).startswith(str(ROOT/'docs/research')):
            K.CLI.write(self.run/'NEEDS.json',dict(candidates=K.CLI.candidate_rows(context)))
            from publish_h4_search_run import prepare as publish_search
            publish_search(ROOT,self.run,self.run.parent/'SEARCH_PUBLICATION.json',run_id=self.spec['round_id'])
        return result

    def finish_report(self, *, include_deferred=True):
        report = getattr(self, '_report', None)
        if report is not None:
            self._report = None
            process, snapshot = report
            with snapshot:
                if process.wait():
                    self.render_snapshot(snapshot)
        deferred = getattr(self, '_deferred_report', None)
        if include_deferred and deferred is not None:
            self._deferred_report = None
            with deferred:
                self.render_snapshot(deferred)

    def render_snapshot(self, snapshot):
        from trottertracks.rpe_plan_history_cache import read_local_exact_snapshot

        snapshot.seek(0)
        run, spec, result, source, owner = read_local_exact_snapshot(snapshot)
        write_width_report(Path(run), self.report_cli(), spec, result, source, owner=owner)

    def report_cli(self):
        trial = selected_worker_policy(self.spec)
        if 'width_report_sha256' not in trial or 'large_shot_adapter' in trial:
            return K.CLI
        import runpy
        from types import SimpleNamespace

        if not hasattr(self, '_report_cli'):
            path = Path(__file__).resolve().with_name('run_rpe_method_optimizer.py')
            imports = sys.path[:]
            try:
                self._report_cli = SimpleNamespace(**runpy.run_path(str(path)))
            finally:
                sys.path[:] = imports
        return self._report_cli

    def report(self, result):
        source = K.CLI.head()
        holds = {method: rows.get('execution_holds', []) for method, rows in result.items()}
        # Older executors reload these holds from ROUND_RESULT at batch start.
        # Publish changes synchronously before another executor can reserve work.
        same_holds = getattr(self, '_reported_holds', None) == holds
        self._reported_holds = deepcopy(holds)
        trial = selected_worker_policy(self.spec)
        if (not same_holds or not getattr(self, 'scaled_shots', False)
                or not trial.get('controller_queue_first')
                or str(self.run).startswith(str(ROOT/'docs/research'))
                or not (self.run/'NORMAL_DRIVER.json').is_file()):
            self.finish_report()
            write_width_report(self.run, self.report_cli(), self.spec, result, source)
            return

        import pickle
        import tempfile

        # A single child renders an immutable snapshot from a private file
        # descriptor while the controller advances its saved checkpoint.
        script = (
            'import importlib.util,runpy,sys\n'
            'from pathlib import Path\n'
            'root,driver,entry=sys.argv[1:]\n'
            'report=runpy.run_path(driver,init_globals={"FIXED_RUN_CODE_ROOT":root})\n'
            'from types import SimpleNamespace\n'
            'cli=SimpleNamespace(**runpy.run_path(entry))\n'
            'codec=Path(driver).parents[1]/"src/trottertracks/rpe_plan_history_cache.py"\n'
            'loader=importlib.util.spec_from_file_location("trottertracks._report_codec",codec)\n'
            'history=importlib.util.module_from_spec(loader)\n'
            'sys.modules[loader.name]=history\n'
            'loader.loader.exec_module(history)\n'
            'run,spec,result,source,owner=history.read_local_exact_snapshot(sys.stdin.buffer)\n'
            'report["write_width_report"](Path(run),cli,spec,result,source,owner=owner)\n')
        with K.CLI.round_phase(self.run, 'WIDTH_QUEUE_REPORT'):
            snapshot = tempfile.TemporaryFile()
            try:
                pickle.dump((str(self.run), self.spec, result, source, os.getpid()),
                            snapshot, protocol=pickle.HIGHEST_PROTOCOL)
                snapshot.seek(0)
                deferred = getattr(self, '_deferred_report', None)
                if deferred is not None:
                    deferred.close()
                    self._deferred_report = None
                previous = getattr(self, '_report', None)
                if previous is not None and previous[0].poll() is None:
                    self._deferred_report = snapshot
                    return
                self.finish_report(include_deferred=False)
                process = subprocess.Popen(
                    [sys.executable, '-B', '-c', script, str(ROOT), str(Path(__file__).resolve()),
                     self.report_cli().__file__],
                    stdin=snapshot, cwd=ROOT)
            except BaseException:
                snapshot.close()
                raise
            self._report = process, snapshot

    def update_evaluated_points(self, context, search, result):
        priced={c['configuration_id']:c['prices'] for c in context['candidates']
                if c.get('ready') and c.get('prices')}
        by_id = {}
        for method, method_result in result.items():
            indexed = by_id.setdefault(method, {})
            for row in method_result.get('by_configuration') or []:
                ids = (row['configuration_id'],)
                if getattr(self, 'scaled_shots', False) and row.get('source_configuration_id'):
                    ids = dict.fromkeys((*ids, row['source_configuration_id']))
                for cid in ids:
                    indexed.setdefault(cid, []).append(row)
        for p in self.state['points']:
            if p['status']not in('REGISTERED','EVALUATED'):continue
            prices=priced.get(p.get('evaluation_configuration_id',p.get('configuration_id')),[])
            if (search.get('review_search_order') is True and len(prices)==26
                    and all(type(v) in (int,Fraction) and v>=0 for v in prices)):
                p['own_price_floor']=sum(prices)
            cid = p.get('evaluation_configuration_id', p.get('configuration_id'))
            matches = by_id.get(p['method'], {}).get(cid, ())
            if getattr(self, 'scaled_shots', False):
                b = min(matches, key=lambda v: (v['best'] is None,
                    v['best']['cost'] if v['best'] else 0, v['M']), default=None)
                if b:
                    p['best_M'] = b['M'] if b['best'] else None
                    b = dict(b, promising=sum(v['promising'] for v in matches))
            else:
                b = next(iter(matches), None)
            if b:
                adopted=b.get('best');best=b.get('confirmed_all_best')or adopted
                p.update(status='EVALUATED',best=None if best is None else best['cost'],
                    rule_adopted_best=None if adopted is None else adopted['cost'],
                    confirmed_all_best=None if b.get('cost_status') or best is None else best['cost'],
                    best_N_X=None if best is None else O.split(best['vector'])[0],
                    best_N_Y=None if best is None else O.split(best['vector'])[1],
                    best_MSE_upper=None if best is None else best.get('MSE_upper'),
                    promising=b.get('promising'))

    def loop(self):
        previous=self.state.get('stop',{}).get('reason')
        transient={'USER_REQUESTED_STOP','REQUESTED_WIDTH_METADATA_ONLY','WAITING_FOR_INPUTS','SCOPE_HELD_INPUTS',
                   'SCOPE_HELD_Q3','MAX_CYCLES','BOUNDED_SCHEDULE_PLAN_ONLY','DISCRETE_SCHEDULE_PLAN_ONLY'}
        trial = selected_worker_policy(self.spec)
        recovery = trial.get('recovered_failure')
        if recovery is None and trial.get('resumed_ast_failure'):
            recovery = dict(kind='ast', invocation=trial['resumed_ast_failure'])
        if previous == 'RUNTIME_EXCEPTION' and recovery and not self.a.plan_only:
            kind, invocation = recovery['kind'], recovery['invocation']
            preserved = O.load_json(self.run/'stop-history'/(kind+'-'+invocation+'.json'))
            if preserved != self.state['stop']:
                raise ValueError('Reconciled controller failure differs from its saved stop')
            self.state.pop('stop')
            self.log(event='RESUMED_AFTER_'+kind.upper()+'_FAILURE', invocation_recovered=invocation)
            previous = None
        if previous and previous not in transient:
            self.log(event='PRESERVED_STOP',reason=previous);return
        if previous:self.state.pop('stop')
        bounded={method:W.kappa_bounds(self.wide_for(method)) for method in self.widths} if self.a.plan_only else {}
        grids={method:W.kappa_values(self.wide_for(method)) for method in self.widths} if self.a.plan_only else {}
        if self.a.plan_only and (any(bounded.values()) or any(grids.values())):
            # Native physical schedule selection only. Missing family/input
            # preparation remains explicit; no Completion, Executor or worker.
            self.state['cycle'] += 1
            proposals, report = self.plan()
            self.state['scope_plan'] = dict(rule=W.RULE,
                kappa_bounds=next(([str(v) for v in b] for b in bounded.values() if b),None),
                kappa_bounds_by_method={m:[str(v) for v in b] if b else None for m,b in bounded.items()},
                kappa_values_by_method={m:[str(v) for v in grid] if grid else None for m,grid in grids.items()},
                bounds_inclusive=[True, True],
                requested_L_D={m:self.problem['methods'][m].get('L_D_values') for m in self.methods},
                family_status={k:dict(L_D=v['L_D'],status=v['status'],reason=v.get('reason'))
                               for k,v in self.state['families'].items()},
                preparation_pending=[k for k,v in self.state['families'].items() if v['status']!='READY_METADATA'],
                scientific_calls=0, queue_operations=0, dry_check_calls=0)
            self.state['stop'] = dict(reason='DISCRETE_SCHEDULE_PLAN_ONLY' if any(grids.values())
                                     else 'BOUNDED_SCHEDULE_PLAN_ONLY',UTC=K.now())
            self.save()
            return
        end=None if self.a.max_cycles is None else self.state['cycle']+self.a.max_cycles
        auto_recheck = end is None and self.work_conserving_search and bool(
            self.spec.get('executor', {}).get('distributed'))
        recheck_delay = 15
        held_q3, unresolved = False, []
        while end is None or self.state['cycle']<end:
            if (self.run/'STOP_REQUESTED.json').exists():
                self.finish_waiting_executor()
            if (self.run/'STOP_REQUESTED.json').exists():
                self.state.pop('recovery_wait', None)
                self.state['stop']=dict(reason='USER_REQUESTED_STOP',UTC=K.now());self.save();return
            self.state['cycle']+=1;started=time.monotonic();self.log(event='PLAN_START')
            self.state.pop('recovery_wait', None)
            distributed_prep=bool((self.spec.get('executor',{}).get('distributed')or{}).get('preparation'))
            overlap=((self.spec.get('overlap_ready_batch')is True or distributed_prep) and not self.a.plan_only and
                     self.spec.get('science_approved')is True)
            if not (distributed_prep and overlap):self.bootstrap_families()
            ready_families=sum(f['status']=='READY_METADATA' for f in self.state['families'].values())
            proposals,report=self.plan();self.save()
            self.log(event='PLAN_END',wall_seconds=time.monotonic()-started,proposals=len(proposals))
            started=time.monotonic();self.log(event='INPUTS_START')
            if not overlap:self.complete_points()
            self.save();self.log(event='INPUTS_END',wall_seconds=time.monotonic()-started,deferred_to_batch=overlap)
            if self.a.plan_only or self.spec.get('science_approved')is not True:
                self.state['stop']=dict(reason='REQUESTED_WIDTH_METADATA_ONLY',UTC=K.now());self.save();return
            started=time.monotonic();self.log(event='V2_BATCH_START')
            result=self.batch(prepare_during=True)if overlap else self.batch()
            self.save();self.log(event='V2_BATCH_END',wall_seconds=time.monotonic()-started,
                methods={m:dict(stop=r['stop_reason'],new_calls=r['new_calls'])for m,r in result.items()})
            if O.shared_stop(result):
                failed={m:r['stop_reason']for m,r in result.items()if r['stop_reason']=='UNKNOWN_RESULT'}
                self.state['stop']=dict(reason='V2_RESULT_REQUIRES_REVIEW',methods=failed,UTC=K.now());self.save();return
            caps=self.spec.get('max_new_q3_per_method',{})
            launched=self.state.get('q3_launched',{})
            budget_stops = {m for m, r in result.items() if r['stop_reason'] == 'Q3_BUDGET_EXHAUSTED'}
            budget_remaining = {m for m in budget_stops
                                if type(caps.get(m)) is not int or launched.get(m, 0) < caps[m]}
            unresolved = next(iter(result.values()), {}).get('executor_metrics', {}).get('unresolved_q3_attempts', [])
            held_q3 = bool(unresolved) or any(r.get('prior_q3_holds') or r['stop_reason'] in O.TASK_HOLD_STOPS or
                any(h['status']=='UNKNOWN' for h in r.get('execution_holds', [])) for r in result.values())
            if (any(launched.values()) and
                    all(type(caps.get(m)) is int and launched.get(m, 0) >= caps[m] for m in result)):
                self.state['stop']=dict(reason='SCOPE_HELD_Q3' if held_q3 else 'Q3_BUDGET_EXHAUSTED',
                    UTC=K.now(),launched=launched,unresolved_q3_attempts=unresolved)
                self.save();return
            scaled = getattr(self, 'scaled_shots', False)
            # The next cycle raises the batch cap, bounded by the same round budget.
            if scaled:
                next_batch = [m for m in sorted(budget_remaining)
                              if type(batch_cap := result[m]['budget']['max_new_q3']) is int
                              and 0 < batch_cap <= launched.get(m, 0)]
                if next_batch:
                    self.log(event='Q3_BATCH_ADVANCE', methods=next_batch)
                    continue
            waiting_inputs=(any(p['status']=='WAITING_INPUTS'for p in self.state['points'])or
                     any(f['status']in('FAMILY_PREPARATION_PENDING','NEEDS_FAMILY_CONFIGURATION_SOURCE')for f in self.state['families'].values())or
                     any(r.get('input_requests') for r in result.values()))
            queued_preparation = getattr(self, 'queued_preparation', 0)
            waiting = waiting_inputs or queued_preparation or any(r.get('prior_q3_holds') for r in result.values())
            pending=any(p['status']in('COMPLETION','REGISTERED')or(p['status']=='EVALUATED'and p.get('promising'))for p in self.state['points'])
            if getattr(self,'rule_stop',False):
                if any(r['stop_reason']=='STEP_LIMIT' for r in result.values()):
                    self.state['stop']=dict(reason='SCOPE_HELD_Q3' if held_q3 else 'EXPLICIT_STEP_LIMIT',
                        UTC=K.now(),unresolved_q3_attempts=unresolved);self.save();return
                if getattr(self,'pf_more',False) or any(r['stop_reason']=='WORK_SLICE_YIELDED' for r in result.values()):
                    continue
                pending=any(p['status'] in ('COMPLETION','REGISTERED') for p in self.state['points'])
            if getattr(self, 'preparation_queue_advanced', False) or getattr(self, 'shortlist_price_changed', False):
                continue
            pf_status_path = self.run/'PF_PREPARATION_STATUS.json'
            pf_status = O.load_json(pf_status_path) if pf_status_path.is_file() else {}
            pf_records = list(pf_status.get('records', {}).values()) or [pf_status]
            pf_held = any(row.get('status') in ('PREPARATION_INCOMPLETE', 'PREPARATION_WAITING',
                                               'PRIOR_OWNER_HOLD', 'PRICE_EVIDENCE_REQUIRED',
                                               'PRICE_ACTION_UNAVAILABLE', 'PRICE_NOT_EVALUATED', 'PRICE_RECHECK_AVAILABLE',
                                               'PRICE_OWNER_WAITING')
                          for row in pf_records)
            if (not proposals and not pending and ready_families ==
                    sum(f['status']=='READY_METADATA' for f in self.state['families'].values())):
                blocked=waiting or any(f['status']!='READY_METADATA' for f in self.state['families'].values()) or any(
                    p['status'] in ('HELD','PRICE_ACTION_UNAVAILABLE','PRICE_NOT_EVALUATED')
                    for p in self.state['points']) or pf_held or any(
                    r.get('not_ready_candidates') or r.get('scaled_shots', {}).get('scope_coverage', {}).get('missing_count')
                    or r.get('scaled_shots', {}).get('unpriced_feasible')
                    for r in result.values())
                reason=('SCOPE_HELD_INPUTS' if blocked else 'REQUESTED_WIDTH_RULE_EXHAUSTED') if getattr(self,'rule_stop',False) else (
                    'WAITING_FOR_INPUTS'if waiting else'REQUESTED_WIDTH_RULE_EXHAUSTED')
                if held_q3 or (scaled and not blocked and (budget_remaining or
                        any(r['stop_reason'] == 'UNRESOLVED_SCALED_SCOPE' for r in result.values()))):
                    reason = 'SCOPE_HELD_Q3'
                elif scaled and not blocked and any(r['stop_reason'] == 'SCALED_SCOPE_INCLUDES_UNSUPPORTED_HORIZON'
                                                    for r in result.values()):
                    reason = 'SCOPE_HELD_Q3'
                elif scaled and not blocked and budget_stops:
                    reason = 'Q3_BUDGET_EXHAUSTED'
                waits = []
                if waiting_inputs or queued_preparation or any(row.get('status') == 'PREPARATION_WAITING' for row in pf_records):
                    waits.append('PREPARATION_OR_INPUT')
                if any(row.get('status') in ('PRICE_RECHECK_AVAILABLE', 'PRICE_OWNER_WAITING')
                       for row in pf_records):
                    waits.append('PF_PRICE_RECHECK')
                if (auto_recheck and 'PR_DF' in result and
                        getattr(self.completion_api(), 'df_price_waiting_for_owner', lambda: False)()):
                    waits.append('DF_PRICE_ORIGINAL_ATTEMPT')
                if unresolved or any(h.get('status') == 'IN_PROGRESS' for r in result.values()
                                     for h in r.get('held', ())):
                    waits.append('OWNED_ATTEMPT')
                if any(r['stop_reason'] == 'HOST_ELIGIBILITY_HOLD' or any(
                       h.get('reason') == 'NO_MATCHING_ELIGIBLE_HOST' or h.get('status') == 'HOST_ELIGIBILITY_HOLD'
                       for h in r.get('held', ())) for r in result.values()):
                    waits.append('ELIGIBLE_HOST')
                if auto_recheck and waits and reason in ('WAITING_FOR_INPUTS', 'SCOPE_HELD_INPUTS', 'SCOPE_HELD_Q3'):
                    self.wait_for_recovery(waits, recheck_delay)
                    recheck_delay = min(recheck_delay * 2, 60)
                    continue
                self.state['stop']=dict(reason=reason,
                    UTC=K.now(),methods={m:r['stop_reason']for m,r in result.items()},
                    unresolved_q3_attempts=unresolved,
                    scope_summary=getattr(self,'scope_summary',None));self.save();return
            recheck_delay = 15
        self.state['stop']=dict(reason='SCOPE_HELD_Q3' if held_q3 else 'MAX_CYCLES',
            UTC=K.now(),unresolved_q3_attempts=unresolved);self.save()

    def finish_waiting_executor(self):
        executor = getattr(self, 'waiting_executor', None)
        if executor is not None:
            self.waiting_executor = None
            executor.finish()

    def recovery_revision(self, executor):
        paths = (
            self.run/'distributed'/'pbs'/'offline-history'/'coverage',
            self.run/'HOST_ADMISSION.json',
            self.run/'completion'/'STEP_PRICE_RULES.json',
            self.run/'completion'/'PRICE_RULES.json',
            self.run/'pf_auto'/'WAVES.json',
        )
        result = []
        for path in paths:
            try:
                stat = path.stat()
                result.append((stat.st_ino, stat.st_size, stat.st_mtime_ns))
            except FileNotFoundError:
                result.append(None)
        if executor is not None:
            with executor.store._connection() as db:
                result.append(db.execute("SELECT max(sequence) FROM events").fetchone()[0])
                unresolved = [tuple(row) for row in db.execute("SELECT method,job_id FROM jobs WHERE state='UNKNOWN'")]
            # An imported owner receipt can arrive outside the queue RPC.  It
            # must wake its current-run owner even before a ledger event.
            for method, job in sorted(unresolved):
                for name in ('OUTCOME.json', 'PROCESS_RETURN.json'):
                    path = self.run/'jobs'/method/job/name
                    try:
                        stat = path.stat()
                        result.append((method, job, name, stat.st_size, stat.st_mtime_ns))
                    except FileNotFoundError:
                        result.append((method, job, name, None))
            # Previous-run ownership is also addressable by its exact job.
            # Observing only these receipts/ledger rows avoids waking the full
            # planner because an unrelated job in the older run changed.
            import sqlite3
            from contextlib import closing
            prior = {}
            for hold in getattr(executor, 'prior_q3_holds', ()):
                if all(hold.get(k) for k in ('parent_directory', 'method', 'job_id')):
                    prior.setdefault(Path(hold['parent_directory']), set()).add((hold['method'], hold['job_id']))
            for owner, jobs in sorted(prior.items()):
                database = owner/'distributed/QUEUE.sqlite3'
                try:
                    with closing(sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True, timeout=.1)) as db:
                        for method, job in sorted(jobs):
                            row = db.execute('SELECT state, completion_sha256, adopted_epoch, updated_epoch FROM jobs WHERE job_id=?',
                                             (job,)).fetchone()
                            result.append((str(owner), method, job, 'ledger', row))
                except (OSError, sqlite3.Error):
                    result.append((str(owner), 'ledger', None))
                for method, job in sorted(jobs):
                    for name in ('OUTCOME.json', 'PROCESS_RETURN.json', 'LAUNCH.json'):
                        path = owner/'jobs'/method/job/name
                        try:
                            stat = path.stat()
                            result.append((str(owner), method, job, name, stat.st_size, stat.st_mtime_ns))
                        except FileNotFoundError:
                            result.append((str(owner), method, job, name, None))
            # A returning worker can make an unqueued candidate eligible.
            # Poll timestamps and fluctuating memory readings are not changes
            # to its identity or admission.
            result.append(tuple(sorted(
                (host, offer.get('capacity'), bool(offer.get('slots')),
                 offer.get('accepting_new_jobs'), json.dumps(offer.get('host_fingerprints'), sort_keys=True),
                 json.dumps(offer.get('host_admission'), sort_keys=True),
                 ((offer.get('resource_report') or {}).get('storage') or {}).get('accepting_new_jobs'))
                for host, offer in dict(executor.hosts).items())))
        return result

    def wait_for_recovery(self, reasons, seconds):
        executor = getattr(self, 'waiting_executor', None)
        baseline = self.waiting_revision if executor is not None else self.recovery_revision(None)
        self.state['recovery_wait'] = dict(reasons=reasons, recheck_epoch=time.time()+seconds)
        self.save()
        self.log(event='WAITING_FOR_RECOVERY', **self.state['recovery_wait'])
        deadline = time.monotonic() + seconds
        # Old run evidence edited outside this controller has no queue event.
        # Do not rebuild every saved candidate periodically while exact
        # current/prior ledger owners and their receipts are unchanged. For
        # a hold without a readable owner location, keep the old watchdog.
        known_owners_only = (set(reasons) == {'OWNED_ATTEMPT'} and executor is not None
                             and all(all(h.get(k) for k in ('parent_directory','method','job_id'))
                                     and Path(h['parent_directory']).is_dir()
                                     for h in getattr(executor, 'prior_q3_holds', ())))
        external_recheck = float('inf') if known_owners_only else time.monotonic() + 900
        stop = self.run/'STOP_REQUESTED.json'
        while not stop.exists():
            if self.recovery_revision(executor) != baseline or time.monotonic() >= external_recheck:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if executor is None or executor.state.failure:
                    return
                # With the relay and readiness services still alive, polling
                # an unchanged queue need not restart the scientific planner.
                deadline = time.monotonic() + seconds
                self.log(event='WAITING_FOR_RECOVERY_HEARTBEAT', reasons=reasons,
                         recheck_epoch=time.time()+seconds)
                remaining = seconds
            time.sleep(min(1, remaining))


@serialize_legacy_ast()
def main(argv=None, *, driver_factory=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True);ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max-cycles', type=int);ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--import-check', action='store_true', help='Load the selected controller and fixed helpers without dispatching or writing run state')
    ap.add_argument('--prepare-readiness', action='store_true', help='plan-only: delta saved history, chosen requests, server2 dry-check and immutable remote delivery')
    ap.add_argument('--preparation-cache', type=Path, help='local disposable cache for --prepare-readiness')
    ap.add_argument('--host-readiness', type=Path, help='saved host READINESS descriptors to join during --prepare-readiness')
    a = ap.parse_args(argv);a.problem = a.problem.resolve()
    if a.prepare_readiness:
        from trottertracks.rpe_plan_prepare import prepare
        return prepare(a, code_root=ROOT, cli=K.CLI)
    problem=O.load_json(a.problem);spec=O.load_json(a.round)
    if a.import_check:
        if trial := selected_worker_policy(spec):
            apply_worker_trial(spec, trial, run_dir=a.run_dir)
        from trottertracks import rpe_scope_map, rpe_scaled_shots, rpe_optimizer_concurrent, rpe_pbs
        print(json.dumps(dict(status='IMPORT_CHECK_PASS', controller=str(Path(__file__).resolve()),
                              run_directory=str(a.run_dir.resolve()), dispatch=False)))
        return
    if a.max_cycles is None and not spec.get('search_to_rule_stop'):a.max_cycles=200
    declared=W.requested_widths(problem,spec)
    if driver_factory is None and declared is None and not spec.get('search_to_rule_stop') and any(m in('FULL_PF','TRUNCATED_PF')for m in spec.get('methods',[])):
        # PF selection and shot/Q3 remain owned by the existing method CLI.
        # This is the same public entry, not another PF driver/optimizer.
        from types import SimpleNamespace
        a.run_dir.mkdir(parents=True,exist_ok=True)
        K.CLI.round_mode(a,problem,[])
        result=O.load_json(a.run_dir/'ROUND_RESULT.json')
        return SimpleNamespace(run=a.run_dir,state=dict(stop=dict(reason='METHOD_ROUND_COMPLETE',
            methods={m:r['stop_reason']for m,r in result['methods'].items()})))
    factory=driver_factory or (WidthDriver if declared is not None or spec.get('search_to_rule_stop') else WideDriver)
    d = factory(a);d.log(event='DRIVER_START', plan_only=a.plan_only)
    try:d.loop()
    except KeyboardInterrupt:
        # Executor cleanup drains owned work. Scientific UNKNOWN/STOP evidence
        # remains in job artifacts; only this operational stop is resumable.
        d.state['stop']=dict(reason='USER_REQUESTED_STOP',UTC=K.now())
        d.save();d.log(event='USER_REQUESTED_STOP')
    except BaseException as error:
        O.record_runtime_error(a.run_dir,error,'DRIVER')
        d.state['stop']=dict(reason='RUNTIME_EXCEPTION',error_type=type(error).__name__,message=str(error),
            error_log='RUNTIME_ERRORS.jsonl',UTC=K.now())
        try:d.save();d.log(event='DRIVER_ERROR',stop=d.state['stop'])
        except BaseException as save_error:O.record_runtime_error(a.run_dir,save_error,'FAILED_DRIVER_STATE_SAVE')
        raise
    finally:
        try:
            if finish_waiting := getattr(d, 'finish_waiting_executor', None):
                finish_waiting()
        finally:
            try:
                finish_report = getattr(d, 'finish_report', None)
                if finish_report is not None:
                    finish_report()
            finally:
                manager = getattr(d, 'run_pbs_manager', None)
                if manager is not None and manager.detached:
                    manager.close()
    if (a.run_dir/'STOP_REQUESTED.json').exists() and d.state.get('stop',{}).get('reason') not in (
            'V2_RESULT_REQUIRES_REVIEW','RUNTIME_EXCEPTION'):
        d.state['stop']=dict(reason='USER_REQUESTED_STOP',UTC=K.now());d.save()
    d.log(event='DRIVER_END', stop=d.state.get('stop'))
    termination_reasons = ('REQUESTED_WIDTH_RULE_EXHAUSTED', 'EXPLICIT_STEP_LIMIT', 'Q3_BUDGET_EXHAUSTED')
    if spec.get('search_to_rule_stop') and d.state.get('stop', {}).get('reason') in termination_reasons:
        K.CLI.write(a.run_dir/'SEARCH_TERMINATION.json', d.state['stop'])
    print(json.dumps(O.jsonable(d.state.get('stop')), default=str)[:2000])
    return d


if __name__ == '__main__':
    main()
