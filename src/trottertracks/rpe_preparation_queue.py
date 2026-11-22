"""Canonical preparation identities, shared by planning, dry-check and dispatch."""
from copy import deepcopy
from pathlib import Path
from .rpe_q3_coordinator import digest
from .rpe_preparation_artifacts import input_manifest,artifact_path


def normalize_preparation_request(request, *, source_manifest=None, dependencies=()):
    request=deepcopy(request)
    if request.get('route',{}).get('kind')!='preparation':raise ValueError('Preparation route required')
    preparation=request.get('preparation')
    if not isinstance(preparation,dict)or preparation.get('stage')not in ('signal','compile','audit'):
        raise ValueError('Preparation stage must be signal, compile, or audit')
    optional = preparation.get('optional_price_validation', False)
    if type(optional) is not bool or (optional and preparation['stage'] != 'compile'):
        raise ValueError('Optional price validation must be a compile preparation')
    if dependencies:preparation['dependencies']=list(dict.fromkeys(dependencies))
    preparation.setdefault('dependencies',[])
    preparation.setdefault('inputs',{})
    if not preparation.get('source'):
        if preparation.get('artificial')is not True or not source_manifest:
            raise ValueError('Fixed preparation SOURCE manifest required')
        preparation['source']=deepcopy(source_manifest)
    if not isinstance(preparation['source'].get('commit'),str)or not isinstance(preparation['source'].get('files'),dict):
        raise ValueError('Preparation SOURCE must pin commit and files')
    preparation['inputs']=input_manifest(request)
    for name in preparation.get('outputs',[]):artifact_path(name)
    # The derived ID is excluded from its own seed; all execution/scientific
    # fields, including source, inputs, dependencies and command, remain bound.
    provided=request.pop('request_id',None)
    identity='prep-'+digest(request)[:24]
    if provided is not None and provided!=identity:raise ValueError('Preparation request ID differs from its full definition')
    request['request_id']=identity
    return request


def preparation_job_id(request, *, source_manifest=None, dependencies=()):
    return normalize_preparation_request(request,source_manifest=source_manifest,dependencies=dependencies)['request_id']


def submit_registered_horizons(runtime, candidates, policy, source_M):
    """Start fixed, scoped extension work while the shorter horizon is running."""
    from .rpe_scaled_shots import projection_id

    bindings = (runtime.settings.get('horizon_extensions') or {}).get('bindings', {})
    requests = []
    for candidate in candidates:
        cid = candidate.get('configuration_id')
        binding = bindings.get(cid)
        if not binding or candidate.get('M', source_M) != binding['source_M']:
            continue
        for horizon in policy['M']:
            if str(horizon) not in binding.get('rounds', {}):
                continue
            rounds = list(range(binding['source_M'] + 1, horizon + 1))
            if rounds:
                requests.append(dict(source_configuration_id=cid, source_M=binding['source_M'],
                    configuration_id=projection_id(cid, binding['source_M'], horizon), M=horizon,
                    extension_rounds=rounds))
    return submit_horizon_preparation(runtime, requests)


def submit_horizon_preparation(runtime, requests):
    """Resolve only the requested extension rounds against a fixed provider catalog.

    The catalog belongs to the release's preparation settings. Candidate aliases
    can point to the same physical round and task name. The normal ledger derives
    the final job identity from the same fixed request with its dependencies.
    """
    catalog = runtime.settings.get('horizon_extensions') or {}
    bindings, tasks = catalog.get('bindings', {}), catalog.get('tasks', {})
    pf_terminal_prices = runtime.settings.get('pf_terminal_price_leaves', {})
    submitted = {}
    budgets = runtime.settings.get('budgets', {})

    def validate(name, order, visiting=()):
        if name in order:
            return
        if name in visiting:
            raise ValueError('Cyclic horizon preparation dependencies')
        task = tasks[name]
        request = task['request']
        stage = request['preparation']['stage']
        if stage not in ('signal', 'compile', 'audit') or request['route']['kind'] != 'preparation':
            raise ValueError('Horizon task requires a fixed preparation provider')
        if not isinstance(request['original_contract'], dict):
            raise ValueError('Horizon task requires a fixed preparation contract')
        parents = task.get('dependencies', ())
        if set(request['preparation'].get('dependency_inputs', {})) - set(parents):
            raise ValueError('Horizon dependency input has no declared producer')
        for dep in parents:
            validate(dep, order, (*visiting, name))
        order[name] = task

    def submit(name, task):
        if name in submitted:
            return
        request = deepcopy(task['request'])
        stage = request['preparation']['stage']
        dependencies = {dep: submitted[dep] for dep in task.get('dependencies', ())}
        if request['preparation'].get('dependency_inputs'):
            request['preparation']['dependency_inputs'] = {
                dependencies[dep]: paths for dep, paths in request['preparation']['dependency_inputs'].items()}
        job = runtime.executor.submit_preparation(request, budget=budgets.get(stage, 0),
            approved=runtime.approved, contract=request['original_contract'], dependencies=dependencies.values())
        submitted[name] = job

    reports = []
    for request in requests:
        rounds = request.get('extension_rounds', ())
        if not rounds:
            continue
        configuration = request.get('source_configuration_id')
        binding = bindings.get(configuration)
        report = dict(configuration_id=request['configuration_id'], source_configuration_id=configuration,
                      M=request['M'], status='HORIZON_PROVIDER_NOT_CONFIGURED', jobs={})
        reports.append(report)
        if binding is None or binding.get('source_M') != request.get('source_M'):
            continue
        incomplete = {}
        for m in rounds:
            entry = binding.get('rounds', {}).get(str(m))
            if not entry:
                incomplete[str(m)] = {'round': 'HORIZON_ROUND_PROVIDER_NOT_CONFIGURED'}
                continue
            if m == request['M']:
                report['terminal_physical_id'] = entry['physical_id']
            successes, failures = {}, {}
            for role, allowed in (('signal', ('signal',)), ('X', ('compile', 'audit')), ('Y', ('compile', 'audit'))):
                try:
                    if (m == request['M'] and role != 'signal' and configuration in pf_terminal_prices
                            and pf_terminal_prices[configuration].get('M', 13) == m):
                        leaf = pf_terminal_prices[configuration]
                        if leaf['physical_id'] != entry['physical_id']:
                            raise ValueError('PF model price and horizon provider bind different physical rounds')
                        failures[role] = 'HORIZON_PRICE_FROM_ADOPTED_PARENT_MODEL'
                        continue
                    name = entry.get(role)
                    if name is None:
                        failures[role] = 'HORIZON_ROLE_PROVIDER_NOT_CONFIGURED'
                        continue
                    preparation = tasks[name]['request']['preparation']
                    if (preparation['stage'] not in allowed or
                            preparation['horizon_round'] != {'m': m, 'physical_id': entry['physical_id']} or
                            (role != 'signal' and role not in preparation.get('price_axes', ()))):
                        raise ValueError('Horizon task does not bind the selected physical round and axis')
                    ordered = {}
                    validate(name, ordered)
                    for dependency, task in ordered.items():
                        submit(dependency, task)
                    successes[role] = submitted[name]
                except (KeyError, TypeError, ValueError) as error:
                    failures[role] = str(error)
            if successes:
                report['jobs'][str(m)] = successes
            if failures:
                incomplete[str(m)] = failures
        if not incomplete:
            report['status'] = 'HORIZON_PREPARATION_REQUESTED'
        else:
            report.update(status='HORIZON_PREPARATION_PARTIAL' if report['jobs'] else
                          'HORIZON_PROVIDER_CONTRACT_HOLD', incomplete=incomplete)
    return reports


def save_horizon_preparation(path, reports, store):
    """Keep discoveries from both the early and final optimizer passes."""
    import json
    from .rpe_release_registry import write

    previous = json.loads(path.read_bytes()) if path.is_file() else []
    terminal_path = path.parent/'pf_auto'/'TERMINAL_PRICES.json'
    terminal_prices = json.loads(terminal_path.read_bytes()) if terminal_path.is_file() else {}
    rows = {(row['source_configuration_id'], row['M']): row for row in previous}
    rows.update(((row['source_configuration_id'], row['M']), row) for row in reports)
    for row in rows.values():
        parent = row['source_configuration_id']
        horizon = row['M']
        terminal = terminal_prices.get(parent)
        # Run-owned dynamic producers use the same stage marker and adopted
        # outcome as the rest of preparation. Do not relabel an owned signal
        # as an absent provider when a later optimizer pass refreshes this row.
        if isinstance(parent, str) and parent not in ('', '.', '..') and Path(parent).name == parent:
            folder = path.parent/'pf_auto'/'horizon'
            marker = folder/'signal-provider'/'horizon-inputs'/parent/'PF_SIGNALS.DISTRIBUTED.json'
            if marker.is_file():
                owned = json.loads(marker.read_bytes())
                physical = owned['request']['preparation']['horizon_round']
                if physical['m'] == horizon:
                    row['terminal_physical_id'] = physical['physical_id']
                    roles = row['jobs'].setdefault(str(horizon), {})
                    if roles.get('signal', owned['job_id']) != owned['job_id']:
                        raise ValueError('Conflicting horizon preparation owners')
                    roles['signal'] = owned['job_id']
                    row['status'] = 'HORIZON_PREPARATION_PARTIAL'
                    row.setdefault('incomplete', {}).setdefault(str(horizon), {}).update(
                        {axis: 'AWAITING_ADOPTED_PRICE_EVIDENCE' for axis in ('X','Y')})
            if horizon == 13:
                pf_detail = folder/parent/'PREPARATION.json'
                df_detail = path.parent/'completion'/'horizons'/parent/'STATUS.json'
                if df_detail.is_file():
                    row['preparation_detail'] = json.loads(df_detail.read_bytes())
                    row['status'] = row['preparation_detail']['status']
                elif pf_detail.is_file():
                    row['preparation_detail'] = json.loads(pf_detail.read_bytes())
                leaf = folder/parent/'LEAF.json'
                if not row.get('terminal_physical_id') and leaf.is_file():
                    row['terminal_physical_id'] = json.loads(leaf.read_bytes())['physical_id']
        if terminal:
            terminal_axes = terminal.get('verified_terminal_axes') or {}
            expected = row.get('terminal_physical_id')
            if (not expected and terminal.get('entry', {}).get('horizon_source_configuration_id') == parent
                    and terminal.get('entry', {}).get('horizon_parent')):
                expected = terminal_axes.get('physical_id')
                row['terminal_physical_id'] = expected
            if (terminal['status'] == 'READY' and terminal_axes.get('m') == horizon
                    and expected and terminal_axes.get('physical_id') == expected
                    and set(terminal_axes.get('axes', ())) == {'X', 'Y'}):
                row['terminal_prices'] = dict(status='READY', source=terminal['entry']['prices'])
                for axis in ('X', 'Y'):
                    row.get('incomplete', {}).get(str(horizon), {}).pop(axis, None)
                if not row.get('incomplete', {}).get(str(horizon)):
                    row.get('incomplete', {}).pop(str(horizon), None)
                if not row.get('incomplete'):
                    row['status'] = 'HORIZON_TERMINAL_PRICES_READY'
                    row.pop('preparation_detail', None)
            elif terminal['status'] == 'WAITING':
                row['terminal_prices'] = dict(status='WAITING', reason=terminal.get('reason'))
        progress = {}
        for round_number, roles in row['jobs'].items():
            progress[round_number] = {}
            for role, job_id in roles.items():
                job = store.row(job_id)
                if job is None:
                    progress[round_number][role] = {'state': 'NOT_RESERVED'}
                    continue
                status = job['detail'].get('outcome_status')
                if status is None and job['state'] == 'ACCEPTED':
                    outcome_path = path.parent/'jobs'/job['method']/job_id/'OUTCOME.json'
                    if outcome_path.is_file():
                        outcome = json.loads(outcome_path.read_bytes())
                        if outcome.get('request_sha256') == job['request_sha256']:
                            status = outcome.get('status')
                progress[round_number][role] = dict(
                    state=job['state'], host_id=job['host_id'], attempt_id=job['attempt_id'],
                    outcome_status=status, adopted=job['adopted_epoch'] is not None)
        row['job_progress'] = progress
    write(path, list(rows.values()))
