"""Canonical preparation identities, shared by planning, dry-check and dispatch."""
from copy import deepcopy
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


def submit_horizon_preparation(runtime, requests):
    """Resolve only the requested extension rounds against a fixed provider catalog.

    The catalog belongs to the release's preparation settings. Candidate aliases
    can point to the same physical round and task name. The normal ledger derives
    the final job identity from the same fixed request with its dependencies.
    """
    catalog = runtime.settings.get('horizon_extensions') or {}
    bindings, tasks = catalog.get('bindings', {}), catalog.get('tasks', {})
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
        for m in rounds:
            entry = binding.get('rounds', {}).get(str(m))
            if not entry:
                continue
            selected, ordered = {}, {}
            try:
                for role, allowed in (('signal', ('signal',)), ('X', ('compile', 'audit')), ('Y', ('compile', 'audit'))):
                    name = entry[role]
                    preparation = tasks[name]['request']['preparation']
                    if (preparation['stage'] not in allowed or
                            preparation['horizon_round'] != {'m': m, 'physical_id': entry['physical_id']} or
                            (role != 'signal' and role not in preparation.get('price_axes', ()))):
                        raise ValueError('Horizon task does not bind the selected physical round and axis')
                    selected[role] = name
                for name in selected.values():
                    validate(name, ordered)
            except (KeyError, TypeError, ValueError) as error:
                report.update(status='HORIZON_PROVIDER_CONTRACT_HOLD', reason=str(error))
                continue
            try:
                for name, task in ordered.items():
                    submit(name, task)
                report['jobs'][str(m)] = {role: submitted[name] for role, name in selected.items()}
            except (KeyError, TypeError, ValueError) as error:
                report.update(status='HORIZON_PROVIDER_CONTRACT_HOLD', reason=str(error))
        if len(report['jobs']) == len(rounds):
            report['status'] = 'HORIZON_PREPARATION_REQUESTED'
    return reports
