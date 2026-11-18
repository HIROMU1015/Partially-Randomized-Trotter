"""Local0 request -> immutable server2 scope -> existing optimizer entry.

No optimizer or scientific adapter is implemented here. Saved Q3 acceptance is
delegated to Executor.prior_records/ingest, including route and identity checks.
"""
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid

METHODS = ('FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF')
ENTRIES = {'method': 'run_rpe_method_optimizer.py', 'schedule': 'run_rpe_schedule_search.py',
           'neighbourhood': 'run_rpe_kappa_neighbourhood.py'}

class RequestError(ValueError):
    def __init__(self, field, reason):
        self.field, self.reason = field, reason
        super().__init__(field + ': ' + reason)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def load(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    temp = path.with_name(path.name + '.tmp'); temp.write_text(raw); os.replace(temp, path)

def immutable(path, value):
    path = Path(path)
    if path.exists():
        if load(path) != value:
            raise RequestError(str(path), 'existing immutable settings differ; preserved')
    else:
        write(path, value)

def keys(value, allowed, field):
    if not isinstance(value,dict):raise RequestError(field,'an object is required')
    for key in value:
        if key not in allowed:
            raise RequestError(field + '.' + key, 'unsupported field; no subset was applied')

def integers(value, field, lo, hi):
    if isinstance(value, dict):
        keys(value, {'min', 'max'}, field)
        a, b = value.get('min'), value.get('max')
        if type(a) is not int or type(b) is not int or a > b:
            raise RequestError(field, 'an inclusive integer min/max is required')
        value = list(range(a, b + 1)) if b - a <= 100000 else []
    if not isinstance(value, list) or not value or any(type(v) is not int or not lo <= v <= hi for v in value):
        raise RequestError(field, 'all requested values must be within %d..%d' % (lo, hi))
    return sorted(set(value))

def number(value, field):
    if not isinstance(value, dict):
        raise RequestError(field, 'declare kind=exact or binary64 explicitly')
    keys(value, {'kind', 'value'}, field)
    raw = value.get('value'); kind = value.get('kind')
    if kind == 'exact':
        if not isinstance(raw, (str, int)) or isinstance(raw, bool):
            raise RequestError(field, 'exact values require a fraction/decimal string or integer')
        try: exact = Fraction(raw)
        except (ValueError, ZeroDivisionError): raise RequestError(field, 'invalid exact fraction')
        native = dict(numerator=exact.numerator, denominator=exact.denominator)
    elif kind == 'binary64':
        try: native = float(raw); exact = Fraction(native)
        except (ValueError, TypeError, OverflowError): raise RequestError(field, 'invalid finite binary64')
    else:
        raise RequestError(field, 'unknown numeric representation')
    if exact <= 0: raise RequestError(field, 'value must be positive')
    return native, dict(kind=kind, exact=str(exact))


def request_service(config, request):
    """Select the adopted physical schedule entry for a PR-Pauli absolute interval."""
    scope = request.get('scope') or {}
    if not isinstance(scope, dict):
        return config
    per_method = scope.get('per_method') or {}
    if not isinstance(per_method, dict) or not isinstance(per_method.get('PR_PAULI', {}), dict):
        return config
    kappa = (per_method.get('PR_PAULI') or {}).get('kappa') or {}
    if not isinstance(kappa, dict):
        return config
    route = config.get('absolute_kappa_route')
    if route and kappa.get('mode') == 'absolute':
        if scope.get('methods', ['PR_PAULI']) != ['PR_PAULI'] and not config.get('combined_width_search'):
            raise RequestError('scope.methods', 'absolute kappa schedule entry currently accepts PR_PAULI alone')
        selected = deepcopy(config)
        selected.pop('absolute_kappa_route')
        selected.update(route)
        selected['absolute_kappa_schedule'] = True
        return selected
    return config

def resolve(config, request):
    """Translate only settings that the selected normal entry actually consumes."""
    keys(request, {'scope', 'instruction', 'action', 'release', 'run_id'}, 'request')
    scope = deepcopy(request.get('scope') or {})
    keys(scope, {'methods', 'per_method', 'shots', 'stop', 'preparation_budgets'}, 'scope')
    problem = load(config['problem']); spec = load(config['round'])
    if config.get('release_manifest'):
        from .rpe_release_deployment import manifest_reference
        manifest=load(config['release_manifest']);reference=manifest_reference(manifest)
        selected_id=config.get('release',{}).get('id')
        if selected_id and reference['id']!=selected_id:
            raise RequestError('release_manifest','component versions belong to another release')
        distributed=spec.get('executor',{}).get('distributed')
        if distributed is not None:
            distributed.update(required_release=reference,release_manifest=config['release_manifest'])
            layouts=config.get('host_layouts',{})
            if layouts.get('local'):distributed['local_deployment_layout']=layouts['local']
            for worker in distributed.get('relay',{}).get('workers',[]):
                if worker.get('enabled',True)and layouts.get(worker['host_id']):
                    layout=load(layouts[worker['host_id']])
                    worker['deployment']=dict(entry=str(Path(layout['component_roots']['optimizer'])/'scripts/manage_rpe_release_deployment.py'),
                        manifest=layout['manifest_path'],layout=layout['layout_path'])
    if config.get('plan_template'):spec.pop('plan_output',None)
    methods = scope.get('methods', spec['methods'])
    if not isinstance(methods, list) or not methods or any(m not in METHODS for m in methods):
        raise RequestError('scope.methods', 'use known method IDs')
    methods = sorted(set(methods), key=METHODS.index)
    if any(m not in problem['methods'] for m in methods):raise RequestError('scope.methods','method is not enabled in the adopted problem')
    spec['methods'] = methods
    spec.setdefault('max_steps_per_method',spec.get('search_overrides',{}).get('max_steps_per_method',problem['search']['max_steps_per_method']))
    spec['method_overrides'] = {m: v for m, v in spec.get('method_overrides', {}).items() if m in methods}
    if 'max_new_q3_per_method' in spec:
        spec['max_new_q3_per_method'] = {m: spec['max_new_q3_per_method'].get(m, 0) for m in methods}
    canonical = dict(methods=methods, per_method={}, shots=scope.get('shots', 'auto'), stop=scope.get('stop', {}))
    fragment_count = problem['problem']['hamiltonian_data']['expected']['fragment_count']
    for method, values in scope.get('per_method', {}).items():
        if method not in methods: raise RequestError('scope.per_method.' + method, 'method is not selected')
        keys(values, {'L_D', 'p', 'formulas', 'delta_pf', 'delta_over_h', 'kappa'}, method)
        current = deepcopy(problem['methods'][method]); override = spec['method_overrides'].setdefault(method, {})
        observed = {}
        for key in ('L_D', 'p'):
            if key not in values: continue
            if (key == 'L_D') != (current['family'] == 'PR'):
                raise RequestError(method + '.' + key, 'parameter does not belong to this method')
            allowed = integers(values[key], method + '.' + key, 1 if key == 'L_D' else 0, fragment_count if method=='FULL_PF' else fragment_count-1)
            override['L_D_values' if key == 'L_D' else 'p_values'] = allowed; observed[key] = allowed
            if key == 'L_D':
                override['L_D_range'] = dict(min=min(allowed), max_offset_from_fragment_count=max(allowed)-fragment_count)
            elif method == 'FULL_PF' and allowed != [fragment_count]:
                raise RequestError(method + '.p', 'Full PF retains all fragments')
        if 'formulas' in values:
            forms = values['formulas']
            if current['family'] != 'PF':
                raise RequestError(method + '.formulas', 'this normal entry cannot restrict saved PR formulas; request is preserved, not narrowed')
            if not isinstance(forms, list) or not forms or any(f not in current['formulas'] for f in forms):
                raise RequestError(method + '.formulas', 'unsupported PF formula')
            override['formulas'] = sorted(set(forms)); observed['formulas'] = override['formulas']
        if 'delta_pf' in values:
            if not isinstance(values['delta_pf'],list) or not values['delta_pf']:raise RequestError(method+'.delta_pf','a nonempty typed width list is required')
            if current['family'] != 'PR' or config.get('entry', 'method') != 'schedule':
                raise RequestError(method + '.delta_pf', 'absolute requested widths require an adopted requested-width schedule entry')
            nums = [number(x, method + '.delta_pf') for x in values['delta_pf']]
            spec.setdefault('search_overrides', {}).setdefault('requested_widths', {})[method] = [x[0] for x in nums]
            observed['delta_pf'] = [x[1] for x in nums]
        if 'delta_over_h' in values:
            if current['family'] != 'PF':
                raise RequestError(method + '.delta_over_h', 'PR widths must use an adopted requested-width entry')
            v = values['delta_over_h']
            if not isinstance(v, list) or len(v) != 2:
                raise RequestError(method + '.delta_over_h', 'exact lower/upper strings are required')
            a, b = Fraction(v[0]), Fraction(v[1])
            if a <= 0 or a > b: raise RequestError(method + '.delta_over_h', 'invalid inclusive width interval')
            override['delta_over_h_range'] = [str(a), str(b)]; observed['delta_over_h'] = [str(a), str(b)]
        if 'kappa' in values:
            k = values['kappa']; keys(k, {'mode', 'min', 'max', 'values'}, method + '.kappa')
            mode = k.get('mode')
            if mode not in ('absolute', 'multiplier'): raise RequestError(method + '.kappa.mode', 'absolute or multiplier is required')
            # Existing multiplier generation is not a restriction on all saved
            # candidates. Never advertise it as an enforced search interval.
            if mode == 'multiplier':
                raise RequestError(method + '.kappa.multiplier', 'selected normal entry does not enforce a multiplier-only candidate range')
            physical_schedule = config.get('entry') == 'schedule' and config.get('absolute_kappa_schedule') is True
            combined=config.get('combined_width_search') and physical_schedule and method=='PR_PAULI'
            if (config.get('entry') != 'neighbourhood' and not physical_schedule) or (methods != ['PR_PAULI'] and not combined):
                raise RequestError(method + '.kappa.absolute', 'absolute min/max needs an adopted PR-Pauli neighbourhood entry; current entry is ' + config.get('entry', 'method'))
            if 'values' in k: raise RequestError(method + '.kappa.values', 'normal entry takes bounds, not a discrete grid')
            a, b = Fraction(k['min']), Fraction(k['max'])
            if not 0 < a <= b: raise RequestError(method + '.kappa', 'invalid positive absolute bounds')
            if physical_schedule:
                if combined:
                    wide=spec.setdefault('search_overrides',{}).setdefault('schedule_wide_by_method',{})
                    wide[method]=dict(spec['search_overrides'].get('schedule_wide',{}),kappa_bounds=[str(a),str(b)],max_kappa=str(b))
                else:
                    spec.setdefault('search_overrides', {}).setdefault('schedule_wide', {})['kappa_bounds'] = [str(a), str(b)]
                    spec['search_overrides']['schedule_wide']['max_kappa'] = str(b)
                override['absolute_kappa_bounds'] = [str(a), str(b)]
                spec['search_overrides']['kappa_neighbourhood'] = dict(outer_planner='SCHEDULE_WIDE_V1')
            else:
                spec.setdefault('search_overrides', {}).setdefault('kappa_neighbourhood', {})['bounds'] = [str(a), str(b)]
            spec['kappa_families'] = override.get('L_D_values', spec.get('kappa_families'))
            observed['kappa'] = dict(mode=mode, min=str(a), max=str(b))
        canonical['per_method'][method] = observed
    shots = scope.get('shots', 'auto')
    if shots != 'auto':
        if not isinstance(shots, dict): raise RequestError('scope.shots', 'use auto or min/max_per_round')
        keys(shots, {'min', 'max_per_round'}, 'scope.shots')
        if shots.get('min', 1) != 1: raise RequestError('scope.shots.min', 'normal optimizer supports a lower bound of 1 only')
        caps = shots.get('max_per_round')
        if type(caps) is int: caps = [caps] * (problem['rpe']['M'] + 1)
        if not isinstance(caps, list) or len(caps) != problem['rpe']['M'] + 1 or any(type(n) is not int or n < 1 for n in caps):
            raise RequestError('scope.shots.max_per_round', 'positive per-round caps required')
        for method in methods: spec['method_overrides'].setdefault(method, {})['caps'] = caps
        canonical['shots'] = dict(min=1, max_per_round=caps)
    defaults = config.get('preparation_budget_defaults', {})
    requested_budgets = scope.get('preparation_budgets', {})
    if 'preparation_budgets' in scope and not requested_budgets:
        raise RequestError('scope.preparation_budgets', 'supply at least one stage or omit the field')
    for field, budgets in (('preparation_budget_defaults', defaults),
                           ('scope.preparation_budgets', requested_budgets)):
        keys(budgets, {'signal', 'compile', 'audit'}, field)
        if any(v != 'UNLIMITED' and (type(v) is not int or v < 0) for v in budgets.values()):
            raise RequestError(field, 'nonnegative stage-task budgets or UNLIMITED required')
    if defaults or 'preparation_budgets' in scope:
        distributed=spec.get('executor',{}).get('distributed')
        if distributed is None:raise RequestError('scope.preparation_budgets','adopted entry has no distributed preparation executor')
        distributed.setdefault('preparation',{}).setdefault('budgets',{}).update(defaults | requested_budgets)
    if 'preparation_budgets' in scope:
        canonical['preparation_budgets'] = deepcopy(requested_budgets)
    stop = scope.get('stop', {})
    keys(stop, {'max_new_q3_per_method', 'max_steps_per_method', 'max_cycles', 'batch'}, 'scope.stop')
    for key in stop:
        value = stop[key]
        if key == 'max_new_q3_per_method':
            if not isinstance(value, dict): value = {m: value for m in methods}
            if set(value) != set(methods) or any(v != 'UNLIMITED' and (type(v) is not int or v < 0) for v in value.values()):
                raise RequestError('scope.stop.' + key, 'nonnegative budgets or UNLIMITED for each selected method required')
            spec[key] = value
        elif type(value) is not int or value < 1:
            raise RequestError('scope.stop.' + key, 'positive integer required')
        elif key == 'max_steps_per_method':
            spec[key] = value
            if key in spec.get('search_overrides',{}):spec['search_overrides'][key]=value
        elif config.get('entry', 'method') == 'method':
            raise RequestError('scope.stop.' + key, 'this entry does not consume cycle/batch limits')
    # A clearly scoped start is the user's execution grant. The zero budgets
    # of a marked plan-only template are preparation gates, not scientific
    # stopping limits. A caller's explicit zero budget remains zero.
    if config.get('plan_template') and not spec.get('science_approved') and 'max_new_q3_per_method' not in stop:
        spec['max_new_q3_per_method'] = {m: 'UNLIMITED' if spec['max_new_q3_per_method'][m] == 0 else spec['max_new_q3_per_method'][m] for m in methods}
    if 'max_new_q3_per_method' in stop and all(spec['max_new_q3_per_method'][m]=='UNLIMITED' for m in methods):
        spec['search_to_rule_stop']=True
        if 'max_steps_per_method' not in stop:
            spec['max_steps_per_method']=None
            spec.setdefault('search_overrides',{}).pop('max_steps_per_method',None)
    widths = spec.get('search_overrides', {}).get('requested_widths') or {}
    if (config.get('live_ready_refill') is True and config.get('entry') == 'schedule'
            and any(widths.get(method) for method in methods if method.startswith('PR_'))):
        spec.update(live_ready_refill=True, overlap_ready_batch=True)
    if request['action'] in ('start','prepare'):
        spec['science_approved'] = True
    return problem, spec, canonical

@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'REGISTRY.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX); yield

def summarize(problem, spec, canonical):
    effective = deepcopy(problem['methods'])
    for method, change in spec.get('method_overrides', {}).items():
        change=deepcopy(change)
        if 'generation' in change:effective[method]['generation']=dict(effective[method].get('generation',{}),**change.pop('generation'))
        effective[method].update(change)
    result = dict(methods=spec['methods'], requested=canonical,
        fixed=dict(problem=problem.get('problem'), accuracy=problem.get('accuracy'), target=problem.get('target'), rpe=problem.get('rpe')),
        method_settings={m:effective[m] for m in spec['methods']},method_overrides=spec.get('method_overrides', {}), search=dict(problem.get('search', {}), **spec.get('search_overrides', {})),
        max_new_q3_per_method=spec.get('max_new_q3_per_method'),max_steps_per_method=spec.get('max_steps_per_method',problem['search']['max_steps_per_method']))
    preparation = (spec.get('executor', {}).get('distributed') or {}).get('preparation')
    if preparation is not None:
        result['preparation_budgets'] = {stage: preparation.get('budgets', {}).get(stage, 0)
                                         for stage in ('signal', 'compile', 'audit')}
    if spec.get('live_ready_refill') is True:
        result['live_ready_refill'] = True
    return result


def local0_progress(applied, spec, run=None):
    """Report central acceptance separately from transferred execution receipts."""
    effective = {key: applied[key] for key in ('preparation_budgets', 'max_new_q3_per_method') if key in applied}
    executor = spec.get('executor', {})
    effective['resources'] = dict(max_concurrent_jobs=executor.get('max_concurrent_jobs'))
    progress = dict(schema='RPE_LOCAL0_PROGRESS_V1', effective=effective, candidates=[], stages=[])
    parents = spec.get('q3_budget_parents', [])
    parent = (spec.get('continuation') or {}).get('parent_run_id')
    if parent or len(parents) == 1:
        progress['continuation'] = dict(parent_run_id=parent or Path(parents[0]['path']).name)
    elif parents:
        progress['continuation'] = dict(parent_run_ids=[Path(row['path']).name for row in parents])
    if run is None:
        return progress
    preflight = run/'DUPLICATE_PREFLIGHT.json'
    if preflight.is_file():
        progress['saved_evaluations'] = load(preflight).get('reused', {})
    inherited_budget = run/'Q3_BUDGET_CONTINUATION.json'
    if inherited_budget.is_file():
        effective['inherited_q3'] = load(inherited_budget)
    observations = {}
    width_state = run/'KAPPA_STATE.json'
    if width_state.is_file():
        state = load(width_state)
        pending = {'UNSUPPORTED_MAPPER': 'UNSUPPORTED_GENERATION', 'WAITING_INPUTS': 'PREPARATION_WAITING',
                   'NEEDS_FAMILY_CONFIGURATION_SOURCE': 'MISSING_FIXED_INPUT', 'HELD': 'TECHNICAL_FAILURE'}
        for key, point in state.get('families', {}).items():
            if point.get('status') in pending:
                observations[key] = dict(candidate_id=key, method=point['method'],
                                         state=pending[point['status']], reason=point.get('reason'))
        for point in state.get('points', []):
            key = point.get('configuration_id') or point.get('family')
            status = pending.get(point.get('status'))
            if point.get('status') == 'EVALUATED' and point.get('saved_point_ids'):
                status = 'REUSED_SAVED_RESULT'
            if key and status:
                observations[key] = dict(candidate_id=key, method=point['method'], state=status, reason=point.get('reason'))
    database = run/'distributed/QUEUE.sqlite3'
    if not database.exists():
        progress['candidates'] = list(observations.values())
        return progress
    import sqlite3
    identified = {}
    try:
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True, timeout=1) as db:
            rows = db.execute('SELECT job_id,method,request,state,adopted_epoch,detail FROM jobs ORDER BY rowid').fetchall()
        stages = {}
        for job, method, raw, state, adopted, detail in rows:
            request = json.loads(raw, parse_int=str)
            prep = request.get('preparation', {})
            stage = prep.get('stage', 'q3')
            count = stages.setdefault(stage, dict(stage=stage, queue_accepted=0, execution_started=0,
                execution_completed=0, execution_unit='evaluation' if stage == 'q3' else 'stage_task',
                active_receipts_pending=0))
            count['queue_accepted'] += 1
            directory = run/'jobs'/method/job
            timing = directory/'JOB_STAGES.jsonl'
            events = set()
            if timing.is_file():
                for line in timing.read_bytes().splitlines(keepends=True):
                    if line.endswith(b'\n'):
                        events.add(json.loads(line).get('event'))
            receipt = 'Q3_BODY' if stage == 'q3' else 'PREPARATION_BODY'
            native_start = receipt+'_START' in events or receipt+'_END' in events
            native_end = receipt+'_END' in events
            # Older routes may have final receipts without the timing markers.
            terminal = (directory/'PREPARATION_RECEIPT.json').is_file() if prep else adopted is not None
            count['execution_started'] += int(native_start or terminal)
            count['execution_completed'] += int(native_end or terminal)
            active = state in ('STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER')
            count['active_receipts_pending'] += int(active and not (native_start or terminal))
            if prep:
                if str(prep.get('provider', '')).startswith('PF_'):
                    candidate = next(iter(prep.get('outputs', [])), '').rsplit('/', 1)[-1]
                    method = 'TRUNCATED_PF' if 'TRUNCATED_PF' in applied.get('methods', []) else 'FULL_PF'
                else:
                    candidate = ''
                code = 'PREPARATION_RUNNING' if active else 'PREPARATION_WAITING'
                if terminal:
                    preparation_result = load(directory/'PREPARATION_RECEIPT.json')
                    if preparation_result.get('returncode') != 0:
                        code = 'TECHNICAL_FAILURE'
                    elif prep.get('provider') == 'PF_FINISH':
                        code = 'EVALUATION_WAITING'
            else:
                candidate = request.get('configuration_id')
                code = 'EVALUATION_COMPLETED' if adopted is not None else 'EVALUATION_RUNNING' if active else 'EVALUATION_WAITING'
            if candidate and state != 'CANCELLED':
                metadata = json.loads(detail)
                if state in ('FAILED', 'UNKNOWN') or (state == 'ACCEPTED' and not prep and adopted is None):
                    code = 'TECHNICAL_FAILURE' if state != 'ACCEPTED' else 'EVALUATION_WAITING'
                key = (candidate, stage)
                priority = 3 if active else 2 if state in ('QUEUED', 'ASSIGNED') else 1
                if key not in identified or priority >= identified[key][0]:
                    identified[key] = (priority, dict(candidate_id=candidate, method=method, state=code,
                                                     reason=metadata.get('reason'), stage=stage))
        for (candidate, stage), (_, row) in identified.items():
            if stage == 'q3' or candidate not in observations:
                observations[candidate] = row
        progress['stages'] = list(stages.values())
    except (sqlite3.Error, OSError, ValueError) as error:
        progress['receipt_status'] = str(error)
    progress['candidates'] = list(observations.values())
    return progress


class Gateway:
    def __init__(self, config, *, preflight=None, launch=None):
        self.config = deepcopy(config); self.state = Path(config['state_directory'])
        self.preflight = preflight or self.normal_preflight; self.launch = launch or self.normal_launch

    def create_run_directory(self, run):
        evidence = self.config.get('run_evidence_root')
        if not evidence:
            run.mkdir(parents=True)
            return
        # The central index remains under state/runs. Preparation publishes Git
        # locators from the run's physical directory in the controller checkout.
        checkout = Path(self.config['code_root']).resolve()
        target = Path(evidence).resolve() / run.name
        if not target.resolve().is_relative_to(checkout) or target.is_symlink():
            raise RequestError('run_evidence_root', 'run evidence must be inside the controller checkout')
        run.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(parents=True, exist_ok=True)
        run.symlink_to(target, target_is_directory=True)

    def handle(self, request):
        from trottertracks.rpe_release_registry import select
        try:
            release_config = select(self.config, request)
        except ValueError as error:
            raise RequestError('release', str(error)) from error
        if release_config is not self.config:
            return Gateway(release_config,
                preflight=None if self.preflight == self.normal_preflight else self.preflight,
                launch=None if self.launch == self.normal_launch else self.launch).handle(request)
        selected = request_service(self.config, request)
        if selected is not self.config:
            return Gateway(selected,
                preflight=None if self.preflight == self.normal_preflight else self.preflight,
                launch=None if self.launch == self.normal_launch else self.launch).handle(request)
        if request.get('action') not in ('preview', 'planonly', 'prepare', 'start', 'resume', 'stop', 'status'):
            raise RequestError('action', 'preview, planonly, prepare, start, resume, stop, or status required')
        if (self.state/'MIGRATED_TO.json').exists() and request['action'] not in ('preview', 'status'):
            raise RequestError('controller', 'this state moved; use '+str(load(self.state/'MIGRATED_TO.json').get('controller','RomuTail')))
        with locked(self.state):
            current = self.state / 'CURRENT_TARGET.json'
            if not request.get('scope'):
                if request.get('run_id'):
                    registry=load(self.state/'RUNS.json') if (self.state/'RUNS.json').exists() else {}
                    if request['run_id'] not in registry:raise RequestError('run_id','unknown existing run')
                    run=Path(registry[request['run_id']]['run_directory'])
                else:
                    if not current.exists(): raise RequestError('target', 'controller has no confirmed target; provide a scope or register a confirmed run')
                    target = load(current); run = Path(target['run_directory'])
                if request['action'] == 'status': return self.result(run, 'STATUS')
                if request['action'] == 'stop': return self.result(run, self.stop(run))
                if request['action'] == 'planonly': return self.result(run, self.plan_only(run))
                if request['action'] in ('preview','prepare'): return self.result(run, 'CONFIRMED_TARGET')
                if not str(request.get('instruction', '')).strip(): raise RequestError('instruction', 'explicit start instruction required')
                status=self.start(run, request)
                return self.result(run, status)
            problem, spec, canonical = resolve(self.config, request)
            entry = self.config.get('entry', 'method')
            if entry not in ENTRIES: raise RequestError('entry', 'unknown normal entry')
            base_identity = {k: v for k, v in problem.items() if k not in ('methods', 'search', 'execution_routes', 'price_rules')}
            evaluation = digest(dict(inputs=base_identity, computation_source=self.config['computation_source']))
            # Original JSON numeric representations also remain in REQUEST.json.
            summary = summarize(problem, spec, canonical)
            signature = digest(dict(problem=problem, method_settings=summary['method_settings'],round={k:v for k,v in spec.items() if k not in ('round_id', 'science_approved','note','method_overrides')}, entry=entry,
                code_source=self.config['code_source']))
            run_id = 'search-' + signature[:20]; run = self.state / 'runs' / run_id
            if request['action'] == 'preview':
                preview = dict(run_id=run_id, status='PREVIEW', applied=summary, started=False,
                               computation_source=self.config['computation_source'], release=self.config.get('release'),
                               controller_source=self.config['code_source'])
                if run.is_dir():
                    preview['runtime'] = self.runtime_status(run)
                preview['progress'] = local0_progress(summary, spec, run if run.is_dir() else None)
                return preview
            if request['action'] in ('stop','resume','status'):
                raise RequestError('scope','use the existing run_id or confirmed target for status/stop/resume')
            if request['action']=='start' and not str(request.get('instruction', '')).strip(): raise RequestError('instruction', 'explicit start instruction required')
            registry_path = self.state / 'RUNS.json'; registry = load(registry_path) if registry_path.exists() else {}
            if not run.exists():
                self.create_run_directory(run)
                prior = list(spec.get('executor', {}).get('prior_run_dirs', []))
                prior += [v['run_directory'] for v in registry.values() if v['evaluation_identity'] == evaluation]
                spec.setdefault('executor', {})['prior_run_dirs'] = sorted(set(str(Path(p).resolve()) for p in prior))
                distributed=spec.get('executor',{}).get('distributed')
                if distributed:
                    for key,name in [('host_admission','HOST_ADMISSION.json'),('host_eligibility','HOST_ELIGIBILITY.json')]:
                        if distributed.get(key):
                            source=Path(distributed[key]);target=run/name
                            target.write_bytes(source.read_bytes());distributed[key]=str(target)
                spec['round_id'] = run_id
                immutable(run/'PROBLEM.json', problem); immutable(run/'ROUND.requested.json', spec)
                immutable(run/'APPLIED_SCOPE.json', dict(schema='LOCAL0_APPLIED_SEARCH_SCOPE_V1',run_id=run_id,signature=signature,
                    evaluation_identity=evaluation,entry=entry,code_source=self.config['code_source'],applied=summary,
                    problem_sha256=digest(problem),round_requested_sha256=digest(spec),scope=canonical))
                immutable(run/'REQUEST.json', request)
                pinned=deepcopy(self.config);pinned['config_path']=str(run/'CONTROLLER_CONFIG.json')
                immutable(run/'CONTROLLER_CONFIG.json',pinned)
                if self.config.get('release'):immutable(run/'RELEASE.json',self.config['release'])
                registry[run_id]=dict(run_directory=str(run),evaluation_identity=evaluation,signature=signature)
                write(registry_path, registry)
            write(current, dict(run_id=run_id,run_directory=str(run)))
            if request['action']=='planonly':return self.result(run,self.plan_only(run))
            if request['action']=='prepare':return self.result(run,'PREPARED_NO_EXECUTION_GRANT')
            status=self.start(run, request)
            return self.result(run, status)

    def start(self, run, request):
        state = load(run/'LAUNCH.json') if (run/'LAUNCH.json').exists() else None
        completed=load(run/'EXIT.json') if (run/'EXIT.json').exists() else {}
        if state and (not state.get('invocation_id') or completed.get('invocation_id')!=state.get('invocation_id')):
            pid = state.get('pid')
            if pid and Path('/proc/%d/cmdline' % pid).exists() and str(run) in Path('/proc/%d/cmdline' % pid).read_bytes().decode(errors='replace'):
                return 'ALREADY_RUNNING'
            raise RequestError('run', 'previous launch has no completion receipt; existing STOP/UNKNOWN is preserved')
        stop_request = run/'STOP_REQUESTED.json'
        user_interrupted = (request.get('action') == 'resume' and completed.get('returncode') == -signal.SIGINT
                            and stop_request.is_file() and load(stop_request).get('reason') == 'USER_REQUESTED_STOP')
        if completed and completed.get('returncode') != 0 and not user_interrupted:
            stop_path = run/'KAPPA_STATE.json'
            stop = (load(stop_path).get('stop') or {}).get('reason') if stop_path.exists() else None
            raise RequestError('run', 'failed invocation requires reconciliation: '+str(stop or completed.get('returncode')))
        if (run/'ROUND_RESULT.json').exists():
            outcomes=load(run/'ROUND_RESULT.json')['methods']
            reasons={v['stop_reason'] for v in outcomes.values()}
            holds={'UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD'}
            if reasons & holds:raise RequestError('run','existing scientific hold/STOP is preserved: '+','.join(sorted(reasons & holds)))
            settings=load(run/'ROUND.requested.json')
            if settings.get('search_to_rule_stop'):
                final=run/'SEARCH_TERMINATION.json'
                if final.exists() and not user_interrupted:
                    return 'REUSED_COMPLETED'
            elif not user_interrupted and reasons and reasons <= {'NO_IMPROVING_PROPOSAL','NO_FEASIBLE_POINT_IN_RANGE','Q3_BUDGET_EXHAUSTED','STEP_LIMIT','NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET','NO_CANDIDATE_LEFT_IN_RANGE','SCHEDULE_WIDE_RULE_EXHAUSTED'}:
                return 'REUSED_COMPLETED'
        spec = load(run/'ROUND.requested.json'); problem = load(run/'PROBLEM.json')
        applied = load(run/'APPLIED_SCOPE.json')
        if applied['code_source']!=self.config['code_source']:raise RequestError('code_source','confirmed target belongs to another fixed deployment; preserve it and prepare the intended scope on this deployment')
        if digest(problem) != applied['problem_sha256'] or digest(spec) != applied['round_requested_sha256']:
            raise RequestError('run', 'saved applied settings changed')
        if spec.get('continuation'):
            from .rpe_search_continuation import verify
            write(run/'CONTINUATION_PREFLIGHT.json', verify(run, problem, spec))
        if spec.get('q3_budget_parents'):
            from .rpe_search_continuation import verify_budget_parents
            immutable(run/'Q3_BUDGET_CONTINUATION.json',verify_budget_parents(run,problem,spec))
        report = self.preflight(run, problem, spec)
        write(run/'DUPLICATE_PREFLIGHT.json', report)
        if report.get('status') != 'PASS': raise RequestError('preflight', '§10.5 evidence is incomplete/conflicting; see DUPLICATE_PREFLIGHT.json')
        immutable(run/'ROUND.json', spec)
        write(run/'AUTHORIZATION.json', dict(kind='EXPLICIT_USER_START',instruction=request['instruction'],scope_signature=applied['signature'],authorized_utc=time.time()))
        if (run/'STOP_REQUESTED.json').exists():
            previous=load(run/'STOP_REQUESTED.json')
            write(run/'stop-history'/(str(time.time_ns())+'.json'),previous)
            (run/'STOP_REQUESTED.json').unlink()
        self.launch(run, problem, spec)
        return 'STARTED'

    def result(self, run, status):
        a = load(run/'APPLIED_SCOPE.json')
        actual = Path(self.config['artificial_controller_directory'])/'run' if self.config.get('artificial_controller_directory') else run
        requested = run/'ROUND.requested.json'
        progress = local0_progress(a['applied'], load(requested) if requested.is_file() else {}, actual)
        result = dict(run_id=a['run_id'],run_directory=str(run),status=status,applied=a['applied'],started=(run/'LAUNCH.json').exists(),
                    computation_source=self.config['computation_source'],release=self.config.get('release'),controller_source=self.config['code_source'],
                    runtime=self.runtime_status(run),progress=progress,
                    preflight=load(run/'DUPLICATE_PREFLIGHT.json') if (run/'DUPLICATE_PREFLIGHT.json').exists() else None)
        if 'continuation' in progress:
            result['continuation'] = progress.pop('continuation')
        return result

    def runtime_status(self,run):
        launch=load(run/'LAUNCH.json')if(run/'LAUNCH.json').exists()else{}
        finished=load(run/'EXIT.json')if(run/'EXIT.json').exists()else{}
        complete=bool(launch)and finished.get('invocation_id')==launch.get('invocation_id')
        pid=launch.get('pid');command=Path('/proc')/str(pid)/'cmdline'
        try:alive=bool(pid)and str(run).encode()in command.read_bytes()
        except OSError:alive=False
        phase='PREPARED'
        if complete:phase='COMPLETE'if finished.get('returncode')==0 else'FAILED'
        elif launch:phase='RUNNING'if alive else'RECONCILIATION_REQUIRED'
        if(run/'STOP_REQUESTED.json').exists():phase='STOP_REQUESTED_DRAINING'if alive else'STOPPED'
        actual=Path(self.config['artificial_controller_directory'])/'run'if self.config.get('artificial_controller_directory')else run
        result=run/'ROUND_RESULT.json'
        reasons={method:value.get('stop_reason')for method,value in load(result).get('methods',{}).items()}if result.exists()else{}
        if phase == 'COMPLETE' and reasons and any(reason in {'UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW',
                'NO_ROUTE_ADAPTER', 'HOST_ELIGIBILITY_HOLD'} for reason in reasons.values()):
            phase = 'RECONCILIATION_REQUIRED'
        value=dict(phase=phase,controller_alive=alive,invocation_id=launch.get('invocation_id'),
            returncode=finished.get('returncode')if complete else None,stop_reasons=reasons)
        width_state = actual/'KAPPA_STATE.json'
        if width_state.is_file():
            state = load(width_state)
            value['search_stop'] = state.get('stop')
            value['candidate_holds'] = [
                {key: row[key] for key in ('method', 'L_D', 'kappa', 'delta_pf', 'status', 'reason') if key in row}
                for row in [*state.get('families', {}).values(), *state.get('points', [])]
                if row.get('status') in ('UNSUPPORTED_MAPPER', 'NEEDS_FAMILY_CONFIGURATION_SOURCE',
                                         'WAITING_INPUTS', 'HELD')]
        database=actual/'distributed/QUEUE.sqlite3'
        if database.exists():
            import sqlite3
            try:
                with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=1)as connection:
                    value['jobs_by_state']=dict(connection.execute('SELECT state,count(*) FROM jobs GROUP BY state'))
                    value['adopted_jobs']=connection.execute('SELECT count(*) FROM jobs WHERE adopted_epoch IS NOT NULL').fetchone()[0]
                    value['jobs_by_stage'] = {}
                    for stage, state, count in connection.execute("""SELECT
                            coalesce(json_extract(request, '$.preparation.stage'), 'q3'), state, count(*)
                            FROM jobs GROUP BY 1, 2"""):
                        value['jobs_by_stage'].setdefault(stage, {})[state] = count
            except sqlite3.Error as error:value['queue_status_error']=str(error)
        return value

    def normal_preflight(self, run, problem, spec):
        if self.config.get('artificial_controller_directory'):
            fixture=Path(self.config['artificial_controller_directory'])
            if not str(problem.get('problem',{}).get('system','')).startswith('ARTIFICIAL'):
                raise RequestError('fixture','artificial controller requires an artificial problem')
            if not load(fixture/'ROUND.json').get('artificial_fixture',{}).get('only'):
                raise RequestError('fixture','artificial controller directory is not explicitly artificial')
            return dict(status='PASS',scientific_calls=0,artificial_only=True,fixture=str(fixture))
        from trottertracks import rpe_method_optimizer as O
        sys.path.insert(0,str(Path(self.config['code_root'])/'scripts'))
        import run_rpe_method_optimizer as CLI
        root=Path(self.config['code_root'])
        fixed_implementation(self.config)
        scoped = CLI.price_overrides(CLI.scoped(problem,spec.get('method_overrides')),spec)
        if spec.get('search_to_rule_stop') or spec.get('live_ready_refill'):
            from .rpe_schedule_search import requested_widths
            for method,widths in (requested_widths(scoped,spec) or {}).items():
                scoped['methods'][method]['requested_delta_pf_values']=[
                    O.enc(Fraction(entry['exact'])) for entry in widths]
        context = O.prepare(Path(self.config['code_root']),scoped,methods=spec['methods'],generate_configurations=False,preparation_dir=run/'preflight-inputs')
        routes = CLI.merged_routes(scoped,spec)
        if spec.get('search_to_rule_stop'):
            from .pf_optimizer_connection import registry_routes
            for method in spec['methods']:
                registered=scoped['methods'][method].get('configuration_ids')
                if method in ('FULL_PF','TRUNCATED_PF') and registered is not None:
                    observed={candidate['configuration_id'] for candidate in context['candidates'] if candidate['method']==method}
                    if observed!=set(registered):
                        raise RequestError('scope',method+' registered configurations are not represented by the fixed candidate index')
            if context.get('pf_evidence_registry'):
                routes=registry_routes(routes,context['pf_evidence_registry'])
            completion=spec.get('candidate_completion') or {}
            if completion.get('resume_preparation_from'):
                from .pr_candidate_completion import Completion
                prepared=Completion(run/'preflight-completion',completion)
                routes,_=prepared.install_snapshot(context,dict(scoped,execution_routes=routes),prepared.state['records'])
        ex = O.Executor(Path(self.config['code_root']),run,routes,context['expected'],scoped,prior_run_dirs=spec.get('executor',{}).get('prior_run_dirs',[]),launch_disabled=True)
        loaded, refused = {}, []
        missing=[str(p) for p in spec.get('executor',{}).get('prior_run_dirs',[]) if not Path(p).is_dir()]
        in_progress=[]
        ids={c['configuration_id'] for c in context['candidates'] if c['method'] in spec['methods']}
        for directory in spec.get('executor',{}).get('prior_run_dirs',[]):
            prior_root=Path(directory)
            if (prior_root/'APPLIED_SCOPE.json').exists() and (prior_root/'LAUNCH.json').exists():
                launch=load(prior_root/'LAUNCH.json');exit=load(prior_root/'EXIT.json') if (prior_root/'EXIT.json').exists() else {}
                if launch.get('invocation_id')!=exit.get('invocation_id'):
                    previous=load(prior_root/'APPLIED_SCOPE.json')
                    if set(previous['applied']['methods']) & set(spec['methods']):in_progress.append(str(prior_root))
            for path in Path(directory).glob('jobs/*/*/REQUEST.json'):
                job=path.parent
                if (job/'OUTCOME.json').exists():continue
                if (job/'LAUNCH.json').exists() or (job/'STARTING.json').exists():
                    if load(path).get('configuration_id') in ids:in_progress.append(str(job))
        for method in spec['methods']:
            known, rejected = ex.for_method(method).prior_records(context['candidates'])
            loaded[method]=len(known);refused.extend(dict(method=method,**r) for r in rejected)
        # A different configuration/route is excluded, never accepted as a
        # match. Missing/corrupt records for a same-key candidate need review.
        holds=[r for r in refused if r['reason'] not in ('CONFIGURATION_NOT_IN_THIS_ROUND','DIFFERENT_JOB_KEY_OR_ROUTE')]
        return dict(status='HOLD' if holds or missing or in_progress else 'PASS',section='10.5',reused=loaded,refused=refused,missing_prior_run_dirs=missing,owned_unresolved_prior_jobs=in_progress,
            checked_prior_run_dirs=spec.get('executor',{}).get('prior_run_dirs',[]),normal_ingest='Executor.prior_records/ingest',scientific_calls=0)

    def normal_launch(self, run, problem, spec):
        invocation=uuid.uuid4().hex
        command=[self.config['python'],'-B',str(Path(self.config['code_root'])/'scripts/run_rpe_local0_search.py'),
                 '--config',str(run/'CONTROLLER_CONFIG.json') if (run/'CONTROLLER_CONFIG.json').exists() else self.config['config_path'],'--execute-run',str(run),'--invocation',invocation]
        with (run/'DRIVER.log').open('a') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),start_new_session=True)
        marker=dict(pid=process.pid,command=command,invocation_id=invocation)
        write(run/'invocations'/invocation/'LAUNCH.json',marker);write(run/'LAUNCH.json',marker)

    def plan_only(self,run):
        applied=load(run/'APPLIED_SCOPE.json')
        command=[self.config['python'],'-B',str(Path(self.config['code_root'])/'scripts'/ENTRIES[applied['entry']]),
            '--problem',str(run/'PROBLEM.json'),'--round',str(run/'ROUND.requested.json'),'--run-dir',str(run/'plan-only'),'--plan-only']
        if self.config.get('artificial_controller_directory'):
            self.normal_preflight(run,load(run/'PROBLEM.json'),load(run/'ROUND.requested.json'))
            command=[self.config['python'],'-B',str(Path(self.config['code_root'])/'scripts/check_rpe_romutail_distributed.py'),
                '--controller',self.config['artificial_controller_directory'],'--control-run',str(run),'--plan-only']
        if not self.config.get('artificial_controller_directory'):
            for key in ('batch','max_cycles'):
                if key in applied['scope'].get('stop',{}):command+=['--'+key.replace('_','-'),str(applied['scope']['stop'][key])]
        with (run/'PLAN_ONLY.log').open('a')as log:
            value=subprocess.run(command,cwd=self.config['code_root'],stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        write(run/'PLAN_ONLY.json',dict(command=command,returncode=value.returncode,scientific_calls=0))
        return 'PLAN_ONLY_COMPLETE'if value.returncode==0 else'PLAN_ONLY_HOLD'

    def stop(self,run):
        write(run/'STOP_REQUESTED.json',dict(reason='USER_REQUESTED_STOP',requested_epoch=time.time()))
        marker=load(run/'NORMAL_DRIVER.json')if(run/'NORMAL_DRIVER.json').exists()else{}
        pid=marker.get('pid');cmd=Path('/proc')/str(pid)/'cmdline'
        try:arguments=cmd.read_bytes().split(b'\0') if pid else ()
        except FileNotFoundError:arguments=()
        if any(os.fsencode(path) in arguments for path in (run,run.resolve())):
            # The driver inside the mount wrapper notices the STOP marker.
            # Interrupting the wrapper itself could publish before it drains.
            if Path(os.fsdecode(arguments[0])).name!='bwrap':os.kill(pid,signal.SIGINT)
            return'STOP_REQUESTED_DRAINING'
        return'STOPPED_NO_ACTIVE_DRIVER'

def fixed_implementation(config):
    """Compare executable code/settings with SOURCE, allowing saved metadata commits.

    Includes staged and working changes, additions and dependency declarations.
    Run inputs retain their separate immutable scope and fingerprint checks.
    """
    root=Path(config['code_root'])
    paths=['src','scripts','tools','configs','schemas','pyproject.toml',
           'requirements.txt','requirements-gpu.txt','setup.py','setup.cfg']
    changed=subprocess.check_output(['git','diff','--name-only',config['code_source'],'--',*paths],cwd=root,text=True).strip()
    untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*paths],cwd=root,text=True).strip()
    if changed or untracked:
        raise RequestError('code_source','fixed implementation/settings changed: '+(changed or untracked))
    for filename,expected in config.get('fixed_settings',{}).items():
        path=Path(filename)
        if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RequestError('fixed_settings','fixed deployment setting changed: '+filename)
    return dict(source=config['code_source'],head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())


def execute_run(config, run, invocation):
    run=Path(run);scope=load(run/'APPLIED_SCOPE.json');spec=load(run/'ROUND.json')
    root=Path(config['code_root'])
    implementation=fixed_implementation(config)
    if load(run/'DUPLICATE_PREFLIGHT.json')['status']!='PASS':raise RequestError('preflight','no matching successful §10.5 preflight')
    auth=load(run/'AUTHORIZATION.json')
    if auth['scope_signature']!=scope['signature'] or auth['kind']!='EXPLICIT_USER_START':raise RequestError('authorization','scope binding differs')
    if digest(load(run/'PROBLEM.json'))!=scope['problem_sha256'] or digest(spec)!=scope['round_requested_sha256']:
        raise RequestError('run','applied settings differ before normal entry')
    if spec.get('continuation'):
        from .rpe_search_continuation import verify
        if verify(run, load(run/'PROBLEM.json'), spec) != load(run/'CONTINUATION_PREFLIGHT.json'):
            raise RequestError('continuation','parent reconciliation changed before launch')
    if spec.get('q3_budget_parents'):
        from .rpe_search_continuation import verify_budget_parents
        if verify_budget_parents(run,load(run/'PROBLEM.json'),spec)!=load(run/'Q3_BUDGET_CONTINUATION.json'):
            raise RequestError('continuation','capped parent reconciliation changed before launch')
    command=[config['python'],'-B',str(Path(config['code_root'])/'scripts'/ENTRIES[scope['entry']]),
        '--problem',str(run/'PROBLEM.json'),'--round',str(run/'ROUND.json'),'--run-dir',str(run)]
    if config.get('artificial_controller_directory'):
        if not str(load(run/'PROBLEM.json').get('problem',{}).get('system','')).startswith('ARTIFICIAL'):
            raise RequestError('fixture','real problem cannot use the artificial controller')
        fixture=Path(config['artificial_controller_directory'])
        if not load(fixture/'ROUND.json').get('artificial_fixture',{}).get('only'):
            raise RequestError('fixture','fixture is not labelled artificial')
        command=[config['python'],'-B',str(Path(config['code_root'])/'scripts/check_rpe_romutail_distributed.py'),
                 '--controller',str(fixture),'--control-run',str(run)]
    stop=scope['scope'].get('stop',{})
    if not config.get('artificial_controller_directory'):
        for key in ('batch','max_cycles'):
            if key in stop:command+=['--'+key.replace('_','-'),str(stop[key])]
    try:
        process=subprocess.Popen(command,cwd=config['code_root'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        write(run/'NORMAL_DRIVER.json',dict(pid=process.pid,command=command,invocation_id=invocation,implementation=implementation))
        code=process.wait()
        if config.get('artificial_controller_directory'):
            result_path=Path(config['artificial_controller_directory'])/'run/ROUND_RESULT.json'
            if result_path.exists():
                write(run/'ROUND_RESULT.json',load(result_path))
        value=dict(returncode=code,command=command,finished_utc=time.time(),invocation_id=invocation)
        write(run/'invocations'/invocation/'EXIT.json',value);write(run/'EXIT.json',value)
        publish_after_exit(config,run)
    except Exception as error:
        value=dict(returncode=1,error=str(error),command=command,invocation_id=invocation)
        write(run/'invocations'/invocation/'EXIT.json',value);write(run/'EXIT.json',value);raise


def publish_after_exit(config,run):
    """Publish saved result metadata through C's normal path; never alter Q3."""
    publication=config.get('publication')
    if not publication:return
    run=Path(run)
    problem=load(run/'PROBLEM.json')
    if config.get('artificial_controller_directory') or str(problem.get('problem',{}).get('system','')).startswith('ARTIFICIAL'):
        write(run/'PUBLICATION_STATUS.json',dict(status='ARTIFICIAL_SKIPPED',scientific_calls=0))
        return
    command=[config['python'],'-B',str(Path(config['code_root'])/'scripts/publish_romutail_search.py'),
        '--repository',publication['repository'],'--run-dir',str(run),
        '--controller-source',config['code_source']]
    if publication.get('branch'):command+=['--branch',publication['branch']]
    if publication.get('push',False):command+=['--push']
    try:
        with (run/'PUBLICATION.log').open('a')as log:
            result=subprocess.run(command,cwd=config['code_root'],stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        # Publisher may write a more specific pending/published locator status.
        status=load(run/'PUBLICATION_STATUS.json')if(run/'PUBLICATION_STATUS.json').exists()else{}
        status.update(command=command,returncode=result.returncode,scientific_calls=0)
        if result.returncode:status['status']='FINAL_PUBLICATION_PENDING'
        status.setdefault('status','PUBLISHED'if result.returncode==0 else'FINAL_PUBLICATION_PENDING')
        write(run/'PUBLICATION_STATUS.json',status)
    except Exception as error:
        write(run/'PUBLICATION_STATUS.json',dict(status='FINAL_PUBLICATION_PENDING',error=str(error),
            command=command,scientific_calls=0))
