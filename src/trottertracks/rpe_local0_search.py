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
        if scope.get('methods', ['PR_PAULI']) != ['PR_PAULI']:
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
    keys(scope, {'methods', 'per_method', 'shots', 'stop'}, 'scope')
    problem = load(config['problem']); spec = load(config['round'])
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
            if (config.get('entry') != 'neighbourhood' and not physical_schedule) or methods != ['PR_PAULI']:
                raise RequestError(method + '.kappa.absolute', 'absolute min/max needs an adopted PR-Pauli neighbourhood entry; current entry is ' + config.get('entry', 'method'))
            if 'values' in k: raise RequestError(method + '.kappa.values', 'normal entry takes bounds, not a discrete grid')
            a, b = Fraction(k['min']), Fraction(k['max'])
            if not 0 < a <= b: raise RequestError(method + '.kappa', 'invalid positive absolute bounds')
            if physical_schedule:
                spec.setdefault('search_overrides', {}).setdefault('schedule_wide', {})['kappa_bounds'] = [str(a), str(b)]
                override['absolute_kappa_bounds'] = [str(a), str(b)]
                spec['search_overrides']['schedule_wide']['max_kappa'] = str(b)
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
    return dict(methods=spec['methods'], requested=canonical,
        fixed=dict(problem=problem.get('problem'), accuracy=problem.get('accuracy'), target=problem.get('target'), rpe=problem.get('rpe')),
        method_settings={m:effective[m] for m in spec['methods']},method_overrides=spec.get('method_overrides', {}), search=dict(problem.get('search', {}), **spec.get('search_overrides', {})),
        max_new_q3_per_method=spec.get('max_new_q3_per_method'),max_steps_per_method=spec.get('max_steps_per_method',problem['search']['max_steps_per_method']))

class Gateway:
    def __init__(self, config, *, preflight=None, launch=None):
        self.config = deepcopy(config); self.state = Path(config['state_directory'])
        self.preflight = preflight or self.normal_preflight; self.launch = launch or self.normal_launch

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
            if request['action'] == 'preview': return dict(run_id=run_id,status='PREVIEW',applied=summary,started=False,computation_source=self.config['computation_source'],release=self.config.get('release'),controller_source=self.config['code_source'])
            if request['action'] in ('stop','resume','status'):
                raise RequestError('scope','use the existing run_id or confirmed target for status/stop/resume')
            if request['action']=='start' and not str(request.get('instruction', '')).strip(): raise RequestError('instruction', 'explicit start instruction required')
            registry_path = self.state / 'RUNS.json'; registry = load(registry_path) if registry_path.exists() else {}
            if not run.exists():
                run.mkdir(parents=True)
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
        if (run/'ROUND_RESULT.json').exists():
            outcomes=load(run/'ROUND_RESULT.json')['methods']
            reasons={v['stop_reason'] for v in outcomes.values()}
            holds={'UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD'}
            if reasons & holds:raise RequestError('run','existing scientific hold/STOP is preserved: '+','.join(sorted(reasons & holds)))
            if reasons and reasons <= {'NO_IMPROVING_PROPOSAL','NO_FEASIBLE_POINT_IN_RANGE','Q3_BUDGET_EXHAUSTED','STEP_LIMIT','NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET','NO_CANDIDATE_LEFT_IN_RANGE','SCHEDULE_WIDE_RULE_EXHAUSTED'}:
                return 'REUSED_COMPLETED'
        spec = load(run/'ROUND.requested.json'); problem = load(run/'PROBLEM.json')
        applied = load(run/'APPLIED_SCOPE.json')
        if applied['code_source']!=self.config['code_source']:raise RequestError('code_source','confirmed target belongs to another fixed deployment; preserve it and prepare the intended scope on this deployment')
        if digest(problem) != applied['problem_sha256'] or digest(spec) != applied['round_requested_sha256']:
            raise RequestError('run', 'saved applied settings changed')
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
        return dict(run_id=a['run_id'],run_directory=str(run),status=status,applied=a['applied'],started=(run/'LAUNCH.json').exists(),
                    computation_source=self.config['computation_source'],release=self.config.get('release'),controller_source=self.config['code_source'],
                    preflight=load(run/'DUPLICATE_PREFLIGHT.json') if (run/'DUPLICATE_PREFLIGHT.json').exists() else None)

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
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=self.config['code_source']:
            raise RequestError('code_source','server2 checkout differs from the fixed deployment')
        if subprocess.check_output(['git','diff','HEAD','--name-only','--','src','scripts'],cwd=root,text=True).strip():
            raise RequestError('code_source','tracked implementation changed')
        scoped = CLI.price_overrides(CLI.scoped(problem,spec.get('method_overrides')),spec)
        context = O.prepare(Path(self.config['code_root']),scoped,methods=spec['methods'],generate_configurations=False,preparation_dir=run/'preflight-inputs')
        routes = CLI.merged_routes(scoped,spec)
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
        if pid and cmd.exists()and str(run).encode()in cmd.read_bytes():
            os.kill(pid,signal.SIGINT)
            return'STOP_REQUESTED_DRAINING'
        return'STOPPED_NO_ACTIVE_DRIVER'

def execute_run(config, run, invocation):
    run=Path(run);scope=load(run/'APPLIED_SCOPE.json');spec=load(run/'ROUND.json')
    root=Path(config['code_root'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=config['code_source'] or subprocess.check_output(['git','diff','HEAD','--name-only','--','src','scripts'],cwd=root,text=True).strip():
        raise RequestError('code_source','fixed server2 implementation changed before normal launch')
    if load(run/'DUPLICATE_PREFLIGHT.json')['status']!='PASS':raise RequestError('preflight','no matching successful §10.5 preflight')
    auth=load(run/'AUTHORIZATION.json')
    if auth['scope_signature']!=scope['signature'] or auth['kind']!='EXPLICIT_USER_START':raise RequestError('authorization','scope binding differs')
    if digest(load(run/'PROBLEM.json'))!=scope['problem_sha256'] or digest(spec)!=scope['round_requested_sha256']:
        raise RequestError('run','applied settings differ before normal entry')
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
        write(run/'NORMAL_DRIVER.json',dict(pid=process.pid,command=command,invocation_id=invocation))
        code=process.wait()
        if config.get('artificial_controller_directory'):
            result_path=Path(config['artificial_controller_directory'])/'run/ROUND_RESULT.json'
            if result_path.exists():
                write(run/'ROUND_RESULT.json',load(result_path))
        value=dict(returncode=code,command=command,finished_utc=time.time(),invocation_id=invocation)
        write(run/'invocations'/invocation/'EXIT.json',value);write(run/'EXIT.json',value)
    except Exception as error:
        value=dict(returncode=1,error=str(error),command=command,invocation_id=invocation)
        write(run/'invocations'/invocation/'EXIT.json',value);write(run/'EXIT.json',value);raise
