"""Local0 request -> immutable server2 scope -> existing optimizer entry.

No optimizer or scientific adapter is implemented here. Saved Q3 acceptance is
delegated to Executor.prior_records/ingest, including route and identity checks.
"""
from collections import Counter
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


def exact_values(value, field):
    """Canonical finite positive rational list; never reinterpret a JSON float."""
    if not isinstance(value, list) or not value:
        raise RequestError(field, 'a nonempty exact string/integer list is required')
    try:
        if any(isinstance(v, bool) or not isinstance(v, (str, int)) for v in value):
            raise ValueError('inexact value')
        values = sorted(set(Fraction(v) for v in value))
        if values[0] <= 0:
            raise ValueError('nonpositive value')
    except (ValueError, ZeroDivisionError):
        raise RequestError(field, 'positive exact fraction/decimal strings or integers required')
    return [str(v) for v in values]


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
    keys(scope, {'system', 'methods', 'per_method', 'shots', 'stop', 'preparation_budgets', 'scaled_shots',
                 'pbs_walltime_overrides', 'execution_hosts'}, 'scope')
    problem = load(config['problem']); spec = load(config['round'])
    if 'system' in scope and scope['system'] != problem.get('problem', {}).get('system'):
        raise RequestError('scope.system', 'system is unavailable on the selected release')
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
    if 'q3_budget_parents' in spec:
        spec['q3_budget_parents'] = [parent for parent in spec['q3_budget_parents']
                                     if parent['method'] in methods]
    executor = spec.get('executor', {})
    if 'prior_q3_holds' in executor:
        executor['prior_q3_holds'] = [hold for hold in executor['prior_q3_holds']
                                      if hold['method'] in methods]
    if 'width_families' in spec:
        spec['width_families'] = [family for family in spec['width_families']
                                  if family['method'] in methods]
    search = spec.get('search_overrides', {})
    widths = search.get('requested_widths', problem['search'].get('requested_widths'))
    if isinstance(widths, dict) and widths:
        scoped_widths = {method: values for method, values in widths.items()
                         if method in methods or method not in ('PR_PAULI', 'PR_DF')}
        if scoped_widths != widths:
            spec.setdefault('search_overrides', {})['requested_widths'] = scoped_widths or None
    walltime = config.get('pbs_walltime_policy')
    overrides = scope.get('pbs_walltime_overrides', {})
    if walltime:
        from .rpe_pbs import apply_walltime_policy
        try:
            if not isinstance(overrides, dict):
                raise ValueError('Use a mapping from PBS rule IDs to seconds')
            pbs = spec['executor']['distributed']['pbs']
            apply_walltime_policy(pbs, load(walltime), overrides,
                                  computation_source=config['computation_source'])
        except (KeyError, ValueError, TypeError) as error:
            raise RequestError('scope.pbs_walltime_overrides', str(error)) from error
    elif 'pbs_walltime_overrides' in scope:
        raise RequestError('scope.pbs_walltime_overrides', 'Selected release has no shared PBS walltime policy')
    spec.setdefault('max_steps_per_method',spec.get('search_overrides',{}).get('max_steps_per_method',problem['search']['max_steps_per_method']))
    spec['method_overrides'] = {m: v for m, v in spec.get('method_overrides', {}).items() if m in methods}
    if 'max_new_q3_per_method' in spec:
        spec['max_new_q3_per_method'] = {m: spec['max_new_q3_per_method'].get(m, 0) for m in methods}
    canonical = dict(methods=methods, per_method={}, shots=scope.get('shots', 'auto'), stop=scope.get('stop', {}))
    if 'execution_hosts' in scope:
        hosts = scope['execution_hosts']
        distributed = executor.get('distributed') or {}
        workers = (distributed.get('pbs') or {}).get('workers', [])
        configured = {worker['host_id'] for worker in workers}
        if (not isinstance(hosts, list) or not hosts or
                any(not isinstance(host, str) or host not in configured for host in hosts)):
            raise RequestError('scope.execution_hosts', 'choose a nonempty list of fixed PBS host IDs')
        hosts = sorted(set(hosts))
        distributed['execution_hosts'] = hosts
        distributed['pbs']['workers'] = [worker for worker in workers if worker['host_id'] in hosts]
        for name in ('local_worker', 'relay'):
            distributed.pop(name, None)
        distributed['required_hosts'] = [host for host in distributed.get('required_hosts', []) if host in hosts]
        distributed['preparation_profiles'] = {
            host: profile for host, profile in (distributed.get('preparation_profiles') or {}).items()
            if host in hosts
        }
        canonical['execution_hosts'] = hosts
    fragment_count = problem['problem']['hamiltonian_data']['expected']['fragment_count']
    for method, values in scope.get('per_method', {}).items():
        if method not in methods: raise RequestError('scope.per_method.' + method, 'method is not selected')
        keys(values, {'L_D', 'p', 'formulas', 'delta_pf', 'delta_over_h', 'delta_over_h_values', 'kappa'}, method)
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
            supported = (current['formulas'] if current['family'] == 'PF' else
                         current.get('generation', {}).get('supported_formulas',
                         current.get('generation', {}).get('formulas', [])))
            if not isinstance(forms, list) or not forms or any(f not in supported for f in forms):
                raise RequestError(method + '.formulas', 'unsupported formula')
            forms = sorted(set(forms))
            if current['family'] == 'PF':
                override['formulas'] = forms
            else:
                override.setdefault('generation', {})['formulas'] = forms
            observed['formulas'] = forms
        if 'delta_over_h_values' in values:
            if 'delta_over_h' in values or 'delta_pf' in values:
                raise RequestError(method + '.delta_over_h_values', 'choose one width representation')
            widths = exact_values(values['delta_over_h_values'], method + '.delta_over_h_values')
            observed['delta_over_h_values'] = widths
            if current['family'] == 'PF':
                override.update(delta_over_h_values=widths,
                                delta_over_h_range=[widths[0], widths[-1]],
                                configuration_ids=None)
                if method == 'FULL_PF':
                    override['p_values'] = [fragment_count]
            else:
                if config.get('entry') != 'schedule':
                    raise RequestError(method + '.delta_over_h_values', 'requested-width schedule entry required')
                raw_h = config.get('absolute_kappa_defaults', {}).get('h')
                if not isinstance(raw_h, dict) or set(raw_h) != {'numerator', 'denominator'}:
                    raise RequestError(method + '.delta_over_h_values', 'fixed base-time h is unavailable')
                h = Fraction(raw_h['numerator'], raw_h['denominator'])
                absolute = [h * Fraction(width) for width in widths]
                spec.setdefault('search_overrides', {}).setdefault('requested_widths', {})[method] = [
                    dict(numerator=v.numerator, denominator=v.denominator) for v in absolute]
                override.setdefault('generation', {})['delta_over_h_values'] = widths
        if 'delta_pf' in values:
            if not isinstance(values['delta_pf'],list) or not values['delta_pf']:raise RequestError(method+'.delta_pf','a nonempty typed width list is required')
            if current['family'] != 'PR' or config.get('entry', 'method') != 'schedule':
                raise RequestError(method + '.delta_pf', 'absolute requested widths require an adopted requested-width schedule entry')
            nums = [number(x, method + '.delta_pf') for x in values['delta_pf']]
            width_settings = spec.setdefault('search_overrides', {})
            width_settings['requested_widths'] = dict(width_settings.get('requested_widths') or {},
                                                       **{method: [x[0] for x in nums]})
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
            single_df = physical_schedule and method == 'PR_DF' and (
                methods == ['PR_DF'] or config.get('combined_width_search'))
            if (config.get('entry') != 'neighbourhood' and not physical_schedule) or (methods != ['PR_PAULI'] and not combined and not single_df):
                raise RequestError(method + '.kappa.absolute', 'absolute min/max needs a supported PR schedule entry')
            discrete = None
            if 'values' in k:
                if 'min' in k or 'max' in k:
                    raise RequestError(method + '.kappa', 'choose exact values or inclusive bounds')
                if not physical_schedule:
                    raise RequestError(method + '.kappa.values', 'discrete kappa requires the physical schedule entry')
                discrete = exact_values(k['values'], method + '.kappa.values')
                a, b = Fraction(discrete[0]), Fraction(discrete[-1])
            else:
                a, b = Fraction(k['min']), Fraction(k['max'])
            if not 0 < a <= b: raise RequestError(method + '.kappa', 'invalid positive absolute bounds')
            if physical_schedule:
                if combined or single_df:
                    wide=spec.setdefault('search_overrides',{}).setdefault('schedule_wide_by_method',{})
                    wide[method]=dict(spec['search_overrides'].get('schedule_wide',{}),kappa_bounds=[str(a),str(b)],max_kappa=str(b))
                else:
                    spec.setdefault('search_overrides', {}).setdefault('schedule_wide', {})['kappa_bounds'] = [str(a), str(b)]
                    spec['search_overrides']['schedule_wide']['max_kappa'] = str(b)
                override['absolute_kappa_bounds'] = [str(a), str(b)]
                if discrete is not None:
                    override['absolute_kappa_values'] = discrete
                    target = (spec['search_overrides']['schedule_wide_by_method'][method]
                              if combined or single_df else spec['search_overrides']['schedule_wide'])
                    target['kappa_values'] = discrete
                spec['search_overrides']['kappa_neighbourhood'] = dict(outer_planner='SCHEDULE_WIDE_V1')
            else:
                spec.setdefault('search_overrides', {}).setdefault('kappa_neighbourhood', {})['bounds'] = [str(a), str(b)]
            spec['kappa_families'] = override.get('L_D_values', spec.get('kappa_families'))
            observed['kappa'] = (dict(mode=mode, values=discrete) if discrete is not None else
                                 dict(mode=mode, min=str(a), max=str(b)))
        canonical['per_method'][method] = observed
    shots = scope.get('shots', 'auto')
    if config.get('scaled_shots_default') is True or 'scaled_shots' in scope:
        from . import rpe_scaled_shots as scaled
        if shots != 'auto':
            raise RequestError('scope.shots', 'scaled_shots determines the whole allocation')
        if Fraction(problem['accuracy']['epsilon_E_decimal']) != Fraction(scaled.EPSILON):
            raise RequestError('accuracy.epsilon_E_decimal', 'scaled search requires the saved R08 exact threshold')
        try:
            policy = scaled.settings(scope.get('scaled_shots'))
        except (TypeError, ValueError) as error:
            raise RequestError('scope.scaled_shots', str(error)) from error
        canonical['scaled_shots'] = policy
        spec.setdefault('search_overrides', {}).update(rule=scaled.RULE, scaled_shots=policy)
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
        if config.get('work_conserving_search') is True:
            spec.setdefault('search_overrides', {})['work_conserving_search'] = True
    if request['action'] in ('start','prepare'):
        spec['science_approved'] = True
    if 'pbs_walltime_overrides' in scope:
        canonical['pbs_walltime_overrides'] = overrides
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
    if result['search'].get('scaled_shots'):
        from .rpe_scaled_shots import allocation_menu
        policy = result['search']['scaled_shots']
        result['scaled_shot_allocations'] = [
            dict(M=M, allocations=[dict(N_X=list(vector[::2]), N_Y=list(vector[1::2]), aliases=aliases)
                                  for vector, aliases in allocation_menu(M, policy).items()])
            for M in policy['M']]
    preparation = (spec.get('executor', {}).get('distributed') or {}).get('preparation')
    if preparation is not None:
        result['preparation_budgets'] = {stage: preparation.get('budgets', {}).get(stage, 0)
                                         for stage in ('signal', 'compile', 'audit')}
    if spec.get('live_ready_refill') is True:
        result['live_ready_refill'] = True
    if spec.get('search_overrides', {}).get('work_conserving_search') is True:
        result['work_conserving_search'] = True
    execution = spec.get('executor', {})
    if 'execution_hosts' in canonical:
        result['execution_hosts'] = canonical['execution_hosts']
    worker_limits = (execution.get('distributed') or {}).get('worker_storage_limits_bytes')
    if worker_limits:
        result['worker_storage_limits_bytes'] = worker_limits
    if set(spec['methods']) & {'PR_PAULI', 'PR_DF'} and any(
            execution.get(key) is True for key in ('compact_q3_evidence', 'parallel_dp_residual')):
        result['reviewed_pr_q3'] = {key: execution[key] for key in
            ('compact_q3_evidence', 'parallel_dp_residual', 'prefix_cache_max_bytes',
             'prefix_checkpoint_rounds', 'prefix_selective_checkpoint_rounds',
             'prefix_selective_window_seconds', 'profile_dp_residual') if key in execution}
    completion = spec.get('candidate_completion') or {}
    if 'PR_DF' in spec['methods'] and 'parallel_df_price_axes' in completion:
        result['parallel_df_price_axes'] = completion['parallel_df_price_axes']
    if 'PR_DF' in spec['methods'] and 'df_signal_source' in completion:
        result['df_signal_source'] = completion['df_signal_source']
    pbs = (execution.get('distributed') or {}).get('pbs') or {}
    if pbs.get('walltime_policy'):
        rules = [row for row in pbs['q3_task_seconds'] if row['method'] in spec['methods']]
        result['pbs_walltime'] = dict(policy=pbs['walltime_policy'], q3=rules,
            preparation=pbs['preparation_task_seconds'], drain_seconds=pbs.get('drain_seconds', 30),
            unconfigured_q3_methods={method: 'PBS_WALLTIME_ESTIMATE_UNAVAILABLE'
                for method in spec['methods'] if not any(
                    row['method'] == method and row['seconds'] > 0 for row in rules)})
        if 'allocation_sizing' in pbs:
            result['pbs_walltime']['allocation_sizing'] = deepcopy(pbs['allocation_sizing'])
    return result


def local0_progress(applied, spec, run=None):
    """Report queue acceptance separately from observed body execution."""
    effective = {key: applied[key] for key in ('preparation_budgets', 'max_new_q3_per_method',
                                               'pbs_walltime') if key in applied}
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
        worker_path = run/'distributed/LOCAL_WORKER.json'
        local_worker = load(worker_path) if worker_path.is_file() else {}
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True, timeout=1) as db:
            rows = db.execute('''SELECT job_id,method,request,state,adopted_epoch,detail,
                host_id,attempt_id,request_sha256 FROM jobs ORDER BY rowid''').fetchall()
        stages = {}
        def body_events(directory):
            events = set()
            for name in ('JOB_STAGES.jsonl','Q3_BODY_TIMING.jsonl'):
                path = directory/name
                if path.is_file():
                    for line in path.read_bytes().splitlines(keepends=True):
                        if line.endswith(b'\n'):events.add(json.loads(line).get('event'))
            return events

        for job, method, raw, state, adopted, detail, host, attempt, checksum in rows:
            request = json.loads(raw, parse_int=str)
            prep = request.get('preparation', {})
            stage = prep.get('stage', 'q3')
            count = stages.setdefault(stage, dict(stage=stage, queue_accepted=0, execution_started=0,
                execution_completed=0, execution_unit='evaluation' if stage == 'q3' else 'stage_task',
                active_receipts_pending=0))
            count['queue_accepted'] += 1
            directory = run/'jobs'/method/job
            receipt = 'Q3_BODY' if stage == 'q3' else 'PREPARATION_BODY'
            events = body_events(directory)
            if (receipt+'_START' not in events and state in ('ASSIGNED','STARTING','RUNNING','COMPLETED_PENDING_TRANSFER')
                    and attempt and host==local_worker.get('host_id') and local_worker.get('work_dir')):
                live=Path(local_worker['work_dir'])/job/attempt
                assignment=live/'ASSIGNMENT.json'
                if assignment.is_file():
                    owner=load(assignment)
                    if (owner.get('job_id'),owner.get('attempt_id'),owner.get('request_sha256'))==(job,attempt,checksum):
                        events.update(body_events(live))
            native_start = receipt+'_START' in events or receipt+'_END' in events
            native_end = receipt+'_END' in events
            # Older routes may have final receipts without the timing markers.
            terminal = (directory/'PREPARATION_RECEIPT.json').is_file() if prep else adopted is not None
            count['execution_started'] += int(native_start or terminal)
            count['execution_completed'] += int(native_end or terminal)
            active = state in ('STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER') or (state=='ASSIGNED' and native_start)
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
    restores_preparation_archives_on_resume = True

    def __init__(self, config, *, preflight=None, launch=None, resume_ast_failure=None):
        self.config = deepcopy(config); self.state = Path(config['state_directory'])
        self.preflight = preflight or self.normal_preflight; self.launch = launch or self.normal_launch
        self.resume_ast_failure = resume_ast_failure

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
        scope = request.get('scope')
        if isinstance(scope, dict) and scope.get('system') in ('H6', 'H8') and self.config.get('readiness_metadata'):
            keys(scope, {'system', 'methods'}, 'scope')
            system = scope['system']
            if request['action'] != 'preview':
                raise RequestError('scope.system', system + ' is available for metadata Preview only')
            if system == 'H6' and scope.get('methods', ['FULL_PF']) != ['FULL_PF']:
                raise RequestError('scope.methods', 'the saved H6 preview covers FULL_PF only')
            evidence = self.config['readiness_metadata'][system]
            readiness = load(evidence['path']) if system == 'H6' else {'status': 'IN_PREPARATION'}
            return dict(status='PREVIEW', started=False, release=self.config.get('release'),
                        controller_source=self.config['code_source'],
                        applied=dict(system=system, methods=['FULL_PF'] if system == 'H6' else [],
                                     readiness=readiness, source=evidence['source']))
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
                if request['action'] == 'status':
                    result = self.result(run, 'STATUS')
                    maintenance = self.state/'maintenance'/(run.name+'.json')
                    if maintenance.is_file():
                        result['central_maintenance'] = load(maintenance)
                    return result
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
        recovered_stop = None
        if completed and completed.get('returncode') != 0 and not user_interrupted:
            stop_path = run/'KAPPA_STATE.json'
            stop = (load(stop_path).get('stop') or {}).get('reason') if stop_path.exists() else None
            if request['action'] != 'resume' or not self.resume_ast_failure:
                raise RequestError('run', 'failed invocation requires reconciliation: '+str(stop or completed.get('returncode')))
            recovered_stop = reconcile_ast_failure(run, state, completed, self.resume_ast_failure)
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
            elif not user_interrupted and reasons and reasons <= {'NO_IMPROVING_PROPOSAL','NO_FEASIBLE_POINT_IN_RANGE','Q3_BUDGET_EXHAUSTED','STEP_LIMIT','NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET','NO_CANDIDATE_LEFT_IN_RANGE','SCHEDULE_WIDE_RULE_EXHAUSTED','SCALED_SHOT_GRID_COMPLETE'}:
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
        archive_recovery = None
        if request['action'] == 'resume' and not getattr(self.preflight, 'restores_preparation_archives', False):
            from .rpe_preparation_retention import restore_resume_archives
            try:
                archive_recovery = restore_resume_archives(self.state, run)
            except (OSError, ValueError, KeyError) as error:
                raise RequestError('preparation_archives', str(error)) from error
        preflight_started = time.monotonic()
        report = dict(self.preflight(run, problem, spec),
                      elapsed_seconds=time.monotonic() - preflight_started)
        if archive_recovery and any(archive_recovery.values()):
            report['preparation_archive_recovery'] = archive_recovery
        write(run/'DUPLICATE_PREFLIGHT.json', report)
        if report.get('status') != 'PASS': raise RequestError('preflight', '§10.5 evidence is incomplete/conflicting; see DUPLICATE_PREFLIGHT.json')
        immutable(run/'ROUND.json', spec)
        if recovered_stop:
            immutable(run/'stop-history'/('ast-'+self.resume_ast_failure+'.json'), recovered_stop)
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
        preflight_path = run/'DUPLICATE_PREFLIGHT.json'
        preflight = load(preflight_path) if preflight_path.exists() else None
        if preflight is not None and isinstance(preflight.get('refused'), list):
            refused = preflight['refused']
            preflight.update(report_path=str(preflight_path),
                             refused_count=len(refused),
                             refused_by_reason=dict(Counter(row.get('reason') for row in refused).most_common(10)),
                             refused=refused[:16])
        result = dict(run_id=a['run_id'],run_directory=str(run),status=status,applied=a['applied'],started=(run/'LAUNCH.json').exists(),
                    computation_source=self.config['computation_source'],release=self.config.get('release'),controller_source=self.config['code_source'],
                    runtime=self.runtime_status(run),progress=progress,
                    preflight=preflight)
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
            if phase == 'COMPLETE' and (state.get('stop') or {}).get('reason') == 'SCOPE_HELD_Q3':
                value['phase'] = 'RECONCILIATION_REQUIRED'
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
        from .rpe_search_continuation import (preparation_rejected_before_science, reconciled_prior_runs,
                                              verify_pbs_prelaunch_rejections, verify_prior_q3_holds,
                                              verify_retired_preparation_attempts)
        prior_holds = verify_prior_q3_holds(root,run,problem,spec)
        retired_preparation = verify_retired_preparation_attempts(run)
        pbs_rejections = verify_pbs_prelaunch_rejections(run, spec.get('executor',{}).get('prior_run_dirs',[]))
        reconciled = reconciled_prior_runs(spec)
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
        ex.prior_q3_holds = list(prior_holds.values())
        loaded, refused = {}, []
        missing=[str(p) for p in spec.get('executor',{}).get('prior_run_dirs',[]) if not Path(p).is_dir()]
        in_progress=[]
        rejected_before_delivery=[]
        # An unresolved candidate has no configuration identity. It must
        # not match unrelated prior attempts that also omit that identity.
        ids = {c['configuration_id'] for c in context['candidates']
               if c['method'] in spec['methods'] and c.get('configuration_id')}
        for directory in spec.get('executor',{}).get('prior_run_dirs',[]):
            prior_root=Path(directory)
            if (prior_root/'APPLIED_SCOPE.json').exists() and (prior_root/'LAUNCH.json').exists():
                launch=load(prior_root/'LAUNCH.json');exit=load(prior_root/'EXIT.json') if (prior_root/'EXIT.json').exists() else {}
                if launch.get('invocation_id')!=exit.get('invocation_id') and prior_root.resolve() not in reconciled:
                    previous=load(prior_root/'APPLIED_SCOPE.json')
                    if set(previous['applied']['methods']) & set(spec['methods']):in_progress.append(str(prior_root))
            for path in Path(directory).glob('jobs/*/*/REQUEST.json'):
                job=path.parent
                if (job/'OUTCOME.json').exists():continue
                if job.resolve() in retired_preparation:continue
                if (job/'LAUNCH.json').exists() or (job/'STARTING.json').exists():
                    if load(path).get('configuration_id') in ids and str(job.resolve()) not in prior_holds:
                        if job.resolve() in pbs_rejections or preparation_rejected_before_science(job):
                            rejected_before_delivery.append(str(job))
                        else:
                            in_progress.append(str(job))
        for method in spec['methods']:
            known, rejected = ex.for_method(method).prior_records(context['candidates'])
            loaded[method]=len(known);refused.extend(dict(method=method,**r) for r in rejected)
        # A different configuration/route is excluded, never accepted as a
        # match. Missing/corrupt records for a same-key candidate need review.
        holds=[r for r in refused if r['reason'] not in ('CONFIGURATION_NOT_IN_THIS_ROUND',
            'DIFFERENT_JOB_KEY_OR_ROUTE','PRIOR_Q3_PERSISTENCE_HOLD')]
        return dict(status='HOLD' if holds or missing or in_progress else 'PASS',section='10.5',reused=loaded,refused=refused,missing_prior_run_dirs=missing,owned_unresolved_prior_jobs=in_progress,
            prior_q3_holds=list(prior_holds.values()),retired_preparation_holds=sorted(str(path) for path in retired_preparation),
            prior_preparation_rejected_before_science=sorted(rejected_before_delivery),
            reconciled_prior_runs={str(path): held for path, held in reconciled.items()},
            checked_prior_run_dirs=spec.get('executor',{}).get('prior_run_dirs',[]),
            normal_ingest='Executor.prior_records/ingest',scientific_calls=0)

    def normal_launch(self, run, problem, spec, *, schedule_resume=None):
        invocation=uuid.uuid4().hex
        command=[self.config['python'],'-B',str(Path(self.config['code_root'])/'scripts/run_rpe_local0_search.py'),
                 '--config',str(run/'CONTROLLER_CONFIG.json') if (run/'CONTROLLER_CONFIG.json').exists() else self.config['config_path'],'--execute-run',str(run),'--invocation',invocation]
        if schedule_resume is not None:
            schedule_driver(self.config, run, schedule_resume)
            command[2] = str(Path(schedule_resume['root'])/'scripts/run_rpe_local0_search.py')
            command.extend(['--schedule-resume', json.dumps(schedule_resume)])
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


SCHEDULE_PREPARATION_MODULES = (
    'rpe_preparation_artifacts', 'rpe_preparation_bridge', 'rpe_completion_acceptance',
)


def reconcile_ast_failure(run, launch, ended, invocation):
    """Allow the existing run to continue after its stopped AST reader failed."""
    import fcntl
    import re
    import sqlite3

    def hold(reason):
        raise RequestError('run', 'AST failure reconciliation: ' + reason)

    if (not re.fullmatch(r'[0-9a-f]{32}', invocation) or not launch
            or launch.get('invocation_id') != invocation or ended.get('invocation_id') != invocation
            or ended.get('returncode') != 1 or (run/'SEARCH_TERMINATION.json').exists()):
        hold('the selected failed invocation or search state differs')
    history = run/'invocations'/invocation
    try:
        stop = load(run/'KAPPA_STATE.json')['stop']
        driver = load(run/'NORMAL_DRIVER.json')
        relay = load(run/'distributed/relay/STATUS.json')
        if (load(history/'LAUNCH.json') != launch or load(history/'EXIT.json') != ended
                or driver['invocation_id'] != invocation or relay['stopped'] is not True):
            hold('the driver and relay exit records do not agree')
        with (run/'RUNTIME_ERRORS.jsonl').open() as log:
            fault = next((row for row in reversed([json.loads(line) for line in log])
                          if row.get('phase') == 'DRIVER'), {})
        if (stop.get('reason') != 'RUNTIME_EXCEPTION' or stop.get('error_type') != 'SystemError'
                or not stop.get('message', '').startswith('AST constructor recursion depth mismatch')
                or fault.get('error_type') != 'SystemError' or fault.get('message') != stop['message']
                or 'pf_optimizer_connection.py' not in fault.get('traceback', '')):
            hold('the failure is outside the fixed PF metadata AST path')
        owners = [launch, driver, relay]
        local = run/'distributed/LOCAL_DAEMON.json'
        if local.is_file():
            owners.append(load(local))
        if any(type(owner.get('pid')) is not int or owner['pid'] <= 0
               or Path('/proc', str(owner['pid'])).exists() for owner in owners):
            hold('a controller or local worker still has a live process')
        allocations = load(run/'distributed/pbs/STATE.json')['allocations'].values()
        if any(a['status'] not in ('F', 'X', 'FINISHED_SAVED_WORKER', 'SUBMIT_REJECTED') for a in allocations):
            hold('a PBS allocation is still open')
        with (run/'FAST_DRIVER.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            queue = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
            with sqlite3.connect(queue, uri=True) as db:
                rows = db.execute("SELECT state, attempt_id, detail FROM jobs WHERE state NOT IN ('ACCEPTED','CANCELLED')")
                for state, attempt, detail in rows:
                    saved = json.loads(detail)
                    if (state != 'UNKNOWN' or not attempt or saved.get('automatic_retry') is not False
                            or saved.get('resource_released') is not True):
                        hold('a queue entry still has unresolved execution or recovery')
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        hold(str(error))
    return stop


def schedule_driver(config, run, selection):
    """Resolve the operator-selected scheduler for this exact pinned run."""
    scope = load(run/'APPLIED_SCOPE.json')
    if (selection['run_id'] != scope['run_id'] or selection['base_source'] != config['code_source']
            or scope['entry'] != 'schedule' or config.get('artificial_controller_directory')):
        raise RequestError('schedule_resume', 'Scheduler selection does not match the fixed run')
    driver = Path(selection['root']).resolve()/'scripts/run_rpe_schedule_search.py'
    if (not driver.is_file()
            or hashlib.sha256(driver.read_bytes()).hexdigest() != selection['driver_sha256']):
        raise RequestError('schedule_resume', 'Scheduler differs from its selected deployment')
    if 'pbs_manager_sha256' in selection:
        manager = driver.parents[1]/'src/trottertracks/rpe_pbs.py'
        if (not manager.is_file()
                or hashlib.sha256(manager.read_bytes()).hexdigest() != selection['pbs_manager_sha256']):
            raise RequestError('schedule_resume', 'PBS manager differs from its selected deployment')
    if 'worker_trial' in selection:
        optimizer = driver.parents[1]/'src/trottertracks/rpe_optimizer_concurrent.py'
        if (not optimizer.is_file()
                or hashlib.sha256(optimizer.read_bytes()).hexdigest() != selection['worker_trial']['optimizer_sha256']):
            raise RequestError('schedule_resume', 'Resource trial optimizer differs from its selected deployment')
    preparation = selection.get('preparation_sha256')
    if preparation is not None:
        if set(preparation) != set(SCHEDULE_PREPARATION_MODULES):
            raise RequestError('schedule_resume', 'Preparation operations selection is incomplete')
        for name in SCHEDULE_PREPARATION_MODULES:
            module = driver.parents[1]/'src/trottertracks'/(name+'.py')
            if not module.is_file() or hashlib.sha256(module.read_bytes()).hexdigest() != preparation[name]:
                raise RequestError('schedule_resume', 'Preparation operations differ from their selected deployment: '+name)
    return driver


def execute_run(config, run, invocation, *, schedule_resume=None):
    run=Path(run);scope=load(run/'APPLIED_SCOPE.json');spec=load(run/'ROUND.json')
    root=Path(config['code_root'])
    implementation=fixed_implementation(config)
    preflight = load(run/'DUPLICATE_PREFLIGHT.json')
    if preflight['status']!='PASS':raise RequestError('preflight','no matching successful §10.5 preflight')
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
    if spec.get('reconciled_prior_runs'):
        from .rpe_search_continuation import reconciled_prior_runs
        reconciled = {str(path): held for path, held in reconciled_prior_runs(spec).items()}
        if reconciled != preflight.get('reconciled_prior_runs'):
            raise RequestError('continuation', 'prior run reconciliation changed before launch')
    if spec.get('executor', {}).get('prior_q3_holds'):
        from .rpe_search_continuation import verify_prior_q3_holds
        holds = verify_prior_q3_holds(root,run,load(run/'PROBLEM.json'),spec)
        if list(holds.values()) != preflight.get('prior_q3_holds'):
            raise RequestError('continuation','prior Q3 holds changed before launch')
    command=[config['python'],'-B',str(Path(config['code_root'])/'scripts'/ENTRIES[scope['entry']]),
        '--problem',str(run/'PROBLEM.json'),'--round',str(run/'ROUND.json'),'--run-dir',str(run)]
    if schedule_resume is not None:
        driver = schedule_driver(config, run, schedule_resume)
        # Load the selected transport and result-acceptance modules before the
        # optimizer. Scientific providers and all other imports stay pinned.
        manager = str(driver.parents[1]/'src/trottertracks/rpe_pbs.py') if 'pbs_manager_sha256' in schedule_resume else ''
        trial = schedule_resume.get('worker_trial')
        optimizer = str(driver.parents[1]/'src/trottertracks/rpe_optimizer_concurrent.py') if trial else ''
        preparation = schedule_resume.get('preparation_sha256')
        runtime_trial = trial
        resumed_ast = schedule_resume.get('resume_ast_failure')
        if resumed_ast and load(run/'EXIT.json').get('invocation_id') == resumed_ast:
            runtime_trial = dict(trial or {}, resumed_ast_failure=resumed_ast)
        operations = [(name, str(driver.parents[1]/'src/trottertracks'/(name+'.py')))
                      for name in SCHEDULE_PREPARATION_MODULES] if preparation is not None else []
        bootstrap = ('import importlib.util,json,runpy,sys\n'
                     'root,driver,manager,optimizer,preparation,trial=sys.argv[1:7]\n'
                     'sys.argv=sys.argv[6:]\n'
                     'sys.path.insert(0,root+"/src")\n'
                     'for name,path in [*json.loads(preparation),("rpe_pbs",manager),("rpe_optimizer_concurrent",optimizer)]:\n'
                     '    if path:\n'
                     '        spec=importlib.util.spec_from_file_location("trottertracks."+name,path)\n'
                     '        module=importlib.util.module_from_spec(spec)\n'
                     '        sys.modules[spec.name]=module\n'
                     '        spec.loader.exec_module(module)\n'
                     'runpy.run_path(driver,run_name="__main__",init_globals={'
                     '"FIXED_RUN_CODE_ROOT":root,"FIXED_RUN_WORKER_TRIAL":json.loads(trial)})')
        command[2:3] = ['-c', bootstrap, str(root), str(driver), manager, optimizer,
                        json.dumps(operations), json.dumps(runtime_trial)]
        implementation['schedule_source'] = schedule_resume['source']
        implementation['schedule_entry'] = str(driver)
        implementation['schedule_sha256'] = schedule_resume['driver_sha256']
        if preparation is not None:
            implementation['preparation_sha256'] = preparation
        if manager:
            implementation['pbs_manager_entry'] = manager
            implementation['pbs_manager_sha256'] = schedule_resume['pbs_manager_sha256']
        if trial:
            implementation['worker_trial'] = trial
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
