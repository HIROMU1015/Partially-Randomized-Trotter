"""Local0 request -> immutable server2 scope -> existing optimizer entry.

No optimizer or scientific adapter is implemented here. Saved Q3 acceptance is
delegated to Executor.prior_records/ingest, including route and identity checks.
"""
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
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
    """Select the adopted physical schedule entry for bounded or discrete PR scopes."""
    scope = request.get('scope') or {}
    if not isinstance(scope, dict):
        return config
    system = scope.get('system')
    if isinstance(system, str) and (hchain := config.get('hchain_profiles', {}).get(system)):
        selected = deepcopy(config)
        selected.pop('hchain_profiles')
        selected.update(hchain)
        return selected
    per_method = scope.get('per_method') or {}
    if not isinstance(per_method, dict):
        return config
    methods = scope.get('methods', ['PR_PAULI'])
    if not isinstance(methods, list):
        return config
    absolute = [m for m in methods if m in ('PR_PAULI', 'PR_DF')
                and isinstance(per_method.get(m), dict) and isinstance(per_method[m].get('kappa'), dict)
                and per_method[m]['kappa'].get('mode') == 'absolute']
    route = config.get('absolute_kappa_route')
    if route and absolute:
        if len(methods) > 1 and not config.get('combined_width_search'):
            raise RequestError('scope.methods', 'combined absolute-kappa search requires a combined schedule entry')
        selected = deepcopy(config)
        selected.pop('absolute_kappa_route')
        selected.update(route)
        selected['absolute_kappa_schedule'] = True
        return selected
    return config

def resolve(config, request):
    """Translate only settings that the selected normal entry actually consumes."""
    if config.get('hchain'):
        from .rpe_hchain_scope import resolve as resolve_hchain
        return resolve_hchain(config, request)
    keys(request, {'scope', 'instruction', 'action', 'release', 'run_id'}, 'request')
    scope = deepcopy(request.get('scope') or {})
    keys(scope, {'system', 'methods', 'per_method', 'shots', 'stop', 'preparation_budgets', 'scaled_shots',
                 'pbs_walltime_overrides', 'execution_hosts', 'exploration_order'}, 'scope')
    problem = load(config['problem']); spec = load(config['round'])
    if config.get('release_worker_policy'):
        from .rpe_release_registry import common_worker_policy
        if config.get('entry') != 'schedule':
            raise RequestError('release_worker_policy','common worker policy requires schedule entry')
        spec['release_worker_policy'] = common_worker_policy(config['release_worker_policy'])
        # Input/route preflight consumes the same fixed mapper selection as
        # the driver, before any worker or independent package is started.
        if mapper := spec['release_worker_policy'].get('df_max_width'):
            completion = spec['candidate_completion']
            completion['df_max_width_source'] = mapper['source']
            completion['df_max_width_code_root'] = config['code_root']
            completion.setdefault('saved_df_descriptors', {})['0'] = mapper['endpoint_descriptor']
            if mapper.get('price_helper_source'):
                completion['df_max_width_price_helper'] = mapper['price_helper_source']
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
            lo = 1 if key == 'L_D' and not (method == 'PR_DF' and config.get('entry') == 'schedule'
                 and 'delta_over_h_values' in values) else 0
            allowed = integers(values[key], method + '.' + key, lo, fragment_count if method=='FULL_PF' else fragment_count-1)
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
    if 'exploration_order' in scope:
        if scope['exploration_order'] != 'coarse_then_intermediate' or config.get('entry') != 'schedule':
            raise RequestError('scope.exploration_order', 'use coarse_then_intermediate with the schedule entry')
        from .rpe_schedule_search import coarse_axes
        axes = {}
        for method, selected in canonical['per_method'].items():
            coordinates = {k: selected[k] for k in ('p', 'L_D', 'delta_over_h_values') if k in selected}
            if 'delta_over_h_values' in coordinates:
                widths = coordinates.pop('delta_over_h_values')
                if method.startswith('PR_'):
                    coordinates['delta_pf'] = spec['search_overrides']['requested_widths'][method]
                else:
                    coordinates['delta_over_h'] = widths
            if selected.get('kappa', {}).get('values'):
                coordinates['kappa'] = selected['kappa']['values']
            axes[method] = coordinates
        canonical['exploration_order'] = scope['exploration_order']
        spec.setdefault('search_overrides', {})['coarse_axes'] = coarse_axes(axes)
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
        if problem.get('problem', {}).get('system') == 'H4':
            spec['search_overrides']['exploratory_cost'] = True
            problem['search']['exploratory_cost'] = True
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
    distributed = spec.get('executor', {}).get('distributed')
    if (distributed is not None and spec.get('executor', {}).get('scheduler') == 'concurrent'
            and spec.get('search_overrides', {}).get('exploratory_cost') is True):
        spec['search_overrides']['controller_parallel'] = True
        if config.get('entry') == 'schedule':
            spec.update(live_ready_refill=True, overlap_ready_batch=True)
            spec['search_overrides']['work_conserving_search'] = True
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
    if policy := spec.get('release_worker_policy'):
        result['release_worker_policy'] = dict(fields=sorted(policy))
        if policy.get('df_max_width'):
            result['release_worker_policy']['pr_df_mapper_source'] = policy['df_max_width']['source']
        if policy.get('pf_price_recheck_provider'):
            provider=policy['pf_price_recheck_provider']
            result['release_worker_policy']['pf_price_provider_source'] = provider['source']
            result['release_worker_policy']['pf_price_provider_hosts'] = ['local', *sorted(provider.get('host_deployments', {}))]
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
    partition_status=run/'PBS_PARTITION_STATUS.json'
    if partition_status.is_file():
        progress['pbs_independent_search']=load(partition_status)
    scope_path=run/'SCOPE_MAP.json'
    scope = load(scope_path) if scope_path.is_file() else None
    preparation_records = []
    width_families, width_points = {}, []
    def join_scope():
        if scope is not None:
            # During a selected Resume, this gateway is loaded by file path
            # over the old scientific package. Its reporting helper belongs
            # to this same controller release, not to the frozen base package.
            from importlib.util import module_from_spec, spec_from_file_location
            name = 'trottertracks.rpe_scope_progress'
            path = Path(__file__).resolve().with_name('rpe_scope_progress.py')
            module = sys.modules.get(name)
            if module is None or Path(module.__file__).resolve() != path:
                loader = spec_from_file_location(name, path)
                module = module_from_spec(loader)
                loader.loader.exec_module(module)
            progress['scope_map'] = dict(path=str(scope_path), schema=scope['schema'],
                mapper_version=scope.get('mapper_version'), snapshot_epoch=scope_path.stat().st_mtime,
                methods=module.scope_progress(scope, preparation_records, observations,
                                              families=width_families, points=width_points))

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
        # The fixed base package may be older than the active controller.
        # Read the journal using this gateway release's implementation.
        from importlib.util import module_from_spec, spec_from_file_location
        journal_path = Path(__file__).resolve().with_name('rpe_width_checkpoint.py')
        if journal_path.is_file():
            loader = spec_from_file_location('trottertracks.rpe_width_checkpoint', journal_path)
            journal = module_from_spec(loader);loader.loader.exec_module(journal)
            progress['point_checkpoint_seq'] = journal.replay(state, run/'POINT_PROGRESS.jsonl')
        width_families, width_points = state.get('families', {}), state.get('points', [])
        if state.get('recovery_wait'):
            progress['recovery_wait'] = state['recovery_wait']
        pending = {'UNSUPPORTED_MAPPER': 'UNSUPPORTED_GENERATION', 'WAITING_INPUTS': 'PREPARATION_WAITING',
                   'NEEDS_FAMILY_CONFIGURATION_SOURCE': 'MISSING_FIXED_INPUT', 'HELD': 'TECHNICAL_FAILURE',
                   'PRICE_ACTION_UNAVAILABLE': 'TECHNICAL_FAILURE', 'PRICE_NOT_EVALUATED': 'NOT_EVALUATED'}
        for key, point in state.get('families', {}).items():
            if point.get('status') in pending:
                observations[key] = dict(candidate_id=key, method=point['method'],
                                         state=pending[point['status']], reason=point.get('reason'))
        for point in state.get('points', []):
            completion = point.get('completion') or {}
            key = (point.get('configuration_id') or completion.get('configuration_id')
                   or completion.get('key') or point.get('family'))
            status = pending.get(point.get('status'))
            if point.get('status') == 'EVALUATED' and point.get('saved_point_ids'):
                status = 'REUSED_SAVED_RESULT'
            if key and status:
                observations[key] = dict(candidate_id=key, method=point['method'], state=status, reason=point.get('reason'))
                for field in ('price_actions', 'preparation_recovery', 'conflicting_signal_rounds',
                              'failure_class','retry_protected'):
                    if (detail := completion.get(field)):
                        observations[key][field] = detail
    pf_status = run/'PF_PREPARATION_STATUS.json'
    waves = run/'pf_auto/WAVES.json'
    if pf_status.is_file() or waves.is_file():
        pending = {'PREPARATION_WAITING': 'PREPARATION_WAITING', 'PREPARATION_INCOMPLETE': 'TECHNICAL_FAILURE',
                   'PRIOR_OWNER_HOLD': 'TECHNICAL_FAILURE', 'PRICE_EVIDENCE_REQUIRED': 'PREPARATION_WAITING',
                   'PRICE_ACTION_UNAVAILABLE': 'TECHNICAL_FAILURE', 'PRICE_RECHECK_AVAILABLE': 'PREPARATION_WAITING',
                   'PRICE_OWNER_WAITING': 'PREPARATION_WAITING', 'PRICE_NOT_EVALUATED': 'NOT_EVALUATED'}
        pf_records = load(pf_status).get('records', {}) if pf_status.is_file() else {}
        if waves.is_file():
            pf_records.update(load(waves))
        for key, record in pf_records.items():
            preparation_records.append(dict(record, configuration_id=key))
            if record.get('method') in ('FULL_PF', 'TRUNCATED_PF') and record.get('status') in pending:
                observations[key] = dict(candidate_id=key, method=record['method'], state=pending[record['status']],
                                         reason=record.get('reason'), differing_fields=record.get('differing_fields'))
                if record.get('price_axis_actions'):
                    observations[key]['price_actions'] = record['price_axis_actions']
                if record.get('preparation_recovery'):
                    observations[key]['preparation_recovery'] = record['preparation_recovery']
    if scope is not None:
        completion = run/'completion/STATE.json'
        if completion.is_file():
            preparation_records.extend(load(completion).get('records', {}).values())
    shortlist = run/'LIVE_SHORTLIST.json'
    if shortlist.is_file():
        progress['live_shortlist'] = dict(load(shortlist), snapshot_epoch=shortlist.stat().st_mtime)
    method_by_candidate = {row.get('configuration_id'): row.get('method') for row in preparation_records
                           if row.get('configuration_id') and row.get('method')}
    database = run/'distributed/QUEUE.sqlite3'
    if not database.exists():
        join_scope()
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
        q3_horizons = {}
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
                active_receipts_pending=0, unknown_attempts=0, accepted_results=0))
            count['queue_accepted'] += 1
            count['unknown_attempts'] += int(state=='UNKNOWN')
            count['accepted_results'] += int(adopted is not None)
            directory = run/'jobs'/method/job
            receipt = 'Q3_BODY' if stage == 'q3' else 'PREPARATION_BODY'
            terminal = (directory/'PREPARATION_RECEIPT.json').is_file() if prep else adopted is not None
            events = set() if terminal else body_events(directory)
            if (not terminal and receipt+'_START' not in events and state in ('ASSIGNED','STARTING','RUNNING','COMPLETED_PENDING_TRANSFER')
                    and attempt and host==local_worker.get('host_id') and local_worker.get('work_dir')):
                live=Path(local_worker['work_dir'])/job/attempt
                assignment=live/'ASSIGNMENT.json'
                if assignment.is_file():
                    owner=load(assignment)
                    if (owner.get('job_id'),owner.get('attempt_id'),owner.get('request_sha256'))==(job,attempt,checksum):
                        events.update(body_events(live))
            native_start = receipt+'_START' in events or receipt+'_END' in events
            native_end = receipt+'_END' in events
            count['execution_started'] += int(native_start or terminal)
            count['execution_completed'] += int(native_end or terminal)
            active = state in ('STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER') or (state=='ASSIGNED' and native_start)
            count['active_receipts_pending'] += int(active and not (native_start or terminal))
            if prep:
                if str(prep.get('provider', '')).startswith('PF_'):
                    candidate = next(iter(prep.get('outputs', [])), '').rsplit('/', 1)[-1]
                    method = method_by_candidate.get(candidate) or observations.get(candidate, {}).get('method')
                    if method is None:
                        pf_methods = set(applied.get('methods', [])) & {'FULL_PF','TRUNCATED_PF'}
                        method = next(iter(pf_methods)) if len(pf_methods) == 1 else 'PF_PREPARATION'
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
                        code = 'PREPARATION_COMPLETED'
            else:
                candidate = request.get('configuration_id')
                code = 'EVALUATION_COMPLETED' if adopted is not None else 'EVALUATION_RUNNING' if active else 'EVALUATION_WAITING'
            if candidate and state != 'CANCELLED':
                metadata = json.loads(detail)
                if state in ('FAILED', 'UNKNOWN') or (state == 'ACCEPTED' and not prep and adopted is None):
                    code = ('UNKNOWN_ATTEMPT' if state == 'UNKNOWN' else
                            'TECHNICAL_FAILURE' if state == 'FAILED' else 'EVALUATION_WAITING')
                if not prep and isinstance(request.get('N_X'), list):
                    horizon = len(request['N_X'])-1
                    counts = q3_horizons.setdefault(candidate, {}).setdefault(horizon, {})
                    counts[code] = counts.get(code, 0) + 1
                key = (candidate, stage)
                priority = 3 if active else 2 if state in ('QUEUED', 'ASSIGNED') else 1
                if key not in identified or priority >= identified[key][0]:
                    identified[key] = (priority, dict(candidate_id=candidate, method=method, state=code,
                                                     reason=metadata.get('reason'), stage=stage))
        for (candidate, stage), (_, row) in identified.items():
            if stage == 'q3' or candidate not in observations:
                previous = observations.get(candidate, {})
                for field in ('price_actions', 'preparation_recovery', 'conflicting_signal_rounds'):
                    if field in previous:
                        row[field] = previous[field]
                observations[candidate] = row
        for candidate, horizons in q3_horizons.items():
            observations[candidate]['q3_horizons'] = horizons
        progress['stages'] = list(stages.values())
    except (sqlite3.Error, OSError, ValueError) as error:
        progress['receipt_status'] = str(error)
    join_scope()
    progress['candidates'] = list(observations.values())
    return progress


class Gateway:
    restores_preparation_archives_on_resume = True

    def __init__(self, config, *, preflight=None, launch=None, resume_ast_failure=None,
                 resume_dispatch_failure=None, resume_pf_request_failure=None,
                 resume_controller_kill=None):
        self.config = deepcopy(config); self.state = Path(config['state_directory'])
        self.preflight = preflight or self.normal_preflight; self.launch = launch or self.normal_launch
        self.resume_ast_failure = resume_ast_failure
        self.resume_dispatch_failure = resume_dispatch_failure
        self.resume_pf_request_failure = resume_pf_request_failure
        self.resume_controller_kill = resume_controller_kill

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
        if (isinstance(scope, dict) and scope.get('system') in ('H6', 'H8')
                and self.config.get('readiness_metadata') and not self.config.get('hchain')):
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
            if not self.config.get('keep_current_target'):
                write(current, dict(run_id=run_id,run_directory=str(run)))
            if request['action']=='planonly':return self.result(run,self.plan_only(run))
            if request['action']=='prepare':return self.result(run,'PREPARED_NO_EXECUTION_GRANT')
            status=self.start(run, request)
            return self.result(run, status)

    def resume_interrupted(self, run, invocation):
        """Continue an already authorized run through the ordinary launch checks."""
        with locked(self.state):
            target = self.state/'CURRENT_TARGET.json'
            selected_target = load(target) if target.is_file() else {}
            if (selected_target.get('run_id') != run.name or
                    Path(selected_target.get('run_directory', '')).resolve() != run.resolve()):
                raise RequestError('run', 'the interrupted run is no longer the current target')
            if (self.state/'MIGRATED_TO.json').exists() or (run/'STOP_REQUESTED.json').exists():
                raise RequestError('run', 'migration or explicit STOP prevents unattended resume')
            launched, ended = load(run/'LAUNCH.json'), load(run/'EXIT.json')
            if (launched.get('invocation_id') != invocation or ended.get('invocation_id') != invocation
                    or ended.get('returncode') != -signal.SIGKILL):
                raise RequestError('run', 'no matching killed-controller receipt')
            authorization = load(run/'AUTHORIZATION.json')
            scope, spec = load(run/'APPLIED_SCOPE.json'), load(run/'ROUND.requested.json')
            if (scope['entry'] != 'schedule' or not spec.get('executor', {}).get('distributed')
                    or spec.get('science_approved') is not True
                    or authorization.get('scope_signature') != scope['signature']
                    or authorization.get('kind') != 'EXPLICIT_USER_START'):
                raise RequestError('run', 'no resumable authorized distributed schedule')
            self.resume_controller_kill = invocation
            previous = run/'AUTO_RESUME.json'
            if previous.is_file() and load(previous).get('authorized_utc') == authorization.get('authorized_utc'):
                raise RequestError('run', 'one unattended resume already used for this start')
            command = launched.get('command') or []
            selected = (json.loads(command[command.index('--schedule-resume')+1])
                        if '--schedule-resume' in command else None)
            return self.start(run, {'action': 'resume'}, interrupted=dict(
                from_invocation=invocation, authorized_utc=authorization['authorized_utc'],
                schedule_resume=selected))

    def start(self, run, request, *, interrupted=None):
        state = load(run/'LAUNCH.json') if (run/'LAUNCH.json').exists() else None
        completed=load(run/'EXIT.json') if (run/'EXIT.json').exists() else {}
        detached = None
        if state and (not state.get('invocation_id') or completed.get('invocation_id')!=state.get('invocation_id')):
            pid = state.get('pid')
            if pid:
                try:
                    if os.fsencode(run) in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0'):
                        return 'ALREADY_RUNNING'
                except FileNotFoundError:
                    pass
            if request.get('action') == 'resume' and self.resume_controller_kill == state.get('invocation_id'):
                detached = reconcile_controller_detachment(run, state, self.resume_controller_kill)
            elif request.get('action') != 'resume' or not restore_pre_dispatch_launch(run, state, completed):
                raise RequestError('run', 'previous launch has no completion receipt; existing STOP/UNKNOWN is preserved')
            else:
                state = load(run/'LAUNCH.json')
        stop_request = run/'STOP_REQUESTED.json'
        user_interrupted = (request.get('action') == 'resume' and completed.get('returncode') == -signal.SIGINT
                            and stop_request.is_file() and load(stop_request).get('reason') == 'USER_REQUESTED_STOP')
        recovered_stop = None
        recovery_kind, recovery_invocation = next(
            ((kind, invocation) for kind, invocation in (
                ('pf_request', self.resume_pf_request_failure),
                ('dispatch', self.resume_dispatch_failure), ('ast', self.resume_ast_failure),
                ('controller_kill', self.resume_controller_kill))
             if invocation), (None, None))
        if detached is not None:
            recovery_kind, recovered_stop = 'controller_detachment', detached
        elif completed and completed.get('returncode') != 0 and not user_interrupted:
            stop_path = run/'KAPPA_STATE.json'
            stop = (load(stop_path).get('stop') or {}).get('reason') if stop_path.exists() else None
            if request['action'] != 'resume' or not recovery_invocation:
                raise RequestError('run', 'failed invocation requires reconciliation: '+str(stop or completed.get('returncode')))
            if recovery_kind == 'controller_kill':
                recovered_stop = reconcile_controller_kill(run, state, completed, recovery_invocation)
            else:
                recovered_stop = reconcile_runtime_failure(run, state, completed, recovery_invocation, recovery_kind)
        suspended_marker = None
        if (run/'ROUND_RESULT.json').exists():
            outcomes=load(run/'ROUND_RESULT.json')['methods']
            reasons={v['stop_reason'] for v in outcomes.values()}
            holds={'UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD'}
            stop_file = run/'KAPPA_STATE.json'
            stop = (load(stop_file).get('stop') or {}).get('reason') if stop_file.is_file() else None
            unresolved_tasks_only = stop in ('SCOPE_HELD_Q3', 'SCOPE_HELD_INPUTS', 'WAITING_FOR_INPUTS')
            fatal = (reasons & {'UNKNOWN_RESULT'}) if unresolved_tasks_only else (reasons & holds)
            if fatal:raise RequestError('run','existing scientific hold/STOP is preserved: '+','.join(sorted(fatal)))
            settings=load(run/'ROUND.requested.json')
            if settings.get('search_to_rule_stop'):
                final=run/'SEARCH_TERMINATION.json'
                if final.exists() and not user_interrupted:
                    # Older schedule drivers wrote this marker even for a suspended run.
                    if request['action'] != 'resume' or load(final).get('reason') not in (
                            'SCOPE_HELD_INPUTS', 'SCOPE_HELD_Q3', 'MAX_CYCLES'):
                        return 'REUSED_COMPLETED'
                    suspended_marker = final
            elif not user_interrupted and reasons and reasons <= {'NO_IMPROVING_PROPOSAL','NO_FEASIBLE_POINT_IN_RANGE','Q3_BUDGET_EXHAUSTED','STEP_LIMIT','NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET','NO_CANDIDATE_LEFT_IN_RANGE','SCHEDULE_WIDE_RULE_EXHAUSTED','SCALED_SHOT_GRID_COMPLETE','EXPLORATORY_GRID_COMPLETE'}:
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
            immutable(run/'stop-history'/(recovery_kind+'-'+recovery_invocation+'.json'), recovered_stop)
        if suspended_marker is not None:
            history = run/'stop-history'
            history.mkdir(exist_ok=True)
            suspended_marker.replace(history/('legacy-search-termination-'+str(time.time_ns())+'.json'))
        if interrupted is None:
            write(run/'AUTHORIZATION.json', dict(kind='EXPLICIT_USER_START',instruction=request['instruction'],scope_signature=applied['signature'],authorized_utc=time.time()))
        elif stop_request.exists():
            raise RequestError('run', 'explicit STOP prevents unattended resume')
        if interrupted is None and stop_request.exists():
            previous=load(run/'STOP_REQUESTED.json')
            write(run/'stop-history'/(str(time.time_ns())+'.json'),previous)
            (run/'STOP_REQUESTED.json').unlink()
        if interrupted is not None:
            if interrupted['schedule_resume'] is not None:
                schedule_driver(self.config, run, interrupted['schedule_resume'])
            write(run/'AUTO_RESUME.json', dict(interrupted, resumed_epoch=time.time()))
            if interrupted['schedule_resume'] is not None:
                self.normal_launch(run, problem, spec, schedule_resume=interrupted['schedule_resume'])
            else:
                self.launch(run, problem, spec)
        else:
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
            if phase == 'RUNNING' and state.get('recovery_wait'):
                value['phase'] = 'WAITING_FOR_RECOVERY'
                value['recovery_wait'] = state['recovery_wait']
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
        if spec.get('hchain'):
            from run_rpe_hchain_round import preflight_context
            try:
                context = preflight_context(problem, spec)
            except (OSError, KeyError, ValueError, subprocess.SubprocessError) as error:
                raise RequestError('hchain', str(error)) from error
        else:
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
            'DIFFERENT_JOB_KEY_OR_ROUTE','PRIOR_Q3_PERSISTENCE_HOLD',
            'CONFIGURATION_PREPARATION_PENDING')]
        report = dict(status='HOLD' if holds or missing or in_progress else 'PASS',section='10.5',reused=loaded,refused=refused,missing_prior_run_dirs=missing,owned_unresolved_prior_jobs=in_progress,
            prior_q3_holds=list(prior_holds.values()),retired_preparation_holds=sorted(str(path) for path in retired_preparation),
            prior_preparation_rejected_before_science=sorted(rejected_before_delivery),
            reconciled_prior_runs={str(path): held for path, held in reconciled.items()},
            checked_prior_run_dirs=spec.get('executor',{}).get('prior_run_dirs',[]),
            normal_ingest='Executor.prior_records/ingest',scientific_calls=0)
        if (legacy := context.get('legacy')) is not None:
            report['hchain_legacy'] = dict(job=legacy['source_job'], completed=len(legacy['records']),
                                           held=len(legacy['holds']), spent=legacy['spent'])
        return report

    def normal_launch(self, run, problem, spec, *, schedule_resume=None):
        invocation=uuid.uuid4().hex
        cpu_resume = schedule_resume and schedule_resume.get('worker_trial',{}).get('hchain_cpu_policy')
        config_path = self.config['config_path'] if cpu_resume else (
            str(run/'CONTROLLER_CONFIG.json') if (run/'CONTROLLER_CONFIG.json').exists() else self.config['config_path'])
        command=[self.config['python'],'-B',str(Path(self.config['code_root'])/'scripts/run_rpe_local0_search.py'),
                 '--config',config_path,'--execute-run',str(run),'--invocation',invocation]
        if schedule_resume is not None:
            schedule_driver(self.config, run, schedule_resume)
            command[2] = str(Path(schedule_resume['root'])/'scripts/run_rpe_local0_search.py')
            command.extend(['--schedule-resume', json.dumps(schedule_resume)])
        # Older fixed launchers may not implement the supervisor switch.
        if (Path(command[2]).is_file() and "'--supervise-run'" in Path(command[2]).read_text()):
            command.append('--supervise-run')
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


def local_worker_for_reattachment(run):
    """Return the live worker entry bound to this run's fixed endpoint."""
    marker = run/'distributed/LOCAL_DAEMON.json'
    if not marker.is_file():
        return None
    owner = load(marker)
    pid = owner['pid']
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid local worker PID')
    try:
        argv = Path('/proc', str(pid), 'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
    except FileNotFoundError:
        return None
    if argv == [b'']:
        return None
    distributed = load(run/'ROUND.json')['executor']['distributed']
    port = distributed.get('port', 8763)
    profile = (run/'distributed/LOCAL_WORKER.json').resolve()
    endpoint = load(run/'distributed/ENDPOINT.json')
    url = 'http://127.0.0.1:'+str(port)
    if (not port or Path(owner['profile']).resolve() != profile or endpoint['url'] != url
            or argv[1:2] != [b'-B'] or len(argv) < 7
            or argv[3:7] != [b'--profile', os.fsencode(profile), b'--controller', os.fsencode(url)]
            or Path(os.fsdecode(argv[2])).name != 'run_rpe_q3_worker.py'):
        raise ValueError('the live worker does not match the saved profile and fixed endpoint')
    return Path(os.fsdecode(argv[2])).resolve()


def reconcile_runtime_failure(run, launch, ended, invocation, kind):
    """Verify ended ownership after a specifically identified controller error."""
    import ast
    import fcntl
    import re
    import sqlite3

    def hold(reason, *, ownership_wait=False):
        field = 'runtime_ownership' if ownership_wait else 'run'
        raise RequestError(field, kind + ' failure reconciliation: ' + reason)

    if (not re.fullmatch(r'[0-9a-f]{32}', invocation) or not launch
            or launch.get('invocation_id') != invocation or ended.get('invocation_id') != invocation
            or ended.get('returncode') != 1 or (run/'SEARCH_TERMINATION.json').exists()):
        hold('the selected failed invocation or search state differs')
    history = run/'invocations'/invocation
    try:
        scope_path = run/'APPLIED_SCOPE.json'
        if (kind == 'dispatch' and scope_path.is_file() and load(scope_path)['entry'] == 'method'
                and (run/'ROUND.json').is_file() and load(run/'ROUND.json').get('hchain')):
            driver = load(run/'NORMAL_DRIVER.json')
            if (load(history/'LAUNCH.json') != launch or load(history/'EXIT.json') != ended
                    or driver['invocation_id'] != invocation):
                hold('the driver and exit records do not agree')
            failure = reconcile_hchain_pinned_pbs_setup(run, history, driver)
            if failure:
                return failure
            failure = verify_pre_dispatch_failure(run, history, driver, has_relay=False)
            if failure:
                return failure
            failure = reconcile_hchain_preparation_setup(run, history, driver)
            if failure:
                return failure
            hold('the H-chain invocation reached a dispatcher or has an unresolved owner')
        stop = load(run/'KAPPA_STATE.json')['stop']
        driver = load(run/'NORMAL_DRIVER.json')
        relay = load(run/'distributed/relay/STATUS.json')
        if (load(history/'LAUNCH.json') != launch or load(history/'EXIT.json') != ended
                or driver['invocation_id'] != invocation):
            hold('the driver and exit records do not agree')
        if relay['stopped'] is not True:
            hold('the relay has not stopped', ownership_wait=True)
        # An operational overlay can fail during imports, before it creates an
        # executor. The earlier run's owned attempts can still be reattached.
        if kind == 'dispatch' and (failure := verify_pre_dispatch_failure(run, history, driver)):
            return failure
        with (run/'RUNTIME_ERRORS.jsonl').open() as log:
            fault = next((row for row in reversed([json.loads(line) for line in log])
                          if row.get('phase') == 'DRIVER'), {})
        message = stop.get('message', '')
        trace = fault.get('traceback', '')
        if (stop.get('reason') != 'RUNTIME_EXCEPTION' or stop.get('error_type') != fault.get('error_type')
                or message != fault.get('message')):
            hold('the driver failure differs from its saved stop')
        damaged_progress = False
        disk_report_failure = False
        if kind == 'dispatch':
            projected_request = (
                stop['error_type'] == 'KeyError' and re.fullmatch("'[0-9a-f]{64}'", message)
                and 'in prefetch_standby' in trace
                and 'source = configurations[projection[\'source_configuration_id\']]' in trace)
            pinned_history_binding = (
                stop['error_type'] == 'AttributeError'
                and message == "'DistributedExecutor' object has no attribute 'prior_route_fields'"
                and 'rpe_plan_history_cache.py' in trace and 'in _context' in trace)
            damaged_progress = (
                stop['error_type'] == 'JSONDecodeError'
                and 'in _metrics_snapshot' in trace and 'in metrics' in trace
                and 'rpe_optimizer_concurrent.py' in trace and 'e=json.loads(line)' in trace)
            disk_report_failure = (stop['error_type'] == 'OSError'
                and message.startswith('[Errno 28]') and 'in report' in trace
                and 'pickle.dump(' in trace and 'run_rpe_schedule_search.py' in trace
                and shutil.disk_usage(run).free > 0)
            if not (projected_request or pinned_history_binding or damaged_progress or disk_report_failure):
                hold('the failure is outside the recoverable dispatch or metrics paths')
            if (damaged_progress or disk_report_failure) and (run/'STOP_REQUESTED.json').exists():
                hold('user STOP prevents metrics recovery')
        elif kind == 'pf_request':
            old_mismatch = (message == 'PF completion directory belongs to another selected request'
                            and 'pf_optimizer_connection.py' in trace)
            inherited_paths = (message == 'PF selected contract changed'
                               and 'pf_distributed_preparation.py' in trace)
            saved_axis_encoding = (stop['error_type'] == 'TypeError' and message == "unhashable type: 'list'"
                                   and 'pf_distributed_preparation.py' in trace
                                   and 'consumed.update(audited)' in trace)
            incomplete_recovery_index = (stop['error_type'] == 'KeyError' and message == "'configuration_id'"
                                         and 'pf_optimizer_connection.py' in trace
                                         and "record['configuration_id'] == candidates[0]" in trace)
            legacy_write_scope = False
            restored_provider = False
            vanished_pf_owner = False
            saved_price_git_input = False
            pf_local_publication = False
            git_suffix = "' returned non-zero exit status 128."
            if (stop['error_type'] == 'CalledProcessError'
                    and 'pf_distributed_preparation.py' in trace
                    and "['git','fetch','--no-tags','--no-write-fetch-head',str(root),commit]" in trace
                    and message.startswith("Command '") and message.endswith(git_suffix)):
                command = ast.literal_eval(message[len("Command '"):-len(git_suffix)])
                root = run.resolve().parents[1]
                saved_price_git_input = (
                    command[:5] == ['git', 'fetch', '--no-tags', '--no-write-fetch-head', str(root)]
                    and len(command) == 6 and re.fullmatch(r'[0-9a-f]{40}', command[5])
                    and subprocess.run(['git', 'cat-file', '-e', command[5]+'^{commit}'],
                                       cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)
            if (stop['error_type'] == 'CalledProcessError' and 'pf_distributed_preparation.py' in trace
                    and message.startswith("Command '") and message.endswith(git_suffix)):
                command = ast.literal_eval(message[len("Command '"):-len(git_suffix)])
                if (len(command) == 6 and command[:4] == ['git','fetch','--no-tags','--no-write-fetch-head']
                        and command[5] == 'HEAD'):
                    workspace = Path(command[4]).resolve()
                    if (workspace.name == 'provider_workspace' and workspace.is_relative_to(run.resolve()/'pf_auto')
                            and (workspace.parent/'PROVIDER_SOURCE.json').is_file()):
                        registry = 'docs/research/distributed_selected_pf/REGISTRY.json'
                        fixed = load(workspace.parent/'PROVIDER_SOURCE.json')
                        head_registry = subprocess.check_output(['git','show','HEAD:'+registry], cwd=workspace)
                        pf_local_publication = (
                            head_registry == (workspace/registry).read_bytes()
                            and subprocess.run(['git','cat-file','-e',fixed['source']+'^{commit}'],
                                               cwd=workspace, capture_output=True).returncode == 0)
            suffix = "' returned non-zero exit status 1."
            if (stop['error_type'] == 'CalledProcessError' and 'pf_distributed_preparation.py' in trace
                    and "source=local('preflight'" in trace and message.startswith("Command '")
                    and message.endswith(suffix) and 'run_pf_preparation_stage.py' in message):
                command = ast.literal_eval(message[len("Command '"):-len(suffix)])
                arguments = launch.get('command', ())
                selection = (json.loads(arguments[arguments.index('--schedule-resume')+1])
                             if '--schedule-resume' in arguments else {})
                trial = selection.get('worker_trial', {})
                adapter = Path(command[2]).resolve()
                provider = Path(command[command.index('--provider-root')+1]).resolve()
                stage_input = Path(command[command.index('--input')+1]).resolve()
                output = Path(command[command.index('--output')+1]).resolve()
                if (selection.get('run_id') == run.name and trial.get('pf_stage_sha256')
                        and adapter == Path(selection['root']).resolve()/'scripts/run_pf_preparation_stage.py'
                        and hashlib.sha256(adapter.read_bytes()).hexdigest() == trial['pf_stage_sha256']
                        and command[0] == trial.get('pf_controller_runtime', {}).get('interpreter')
                        and output.name == 'preflight.METADATA_RESULT.json' and not output.exists()
                        and stage_input == output.with_name('preflight.METADATA_INPUT.json')
                        and stage_input.is_relative_to(provider/'docs/research/distributed_selected_pf')
                        and provider.name == 'provider_workspace' and provider.is_relative_to(run.resolve())
                        and load(stage_input)['phase'] == 'preflight'
                        and not any(p.name.endswith(('.DISTRIBUTED.json', '.STARTED.json'))
                                    for p in output.parent.rglob('PF_*.json'))):
                    with (run/'DRIVER.log').open('rb') as log:
                        log.seek(max(0, log.seek(0, os.SEEK_END) - 2 * 1024**2))
                        tail = log.read().decode(errors='replace')
                    vanished_pf_owner = any(
                        str(provider/'scripts/pf_price_stages.py') in block and 'in active_price_jobs' in block
                        and 'ProcessLookupError: [Errno 3] No such process' in block
                        for block in tail.split('Traceback (most recent call last):'))
            if (stop['error_type'] == 'CalledProcessError' and 'pf_distributed_preparation.py' in trace
                    and message.startswith("Command '") and message.endswith(suffix)
                    and 'check_pf_preparation_owner.py' in message):
                try:
                    command = ast.literal_eval(message[len("Command '"):-len(suffix)])
                    output = Path(command[command.index('--output') + 1]).resolve()
                except (AttributeError, IndexError, SyntaxError, TypeError, ValueError):
                    hold('invalid PF launcher command in the failure record')
                fixed = load(run/'ROUND.json')['pf_completion']
                pinned = fixed['interpreter']
                arguments = launch.get('command', ())
                prior_worker = (json.loads(arguments[arguments.index('--schedule-resume')+1])['worker_trial']
                                if '--schedule-resume' in arguments else {})
                runtime = prior_worker.get('pf_controller_runtime', {})
                allowed = {pinned}
                if runtime.get('replaces') == pinned and Path(runtime['run_directory']).resolve() == run.resolve():
                    allowed.add(runtime['interpreter'])
                if (command[0] in allowed and Path(command[2]).name == 'check_pf_preparation_owner.py'
                        and output.name == 'ORIGINAL_OWNER_CHECK.json' and output.is_relative_to(run.resolve())):
                    with (run/'DRIVER.log').open('rb') as log:
                        log.seek(max(0, log.seek(0, os.SEEK_END) - 2 * 1024**2))
                        evidence = log.read().decode(errors='replace')
                    legacy_write_scope = command[0] == pinned and any(str(output) in line and 'is not in the subpath of' in line
                                             for line in evidence.splitlines() if line.startswith('ValueError:'))
                    if fixed.get('entry') and '--provider-root' in command and '--request' in command:
                        provider = Path(fixed['entry']).resolve()
                        owner_root = Path(command[command.index('--provider-root')+1]).resolve()
                        owner_request = Path(command[command.index('--request')+1]).resolve()
                        restored_provider = (provider.is_file() and provider.parents[1] == owner_root
                            and owner_request == output.with_name('ORIGINAL_OWNER_REQUEST.json')
                            and owner_request.is_file() and not output.exists()
                            and any(repr(str(provider)) in line for line in evidence.splitlines()
                                    if line.startswith('FileNotFoundError: [Errno 2]'))
                            and subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', 'scripts', 'src'],
                                               cwd=owner_root).returncode == 0)
            if (not ((stop['error_type'] == 'ValueError' and (old_mismatch or inherited_paths))
                     or saved_axis_encoding or incomplete_recovery_index or legacy_write_scope or restored_provider
                     or vanished_pf_owner or saved_price_git_input or pf_local_publication)
                    or 'in complete_selected' not in trace
                    or 'pf_candidate_selection.py' not in trace):
                hold('the failure is outside the PF candidate request binding')
            if ((incomplete_recovery_index or legacy_write_scope or vanished_pf_owner or saved_price_git_input or pf_local_publication)
                    and (run/'STOP_REQUESTED.json').exists()):
                hold('user STOP prevents recovery')
        elif (kind != 'ast' or stop['error_type'] != 'SystemError'
                or not message.startswith('AST constructor recursion depth mismatch')
                or 'pf_optimizer_connection.py' not in trace):
            hold('the failure is outside the fixed PF metadata AST path')
        reattach = damaged_progress or disk_report_failure or (kind == 'pf_request'
            and (incomplete_recovery_index or legacy_write_scope or restored_provider or vanished_pf_owner
                 or saved_price_git_input or pf_local_publication))
        owners = [launch, driver, relay]
        local = run/'distributed/LOCAL_DAEMON.json'
        if reattach:
            local_worker_for_reattachment(run)
        elif local.is_file():
            owners.append(load(local))
        if any(type(owner.get('pid')) is not int or owner['pid'] <= 0
               or Path('/proc', str(owner['pid'])).exists() for owner in owners):
            hold('a controller or local worker still has a live process', ownership_wait=True)
        allocations = load(run/'distributed/pbs/STATE.json')['allocations'].values()
        if not reattach and any(a['status'] not in ('F', 'X', 'FINISHED_SAVED_WORKER', 'SUBMIT_REJECTED')
                                for a in allocations):
            hold('a PBS allocation is still open', ownership_wait=True)
        with ExitStack() as locks:
            for filename in ('FAST_DRIVER.lock', 'distributed/relay/RELAY.lock', 'distributed/pbs/MANAGER.lock'):
                path = run/filename
                if filename == 'FAST_DRIVER.lock' or (reattach and path.exists()):
                    lock = locks.enter_context(path.open('rb'))
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        hold(filename+' still has an owner', ownership_wait=True)
            queue = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
            with sqlite3.connect(queue, uri=True) as db:
                if reattach:
                    verify_reattachable_attempts(run, db)
                    return stop
                excluded = "('ACCEPTED','CANCELLED')"
                if kind == 'dispatch' or (kind == 'pf_request' and (inherited_paths or saved_axis_encoding)):
                    launched = (history/'LAUNCH.json').stat().st_mtime
                    if db.execute("SELECT 1 FROM jobs WHERE assigned_epoch >= ? OR started_epoch >= ?"
                                  " OR (state='QUEUED' AND (host_id IS NOT NULL OR attempt_id IS NOT NULL"
                                  " OR assigned_epoch IS NOT NULL OR started_epoch IS NOT NULL)) LIMIT 1",
                                  (launched, launched)).fetchone():
                        hold('the failed invocation acquired an attempt')
                    excluded = "('ACCEPTED','CANCELLED','QUEUED')"
                rows = db.execute("SELECT state, attempt_id, detail FROM jobs WHERE state NOT IN " + excluded)
                for state, attempt, detail in rows:
                    saved = json.loads(detail)
                    if (state != 'UNKNOWN' or not attempt or saved.get('automatic_retry') is not False
                            or saved.get('resource_released') is not True):
                        hold('a queue entry still has unresolved execution or recovery', ownership_wait=True)
    except (OSError, ValueError, KeyError, IndexError, SyntaxError, TypeError, sqlite3.Error) as error:
        hold(str(error))
    return stop


def verify_pre_dispatch_failure(run, history, driver, *, has_relay=True):
    """Return a replayable startup failure only if no new executor took ownership."""
    import sqlite3

    launched = (history/'LAUNCH.json').stat().st_mtime
    endpoint = run/'distributed/ENDPOINT.json'
    progress = run/'PROGRESS.jsonl'
    if (not endpoint.is_file() or not progress.is_file() or endpoint.stat().st_mtime >= launched
            or load(endpoint)['pid'] == driver['pid'] or progress.stat().st_mtime >= launched
            or (run/'STOP_REQUESTED.json').exists()):
        return None
    owners = [load(history/'LAUNCH.json'), driver]
    lock_names = ['FAST_DRIVER.lock', 'distributed/pbs/MANAGER.lock']
    if has_relay:
        owners.append(load(run/'distributed/relay/STATUS.json'))
        lock_names.insert(1, 'distributed/relay/RELAY.lock')
    for owner in owners:
        try:
            live = (Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().strip(b'\0')
        except FileNotFoundError:
            live = False
        if live:
            raise BlockingIOError('a startup or previous relay owner remains active')
    local_worker_for_reattachment(run)
    with ExitStack() as locks:
        for name in lock_names:
            lock = locks.enter_context((run/name).open('rb'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        database = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
        with sqlite3.connect(database, uri=True) as db:
            if db.execute('SELECT 1 FROM events WHERE epoch>=? LIMIT 1', (launched,)).fetchone():
                return None
            verify_reattachable_attempts(run, db)
    return dict(reason='CONTROLLER_STARTUP_EXIT_BEFORE_DISPATCH', invocation_id=history.name)


def reconcile_hchain_pinned_pbs_setup(run, history, driver):
    """A PBS-only parent rejected its operational profile before manager intake."""
    import sqlite3
    spec = load(run/'ROUND.json')
    d = spec.get('executor',{}).get('distributed',{})
    if not (spec.get('hchain') and d.get('pbs') and not d.get('local_worker')
            and not d.get('relay') and d.get('pbs_hold_node')
            and (run/'PBS_PARTITION_PLAN.json').is_file()):
        return None
    with (run/'DRIVER.log').open('rb') as stream:
        stream.seek(max(0,os.fstat(stream.fileno()).st_size-8192))
        trace=stream.read().decode(errors='replace')
    if not trace.rstrip().endswith("ValueError: Existing run retains its pinned PBS settings"):
        return None
    launched=(history/'LAUNCH.json').stat().st_mtime
    for name in ('distributed/pbs/STATE.json','distributed/pbs/EVENTS.jsonl'):
        if (run/name).stat().st_mtime >= launched:
            return None
    for owner in (load(history/'LAUNCH.json'),driver):
        try:
            if (Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().strip(b'\0'):
                raise BlockingIOError('The failed controller still has a live process')
        except FileNotFoundError:
            pass
    with ExitStack() as locks:
        for name in ('FAST_DRIVER.lock','distributed/pbs/MANAGER.lock'):
            fcntl.flock(locks.enter_context((run/name).open('rb')),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with sqlite3.connect((run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
            if db.execute('SELECT 1 FROM jobs LIMIT 1').fetchone() or db.execute('SELECT 1 FROM events LIMIT 1').fetchone():
                return None
    return dict(reason='HCHAIN_PBS_PROFILE_REJECTED_BEFORE_DISPATCH',invocation_id=history.name)


def reconcile_hchain_preparation_setup(run, history, driver):
    """Recover the preparation setup exception before any new attempt was assigned."""
    import sqlite3

    if ((run/'STOP_REQUESTED.json').exists() or (run/'SEARCH_TERMINATION.json').exists()
            or load(run/'distributed/ENDPOINT.json')['pid'] != driver['pid']):
        return None
    with (run/'DRIVER.log').open('rb') as stream:
        stream.seek(max(0, os.fstat(stream.fileno()).st_size - 8192))
        trace = stream.read().decode(errors='replace')
    if ('in preparation_priorities' not in trace
            or "worker.h8_preparation_priority(plan, run, circuit_size)" not in trace
            or not trace.rstrip().endswith("AttributeError: 'dict' object has no attribute 'h8_preparation_priority'")):
        return None
    for owner in (load(history/'LAUNCH.json'), driver):
        try:
            if (Path('/proc')/str(owner['pid'])/'cmdline').read_bytes().strip(b'\0'):
                raise BlockingIOError('the failed H-chain controller still has a live process')
        except FileNotFoundError:
            pass
    local_worker_for_reattachment(run)
    with ExitStack() as locks:
        for name in ('FAST_DRIVER.lock', 'distributed/pbs/MANAGER.lock'):
            lock = locks.enter_context((run/name).open('rb'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        database = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
        with sqlite3.connect(database, uri=True) as db:
            allowed = ('CANCEL_QUEUED', 'HOST_ELIGIBILITY_HOLD', 'COMPLETED_PENDING_TRANSFER')
            if db.execute('SELECT 1 FROM events WHERE epoch>=? AND event NOT IN (?,?,?) LIMIT 1',
                          ((history/'LAUNCH.json').stat().st_mtime, *allowed)).fetchone():
                return None
            verify_reattachable_attempts(run, db)
    return dict(reason='HCHAIN_PREPARATION_SETUP_EXIT_WITH_NO_NEW_ASSIGNMENT', invocation_id=history.name)


def verify_reattachable_attempts(run, db):
    from .rpe_q3_store import digest as request_digest

    rows = db.execute("""SELECT job_id, method, state, host_id, attempt_id, request_sha256, request
                         FROM jobs WHERE state NOT IN ('ACCEPTED', 'CANCELLED')""")
    for job, method, state, host, attempt, sha, request in rows:
        directory = run/'jobs'/method/job
        if (directory.resolve().parent != (run/'jobs'/method).resolve()
                or not directory.resolve().is_relative_to((run/'jobs').resolve())
                or request_digest(json.loads(request)) != sha
                or request_digest(load(directory/'REQUEST.json')) != sha):
            raise ValueError('saved request differs from the queue: '+job)
        if state == 'QUEUED' and host is None and attempt is None:
            continue
        if state not in ('ASSIGNED', 'STARTING', 'RUNNING', 'COMPLETED_PENDING_TRANSFER', 'UNKNOWN') or not host or not attempt:
            raise ValueError('queue ownership is incomplete: '+job)
        marker = directory/'LAUNCH.json'
        if marker.is_file():
            launched = load(marker)
            if (launched.get('host_id'), launched.get('attempt_id')) != (host, attempt):
                raise ValueError('saved launch owner differs from the queue: '+job)


def reconcile_controller_detachment(run, launch, invocation):
    """Reattach the queue when the named controller disappeared before writing EXIT."""
    import re
    import sqlite3

    try:
        history = run/'invocations'/invocation
        if (not re.fullmatch(r'[0-9a-f]{32}', invocation) or launch['invocation_id'] != invocation
                or launch != load(history/'LAUNCH.json') or (history/'EXIT.json').exists()):
            raise ValueError('the detached invocation differs')
        driver = load(run/'NORMAL_DRIVER.json')
        endpoint = load(run/'distributed/ENDPOINT.json')
        spec = load(run/'ROUND.json')
        distributed = spec.get('executor', {}).get('distributed', {})
        hchain_pbs = (bool(spec.get('hchain')) and bool(distributed.get('pbs'))
                      and load(run/'APPLIED_SCOPE.json')['entry'] == 'method'
                      and not distributed.get('relay') and not distributed.get('local_worker'))
        owners = [('launcher', launch), ('driver', driver)]
        if not hchain_pbs:
            owners.append(('relay', load(run/'distributed/relay/STATUS.json')))
        if driver['invocation_id'] != invocation:
            raise ValueError('the detached driver and endpoint differ')
        for name, owner in owners:
            pid = owner['pid']
            if type(pid) is not int or pid <= 0:
                raise ValueError('invalid '+name+' PID')
            try:
                if Path('/proc', str(pid), 'cmdline').read_bytes().strip(b'\0'):
                    raise BlockingIOError(name+' still has a live process')
            except FileNotFoundError:
                pass
        if hchain_pbs:
            if (run/'STOP_REQUESTED.json').exists() or (run/'SEARCH_TERMINATION.json').exists():
                raise ValueError('STOP_OR_SEARCH_TERMINATION')
        else:
            hold = driver_restart_hold(run, spec)
            if hold:
                raise ValueError(hold)
        if endpoint['pid'] != driver['pid']:
            proof = verify_pre_dispatch_failure(run, history, driver, has_relay=not hchain_pbs)
            if proof is None:
                raise ValueError('the detached driver and endpoint differ')
            return proof
        local_worker_for_reattachment(run)
        with ExitStack() as locks:
            for name in ('FAST_DRIVER.lock', 'distributed/relay/RELAY.lock', 'distributed/pbs/MANAGER.lock'):
                path = run/name
                if name == 'FAST_DRIVER.lock' or path.exists():
                    fcntl.flock(locks.enter_context(path.open('rb')), fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
            with sqlite3.connect(path, uri=True) as db:
                if hchain_pbs:
                    # The H-chain method entry has no width-schedule checkpoint.
                    # Reattach only a PBS reservation before any new task acquired ownership.
                    if (db.execute('PRAGMA quick_check').fetchall() != [('ok',)]
                            or db.execute('SELECT 1 FROM jobs LIMIT 1').fetchone()
                            or db.execute('SELECT 1 FROM events LIMIT 1').fetchone()):
                        raise ValueError('H-chain dispatch ownership requires reconciliation')
                    pbs = load(run/'distributed/pbs/STATE.json')
                    allocations = [a for a in pbs['allocations'].values()
                                   if a['status'] != 'SUPERSEDED_UNSUBMITTED']
                    if (pbs['requests'] or not allocations
                            or any(a['status'] not in ('Q', 'R') or a.get('inflight')
                                   or not (a.get('observed') or {}).get('job_id') for a in allocations)):
                        raise ValueError('H-chain PBS reservation ownership is incomplete')
                verify_reattachable_attempts(run, db)
        return dict(reason='CONTROLLER_DETACHED_WITHOUT_EXIT_RECEIPT', invocation_id=invocation)
    except BlockingIOError as error:
        raise RequestError('runtime_ownership', 'controller detachment: '+str(error)) from error
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        raise RequestError('run', 'controller detachment: '+str(error)) from error


def reconcile_controller_kill(run, launch, ended, invocation):
    """Resume an ended controller; reattach exactly owned work on a fixed endpoint."""
    import sqlite3

    try:
        archived = run/'invocations'/invocation
        driver = load(run/'NORMAL_DRIVER.json')
        spec = load(run/'ROUND.json')
        distributed = spec.get('executor', {}).get('distributed', {})
        if (spec.get('hchain') and distributed.get('pbs')
                and not distributed.get('relay') and not distributed.get('local_worker')
                and load(run/'APPLIED_SCOPE.json')['entry'] == 'method'):
            # A deliberate controller-only interruption preserves the PBS
            # workers, their spools, and every durable attempt. This is not
            # the schedule restart path that requires closed allocations.
            if (launch.get('invocation_id') != invocation or ended.get('invocation_id') != invocation
                    or ended.get('returncode') != -signal.SIGTERM
                    or launch != load(archived/'LAUNCH.json') or ended != load(archived/'EXIT.json')
                    or driver.get('invocation_id') != invocation
                    or load(run/'distributed/ENDPOINT.json')['pid'] != driver['pid']):
                raise ValueError('the interrupted H-chain invocation differs')
            if (run/'STOP_REQUESTED.json').exists() or (run/'SEARCH_TERMINATION.json').exists():
                raise ValueError('STOP_OR_SEARCH_TERMINATION')
            for owner in (launch, driver):
                pid = owner['pid']
                if type(pid) is not int or pid <= 0:
                    raise ValueError('invalid controller PID')
                try:
                    if Path('/proc', str(pid), 'cmdline').read_bytes().strip(b'\0'):
                        raise ValueError('an old controller still has a live process')
                except FileNotFoundError:
                    pass
            local_worker_for_reattachment(run)
            with ExitStack() as locks:
                for name in ('FAST_DRIVER.lock', 'distributed/pbs/MANAGER.lock'):
                    fcntl.flock(locks.enter_context((run/name).open('rb')), fcntl.LOCK_EX | fcntl.LOCK_NB)
                pbs = load(run/'distributed/pbs/STATE.json')
                allocations = [a for a in pbs['allocations'].values() if a['status'] in ('Q', 'R')]
                if (not allocations or any(not (a.get('observed') or {}).get('job_id') for a in allocations)):
                    raise ValueError('existing H-chain PBS allocation ownership is missing')
                queue = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
                with sqlite3.connect(queue, uri=True) as db:
                    if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                        raise ValueError('H-chain queue integrity check failed')
                    verify_reattachable_attempts(run, db)
            return dict(reason='HCHAIN_CONTROLLER_INTERRUPTED_PBS_REATTACHED',
                        invocation_id=invocation, returncode=ended['returncode'],
                        allocations=[a['observed']['job_id'] for a in allocations])
        relay = load(run/'distributed/relay/STATUS.json')
        if (launch.get('invocation_id') != invocation or ended.get('invocation_id') != invocation
                or ended.get('returncode') != -signal.SIGKILL
                or launch != load(archived/'LAUNCH.json') or ended != load(archived/'EXIT.json')
                or driver.get('invocation_id') != invocation):
            raise ValueError('the killed invocation differs')
        for owner in (launch, driver):
            pid = owner['pid']
            if type(pid) is not int or pid <= 0:
                raise ValueError('invalid owner PID')
            process = Path('/proc', str(pid), 'cmdline')
            try:
                if process.read_bytes().strip(b'\0'):
                    raise ValueError('an owner still has a live process')
            except FileNotFoundError:
                pass
        hold = driver_restart_hold(run, load(run/'ROUND.json'))
        if hold:
            raise ValueError(hold)
        # Same fixed-endpoint reattachment as the in-invocation SIGKILL
        # recovery: outstanding attempts remain owned by their original hosts.
        # The controller and manager must be gone, not their valid workers.
        spec=load(run/'ROUND.json');distributed=spec['executor']['distributed']
        endpoint_path=run/'distributed/ENDPOINT.json'
        if endpoint_path.is_file() and distributed.get('port',8763):
            endpoint=load(endpoint_path);port=distributed.get('port',8763)
            if endpoint['pid']!=driver['pid'] or endpoint['url']!='http://127.0.0.1:'+str(port):
                raise ValueError('previous controller endpoint differs')
            local_worker_for_reattachment(run)
            relay_dir=(run/'distributed/relay').resolve();relay_config=relay_dir/'CONFIG.json'
            try:args=Path('/proc',str(relay['pid']),'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
            except FileNotFoundError:args=[]
            if args and args!=[b'']:
                if (len(args)<5 or Path(os.fsdecode(args[2])).name!='run_rpe_q3_relay.py'
                        or args[3:5]!=[b'--config',os.fsencode(relay_config)]
                        or Path(relay['configuration']).resolve()!=relay_config
                        or load(relay_config)['controller']['local_port']!=port
                        or (relay_dir/'STOP').exists()):
                    raise ValueError('surviving relay is not the fixed endpoint owner')
                with (relay_dir/'RELAY.lock').open('rb') as lock:
                    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:pass
                    else:raise ValueError('surviving relay has no ownership lock')
            pbs=run/'distributed/pbs';state=pbs/'STATE.json'
            with ExitStack() as locks:
                for path in (run/'FAST_DRIVER.lock',pbs/'MANAGER.lock'):
                    if path.exists():fcntl.flock(locks.enter_context(path.open('rb')),fcntl.LOCK_EX|fcntl.LOCK_NB)
                if state.is_file():
                    for record in load(state)['allocations'].values():
                        if record['status'] not in ('F','X','FINISHED_SAVED_WORKER','SUBMIT_REJECTED','SUPERSEDED_UNSUBMITTED'):
                            if Path(record['spec']['run_id']).resolve()!=run.resolve():
                                raise ValueError('open PBS allocation belongs to another run')
                queue=(run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
                with sqlite3.connect(queue,uri=True) as db:
                    verify_reattachable_attempts(run,db)
                    rows=db.execute("SELECT job_id,attempt_id,state FROM jobs WHERE state NOT IN ('ACCEPTED','CANCELLED','QUEUED')").fetchall()
            return dict(reason='CONTROLLER_KILLED_FIXED_ENDPOINT_REATTACHED',invocation_id=invocation,
                        returncode=ended['returncode'],reattached_attempts=[dict(job_id=j,attempt_id=a,state=s)for j,a,s in rows if s!='UNKNOWN'],
                        held_attempts=[dict(job_id=j,attempt_id=a)for j,a,s in rows if s=='UNKNOWN'])
        for owner in (relay,load(run/'distributed/LOCAL_DAEMON.json')):
            pid=owner['pid']
            if type(pid) is not int or pid<=0:raise ValueError('invalid worker PID')
            try:
                if Path('/proc',str(pid),'cmdline').read_bytes().strip(b'\0'):
                    raise ValueError('an owner still has a live process')
            except FileNotFoundError:pass
        terminal_pbs = ('F', 'X', 'FINISHED_SAVED_WORKER', 'SUBMIT_REJECTED')
        if any(a['status'] not in terminal_pbs or a.get('inflight')
               for a in load(run/'distributed/pbs/STATE.json')['allocations'].values()):
            raise ValueError('a run-owned PBS allocation is still open')
        with (run/'FAST_DRIVER.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            queue = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
            with sqlite3.connect(queue, uri=True) as db:
                rows = db.execute("SELECT job_id,state,attempt_id,detail,host_id,assigned_epoch,started_epoch "
                                  "FROM jobs WHERE state NOT IN "
                                  "('ACCEPTED','CANCELLED')")
                held = []
                for job, state, attempt, detail, host, assigned, started in rows:
                    if state == 'QUEUED' and all(value is None for value in (attempt, host, assigned, started)):
                        continue
                    detail = json.loads(detail)
                    if (state != 'UNKNOWN' or not attempt or detail.get('automatic_retry') is not False
                            or detail.get('resource_released') is not True):
                        raise ValueError('job still has unresolved ownership: '+job)
                    held.append(dict(job_id=job, attempt_id=attempt))
        return dict(reason='CONTROLLER_KILLED_CHECKPOINT_RECONCILED', invocation_id=invocation,
                    returncode=ended['returncode'], held_attempts=held)
    except BlockingIOError as error:
        raise RequestError('runtime_ownership','controller kill reconciliation: '+str(error)) from error
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        raise RequestError('run', 'controller kill reconciliation: '+str(error)) from error


def schedule_driver(config, run, selection):
    """Resolve the operator-selected scheduler for this exact pinned run."""
    scope = load(run/'APPLIED_SCOPE.json')
    entry = scope['entry']
    dispatch = selection.get('worker_trial', {}).get('pbs_dispatch_gateway')
    method_transport = entry == 'method' and dispatch is not None
    if (selection['run_id'] != scope['run_id'] or selection['base_source'] != config['code_source']
            or (entry != 'schedule' and not method_transport) or config.get('artificial_controller_directory')):
        raise RequestError('schedule_resume', 'Scheduler selection does not match the fixed run')
    if method_transport:
        spec = load(run/'ROUND.requested.json')
        distributed = spec.get('executor', {}).get('distributed', {})
        fields = {'run_id', 'base_source', 'root', 'source', 'driver_sha256',
                  'worker_trial', 'pbs_manager_sha256'}
        trial = selection['worker_trial']
        transport_fields = set(trial)-{'pbs_q3_measured_memory','pbs_q3_memory_learning'}
        hchain_cpu = transport_fields == {'pbs_dispatch_gateway','pbs_future_storage_launcher','hchain_cpu_policy'}
        if (not fields <= selection.keys() <= fields | {'resume_dispatch_failure', 'resume_controller_kill', 'q3_store_sha256'}
                or not hchain_cpu and transport_fields not in ({'pbs_dispatch_gateway'},
                                     {'pbs_dispatch_gateway', 'hchain_preparation_memory', 'optimizer_sha256'},
                                     {'pbs_dispatch_gateway', 'hchain_preparation_memory', 'optimizer_sha256', 'hchain_round_sha256'},
                                     {'pbs_dispatch_gateway', 'hchain_preparation_memory', 'optimizer_sha256',
                                      'hchain_round_sha256', 'pbs_future_storage_launcher'},
                                     {'pbs_dispatch_gateway', 'hchain_preparation_memory', 'optimizer_sha256',
                                      'hchain_round_sha256', 'pbs_future_storage_launcher', 'pbs_replacement'})
                or not isinstance(dispatch, dict) or set(dispatch) != {'path', 'sha256'}
                or not Path(dispatch['path']).is_absolute() or not re.fullmatch('[0-9a-f]{64}', dispatch['sha256'])
                or not spec.get('hchain') or not distributed.get('pbs_dispatch_gateway')
                or distributed.get('pbs_hold_node') and not hchain_cpu):
            raise RequestError('schedule_resume', 'Invalid operational selection for this fixed H-chain PBS run')
        if hchain_cpu:
            cpu = trial['hchain_cpu_policy']
            if (set(cpu) != {'base_root','overlay','policy'} or cpu['base_root'] != config['code_root']
                    or spec['hchain'].get('pbs_offline_search',{}).get('mode') != 'independent'
                    or not distributed.get('pbs_hold_node')
                    or cpu['overlay'].get('base_source') != config['code_source']
                    or not cpu['policy'].get('classes')):
                raise RequestError('schedule_resume','Invalid independent H-chain CPU policy')
        if reference := trial.get('pbs_q3_measured_memory'):
            if (set(reference)!={'path','sha256'} or not Path(reference['path']).is_absolute()
                    or hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest()!=reference['sha256']):
                raise RequestError('schedule_resume','Q3 measured memory operations differ')
        if 'pbs_q3_memory_learning' in trial:
            from .rpe_q3_memory import learning_policy
            learning_policy(trial['pbs_q3_memory_learning'])
        if (reference := trial.get('hchain_preparation_memory')):
            path = Path(reference['path'])
            optimizer = Path(selection['root'])/'src/trottertracks/rpe_optimizer_concurrent.py'
            if (not path.is_absolute() or not path.is_file() or not optimizer.is_file() or
                    hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']
                    or hashlib.sha256(optimizer.read_bytes()).hexdigest() != trial['optimizer_sha256']):
                raise RequestError('schedule_resume', 'H-chain preparation memory operations differ')
        if replacement := trial.get('pbs_replacement'):
            fields = set(replacement)
            if (fields not in ({'job_id','host_id'}, {'job_id','host_id','retain_predecessor','successor_token'},
                               {'job_id','host_id','retain_predecessor','successor_token','transfer_successor'},
                               {'job_id','host_id','retain_predecessor','successor_token','transfer_successor','restore_predecessor'})
                    or ('restore_predecessor' in replacement and replacement['restore_predecessor'] is not True)
                    or ('transfer_successor' in replacement and replacement['transfer_successor'] is not True)
                    or ('retain_predecessor' in replacement and (replacement['retain_predecessor'] is not True
                        or not re.fullmatch(r'[0-9a-f]{32}',replacement['successor_token'])))
                    or not re.fullmatch(r'[0-9]+\.[A-Za-z0-9_-]+',replacement['job_id']) or
                    replacement['host_id'] not in {w['host_id'] for w in distributed['pbs']['workers']}):
                raise RequestError('schedule_resume','Invalid PBS successor scope')
        if reference := trial.get('pbs_future_storage_launcher'):
            if (set(reference) != {'path', 'sha256'} or not Path(reference['path']).is_absolute()
                    or not re.fullmatch('[0-9a-f]{64}', reference['sha256'])):
                raise RequestError('schedule_resume', 'Invalid future PBS launcher reference')
        if 'hchain_round_sha256' in trial:
            round_script = Path(selection['root'])/'scripts/run_rpe_hchain_round.py'
            if hashlib.sha256(round_script.read_bytes()).hexdigest() != trial['hchain_round_sha256']:
                raise RequestError('schedule_resume', 'H-chain preparation priority operations differ')
    driver = Path(selection['root']).resolve()/'scripts'/ENTRIES[entry]
    if (not driver.is_file()
            or hashlib.sha256(driver.read_bytes()).hexdigest() != selection['driver_sha256']):
        raise RequestError('schedule_resume', 'Scheduler differs from its selected deployment')
    if 'publisher_sha256' in selection:
        for name in ('publish_romutail_search.py', 'publish_h4_search_run.py'):
            file = driver.with_name(name)
            if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != selection['publisher_sha256'].get(name):
                raise RequestError('schedule_resume', 'Saved-result publisher differs from deployment: '+name)
    if 'pbs_manager_sha256' in selection:
        manager = driver.parents[1]/'src/trottertracks/rpe_pbs.py'
        if (not manager.is_file()
                or hashlib.sha256(manager.read_bytes()).hexdigest() != selection['pbs_manager_sha256']):
            raise RequestError('schedule_resume', 'PBS manager differs from its selected deployment')
    if 'worker_trial' in selection and not method_transport:
        trial = selection['worker_trial']
        for name, field in (('rpe_optimizer_concurrent', 'optimizer_sha256'),
                            ('rpe_method_optimizer', 'method_optimizer_sha256')):
            if field != 'optimizer_sha256' and field not in trial:
                continue
            path = driver.parents[1]/'src/trottertracks'/(name+'.py')
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != trial[field]:
                raise RequestError('schedule_resume', 'Selected optimizer differs from deployment: '+name)
        if 'resume_pf_request_failure' in selection or 'resume_controller_kill' in selection:
            try:
                worker = local_worker_for_reattachment(run)
                settings = trial.get('local_resource_worker', {})
                if worker is not None:
                    legacy_root = Path(settings.get('root', driver.parents[1])).resolve()
                    transport_root = driver.parents[1].resolve()
                    # A completed rollover is also a legitimate owner on a
                    # later Resume. Admit only the two explicitly selected
                    # implementations, never another run/spool.
                    if worker == legacy_root/'scripts/run_rpe_q3_worker.py':
                        worker_root, files = legacy_root, settings.get('files')
                    elif (settings.get('transport_from_release', True)
                          and worker == transport_root/'scripts/run_rpe_q3_worker.py'):
                        worker_root, files = transport_root, settings.get('transport_files')
                    else:
                        worker_root, files = None, None
                    if not files:
                        raise ValueError('selected resume would start a second local worker')
                    for name, sha in files.items():
                        if hashlib.sha256((worker_root/name).read_bytes()).hexdigest() != sha:
                            raise ValueError('selected local worker file differs: '+name)
            except (OSError, ValueError, KeyError) as error:
                raise RequestError('schedule_resume', str(error)) from error
    if 'q3_store_sha256' in selection:
        store = driver.parents[1]/'src/trottertracks/rpe_q3_store.py'
        if not store.is_file() or hashlib.sha256(store.read_bytes()).hexdigest() != selection['q3_store_sha256']:
            raise RequestError('schedule_resume', 'Q3 store differs from its selected deployment')
    preparation = selection.get('preparation_sha256')
    if preparation is not None:
        required = set(SCHEDULE_PREPARATION_MODULES)
        optional = {'saved_rpe_cost_catalog', 'pr_step_price_registry', 'pr_candidate_completion', 'pf_optimizer_connection',
                    'rpe_q3_timing', 'rpe_worker_storage'}
        if not required <= set(preparation) <= required | optional:
            raise RequestError('schedule_resume', 'Preparation operations selection is incomplete')
        for name in preparation:
            module = driver.parents[1]/'src/trottertracks'/(name+'.py')
            if not module.is_file() or hashlib.sha256(module.read_bytes()).hexdigest() != preparation[name]:
                raise RequestError('schedule_resume', 'Preparation operations differ from their selected deployment: '+name)
    if 'pr_price_service_sha256' in selection:
        service = driver.with_name('pr_candidate_services.py')
        if (not preparation or not {'pr_candidate_completion', 'pr_step_price_registry'} <= set(preparation)
                or not service.is_file()
                or hashlib.sha256(service.read_bytes()).hexdigest() != selection['pr_price_service_sha256']):
            raise RequestError('schedule_resume', 'PR price metadata service differs from its selected deployment')
    return driver


def driver_restart_hold(run, spec):
    """Check whether the same invocation may reattach after its child was killed."""
    import sqlite3

    if (run/'STOP_REQUESTED.json').exists() or (run/'SEARCH_TERMINATION.json').exists():
        return 'STOP_OR_SEARCH_TERMINATION'
    try:
        checkpoint = load(run/'KAPPA_STATE.json')
        if (checkpoint['round_id'] != spec['round_id'] or checkpoint.get('stop')
                or checkpoint.get('schema') != 'RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1'):
            return 'CHECKPOINT_NOT_RESUMABLE'
        with (run/'FAST_DRIVER.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            database = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
            with sqlite3.connect(database, uri=True, timeout=1) as db:
                if db.execute('PRAGMA quick_check').fetchall() != [('ok',)]:
                    return 'QUEUE_INTEGRITY'
        if spec['executor']['distributed'].get('port') == 0:
            for path in ('distributed/LOCAL_DAEMON.json', 'distributed/relay/LAUNCH.json'):
                if (run/path).is_file() and Path('/proc', str(load(run/path)['pid']), 'cmdline').exists():
                    return 'DYNAMIC_ENDPOINT_STILL_HAS_LOCAL_OWNER'
    except (OSError, sqlite3.Error, ValueError, KeyError) as error:
        return 'RECONCILIATION_REQUIRED: '+str(error)
    return None


def restore_pre_dispatch_launch(run, launch, completed):
    """Restore the prior launch pointer only if its interrupted successor never entered dispatch."""
    import sqlite3

    invocation = launch.get('invocation_id')
    previous = completed.get('invocation_id')
    if not invocation or not previous or completed.get('returncode') != 0:
        return False
    archived = run/'invocations'/invocation
    before = run/'invocations'/previous
    try:
        proof = load(archived/'LAUNCH.json')
        normal = load(run/'NORMAL_DRIVER.json')
        old_launch = load(before/'LAUNCH.json')
        published_at = min((archived/'LAUNCH.json').stat().st_mtime, (run/'NORMAL_DRIVER.json').stat().st_mtime)
        if (launch != proof or normal['invocation_id'] != invocation
                or completed != load(before/'EXIT.json') or old_launch['invocation_id'] != previous
                or (archived/'EXIT.json').exists() or (run/'STOP_REQUESTED.json').exists()
                or (run/'SEARCH_TERMINATION.json').exists()):
            return False
        # A process in another PID namespace can have a different PID in its
        # marker. Use invocation and run arguments, not the marker PID alone.
        for proc in Path('/proc').iterdir():
            if not proc.name.isdecimal():
                continue
            try:
                if proc.stat().st_uid == os.getuid():
                    args = (proc/'cmdline').read_bytes().split(b'\0')
                    if os.fsencode(invocation) in args or os.fsencode(run) in args or os.fsencode(run.resolve()) in args:
                        return False
            except FileNotFoundError:
                pass
        with (run/'FAST_DRIVER.lock').open('rb') as driver, (run/'distributed/relay/RELAY.lock').open('rb') as relay:
            for lock in (driver, relay):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (load(run/'LAUNCH.json') != launch or load(run/'EXIT.json') != completed
                    or (run/'STOP_REQUESTED.json').exists() or (run/'PROGRESS.jsonl').stat().st_mtime >= published_at):
                return False
            database = (run/'distributed/QUEUE.sqlite3').resolve().as_uri()+'?mode=ro'
            with sqlite3.connect(database, uri=True, timeout=1) as db:
                if db.execute("""SELECT 1 FROM jobs WHERE assigned_epoch>=? OR started_epoch>=?
                    OR state IN ('ASSIGNED','STARTING','RUNNING','COMPLETED_PENDING_TRANSFER') LIMIT 1""",
                              (published_at, published_at)).fetchone():
                    return False
            immutable(archived/'PRE_DISPATCH_RECOVERY.json', dict(invocation_id=invocation,
                previous_invocation_id=previous, status='PREVIOUS_COMPLETED_LAUNCH_POINTER_RESTORED'))
            write(run/'LAUNCH.json', old_launch)
        return True
    except (OSError, KeyError, ValueError, sqlite3.Error):
        return False


def supervise_run(run, invocation, command):
    """Save a real wait receipt if the controller wrapper itself is killed."""
    run = Path(run)
    process = subprocess.Popen(command, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    code = process.wait()
    if code != -signal.SIGKILL:
        return code
    launch_path = run/'LAUNCH.json'
    history = run/'invocations'/invocation
    # The gateway publishes the launch while holding the registry lock.
    # Synchronize even if the child exited before that publication.
    config = load(run/'CONTROLLER_CONFIG.json')
    with locked(Path(config['state_directory'])):
        if (not launch_path.is_file() or load(launch_path).get('invocation_id') != invocation
                or (history/'EXIT.json').exists()):
            return code
        value = dict(returncode=code, command=command, invocation_id=invocation,
                     finished_utc=time.time(), observed_by='DETACHED_CONTROLLER_SUPERVISOR')
        write(history/'EXIT.json', value)
        write(run/'EXIT.json', value)
    return code


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
        optimizer = (str(driver.parents[1]/'src/trottertracks/rpe_optimizer_concurrent.py')
                     if trial and 'optimizer_sha256' in trial else '')
        preparation = schedule_resume.get('preparation_sha256')
        runtime_trial = trial
        for kind in ('ast', 'dispatch', 'pf_request'):
            recovered = schedule_resume.get('resume_'+kind+'_failure')
            if recovered and load(run/'EXIT.json').get('invocation_id') == recovered:
                runtime_trial = dict(trial or {}, recovered_failure=dict(kind=kind, invocation=recovered))
        operations = [(name, str(driver.parents[1]/'src/trottertracks'/(name+'.py')))
                      for name in ('pf_optimizer_connection', *SCHEDULE_PREPARATION_MODULES,
                                   'pr_step_price_registry', 'pr_candidate_completion')
                      if name in (preparation or {})]
        if 'q3_store_sha256' in schedule_resume:
            operations.insert(0, ('rpe_q3_store', str(driver.parents[1]/'src/trottertracks/rpe_q3_store.py')))
            implementation['q3_store_sha256'] = schedule_resume['q3_store_sha256']
        if trial and 'method_optimizer_sha256' in trial:
            # Executors subclass this module. Bind the selected class before
            # loading any helper or executor that imports it.
            operations.insert(0, ('rpe_method_optimizer',
                                  str(driver.parents[1]/'src/trottertracks/rpe_method_optimizer.py')))
        if 'saved_rpe_cost_catalog' in (preparation or {}):
            # Optimizer and provider imports bind Reader at module load time.
            operations.insert(0, ('saved_rpe_cost_catalog',
                                  str(driver.parents[1]/'src/trottertracks/saved_rpe_cost_catalog.py')))
        if 'rpe_worker_storage' in (preparation or {}):
            operations.append(('rpe_worker_storage', str(driver.parents[1]/'src/trottertracks/rpe_worker_storage.py')))
        if 'rpe_q3_timing' in (preparation or {}):
            operations.insert(0, ('rpe_q3_timing', str(driver.parents[1]/'src/trottertracks/rpe_q3_timing.py')))
        bootstrap = ('import importlib.util,json,runpy,sys\n'
                     'root,driver,manager,optimizer,preparation,trial,pr_price_service=sys.argv[1:8]\n'
                     'sys.argv=sys.argv[7:]\n'
                     'sys.path.insert(0,root+"/src")\n'
                     'import trottertracks\n'
                     'for name,path in [*json.loads(preparation),("rpe_pbs",manager),("rpe_optimizer_concurrent",optimizer)]:\n'
                     '    if path:\n'
                     '        spec=importlib.util.spec_from_file_location("trottertracks."+name,path)\n'
                     '        module=importlib.util.module_from_spec(spec)\n'
                     '        if name=="pr_candidate_completion":\n'
                     '            module.FIXED_RUN_CODE_ROOT=root\n'
                     '            module.FIXED_RUN_PR_PRICE_SERVICE=pr_price_service\n'
                     '        sys.modules[spec.name]=module\n'
                     '        spec.loader.exec_module(module)\n'
                     '        setattr(trottertracks,name,module)\n'
                     'runpy.run_path(driver,run_name="__main__",init_globals={'
                     '"FIXED_RUN_CODE_ROOT":root,"FIXED_RUN_WORKER_TRIAL":json.loads(trial)})')
        price_service = str(driver.with_name('pr_candidate_services.py')) if 'pr_price_service_sha256' in schedule_resume else ''
        command[2:3] = ['-c', bootstrap, str(root), str(driver), manager, optimizer,
                        json.dumps(operations), json.dumps(runtime_trial), price_service]
        implementation['schedule_source'] = schedule_resume['source']
        implementation['schedule_entry'] = str(driver)
        implementation['schedule_sha256'] = schedule_resume['driver_sha256']
        if preparation is not None:
            implementation['preparation_sha256'] = preparation
        if price_service:
            implementation['pr_price_service_sha256'] = schedule_resume['pr_price_service_sha256']
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
    restartable = (scope['entry'] == 'schedule'
                   and (schedule_resume is None or schedule_resume['source'] == config['code_source']
                        or schedule_resume.get('restart_driver_after_kill') is True)
                   and not config.get('artificial_controller_directory')
                   and bool(spec.get('executor', {}).get('distributed'))
                   and spec.get('search_overrides', {}).get('work_conserving_search') is True
                   and 'max_cycles' not in stop and spec.get('search_to_rule_stop') is True)
    try:
        if scope['entry'] == 'schedule' and not config.get('artificial_controller_directory') and (
                config.get('controller_import_check') or (schedule_resume or {}).get('controller_import_check')):
            checked = subprocess.run([*command, '--import-check'], cwd=config['code_root'],
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), capture_output=True, text=True, check=True)
            if json.loads(checked.stdout.splitlines()[-1]).get('status') != 'IMPORT_CHECK_PASS':
                raise ValueError('Selected controller import check did not complete')
        for generation in range(2):
            process = subprocess.Popen(command,cwd=config['code_root'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            driver = dict(pid=process.pid, command=command, invocation_id=invocation,
                          implementation=implementation)
            write(run/'NORMAL_DRIVER.json', driver)
            code = process.wait()
            if not restartable or generation or code != -signal.SIGKILL:
                break
            hold = driver_restart_hold(run, spec)
            write(run/'invocations'/invocation/'DRIVER_INTERRUPTION.json',
                  dict(driver, returncode=code, observed_epoch=time.time(), recovery_hold=hold))
            if hold is not None:
                break
            fixed_implementation(config)
            if schedule_resume is not None:
                schedule_driver(config, run, schedule_resume)
        if config.get('artificial_controller_directory'):
            result_path=Path(config['artificial_controller_directory'])/'run/ROUND_RESULT.json'
            if result_path.exists():
                write(run/'ROUND_RESULT.json',load(result_path))
        value=dict(returncode=code,command=command,finished_utc=time.time(),invocation_id=invocation)
        write(run/'invocations'/invocation/'EXIT.json',value);write(run/'EXIT.json',value)
        publish_after_exit(config, run, publisher_root=(schedule_resume['root'] if schedule_resume and
                                                       'publisher_sha256' in schedule_resume else None),
                           operations_source=schedule_resume['source'] if schedule_resume and
                           'publisher_sha256' in schedule_resume else None)
    except Exception as error:
        value=dict(returncode=1,error=str(error),command=command,invocation_id=invocation)
        write(run/'invocations'/invocation/'EXIT.json',value);write(run/'EXIT.json',value);raise


def publish_after_exit(config,run, *, publisher_root=None, operations_source=None):
    """Publish saved result metadata through C's normal path; never alter Q3."""
    publication=config.get('publication')
    if not publication:return
    run=Path(run)
    problem=load(run/'PROBLEM.json')
    if config.get('artificial_controller_directory') or str(problem.get('problem',{}).get('system','')).startswith('ARTIFICIAL'):
        write(run/'PUBLICATION_STATUS.json',dict(status='ARTIFICIAL_SKIPPED',scientific_calls=0))
        return
    command=[config['python'],'-B',str(Path(publisher_root or config['code_root'])/'scripts/publish_romutail_search.py'),
        '--repository',publication['repository'],'--run-dir',str(run),
        '--controller-source',config['code_source']]
    if publication.get('branch'):command+=['--branch',publication['branch']]
    if operations_source:command+=['--operations-source',operations_source]
    if publication.get('push',False):command+=['--push']
    try:
        with (run/'PUBLICATION.log').open('a')as log:
            result=subprocess.run(command,cwd=config['code_root'],stdout=log,stderr=subprocess.STDOUT,timeout=180,
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
