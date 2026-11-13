"""Approved P0--P4 saved-result arithmetic, not a signal/N/C/G optimizer.

Reuses pinned artifacts and exact accounting. No simulation/compiler/provider
is imported or called. All ratios retain their conditional or cohort meaning.
"""
from copy import deepcopy
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import re
import subprocess

from . import pf_pr_rpe_pipeline as p

a = p.accounting
PREPARATION = '80dade8e9b21bf62f54dabf5b554e42f157dc7f3'
SCOPE_PATH = 'docs/research/rpe_cost_preliminary_analysis_preparation_20261113/SOURCE_SCOPE.json'
TAILS = (6, 9, 10, 12)


def exact(value):
    # D's existing json_safe encodes Fraction as an integer or rational string.
    if isinstance(value, str) and re.fullmatch(r'-?\d+(?:/[1-9]\d*)?', value):
        return Fraction(value)
    return a.exact_scalar(value)


def number(value):
    if value is None:
        return None
    value = value if isinstance(value, Fraction) else exact(value)
    try:
        display = float(value)
    except OverflowError:
        display = None
    return dict(exact=a.exact_value(value), display=display if display is None or math.isfinite(display) else None)


def ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def sum_complete(values):
    return None if any(v is None for v in values) else sum(values, Fraction())


def structural(cells, *, tails=TAILS):
    """Exact proxy/box weights; missing terms never become zero."""
    rows = []; previous = {}
    for cell in sorted(cells, key=lambda v: (v['m'], v['axis'])):
        row = deepcopy(cell); issues = list(row.get('issues', [])); values = {}
        for name in ('N_nominal', 'S', 'lower', 'upper', 'T'):
            try:
                values[name] = exact(row[name])
            except (ValueError, TypeError, ZeroDivisionError):
                values[name] = None; issues.append('MISSING_OR_INVALID_' + name)
        n = row['N_nominal']
        if type(n) is not int or n < 1:
            issues.append('INVALID_INTEGER_N')
        if not issues and not (0 <= values['lower'] <= values['S'] <= values['upper'] and values['T'] > 0):
            issues.append('INVALID_STRUCTURAL_ENCLOSURE_OR_TIME')
        valid = not issues; s = values['S'] if valid else None
        row.update(issues=issues, status='UNKNOWN' if issues else 'CONDITIONAL_STRUCTURAL_PROXY',
            H=None if not valid else n*s,
            H_lower=None if not valid else n*values['lower'],
            H_upper=None if not valid else n*values['upper'],
            S_over_T_squared=None if not valid else s/values['T']**2,
            adjacent_S_ratio=ratio(s, previous.get((row['m']-1, row['axis']))))
        previous[row['m'], row['axis']] = s
        rows.append(row)
    rounds = []
    for m in sorted({v['m'] for v in rows}):
        selected = [v for v in rows if v['m'] == m]
        complete = {v['axis'] for v in selected} == set(p.AXES) and len(selected) == 2
        rounds.append(dict(m=m, H=sum_complete([v['H'] for v in selected]) if complete else None,
            lower=sum_complete([v['H_lower'] for v in selected]) if complete else None,
            upper=sum_complete([v['H_upper'] for v in selected]) if complete else None))
    total = sum_complete([v['H'] for v in rounds])
    total_low = sum_complete([v['lower'] for v in rounds])
    total_high = sum_complete([v['upper'] for v in rounds])
    def share_box(group):
        other = [v for v in rounds if v not in group]
        lo, hi = [sum_complete([v[k] for v in group]) for k in ('lower', 'upper')]
        clo, chi = [sum_complete([v[k] for v in other]) for k in ('lower', 'upper')]
        lower = ratio(lo, None if lo is None or chi is None else lo+chi)
        upper = ratio(hi, None if hi is None or clo is None else hi+clo)
        return [lower, upper]
    for row in rounds:
        row.update(share=ratio(row['H'], total), share_enclosure=share_box([row]))
    tail_rows = []
    for k in tails:
        group = [v for v in rounds if v['m'] >= k]
        tail_rows.append(dict(start_m=k, H=sum_complete([v['H'] for v in group]),
            share=ratio(sum_complete([v['H'] for v in group]), total), share_enclosure=share_box(group)))
    return dict(quantity='NOMINAL_WEIGHTED_IDEAL_STRUCTURAL_RZ_PROXY_NOT_G',
        cells=rows, rounds=rounds, tails=tail_rows, total=total, numerical_enclosure=[total_low, total_high],
        confidence_interval=None, sufficient_N=None, G=None, winner=None)


def pilot_ratios(summary):
    """Ratios of existing fresh summary means only; no new moments or CI."""
    out = []
    for row in summary:
        if (row['cohort'] != 'D_P2_DIRECT_PILOT_V1' or row['role'] != 'DEVELOPMENT'
                or row['expectation'] != 'FINITE_COHORT_ONLY' or row['n_paths'] != 4):
            raise ValueError('Fixed fresh pilot summary required')
        for i, axis in enumerate(p.AXES):
            c, s, d = [exact(v) for v in (row['C']['mean'][i], row['S']['mean'][i], row[axis]['mean_D'])]
            out.append(dict(m=row['m'], axis=axis, candidate_id=row.get('candidate_id'),
                cohort=row['cohort'], n_paths=row['n_paths'], role=row['role'], expectation=row['expectation'],
                C_mean=c, S_mean=s, D_mean=d, C_over_S=ratio(c,s), D_over_S=ratio(d,s),
                saved_C_moments=deepcopy(row['C']), saved_S_moments=deepcopy(row['S']),
                saved_D_moments=deepcopy(row[axis]), saved_XY_covariance_D=row['XY_covariance_D'],
                uncertainty=deepcopy(row['uncertainty']),
                qualification='DEVELOPMENT_FINITE_COHORT_RATIO_OF_SAVED_MEANS_NOT_LAW_EXPECTATION', CI=None))
    return out


def _read_gzip(store, ref):
    pair = ref['commit'], ref['path']
    if pair not in store.cache:
        raw = subprocess.check_output(['git','show',ref['commit']+':'+ref['path']], cwd=store.root)
        store.cache[pair] = json.loads(gzip.decompress(raw))
    return deepcopy(store.cache[pair][ref['key']])


def analyze(root):
    """Exactly the fixed P0--P4 scope; no fallback data sources or conditions."""
    store = p.SavedArtifacts(Path(root))
    scope = store.read(dict(commit=PREPARATION, path=SCOPE_PATH))
    refs = scope['sources']
    A,C,D,PF,screen,S = [store.read(refs[k]) for k in ('A_PR','C_S','D_P2','D_PF','A_SCREEN','B_S3_INPUT')]
    cid = scope['configuration_id']
    if any(v['configuration_id'] != cid for v in (A,C,S)):
        raise ValueError('Fixed canonical configuration mismatch')
    if [r['m'] for r in A['rounds']] != list(range(13)):
        raise ValueError('Fixed thirteen-round source required')
    s_index = {(v['m'], scope['structural_axis_to_N_axis'][v['axis']]):v for v in C['values']}
    if len(C['values']) != 26 or set(s_index) != {(m,axis) for m in range(13) for axis in p.AXES}:
        raise ValueError('Duplicate/missing structural axes')
    cells = []; signals = []
    for n, prepared, physical in zip(A['rounds'], S['rows'], scope['PR_rows']):
        m = n['m']
        if physical['m'] != m or prepared['m'] != m:
            raise ValueError('Round order mismatch')
        if n['source_candidate_id'] != physical['physical_candidate_id']:
            raise ValueError('Original physical ID mismatch')
        for axis in p.AXES:
            value = s_index[m,axis]
            if any(not p.same(value[k], n[k]) or not p.same(n[k],prepared[k]) for k in ('identity','law_binding')):
                raise ValueError('N/S/mean identity or law mismatch')
            if value['quantity'] != 'FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT':
                raise ValueError('Structural quantity changed')
            if n['N_'+axis] is not None or n['nominal_N_'+axis] != prepared['shots']['N_'+axis]:
                raise ValueError('Nominal scope/meaning mismatch')
            lo,hi = value['numerical_enclosure']
            cells.append(dict(m=m,axis=axis,N_nominal=n['nominal_N_'+axis],S=value['mean_S'],lower=lo,upper=hi,
                T=n['identity']['candidate']['T'],q_PF=value['q_PF'],R_outer=value['R_outer'],
                physical_candidate_id=physical['physical_candidate_id'],
                identity_law_source=dict(**refs['A_PR'],selector='rounds[m='+str(m)+']'),
                structural_source=dict(**refs['C_S'],selector='values[m='+str(m)+',axis='+value['axis']+']'),
                kernel_source=deepcopy(value['kernel_source'])))
        saved_mean = {k:deepcopy(v) for k,v in prepared['mean'].items() if k not in ('identity','law_binding','target')}
        signals.append(dict(family='PR',m=m,physical_candidate_id=physical['physical_candidate_id'],
            source=dict(**refs['B_S3_INPUT'],selector='rows[m='+str(m)+'].mean'), saved_mean=saved_mean,
            sufficient_N=None, global_RMSE_bound=None))
    proxy = structural(cells)
    break_even = []
    for lane in scope['full_pf_lanes']:
        for cell in lane['cells']:
            found = [v for v in PF['rows'] if v['physical_candidate_id'] == cell['physical_candidate_id']]
            if len(found) != 1:
                raise ValueError('Ambiguous fixed Full PF candidate')
            row = found[0]
            if row['system'] != 'H4' or row['prefix'] != 10 or row['formula'] != lane['formula']:
                raise ValueError('Full PF physical recipe mismatch')
            signal = _read_gzip(store, cell['signal_source'])
            if not p.same(signal['candidate'], row['physical_candidate']):
                raise ValueError('Full PF signal carrier identity mismatch')
            signals.append(dict(family='FULL_PF',label=lane['label'],m=cell['m'],source=deepcopy(cell['signal_source']),
                physical_candidate_id=cell['physical_candidate_id'],saved_record=signal,
                saved_numerical_status=deepcopy(row['numerical']), sufficient_N=None,global_RMSE_bound=None))
            for axis in p.AXES:
                choices = [v for v in row['prices'] if v['axes'][axis]['status'] == 'EXACT_METADATA_MATCH']
                selected = choices[0] if len(choices) == 1 else None
                value = None if selected is None else exact(selected['axes'][axis]['rz_count'])
                if selected is not None and selected.get('compiler') != p.PROFILE['compiler']:
                    selected = None; value = None
                s = exact(s_index[cell['m'],axis]['mean_S'])
                break_even.append(dict(label=lane['label'],m=cell['m'],axis=axis,
                    physical_candidate_id=cell['physical_candidate_id'],PF_C=value,PR_S=s,
                    theta=ratio(value,s), status='CONDITIONAL_BREAK_EVEN_THRESHOLD_NOT_COST_WINNER' if selected else 'UNKNOWN_MISSING_OR_AMBIGUOUS_PF_PRICE',
                    selected_price=deepcopy(selected),price_source=deepcopy(refs['D_PF']),
                    missing_N_ratio=None,missing_PR_compiled_multiplier=None,
                    comparison='r_N * rho_compiled < theta; no r_N/rho value adopted'))
    for cell in scope['screen_saved_rows']:
        ref=dict(commit=refs['A_SCREEN']['commit'],path=refs['A_SCREEN']['path'].rsplit('/',1)[0]+'/'+cell['record_path'])
        row=store.read(ref)
        if row['candidate']['id'] != cell['physical_candidate_id'] or row['m'] != cell['m']:
            raise ValueError('Screen source candidate mismatch')
        signals.append(dict(family='FULL_PF',label=cell['label'],m=cell['m'],source=ref,
            physical_candidate_id=cell['physical_candidate_id'],saved_record=row,sufficient_N=None,global_RMSE_bound=None))
    if [(v['m'],v['candidate_id']) for v in D['summary']] != [(v['m'],v['candidate_id']) for v in scope['D_pilot_rows']]:
        raise ValueError('Pilot scope mismatch')
    return dict(status='FINITE_SAVED_REANALYSIS_P0_P4_COMPLETE_CONDITIONAL_ONLY',
        input_scope=dict(commit=PREPARATION,path=SCOPE_PATH),configuration_id=cid,sources=refs,
        P0_P1=proxy,P2=break_even,P3=pilot_ratios(D['summary']),P4=signals,
        saved_target_and_shot_policy=deepcopy({k:S[k] for k in ('target','shot_policy')}),
        pilot_source=deepcopy(refs['D_P2']),pilot_execution_source=deepcopy(D['source']),
        screen_saved_status=deepcopy(screen),
        new_signal_MC_compile_fit=0,new_N_or_G_optimization=False,
        sufficient_N=None,global_RMSE_bound=None,G=None,winner=None,mandatory_stop=True)


def json_values(value):
    if isinstance(value,Fraction):
        return number(value)
    if isinstance(value,dict):
        return {k:json_values(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_values(v) for v in value]
    return value
