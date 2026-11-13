"""Passive PF/PR resource integration; planning never solves N or sums G.

Numerical entry points are explicit and require their own scientific scope.
Providers are batch callables, not command runners; no sampler/compiler/fitter
is imported. Existing supplied-axis accounting performs the cost arithmetic.
"""
from copy import deepcopy
import ast
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import json
import math
import re
import subprocess
import sys

from . import h4_df10_rpe_connection as accounting

AXES = ('X', 'Y')
ARMS = ('PF', 'PR')
SCOPE = accounting.SCOPE
MEASUREMENT = dict(X=dict(axis='cosine',observable='Re(z_raw)'),Y=dict(axis='sine',observable='Im(z_raw)'))
PROFILE = dict(compiler='1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b',
    control='symmetric_directional', wrapper='full_Hadamard_with_measurement',
    scope=SCOPE, P=0, qiskit_version='1.3.0', basis_gates=['rz','sx','x','cx'],
    optimization_level=1, seed_transpiler=17)


def same(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def arm(candidate):
    if candidate['method'] not in ('FULL', 'B0', 'B0I', 'PR'):
        raise ValueError('Unsupported algorithm family')
    return 'PR' if candidate['method'] == 'PR' else 'PF'


def key(row):
    return row['m'], row['candidate']['id']


class SavedArtifacts:
    """Immutable Git JSON selection only. No cache misses/builds or reanalysis."""
    def __init__(self, root):
        self.root, self.sources, self.cache = root, {}, {}

    def read(self, ref):
        commit, path = ref['commit'], ref['path']
        if not re.fullmatch('[0-9a-f]{40}', commit) or path.startswith('/') or '..' in path.split('/'):
            raise ValueError('Pinned commit and repository-relative path required')
        pair = commit, path
        if pair not in self.cache:
            data = subprocess.check_output(['git','show',commit+':'+path], cwd=self.root)
            self.sources[commit+':'+path] = hashlib.sha256(data).hexdigest()
            self.cache[pair] = json.loads(data)
        return deepcopy(self.cache[pair])


@lru_cache(maxsize=1)
def _schedule():
    path=Path(__file__).resolve().parents[1]/'trotterlib/h46_m5_plan.py'
    spec=importlib.util.spec_from_file_location('_pf_pr_metadata_schedule',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _candidate_id_parser(path, source):
    """Cache the callable, never candidate results; changed SOURCE reparses."""
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='cid')
    scope={'json':json,'hashlib':hashlib}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['cid']


def candidate_id(candidate):
    path=Path(__file__).resolve().parents[2]/'scripts/h46_long_time_signal.py'
    # Read exact bytes each time: an edited rule cannot reuse a stale callable,
    # even when file size/mtime are unchanged. Each candidate is evaluated anew.
    return _candidate_id_parser(str(path), path.read_bytes())(candidate)


def structure(c):
    """Declared saved recipe, not a compiled-price or action qualification."""
    if c['formula'] == '2nd':
        weights, formula_id = [1.], 'S2'
    elif c['formula'] in ('4th', '4th(m5_best)'):
        if c['formula'] == '4th' and arm(c) != 'PF':
            raise ValueError('Standard Yoshida4 extension covers deterministic PF only')
        # Read the existing fixed binary64 schedule through its passive module.
        schedule=_schedule()
        weights = list(schedule.weights(c['formula']))
        formula_id = ('PF_STANDARD_YOSHIDA4_BINARY64_V1' if c['formula'] == '4th' else schedule.FORMULA_ID)
    else:
        raise ValueError('Supply a source-bound structure for other formulas')
    if arm(c) == 'PF':
        allocation = [0]*len(weights)
    elif c['formula'] == '2nd':
        allocation = [c['R_outer']]
    else:
        schedule=_schedule()
        allocation = list(schedule.allocation(weights, c['R_outer']))
    return dict(formula_id=formula_id, signed_weights=weights, allocation=allocation,
        event_count=c['q']*sum(allocation), scalar_policy='scalar_identity_once_at_total_T',
        allocation_policy='ABS_TIME_PER_OUTER_INTEGER_MIN_ONE_PER_PIECE',
        law_model='ideal_finite_RTE_law' if arm(c)=='PR' else 'deterministic')


def task_from_saved(config, binding, rows, *, task_id, source, contract):
    """Transfer a declared schedule; never choose h/M/beta/prior or candidates."""
    entries=[]
    for row in rows:
        entries.append(dict(m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],
            candidate=deepcopy(row['candidate']),structure=structure(row['candidate']),
            adoption=row.get('adoption','SAVED_EVALUATED_CANDIDATE')))
    task=dict(schema_version=1, task_id=task_id, configuration=deepcopy(config),
        binding=deepcopy(binding), contract=deepcopy(contract), profile=deepcopy(PROFILE),
        candidates=entries, source=source, energy_target='saved_DF_H_sector_ground',
        state_assumption='saved_numerical_state_conditional_ground',
        full_RPE_qualified=False)
    validate_task(task)
    return task


def validate_task(task):
    c=task['configuration'];h=c['base_time'];M=c['horizon']['maximum_round_index_M']
    if (type(M) is not int or M<0 or not math.isfinite(h) or h<=0
            or c['horizon']['round_count_M_plus_one']!=M+1
            or not 0<c['beta_stat']<c['beta_total']<math.pi/3
            or c['energy_precision_ha']<=0 or not 0<c['alpha_total_per_method']<1
            or not 0<c['alpha_axis']<1
            or math.fsum([c['alpha_axis']]*(2*(M+1)))>c['alpha_total_per_method']):
        raise ValueError('Invalid declared RPE policy; no schedule optimization')
    if (not task.get('task_id') or not task.get('source') or not task.get('contract')
            or task['profile']['scope']!=SCOPE or task['profile']['P']!=0
            or task['energy_target']!='saved_DF_H_sector_ground'):
        raise ValueError('Explicit scientific task, contract and full-wrapper scope required')
    if any(not task['binding'].get(k) for k in ('system','hamiltonian','state','snapshot','rank_policy')):
        raise ValueError('H/state/snapshot/rank binding required')
    prior=c.get('energy_prior_ha')
    if not prior or len(prior)!=2 or not all(math.isfinite(v) for v in prior) or prior[0]>=prior[1] or not c.get('decoder'):
        raise ValueError('Supplied prior and decoder required; no implicit energy oracle')
    if len({key(r) for r in task['candidates']})!=len(task['candidates']):
        raise ValueError('Duplicate candidate row')
    for r in task['candidates']:
        v=r['candidate'];m=r['m'];arm(v)
        if (type(m) is not int or m not in range(M+1) or r['q_RPE']!=1<<m
                or type(v['q']) is not int or v['q']<=0 or r['q_PF']!=v['q']
                or v['T']!=math.ldexp(h,m) or v['delta']!=v['T']/v['q']
                or v['system']!=task['binding']['system']):
            raise ValueError('Dyadic time / q_PF / candidate mismatch')
        if v['id']!=candidate_id(v):raise ValueError('Candidate ID differs from existing canonical tuple')
        s=r['structure']
        if not same(s,structure(v)):raise ValueError('Recipe differs from existing fixed schedule')
        if (not s.get('formula_id') or not s['signed_weights']
                or len(s['allocation'])!=len(s['signed_weights'])
                or any(type(n) is not int or n<0 for n in s['allocation'])
                or s['event_count']!=v['q']*sum(s['allocation'])
                or s['scalar_policy']!='scalar_identity_once_at_total_T'):
            raise ValueError('Explicit signed recipe and single scalar accounting required')
        if arm(v)=='PR' and (sum(s['allocation'])!=v['R_outer'] or any(n<1 for n in s['allocation'])):
            raise ValueError('R_outer is per entire signed schedule')


def identity(task, row):
    return dict(m=row['m'],q_RPE=row['q_RPE'],q_PF=row['q_PF'],
        candidate=deepcopy(row['candidate']),structure=deepcopy(row['structure']),
        binding=deepcopy(task['binding']),contract=deepcopy(task['contract']),
        profile=deepcopy(task['profile']),configuration=deepcopy(task['configuration']))


def shot_record(task, row, saved, source):
    """Copy saved diagnostics and ideal/actual N; no new phase/N arithmetic."""
    if (not same(saved['candidate'],row['candidate']) or saved['m']!=row['m']
            or saved['q_PF']!=row['q_PF'] or saved['q_RPE']!=row['q_RPE']
            or not same(saved['ideal_numerical_pilot']['binding'],task['binding'])):
        raise ValueError('Saved N candidate/binding differs')
    for field in ('signed_weights','allocation','event_count'):
        if field in saved and not same(saved[field],row['structure'][field]):
            raise ValueError('Saved action recipe differs')
    lane=deepcopy(saved['ideal_numerical_pilot'])
    if lane['base_time']!=task['configuration']['base_time']:
        raise ValueError('Saved N base time differs')
    if any(lane['alpha_'+a]!=task['configuration']['alpha_axis'] for a in AXES):
        raise ValueError('Saved N failure allocation differs')
    return dict(identity=identity(task,row),plan=lane,actual=deepcopy(saved['actual_implementation']),
        diagnostics=deepcopy(saved),source=deepcopy(source),origin='SAVED_N_NOT_RECOMPUTED')


def measurement(row, axis, *, saved_sampler=False):
    return dict(measurement_axis=MEASUREMENT[axis]['axis'],observable=MEASUREMENT[axis]['observable'],
        outcome_convention='P0_minus_P1',units={'rz_count':'gate_count'},
        law_context=dict(N_law=row['structure']['law_model'],
            price_law='PCG64_numpy_float64_CDF_saved_cohort' if saved_sampler else row['structure']['law_model'],
            ideal_sampler_distance_qualified=not saved_sampler))


def saved_direct_prices(task, row, saved):
    """Keep cohorts distinct; copy already saved full-wrapper axis RZ means."""
    if not same(task['profile'],PROFILE):raise ValueError('Saved direct compiler/profile differs')
    records=[]
    for p in saved['compiled_prices']:
        for a in AXES:
            records.append(dict(**measurement(row,a,saved_sampler=arm(row['candidate'])=='PR'),identity=identity(task,row),axis=a,
                classification=p['classification'],cohort=p['cohort'],n_paths=p['n'],
                mean=dict(rz_count=p['rz_'+a]),source=deepcopy(p['source']),
                expectation='DETERMINISTIC' if p['classification']=='DIRECT_COMPILED_DETERMINISTIC' else 'FINITE_COHORT_ONLY',
                model=None,uncertainty=None,normalization_is_price_multiplier=False,
                preparation_evidence='UPSTREAM_SAVED_CANDIDATE_BINDING',peak_qubits=None))
    return records


def _match(task, row, record):
    if not same(record['identity'],identity(task,row)) or not record.get('source'):
        raise ValueError('Scientific/recipe/compiler/wrapper/SOURCE mismatch')


def _price_available(p):
    value=p.get('mean',{}).get('rz_count')
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value>=0


def _eligible_price(p, row, price_policy):
    allowed={'DETERMINISTIC','QUALIFIED_EXPECTATION','EXPLICIT_CONDITIONAL_MODEL'}
    if price_policy=='finite_cohort_scenario':allowed.add('FINITE_COHORT_ONLY')
    if not _price_available(p) or p['classification'] not in accounting.PRICE_KINDS or p['expectation'] not in allowed:
        return False
    if p['classification']!='MODEL_PREDICTION':return True
    model=p.get('model') or {}
    return (model.get('qualified_for_candidate') is True and model.get('q_variable')=='q_PF'
        and model.get('q')==row['q_PF'] and bool(model.get('validation_source'))
        and bool(model.get('coverage')) and bool(model.get('source')))


def _expected_price(p, row):
    return _eligible_price(p,row,'qualified_expectation')


def h4_saved_price_provider(read_json, *, selections=None):
    """B entry for A's frozen H4 catalog. No fresh prediction/compile fallback."""
    from .h4_df10_price_provider import H4PriceProvider
    return H4PriceProvider(read_json,pipeline=sys.modules[__name__],selections=selections)


def h6_saved_price_provider(read_json, *, selections=None):
    """H6 fixed-domain saved direct/Affine lookup; no numerical fallback."""
    from .h6_df19_price_provider import H6AffinePriceProvider
    return H6AffinePriceProvider(read_json,pipeline=sys.modules[__name__],selections=selections)


def plan(task, shots, prices, *, price_policy='qualified_expectation'):
    """Static coverage only: no prices predicted, N solved, or N*C evaluated."""
    if price_policy not in ('qualified_expectation','finite_cohort_scenario'):
        raise ValueError('Explicit price qualification policy required')
    validate_task(task);lookup={key(r):r for r in task['candidates']}
    nmap={};pmap={}
    for n in shots:
        k=key(n['identity'])
        if k not in lookup:raise ValueError('N outside declared candidate menu')
        _match(task,lookup[k],n)
        if k in nmap:raise ValueError('Repeated N result; do not add reruns')
        lane=n['plan'];row=lookup[k]
        if (not same(lane['candidate'],row['candidate']) or not same(lane['binding'],task['binding'])
                or lane['m']!=row['m'] or lane['q_RPE']!=row['q_RPE'] or lane['q_PF']!=row['q_PF']
                or lane['base_time']!=task['configuration']['base_time']
                or lane['target_id']!=task['task_id']
                or any(lane['alpha_'+a]!=task['configuration']['alpha_axis'] for a in AXES)):
            raise ValueError('N policy/task differs from the common identity')
        nmap[k]=n
    for p in prices:
        k=key(p['identity'])
        if k not in lookup:raise ValueError('C outside declared candidate menu')
        _match(task,lookup[k],p)
        if p['axis'] not in AXES or p.get('normalization_is_price_multiplier') is not False:
            raise ValueError('Axis and normalization accounting required')
        a=p['axis']
        if (p.get('measurement_axis')!=MEASUREMENT[a]['axis'] or p.get('observable')!=MEASUREMENT[a]['observable']
                or p.get('outcome_convention')!='P0_minus_P1' or p.get('units',{}).get('rz_count')!='gate_count'
                or p.get('law_context',{}).get('N_law')!=lookup[k]['structure']['law_model']):
            raise ValueError('Measurement sign/units or declared N-law mismatch')
        pk=k+(p['axis'],p['cohort'])
        if pk in pmap:raise ValueError('Duplicate price/cohort/axis')
        pmap[pk]=p
    items=[]
    for row in task['candidates']:
        k=key(row);n=nmap.get(k)
        qualified=(n is not None and n['plan']['status']=='CONDITIONAL_ROUND_PASS'
            and all(type(n['plan'].get('N_'+a)) is int and n['plan']['N_'+a]>0 for a in AXES))
        available=[p for pk,p in pmap.items() if pk[:2]==k]
        expected_cohorts=sorted({p['cohort'] for p in available if all(
            any(v['cohort']==p['cohort'] and v['axis']==a and _expected_price(v,row) for v in available) for a in AXES)})
        eligible_cohorts=sorted({p['cohort'] for p in available if all(
            any(v['cohort']==p['cohort'] and v['axis']==a and _eligible_price(v,row,price_policy) for v in available) for a in AXES)})
        items.append(dict(m=row['m'],candidate_id=k[1],arm=arm(row['candidate']),
            N_status='SAVED_CONDITIONAL' if qualified else ('MISSING' if n is None else 'UNQUALIFIED'),
            C_axes=sorted({p['axis'] for p in available if _price_available(p)}),
            C_record_axes=sorted({p['axis'] for p in available}),
            price_contexts=[dict(axis=p['axis'],cohort=p['cohort'],classification=p['classification'],expectation=p['expectation'],
                value_available=_price_available(p),price_unit=p.get('provider_quote',{}).get('price_unit'),
                validation=p.get('provider_quote',{}).get('validation'),reason=p.get('provider_quote',{}).get('reason'),
                model_qualified=(p.get('model') or {}).get('qualified_for_candidate'),source=deepcopy(p['source'])) for p in available],
            adoption=row['adoption'],
            expected_C_axes=sorted({p['axis'] for p in available if _expected_price(p,row)}),
            expected_C_cohorts=expected_cohorts,eligible_C_cohorts=eligible_cohorts))
    M=task['configuration']['horizon']['maximum_round_index_M']
    missing=[dict(m=m,arm=a) for m in range(M+1) for a in ARMS
             if not any(i['m']==m and i['arm']==a for i in items)]
    return dict(status='STATIC_PLAN_ONLY',rows=items,missing_candidate_slots=missing,
        missing_N=[i['candidate_id'] for i in items if i['N_status']=='MISSING'],
        unqualified_N=[i['candidate_id'] for i in items if i['N_status']=='UNQUALIFIED'],
        missing_C=[i['candidate_id'] for i in items if i['C_axes']!=list(AXES)],
        unqualified_expected_C=[i['candidate_id'] for i in items if not i['expected_C_cohorts']],
        unqualified_policy_C=[i['candidate_id'] for i in items if not i['eligible_C_cohorts']],
        C_provider_requests=[i['candidate_id'] for i in items if i['C_record_axes']!=list(AXES) and not i['eligible_C_cohorts']],
        blocked_C=[i['candidate_id'] for i in items if i['C_record_axes']==list(AXES) and not i['eligible_C_cohorts']],
        price_policy=price_policy,
        totals=None,science_calls=0,full_RPE_qualified=False,mandatory_stop=True)


def evaluate(task, shots, prices, *, cohort_selection, price_policy='qualified_expectation', joint=None):
    """Explicit future numerical entry. Fixed candidates first, no optimizer.

    cohort_selection[(candidate_id, axis)] chooses exactly one price scenario.
    No missing covariance, model systematic error or finite tail is set to zero.
    """
    inventory=plan(task,shots,prices,price_policy=price_policy)
    if inventory['missing_candidate_slots'] or inventory['missing_N'] or inventory['unqualified_N']:
        return dict(status='INCOMPLETE_OR_UNQUALIFIED_N',totals=None,inventory=inventory)
    M=task['configuration']['horizon']['maximum_round_index_M']
    if any(sum(r['m']==m and arm(r['candidate'])==a for r in task['candidates'])!=1
            for m in range(M+1) for a in ARMS):
        raise ValueError('Select one fixed candidate per arm/round before evaluation')
    chosen=[]
    for row in task['candidates']:
        for a in AXES:
            cohort=cohort_selection.get((row['candidate']['id'],a))
            matches=[p for p in prices if key(p['identity'])==key(row) and p['axis']==a and p['cohort']==cohort]
            if len(matches)!=1:return dict(status='MISSING_SELECTED_C',totals=None)
            p=matches[0]
            if not _price_available(p):return dict(status='MISSING_SELECTED_C_VALUE',totals=None)
            if p['classification'] not in accounting.PRICE_KINDS:
                return dict(status='UNVALIDATED_PRICE_EXTRAPOLATION',totals=None)
            if price_policy not in ('qualified_expectation','finite_cohort_scenario'):
                raise ValueError('Explicit price qualification policy required')
            allowed={'DETERMINISTIC','QUALIFIED_EXPECTATION','EXPLICIT_CONDITIONAL_MODEL'}
            if price_policy=='finite_cohort_scenario':allowed.add('FINITE_COHORT_ONLY')
            if p['expectation'] not in allowed:return dict(status='UNQUALIFIED_PRICE_EXPECTATION',totals=None)
            if p['classification']=='MODEL_PREDICTION':
                model=p.get('model') or {}
                if (model.get('qualified_for_candidate') is not True or model.get('q_variable')!='q_PF'
                        or model.get('q')!=row['q_PF'] or not model.get('validation_source')
                        or not model.get('coverage') or not model.get('source')):
                    return dict(status='UNQUALIFIED_MODEL_DOMAIN',totals=None)
            chosen.append(p)
    for row in task['candidates']:
        if len({x['cohort'] for x in chosen if key(x['identity'])==key(row)})!=1:
            raise ValueError('Paired X/Y must use a single declared price cohort/scenario')
    methods={}
    for alg in ARMS:
        ns=[deepcopy(n['plan']) for n in shots if arm(n['identity']['candidate'])==alg]
        cs=[]
        for p in chosen:
            i=p['identity']
            if arm(i['candidate'])!=alg:continue
            cs.append(dict(m=i['m'],axis=p['axis'],candidate=i['candidate'],binding=i['binding'],
                compiler=i['profile']['compiler'],scope=SCOPE,classification=p['classification'],
                source=p['source'],mean=p['mean'],model=p.get('model'),peak_qubits=p.get('peak_qubits')))
        methods[alg]=accounting.aggregate_supplied_axes(ns,cs,expected_rounds=range(M+1),
            alpha_total=task['configuration']['alpha_total_per_method'],compiler=task['profile']['compiler'],metrics=('rz_count',))
    if any(v.get('totals') is None for v in methods.values()):
        return dict(status='UNQUALIFIED_COST_SUM',totals=None,methods=methods)
    # Explicit joint order spans arms, rounds and paired X/Y, retaining shared fits.
    selected=sorted(chosen,key=lambda p:(arm(p['identity']['candidate']),p['identity']['m'],p['axis']))
    labels=[f"{arm(p['identity']['candidate'])}:{p['identity']['m']}:{p['identity']['candidate']['id']}:{p['axis']}" for p in selected]
    nlookup={key(n['identity']):n['plan'] for n in shots}
    weights=[nlookup[key(p['identity'])]['N_'+p['axis']] for p in selected]
    uncertainty=joint_uncertainty(labels,weights,[arm(p['identity']['candidate']) for p in selected],joint,means=[p['mean']['rz_count'] for p in selected])
    comparison=accounting.existing.compare_totals(
        dict(totals=methods['PF']['totals'],cost_scope=SCOPE,intervals=uncertainty['intervals']['PF']),
        dict(totals=methods['PR']['totals'],cost_scope=SCOPE,intervals=uncertainty['intervals']['PR']))
    comparison['verdict']=comparison['verdict'].replace('CDF','PF').replace('XDF','PR')
    return dict(status='CONDITIONAL_FIXED_CANDIDATE_COST',methods=methods,
        totals={a:methods[a]['totals'] for a in ARMS},price_policy=price_policy,
        uncertainty=uncertainty,comparison=comparison,price_order=labels,
        full_RPE_qualified=False,actual_success_guaranteed=False,
        normalization_is_price_multiplier=False,conditional_on_fixed_N=True,N_model_error_qualified=False,mandatory_stop=True)


def joint_uncertainty(labels, weights, arms, evidence, *, means=None):
    """Explicit W Sigma W^T, with paired/shared-calibration covariance intact.

    Track D's fixed 52-control-row transform cannot accept this PF/PR layout.
    Simultaneous price bounds are optional *supplied* evidence; no confidence
    level, residual bound or independence is inferred from covariance alone.
    """
    result=dict(covariance=None,systematic_bounds=None,total_confidence_interval=None,
        intervals={'PF':None,'PR':None},missing_covariance_is_zero=False,
        comparison_qualified=False,status='JOINT_UNCERTAINTY_MISSING')
    if evidence is None:return result
    if evidence.get('labels')!=labels or evidence.get('basis')!='mean_price_estimator':
        raise ValueError('Full paired mean-price covariance order required')
    if not evidence.get('source'):
        raise ValueError('Uncertainty evidence source required')
    matrix=evidence.get('covariance')
    if matrix is not None:
        if (evidence.get('cross_round_method_complete') is not True
                or evidence.get('shared_calibration_included') is not True):
            return dict(result,status='INCOMPLETE_DEPENDENCE')
        import numpy as np
        c=np.asarray(matrix,dtype=float);n=len(labels)
        if c.shape!=(n,n) or not np.isfinite(c).all() or not np.allclose(c,c.T,rtol=1e-12,atol=1e-12):
            raise ValueError('Joint covariance schema')
        scale=max(1.,float(np.max(np.abs(c))))
        if float(np.linalg.eigvalsh(c).min()) < -1e-10*scale:
            raise ValueError('Joint covariance must be positive semidefinite')
        w=np.array([[v if a==alg else 0 for a,v in zip(arms,weights)] for alg in ARMS],float)
        w=np.vstack([w,w[1]-w[0]])
        result.update(covariance=(w@c@w.T).tolist(),labels=['PF','PR','PR_minus_PF'],
            status='STATISTICAL_COMPONENT_ONLY')
    bounds=evidence.get('simultaneous_price_bounds')
    if bounds is not None and evidence.get('bounds_cover_stat_model_and_law') is True:
        if len(bounds)!=len(labels) or means is None:
            raise ValueError('Full ordered simultaneous price bounds required')
        for (lo,hi),mean in zip(bounds,means):
            if not 0<=lo<=mean<=hi or not math.isfinite(hi):raise ValueError('Price bounds must contain means')
        for alg in ARMS:
            result['intervals'][alg]={'rz_count':[math.fsum(v*b[j] for v,a,b in zip(weights,arms,bounds) if a==alg) for j in (0,1)]}
        result.update(comparison_qualified=True,status='SUPPLIED_SIMULTANEOUS_BOUND_COMPARISON')
    result.update(systematic_bounds=deepcopy(evidence.get('systematic_bounds')),source=evidence['source'])
    return result


def run_missing(task, shots, prices, *, n_provider=None, c_provider=None,price_policy='qualified_expectation'):
    """Future permitted batch: one N provider call and one C provider call.

    Reuse existing N even on FAIL; never retry unqualified candidates. Provider
    results are validated before proceeding. C response records (including
    unavailable) suppress automatic retries when both axes were observed.
    Keep returned prices for the next call. No aggregation or optimization.
    """
    if price_policy not in ('qualified_expectation','finite_cohort_scenario'):
        raise ValueError('Explicit price qualification policy required')
    initial=plan(task,shots,prices,price_policy=price_policy)
    if initial['unqualified_N']:
        return dict(status='STOP_UNQUALIFIED_N',shots=shots,prices=prices,plan=initial)
    missing=set(initial['missing_N']);newshots=list(shots);newprices=list(prices)
    if missing:
        if n_provider is None:return dict(status='N_PROVIDER_REQUIRED',plan=initial)
        requested=[r for r in task['candidates'] if r['candidate']['id'] in missing]
        batch=list(n_provider(task,requested))
        if {key(n['identity']) for n in batch}!={key(r) for r in requested} or len(batch)!=len(requested):
            raise ValueError('N batch returned different scope')
        newshots.extend(batch)
    middle=plan(task,newshots,newprices,price_policy=price_policy)
    if middle['unqualified_N']:
        return dict(status='STOP_UNQUALIFIED_N',shots=newshots,prices=newprices,plan=middle)
    # A response with null value is still a recorded lookup outcome. Finite
    # cohorts, domain failures and extrapolations need a scientific decision,
    # not another implicit provider/compile call on each invocation.
    missing=set(middle['C_provider_requests'])
    if missing and c_provider is not None:
        requested=[r for r in task['candidates'] if r['candidate']['id'] in missing]
        batch=list(c_provider(task,requested))
        expected={(key(r),a) for r in requested for a in AXES}
        if {(key(p['identity']),p['axis']) for p in batch}!=expected or len(batch)!=len(expected):
            raise ValueError('C batch returned different scope')
        # Validate even an unavailable overlap before deciding to retain cache.
        plan(task,newshots,batch,price_policy=price_policy)
        for price in batch:
            pk=key(price['identity'])+(price['axis'],price['cohort'])
            old=next((i for i,v in enumerate(newprices) if key(v['identity'])+(v['axis'],v['cohort'])==pk),None)
            if old is None:newprices.append(price)
            elif same(newprices[old],price):continue
            elif not _price_available(newprices[old]):newprices[old]=price
            elif _price_available(price):raise ValueError('Conflicting saved price; no automatic replacement')
    return dict(status='BATCH_PREPARED_NO_TOTALS',shots=newshots,prices=newprices,
        plan=plan(task,newshots,newprices,price_policy=price_policy),totals=None,automatic_retry=False,mandatory_stop=True)


def saved_model_prices(task, document, source, *, format):
    """Copy published summary predictions only; never fit/predict/average paths.

    The two historical formats remain unqualified. A/C future providers must
    supply explicit candidate-domain validation in the common price interface.
    """
    if not same(task['profile'],PROFILE):raise ValueError('Saved model compiler/profile differs')
    if not same(document['scientific_binding'],task['binding']):
        raise ValueError('Foreign saved model binding')
    records=[]
    for summary in document['summaries']:
        if format=='H6_V1':
            domain={(x['candidate_id'],x['T'],x['q_PF']) for x in document['rows'] if x['family']==summary['family']}
            if len(domain)!=1:raise ValueError('Ambiguous H6 summary domain')
            cid,T,q=next(iter(domain))
            matches=[r for r in task['candidates'] if r['candidate']['id']==cid and r['candidate']['T']==T and r['q_PF']==q]
            value=summary['mean_predicted'];name=document['model'];cohort=document['stage']+':'+summary['family']
        elif format=='H4_V2_S1':
            matches=[r for r in task['candidates'] if arm(r['candidate'])==summary['method'] and r['q_PF']==summary['q']
                     and r['candidate']['formula']=='2nd' and r['candidate']['delta']==.1
                     and r['candidate']['prefix']==(1 if summary['method']=='PR' else 7)
                     and (r['candidate']['R_outer'],r['candidate']['K'])==((4,2) if summary['method']=='PR' else (None,None))]
            value=summary['metrics']['prediction_mean'];name=summary['model']
            cohort=summary['role']+':'+summary['cohort']+':'+name
        else:raise ValueError('Unsupported saved model format')
        if not matches:continue
        if len(matches)!=1:raise ValueError('Ambiguous candidate model join')
        row=matches[0];axis={'cosine':'X','sine':'Y'}[summary['axis']]
        records.append(dict(**measurement(row,axis,saved_sampler=arm(row['candidate'])=='PR'),identity=identity(task,row),axis=axis,classification='MODEL_PREDICTION',
            cohort='MODEL:'+cohort,n_paths=summary.get('n_paths'),mean=dict(rz_count=value),
            source=deepcopy(source),expectation='UNQUALIFIED_MODEL',
            model=dict(name=name,source=deepcopy(source),validation_source=deepcopy(source),
                coverage='PUBLISHED_KNOWN_Q_ONLY_NOT_FULL_HORIZON',q_variable='q_PF',q=row['q_PF'],
                binding=deepcopy(task['binding']),compiler=task['profile']['compiler'],qualified_for_candidate=False),
            uncertainty=None,normalization_is_price_multiplier=False,
            preparation_evidence='UPSTREAM_PUBLISHED_DOMAIN_ONLY',peak_qubits=None))
    return records


def run_fixed_evaluation(task, shots, prices, *, cohort_selection, n_provider=None,
                         c_provider=None, price_policy='qualified_expectation',joint=None):
    """Explicit future science entry from fixed menu through conditional costs.

    Callbacks supply only missing records; upstream contracts govern execution.
    No retries, new candidates, schedule changes, fitting or compile fallback.
    """
    batch=run_missing(task,shots,prices,n_provider=n_provider,c_provider=c_provider,price_policy=price_policy)
    if batch['status']!='BATCH_PREPARED_NO_TOTALS':
        return dict(status=batch['status'],totals=None,batch=batch)
    result=evaluate(task,batch['shots'],batch['prices'],cohort_selection=cohort_selection,
        price_policy=price_policy,joint=joint)
    return dict(result,batch_plan=batch['plan'])


def saved_deterministic_points(task, document, source):
    """Copy individual saved deterministic PF wrapper counts, never average."""
    if not same(document['scientific_binding'],task['binding']) or not same(task['profile'],PROFILE):
        raise ValueError('Saved deterministic compiler/scientific binding differs')
    records=[]
    for point in document['rows']:
        if point.get('method')!='PF' or point.get('cohort')!='DETERMINISTIC':continue
        matches=[r for r in task['candidates'] if r['candidate']['id']==point['candidate_id']
            and r['candidate']['T']==point['T'] and r['q_PF']==point['q']]
        if not matches:continue
        row=matches[0];a={'cosine':'X','sine':'Y'}[point['axis']]
        records.append(dict(**measurement(row,a),identity=identity(task,row),axis=a,
            classification='DIRECT_COMPILED_DETERMINISTIC',cohort='SAVED_DETERMINISTIC_POINT',n_paths=1,
            mean=dict(rz_count=point['compiled_rz']),source=dict(source,selector=point['candidate_id']+':'+a),
            expectation='DETERMINISTIC',model=None,uncertainty=None,
            normalization_is_price_multiplier=False,preparation_evidence='PUBLISHED_V1_MATCHED_RECEIPT',peak_qubits=None))
    return records
