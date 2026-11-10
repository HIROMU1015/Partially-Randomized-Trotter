"""Phase 1 metadata, saved C/S joins and explicit paired numerical analysis.

Import/plan never sample, build, count S, fit or evaluate scientific prices.
Numerical functions are used on mock data in preparation; real use needs scope.
"""
import ast
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import types

from . import pf_pr_rpe_pipeline as p

BASE = '8457c443554af5fadaa3ba36f97f6316ca3c5050'
B = '7bca8f9642d1c42ca3fede4d31b2cdd8dcef46ae'
A = '5ca2dce3234f3e4a971f4a8c852da676f050e315'
CARRIER = '155529e20a8fac39e00786c40b57830b71203c4f'
PROVIDER = 'c31ef424175c02379eafebe8fc6c7548c1d9aabe'
H4 = '1cf87201ebc6c9e0296fe647354981b6f7f8b0c1'
H6 = '73dcbe23678bfc50d5654f09e600d4cfbe75e697'
H4V1 = 'eec5bdf9cc391fe49b55b5673f3e385deeb20d42'
H6V1 = 'a4d743cb5007d5fc6b6e93c3b43a4701ff956ac4'
C_HELPERS = 'ae44283d169b61deace7e1c0d18c4538d4a3fc0c'
AXES = ('cosine', 'sine')
METRICS = ('bias', 'MAE', 'RMSE', 'relative_bias', 'relative_RMSE', 'max_relative_error', 'paired_path_and_mean_covariance')
SOURCE_FILES = ('src/trottertracks/r_price_phase1.py', 'src/trottertracks/r_price_phase1_runner.py',
                'src/trottertracks/r_price_phase1_science.py', 'scripts/run_r_price_phase1.py')


def ref(commit, path): return dict(commit=commit, path=path)
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def blob(root, commit, path): return subprocess.check_output(['git', 'show', commit+':'+path], cwd=root, stderr=subprocess.PIPE)


def fixed_module(root, commit, path, name):
    module = types.ModuleType(name);module.__package__ = 'trottertracks';module.__file__ = str(Path(root)/path)
    exec(compile(blob(root, commit, path), commit+':'+path, 'exec'), module.__dict__)
    return module


@lru_cache(maxsize=None)
def helpers(root):
    """Reuse only pure fit/summary/seed bodies; no historical launcher imports."""
    import numpy as np
    scope = dict(np=np, defaultdict=defaultdict, hashlib=hashlib, json=json)
    for commit, path, names in [
        (BASE, 'src/trottertracks/track_d_c7_4_models.py', ('fit',)),
        (BASE, 'src/trottertracks/h4_df10_price_v2.py', ('summarize',)),
        (CARRIER, 'scripts/h46_long_time_native.py', ('trial_seed',)),
    ]:
        nodes = [n for n in ast.parse(blob(root, commit, path)).body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in nodes} != set(names): raise ValueError('PINNED_HELPER_MISSING')
        exec(compile(ast.Module(body=nodes, type_ignores=[]), commit+':'+path, 'exec'), scope)
    return scope


def make_plan(root, source):
    """Public metadata only, including candidate/roles/seeds before acquisition."""
    store = p.SavedArtifacts(root)
    b = fixed_module(root, B, 'src/trottertracks/rpe_r_selection.py', 'trottertracks._phase1_fixed_B')
    law = dict(model='ideal_finite_RTE_law', sampling='PCG64_numpy_float64_CDF_saved_cohort',
               source=ref(PROVIDER, 'src/trotterlib/rte.py'))
    menus = {};cells = []
    for system, ms, rs, ns in [('H4', (3,4), (4,8), (9,33)), ('H6', (4,5), (16,32), (9,8))]:
        task = store.read(ref('3b742fff26381514e12c07b51701276b7d1b26ec',
            'docs/research/pf_pr_rpe_cost_integration_20261110/static_plan/'+system+'_PLAN.json'))['task']
        menus[system] = b.build_menu(task, {m:list(rs) for m in ms}, sampling_law=law)
        for m in ms:
            for R in rs:
                row = next(r for r in menus[system]['candidates'] if r['m']==m and r['candidate']['method']=='PR' and r['candidate']['R_outer']==R)
                saved = R==rs[0];role = 'CALIBRATION' if saved or m==ms[0] else 'VALIDATION'
                cohort = ('EXPLORATION' if m==ms[0] else 'H4_DF10_V2_INDEPENDENT_Q_20261110' if system=='H4' else 'H6_PRICE_V2_20261110_Q2_CALIBRATION') if saved else f'R_PRICE_P1_20261111_{system}_R{R}_{role}'
                cells.append(dict(id=system+f'_q{row["q_PF"]}_R{R}', system=system, m=m, candidate_id=row['candidate']['id'],
                    role=role, saved=saved, n=ns[ms.index(m)] if saved else 8, cohort=cohort,
                    request=b.request_context(menus[system], row)))
    plan = dict(schema_version=1, source=source, B_source=B, A_source=A, menus=menus, cells=cells,
        axes=list(AXES), features=['intercept','q_PF','q_PF_times_R_outer'], metrics=list(METRICS),
        model='STRUCTURAL_PLUS_THREE_COMPILED_RESIDUAL_TERMS', acceptance_threshold=None,
        scientific_status='PREPARED_NOT_EXECUTED_NOT_APPROVED', new_normal_wrapper_budget=64, new_rare_wrapper_budget=16,
        rare=dict(pairs_per_system=2, path_versions=['order0','order2'], unit='FORCED_LEGAL_PAIR_NOT_UNCONDITIONAL_COHORT',
            H4=['head_tail_first_event','internal_different_basis'], H6=['positive_piece_first_event','negative_piece_first_event']),
        runtime=dict(provider_root='/home/AbeHiromu/projects/h46-higher-pf-acquire/20261018/private/provider',
            input_config='/home/AbeHiromu/projects/h46-pr-exploration-execution/20261021/private/authorization_run01/H4_PR_EXPLORATION/config_APPROVED_PRIVATE.json',
            old_cache='/home/AbeHiromu/projects/h46-long-time-pf-pr/20261025/compiled_cache'),
        source_files=list(SOURCE_FILES), mandatory_stop=True, full_RPE_qualified=False)
    seeds=[]
    for cell in cells:
        if cell['saved']: continue
        row=row_for(plan,cell);c=row['candidate'];J=len(row['structure']['allocation'])
        cell['seeds']=[[helpers(str(root))['trial_seed'](c,cell['cohort'],i,o,j) for o in range(c['q']) for j in range(J)] for i in range(cell['n'])]
        seeds += [v for vs in cell['seeds'] for v in vs]
    if len(seeds)!=len(set(seeds)): raise ValueError('NEW_SEED_COLLISION')
    normal_stream_count=len(seeds)
    plan['rare_groups']=[]
    for system in ('H4','H6'):
        cell=next(c for c in cells if c['system']==system and not c['saved'] and c['role']=='CALIBRATION')
        row=row_for(plan,cell);c=row['candidate'];J=len(row['structure']['allocation'])
        cohort='R_PRICE_P1_20261111_'+system+'_FORCED_DIAGNOSTIC'
        for pair in (0,1):
            values=[helpers(str(root))['trial_seed'](c,cohort,pair,o,j) for o in range(c['q']) for j in range(J)]
            plan['rare_groups'].append(dict(system=system,pair=pair,cohort=cohort,candidate_id=c['id'],seeds=values))
            seeds+=values
    if len(seeds)!=len(set(seeds)):raise ValueError('DIAGNOSTIC_SEED_COLLISION')
    plan['seed_coverage'] = dict(new_normal_streams=normal_stream_count,new_diagnostic_streams=len(seeds)-normal_stream_count,
        distinct_new_streams=len(seeds),new_collisions=0,complete_historical_coverage=False)
    validate_plan(plan)
    return plan


def row_for(plan, cell):
    return next(r for r in plan['menus'][cell['system']]['candidates'] if r['candidate']['id']==cell['candidate_id'])


@lru_cache(maxsize=2)
def fixed_task(system):
    root=Path(__file__).resolve().parents[2]
    return p.SavedArtifacts(root).read(ref('3b742fff26381514e12c07b51701276b7d1b26ec',
        'docs/research/pf_pr_rpe_cost_integration_20261110/static_plan/'+system+'_PLAN.json'))['task']


def validate_plan(plan):
    if (tuple(plan['source_files'])!=SOURCE_FILES or plan['B_source']!=B or plan['A_source']!=A or len(plan.get('rare_groups',[]))!=4):raise ValueError('PINNED_SOURCE_OR_RARE_SCOPE')
    if (plan['axes']!=list(AXES) or plan['features']!=['intercept','q_PF','q_PF_times_R_outer'] or
            plan['new_normal_wrapper_budget']!=64 or plan['new_rare_wrapper_budget']!=16 or len(plan['cells'])!=8):
        raise ValueError('FIXED_PHASE1_PLAN')
    for s, expected in [('H4', {(8,4),(16,4),(8,8),(16,8)}), ('H6', {(1,16),(2,16),(1,32),(2,32)})]:
        task=plan['menus'][s]
        if any(not p.same(task.get(k),v) for k,v in fixed_task(s).items() if k!='candidates'):raise ValueError('FROZEN_SCIENTIFIC_TASK_CHANGED')
        p.validate_task(task);cells=[c for c in plan['cells'] if c['system']==s]
        observed={(row_for(plan,c)['q_PF'],row_for(plan,c)['candidate']['R_outer']) for c in cells}
        if observed!=expected or sum(c['role']=='CALIBRATION' for c in cells)!=3: raise ValueError('CELL_ROLES_OR_DOMAIN')
        for c in cells:
            r=row_for(plan,c);v=r['candidate'];qlo=min(q for q,R in expected);Rlo=min(R for q,R in expected)
            expected_role='CALIBRATION' if v['R_outer']==Rlo or v['q']==qlo else 'VALIDATION'
            if (c['role']!=expected_role or c['saved']!=(v['R_outer']==Rlo) or c['request']['identity']!=p.identity(plan['menus'][s],r)
                    or v['K']!=2 or v['prefix']!=(1 if s=='H4' else 7) or v['delta']!=(.1 if s=='H4' else 1.6)):
                raise ValueError('FROZEN_CELL_ROLE_IDENTITY')
            if c['saved'] and c['n']!=((9 if v['q']==qlo else 33) if s=='H4' else (9 if v['q']==qlo else 8)):raise ValueError('FIXED_SAVED_COHORT')
            if not c['saved'] and (c['n']!=8 or len(c['seeds'])!=8): raise ValueError('FIXED_NEW_COHORT')
    if {(v['system'],v['pair']) for v in plan['rare_groups']}!={(s,i) for s in ('H4','H6') for i in (0,1)}:raise ValueError('RARE_PAIR_SCOPE')
    if plan['metrics']!=list(METRICS) or plan['acceptance_threshold'] is not None: raise ValueError('FROZEN_METRICS_POLICY')


def _binding_matches(saved, expected):
    ignored={'T','profile','profile_is_reference_only','rank_is_saved_observation_not_generation_argument'}
    if any(not p.same(expected.get(k),v) for k,v in saved.items() if k not in ignored):
        raise ValueError('SAVED_SCIENTIFIC_BINDING')


def saved_cell(root, plan, cell, read=None):
    """Explicit execution-only join: copy C/S by exact path/axis, no recount."""
    if not cell['saved']:raise ValueError('SAVED_CELL_ONLY')
    read = read or p.SavedArtifacts(root).read
    s=cell['system'];cid=cell['candidate_id'];row=row_for(plan,cell);records=[]
    binding=plan['menus'][s]['binding']
    def record(index, C, S, source, event_digest):
        return dict(cell_id=cell['id'],candidate_id=cid,index=index,cohort=cell['cohort'],conditional=False,
            C=C,S=S,source=source,event_digest=event_digest,compiler=plan['menus'][s]['profile']['compiler'])
    if row['q_PF']==(8 if s=='H4' else 1):
        commit=H4V1 if s=='H4' else H6V1;path=f'docs/research/{s.lower()}_df'+('10' if s=='H4' else '19')+'_price_v1_20261110/DEVELOPMENT.json'
        doc=read(ref(commit,path));_binding_matches(doc['scientific_binding'],binding)
        if doc['compiler_hash']!=p.PROFILE['compiler']: raise ValueError('SAVED_COMPILER')
        rows=[r for r in doc['rows'] if r['candidate_id']==cid and r['cohort']==cell['cohort']]
        bindings={b['index']:b for b in doc['private_trial_bindings'] if s=='H4' or b['family']=='PR_B'}
        for index in sorted({r['path_index'] for r in rows}):
            paired={r['axis']:r for r in rows if r['path_index']==index}
            if set(paired)!=set(AXES) or len([r for r in rows if r['path_index']==index])!=2: raise ValueError('SAVED_PAIRED_AXIS')
            records.append(record(index,[paired[a]['compiled_rz'] for a in AXES],
                [paired[a]['prediction']['structural_rz'] for a in AXES],ref(commit,path),bindings[index]['events_sha256']))
    elif s=='H4':
        base='docs/research/h4_df10_price_v2_independent_q_20261110/'
        pred=read(ref(H4,base+'PREDICTIONS.json'));result=read(ref(H4,base+'completion01/RESULT.json'));prov=read(ref(H4,base+'completion01/PROVENANCE.json'))
        _binding_matches(prov['scientific_binding'],binding)
        if prov['compiler']!=p.PROFILE['compiler'] or prov['cohort']!=cell['cohort']:raise ValueError('SAVED_COMPILER_COHORT')
        predictions={(r['path_index'],r['axis']):r for r in pred['predictions'] if r['candidate_id']==cid}
        binds={b['path_index']:b for b in prov['trial_bindings'] if b['candidate_id']==cid}
        for trial in result['trials']:
            if trial['candidate_id']!=cid: continue
            i=trial['index'];dig=binds[i]['canonical_events_sha256']
            if any(predictions[i,a]['events_sha256']!=dig for a in AXES): raise ValueError('SAVED_EVENT_C_S_JOIN')
            if {b['axis'] for b in trial['wrapper_bindings']}!=set(AXES) or len(trial['wrapper_bindings'])!=2 or any(b['compiler']!=p.PROFILE['compiler'] for b in trial['wrapper_bindings']): raise ValueError('SAVED_COMPILER')
            records.append(record(i,[trial['prices'][a]['rz_count'] for a in AXES],
                [predictions[i,a]['structural_rz'] for a in AXES],ref(H4,base+'completion01/RESULT.json'),dig))
    else:
        path='docs/research/h6_df19_price_v2_20261110/Q2_DIRECT.json';doc=read(ref(H6,path))
        if doc['compiler_hash']!=p.PROFILE['compiler']: raise ValueError('SAVED_COMPILER')
        for trial in doc['rows']:
            if trial['candidate']['id']==cid:
                if trial.get('status')!='PASS_PRICE_ACQUIRED':raise ValueError('SAVED_PRICE_NOT_COMPLETE')
                if not p.same(trial['candidate'],row['candidate']) or trial['cohort']!=cell['cohort']: raise ValueError('SAVED_CANDIDATE')
                records.append(record(trial['index'],[trial['prices'][a]['rz_count'] for a in AXES],
                    [trial['predictions'][a]['structural_rz'] for a in AXES],ref(H6,path),None))
    validate_records(plan,cell,records)
    return records


def validate_records(plan, cell, rows):
    import math
    if not any(p.same(cell,c) for c in plan['cells']):raise ValueError('CELL_OUTSIDE_PHASE1_NO_EXTRAPOLATION')
    if len(rows)!=cell['n'] or {r['index'] for r in rows}!=set(range(cell['n'])): raise ValueError('COMPLETE_COHORT_REQUIRED')
    for r in rows:
        if (r['cell_id']!=cell['id'] or r['candidate_id']!=cell['candidate_id'] or r['cohort']!=cell['cohort'] or r['conditional'] is not False
                or r['compiler']!=p.PROFILE['compiler']): raise ValueError('CELL_OR_CONDITIONAL_BINDING')
        if any(len(r[k])!=2 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in r[k]) for k in ('C','S')):
            raise ValueError('FINITE_PAIRED_C_S_REQUIRED')


def calibrate(root, plan, data):
    """3 cell means, independent system fits and nominal paired covariance."""
    import numpy as np
    validate_plan(plan);models={}
    expected={c['id'] for c in plan['cells'] if c['role']=='CALIBRATION'}
    if set(data)!=expected: raise ValueError('CALIBRATION_ONLY_NO_HOLDOUT')
    for s in ('H4','H6'):
        cells=[c for c in plan['cells'] if c['system']==s and c['role']=='CALIBRATION'];X=[];Y=[];covs=[]
        for cell in cells:
            rs=data[cell['id']];validate_records(plan,cell,rs);c=row_for(plan,cell)['candidate']
            residual=np.array([r['C'] for r in rs])-np.array([r['S'] for r in rs]);X.append([1,c['q'],c['q']*c['R_outer']]);Y.append(residual.mean(axis=0))
            covs.append(np.cov(residual.T,ddof=1)/len(rs))
        x=np.asarray(X,float);y=np.asarray(Y,float)
        rank=int(np.linalg.matrix_rank(x))
        if rank!=3: raise ValueError('UNIDENTIFIABLE_THREE_PARAMETER_MODEL')
        # Unique cell labels make the reused helper's weights equal per cell.
        beta=helpers(str(root))['fit'](x,y,np.arange(3));inverse=np.linalg.inv(x)
        sigma=np.zeros((6,6))
        for k,V in enumerate(covs): sigma+=np.kron(np.outer(inverse[:,k],inverse[:,k]),V)
        if not np.isfinite(beta).all() or not np.isfinite(sigma).all(): raise ValueError('NONFINITE_FIT')
        models[s]=dict(coefficients=beta.tolist(),rank=rank,cell_level_lack_of_fit_DOF=0,
            features=plan['features'],axes=list(AXES),calibration_cells=[c['id'] for c in cells],
            point_roles={c['id']:c['role'] for c in plan['cells'] if c['system']==s},unobserved_q_R_qualified=False,
            nominal_coefficient_covariance=sigma.tolist(),covariance_order='feature_then_axis',
            covariance_qualification='Independent calibration cohorts, paired C-S/X/Y; finite nominal only, excludes rare/law/systematic',
            shared_calibration_group='R_PRICE_P1:'+s+':'+digest({k:data[k] for k in expected if k.startswith(s)}),
            R_q_systematic=None,all_law_expected_price=None,total_CI=None,qualified_for_candidate=False)
    return dict(plan_digest=digest(plan),model=plan['model'],models=models,metrics=plan['metrics'],
        validation_prices_observed=False,validation_refit=False,acceptance_threshold=None,full_RPE_qualified=False)


def evaluate_cell(root, plan, cell, rows, frozen):
    import numpy as np
    if frozen['plan_digest']!=digest(plan): raise ValueError('FROZEN_PLAN_MISMATCH')
    validate_records(plan,cell,rows);c=row_for(plan,cell)['candidate'];model=frozen['models'][cell['system']]
    correction=np.array([1,c['q'],c['q']*c['R_outer']])@np.array(model['coefficients']);output=[]
    for r in rows:
        prediction=np.array(r['S'])+correction
        if not np.isfinite(prediction).all() or min(prediction)<0: raise ValueError('INVALID_MODEL_PREDICTION_NO_CLAMP')
        for j,axis in enumerate(AXES):
            actual=r['C'][j];pred=float(prediction[j]);e=actual-pred
            output.append(dict(system=cell['system'],role=cell['role'],method='PR',cohort=cell['cohort'],model=plan['model'],
                q=c['q'],R_outer=c['R_outer'],candidate_id=c['id'],path_index=r['index'],axis=axis,
                compiled_rz=actual,structural_rz=r['S'][j],prediction_rz=pred,error_rz=e,
                relative_error=e/actual if actual else None,diagnostic='FINITE'))
    actual=np.array([r['C'] for r in rows]);S=np.array([r['S'] for r in rows]);predicted=S+correction;error=actual-predicted
    return dict(cell_id=cell['id'],role=cell['role'],summaries=helpers(str(root))['summarize'](output),
        observed_cohort_mean=actual.mean(axis=0).tolist(),predicted_cohort_mean=predicted.mean(axis=0).tolist(),
        nominal_mean_price_covariance=(np.cov(actual.T,ddof=1)/len(rows)).tolist(),
        paired_price_covariance=np.cov(actual.T,ddof=1).tolist(),paired_residual_covariance=np.cov(error.T,ddof=1).tolist(),
        nominal_mean_residual_covariance_given_frozen_fit=(np.cov(error.T,ddof=1)/len(rows)).tolist(),
        price_unit='FINITE_COHORT_ONLY',validation='CALIBRATION' if cell['role']=='CALIBRATION' else 'FINITE_INDEPENDENT_POINT_EVALUATED_NOT_ADOPTED',
        qualified_for_candidate=False,total_CI=None,all_law_expected_price=None),output


def price_records(plan, cell, evaluation, frozen, source):
    """B form incl. full R domain/law; no G, no all-law promotion."""
    if evaluation['cell_id']!=cell['id'] or evaluation['role']!=cell['role'] or frozen['plan_digest']!=digest(plan):raise ValueError('PRICE_EVALUATION_CELL_BINDING')
    row=row_for(plan,cell);task=plan['menus'][cell['system']];records=[]
    for j,axis in enumerate(p.AXES):
        common=dict(**p.measurement(row,axis,saved_sampler=True),identity=deepcopy(cell['request']['identity']),
            law_binding=deepcopy(cell['request']['law_binding']),axis=axis,n_paths=cell['n'],source=source,
            expectation='FINITE_COHORT_ONLY',normalization_is_price_multiplier=False,
            uncertainty=dict(shared_calibration_group=frozen['models'][cell['system']]['shared_calibration_group'],
                nominal_coefficient_covariance=frozen['models'][cell['system']]['nominal_coefficient_covariance'],
                R_q_systematic=None,joint_covariance=None,total_CI=None,all_law_expectation=None))
        for kind,field in [('DIRECT_COMPILED_FINITE_SAMPLE','observed_cohort_mean'),('MODEL_PREDICTION','predicted_cohort_mean')]:
            value=deepcopy(common);value.update(classification=kind,cohort=('DIRECT:' if kind.startswith('DIRECT') else 'MODEL:')+cell['cohort'],
                mean={'rz_count':evaluation[field][j]},model=None)
            if kind.startswith('DIRECT'):
                value['uncertainty']=dict(paired_group='DIRECT:'+cell['id'],nominal_mean_price_covariance=evaluation['nominal_mean_price_covariance'],
                    calibration_data_group=frozen['models'][cell['system']]['shared_calibration_group'] if cell['role']=='CALIBRATION' else None,
                    coefficient_uncertainty=None,joint_covariance=None,R_q_systematic=None,total_CI=None,all_law_expectation=None)
            if kind=='MODEL_PREDICTION':value['model']=dict(name=plan['model'],q_variable='q_PF',q=row['q_PF'],
                domain=deepcopy(row['candidate']),binding=deepcopy(task['binding']),compiler=task['profile']['compiler'],
                source=source,validation_source=source,coverage=evaluation['validation'],qualified_for_candidate=False)
            records.append(value)
    return records
