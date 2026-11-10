"""H6 V2 saved-scalar corrections; no circuit, sampling, or V1 recount calls."""
import ast
from collections import defaultdict
import math
import subprocess
import numpy as np

V1='a4d743cb5007d5fc6b6e93c3b43a4701ff956ac4'
H4='cefc01622a3cd75dd16d30de7b9fb68274f3eddd'
AXES=('cosine','sine')
V1_MODEL='H6_DF19_M5_SOURCE_NATIVE_BASIS_CANCEL_ZFOLD_V1'


def design(pairs,model):
    if model=='Moffset':return np.ones((len(pairs),1))
    if model=='Mbasis':return np.column_stack((np.ones(len(pairs)),[p['basis_groups'] for p in pairs]))
    raise ValueError('Fixed H6 correction design only')


def reuse(root):
    scope=dict(np=np,defaultdict=defaultdict,AXES=AXES,V1_MODEL=V1_MODEL,design=design)
    raw=subprocess.check_output(['git','show',H4+':src/trottertracks/h4_df10_price_v2.py'],cwd=root)
    nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('pair_rows','evaluate','summarize')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),H4+':h4_df10_price_v2.py','exec'),scope)
    raw=subprocess.check_output(['git','show',H4+':src/trottertracks/track_d_c7_4_models.py'],cwd=root)
    node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='fit')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),H4+':fixed_fit','exec'),scope)
    return scope


def pairs(rows,helpers):
    converted=[];family={r['candidate_id']:r['family'] for r in rows}
    for row in rows:
        r=dict(row,method='PF' if row['family'].startswith('PF') else 'PR',q=row['q_PF'])
        converted.append(r)
    ps=helpers['pair_rows'](converted)
    for p in ps:p['family']=family[p['candidate_id']]
    return ps


def calibrate(ps,model,fit):
    x=design(ps,model);y=np.asarray([p['actual'] for p in ps])-np.asarray([p['baseline'] for p in ps])
    try:beta=fit(x,y,np.asarray([p['q'] for p in ps]))
    except ValueError as error:return dict(status='EVALUATION_UNAVAILABLE',model=model,coefficients=None,reason=str(error),rank=int(np.linalg.matrix_rank(x)))
    if not np.isfinite(beta).all():return dict(status='EVALUATION_UNAVAILABLE',model=model,coefficients=None,reason='NONFINITE_COEFFICIENT')
    rank=x.shape[1];dof=len(ps)-rank;covariance=None
    if dof>0:
        e=y-x@beta;sigma=e.T@e/dof
        covariance=np.kron(np.linalg.inv(x.T@x),sigma).tolist()
    return dict(status='FITTED',model=model,coefficients=beta.tolist(),rank=rank,training_paths=len(ps),degrees_of_freedom=dof,features=['intercept'] if model=='Moffset' else ['intercept','tail_basis_groups'],response_axes=list(AXES),nominal_OLS_coefficient_covariance=covariance,covariance_order='feature_then_axis',coefficient_covariance_qualification='Nominal IID homoskedastic residual diagnostic; not total model or finite-bit-law uncertainty',q_slope_identified=False)


def campaign(development,known,auxiliary,helpers):
    train=pairs(development,helpers);check=pairs(known,helpers);aux=pairs(auxiliary,helpers)
    calibrations=[];rows=[];selected={};summaries=[]
    for family in ('PF_FULL19','PR_A','PR_B'):
        ps=[p for p in train if p['family']==family];ks=[p for p in check if p['family']==family]
        models=('Moffset',) if family=='PF_FULL19' else ('Moffset','Mbasis')
        local=[]
        for role,points in [('TRAINING',ps),('KNOWN_V1_USED',ks)]:local.extend(helpers['evaluate'](points,role,'M0'))
        for model in models:
            fit=calibrate(ps,model,helpers['fit']);fit.update(family=family);calibrations.append(fit)
            for role,points in [('TRAINING',ps),('KNOWN_V1_USED',ks)]:local.extend(helpers['evaluate'](points,role,model,fit))
        local_summaries=helpers['summarize'](local)
        for s in local_summaries:s['family']=family
        candidates=[]
        for model in ('M0',*models):
            ss=[s for s in local_summaries if s['role']=='KNOWN_V1_USED' and s['model']==model]
            if len(ss)==2 and all(s['metrics'] is not None and s['negative_predictions']==0 for s in ss):
                candidates.append((max(s['metrics']['RMSE'] for s in ss),0 if model=='M0' else 1 if model=='Moffset' else 2,model))
        if not candidates:raise ValueError('No evaluable frozen candidate')
        choice=min(candidates)[2];selected[family]=dict(model=choice,calibration=next((f for f in calibrations if f['family']==family and f['model']==choice),None),selection='Exploratory minimum maximum-axis known-data RMSE; tie favors fewer parameters; no percentage threshold')
        rows.extend(local);summaries.extend(local_summaries)
    for model in ('M0','Moffset'):
        calibration=next((f for f in calibrations if f['family']=='PF_FULL19' and f['model']==model),None)
        extra=helpers['evaluate'](aux,'AUX_CHANGED_DELTA',model,calibration);rows.extend(extra)
        for s in helpers['summarize'](extra):s['family']='PF_AUX_Q2';summaries.append(s)
    return dict(calibrations=calibrations,selected=selected,summaries=summaries,known_data_comparison='EXPLORATORY_V1_ALREADY_USED',formal_percentage_threshold=None,full_provider_qualified=False),rows
