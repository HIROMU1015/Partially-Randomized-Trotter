"""Freeze q1/q2 Affine before q4; compare without q4 refit."""
import csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.h6_df19_price_v2 import reuse,V1
PUBLIC=ROOT/'docs/research/h6_df19_price_v2_20261110'
AXES=('cosine','sine')

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def saved(name):return json.loads(subprocess.check_output(['git','show',V1+':docs/research/h6_df19_price_v1_20261110/'+name],cwd=ROOT))
def mean_cov(a,deterministic=False):
    a=np.asarray(a,float);return dict(n=len(a),mean=a.mean(axis=0).tolist(),path_covariance=None if deterministic else np.cov(a.T,ddof=1).tolist(),mean_covariance=None if deterministic else (np.cov(a.T,ddof=1)/len(a)).tolist(),population_guarantee=False)

def freeze():
    q2=read(PUBLIC/'Q2_DIRECT.json');dev=saved('DEVELOPMENT.json')['rows'];known=saved('HOLDOUT.json')['rows'];families={};h=reuse(ROOT)
    for family in ('PF_FULL19','PR_B'):
        c1=[r for r in (known if family=='PF_FULL19' else dev) if r['family']==family]
        indices=sorted({r['path_index'] for r in c1});a1=[[next(r['compiled_rz'] for r in c1 if r['path_index']==i and r['axis']==axis) for axis in AXES] for i in indices]
        rs=[r for r in q2['rows'] if (r['candidate']['method']=='FULL')==(family=='PF_FULL19')];a2=[[r['prices'][axis]['rz_count'] for axis in AXES] for r in rs]
        c1s=mean_cov(a1,family=='PF_FULL19');c2s=mean_cov(a2,family=='PF_FULL19')
        coefficients=h['fit'](np.array([[1,1],[1,2]],float),np.array([c1s['mean'],c2s['mean']]),np.array([1,2]))
        v1=c1s['mean_covariance'];v2=c2s['mean_covariance'];covariance=None;prediction_cov=None
        if v1 is not None:
            v1=np.array(v1);v2=np.array(v2)
            covariance=np.block([[4*v1+v2,-2*v1-v2],[-2*v1-v2,v1+v2]]).tolist();prediction_cov=(4*v1+9*v2).tolist()
        families[family]=dict(model='AFFINE_FIXED_DELTA',formula='intercept + slope*q',coefficients=coefficients.tolist(),features=['intercept','q'],response_axes=list(AXES),rank=2,q1=c1s,q2=c2s,q1_cohort='SAVED_DETERMINISTIC_T1.6' if family=='PF_FULL19' else 'SAVED_EXPLORATION_N9_T1.6',q2_cohort='Q2_CALIBRATION',nominal_coefficient_covariance=covariance,covariance_order='feature_then_axis',q4_mean_prediction=(np.array([1,4])@coefficients).tolist(),q4_predicted_mean_covariance_under_independent_IID_cohorts=prediction_cov,systematic_error_bound=None,model_law_identified=False)
    write(PUBLIC/'Q4_FROZEN_MODELS.json',dict(analysis_script_sha256=sha(__file__),q2_direct_sha256=sha(PUBLIC/'Q2_DIRECT.json'),q4_prices_observed=False,q4_execution_script_sha256=sha(ROOT/'scripts/run_h6_df19_price_v2_q4.py'),selected_V2=read(PUBLIC/'SAVED_CORRECTIONS.json')['selected'],families=families,comparison_metrics=['bias','MAE','RMSE','relative_bias','relative_RMSE','maximum_absolute_error','maximum_absolute_relative_error','path_residual_SD','paired_mean_covariance'],prediction_unit=dict(M0='individual path',V2='individual path, frozen constant offset',Affine='cohort mean, individual path variation remains'),q4_refit=False,qualification='Two-point affine interpolation/extrapolation; no lack-of-fit DOF; finite-cohort covariance does not include systematic'))
    print(json.dumps(families),flush=True)


def metrics(actual,predicted):
    a=np.asarray(actual,float);p=np.asarray(predicted,float);e=a-p;rel=e/a
    return dict(n=len(a),observed_mean=float(a.mean()),predicted_mean=float(p.mean()),bias=float(e.mean()),MAE=float(np.mean(abs(e))),RMSE=float(np.sqrt(np.mean(e*e))),relative_bias=float(rel.mean()),relative_RMSE=float(np.sqrt(np.mean(rel*rel))),maximum_absolute_error=float(max(abs(e))),maximum_absolute_relative_error=float(max(abs(rel))),path_residual_SD=float(np.std(e,ddof=1)) if len(a)>1 else None,total_prediction_CI=None)


def summarize():
    frozen=read(PUBLIC/'Q4_FROZEN_MODELS.json');summary=[];rows=[];covariances=[]
    require_hash=frozen['analysis_script_sha256'];assert require_hash==sha(__file__),'Analysis changed after freeze'
    for q in (2,4):
        d=read(PUBLIC/('Q'+str(q)+'_DIRECT.json'))
        for family in ('PF_FULL19','PR_B'):
            group=[r for r in d['rows'] if (r['candidate']['method']=='FULL')==(family=='PF_FULL19')]
            calibration=frozen['selected_V2'][family]['calibration'];offset=np.asarray(calibration['coefficients'])[0]
            affine=np.array([1,q])@np.array(frozen['families'][family]['coefficients'])
            actual=np.array([[r['prices'][axis]['rz_count'] for axis in AXES] for r in group],float)
            base=np.array([[r['predictions'][axis]['structural_rz'] for axis in AXES] for r in group],float)
            predictions=dict(M0=base,V2=base+offset,Affine=np.tile(affine,(len(group),1)))
            for model,pred in predictions.items():
                for ai,axis in enumerate(AXES):
                    summary.append(dict(q_PF=q,T=1.6*q,family=family,model=model,axis=axis,role='CALIBRATION' if q==2 else 'NEW_Q_INDEPENDENT_VERIFICATION',prediction_unit='COHORT_MEAN_REPEATED_FOR_DESCRIPTIVE_PATH_ERRORS' if model=='Affine' else 'INDIVIDUAL_PATH',metrics=metrics(actual[:,ai],pred[:,ai])))
                    for i,r in enumerate(group):rows.append(dict(q_PF=q,T=1.6*q,family=family,candidate_id=r['candidate']['id'],cohort=r['cohort'],path_index=r['index'],axis=axis,model=model,compiled_rz=int(actual[i,ai]),predicted_rz=float(pred[i,ai]),residual_rz=float(actual[i,ai]-pred[i,ai]),relative_error=float((actual[i,ai]-pred[i,ai])/actual[i,ai])))
                if len(group)>1:
                    e=actual-pred
                    covariances.append(dict(q_PF=q,family=family,model=model,nominal_path_error_covariance=np.cov(e.T,ddof=1).tolist(),nominal_mean_error_covariance_given_frozen_fit=(np.cov(e.T,ddof=1)/len(group)).tolist(),qualification='Paired X/Y; finite n8; excludes fit uncertainty/systematic. Affine mean uncertainty is separately recorded in frozen coefficients.'))
            covariances.append(dict(q_PF=q,family=family,model='DIRECT_PRICE',statistics=mean_cov(actual,family=='PF_FULL19')))
    out=dict(analysis_script_sha256=sha(__file__),q4_frozen_models_sha256=sha(PUBLIC/'Q4_FROZEN_MODELS.json'),summaries=summary,paired_covariances=covariances,q4_refit=False,full_rpe_resources=None,long_q_gt4_qualified=False,all_sampling_law_expected_price_certified=False,mandatory_stop=True)
    write(PUBLIC/'Q_VALIDATION.json',out)
    with (PUBLIC/'Q_PATH_RESIDUALS.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    for s in summary:
        if s['axis']=='cosine':print(json.dumps(s),flush=True)


if __name__=='__main__':
    if sys.argv[1]=='freeze':freeze()
    elif sys.argv[1]=='summarize':summarize()
    else:raise ValueError('freeze/summarize')
