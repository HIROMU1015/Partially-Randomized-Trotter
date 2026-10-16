"""One-shot H8 saved-cost fitting. No compiler, signals, or total-cost accounting."""
from fractions import Fraction
import hashlib
import json
from h8_phase_b1_contract import PROFILE,COMPILER,eligible

SCOPE='H8_B1_REUSED_B0_RUN02_B0I16_FIT_V1'
STRATA=[(m,a) for m in ('B0','B0I') for a in ('cosine','sine')]
TRAIN={(14,1),(21,1),(14,8)};HOLDOUT={(21,8)}
RESOURCES=dict(driver_AS_bytes=512*2**20,driver_RSS_bytes=256*2**20,
 observer_AS_bytes=512*2**20,observer_RSS_bytes=256*2**20,AS_bytes=512*2**20,RSS_bytes=256*2**20,
 CPU_seconds=240,worker_CPU_seconds=240,observer_CPU_seconds=120,aggregate_CPU_seconds=360,
 wall_seconds=600,aggregate_RSS_bytes=512*2**20,admission_reserve_bytes=512*2**20,
 max_workers=0,minimum_free_inodes=40000,stage_storage_bytes=128*2**20,output_bytes=64*2**20,
 log_bytes=16*2**20,output_files_max=1000)
CAPS=dict(cost_fit=4,B0_training_checks=6,B0_holdout_checks=2,B0I_anchor_checks=16,
 B2_prediction_rows=4,new_compile=0,new_signal=0,cost_reaccount=0,PR_sampling=0,
 SCF=0,DF=0,state=0,S4_compile=0,B0I_compile=0)


def require(ok,reason):
 if not ok:raise ValueError(reason)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def plan():
 return dict(schema='H8_PHASE_B1_EXECUTION_V1',scope=SCOPE,resources=RESOURCES,call_caps=CAPS,
  stratum_order=[m+'/'+a for m,a in STRATA],S2='a+b*q+d*prefix*q',S4='a+3*b*q+3*d*prefix*q',
  B0_TRAIN=[[14,1],[21,1],[14,8]],B0_HOLDOUT=[[21,8]],B0I='ALL_EIGHT_SAVED_ANCHORS_PER_AXIS',
  relative_error_denominator='MEASURED_RZ',stop_at_or_above='1/50',exact_rational_least_squares=True,
  original_B0_run02_status='STOP',saved_B0_reuse_requires_audit=True,historical_thread_compliance_certified=False,
  independent_axes=True,no_H6_coefficients=True,no_old_B0_anchors=True,one_shot=True,retry=False,
  refit=False,resume=False,mandatory_STOP=True,B2_authorized=False,C_authorized=False,next_stage_authorized=False)


def validate_anchors(rows):
 expected={(m,p,q,a) for m,a in STRATA for p in (14,21) for q in ((1,8) if m=='B0' else (1,2,4,8))}
 require(len(rows)==24 and {(r['method'],r['prefix'],r['q'],r['axis']) for r in rows}==expected,'EXACT_24_SAVED_AXES')
 for r in rows:
  require(eligible(r) and r['control']=='symmetric_directional' and r['formula']=='2nd','ANCHOR_PROVENANCE')
  require(type(r['saved_RZ']) is int and r['saved_RZ']>0,'POSITIVE_SAVED_INTEGER_RZ')
  require(r['role']==('HOLDOUT' if r['method']=='B0' and (r['prefix'],r['q']) in HOLDOUT else 'TRAIN'),'FIXED_TRAIN_HOLDOUT')
  require(r['origin_SOURCE']==('db559a518214fd27fa96e2f1451911df05c4f4a8' if r['method']=='B0' else '54ed98c5fa5b7190f77255271f4e3bec8db15630'),'ANCHOR_SOURCE')
  require(r['scalar_policy']==('ORIGINAL_CONSTANT_ONCE_NO_TAIL_IDENTITY' if r['method']=='B0' else 'ORIGINAL_PLUS_FULL_FOCK_TAIL_IDENTITY_ONCE'),'METHOD_SCALAR_POLICY')
 return rows


def solve(rows):
 """Exact 3-column normal equations; no data-dependent model selection."""
 require(len(rows)>=3,'THREE_NONCOLLINEAR_TRAINING_ANCHORS')
 X=[[Fraction(1),Fraction(r['q']),Fraction(r['prefix']*r['q'])] for r in rows]
 A=[[sum(x[i]*x[j] for x in X) for j in range(3)]+[sum(x[i]*r['saved_RZ'] for x,r in zip(X,rows))] for i in range(3)]
 for i in range(3):
  pivot=next((j for j in range(i,3) if A[j][i]),None);require(pivot is not None,'SINGULAR_DESIGN_NO_FIT')
  A[i],A[pivot]=A[pivot],A[i];v=A[i][i];A[i]=[x/v for x in A[i]]
  for j in range(3):
   if i!=j:
    v=A[j][i];A[j]=[x-v*y for x,y in zip(A[j],A[i])]
 return [A[i][3] for i in range(3)]


def predict(coefficients,p,q,pieces=1):
 a,b,d=coefficients;value=a+pieces*q*(b+d*p);require(value>0,'NONPOSITIVE_PREDICTION');return value


def check(prediction,measured):
 require(type(measured) is int and measured>0,'MEASURED_DENOMINATOR')
 error=abs(Fraction(prediction)-measured)/measured
 return dict(exact_prediction=str(prediction),measured_RZ=measured,relative_error=str(error),
  relative_error_percent=float(error*100),PASS=error<Fraction(1,50))


def execute(rows,B2_axes,*,take,write,guard):
 validate_anchors(rows);models=[]
 require(len(B2_axes)==4 and [(x['candidate_id'],x['axis']) for x in B2_axes]==[
  ('B0_p15_q1_4th','cosine'),('B0_p15_q1_4th','sine'),('B0I_p21_q8_4th','cosine'),('B0I_p21_q8_4th','sine')],'PREREGISTERED_B2_AXES')
 for method,axis in STRATA:
  guard();chosen=[r for r in rows if r['method']==method and r['axis']==axis]
  train=[r for r in chosen if r['role']=='TRAIN']
  require(len(train)==(3 if method=='B0' else 8),'FIXED_TRAIN_COUNT')
  take('cost_fit');coefficients=solve(train)
  model=dict(method=method,axis=axis,coefficients=[str(x) for x in coefficients],rank=3,
   TRAIN_ids=[r['id'] for r in train],checks=[],status='PENDING_AXIS_CHECKS',
   S4_status='UNVALIDATED_SCHEDULE_COMPOSITION_HYPOTHESIS',full_grid_guaranteed=False)
  write(method+'_'+axis+'_partial.json',model)
  for r in train+[r for r in chosen if r['role']=='HOLDOUT']:
   guard();key='B0I_anchor_checks' if method=='B0I' else ('B0_holdout_checks' if r['role']=='HOLDOUT' else 'B0_training_checks')
   take(key);diagnostic=dict(id=r['id'],role=r['role'],**check(predict(coefficients,r['prefix'],r['q']),r['saved_RZ']))
   model['checks'].append(diagnostic);write(method+'_'+axis+'_partial.json',model)
   require(diagnostic['PASS'],'AXIS_MODEL_ERROR_GE_2_PERCENT')
  model['status']='S2_ANCHORS_AND_REGISTERED_HOLDOUT_PASS';models.append(model)
  write(method+'_'+axis+'_model.json',model)
 frozen=dict(schema='H8_FROZEN_COST_MODEL_V1',models=models,S2=plan()['S2'],S4=plan()['S4'],
  profile_sha256=PROFILE,compiler_hash=COMPILER,H6_coefficients_transferred=False,
  old_B0_used=False,S4_measured_validation=False,compiled_grid_guaranteed=False,
  rigorous_bias_or_PR_population_guarantee=False,B2_authorized=False,C_authorized=False,next_stage_authorized=False)
 model_sha=digest(frozen);write('frozen_models.json',frozen);predictions=[]
 for row in B2_axes:
  guard();take('B2_prediction_rows');method,p,q,_=row['candidate_id'].split('_');p=int(p[1:]);q=int(q[1:])
  m=next(x for x in models if x['method']==method and x['axis']==row['axis'])
  predictions.append(dict(candidate_id=row['candidate_id'],axis=row['axis'],exact_predicted_RZ=str(predict(list(map(Fraction,m['coefficients'])),p,q,3)),
   model_sha256=model_sha,status='MODEL_ESTIMATE_NOT_MEASURED',compile_authorized=False))
 write('B2_predictions_UNAPPROVED.json',dict(axes=predictions,model_sha256=model_sha,B2_authorized=False))
 return dict(models=len(models),model_sha256=model_sha,all_four_strata_PASS=True,S4_validation=False,
  B2_authorized=False,C_authorized=False,next_stage_authorized=False)
