"""q64-only domain adapter; scientific builders/checkers remain unchanged."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

_path=Path(__file__).with_name('track_d_c7_3_q32.py')
_spec=importlib.util.spec_from_file_location(__name__+'_q32_private',_path)
_domain=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_domain
_spec.loader.exec_module(_domain)
api,calibration,digest=_domain.api,_domain.calibration,_domain.digest
BASE='8828b298975c88a2c0d1fd57ff6ed905233840f5'
Q,SAMPLES,CUT,CX_CANDIDATE=64,8,32,-14128
COHORT='track_d_c8_1_q64_independent_price_v1'
MASTER=2026110801
DOMAIN=COHORT+':fresh_independent_holdout:q64'
METRICS=calibration.METRICS


def binding(sample):
    from .c1_stage3_connection import trajectory_seed
    if type(sample) is not int or not 0<=sample<SAMPLES:raise ValueError('C8_FIXED_SAMPLE')
    seeds=[trajectory_seed((MASTER,DOMAIN,BASE),'proposed',api.c3.paired.ENDPOINT,4,
        'C8_1_PAIRED_PRICE_ONLY',sample,i) for i in range(Q)]
    return dict(q=Q,sample=sample,sample_id=f'q64_sample{sample:02d}',seed_domain=DOMAIN,
        seeds=seeds,cut_after_complete_step=CUT)


def seed_plan():
    root=Path(__file__).resolve().parents[2]
    paths=[_domain.CALDIR+'/seed_plan.json',
        'artifacts/track_d_c6b_q8/2026-11-07/technical/seed_plan.json',
        'artifacts/track_d_c7_1_q16/2026-11-07/technical/seed_plan.json',
        'artifacts/track_d_c7_3_q32/2026-11-07/technical/seed_plan.json']
    packets=[json.loads((root/p).read_text()) for p in paths]
    prior={seed for p in packets for row in p['generation_order'] for seed in row['seeds']}
    # Same existing single-occurrence price-cohort trajectory_seed convention.
    from .c1_stage3_connection import trajectory_seed
    prior.update(trajectory_seed((2026110304,api.c3.paired.COHORT,api.c3.paired.COHORT_SOURCE),
        'proposed',api.c3.paired.ENDPOINT,-1,'NEAR_PRACTICAL_PR_PRICE_ONLY',i,0) for i in range(32))
    rows=[binding(i) for i in range(8)];seeds=[v for r in rows for v in r['seeds']]
    if len(set(seeds))!=512 or set(seeds)&prior:raise ValueError('SEED_DOMAIN_COLLISION')
    return dict(cohort=COHORT,q=64,cut=32,samples=8,master=MASTER,master_tuple=[MASTER,DOMAIN,BASE],
        generation_order=rows,previous_distinct_seeds=len(prior),previous_seed_overlap=0,
        previous_seed_plan_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        seed_function='existing trajectory_seed SHA256(repr(tuple)) first8 little endian',
        seed_key_fields=['master_tuple','proposed',api.c3.paired.ENDPOINT,4,'C8_1_PAIRED_PRICE_ONLY','sample','occurrence'],
        PRNG='numpy1.26.4 PCG64',sampler='existing configure / sample_occurrence / iter_sample_rte_events',
        same_path_methods_XY=True,ideal_IID_guaranteed=False,implemented_law_distance=None)


for _key in ('BASE','Q','SAMPLES','CUT','CX_CANDIDATE','COHORT','MASTER','DOMAIN'):
    setattr(_domain,_key,globals()[_key]);setattr(_domain._shared,_key,globals()[_key])
_domain.binding=binding;_domain._shared.binding=binding


@dataclass(frozen=True)
class Q64Request(_domain.Q32Request):
    q:int=64
    cut_after_complete_step:int=32
    cohort:str=COHORT
    @property
    def ordered_samples(self):return tuple(f'q64_sample{self.sample:02d}_step{i:02d}' for i in range(64))


_domain._shared.Q16Request=Q64Request
sample_record=_domain.sample_record
request_from_record=_domain.request_from_record
sampling_context=_domain.sampling_context
wrapper_features=_domain.wrapper_features
raw_features=_domain.raw_features


def prepare_fresh(context,record,reference_context):
    bundle,request=_domain.prepare_fresh(context,record,reference_context)
    bundle.identity['fixture_selection']='pre-fixed q64 independent domain, eight fresh paths, cut32; pseudo-independent only'
    return bundle,request


def price_record(request,**kwargs):
    row=_domain.price_record(request,**kwargs);row['estimate_kind']='exploratory_independent_q64_holdout_compiled'
    return row


def paired_statistics(records):
    row=_domain.paired_statistics(records);row['status']='COMPLETE_FINITE_Q64_HOLDOUT_DESCRIPTIVE'
    return row


def cx_candidate_residuals(summary):
    result=_domain._shared.cx_candidate_residuals(summary)
    for r in result['rows']:r['residual_definition']='observed DeltaCX + 14128'
    result['global_formula_status']='CX_GLOBAL_FORMULA_UNQUALIFIED'
    return result


def model_packet(root):
    """Fixed published q64 forecasts, not a new fit or model selection."""
    import numpy as np
    commit='64072ba81bb70db1df1b34e94adb08e78926f24b'
    directory='artifacts/track_d_c7_4_models/2026-11-07/run02/'
    references={}
    def read(path):
        data=subprocess.check_output(['git','show',commit+':'+path],cwd=root)
        references[path]=dict(commit=commit,sha256=hashlib.sha256(data).hexdigest())
        return json.loads(data)
    models={}
    for scenario,name in [('A','affine'),('S','sqrt_lane')]:
        path=directory+'model_'+name+'.json';d=read(path);future=d['future'];index=future['q'].index(64)
        labels=d['quantity_labels'];x=np.array([1.,64.] if scenario=='A' else [1.,64.,8.])
        transform=np.kron(x.reshape(1,-1),np.eye(36))
        models[scenario]=dict(name=name,source_path=path,design=d['design'],quantity_labels=labels,
            prediction=future['predictions'][index],calibration_covariance=(transform@np.array(d['coefficient_covariance'])@transform.T).tolist(),
            qualification='saved stratified pseudo-IID bootstrap calibration only; no bias/law/long-q coverage')
    read(directory+'model_frozen_C6a.json')
    f=read(directory+'frozen_C6a_future.json');labels=f['quantity_labels'];old=f['saved_parameter_covariance'];oldlabels=[tuple(r) for r in old['labels']]
    transform=np.zeros((36,48))
    for i,(m,a,g) in enumerate(labels):
        methods=[(m,1)] if m!='paired_delta' else [('reference',1),('symmetric_directional',-1)]
        for method,sign in methods:
            transform[i,oldlabels.index((method,a,g,'slope'))]+=sign*64
            transform[i,oldlabels.index((method,a,g,'intercept'))]+=sign
    models['F']=dict(name='frozen_C6a',source_path=directory+'frozen_C6a_future.json',design=['1','q'],
        quantity_labels=labels,prediction=f['predictions'][f['q'].index(64)],
        calibration_covariance=(transform@np.array(old['matrix'])@transform.T).tolist(),qualification=old['formula'])
    # Auxiliary versions remain read-only references, not future-law price inputs.
    for name in ('basis_runs','lane_runs'):read(directory+'model_'+name+'.json')
    return dict(q=64,primary='A',models={s:models[s] for s in ('A','F','S')},references=references,
        priority=['cx_count','total_depth'],difference='Reference minus symmetric_directional',
        integer_CX_candidate=-14128,price_width_diagnostic_used=False,confirmatory_PASS=False,
        new_fit=False,bias_bound=None,law_bound=None)


def diagnose(summary,packet):
    import math
    rows=[]
    for scenario,model in packet['models'].items():
        labels=[tuple(v) for v in model['quantity_labels']]
        for stat in summary['rows']:
            if stat['metric']=='peak_qubits':continue
            for method,field in [('symmetric_directional','symmetric'),('reference','reference'),('paired_delta','paired_delta')]:
                i=labels.index((method,stat['axis'],stat['metric']));value=stat[field]
                rows.append(dict(scenario=scenario,method=method,axis=stat['axis'],metric=stat['metric'],
                    n=8,prediction=model['prediction'][i],observed_mean=value['mean'],
                    observed_minus_prediction=value['mean']-model['prediction'][i],
                    observed_sample_SD=value['sample_SD'],observed_min=value['min'],observed_max=value['max'],
                    saved_calibration_variance=model['calibration_covariance'][i][i],
                    nominal_mean_SD=value['sample_SD']/math.sqrt(8),confidence_interval=None))
    return dict(status='EXPLORATORY_INDEPENDENT_Q64_MODEL_DIAGNOSTICS',rows=rows,
        total_model_bias=None,ideal_IID_guaranteed=False,confirmatory_PASS=False,model_refits=0,
        note='fresh domain not used in fixed models; conditional calibration/holdout variation, no confidence guarantee')
