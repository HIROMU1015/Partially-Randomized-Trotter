"""C7-4 only: saved-price reanalysis. No sampling, circuit build or compile."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess

import numpy as np
from .track_d_c6a_pilot import describe

METRICS = ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')
METHODS = ('symmetric_directional','reference')
AXES = ('X','Y')
RUNS = (
 ('track_d_c6a_paired_price/2026-11-06/run01','track-d-c6a-paired-price-pilot-20261106',
  'e9d4e2f602d5f9e014507ca7e391719bab017b88',((2,16),(4,16))),
 ('track_d_c6b_q8/2026-11-07/run01_accelerated_completion','track-d-c6b-q8-exploratory-20261107',
  'fed5f458be2f625de4e1b8c98b8cc5da6e551b14',((8,32),)),
 ('track_d_c7_1_q16/2026-11-07/run01','track-d-c7-1-q16-exploratory-20261107',
  '86c5c9d48dbddc085f388b6521be8cd92ee49d0d',((16,16),)),
 ('track_d_c7_3_q32/2026-11-07/run01_accelerated_completion','track-d-c7-3-q32-accelerated-20261107',
  'acea43987edae0d5eb0c186a89aa8cd7139fbd7b',((32,8),)),
)
PACKET='1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db'
ENDPOINT='a8813acaa6dd67abf843c50b103c29b84f015e4f3cabdb636e2a2db6401a863a'
LAW='c347d7f2ec2ed40adb6f2490960dbebf693862cddcce96418c1027c48a68d6a5'
LABELS=[(m,a,g) for m in METHODS for a in AXES for g in METRICS]+[
 ('paired_delta',a,g) for a in AXES for g in METRICS]
CANDIDATES=('affine','sqrt_lane','basis_runs','lane_runs')
FUTURE=(64,128,256,512,1024,2048,4096)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def event_features(record):
    """Aggregate saved application descriptors, never emit gates or events."""
    q=record['q'];runs=[];work=[];signs=Counter();supports=Counter();zeros=0
    for step in record['ordered_events']:
        previous=None;n=0;w=0
        for event in step:
            zeros+=int(event['rotation_angle']==0)
            for app in event['application_sequence']:
                basis=app['basis_id'];n+=basis!=previous;previous=basis
                support=app['diagonal_pauli_support'];w+=max(1,2*len(support)-1)
                supports[str(len(support))]+=1;signs[str(app['coefficient_sign'])]+=1
        runs.append(n);work.append(w)
    left,right=sum(runs[:q//2]),sum(runs[q//2:])
    return dict(q=q,sample_id=record['sample_id'],basis_runs=sum(runs),
        different_basis_transitions=sum(runs)-q,lane_A_runs=right,lane_B_runs=left,
        lane_max_runs=max(left,right),lane_imbalance=abs(left-right),
        support_work=sum(work),support_size_counts=dict(supports),sign_counts=dict(signs),
        zero_angles=zeros,events=sum(map(len,record['ordered_events'])),
        step_boundaries=q-1,reference_internal_boundaries=q-2,global_cut_losses=1)


def ingest(root,evidence_root):
    rows=[];bindings={};features=[];checks=[];private={}
    for directory,task,commit,groups in RUNS:
        folder=root/'artifacts'/directory;local=evidence_root/task/'artifacts'/directory/'local'
        source=read(folder/'SOURCE.json');result=read(folder/'result.json')
        if result['status']!='TECHNICAL_PASS_MANDATORY_STOP' or not read(folder/'STOP.json')['mandatory_stop']:
            raise ValueError('SAVED_RUN_NOT_COMPLETE:'+directory)
        leaves=['prices.json','SOURCE.json','STOP.json','result.json','event_bindings.json','seed_plan.json','paired_statistics.json']
        if groups[0][0]==2:leaves+=['exploratory_affine.json','C6b_planning_proposal.json']
        if groups[0][0]==32:leaves+=['structural_feature_summary.json','exploratory_diagnostics.json']
        for leaf in leaves:
            p=folder/leaf;relative=str(p.relative_to(root))
            if p.read_bytes()!=subprocess.check_output(['git','show',commit+':'+relative],cwd=root):
                raise ValueError('SAVED_PUBLIC_BYTES_CHANGED:'+relative)
            bindings[relative]=dict(commit=commit,sha256=sha(p))
        prices=read(folder/'prices.json');table={(r['q'],r['sample_id'],r['method'],r['axis']):r for r in prices}
        expected={(q,f'q{q}_sample{i:02d}',m,a) for q,n in groups for i in range(n) for m in METHODS for a in AXES}
        if len(table)!=len(prices) or set(table)!=expected:raise ValueError('COMPLETE_FIXED_COHORT')
        event_index={r['sample_id']:r for r in read(folder/'event_bindings.json')}
        for q,n in groups:
            for i in range(n):
                sid=f'q{q}_sample{i:02d}';p=local/'events'/(sid+'.json')
                if sha(p)!=event_index[sid]['record_sha256']:raise ValueError('SAVED_EVENT_HASH:'+sid)
                event=read(p)
                if (event['q'],event['cut_after_complete_step'],event['endpoint'],event['input_packet_sha256'],event['law_sha256'])!=(q,q//2,ENDPOINT,PACKET,LAW):
                    raise ValueError('SAVED_EVENT_IDENTITY:'+sid)
                features.append(event_features(event));private[sid]=local/sid
                for m in METHODS:
                    for a in AXES:
                        r=table[q,sid,m,a];request=r['request']
                        if (request['endpoint'],request['input_packet_sha256'],request['law_sha256'],request['cut_after_complete_step'])!=(ENDPOINT,PACKET,LAW,q//2):
                            raise ValueError('PRICE_IDENTITY:'+sid)
                        if (r['compiler']!={'qiskit':'1.3.0','optimization_level':1,'seed_transpiler':17,
                            'basis_gates':['rz','sx','x','cx'],'approximation_degree':1.0,'qubits_initially_zero':False}
                            or r['precision']!={'component_abs':1e-10,'observable_frobenius':2e-10,'rtol':0,'global_phase_fit':False}
                            or r['topology']!='free' or r['cost_scope']!='single_hadamard_interrogation_without_state_preparation'
                            or r['qualification']['postcheck']!='PASS'):
                            raise ValueError('PRICE_COMPARISON_CONDITIONS:'+sid)
                        rows.append(dict(q=q,sample_id=sid,method=m,axis=a,metrics=r['metrics'],peak_qubits=r['peak_qubits'],
                            execution_source=r['execution_source'],sampling_source=event['execution_source'],event_sha256=sha(p)))
                        c=read(local/sid/(m+'_'+a+'_check.json'))
                        if c['status']!='PASS':raise ValueError('SAVED_CHECK_NOT_PASS:'+sid)
                        checks.append(dict(q=q,sample_id=sid,method=m,axis=a,checks=c['checks'],
                            reference_leakage=c.get('reference_leakage')))
                raw=read(local/sid/'precompile_checks.json')
                if raw['status']!='PASS':raise ValueError('RAW_CHECK_NOT_PASS:'+sid)
                checks.append(dict(q=q,sample_id=sid,method='raw',axis='both',checks=raw['checks'],
                    reference_leakage=raw.get('reference_leakage')))
    if len(rows)!=352 or len(features)!=88:raise ValueError('FIXED88')
    return rows,features,checks,bindings,private


def matrix(rows,features):
    table={(r['sample_id'],r['method'],r['axis']):r for r in rows}
    def value(sid,m,a,g):return table[sid,m,a]['metrics'][g]
    y=[]
    for f in features:
        sid=f['sample_id'];v=[]
        for m,a,g in LABELS:
            v.append(value(sid,m,a,g) if m!='paired_delta' else value(sid,'reference',a,g)-value(sid,'symmetric_directional',a,g))
        y.append(v)
    return np.asarray(y,float)


def design(features,name):
    q=np.array([f['q'] for f in features],float)
    if name=='affine':return np.column_stack((np.ones(len(q)),q))
    z=np.sqrt(q) if name=='sqrt_lane' else np.array([f['basis_runs' if name=='basis_runs' else 'lane_max_runs'] for f in features],float)
    return np.column_stack((np.ones(len(q)),q,z))


def fit(x,y,q):
    """Equal-q empirical objective; unequal n and paired response rows retained."""
    w=np.array([1/np.count_nonzero(q==v) for v in q]);sw=np.sqrt(w)
    b,res,rank,s=np.linalg.lstsq(x*sw[:,None],y*sw[:,None],rcond=None)
    if rank!=x.shape[1]:raise ValueError('UNIDENTIFIABLE_MODEL')
    return b


def residual_summary(y,pred,q):
    rows=[]
    for group in sorted(set(q)):
        select=q==group
        for j,label in enumerate(LABELS):
            e=y[select,j]-pred[select,j]
            rows.append(dict(q=int(group),quantity=list(label),n=int(select.sum()),
                observed_mean=float(y[select,j].mean()),prediction_mean=float(pred[select,j].mean()),
                mean_error=float(e.mean()),RMSE=float(np.sqrt(np.mean(e**2))),
                residual_SD=float(np.std(e,ddof=1)),max_abs_error=float(np.max(abs(e)))))
    return rows


def integrate(rows,features,y):
    q=np.array([f['q'] for f in features]);byq=[]
    for v in sorted(set(q)):
        select=q==v;means=[]
        for j,label in enumerate(LABELS):means.append(dict(quantity=list(label),**describe(y[select,j].tolist())))
        byq.append(dict(q=int(v),n=int(select.sum()),quantities=means,
            sample_covariance=np.cov(y[select],rowvar=False,ddof=1).tolist(),
            covariance_rank_limit=int(select.sum())-1))
    cx=[int(row[LABELS.index(('paired_delta','X','cx_count'))]) for row in y]
    residual=[d-(208-224*int(v)) for d,v in zip(cx,q)]
    size_identity=[]
    for a in AXES:
        delta=lambda g:y[:,LABELS.index(('paired_delta',a,g))]
        size_identity.append(dict(axis=a,max_abs_residual=float(np.max(abs(delta('circuit_size')-delta('rz_count')-delta('cx_count')-192)))))
    return dict(status='88_SAVED_PATH_REANALYSIS',by_q=byq,quantity_labels=LABELS,ddof=1,
        XY_independent=False,ideal_IID_guaranteed=False,observed_support_is_population_support=False,
        CX_integer_candidate=dict(formula='208-224*q',n=88,residuals=residual,all_zero=not any(residual),all_path_theorem=False),
        size_minus_RZ_minus_CX_candidate=dict(formula='192',rows=size_identity,all_path_theorem=False,
            selection='exploratory relation suggested by saved q means, not an independent hypothesis test'),
        peak_qubits=dict(symmetric=9,reference=17,additive_round_cost=False))


def old_predictions(root,features):
    path=root/'artifacts/track_d_c6a_paired_price/2026-11-06/run01/exploratory_affine.json'
    model=read(path);q=np.array([f['q'] for f in features],float)
    result=[]
    for m,a,g in LABELS:
        def predict(method):
            r=next(r for r in model['models'][method] if r['metric']==g and r['axis']=={'X':'cosine','Y':'sine'}[a])
            return r['intercept']+r['slope']*q
        result.append(predict(m) if m!='paired_delta' else predict('reference')-predict('symmetric_directional'))
    return np.array(result).T


def analyze_models(root,features,y):
    q=np.array([f['q'] for f in features]);rng=np.random.default_rng(2026110704)
    groups=[np.flatnonzero(q==v) for v in sorted(set(q))]
    draws=[np.concatenate([rng.choice(g,size=len(g),replace=True) for g in groups]) for _ in range(256)]
    results={};old=old_predictions(root,features)
    results['frozen_C6a']=dict(parameters=2,fit_data='original q2/q4 only; unchanged',
        residuals=residual_summary(y,old,q),independent_holdout_for_C7_4=False)
    for name in CANDIDATES:
        x=design(features,name);b=fit(x,y,q);pred=x@b;fold=np.empty_like(y)
        for held in sorted(set(q)):
            train=q!=held;test=~train
            fold[test]=x[test]@fit(x[train],y[train],q[train])
        boot=np.stack([fit(x[index],y[index],q[index]) for index in draws])
        cov=np.cov(boot.reshape(len(draws),-1),rowvar=False,ddof=1)
        model=dict(parameters=x.shape[1],coefficients=b.tolist(),coefficient_order='design column, then quantity',
            design=['1','q']+([] if name=='affine' else [{'sqrt_lane':'sqrt(q)','basis_runs':'basis_runs','lane_runs':'max(A_runs,B_runs)'}[name]]),
            quantity_labels=LABELS,coefficient_covariance=cov.tolist(),bootstrap_replicates=256,
            bootstrap_scope='stratified saved whole-path resampling, shared indices for every method/axis/resource; pseudo-IID conditional only',
            residuals=residual_summary(y,pred,q),leave_one_q_out=residual_summary(y,fold,q),
            residual_by_path=(y-pred).tolist(),fold_residual_by_path=(y-fold).tolist(),
            independent_validation=False,qualified_coverage=False)
        if name in ('affine','sqrt_lane'):
            future_features=[dict(q=v) for v in FUTURE];xf=design(future_features,name)
            future_boot=np.einsum('qp,bpd->bqd',xf,boot)
            paired=[j for j,label in enumerate(LABELS) if label[0]=='paired_delta']
            model['future']=dict(q=list(FUTURE),predictions=(xf@b).tolist(),quantity_labels=LABELS,
                nominal_bootstrap_SD=future_boot.std(axis=0,ddof=1).tolist(),
                paired_cross_round_covariance=np.cov(future_boot[:,:,paired].reshape(256,-1),rowvar=False,ddof=1).tolist(),
                covariance_order='q then paired quantity',bias_bound=None,law_error_bound=None,
                actual_compile_prices=False,guaranteed_expectation=False)
        else:model['future']=dict(numeric_prediction=None,missing='future joint feature law and expected maximum; max(E) is not E[max]')
        results[name]=model
    return results


def numerical_trends(checks):
    rows=[]
    for q in (2,4,8,16,32):
        labels=sorted({c['label'] for r in checks if r['q']==q for c in r['checks']})
        for label in labels:
            for method in ('raw',*METHODS):
                values=[c for r in checks if r['q']==q and r['method']==method for c in r['checks'] if c['label']==label]
                if values:rows.append(dict(q=q,method=method,label=label,n=len(values),
                    max_entry=max(c.get('max_entry',0) for c in values),
                    max_frobenius=max(c.get('frobenius',0) for c in values)))
    selected=sorted([(c.get('frobenius',0),r['sample_id'],r['method'],r['axis'],c['label'])
        for r in checks if r['q']==32 and r['method']!='raw' for c in r['checks']
        if c['label'] in ('compiled_vs_CAR_observable','compiled_vs_raw_observable')],key=lambda r:(-r[0],r[1:]))[0]
    return dict(status='SAVED_RESIDUAL_REANALYSIS',rows=rows,
        selected_worst=dict(frobenius=selected[0],sample_id=selected[1],method=selected[2],axis=selected[3],label=selected[4]),
        selection='maximum recorded q32 observable residual; lexical tie-break; diagnostic selection only',
        component_tolerance=1e-10,observable_tolerance=2e-10,phase_fit=False)


def saved_recheck(directory,selected):
    """One saved path, arrays only. No circuit or state-action rerun."""
    with np.load(directory/'references.npz',allow_pickle=False) as z:refs={k:z[k] for k in z.files}
    records=[]
    def check(a,b,label,tol=1e-10,observable=False):
        d=np.asarray(a,dtype=np.clongdouble)-np.asarray(b,dtype=np.clongdouble)
        maximum=float(abs(d).max());frobenius=float(np.sqrt(np.sum(d.real*d.real+d.imag*d.imag,dtype=np.longdouble)))
        passed=(frobenius if observable else maximum)<=tol
        records.append(dict(label=label,max_entry=maximum,frobenius=frobenius,tolerance=tol,status='PASS' if passed else 'FAIL'))
    v=refs['V'].astype(np.clongdouble)
    check(refs['A'].astype(np.clongdouble)@refs['B'].astype(np.clongdouble),v,'saved_AB_extended_product')
    check(refs['A_adjoint'],refs['A'].conj().T,'saved_actual_adjoint')
    check(v.conj().T@v,np.eye(256),'saved_CAR_unitarity_extended_product')
    for m in METHODS:
        for a in AXES:
            with np.load(directory/(m+'_'+a+'_observable.npz'),allow_pickle=False) as z:o=z['observable']
            target=(v+v.conj().T)/2 if a=='X' else (v-v.conj().T)/(2j)
            check(o,target,m+'_'+a+'_vs_CAR_extended_subtraction',2e-10,True)
            check(o,refs['raw_'+m+'_'+a],m+'_'+a+'_vs_saved_raw',2e-10,True)
            check(o,o.conj().T,m+'_'+a+'_Hermiticity')
    return dict(status='PASS' if all(r['status']=='PASS' for r in records) else 'NUMERICAL_DEPENDENCY_STOP',
        selected=selected,checks=records,extended_precision_mantissa_bits=int(np.finfo(np.longdouble).nmant),
        reference_arrays_repropagated=False,phase_fit=False,threshold_relaxed=False,old_outputs_modified=False)
