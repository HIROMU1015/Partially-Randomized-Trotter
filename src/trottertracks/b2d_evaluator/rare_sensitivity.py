"""Two fixed saved-data cost comparisons; no physical or sampling APIs."""
from collections import Counter
from decimal import Decimal as D, localcontext
from fractions import Fraction
from math import factorial
from pathlib import Path
import csv,json,os,platform,subprocess,sys,time

from .followup_io import FollowupLedger,registered_scope
from .launcher import Evidence,rejection
from .preflight import _check_source

BASE='bef63edbd8cc564444b44e4e8cf23baa90f05fe5'
B1_SOURCE='11e2532a62d54a65f8fad5489be6b135f826011b'
B0_SOURCE='b190fe28b98d2f94e1509e19cd3d36a63b2bb99e'
B1='artifacts/research_b_outer_b1/20261025/'
B0='artifacts/research_b_outer_b0/20261025/WRAPPERS_v5.csv'
IDS=('B4','B3','D1','R4');TIMES=('9/50','-9/50');AXES=('cosine','sine');METRICS=('rz_count','cx_count')
PAIRS=(('A','B3','B4'),('B','R4','D1'))
INPUTS=tuple(B1+x for x in ('SAMPLES.csv','STATISTICS.csv','SUMMARY.json',
    'iid256-once/binding.json','iid256-once/graph.json','iid256-once/inputs.json',
    'iid256-once/result.json','iid256-once/trajectories.jsonl'))+(B0,)
OUTPUTS=('provenance.json','result.json','PROBABILITIES.csv','BOUNDARIES.csv',
    'events.jsonl','partial_results.json','status.json','STOP.json','failure.json')
LIMITS=dict(total_bytes=4*1024**2,json_document_bytes=2*1024**2,
            event_record_bytes=64*1024,files=len(OUTPUTS),stop_reserve_bytes=64*1024)
PRECISION=50

def require(condition,message):
    if not condition:raise ValueError(message)

def dec(v):
    # Reuse Stage-B dec/finite_law from e16c3be8:scripts/rare_rte_stage_b_core.py.
    # Saved binary64 tau/probability is lifted exactly; no new physical precision.
    if isinstance(v,Fraction):return D(v.numerator)/D(v.denominator)
    if isinstance(v,float):return D.from_float(v)
    return D(v)

def finite_law(tau,K):
    require(K in (2,4),'Unsupported saved cutoff')
    t=abs(dec(tau))
    weights={k:(D(1) if k==0 else t**k)/D(factorial(k))*(1+t*t/D((k+1)**2)).sqrt()
             for k in range(0,K+1,2)}
    normalization=sum(weights.values())
    return {k:v/normalization for k,v in weights.items()},normalization

def pack(value):
    if isinstance(value,(D,Fraction,Path)):return str(value)
    if isinstance(value,dict):return {str(k):pack(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [pack(v) for v in value]
    return value

def load(path):return json.loads(Path(path).read_text())
def table(path):return list(csv.DictReader(Path(path).open()))

def scope():
    calls=[('rare.probability',c+':'+t) for c in IDS for t in TIMES]
    calls += [('rare.boundary',':'.join((p,t,a,m))) for p,pos,neg in PAIRS for t in TIMES for a in AXES for m in METRICS]
    return registered_scope('outer_rare_sensitivity',calls,candidate_time_conditions=8,
        comparison_time_conditions=4,axis_metric_boundaries=16,SE_corner_rows=80,
        numerical_policy='Decimal50;exact binary64 lifts;saved p0 product;SE box corners',
        scientific_status='CONDITIONAL_SENSITIVITY_ONLY',new_sampling=0,new_build_compile=0,
        new_H_operator_signal=0,shot_evaluations=0)

def preflight(root,source,output):
    root=Path(root).resolve();out=Path(output).resolve()
    _check_source(root,source,output=out)
    parent=root/'artifacts/research_b_outer_rare_sensitivity'
    require(out.is_relative_to(parent) and out!=parent and not out.exists(),'Fresh dedicated output required')
    blobs={}
    for path in INPUTS:
        expected=subprocess.check_output(['git','rev-parse',BASE+':'+path],cwd=root,text=True).strip()
        actual=subprocess.check_output(['git','hash-object','--',path],cwd=root,text=True).strip()
        require(actual==expected,'Saved input identity mismatch: '+path);blobs[path]=actual
    return dict(root=root,output=out,source=source,input_commit=BASE,input_blobs=blobs,scope=scope())

def saved_inputs(root):
    """Read only the four candidates' laws/statistics; preserve all n/SE values."""
    root=Path(root);raw=root/B1/'iid256-once'
    result=load(raw/'result.json');binding=load(raw/'binding.json');graph=load(raw/'graph.json');inputs=load(raw/'inputs.json')
    require(result['source']==binding['source']==B1_SOURCE,'B1 SOURCE identity mismatch')
    require(result['fixed_sample_set_complete'] and result['scientific_status']=='INCONCLUSIVE_RARE_EVENT_COVERAGE','B1 status changed')
    require(inputs['recipe']['ld']==2 and inputs['occupied_modes']==[0] and inputs['sector_basis_indices']==[1,2],'Fixture/sector/state mismatch')
    rows={r['candidate_id']+':'+r['case']['total_time']:r for r in graph['cases'] if r['candidate_id'] in IDS}
    require(set(rows)=={c+':'+t for c in IDS for t in TIMES},'Comparison scope mismatch')
    laws={};seen=Counter();observations=Counter()
    for line in (raw/'trajectories.jsonl').open():
        job=json.loads(line);key=':'.join(job['id'].split(':')[:2])
        if key not in rows:continue
        require(job['id']==key+':'+str(job['index']) and 0<=job['index']<256,'Sample identity mismatch')
        signature=[]
        for o in job['details']:
            signature.append({k:v for k,v in o.items() if k!='events'})
            require(len(o['events'])==o['spec']['rte_steps'],'Short-step count mismatch')
            require(all(e['order']==0 for e in o['events']),'Observed non-S0 trajectory')
            observations[key]+=len(o['events'])
        require(key not in laws or signature==laws[key],'Saved occurrence law changed within case')
        laws[key]=signature;seen[key]+=1
    require(all(seen[k]==256 for k in rows),'Missing fixed sample provenance')
    costs={};ids=set()
    for sample in table(root/B1/'SAMPLES.csv'):
        if sample['candidate'] not in IDS:continue
        key=sample['candidate']+':'+sample['T'];identity=(sample['id'],sample['axis'])
        require(key in rows and sample['axis'] in AXES and identity not in ids,'Cost sample scope mismatch')
        ids.add(identity)
        require(sample['id']==key+':'+sample['sample'],'Cost identity mismatch')
        for m in METRICS:costs.setdefault((key,sample['axis'],m),[]).append(int(sample[m]))
    stats={}
    for row in table(root/B1/'STATISTICS.csv'):
        c,t,axis=row['group'].split(':');m=row['metric']
        if c not in IDS or m not in METRICS:continue
        require(t in TIMES and axis in AXES,'Statistics scope mismatch')
        key=c+':'+t;v=result['statistics']['groups'][row['group']][m];values=costs[(key,axis,m)]
        require(len(values)==int(row['sample_count'])==v['sample_count']==256,'Paired n must remain256')
        for f in ('mean','standard_error','unbiased_sample_variance','minimum','maximum'):
            require(D(row[f])==D(str(v[f])),'CSV/result statistics mismatch')
        require(abs(D(row['mean'])-D(sum(values))/256)<D('1e-8'),'Saved empirical mean mismatch')
        stats.setdefault(key,{})[axis+':'+m]=dict(mean=D(row['mean']),SE=D(row['standard_error']),
            variance=D(row['unbiased_sample_variance']),minimum=D(row['minimum']),maximum=D(row['maximum']),n=256)
    require(len(ids)==4096 and sum(len(v) for v in stats.values())==32,'S0 statistics incomplete')
    b0={}
    for row in table(root/B0):
        if row['candidate_id'] not in IDS:continue
        key=row['candidate_id']+':'+row['T']+':'+row['axis']
        b0.setdefault(key,[]).append(dict(representative=int(row['representative']),
            rz_count=int(row['rz_count']),cx_count=int(row['cx_count'])))
    require(len(b0)==16 and all(len(v)==4 for v in b0.values()),'B0 reference scope mismatch')
    return rows,laws,stats,b0,observations,inputs

def probability(row,occurrences,lam):
    """Actual heterogeneous signed occurrence sequence, no common-step shortcut."""
    stages=[s['payload'] for s in row['plan']['body'] if 'payload' in s]
    require(stages==[o['spec'] for o in occurrences],'Signed outer stage order mismatch')
    require(row['independent_occurrences'] and row['case']['n_d']==1 and row['case']['macro_steps']==2,'Fixed case identity mismatch')
    require(row['rte_parameters']==dict(K=4 if row['candidate_id']=='R4' else 2,r_rte=2),'Fixed r/K mismatch')
    product=canonical_product=D(1);checks=[];steps=0
    for j,o in enumerate(occurrences):
        s=o['spec'];tau=dec(o['dimensionless_step_time']);duration=dec(Fraction(s['duration']))
        require(o['occurrence']==j and s['rte_steps']==2 and s['seed']==41+j,'Occurrence identity/r mismatch')
        require((tau>0)==(duration>0) and abs(tau-lam*duration/2)<D('2e-16'),'Signed tau/lambda mismatch')
        p={k:dec(v) for k,v in zip(o['orders'],o['order_probabilities'])};canonical,norm=finite_law(tau,s['cutoff'])
        require(list(p)==list(range(0,s['cutoff']+1,2)) and all(v>0 for v in p.values()),'Law order/support mismatch')
        probability_error=max(abs(p[k]-canonical[k]) for k in p)
        norm_error=abs(dec(o['exact_finite_distribution'])-norm)
        require(abs(sum(p.values())-1)<D('1e-15') and probability_error<D('1e-15') and norm_error<D('1e-15'),'Saved distribution/normalization mismatch')
        coverage=o['spec']['rte_steps'];steps+=coverage
        product*=p[0]**coverage;canonical_product*=canonical[0]**coverage
        checks.append(dict(occurrence=j,spec=s,tau=tau,probabilities=p,normalization=dec(o['exact_finite_distribution']),
            saved_probability_sum_minus_one=sum(p.values())-1,maximum_probability_error=probability_error,normalization_error=norm_error))
    require(steps==row['cost']['RTE_short_steps'],'Outer short-step inventory mismatch')
    require(0<product<1,'Nontrivial S0 probability required')
    return dict(p_S0=product,p_rare=1-product,canonical_p_rare_check=1-canonical_product,
        saved_canonical_mass_difference=abs(product-canonical_product),independent_short_steps=steps,
        occurrences=checks,unknown_rare_mean=None,unknown_rare_variance=None,
        bit_level_PRNG_law_certified=False,normalization_changed=False)

def boundary(p_pos,p_neg,pos,neg):
    """Delta = a + p_pos*u_pos - p_neg*u_neg; SE rectangular sensitivity."""
    require(0<p_pos<1 and 0<p_neg<1 and pos['mean']>0 and neg['mean']>0,'Positive law/mean required')
    a=(1-p_pos)*pos['mean']-(1-p_neg)*neg['mean']
    width=(1-p_pos)*pos['SE']+(1-p_neg)*neg['SE']
    slope=p_pos/p_neg
    sensitivity=[]
    for shift in (-2,-1,0,1,2):
        value=a+shift*width
        sensitivity.append(dict(SE_corner=shift,delta_S0=value,
            u_negative_intercept=value/p_neg,u_negative_slope=slope,
            normalized_intercept=value/(p_neg*neg['mean']),
            normalized_slope=slope*pos['mean']/neg['mean'],
            u_positive_axis_intercept=-value/p_pos,
            mu0_positive_shift=shift*pos['SE'],mu0_negative_shift=-shift*neg['SE']))
    return dict(saved_positive=pos,saved_negative=neg,p_positive=p_pos,p_negative=p_neg,
        delta_S0=a,SE_box_width=width,delta_SE1=[a-width,a+width],delta_SE2=[a-2*width,a+2*width],
        sensitivity=sensitivity,unknown_means=dict(positive=None,negative=None),
        positive_means_negative_candidate_cheaper=True,
        nonnegative_cost_alone_fixes_sign=False,unconditional_ranking=None,
        covariance=None,SE_policy='Opposite corners of individual mean +/-kSE boxes;no independence or coverage claim')

def analyze(root,ledger):
    with localcontext() as ctx:
        ctx.prec=PRECISION
        rows,laws,stats,b0,observations,inputs=saved_inputs(root);probs={};bounds=[]
        for key,row in rows.items():
            probs[key]=ledger.call('rare.probability',lambda row=row,key=key:probability(row,laws[key],dec(inputs['exact_rte_lambda_r'])),identity=key)
            probs[key]['observed_order0_steps']=observations[key]
        for name,pos,neg in PAIRS:
            for T in TIMES:
                for axis in AXES:
                    for metric in METRICS:
                        key=':'.join((name,T,axis,metric));pk=pos+':'+T;nk=neg+':'+T;sm=axis+':'+metric
                        b=ledger.call('rare.boundary',lambda:boundary(probs[pk]['p_rare'],probs[nk]['p_rare'],stats[pk][sm],stats[nk][sm]),identity=key)
                        b.update(comparison=name,T=T,axis=axis,metric=metric,positive_candidate=pos,negative_candidate=neg)
                        bounds.append(b)
        return dict(scientific_status='CONDITIONAL_SENSITIVITY_ONLY',B1_status_preserved='INCONCLUSIVE_RARE_EVENT_COVERAGE',
            probabilities=probs,boundaries=bounds,S0_statistics=stats,B0_reference_costs=b0,
            B0_interpretation='Four deterministic representatives only;not rare mean,unbiased estimate or support bound',
            unknown_rare_costs_zero_filled=False,unconditional_optimality=None,
            gamma=None,u_num=None,shots=None,total_resource_G=None)

def execute(run,*,mock_only=False,analysis=analyze):
    out=run['output'];out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(mode=0o700)
    evidence=Evidence(out,outputs=OUTPUTS,limits=LIMITS,mock_only=mock_only);ledger=FollowupLedger(run['scope']);ledger.sink=evidence.event
    tick=time.monotonic();cpu_tick=time.process_time()
    try:
        import resource
        evidence.precision_status='CONDITIONAL_SENSITIVITY_ONLY_NOT_A_CERTIFICATE'
        evidence.write('provenance.json',pack(dict(source=run['source'],input_commit=BASE,input_blobs=run['input_blobs'],
            B1_source=B1_SOURCE,B0_source=B0_SOURCE,scope=run['scope'],python=sys.version,platform=platform.platform(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),resource_policy='Small saved-file scan and44 occurrence checks;single streaming pass,not a worker cap')))
        ledger.emit(dict(event='available_resources',memory=next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')),load_average=os.getloadavg()))
        require(not mock_only or analysis is not analyze,'Mock analysis dependency required')
        result=analysis(run['root'],ledger);ledger.finish()
        result.update(source=run['source'],wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu_tick,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        for name,rows in (
            ('PROBABILITIES.csv',[dict(case=k,**{f:v[f] for f in ('p_S0','p_rare','canonical_p_rare_check','saved_canonical_mass_difference','independent_short_steps','observed_order0_steps')}) for k,v in result['probabilities'].items()]),
            ('BOUNDARIES.csv',[dict(**{k:b[k] for k in ('comparison','T','axis','metric','positive_candidate','negative_candidate')},**point) for b in result['boundaries'] for point in b['sensitivity']])):
            with (out/name).open('x',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(pack(rows));f.flush();os.fsync(f.fileno())
        return evidence.finish(dict(status='COMPLETED_DESCRIPTIVE_MANDATORY_STOP'),ledger,result=pack(result))
    except BaseException as error:
        import traceback
        evidence.write('failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        return evidence.finish(dict(status='FAILED_MANDATORY_STOP'),ledger,error=error)

def main():
    import argparse
    parser=argparse.ArgumentParser(description='Fixed two-pair saved RTE sensitivity;no sampling/build/compile')
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--inspect',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--source-commit');parser.add_argument('--output');args=parser.parse_args()
    if args.inspect:print(json.dumps(scope()));return 0
    if not args.source_commit or not args.output:parser.error('--execute requires --source-commit and --output')
    try:status=execute(preflight(Path(__file__).resolve().parents[3],args.source_commit,args.output))
    except BaseException as error:status=rejection(error)
    print(json.dumps(status));return 0 if status['status']=='COMPLETED_DESCRIPTIVE_MANDATORY_STOP' else 2
