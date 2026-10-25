"""Fixed saved-data shot accounting. Metadata imports never evaluate science."""
from collections import defaultdict
from decimal import localcontext
from fractions import Fraction as F
from math import factorial
from pathlib import Path
import csv,importlib.metadata,json,math,os,platform,subprocess,sys,time

from .b0_native import PROFILE
from .followup_io import FollowupLedger,registered_scope
from .launcher import Evidence,rejection
from .preflight import _check_source
from .scope import Stage1Failure,STOP
from .rare_sensitivity import dec,pack,require
from ..pr_dictionary_validation.intervals import scalar_context,rational_interval,endpoints
from ..pr_dictionary_validation.gates import shot_count_from_enclosure

BASE='488d6ab544295905dd99fa2c6357202d9b60ea31'
IDS=('B3','D1','R2','R4','B4','F1');TIMES=('9/50','-9/50');EPS=('1e-8','1e-9')
AXES=('cosine','sine');METRICS=('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')
PAIRS=(('A','B3','B4'),('B','R4','D1'),('C','R2','D1'))
EXPECTED={
    'B3':('outer-Yoshida4','A2',None,2,2),'D1':('outer-Yoshida4','A3',None,2,2),
    'R2':('outer-Yoshida4','A3',None,4,2),'R4':('outer-Yoshida4','A3',None,2,4),
    'B4':('SABA2','A2',None,2,2),'F1':('refined-S2','A2',8,2,2)}
PATHS={
    'analytic':'artifacts/research_b_outer_rte22/20261025/analytic-once/',
    'a0':'artifacts/research_b_outer_a0/20261025/independent-once/',
    'b0':'artifacts/research_b_outer_b0/20261025/corrected-2/',
    'b1':'artifacts/research_b_outer_b1/20261025/',
    'rare':'artifacts/research_b_outer_rare_sensitivity/20261025/saved-once/'}
COMMITS=dict(analytic='33715848bfb83d49ec4e54d6eef481012a7441e4',a0='31662b49157b15f2d1c03eef1c2faa07e2f905e7',
    b0='0db9b3ed189eb6cd9d66e070993dbd0e72d29104',b1='bef63edbd8cc564444b44e4e8cf23baa90f05fe5',rare='cae3cd784dbdfcf2fe96902345379aca4b9606c7')
SOURCES=dict(analytic='26e8734aff3969dc8077cb058826d12394be28e1',a0='1e3921ba16e5d531e9a3010cffa9ff717609a93a',
    b0='b190fe28b98d2f94e1509e19cd3d36a63b2bb99e',b1='11e2532a62d54a65f8fad5489be6b135f826011b',rare='a8cf99156af692f0facaa1885ab9ed52b28c6f4c')
INPUTS=tuple((study,PATHS[study]+name) for study,names in (
    ('analytic',('result.json','inputs.json')),('a0',('result.json','inputs.json')),('b0',('result.json',)),
    ('b1',('iid256-once/result.json','iid256-once/inputs.json','iid256-once/graph.json','iid256-once/binding.json',
           'SAMPLES.csv','STATISTICS.csv','iid256-once/trajectories.jsonl')),('rare',('result.json',))) for name in names)
OUTPUTS=('provenance.json','result.json','SHOTS.csv','RESOURCES.csv','COMPARISONS.csv',
    'events.jsonl','partial_results.json','status.json','STOP.json','failure.json')
LIMITS=dict(total_bytes=32*1024**2,json_document_bytes=8*1024**2,event_record_bytes=64*1024,
    files=len(OUTPUTS),stop_reserve_bytes=256*1024)
IDENTITY=('hamiltonian_id','sector_id','state_id','input_data_sha256','tail_id','occupied_modes','sector_basis_indices',
          'deterministic_fragment_indices','randomized_block_indices','fixture_definition')

def frac(value):
    require(not isinstance(value,bool),'Boolean is not a numeric input')
    require(not isinstance(value,float) or math.isfinite(value),'Nonfinite numeric input')
    return F.from_float(value) if isinstance(value,float) else F(value)

def integer_cost(value):
    require(type(value) is int or (isinstance(value,str) and value.isdecimal()),'Integer cost count required')
    count=int(value);require(count>=0,'Negative cost count');return count

def key(row):return row['candidate_id']+':'+row['case']['total_time']
def case_keys():return tuple(c+':'+t for c in IDS for t in TIMES)

def scope():
    calls=[]
    for k in case_keys():
        calls.extend((name,k) for name in ('shot.bind','shot.normalization'))
        if k.split(':')[0] in ('R2','F1'):calls.append(('shot.rare',k))
        calls.extend(('shot.joint',k+':'+m) for m in METRICS)
        for e in EPS:
            q=k+':'+e;calls.extend((name,q) for name in ('shot.bias','shot.qualification'))
            calls.extend(('shot.count',q+':'+a) for a in AXES)
            calls.extend(('shot.resource',q+':'+m) for m in METRICS)
    calls.extend(('shot.compare',':'.join((pair,t,e,m))) for pair,p,n in PAIRS for t in TIMES for e in EPS for m in METRICS[:2])
    return registered_scope('shot_weighted',calls,candidate_time_conditions=12,case_epsilon_conditions=24,
        shots=48,joint_moments=72,resources=144,comparisons=24,normalization_occurrences=88,short_step_factors=200,
        precision='mpmath1.3.0 outward scalar90dps;exact Fraction and integer;Decimal50 saved rare mass',
        error_budget=dict(bias_plus_numerical='epsilon/2',implementation='epsilon/10',statistical='2*epsilon/5'),
        confidence=dict(alpha_axis='1/40',scope='PER_CANDIDATE_T_EPSILON',familywise=None),
        new_sampling=0,new_build_compile=0,new_H_operator_signal=0,quantum_shots_executed=0)

def preflight(root,source,output):
    root=Path(root).resolve();out=Path(output).resolve()
    require(root==Path(__file__).resolve().parents[3],'Loaded SOURCE root mismatch')
    require(len(source)==40 and all(c in '0123456789abcdef' for c in source),'Full SOURCE required')
    _check_source(root,source,output=out)
    parent=root/'artifacts/research_b_outer_shot_weighted'
    require(out.is_relative_to(parent) and out!=parent and not out.exists(),'Fresh dedicated output required')
    version=importlib.metadata.version('mpmath');require(version=='1.3.0','Unsupported scalar interval API version')
    blobs={}
    for study,path in INPUTS:
        expected=subprocess.check_output(['git','rev-parse',COMMITS[study]+':'+path],cwd=root,text=True).strip()
        actual=subprocess.check_output(['git','hash-object','--',path],cwd=root,text=True).strip()
        require(actual==expected,'Saved input identity mismatch: '+path);blobs[path]=dict(commit=COMMITS[study],blob=actual)
    return dict(root=root,output=out,source=source,input_blobs=blobs,scope=scope(),environment=dict(
        python=sys.version,mpmath=version,platform=platform.platform(),executable=sys.executable,
        affinity=sorted(os.sched_getaffinity(0)),load_average=os.getloadavg(),
        available_memory=next(v for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')),
        policy='One streaming saved-data scan;scalar caches;no artificial CPU/RAM/worker cap'))

def read_payload(root):
    root=Path(root)
    def load(study,name):return json.loads((root/PATHS[study]/name).read_text())
    data={s:load(s,'result.json') for s in ('analytic','a0','b0','rare')}
    data['b1']=load('b1','iid256-once/result.json');data['graph']=load('b1','iid256-once/graph.json')
    data['binding']=load('b1','iid256-once/binding.json')
    data['inputs']={s:load(s,'inputs.json' if s!='b1' else 'iid256-once/inputs.json') for s in ('analytic','a0','b1')}
    data['samples']=defaultdict(list);data['trajectories']=defaultdict(list);data['statistics']={}
    with (root/PATHS['b1']/'SAMPLES.csv').open() as f:
        for row in csv.DictReader(f):data['samples'][row['candidate']+':'+row['T']].append(row)
    with (root/PATHS['b1']/'STATISTICS.csv').open() as f:
        for row in csv.DictReader(f):data['statistics'][(row['group'],row['metric'])]=row
    with (root/PATHS['b1']/'iid256-once/trajectories.jsonl').open() as f:
        for line in f:
            row=json.loads(line);data['trajectories'][':'.join(row['id'].split(':')[:2])].append(row)
    return data

def bind_case(k,data):
    c,t=k.split(':');require(c in IDS and t in TIMES,'Candidate/time scope mismatch')
    for study in ('analytic','a0','b0','b1','rare'):
        field='source_commit' if study=='analytic' else 'source'
        require(data[study][field]==SOURCES[study],'Saved science SOURCE mismatch: '+study)
    require(data['binding']['source']==SOURCES['b1'] and data['b1']['fixed_sample_set_complete'],'B1 binding/incomplete')
    require(data['b1']['scientific_status']=='INCONCLUSIVE_RARE_EVENT_COVERAGE' and data['rare']['scientific_status']=='CONDITIONAL_SENSITIVITY_ONLY','Prior scientific status mismatch')
    reference=data['inputs']['analytic']
    require(reference['recipe']['ld']==2 and reference['recipe']['n_electrons']==1 and reference['occupied_modes']==[0]
        and reference['sector_basis_indices']==[1,2] and reference['recipe']['n_qubits']==2
        and reference['recipe']['weight_rule']=='lambda_frobenius_squared'
        and reference['deterministic_fragment_indices']==[0,1] and reference['randomized_block_indices']==[2],'H/sector/state/DF split mismatch')
    for inputs in data['inputs'].values():
        require(all(inputs[f]==reference[f] for f in IDENTITY),'Input H/sector/state/DF identity mismatch')
    require(pack(data['graph']['scope']['compiler'])==pack(PROFILE),'Full-wrapper compiler profile mismatch')
    require(pack(data['b0']['provenance']['compiler'])==pack(PROFILE)
        and data['b0']['provenance']['preparation_id']==reference['preparation_id']
        and data['b0']['provenance']['tail_id']==reference['tail_id'],'B0 native provenance mismatch')
    a=next((r for r in data['analytic']['rows'] if key(r)==k),None)
    g=next((r for r in data['graph']['cases'] if key(r)==k),None)
    require(a is not None and g is not None and a['case']==g['case'] and a['rte_parameters']==g['rte_parameters'],'Case join mismatch')
    method,arm,refinement,r,K=EXPECTED[c];case=a['case']
    require((case['method'],case['inner_arm'],case['refinement'],case['macro_steps'],case['n_d'])==(method,arm,refinement,2,1)
        and a['rte_parameters']==dict(r_rte=r,K=K),'Fixed scientific conditions mismatch')
    specs=[s['payload'] for s in g['plan']['body'] if 'payload' in s]
    stages=a['outputs']['rte_stages'];require(len(stages)==len(specs),'Occurrence inventory mismatch')
    traces=data['trajectories'][k];require(len(traces)==256 and {x['index'] for x in traces}==set(range(256)),'Trajectory ID coverage mismatch')
    first=traces[0]['details']
    trace_keys={job['index']:job['key'] for job in traces}
    for job in traces:
        require(job['id']==k+':'+str(job['index']) and len(job['details'])==len(specs),'Trajectory join mismatch')
        for i,(o,spec,stage) in enumerate(zip(job['details'],specs,stages)):
            conf=stage['config'];law=stage['event_distribution']
            require(o['occurrence']==i and o['spec']==spec and o['spec']==first[i]['spec'],'Signed occurrence order mismatch')
            require(conf['rte_steps']==spec['rte_steps']==r and conf['finite_taylor_order']==spec['cutoff']==K
                and conf['seed']==spec['seed'] and conf['evolution_time']==float(F(spec['duration']))
                and conf['tail_hash']==reference['tail_id'] and stage['append_occurrence']==i
                and conf['taylor_distribution']=='rte_even_taylor_paired','Signed config/r/K identity mismatch')
            require(o['dimensionless_step_time']==conf['dimensionless_step_time']==law['dimensionless_step_time']
                and o['orders']==law['orders'] and o['order_probabilities']==law['order_probabilities']
                and o['exact_finite_distribution']==law['exact_finite_distribution'],'Saved normalization/distribution identity mismatch')
            probabilities=[frac(p) for p in law['order_probabilities']]
            require(law['orders']==list(range(0,K+1,2)) and len(probabilities)==len(law['orders'])
                and all(0<p<=1 for p in probabilities) and abs(sum(probabilities)-1)<F(1,10**14),'Invalid saved probability support')
            require(len(o['events'])==r and all(e['order']==0 for e in o['events']),'Non-S0 or incomplete cost cohort')
    paired={i:{} for i in range(256)}
    for row in data['samples'][k]:
        i=int(row['sample']);axis=row['axis']
        require(i in paired and axis in AXES and axis not in paired[i] and row['id']==k+':'+str(i),'Paired sample ID mismatch')
        require(row['physical_key']==trace_keys[i],'Physical trajectory identity mismatch')
        require(int(row['measurement_count'])==1,'Full-wrapper measurement missing')
        paired[i][axis]=row
    require(all(set(v)==set(AXES) and v[AXES[0]]['physical_key']==v[AXES[1]]['physical_key'] for v in paired.values()),'Paired axes incomplete')
    saved={}
    for axis in AXES:
        for m in METRICS:
            group=k+':'+axis;v=data['b1']['statistics']['groups'][group][m];csvrow=data['statistics'][(group,m)]
            require(v['sample_count']==int(csvrow['sample_count'])==256,'Saved n must remain256')
            require(all(F(str(v[f]))==F(csvrow[f]) for f in ('mean','unbiased_sample_variance','standard_error','minimum','maximum')),'Saved cost statistic identity mismatch')
            saved[axis+':'+m]=v
    auxiliary=next((v for v in data['a0']['comparisons'] if v['key']==k),None)
    require((auxiliary is not None)==(c in ('B3','F1','R4')),'A0 auxiliary scope mismatch')
    require(a['outputs']['bias_qualification']['observed_bias']==a['outputs']['signals']['corrected_physical_signal_error'],'Direct physical bias field mismatch')
    return dict(key=k,candidate=c,T=t,case=case,rte=a['rte_parameters'],stages=stages,
        N_used=frac(a['outputs']['normalization']['normalization_product']),
        observed_bias=frac(a['outputs']['signals']['corrected_physical_signal_error']),
        A0_auxiliary=auxiliary,pairs=[paired[i] for i in range(256)],saved_statistics=saved,
        p_rare_saved=data['rare']['probabilities'].get(k,{}).get('p_rare'))

def normalization_enclosure(case):
    ctx=scalar_context();total=ctx.mpf(1);saved_total=F(1);occurrences=[];blocks={}
    for stage in case['stages']:
        conf=stage['config'];dist=stage['event_distribution'];tau=frac(conf['dimensionless_step_time']);r=conf['rte_steps'];K=conf['finite_taylor_order']
        require(type(r) is int and r>0 and K in (2,4) and tau!=0,'Invalid scalar normalization input')
        require((tau>0)==(frac(conf['evolution_time'])>0),'Signed normalization time mismatch')
        require(abs(tau-frac(conf['lambda_r'])*frac(conf['evolution_time'])/r)<F(1,10**15),'Tau/lambda identity mismatch')
        cache_key=(abs(tau),K)
        if cache_key not in blocks:
            x=abs(rational_interval(ctx,tau))
            blocks[cache_key]=sum((x**n/factorial(n)*ctx.sqrt(1+(x/(n+1))**2) for n in range(0,K+1,2)),ctx.mpf(0))
        b=blocks[cache_key]
        lo,hi=endpoints(b);saved_b=frac(dist['exact_finite_distribution'])
        require(abs(saved_b-(lo+hi)/2)<F(1,10**12),'Saved distribution normalization mismatch')
        total*=b**r;saved_total*=frac(stage['normalization_product'])
        occurrences.append(dict(tau=tau,r=r,K=K,B_low=lo,B_high=hi))
    lo,hi=endpoints(total);used=case['N_used'];require(0<lo<=hi and used>0,'Invalid normalization range')
    require(abs(used-(lo+hi)/2)<F(1,10**12) and abs(saved_total-used)<F(1,10**12),'Saved full normalization mismatch')
    upper=max(hi,abs(used));scale=max(abs(used-lo),abs(used-hi))*endpoints(ctx.sqrt(2))[1]
    return dict(N_used=used,N_model_low=lo,N_model_high=hi,U=upper,rescaling_bias_upper=scale,
        u_num_total=None,model='saved_binary64_signed_tau_finite_paired_Taylor',occurrences=occurrences,
        unique_scalar_B_evaluations=len(blocks),
        qualification='SCALAR_MODEL_ENCLOSURE_ONLY_NOT_SIGNAL_U_NUM_OR_BIT_LAW_CERTIFICATE')

def extra_rare_probability(case):
    with localcontext() as ctx:
        ctx.prec=50;p=dec(1)
        for stage in case['stages']:
            conf=stage['config'];law=stage['event_distribution']
            require(law['orders']==list(range(0,conf['finite_taylor_order']+1,2)),'Rare law support mismatch')
            p*=dec(law['order_probabilities'][0])**conf['rte_steps']
        require(0<p<1,'Invalid S0 probability')
        return F(str(1-p))

def joint_moments(pairs,metric,saved=None):
    n=len(pairs);require(n==256,'Exactly256 paired samples required')
    xs=[integer_cost(p['cosine'][metric]) for p in pairs];ys=[integer_cost(p['sine'][metric]) for p in pairs]
    x=F(sum(xs),n);y=F(sum(ys),n)
    vx=(sum(v*v for v in xs)-n*x*x)/(n-1);vy=(sum(v*v for v in ys)-n*y*y)/(n-1)
    cov=(sum(a*b for a,b in zip(xs,ys))-n*x*y)/(n-1)
    if saved is not None:
        for axis,mean,var in (('cosine',x,vx),('sine',y,vy)):
            original=saved[axis+':'+metric]
            require(abs(frac(original['mean'])-mean)<F(1,10**8) and abs(frac(original['unbiased_sample_variance'])-var)<F(1,10**6),'Saved statistics do not describe paired cohort')
    return dict(n=n,mean_X=x,mean_Y=y,variance_X=vx,variance_Y=vy,sample_covariance_XY=cov,
        saved_statistics={a:saved[a+':'+metric] for a in AXES} if saved else None,
        mean_rule='exact same256 integer observations;original binary64 statistics preserved')

def scalar_shots(norm,epsilon,axis):
    require(axis in AXES,'Axis mismatch');eps=frac(epsilon);require(eps>0,'Positive epsilon required')
    ctx=scalar_context();s=2*eps/5;alpha=F(1,40);u=frac(norm['U']);require(u>0,'Positive range upper required')
    expr=4*rational_interval(ctx,u)**2/rational_interval(ctx,s)**2*ctx.ln(rational_interval(ctx,2/alpha))
    lo,hi=endpoints(expr)
    try:count=shot_count_from_enclosure(lo,hi)
    except ValueError as error:raise Stage1Failure('SHOT_COUNT_UNRESOLVED') from error
    require(count>0,'Positive shots required')
    return dict(axis=axis,epsilon=str(epsilon),s=s,alpha_axis=alpha,**{k:norm[k] for k in ('N_used','N_model_low','N_model_high','U')},
        expression_low=lo,expression_high=hi,m_decimal_string=str(count),shot_status='CONDITIONAL_MODEL_SUFFICIENT_SHOTS',
        IID='fresh full RTE trajectory per quantum shot',quantum_shots_executed=0,familywise_qualified=None)

def bias_status(case,epsilon):
    eps=frac(epsilon);b=case['observed_bias'];require(b>=0,'Invalid observed bias')
    aux=case.get('A0_auxiliary');match=None
    if aux is not None:match=(frac(aux['bias']['real'])<=eps/2)==(b<=eps/2)
    return dict(observed_bias=b,bias_plus_numerical_budget=eps/2,implementation_budget=eps/10,statistical_budget=2*eps/5,
        observed_bias_budget_status='WITHIN_POINT_BUDGET' if b<=eps/2 else 'OUTSIDE_POINT_BUDGET',
        auxiliary_A0_point_flag_matches=match,input_numerics='INCONCLUSIVE_INPUT_NUMERICS' if match is False else 'EMPIRICAL_ONLY',
        u_num=None,gamma=None,deterministic_qualified=None,implementation_qualified=None,end_to_end_qualified=None)

def weighted_resource(joint,shots,p):
    mx,my=(int(shots[a]['m_decimal_string']) for a in AXES);p=frac(p);require(0<p<1,'Positive rare mass required')
    mean=mx*joint['mean_X']+my*joint['mean_Y']
    variance=mx*mx*joint['variance_X']+my*my*joint['variance_Y']+2*mx*my*joint['sample_covariance_XY']
    require(variance>=0,'Negative paired variance')
    ctx=scalar_context();low,high=endpoints(ctx.sqrt(rational_interval(ctx,variance/joint['n'])))
    return dict(n_S0=joint['n'],m_X=str(mx),m_Y=str(my),G_S0_fraction=mean,variance_S0_sample=variance,
        SE_S0_low=low,SE_S0_high=high,p_rare=p,known_contribution=(1-p)*mean,
        SE_known_upper=(1-p)*high,coefficient_u_X=p*mx,coefficient_u_Y=p*my,
        unknown_mean_u_X=None,unknown_mean_u_Y=None,unconditional_G=None,unconditional_SE=None,
        S0_SE_sensitivity=[dict(k=k,G_S0=mean+k*high) for k in (-2,-1,0,1,2)],
        uncertainty='PAIRED_COST_MC_SAMPLE_SE_ONLY_NOT_QUANTUM_CONFIDENCE_OR_RARE_TAIL',qualification='S0_REFERENCE_AND_SYMBOLIC_RARE_ONLY')

def compare_resources(pos,neg):
    pp,pn=pos['p_rare'],neg['p_rare'];mp=int(pos['m_X'])+int(pos['m_Y']);mn=int(neg['m_X'])+int(neg['m_Y'])
    a=pos['known_contribution']-neg['known_contribution'];width=pos['SE_known_upper']+neg['SE_known_upper']
    require(pp*mp>0 and pn*mn>0,'Undefined rare boundary')
    return dict(delta_known=a,SE_box_width=width,positive_coefficients=dict(X=pos['coefficient_u_X'],Y=pos['coefficient_u_Y']),
        negative_coefficients=dict(X=neg['coefficient_u_X'],Y=neg['coefficient_u_Y']),
        rare_mixture_definition='u_i^m=(m_X*u_X+m_Y*u_Y)/(m_X+m_Y);not equal-axis cost assumption',
        mixture_boundary_slope=pp*mp/(pn*mn),
        sensitivity=[dict(k=k,delta_known=a+k*width,negative_mixture_intercept=(a+k*width)/(pn*mn)) for k in (-2,-1,0,1,2)],
        direction='Delta>0 means negative candidate cheaper;u_negative below boundary',
        unknown_rare_means=dict(positive_X=None,positive_Y=None,negative_X=None,negative_Y=None),
        unconditional_ranking=None,candidate_covariance=None,confidence_interval=None)

def qualifications(bias,shots):
    return dict(observed_bias_budget_status=bias['observed_bias_budget_status'],u_num=None,gamma=None,
        deterministic_qualified=None,implementation_qualified=None,unseen_rare_costs=True,
        model_shots_evaluated=all(a in shots for a in AXES),statistical_status='CONDITIONAL_MODEL_ONLY',
        end_to_end_qualified=None,end_to_end_status='NOT_CERTIFIED',input_numerics=bias['input_numerics'])

def evaluate(root,ledger,emit,*,payload_reader=read_payload):
    """One saved-data pass; scientific functions are dormant until called here."""
    data=payload_reader(root);cases={};normalizations={};rare={};joints={};resources={}
    shot_rows=[];resource_rows=[];comparisons=[];qualified_rows=[]
    def retain(kind,identity,value):
        ledger.rows.append(dict(kind=kind,identity=identity,value=pack(value)))
        return value
    # Finish every saved-input join before starting numerical accounting.
    for k in case_keys():
        cases[k]=ledger.call('shot.bind',lambda k=k:bind_case(k,data),identity=k)
    for k,case in cases.items():
        norm=ledger.call('shot.normalization',lambda:normalization_enclosure(case),identity=k)
        normalizations[k]=retain('normalization',k,norm)
        if case['candidate'] in ('R2','F1'):
            p=ledger.call('shot.rare',lambda:extra_rare_probability(case),identity=k)
            origin='NEW_SCALAR_PRODUCT_OF_SAVED_B1_PROBABILITIES'
        else:
            require(case['p_rare_saved'] is not None,'Existing rare probability missing')
            p=frac(case['p_rare_saved']);origin='SAVED_CONDITIONAL_SENSITIVITY'
        require(0<p<1,'Invalid rare probability');rare[k]=dict(p_rare=p,origin=origin,unknown_mean=None)
        for metric in METRICS:
            joint=ledger.call('shot.joint',lambda:joint_moments(case['pairs'],metric,case['saved_statistics']),identity=k+':'+metric)
            joints[k+':'+metric]=retain('joint',k+':'+metric,joint)
        for epsilon in EPS:
            q=k+':'+epsilon
            bias=ledger.call('shot.bias',lambda:bias_status(case,epsilon),identity=q);shots={}
            retain('bias',q,bias)
            for axis in AXES:
                record=ledger.call('shot.count',lambda:scalar_shots(norm,epsilon,axis),identity=q+':'+axis)
                shots[axis]=record
                row=dict(candidate=case['candidate'],T=case['T'],**record,
                    observed_bias_budget_status=bias['observed_bias_budget_status'],
                    deterministic_qualified=None,implementation_qualified=None,end_to_end_qualified=None)
                shot_rows.append(row);retain('shot',q+':'+axis,row);emit('SHOTS.csv',row)
            for metric in METRICS:
                record=ledger.call('shot.resource',lambda:weighted_resource(joints[k+':'+metric],shots,p),identity=q+':'+metric)
                record.update(candidate=case['candidate'],T=case['T'],epsilon=epsilon,metric=metric,
                    observed_bias_budget_status=bias['observed_bias_budget_status'])
                resources[q+':'+metric]=record;resource_rows.append(record)
                retain('resource',q+':'+metric,record);emit('RESOURCES.csv',record)
            qualification=ledger.call('shot.qualification',lambda:qualifications(bias,shots),identity=q)
            qualification.update(candidate=case['candidate'],T=case['T'],epsilon=epsilon,
                bias=bias,A0_auxiliary=case['A0_auxiliary'])
            qualified_rows.append(retain('qualification',q,qualification))
    require(sum(len(n['occurrences']) for n in normalizations.values())==88,'Normalization occurrence count mismatch')
    require(sum(o['r'] for n in normalizations.values() for o in n['occurrences'])==200,'Short-step factor count mismatch')
    for pair,pos,neg in PAIRS:
        for t in TIMES:
            for e in EPS:
                for metric in METRICS[:2]:
                    identity=':'.join((pair,t,e,metric))
                    record=ledger.call('shot.compare',lambda:compare_resources(
                        resources[':'.join((pos,t,e,metric))],resources[':'.join((neg,t,e,metric))]),identity=identity)
                    record.update(comparison=pair,positive_candidate=pos,negative_candidate=neg,T=t,epsilon=e,metric=metric)
                    comparisons.append(record);retain('comparison',identity,record);emit('COMPARISONS.csv',record)
    return dict(scientific_status='INCONCLUSIVE_QUALIFICATIONS_AND_RARE_COSTS',
        normalizations=normalizations,rare_probabilities=rare,joint_moments=joints,shots=shot_rows,
        resources=resource_rows,comparisons=comparisons,qualifications=qualified_rows,
        B1_status_preserved='INCONCLUSIVE_RARE_EVENT_COVERAGE',rare_analysis_status_preserved='CONDITIONAL_SENSITIVITY_ONLY',
        u_num=None,gamma=None,certified_shots=None,unconditional_G=None,unconditional_G_SE=None,
        unconditional_winner=None,familywise_guarantee=None,quantum_shots_executed=0)

def stop_category(error):
    message=str(error)
    if 'SHOT_COUNT_UNRESOLVED' in message:return 'INCONCLUSIVE_NUMERICAL_STOP'
    if isinstance(error,Stage1Failure) and any(s in error.code for s in ('SCOPE','INCOMPLETE','DUPLICATE')):return 'SCOPE_FAILURE'
    if isinstance(error,(OverflowError,ZeroDivisionError)) or 'Nonfinite' in message:return 'NUMERICAL_FAILURE'
    if isinstance(error,(ValueError,KeyError)):return 'INPUT_OR_CORRECTNESS_FAILURE'
    return 'TECHNICAL_STOP'

def execute(run,*,mock_only=False,payload_reader=read_payload,evaluator=evaluate,evidence_factory=Evidence):
    """Exclusive fresh output, streamed tables, retained partial values, no retry."""
    out=run['output'];out.parent.mkdir(parents=True,exist_ok=True);out.mkdir(mode=0o700)
    ledger=FollowupLedger(run['scope']);streams={};writers={};evidence=None
    tick=time.monotonic();cpu=time.process_time()
    try:
        require(not mock_only or payload_reader is not read_payload,'Mock payload reader required')
        evidence=evidence_factory(out,outputs=OUTPUTS,limits=LIMITS,mock_only=mock_only)
        evidence.precision_status='SCALAR_ENCLOSURES_CONDITIONAL_MODEL_ONLY_U_NUM_GAMMA_NULL';ledger.sink=evidence.event
        evidence.write('provenance.json',pack({k:v for k,v in run.items() if k not in ('root','output')}))
        def emit(name,value):
            require(name in ('SHOTS.csv','RESOURCES.csv','COMPARISONS.csv'),'CSV output scope mismatch')
            packed=pack(value)
            row={k:json.dumps(v,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list)) else
                 'null' if v is None else v for k,v in packed.items()}
            if name not in writers:
                stream=(out/name).open('x',newline='',encoding='utf-8');streams[name]=stream
                writers[name]=csv.DictWriter(stream,fieldnames=list(row));writers[name].writeheader()
            writers[name].writerow(row);streams[name].flush();os.fsync(streams[name].fileno());evidence.audit()
        result=evaluator(run['root'],ledger,emit,payload_reader=payload_reader);ledger.finish()
        import resource
        result.update(source=run['source'],wall_seconds=time.monotonic()-tick,CPU_seconds=time.process_time()-cpu,
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return evidence.finish(dict(status='COMPLETED_DESCRIPTIVE_MANDATORY_STOP',
            scientific_status=result['scientific_status']),ledger,result=pack(result))
    except BaseException as error:
        import traceback
        if evidence is None:
            status=dict(status='SETUP_FAILED_MANDATORY_STOP',error_type=type(error).__name__,stop=STOP)
            for name in ('status.json','STOP.json'):
                with (out/name).open('x') as f:json.dump(status,f)
            return status
        try:evidence.write('failure.json',dict(category=stop_category(error),error_type=type(error).__name__,
            message=str(error),traceback=traceback.format_exc()))
        except BaseException:pass
        return evidence.finish(dict(status='FAILED_MANDATORY_STOP',failure_category=stop_category(error)),ledger,error=error)
    finally:
        for stream in streams.values():stream.close()

def main():
    import argparse
    parser=argparse.ArgumentParser(description='Fixed24-condition saved shot accounting;no quantum execution or automatic retry')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--inspect',action='store_true');mode.add_argument('--preflight',action='store_true');mode.add_argument('--execute',action='store_true')
    parser.add_argument('--source-commit');parser.add_argument('--output');args=parser.parse_args()
    if args.inspect:
        print(json.dumps(dict(status='METADATA_ONLY_NO_SCIENCE',scope=scope())));return 0
    if not args.source_commit or not args.output:parser.error('--preflight/--execute require --source-commit and --output')
    try:
        run=preflight(Path(__file__).resolve().parents[3],args.source_commit,args.output)
        if args.preflight:
            status=dict(status='READY_READ_ONLY_NO_SCIENCE',**{k:v for k,v in run.items() if k!='root'})
        else:status=execute(run)
    except BaseException as error:status=rejection(error)
    print(json.dumps(pack(status)));return 0 if status['status'] in ('READY_READ_ONLY_NO_SCIENCE','COMPLETED_DESCRIPTIVE_MANDATORY_STOP') else 2
