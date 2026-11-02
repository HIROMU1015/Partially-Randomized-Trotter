"""Conditional G4-B bounded matched CTS acquisition; one process, no retry."""
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('cts_fixed',Path(__file__).with_name('g4_cts_specialization.py'))
cts=importlib.util.module_from_spec(spec);spec.loader.exec_module(cts);v=cts.v
OUT=v.OUT;PRECISIONS=v.PRECISIONS;AXES=('T','CX','1Q');DEN=2**60
ELL=v.log_enclosure().hi
CONTRACT='artifacts/track_b_rte_reallocation_r1_source/2026-10-06/contract_v2.json'
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_reallocation.model import Event
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle,lower_event
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,synthesize,synthesis_key
from trottertracks.algorithm_codesign.rte_reallocation.accounting import native_cost
from trottertracks.algorithm_codesign.rte_reallocation.launch import BudgetGuard
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime

def planned_keys(defs):
    return dict(sorted({synthesis_key(Angle('atan',F(d['fixed_rational_rotation_ratio']),F(sign)),eps):
        (Angle('atan',F(d['fixed_rational_rotation_ratio']),F(sign)),eps)
        for d in defs.values() for eps,sign in product(PRECISIONS,(-1,1))}.items()))

def lower(definition,e):
    if e['real_event']:
        event=Event('CTS_M3',e['label'],F(1),F(0),F(1),word=() if e['axis']=='II' else (e['axis'],),phase=2)
    else:
        event=Event('CTS_M3',e['label'],F(1),F(definition['fixed_rational_rotation_ratio']),F(1),rotation=e['axis'],rotation_sign=e['rotation_sign'])
    return lower_event('distinct_basis',event,True)

def dyadic(q):
    z=sum(q);q=[p/z for p in q]
    if min(q)<=0:raise ValueError('support lost before rounding')
    ns=[(p*DEN).__floor__() for p in q];left=DEN-sum(ns)
    for i in sorted(range(len(q)),key=lambda i:(-(q[i]*DEN-ns[i]),i))[:left]:ns[i]+=1
    if min(ns)<=0:raise ValueError('support lost on rounding')
    return [F(n,DEN) for n in ns]

def proposals(es,axis):
    a=[F(e['coefficient_mid']) for e in es];B=sum(a);can=[z/B for z in a]
    zero=[i for i,e in enumerate(es) if F(e['native_cost'][axis])==0];pos=[i for i in range(len(es)) if i not in zero]
    if not pos:return [('all_zero_canonical',dyadic(can))]
    raw={i:a[i]/((v.sqrt_enclosure(es[i]['native_cost'][axis]).lo+v.sqrt_enclosure(es[i]['native_cost'][axis]).hi)/2) for i in pos}
    S=sum(raw.values());W=sum(a[i] for i in zero)
    masses=[('canonical',W/B)]+[(str(z),z) for z in (F(1,64),F(1,16),F(1,4),F(1,2),F(3,4))] if zero else [('0',F(0))]
    out=[];seen=set()
    for (name,z),eta in product(masses,(F(0),F(1,2),F(1))):
        law=[z*a[i]/W if i in zero else (1-z)*raw[i]/S for i in range(len(es))]
        q=dyadic([(1-eta)*p+eta*u for p,u in zip(can,law)])
        if tuple(q) not in seen:seen.add(tuple(q));out.append((f'zero_mass={name};mix={eta}',q))
    return out

def events(definition,precision,costs):
    output=[]
    for e in definition['events']:
        eps='exact' if e['real_event'] else precision[e['axis']]
        cost=next(z['native_cost'] for z in costs if z['x']==definition['x'] and z['label']==e['label'] and z['epsilon']==eps)
        lo,hi=map(F,(e['ideal_coefficient']['lo'],e['ideal_coefficient']['hi']))
        output.append({**e,'epsilon':eps,'native_cost':cost,'coefficient_mid':str((lo+hi)/2)})
    return output

def make_law(xs,es,q,precision,axis,rule):
    a=[F(e['coefficient_mid']) for e in es];w=[z/p for z,p in zip(a,q)]
    coef=sum(max(abs(z-F(e['ideal_coefficient']['lo'])),abs(F(e['ideal_coefficient']['hi'])-z)) for z,e in zip(a,es))
    synth=2*sum(z*(F(e['native_cost']['strict_event_error_upper'])+F(e['angle_error_upper'])) for z,e in zip(a,es))
    s=v.EPS-coef-synth
    if s<=0:return None
    m2=sum(p*z*z for p,z in zip(q,w));L=max(w)
    n=(ELL*(2*m2/(s*s)+F(4,3)*L/s)).__ceil__()
    if n>10**9:return None
    ec={k:sum(p*F(e['native_cost'][k]) for p,e in zip(q,es)) for k in AXES}
    return {'x':xs,'arm':'CTS_literal_M3','optimized_axis':axis,'precision':precision,'rule':rule,
        'event_source':es,'q_exact':list(map(str,q)),'weights_exact':list(map(str,w)),
        'm2_exact':str(m2),'L_exact':str(L),'shots_per_axis':n,'coefficient_error':str(coef),'synthesis_and_angle_bias':str(synth),'remaining':str(s),
        'resource_total':{k:str(n*(2*ec[k]+(5 if k=='1Q' else 0))) for k in AXES},
        'expected_cost':{k:str(z) for k,z in ec.items()},'workspace_peak':1,'sampler_denominator':str(DEN)}

def certify(c,definition,cost_rows):
    es=c['event_source'];q=list(map(F,c['q_exact']));w=list(map(F,c['weights_exact']));a=[p*z for p,z in zip(q,w)]
    if len(q)!=6 or len(es)!=6 or len(w)!=6 or sum(q)!=1 or min(q)<=0 or any(DEN%p.denominator for p in q):raise ValueError('CTS finite law')
    coef=F(0);synth=F(0);I=v.Interval(0);imag={k:v.Interval(0) for k in ('ZI','IZ','XY','YX')};realZZ=v.Interval(0)
    ratio=F(definition['fixed_rational_rotation_ratio']);norm=v.sqrt_enclosure(1+ratio*ratio)
    for e,z in zip(es,a):
        ref=next(t for t in definition['events'] if t['label']==e['label'])
        if any(e[k]!=ref[k] for k in ('axis','real_event','phase_i_power','rotation_sign','ideal_coefficient','angle_error_upper')):raise ValueError('CTS source semantics changed')
        if e['epsilon']!=('exact' if ref['real_event'] else c['precision'][ref['axis']]):raise ValueError('CTS precision binding')
        saved=next(t for t in cost_rows if (t['x'],t['label'],t['epsilon'])==(c['x'],e['label'],e['epsilon']))
        if e['native_cost']!=saved['native_cost'] or z!=F(e['coefficient_mid']) or z<=0:raise ValueError('CTS cost/reweight binding')
        coef+=max(abs(z-F(ref['ideal_coefficient']['lo'])),abs(F(ref['ideal_coefficient']['hi'])-z))
        synth+=2*z*(F(e['native_cost']['strict_event_error_upper'])+F(e['angle_error_upper']))
        if e['real_event']:
            if e['axis']=='II':I-=z
            else:realZZ-=z
        else:
            part=v.Interval(z)/norm;I+=part;imag[e['axis']]+=part*(-e['rotation_sign']*ratio)
    s=v.EPS-coef-synth;m2=sum(p*z*z for p,z in zip(q,w));L=max(abs(z) for z in w);n=c['shots_per_axis']
    if s<=0 or not 1<=n<=10**9 or coef>F(1,10**25):raise ValueError('CTS feasibility')
    if (F(c['m2_exact'])!=m2 or F(c['L_exact'])!=L or F(c['remaining'])!=s
        or F(c['coefficient_error'])!=coef or F(c['synthesis_and_angle_bias'])!=synth):raise ValueError('CTS stored accounting mismatch')
    margin=n*s*s-ELL*(2*m2+F(4,3)*L*s)
    if margin<0:raise ValueError('CTS confidence')
    target=cts.collected_target(c['x']);bound=F(0)
    for (axis,p),z in target.items():
        actual=I if axis=='II' else realZZ if axis=='ZZ' else imag[axis]
        diff=actual-z.interval();bound+=max(abs(diff.lo),abs(diff.hi))
    if bound>F(1,10**12):raise ValueError('full operator coefficient mean not certified')
    totals={k:n*(2*sum(p*F(e['native_cost'][k]) for p,e in zip(q,es))+(5 if k=='1Q' else 0)) for k in AXES}
    if totals!={k:F(z) for k,z in c['resource_total'].items()} or c['workspace_peak']!=1:raise ValueError('CTS resource accounting')
    return {'pass':True,'Pauli_coefficient_mean_L1_residual_upper':str(bound),'coefficient_L1_error':str(coef),
        'synthesis_plus_angle_bias':str(synth),'m2':str(m2),'L':str(L),'confidence_margin':str(margin),
        'resource_vector_exact':{k:str(z) for k,z in totals.items()},'operator_not_only_channel_mean':True,
        'phase_policy':'strict, controlled relative phases retained','sampling_full_support':True,'quantum_measurements':0}

def run():
    scope=json.loads((OUT/'scope_B.json').read_bytes())
    for path,h in scope['fixed_hashes'].items():
        if sha256((ROOT/path).read_bytes()).hexdigest()!=h:raise PermissionError('CTS source binding '+path)
    if not json.loads((OUT/'result_A.json').read_bytes())['G4_B_conditional_gate']:raise PermissionError('A gate missing')
    defs=json.loads((OUT/'cts_definition.json').read_bytes())
    if defs!={x:cts.definition(x) for x in ('1/8','1/4')}:raise PermissionError('CTS finite specialization changed')
    inventory=planned_keys(defs)
    if list(inventory)!=scope['new_synthesis_keys'] or len(inventory)!=12:raise PermissionError('CTS key expansion')
    contract=json.loads((ROOT/CONTRACT).read_bytes());runtime=verify_runtime(ROOT,contract)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise PermissionError('CTS source dirty')
    with (OUT/'phase_B_consumed.json').open('x') as f:json.dump({'source':head,'scope_sha256':sha256((OUT/'scope_B.json').read_bytes()).hexdigest(),'runs':1,'retries':0},f)
    result={'status':'G4_B_TECHNICAL_INCONCLUSIVE','source_commit':head,'runtime':runtime,'synthesis_rows':[],
        'event_rows':[],'synthesis_attempts':0,'new_native_event_conditions':0,'profiles':0,'law_candidates':0,'rejected':0,
        'retries':0,'new_LP_molecule_DF_GPU_science_signal_trajectory_sampling_calls':0,'mandatory_STOP':True}
    guard=BudgetGuard(scope['caps']);start=time.monotonic()
    try:
        with guard:
            configure(100);cache={}
            for key,(angle,eps) in inventory.items():
                guard.begin_key();result['synthesis_attempts']+=1
                row=synthesize(angle,eps,contract['synthesizer_options'],max_characters=20000);row['usage']=guard.end_key()
                if not row['error_pass']:raise ArithmeticError('CTS strict synthesis guard failure; no retry')
                cache[key]=row;result['synthesis_rows'].append(row)
            for xs,d in defs.items():
                for e in d['events']:
                    for eps in ('exact',) if e['real_event'] else PRECISIONS:
                        circuit=lower(d,e);cost=native_cost(circuit,cache,eps);result['new_native_event_conditions']+=1
                        result['event_rows'].append({'x':xs,'label':e['label'],'epsilon':eps,
                            'native_cost':{k:str(z) if isinstance(z,F) else z for k,z in cost.items()},'workspace_peak':1})
            best=[];labels=('ZI','IZ','XY','YX')
            for xs,d in defs.items():
                for ps in product(PRECISIONS,repeat=4):
                    guard.check();precision=dict(zip(labels,ps));result['profiles']+=1;es=events(d,precision,result['event_rows'])
                    for axis in AXES:
                        winner=None
                        for rule,q in proposals(es,axis):
                            result['law_candidates']+=1
                            if result['law_candidates']>scope['caps']['law_candidates']:raise TimeoutError('CTS law call cap')
                            c=make_law(xs,es,q,precision,axis,rule)
                            if c is None:result['rejected']+=1;continue
                            c['certificate']=certify(c,d,result['event_rows'])
                            if winner is None or F(c['resource_total'][axis])<F(winner['resource_total'][axis]):winner=c
                        if winner:best.append(winner)
            if result['profiles']!=162 or len(best)!=486 or len(result['synthesis_rows'])!=12 or len(result['event_rows'])!=28:raise ArithmeticError('incomplete CTS packet')
            selected={};summary={}
            for xs in defs:
                summary[xs]={}
                for axis in AXES:
                    c=min([c for c in best if c['x']==xs and c['optimized_axis']==axis],key=lambda c:F(c['resource_total'][axis]))
                    key=xs+'/'+axis;selected[key]=c;summary[xs][axis]={k:z for k,z in c.items() if k not in ('event_source','q_exact','weights_exact')}
            v.write(OUT/'CTS_best_per_profile.json',[{k:z for k,z in c.items() if k not in ('event_source','q_exact','weights_exact')} for c in best])
            v.write(OUT/'CTS_selected_complete_laws.json',selected)
            result['summary']=summary;result['usage']=guard.usage();result['wall']=time.monotonic()-start
            result['status']='G4_B_MATCHED_CTS_COMPLETE'
            v.write(OUT/'result_B.json',result)
            if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())>scope['caps']['output_bytes']:raise RuntimeError('G4 output cap')
            print(json.dumps({'status':result['status'],'keys':12,'events':28,'profiles':162,'laws':result['law_candidates']}))
    except Exception as e:
        result['status']='G4_B_TECHNICAL_INCONCLUSIVE';result['error']=str(e);result['prefix_not_final']=True;result['usage']=guard.usage();v.write(OUT/'result_B.json',result);raise
    finally:v.write(OUT/'STOP.json',{'mandatory_STOP':True,'next_science_stage_authorized':False,'retries':0,'research_owner':'GPT','G4_C_design_only':True})

if __name__=='__main__':run()
