"""Conditionally authorized known-return comparator, fixed 12 event conditions.

No Hamiltonian, wavefunction, molecule, DF, trajectory, LP or old run. Only
new returned-zero-degree native IR and 12 primitive synthesis keys are acquired.
"""
from copy import deepcopy
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
spec=importlib.util.spec_from_file_location('g3_A_helpers',Path(__file__).with_name('g3_finite_law.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
OUT=a.OUT
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.rte_reallocation.model import Event
from trottertracks.algorithm_codesign.rte_reallocation.native import Angle,lower_event
from trottertracks.algorithm_codesign.rte_reallocation.numeric import configure,synthesize,synthesis_key
from trottertracks.algorithm_codesign.rte_reallocation.accounting import native_cost
from trottertracks.algorithm_codesign.rte_reallocation.launch import BudgetGuard
from trottertracks.algorithm_codesign.synthesis_placement.wrapper_launch import verify_runtime


def return_ab(xs):
    x=F(xs);chi=F(5,8)
    return 1-chi*x*x/2,x-chi*x**3/6


def keys():
    result={}
    for x,eps,sign in product(('1/8','1/4'),a.g.EPS,(-1,1)):
        aa,b=return_ab(x);angle=Angle('atan',b/aa,F(sign))
        result[synthesis_key(angle,eps)]=(angle,eps)
    if len(result)!=12:raise ValueError('fixed return key inventory')
    return dict(sorted(result.items()))


def event_column(xs,eps,cost_rows):
    aa,b=return_ab(xs);norm=a.g.sqrt_i(aa*aa+b*b)
    direction=[]
    for numer in (aa,b,F(0),F(0)):
        direction.append([str(numer/norm.hi),str(numer/norm.lo)])
    events=[]
    for label,p in enumerate((F(3,4),F(1,4))):
        cost=next(r['native_cost'] for r in cost_rows if (r['x'],r['epsilon'],r['label'])==(xs,eps,label))
        events.append({'source_label':f'0:({label},)','label_probability':str(p),
            'word':[],'rotation':label,'rotation_sign':1,'phase_i_power':0,'complement':False,
            'native_cost':cost})
    return {'id':'RET0:'+eps,'prototype':'RET0','degree':0,'epsilon':eps,
        'saved_ideal_ab_exact':[str(aa),str(b)],'D_intervals':direction,'events':events,
        'workspace_peak':1,'new_event_IRs':2,'new_angle_ratio_fixed':str(b/aa)}


def offdiagonal_column(xs,original):
    """R^2-chi I reuse. Effective polynomial coefficients include low returns."""
    chi=F(5,8);den=1-chi
    col=deepcopy(original);col['id']='RET2:'+original['epsilon'];col['prototype']='RET2'
    source=original['D_intervals']
    col['D_intervals']=[
        [str(chi*F(z)/den) for z in source[2]],
        [str(chi*F(z)/den) for z in source[3]],
        [str(F(z)/den) for z in source[2]],
        [str(F(z)/den) for z in source[3]]]
    col['events']=[deepcopy(e) for e in original['events'] if e['word'][0]!=e['word'][1]]
    if len(col['events'])!=4:raise ValueError('conditional unequal-word support')
    for e in col['events']:e['label_probability']=str(F(e['label_probability'])/den)
    if sum(F(e['label_probability']) for e in col['events'])!=1:raise ValueError('conditional word law')
    col['source_O2_id']=original['id'];col['source_O2_identity']=original['implementation_identity_sha256']
    return col


def return_mean_check(xs):
    x=F(xs);chi=F(5,8);aa,b=return_ab(xs)
    degree=(aa+chi*x*x/2,b+chi*x**3/6,x*x/2,x**3/6)
    if degree!=(F(1),x,x*x/2,x**3/6):raise ValueError('return finite target mismatch')


def return_events(xs,precision,columns):
    return_mean_check(xs)
    output=[]
    for name,weight in [('RET0',F(1)),('RET2',F(3,8))]:
        col=columns[name+':'+precision[name]];aa,b=map(F,col['saved_ideal_ab_exact'])
        norm=a.g.sqrt_i(aa*aa+b*b)
        for e in col['events']:
            p=F(e['label_probability']);scale=weight*p
            output.append({'id':col['id']+'/'+e['source_label'],'column_id':col['id'],
                'coefficient_mid':str(scale*a.mid(norm)),'coefficient_lo':str(scale*norm.lo),
                'coefficient_hi':str(scale*norm.hi),'D_intervals':col['D_intervals'],
                'conditional_probability':str(p),'degree':col['degree'],
                **{k:e[k] for k in ('word','rotation','phase_i_power','rotation_sign','complement','native_cost')}})
    return output


def run():
    scope=json.loads((OUT/'phase_B_scope.json').read_bytes())
    for path,h in scope['fixed_hashes'].items():
        if sha256((ROOT/path).read_bytes()).hexdigest()!=h:raise PermissionError('Phase B binding: '+path)
    previous=json.loads((OUT/'phase_A_result.json').read_bytes())
    if previous['status']!='G3_PHASE_A_FINITE_LAW_DIAGNOSIS_COMPLETE' or not previous['Phase_B_technical_gate']:
        raise PermissionError('Phase A conditional gate not met')
    contract=json.loads((ROOT/a.g.R1_CONTRACT).read_bytes())
    runtime=verify_runtime(ROOT,contract)
    if list(keys())!=scope['new_synthesis_keys']:raise PermissionError('return key scope changed')
    if (OUT/'phase_B_consumed.json').exists():raise FileExistsError('Phase B marker consumed; no retry')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise PermissionError('source worktree dirty')
    with (OUT/'phase_B_consumed.json').open('x') as f:json.dump({'source_commit':head,'scope_sha256':sha256((OUT/'phase_B_scope.json').read_bytes()).hexdigest(),'runs':1,'retries':0},f)
    start=time.monotonic();raw=json.loads((ROOT/a.g.RAW).read_bytes())
    cache={r['key']:r for r in raw['synthesis_rows']}
    result={'status':'G3_PHASE_B_TECHNICAL_INCONCLUSIVE','source_commit':head,'runtime_identity':runtime,
        'synthesis_rows':[],'event_rows':[],'synthesis_attempts':0,'native_event_IRs_built':0,
        'retries':0,'mandatory_STOP':True,'next_stage_authorized':False,
        'science_Hamiltonian_DF_molecule_trajectory_LP_GPU_calls':0,
        'law_candidates':0,'rejected_law_candidates':0,'quantum_shots_executed':0,
        'full_registered_LP_or_old_run_reexecution':False}
    guard=BudgetGuard(scope['caps'])
    try:
        with guard:
            configure(100)
            for key,(angle,eps) in keys().items():
                guard.check();guard.begin_key();result['synthesis_attempts']+=1
                row=synthesize(angle,eps,contract['synthesizer_options'],max_characters=20000)
                row['resource_usage']=guard.end_key()
                if not row['error_pass']:raise ArithmeticError('strict returned synthesis guard failed; no retry')
                cache[key]=row;result['synthesis_rows'].append(row)
            for xs,eps,label in product(('1/8','1/4'),a.g.EPS,(0,1)):
                aa,b=return_ab(xs);p=(F(3,4),F(1,4))[label]
                event=Event('known_return',f'0:({label},)',aa,b,p,rotation=label)
                circuit=lower_event('distinct_basis',event,True)
                cost=native_cost(circuit,cache,eps);result['native_event_IRs_built']+=1
                result['event_rows'].append({'x':xs,'epsilon':eps,'label':label,
                    'returned_ratio':str(b/aa),'native_cost':{k:str(v) if isinstance(v,F) else v for k,v in cost.items()},
                    'phase_policy':'strict operator error; no global phase minimization',
                    'workspace_peak':1,'I0_implementation':'Q0=ZI,Q1=V^dag IZ V; original basis/conjugator unchanged'})
            table=json.loads((ROOT/a.g.TABLE).read_bytes());best=[];count=0;all_columns={}
            for xs,t in table['tables'].items():
                cols={}
                for eps in a.g.EPS:
                    ret0=event_column(xs,eps,result['event_rows']);cols[ret0['id']]=ret0
                    source=next(c for c in t['columns'] if c['id']=='O2:'+eps)
                    ret2=offdiagonal_column(xs,source);cols[ret2['id']]=ret2
                all_columns[xs]=cols
                for e0,e2 in product(a.g.EPS,repeat=2):
                    precision={'RET0':e0,'RET2':e2};es=return_events(xs,precision,cols)
                    for axis in a.AXES:
                        winner=None
                        for rule,q in a.proposals(es,axis):
                            guard.check();count+=1
                            result['law_candidates']=count
                            if count>scope['caps']['law_candidates']:raise TimeoutError('return law call cap')
                            ident=f'{xs}/known_return/RET0={e0},RET2={e2}/{axis}/{rule}'
                            c=a.candidate(es,q,xs,ident,axis,rule)
                            if c is None:result['rejected_law_candidates']+=1;continue
                            c['arm']='known_return';c['precision']=precision
                            c['certificate']=a.certify(c,source_columns=cols)
                            c['certificate']['native_semantics']='new fixed returned-zero-degree strict synthesized IR; off-diagonal saved O2 IR/error reused'
                            if winner is None or F(c['resource_total'][axis])<F(winner['resource_total'][axis]):winner=c
                        if winner:best.append(winner)
            if len(result['synthesis_rows'])!=12 or len(result['event_rows'])!=12 or len(best)!=54:
                raise ArithmeticError('incomplete fixed return inventory; prefix not final')
            result['status']='G3_KNOWN_RETURN_COMPARATOR_COMPLETE'
            result['profiles']=18;result['law_candidates']=count
            result['acquisition']={'p_access':'existing exact toy p,2 entries','chi_exact':'5/8','chi_sum_squared_operations':2,
                'conditional_word_law':'unequal0/1 words each1/2; rotation label independent3/4,1/4',
                'offdiagonal_IRs_newly_built':0,'offdiagonal_original_O2_reuse':True,
                'new_angle_search_or_new_dictionary':False,'general_DF_or_classical_scale_advantage':False}
            dump=a.dump
            dump(OUT/'phase_B_return_columns.json',all_columns)
            dump(OUT/'phase_B_best_laws.json',best)
            if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())>scope['caps']['output_bytes']:
                raise RuntimeError('packet output cap; prefix not final')
            result['resource_usage']=guard.usage();result['wall_seconds']=time.monotonic()-start
            dump(OUT/'phase_B_result.json',result)
            print(json.dumps({'status':result['status'],'keys':len(result['synthesis_rows']),
                'event_conditions':len(result['event_rows']),'return_laws':count}))
    except Exception as e:
        result['error_type']=type(e).__name__;result['error']=str(e)
        result['status']='G3_PHASE_B_TECHNICAL_INCONCLUSIVE'
        result['resource_usage']=guard.usage();result['wall_seconds']=time.monotonic()-start
        result['partial_prefix_not_final_comparison']=True
        a.dump(OUT/'phase_B_result.json',result)
        raise
    finally:
        a.dump(OUT/'STOP.json',{'mandatory_STOP':True,'next_stage_authorized':False,'retries':0,'return_to':'GPT_G3'})


if __name__=='__main__':run()
