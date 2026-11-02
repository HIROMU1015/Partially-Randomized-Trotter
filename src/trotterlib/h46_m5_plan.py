"""Metadata-only m5_best schedules, candidate scopes and parallel dependencies.

No chemistry, RNG, circuit construction, compilation, fitting or real-data ranking.
The module is loaded directly to avoid scientific package imports.
"""
import hashlib
import json
import math
import ast
from functools import lru_cache
from pathlib import Path

TAIL=(.11225428783206336,.14181961643121604,.1211920365713068,
      .13781643533753404,.12547397927357543)
FORMULA_ID='PF_M5_BEST_20D64C2_BINARY64_V1'
FORMULAS=('2nd','4th','4th(m5_best)')
SYSTEMS={'H4':{'rank':12,'anchor':3,'neighbors':(2,4),'cuts':(2,3,4,6,9,11)},
         'H6':{'rank':19,'anchor':5,'neighbors':(3,4),'cuts':tuple(range(5,16))}}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def weights(label):
    if label=='2nd':return (1.,)
    if label=='4th':
        u=2**(1/3);a=1/(2-u);return(a,-u/(2-u),a)
    if label=='4th(m5_best)':return tuple(reversed(TAIL))+(1-2*sum(TAIL),)+TAIL
    raise ValueError('FIXED_FORMULA_LABEL')


def allocation(w,total):
    """Execute only the frozen pure allocation function; never import chemistry."""
    if any(not math.isfinite(v) or v==0 for v in w):raise ValueError('FINITE_NONZERO_WEIGHTS')
    return frozen_allocation()(w,total)


@lru_cache(maxsize=1)
def frozen_allocation():
    path=Path(__file__).resolve().parents[2]/'scripts/h46_new2_core.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='allocation')
    scope={'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path), 'exec'),scope)
    return scope['allocation']


def schedule(label,q,policy,value,T=.8):
    if type(q) is not int or q<1 or not math.isfinite(T) or T==0:raise ValueError('SCHEDULE_SCOPE')
    if type(value) is not int or value<1:raise ValueError('INTEGER_RTE_DENSITY')
    w=weights(label)
    if policy=='PER_PIECE':counts=(value,)*len(w)
    elif policy=='ABS_TIME_PER_OUTER':counts=allocation(w,value)
    else:raise ValueError('RTE_POLICY_REQUIRED')
    return [{'outer':outer,'piece':piece,'signed_time':T*coefficient/q,'rte_steps':count}
            for outer in range(q) for piece,(coefficient,count) in enumerate(zip(w,counts))]


def lowered_schedule(label,q,policy,value,*,mode,scalar,tail_identity=0.,T,lower_s2):
    """Artificial/callback adapter: preserve order, scalar and control in each piece.

    Caller supplies a lowerer; no production dispatcher is installed by this plan.
    PR tail sampling and B0/B0I preparation remain responsibilities of sealed adapters.
    """
    if mode not in ('FULL','B0','B0I','PR'):raise ValueError('METHOD_SCOPE')
    if not math.isfinite(scalar) or not math.isfinite(tail_identity):raise ValueError('SCALAR_FINITE')
    effective=scalar+(tail_identity if mode in ('B0I','PR') else 0.)
    pieces=schedule(label,q,policy,value,T)
    if mode!='PR':pieces=[{**piece,'rte_steps':0} for piece in pieces]
    return [lower_s2(**piece,mode=mode,scalar_time=effective*piece['signed_time'],
                     control_policy='symmetric_directional') for piece in pieces]


def candidate(system,mode,label,prefix,q,K=None,policy=None,value=None):
    v={'system':system,'method':mode,'formula':label,'prefix':prefix,'q':q,'K':K,'policy':policy,'r_value':value}
    return {'id':system+'_'+mode+'_'+digest(v)[:16],**v}


def pilot(system):
    if system not in SYSTEMS:raise ValueError('SYSTEM_SCOPE')
    s=SYSTEMS[system];det=[];pr=[]
    for label in FORMULAS:
        for mode in ('FULL','B0','B0I'):
            det.append(candidate(system,mode,label,s['rank'] if mode=='FULL' else s['anchor'],2))
        pr.append(candidate(system,'PR',label,s['anchor'],2,4,'PER_PIECE',2))
    return {'deterministic':det,'PR':pr,'PR_technical_trajectories_per_group':2,
            'reference_jobs':1,'signal_jobs':12,'native_evolutions':15,'compile_axes':30,
            'representative_correctness_trajectories':12,'correctness_comparisons_proposed':84,
            'independent_confirmation':False,'ranking_authorized':False}


def screen(system):
    s=SYSTEMS[system];pr=[];det=[]
    for f in FORMULAS:
        for q in (1,2,4,8):
            for policy,value in (('PER_PIECE',2),('PER_PIECE',4),('ABS_TIME_PER_OUTER',16)):
                pr.append(candidate(system,'PR',f,s['anchor'],q,4,policy,value))
        for p in s['neighbors']:
            for q in (2,8):pr.append(candidate(system,'PR',f,p,q,4,'PER_PIECE',4))
        for q in (1,4):pr.append(candidate(system,'PR',f,s['anchor'],q,2,'PER_PIECE',4))
    for q in (1,2,4,8):
        det.append(candidate(system,'FULL','4th(m5_best)',s['rank'],q))
        for p in s['cuts']:
            for mode in ('B0','B0I'):det.append(candidate(system,mode,'4th(m5_best)',p,q))
    return {'PR':pr,'new_M5_deterministic':det,'existing_S2_Y4_deterministic_domains_retained':True,
            'automatic_q16_extension':False,'real_science_execution_approved':False}


def graph():
    nodes=[]
    for system in SYSTEMS:
        for lane in ('DET','PR'):
            pid=system+'_P2_'+lane
            nodes.append({'id':pid,'phase':2,'depends':[],'approval_required':True,
                          'own_grant_ledger_output':True,'stop_domain':pid,'end_STOP':True})
        signal=system+'_P3_SIGNAL'
        nodes.append({'id':signal,'phase':3,'depends':[system+'_P2_DET',system+'_P2_PR'],
                      'approval_required':True,'end_STOP':True})
        for lane in ('DET_COST','PR_COST'):
            nodes.append({'id':system+'_P3_'+lane,'phase':3,'depends':[signal],
                          'approval_required':True,'scope_to_be_fixed_after_pilot':True,'end_STOP':True})
        nodes.append({'id':system+'_P3_COMPARE','phase':3,'depends':[system+'_P3_DET_COST',system+'_P3_PR_COST'],
                      'approval_required':True,'end_STOP':True})
        nodes.append({'id':system+'_P4_CONFIRM','phase':4,'depends':[system+'_P3_COMPARE'],
                      'approval_required':True,'selection_and_sample_design_required':True,'end_STOP':True})
    return nodes


def ready(nodes,completed,approved,failed=(),*,active_phase=2):
    done=set(completed);permit=set(approved);bad=set(failed)
    return [n['id'] for n in nodes if n['phase']==active_phase and n['id'] in permit and n['id'] not in done|bad
            and set(n['depends'])<=done and not set(n['depends'])&bad]


def specification():
    return {'status':'PLAN_ONLY_UNAPPROVED','formula_id':FORMULA_ID,'pf_label':'4th(m5_best)',
            'systems':SYSTEMS,'pilot':{s:pilot(s) for s in SYSTEMS},'screen':{s:screen(s) for s in SYSTEMS},
            'parallel_graph':graph(),'production_adapter_sealed':False,'valid_grants':0,
            'scientific_execution':0,'mandatory_STOP':True}
