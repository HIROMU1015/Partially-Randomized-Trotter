"""New post-hoc finite-law diagnosis. No old run, LP, or science execution.

Phase A uses saved event data only. Cost-zero events stay zero. Rational
coefficients and dyadic proposals are verified by a separate exact verifier.
"""
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
from pathlib import Path
import csv
import importlib.util
import json
import resource
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/track_b_g3_finite_law/2026-10-09'
spec = importlib.util.spec_from_file_location('g2_arithmetic', Path(__file__).with_name('g2_saved_diagnostic.py'))
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)  # pure helpers only; old main/markers are never called
N = 2**60
E = F(1,200)
ELL = g.log_integer(10560).hi
AXES = ('T','CX','1Q')
ARMS = ('ordinary','PTSC_K0','A','J1')
MIX = (F(0),F(1,2),F(1))
ZERO_MASSES = (F(1,64),F(1,16),F(1,4),F(1,2),F(3,4))


def dump(path, value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def mid(v):
    return (v.lo+v.hi)/2


def dyadic(q):
    """Largest remainder with fixed index ties. Never clip/repair lost support."""
    q=list(map(F,q))
    if not q or min(q)<=0:raise ValueError('positive finite support required')
    q=[v/sum(q) for v in q]
    scaled=[v*N for v in q]; counts=[v.__floor__() for v in scaled]
    left=N-sum(counts)
    indices=sorted(range(len(q)),key=lambda i:(-(scaled[i]-counts[i]),i))
    for i in indices[:left]:counts[i]+=1
    if min(counts)<=0:raise ValueError('dyadic rounding lost positive support')
    return [F(c,N) for c in counts]


def make_events(xs, weights, precision, columns):
    g.mean_check(F(xs),weights)
    result=[]
    for name in g.ORDER:
        if not weights[name]:continue
        col=columns[name+':'+precision[name]]
        a,b=map(F,col['saved_ideal_ab_exact'])
        norm=g.sqrt_i(a*a+b*b)
        for e in col['events']:
            p=F(e['label_probability']);scale=weights[name]*p
            # Midpoint the group norm BEFORE multiplying conditional IID law.
            # This retains exact conditional shares under rational reweighting.
            alo,ahi=scale*norm.lo,scale*norm.hi
            amid=scale*mid(norm)
            degree=col['degree']
            result.append({'id':col['id']+'/'+e['source_label'], 'column_id':col['id'],
                'coefficient_mid':str(amid), 'coefficient_lo':str(alo),
                'coefficient_hi':str(ahi),'D_intervals':col['D_intervals'],
                'conditional_probability':str(p),
                'degree':degree,'word':e['word'],'rotation':e['rotation'],
                'phase_i_power':e['phase_i_power'],'rotation_sign':e['rotation_sign'],
                'complement':e['complement'],'native_cost':e['native_cost']})
    return result


def proposals(events, axis):
    coeff=[F(e['coefficient_mid']) for e in events]
    B=sum(coeff); canonical=[a/B for a in coeff]
    zero=[i for i,e in enumerate(events) if F(e['native_cost'][axis])==0]
    positive=[i for i in range(len(events)) if i not in zero]
    if not positive:
        return [('all_zero_cost_canonical',dyadic(canonical))]
    raw={i:coeff[i]/mid(g.sqrt_i(events[i]['native_cost'][axis])) for i in positive}
    S=sum(raw.values());pos={i:raw[i]/S for i in positive}
    if zero:
        W=sum(coeff[i] for i in zero)
        masses=(W/B,)+ZERO_MASSES
    else:
        masses=(F(0),)
    output=[]; seen=set()
    for z,eta in product(masses,MIX):
        law=[z*coeff[i]/W if i in zero else (1-z)*pos[i] for i in range(len(events))]
        q=dyadic([(1-eta)*base+eta*value for base,value in zip(canonical,law)])
        ident=tuple(q)
        if ident in seen:continue
        seen.add(ident)
        output.append((f'zero_mass={z};mix={eta}',q))
    return output


def candidate(events, q, xs, ident, axis, rule):
    a=[F(e['coefficient_mid']) for e in events]
    w=[v/p for v,p in zip(a,q)]
    mean_radius=sum(max(abs(v-F(e['coefficient_lo'])),abs(F(e['coefficient_hi'])-v)) for v,e in zip(a,events))
    synthesis=2*sum(v*F(e['native_cost']['strict_event_error_upper']) for v,e in zip(a,events))
    s=E-mean_radius-synthesis
    if s<=0:return None
    m2=sum(v*v/p for v,p in zip(a,q));L=max(w)
    n=(ELL*(2*m2/(s*s)+F(4,3)*L/s)).__ceil__()
    if n>10**9:return None
    ec={k:sum(p*F(e['native_cost'][k]) for p,e in zip(q,events)) for k in AXES}
    totals={k:n*(2*ec[k]+(5 if k=='1Q' else 0)) for k in AXES}
    return {'id':ident,'x':xs,'optimized_axis':axis,'proposal_rule':rule,
            'q_exact':list(map(str,q)),'weights_exact':list(map(str,w)),
            'coefficient_mean_bias_upper':str(mean_radius),'synthesis_bias_upper':str(synthesis),
            'remaining_stat':str(s),'m2_exact':str(m2),'L_exact':str(L),
            'shots_per_axis':n,'expected_cost':{k:str(v) for k,v in ec.items()},
            'resource_total':{k:str(v) for k,v in totals.items()},
            'workspace_peak':1,'sampler_denominator':str(N),
            'target':'same finite P3; no exponential truncation claim',
            'event_source':events,'quantum_measurements_executed':0}


def certify(c, caps=None, source_columns=None):
    """Independent verification from emitted events/q/weights, not candidate totals.

    Applies to weighted event IS, not the old constant-weight q,y certificate.
    """
    es=c['event_source'];q=list(map(F,c['q_exact']));w=list(map(F,c['weights_exact']))
    if len(q)!=len(es) or len(w)!=len(es) or min(q)<=0 or sum(q)!=1 or any(N%v.denominator for v in q):
        raise ValueError('invalid dyadic law/support')
    a=[p*v for p,v in zip(q,w)]
    if min(a)<=0:raise ValueError('positive corrected coefficients required')
    if any(v!=F(e['coefficient_mid']) for v,e in zip(a,es)):
        raise ValueError('exact reweighting fails')
    groups={}
    for e,v in zip(es,a):
        share=F(e['conditional_probability'])
        if share<=0:raise ValueError('invalid conditional share')
        group=e['column_id'];ratio=v/share
        if group in groups and groups[group]!=ratio:raise ValueError('conditional law not preserved')
        groups[group]=ratio
        if source_columns is not None:
            col=source_columns[group]
            label=e['id'].split('/',1)[1]
            match=[z for z in col['events'] if z['source_label']==label]
            if len(match)!=1:raise ValueError('source event missing')
            original=match[0]
            if share!=F(original['label_probability']) or e['D_intervals']!=col['D_intervals']:
                raise ValueError('source law/direction changed')
            for key in ('word','rotation','phase_i_power','rotation_sign','complement','native_cost'):
                if e[key]!=original[key]:raise ValueError('saved phase/cost/error identity changed')
    for group in groups:
        if sum(F(e['conditional_probability']) for e in es if e['column_id']==group)!=1:
            raise ValueError('conditional support incomplete')
    # Direct first-degree moment enclosure uses saved normalized coefficients.
    target=[F(1),F(c['x']),F(c['x'])**2/2,F(c['x'])**3/6]
    residual=[]
    for k,t in enumerate(target):
        lo=sum(v*F(e['D_intervals'][k][0]) for v,e in zip(a,es))-t
        hi=sum(v*F(e['D_intervals'][k][1]) for v,e in zip(a,es))-t
        residual.append(max(abs(lo),abs(hi)))
    xi=sum(residual)
    coef_error=sum(max(abs(v-F(e['coefficient_lo'])),abs(F(e['coefficient_hi'])-v)) for v,e in zip(a,es))
    if xi>F(1,10**12) or coef_error>F(1,10**25):raise ValueError('numerical mean/coefficient cap')
    # Observable bound from coefficient approximation + strict joint gate error.
    bias=coef_error+sum(v*2*F(e['native_cost']['strict_event_error_upper']) for v,e in zip(a,es))
    stat=E-bias
    if stat<=0:raise ValueError('bias exhausted')
    m2=sum(p*v*v for p,v in zip(q,w));L=max(abs(v) for v in w)
    n=c['shots_per_axis']
    confidence=n*stat*stat-ELL*(2*m2+F(4,3)*L*stat)
    if type(n)!=int or not 1<=n<=10**9 or confidence<0:raise ValueError('confidence/shot certificate')
    totals={k:n*(2*sum(p*F(e['native_cost'][k]) for p,e in zip(q,es))+(5 if k=='1Q' else 0)) for k in AXES}
    if totals!={k:F(v) for k,v in c['resource_total'].items()}:raise ValueError('resource mismatch')
    if caps and any(totals[k]>F(v) for k,v in caps.items()):raise ValueError('other resource cap')
    if c['workspace_peak']!=1:raise ValueError('workspace capacity')
    return {'certified':True,'exact_sampling_sum':str(sum(q)),
            'exact_weighted_first_mean_cancellation':True,
            'source_phase_cost_error_binding_verified':source_columns is not None,
            'conditional_IID_shares_preserved_exactly':True,
            'ideal_degree_mean_semantics':'exact algebra; implemented numeric coefficient approximation bounded separately',
            'degree_mean_residual_upper':str(xi),'coefficient_L1_error_upper':str(coef_error),
            'total_bias_upper':str(bias),'confidence_margin_exact':str(confidence),
            'm2_exact':str(m2),'L_exact':str(L),'shots_per_axis':n,
            'resource_vector_exact':{k:str(v) for k,v in totals.items()},
            'new_cap_feasibility':'no unspecified query cap invented; optional declared cap checks only',
            'native_semantics':'saved R1 phase-preserving IR/error bound; no new circuit execution'}


def dominates(left,right):
    a,b=left['resource_total'],right['resource_total']
    return all(F(a[k])<=F(b[k]) for k in AXES) and any(F(a[k])<F(b[k]) for k in AXES)


def summarize(candidates):
    summaries={};selected={}
    for xs in ('1/8','1/4'):
        rows=[c for c in candidates if c['x']==xs]
        original=[c for c in rows if c['arm']!='J1'];j1=[c for c in rows if c['arm']=='J1']
        record={'axes':{},'B2_full_mixture_global_optimality':False}
        for k in AXES:
            old=min(original,key=lambda c:F(c['resource_total'][k]));new=min(j1,key=lambda c:F(c['resource_total'][k]))
            ratio=F(new['resource_total'][k])/F(old['resource_total'][k])
            nondominated=not any(dominates(c,new) for c in original)
            record['axes'][k]={'original_id':old['id'],'J1_id':new['id'],
                'original_totals':old['resource_total'],'J1_totals':new['resource_total'],
                'J1_over_original_exact':str(ratio),'ratio_display':float(ratio),
                'strict_cost_difference':ratio<1,'J1_not_dominated_by_original_candidate_set':nondominated,
                'gap_magnitude_is_not_scientific_materiality_decision':True}
            selected[old['id']]=old;selected[new['id']]=new
        summaries[xs]=record
    return summaries,list(selected.values())


def verify_bindings(scope):
    for p,h in scope['fixed_hashes'].items():
        if sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise PermissionError('source/input mismatch: '+p)


def run_A():
    scope=json.loads((OUT/'scope_v1.json').read_bytes());verify_bindings(scope)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise PermissionError('source worktree dirty')
    with (OUT/'phase_A_consumed.json').open('x') as f:json.dump({'source_commit':source,'scope_sha256':sha256((OUT/'scope_v1.json').read_bytes()).hexdigest(),'runs':1,'retries':0},f)
    start,cpu=time.monotonic(),time.process_time()
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,resource.getrlimit(resource.RLIMIT_AS)[1]))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Phase A cap')));signal.alarm(600)
    best=[];attempts=0;failures=[];profiles=0
    try:
        table=json.loads((ROOT/g.TABLE).read_bytes());identity=g.verify_inputs(table,(ROOT/g.RAW).read_bytes())
        with (OUT/'phase_A_all_candidates.csv').open('w',newline='') as f:
            names=['id','x','arm','axis','precision','proposal','shots','m2','L','mean_bias','T','CX','1Q']
            writer=csv.DictWriter(f,fieldnames=names,lineterminator='\n');writer.writeheader()
            for xs,t in table['tables'].items():
                cols={c['id']:c for c in t['columns']}
                for arm in ARMS:
                    weights=g.vertices(F(xs))[arm];active=[v for v in g.ORDER if weights[v]]
                    for eps in product(g.EPS,repeat=len(active)):
                        precision=dict(zip(active,eps));profiles+=1
                        es=make_events(xs,weights,precision,cols)
                        for axis in AXES:
                            winner=None
                            for rule,q in proposals(es,axis):
                                attempts+=1
                                if attempts>15000 or time.process_time()-cpu>480:raise TimeoutError('Phase A call/CPU cap')
                                ident=f'{xs}/{arm}/'+','.join(v+'='+precision[v] for v in active)+'/'+axis+'/'+rule
                                c=candidate(es,q,xs,ident,axis,rule)
                                if c is None:failures.append({'id':ident,'reason':'BIAS_OR_SHOT_CAP'});continue
                                c['arm']=arm;c['precision']=precision;c['certificate']=certify(c,source_columns=cols)
                                vals={k:float(F(c['resource_total'][k])) for k in AXES}
                                writer.writerow({'id':ident,'x':xs,'arm':arm,'axis':axis,'precision':str(precision),'proposal':rule,'shots':c['shots_per_axis'],
                                    'm2':float(F(c['m2_exact'])),'L':float(F(c['L_exact'])),'mean_bias':float(F(c['certificate']['degree_mean_residual_upper'])),**vals})
                                if winner is None or F(c['resource_total'][axis])<F(winner['resource_total'][axis]):winner=c
                            if winner:best.append(winner)
        if profiles!=288:raise ValueError('fixed profile inventory mismatch')
        summaries,selected=summarize(best)
        gate=any(v['strict_cost_difference'] and v['J1_not_dominated_by_original_candidate_set'] for x in summaries.values() for k,v in x['axes'].items() if k in ('T','1Q'))
        dump(OUT/'phase_A_selected_laws.json',selected)
        dump(OUT/'phase_A_best_per_profile.json',[
            {k:v for k,v in c.items() if k not in ('event_source','q_exact','weights_exact')}
            | {'law_digest':g.digest({'events':c['event_source'],'q':c['q_exact'],'weights':c['weights_exact']})}
            for c in best])
        dump(OUT/'phase_A_identity_audit.json',identity)
        dump(OUT/'phase_A_result.json',{'status':'G3_PHASE_A_FINITE_LAW_DIAGNOSIS_COMPLETE','source_commit':source,
            'profiles':profiles,'law_candidates':attempts,'rejected_candidate_count':len(failures),'rejected_candidates':failures,
            'summary':summaries,'Phase_B_technical_gate':gate,
            'gate_meaning':'finite certified T/1Q strict candidate-set difference plus nondominance; no materiality/science-GO/global-B2 claim',
            'full_B2_mixture_or_all_finite_IS_optimality':False,'zero_cost_not_modified':True,
            'measurement_synthesis_LP_matrix_circuit_GPU_calls':0,'sampler_D':str(N),'actual_quantum_shots_executed':0,
            'wall_seconds':time.monotonic()-start,'CPU_seconds':time.process_time()-cpu,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'runs':1,'retries':0,
            'Phase_B_synthesis_authorized_conditionally_by_user_review':True,'research_decision_owner':'GPT_G3'})
        print(json.dumps({'status':'PHASE_A_COMPLETE','profiles':profiles,'law_candidates':attempts,'Phase_B_technical_gate':gate}))
    except Exception as e:
        dump(OUT/'phase_A_failure.json',{'status':'G3_PHASE_A_TECHNICAL_INCONCLUSIVE','error':str(e),'retries':0,'partial_prefix_not_used':True,'mandatory_STOP':True})
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':run_A()
