"""G5 saved-only class closure. No G1/G2/G3 evaluator, solver or science import.

G4 stdlib interval/root/log primitives are shared and source-bound. New vertex,
price and CTS coefficient verification; never calls any previous runner.
"""
from fractions import Fraction as F
from hashlib import sha256
from itertools import combinations,product
from pathlib import Path
import importlib.util
import json
import resource
import signal
import subprocess
import sys
import time

# Fixed, trusted rational certificates can exceed Python's default 4300 digits.
# Keep a finite serialization limit in addition to the output/RSS caps.
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(50000)

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/track_b_g5_fixed_dictionary_closure/2026-10-10'
G4='artifacts/track_b_g4_conditional_separation/2026-10-09/'
spec=importlib.util.spec_from_file_location('g4_rational',Path(__file__).with_name('g4_independent_certificate.py'))
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
ORDER=('O0','O2','P2','P3','A0','A1','A2');PRECISIONS=('1e-3','1e-4','1e-6')
RADII={'T':F(2150),'CX':F(350),'1Q':F(3500)}

def data(n):return json.loads((ROOT/n).read_bytes())
def put(n,d):v.write(OUT/n,d)

def formal_columns(x):
    x=F(x)
    if x<=0:raise ValueError('positive fixed formal x required')
    rho=(x+x**3/6)/(1+x*x/2)
    return {'O0':(0,F(1),x),'O2':(2,x*x/2,x**3/6),'P2':(2,x*x/2,F(0)),
        'P3':(3,x**3/6,F(0)),'A0':(0,F(1),rho),
        'A1':(1,x-rho,(x-rho)/rho),'A2':(2,x**3/(6*rho),x**3/6)}

def degree_system(x):
    proto=formal_columns(x);D=[[F(0)]*7 for _ in range(4)]
    for j,name in enumerate(ORDER):
        k,a,b=proto[name];D[k][j]=a
        if b:D[k+1][j]=b
    return D,[F(1),F(x),F(x)**2/2,F(x)**3/6]

def solve_square(A,t):
    """Exact rational Gaussian elimination of a coefficient identity, not LP."""
    a=[list(row)+[z] for row,z in zip(A,t)];n=len(t)
    for j in range(n):
        piv=next((i for i in range(j,n) if a[i][j]),None)
        if piv is None:return None
        a[j],a[piv]=a[piv],a[j];pivot=a[j][j];a[j]=[z/pivot for z in a[j]]
        for i in range(n):
            if i!=j:
                m=a[i][j];a[i]=[z-m*w for z,w in zip(a[i],a[j])]
    return [row[-1] for row in a]

def vertices(x):
    D,t=degree_system(x);found=set()
    # All 35 possible four-column bases of rank-4 D; no optimization/backend.
    for chosen in combinations(range(7),4):
        z=solve_square([[row[j] for j in chosen] for row in D],t)
        if z is None or min(z)<0:continue
        full=[F(0)]*7
        for j,q in zip(chosen,z):full[j]=q
        if [sum(d*q for d,q in zip(row,full)) for row in D]!=t:raise ValueError('vertex mean')
        found.add(tuple(full))
    mu=(F(x)**2+2)/(F(x)**2+6)
    def point(s,r,b):return (s,b,mu+(1-mu)*s-mu*r-b,1-r-b,1-s,1-s,r)
    expected={name:point(*map(F,p)) for name,p in {'ordinary':(1,0,1),'PTSC_K0':(1,0,0),'A':(0,1,0),
        'J1':(0,0,0),'J2':(0,0,mu),'J3':(1,1,0)}.items()}
    if found!=set(expected.values()):raise ValueError('six vertex completeness mismatch')
    # Every column sum is strictly positive: D gamma=t and gamma>=0 is bounded.
    if any(sum(D[i][j] for i in range(4))<=0 for j in range(7)):raise ValueError('unboundedness proof unavailable')
    return {name:{k:z for k,z in zip(ORDER,g) if z} for name,g in expected.items()}

def price(x,vertex,gamma,precision,cols,axis):
    K=v.Interval(0);bias=v.Interval(0)
    for name,g in gamma.items():
        _,a,b=formal_columns(x)[name];norm=v.sqrt_enclosure(a*a+b*b)
        c=cols[name+':'+precision[name]];h=v.Interval(0);d=F(0)
        for e in c['events']:
            p=F(e['label_probability']);h+=p*v.sqrt_enclosure(e['native_cost'][axis])
            d+=2*p*F(e['native_cost']['strict_event_error_upper'])
        K+=g*norm*h;bias+=g*norm*d
    s=v.Interval(v.EPS)-bias
    if s.lo<=0:raise ValueError('positive accuracy unresolved')
    r=K/s
    return {'vertex':vertex,'precision':precision,'axis':axis,'K':K.json(),'remaining':s.json(),'ratio':r.json(),'Phi':(r*r).json()}

def bridge(cols,axis,r):
    rows=[]
    for c in cols.values():
        for e in c['events']:
            d=2*F(e['native_cost']['strict_event_error_upper']);C=F(e['native_cost'][axis])
            slack=r*r*(1-d)**2-C
            rows.append({'column_id':c['id'],'source_label':e['source_label'],'C':str(C),'d':str(d),
                'squared_slack':str(slack),'pass':0<=d<=1 and C>=0 and slack>=0})
    return {'pass':all(z['pass'] for z in rows),'events':len(rows),'minimum_squared_slack':str(min(F(z['squared_slack']) for z in rows)), 'rows':rows}

def cts_targets(x):
    """Independent radical enclosures of the fixed full P3 Pauli coefficients."""
    x=F(x);r2=v.sqrt_enclosure(2,bits=768)
    c=v.Interval(v.sqrt_enclosure(2+r2.lo,bits=768).lo/2,v.sqrt_enclosure(2+r2.hi,bits=768).hi/2)
    s=v.Interval(v.sqrt_enclosure(2-r2.hi,bits=768).lo/2,v.sqrt_enclosure(2-r2.lo,bits=768).hi/2)
    return {'II':v.Interval(1-F(5,16)*x*x),'ZZ':-3*c*x*x/16,
        'ZI':x*(x*x*(c*c+5)-48)/64,'IZ':c*x*(7*x*x-24)/96,
        'XY':s*x*(5*x*x-48)/192,'YX':c*s*x**3/64},c,s

def ideal_cts_intervals(x):
    target,_,_=cts_targets(x)
    signed={k:target[k] for k in ('ZI','IZ','XY','YX')}
    if any(z.lo<=0<=z.hi for z in signed.values()):raise ValueError('CTS target sign unresolved')
    absolute={k:(-z if z.hi<0 else z) for k,z in signed.items()}
    ls=sum(absolute.values(),v.Interval(0));ls2=1+ls*ls
    norm=v.Interval(v.sqrt_enclosure(ls2.lo,bits=768).lo,v.sqrt_enclosure(ls2.hi,bits=768).hi)
    coeff={'real_minus_II':v.Interval(F(5,16)*F(x)**2),'real_minus_ZZ':-target['ZZ']}
    coeff.update({'rotation_'+k:norm*z/ls for k,z in absolute.items()})
    return coeff,ls,{k:(1 if z.hi<0 else -1) for k,z in signed.items()}

def verify_cts(c,definition,costs,expected_x='1/4'):
    es=c['event_source'];q=list(map(F,c['q_exact']));w=list(map(F,c['weights_exact']))
    ideal,ls,signs=ideal_cts_intervals(c['x'])
    if c['x']!=expected_x or c['optimized_axis']!='1Q' or len(es)!=6 or len(q)!=6 or len(w)!=6:raise ValueError('specified CTS point changed')
    if {e['label'] for e in es}!=set(ideal) or {e['label'] for e in definition['events']}!=set(ideal):raise ValueError('CTS support identity')
    if sum(q)!=1 or min(q)<=0 or min(w)<=0 or F(c['sampler_denominator'])!=2**60 or any(2**60%p.denominator for p in q):raise ValueError('CTS sampler')
    coef=F(0);bias=F(0);a=[p*z for p,z in zip(q,w)];L=max(w);m2=sum(p*z*z for p,z in zip(q,w))
    ratio=F(definition['fixed_rational_rotation_ratio']);N=v.sqrt_enclosure(1+ratio*ratio)
    actual={k:v.Interval(0) for k in ('II','ZZ','ZI','IZ','XY','YX')}
    for e,z in zip(es,a):
        ref=next(d for d in definition['events'] if d['label']==e['label'])
        if e['axis']!=ref['axis'] or e['phase_i_power']!=ref['phase_i_power'] or e['rotation_sign']!=ref['rotation_sign'] or e['real_event']!=ref['real_event']:raise ValueError('CTS phase/label')
        expected_axis=e['label'].removeprefix('real_minus_').removeprefix('rotation_')
        is_real=e['label'].startswith('real_minus_')
        if e['axis']!=expected_axis or e['real_event']!=is_real or e['phase_i_power']!=(2 if is_real else 0) or e['rotation_sign']!=(0 if is_real else signs[e['axis']]):raise ValueError('independent CTS direction')
        if z!=F(e['coefficient_mid']) or e['ideal_coefficient']!=ref['ideal_coefficient'] or e['angle_error_upper']!=ref['angle_error_upper']:raise ValueError('CTS coefficient identity')
        if e['epsilon']!=('exact' if is_real else c['precision'][e['axis']]):raise ValueError('CTS precision identity')
        row=next(d for d in costs if (d['x'],d['label'],d['epsilon'])==(c['x'],e['label'],e['epsilon']))
        if e['native_cost']!=row['native_cost']:raise ValueError('CTS cost binding')
        lo,hi=F(ref['ideal_coefficient']['lo']),F(ref['ideal_coefficient']['hi'])
        independent=ideal[e['label']]
        if not lo<=independent.lo<=independent.hi<=hi:raise ValueError('CTS independent ideal enclosure')
        angle=F(e['angle_error_upper'])
        if angle<0 or (is_real and angle!=0) or (not is_real and angle<2*max(abs(ratio-ls.lo),abs(ratio-ls.hi))):raise ValueError('CTS angle guard')
        coef+=max(abs(z-lo),abs(z-hi));bias+=2*z*(F(row['native_cost']['strict_event_error_upper'])+F(e['angle_error_upper']))
        if e['real_event']:
            if e['phase_i_power']!=2:raise ValueError('missing minus relative phase')
            actual[e['axis']]+= -z
        else:
            part=v.Interval(z)/N;actual['II']+=part;actual[e['axis']]+=part*(-e['rotation_sign']*ratio)
    target,_,_=cts_targets(c['x']);residual=F(0)
    for k,z in actual.items():
        d=z-target[k];residual+=max(abs(d.lo),abs(d.hi))
    if residual>F(1,10**12) or coef>F(1,10**25):raise ValueError('CTS full coefficient mean')
    s=v.EPS-coef-bias;n=c['shots_per_axis'];ell=v.log_enclosure()
    if s<=0 or not 1<=n<=10**9 or n*s*s<ell.hi*(2*m2+F(4,3)*L*s):raise ValueError('CTS confidence')
    totals={k:n*(2*sum(p*F(e['native_cost'][k]) for e,p in zip(es,q))+(5 if k=='1Q' else 0)) for k in RADII}
    if totals!={k:F(z) for k,z in c['resource_total'].items()} or c['workspace_peak']!=1:raise ValueError('CTS resources')
    if m2!=F(c['m2_exact']) or L!=F(c['L_exact']):raise ValueError('CTS moments')
    if any(F(c[k])!=z for k,z in (('coefficient_error',coef),('synthesis_and_angle_bias',bias),('remaining',s))):raise ValueError('CTS saved confidence accounting')
    return {'pass':True,'full_Pauli_coefficient_mean_residual_upper':str(residual),'coefficient_L1':str(coef),
        'joint_synthesis_angle_bias':str(bias),'m2':str(m2),'L':str(L),'remaining':str(s),'shots_per_axis':n,
        'confidence_margin':str(n*s*s-ell.hi*(2*m2+F(4,3)*L*s)), 'workspace':1,'resource_total':{k:str(z) for k,z in totals.items()}}

def run():
    scope=json.loads((OUT/'scope_v1.json').read_bytes())
    for n,h in scope['fixed_hashes'].items():
        if sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise PermissionError('G5 input/source changed '+n)
    for n,h in json.loads((OUT/'prior_protected_hashes.json').read_bytes()).items():
        if sha256((ROOT/n).read_bytes()).hexdigest()!=h:raise PermissionError('prior evidence changed '+n)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','rev-parse','HEAD^'],cwd=ROOT,text=True).strip()!=scope['base_G4_commit']:raise PermissionError('G5 source is not direct child of fixed G4 evidence')
    if scope['runtime_version']!=sys.version:raise PermissionError('fixed stdlib runtime changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise PermissionError('G5 source dirty')
    with (OUT/'one_shot_consumed.json').open('x') as f:json.dump({'source_commit':head,'scope_sha256':sha256((OUT/'scope_v1.json').read_bytes()).hexdigest(),'runs':1,'retries':0,'saved_only':True},f)
    start,cpu=time.monotonic(),time.process_time()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('G5 wall cap; no retry')));signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,resource.getrlimit(resource.RLIMIT_AS)[1]));resource.setrlimit(resource.RLIMIT_CPU,(480,resource.getrlimit(resource.RLIMIT_CPU)[1]))
    try:
        table=data(v.TABLE);subset={**table,'tables':{'1/4':table['tables']['1/4']}}
        verified=v.verify_table(subset,(ROOT/v.RAW).read_bytes());cols={c['id']:c for c in subset['tables']['1/4']['columns']}
        vs=vertices('1/4');profiles=[];axes={};g2=data(v.G2)['summary']['1/4']['axes']
        if v.exp_upper(F(37,4))>=10560:raise ArithmeticError('log bound')
        law=data(G4+'CTS_selected_complete_laws.json')['1/4/1Q'];defs=data(G4+'cts_definition.json')['1/4'];saved_B=data(G4+'result_B.json');costs=saved_B['event_rows']
        for seq in saved_B['synthesis_rows']:
            z=seq['sequence']
            if sha256(z.encode()).hexdigest()!=seq['sequence_sha256'] or z.count('T')+z.count('t')!=seq['T_count'] or z.count('t')!=seq['Tdagger_count'] or seq['error_pass'] is not True or not 0<=F(seq['strict_operator_error_upper'])<=F(seq['epsilon']):raise ValueError('saved G4 sequence/count/error identity')
        witness=verify_cts(law,defs,costs)
        for axis,r in RADII.items():
            rows=[]
            for name,g in vs.items():
                active=list(g)
                for ps in product(PRECISIONS,repeat=len(active)):rows.append(price('1/4',name,g,dict(zip(active,ps)),cols,axis))
            if len(rows)!=252:raise ValueError('252 pure precision coverage')
            lo=min(F(z['Phi']['lo']) for z in rows);hi=min(F(z['Phi']['hi']) for z in rows)
            ref=g2[axis]['same_IS_all_vertices']['value'];overlap=not (hi<F(ref['lo']) or lo>F(ref['hi']))
            if not overlap or lo<=r*r:raise ValueError('independent G2 price or fixed r lower bound failed')
            b=bridge(cols,axis,r);lower=37*r*r;cost=F(witness['resource_total'][axis])
            if lower<=cost:raise ValueError('same CTS witness does not separate fixed ideal lower')
            axes[axis]={'Phi_min_enclosure':v.Interval(lo,hi).json(),'G2_saved_interval_overlap':overlap,'r':str(r),
                'r_squared_lower_certified':True,'policy_resource_lower':str(lower),'CTS_same_law_resource':str(cost),
                'gap':str(lower-cost),'digital_bridge':b,'ideal_exclusion':True,'digital_exclusion':b['pass']}
            profiles+=rows
        digital=all(z['digital_exclusion'] for z in axes.values())
        result={'status':'G5_DIGITAL_SIX_VERTEX_CLASS_EXCLUDED_BY_SAVED_CTS_LAW' if digital else 'G5_IDEAL_SIX_VERTEX_CLASS_EXCLUSION_ONLY',
            'source_commit':head,'x':'1/4','degree':3,'vertex_count':6,'rational_coefficient_bases_checked':35,'saved_G4_sequences_verified':len(saved_B['synthesis_rows']),
            'profiles':252,'priced_evaluations':len(profiles),'source_event_conditions_bound':verified,'vertices':{k:{n:str(g) for n,g in gs.items()} for k,gs in vs.items()},
            'axes':axes,'CTS_same_complete_law_certificate':witness,'log_lower':'37/4','exp_log_upper':str(v.exp_upper(F(37,4))),
            'wall':time.monotonic()-start,'CPU':time.process_time()-cpu,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'new_synthesis_LP_Hamiltonian_matrix_circuit_sampling_DF_NPZ_GPU_calls':0,'G1_G2_G3_evaluator_imports':0,'G4_rational_primitives_shared':True,
            'runs':1,'retries':0,'old_K3_all_class_claim':False,'physical_or_information_theoretic_lower_bound':False,'mandatory_STOP':True}
        put('independent_profiles.json',profiles);put('result_v1.json',result)
        if sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())>16*1024**2:raise RuntimeError('G5 output cap')
        print(json.dumps({'status':result['status'],'profiles':252,'prices':len(profiles),'digital':digital}))
    except Exception as e:
        put('failure.json',{'status':'G5_ADDITIONAL_CLASS_CLAIM_NOT_ESTABLISHED','error':str(e),'prefix_not_final':True,'runs':1,'retries':0,'mandatory_STOP':True});raise
    finally:
        signal.alarm(0);put('STOP.json',{'mandatory_STOP':True,'next_science_authorized':False,'new_hypothesis_owner':'GPT','retries':0})

if __name__=='__main__':run()
