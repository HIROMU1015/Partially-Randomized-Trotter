"""G4-A independent rational/dyadic verifier; imports no G2/G3 helpers.

Only fixed saved JSON arithmetic. No solver, sampling, synthesis, native lowering
or matrix. The theorem is confined to the stated coefficient/budget class.
"""
import ast
from fractions import Fraction as F
from hashlib import sha256
from itertools import product
from math import isqrt
from pathlib import Path
import json
import resource
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'artifacts/track_b_g4_conditional_separation/2026-10-09'
TABLE='artifacts/track_b_ra_d0_preparation/2026-10-06/candidate_table_v1.json'
RAW='artifacts/track_b_rte_reallocation_r1_result/2026-10-06/v1/result.json'
LAWS='artifacts/track_b_g3_finite_law/2026-10-09/phase_A_selected_laws.json'
G2='artifacts/track_b_g2_saved_diagnostic/2026-10-09/result_v1.json'
EPS=F(1,200);PRECISIONS=('1e-3','1e-4','1e-6');BITS=384

def write(path,v):path.write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')
def digest(v):return sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class Interval:
    def __init__(self,lo,hi=None):
        self.lo=F(lo);self.hi=self.lo if hi is None else F(hi)
        if self.lo>self.hi:raise ValueError('unordered interval')
    def __add__(self,z):
        z=z if isinstance(z,Interval) else Interval(z)
        return Interval(self.lo+z.lo,self.hi+z.hi)
    __radd__=__add__
    def __neg__(self):return Interval(-self.hi,-self.lo)
    def __sub__(self,z):return self+-as_interval(z)
    def __mul__(self,z):
        z=as_interval(z);v=[x*y for x in (self.lo,self.hi) for y in (z.lo,z.hi)]
        return Interval(min(v),max(v))
    __rmul__=__mul__
    def __truediv__(self,z):
        z=as_interval(z)
        if z.lo<=0<=z.hi:raise ZeroDivisionError('zero enclosing denominator')
        return self*Interval(1/z.hi,1/z.lo)
    def json(self):return {'lo':str(self.lo),'hi':str(self.hi)}

def as_interval(z):return z if isinstance(z,Interval) else Interval(z)
def sqrt_enclosure(q,bits=BITS):
    q=F(q)
    if q<0:raise ValueError('negative square root')
    den=2**bits;k=isqrt((q.numerator*den*den)//q.denominator)
    lo=F(k,den)
    return Interval(lo,lo if lo*lo==q else F(k+1,den))

def exp_upper(z,terms=60):
    z=F(z)
    if z<0 or z>=terms+2:raise ValueError('geometric tail unavailable')
    term=F(1);total=term
    for k in range(1,terms+1):term*=z/k;total+=term
    return total+(term*z/(terms+1))/(1-z/(terms+2))

def log_enclosure():
    # ln10560=13 ln2+ln(165/128), evaluated by positive atanh series.
    def log_ratio(n,d):
        z=F(n-d,n+d);s=F(0);power=z
        for k in range(160):s+=power/(2*k+1);power*=z*z
        return Interval(2*s,2*s+2*power/((321)*(1-z*z)))
    return 13*log_ratio(2,1)+log_ratio(165,128)

def prototypes(x):
    x=F(x);rho=x*(1+x*x/6)/(1+x*x/2)
    return {'O0':(0,F(1),x),'O2':(2,x*x/2,x**3/6),
        'P2':(2,x*x/2,F(0)),'P3':(3,x**3/6,F(0)),
        'A0':(0,F(1),rho),'A1':(1,2*x**3/(3*(x*x+2)),2*x*x/(x*x+6)),
        'A2':(2,x*x*(x*x+2)/(2*(x*x+6)),x**3/6)}

def weights(x,arm):
    if arm=='ordinary':return {'O0':F(1),'O2':F(1)}
    if arm=='PTSC_K0':return {'O0':F(1),'P2':F(1),'P3':F(1)}
    if arm=='A':return {'A0':F(1),'A1':F(1),'A2':F(1)}
    if arm=='J1':return {'A0':F(1),'A1':F(1),'P2':(F(x)**2+2)/(F(x)**2+6),'P3':F(1)}
    raise ValueError('unregistered arm')

def ideal_mean(x,arm):
    v=[F(0)]*4
    for name,w in weights(x,arm).items():
        k,a,b=prototypes(x)[name];v[k]+=w*a
        if b:v[k+1]+=w*b
    if v!=[F(1),F(x),F(x)**2/2,F(x)**3/6]:raise ValueError('formal degree identity')

def verify_table(table,raw):
    if table['source_result_sha256']!=sha256(raw).hexdigest():raise ValueError('raw identity')
    data=json.loads(raw);count=0
    for seq in data['synthesis_rows']:
        s=seq['sequence']
        if sha256(s.encode()).hexdigest()!=seq['sequence_sha256'] or s.count('T')+s.count('t')!=seq['T_count'] or s.count('t')!=seq['Tdagger_count'] or seq['error_pass'] is not True:raise ValueError('saved sequence identity')
    for xs,t in table['tables'].items():
        if len(t['columns'])!=21 or t['workspace_exclusions']:raise ValueError('column inventory')
        for c in t['columns']:
            name=c['prototype'];k,a,b=prototypes(xs)[name];norm2=a*a+b*b
            if (c['degree'],list(map(F,c['saved_ideal_ab_exact'])))!=(k,[a,b]):raise ValueError('prototype')
            for j,(lo,hi) in enumerate(c['D_intervals']):
                lo,hi=F(lo),F(hi);numer=a if j==k else b if j==k+1 else F(0)
                if not 0<=lo<=hi or not lo*lo*norm2<=numer*numer<=hi*hi*norm2:raise ValueError('normalized column enclosure')
            if c['workspace_peak']!=1:raise ValueError('workspace')
            for e in c['events']:
                labels=ast.literal_eval(e['source_label'].split(':',1)[1]);p=F(1)
                for i in labels:p*=(F(3,4),F(1,4))[i]
                expected_word=list(labels[1:] if b else labels);expected_rotation=labels[0] if b else None
                if (len(labels),F(e['label_probability']),e['word'],e['rotation'],e['phase_i_power'],e['rotation_sign'],e['complement'])!=(k+int(b!=0),p,expected_word,expected_rotation,(-k)%4,1,name=='A1'):raise ValueError('word/phase/conditional law')
                arms={'O0':'ordinary','O2':'ordinary','P2':'PTSC_K0','P3':'PTSC_K0','A0':'A','A1':'A','A2':'A'}
                native=[]
                for row in data['resource_rows']:
                    if (row['context'],row['controlled'],row['x'],row['arm'],row['epsilon'],row['sigma'])==('distinct_basis',True,xs,arms[name],c['epsilon'],1):
                        native += [z for z in row['profile']['events'] if z['label']==e['source_label']]
                if len(native)!=1 or native[0]['native_cost']!=e['native_cost']:raise ValueError('raw source event binding')
                if any(F(e['native_cost'][z])<0 for z in ('T','CX','1Q','strict_event_error_upper')):raise ValueError('negative physical cost/error')
                count+=1
            if sum(F(e['label_probability']) for e in c['events'])!=1:raise ValueError('conditional support')
            if F(c['d_upper'])!=2*sum(F(e['label_probability'])*F(e['native_cost']['strict_event_error_upper']) for e in c['events']):raise ValueError('event bias aggregate')
    return count

def profile(x,arm,precision,cols,axis,readout=False):
    ideal_mean(x,arm);K=Interval(0);bias=Interval(0)
    for name,w in weights(x,arm).items():
        c=cols[name+':'+precision[name]];_,a,b=prototypes(x)[name];norm=sqrt_enclosure(a*a+b*b)
        d=F(0);price=Interval(0)
        for e in c['events']:
            p=F(e['label_probability']);d+=2*p*F(e['native_cost']['strict_event_error_upper'])
            price+=p*sqrt_enclosure(F(e['native_cost'][axis])+(F(5,2) if readout else 0))
        K+=w*norm*price;bias+=w*norm*d
    s=Interval(EPS)-bias
    if s.lo<=0:raise ValueError('registered profile feasibility unresolved')
    r=K/s
    return {'arm':arm,'precision':precision,'axis':axis,'readout_price_added':readout,
        'K':K.json(),'remaining':s.json(),'ratio':r.json(),'Phi':(r*r).json()}

def finite_law_certificate(c,cols):
    es=c['event_source'];q=list(map(F,c['q_exact']));w=list(map(F,c['weights_exact']))
    if len(es)!=len(q) or len(w)!=len(q) or sum(q)!=1 or min(q)<=0 or any(2**60%v.denominator for v in q):raise ValueError('finite sampler')
    x=F(c['x']);ideal_mean(x,'J1');coef=F(0);synth=F(0);B=[F(0)]*4;groups={}
    for e,p,z in zip(es,q,w):
        v=p*z;col=cols[e['column_id']];name=col['prototype'];gamma=weights(x,'J1')[name]
        share=F(e['conditional_probability']);source=next(s for s in col['events'] if e['id'].split('/',1)[1]==s['source_label'])
        if v!=F(e['coefficient_mid']) or share!=F(source['label_probability']):raise ValueError('reweight/source share')
        if e['D_intervals']!=col['D_intervals'] or any(e[k]!=source[k] for k in ('word','rotation','rotation_sign','phase_i_power','complement','native_cost')):raise ValueError('source direction/phase')
        k,a,b=prototypes(x)[name];scale=gamma*share;norm2=a*a+b*b
        lo,hi=F(e['coefficient_lo'])/scale,F(e['coefficient_hi'])/scale
        if not 0<=lo<=hi or not lo*lo<=norm2<=hi*hi:raise ValueError('coefficient interval not enclosing ideal')
        if min(v,z)<=0:raise ValueError('coefficient positive support')
        coef+=max(abs(v-F(e['coefficient_lo'])),abs(v-F(e['coefficient_hi'])))
        synth+=2*v*F(e['native_cost']['strict_event_error_upper'])
        ratio=v/share
        if name in groups and groups[name]!=ratio:raise ValueError('IID shares not retained')
        groups[name]=ratio
    for name in groups:
        if sum(F(e['conditional_probability']) for e in es if cols[e['column_id']]['prototype']==name)!=1:raise ValueError('support incomplete')
    remaining=EPS-coef-synth;m2=sum(p*z*z for p,z in zip(q,w));L=max(w);n=c['shots_per_axis'];ell=log_enclosure()
    if remaining<=0 or not 1<=n<=10**9 or coef>F(1,10**25):raise ValueError('feasibility/bias')
    margin=n*remaining*remaining-ell.hi*(2*m2+F(4,3)*L*remaining)
    if margin<0:raise ValueError('Bernstein confidence')
    totals={k:n*(2*sum(p*F(e['native_cost'][k]) for p,e in zip(q,es))+(5 if k=='1Q' else 0)) for k in ('T','CX','1Q')}
    if totals!={k:F(v) for k,v in c['resource_total'].items()}:raise ValueError('resource arithmetic')
    if c['workspace_peak']!=1:raise ValueError('workspace')
    # Formal residual is auxiliary; ideal algebra plus coefficient L1 is the bias proof.
    residual=F(0)
    for j,t in enumerate((1,x,x*x/2,x**3/6)):
        ends=[sum(p*z*F(e['D_intervals'][j][k]) for e,p,z in zip(es,q,w))-t for k in (0,1)]
        residual+=max(map(abs,ends))
    if residual>F(1,10**12):raise ValueError('digital formal mean residual')
    return {'pass':True,'law_id':c['id'],'mean_L1_residual_upper':str(residual),
        'coefficient_L1_error':str(coef),'synthesis_bias':str(synth),'remaining':str(remaining),
        'm2':str(m2),'L':str(L),'shots_per_axis':n,'confidence_margin':str(margin),
        'resource_totals':{k:str(v) for k,v in totals.items()},'independent_G2_G3_code_imports':0}

def run():
    scope=json.loads((OUT/'scope_A.json').read_bytes())
    for path,digest in scope['fixed_hashes'].items():
        if sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise PermissionError('source binding '+path)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise PermissionError('source worktree dirty')
    with (OUT/'phase_A_consumed.json').open('x') as f:json.dump({'source':head,'scope_sha256':sha256((OUT/'scope_A.json').read_bytes()).hexdigest(),'runs':1,'retries':0},f)
    start,cpu=time.monotonic(),time.process_time();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('G4 A cap')));signal.alarm(600)
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,resource.getrlimit(resource.RLIMIT_AS)[1]))
    resource.setrlimit(resource.RLIMIT_CPU,(480,resource.getrlimit(resource.RLIMIT_CPU)[1]))
    try:
        table=json.loads((ROOT/TABLE).read_bytes());verified=verify_table(table,(ROOT/RAW).read_bytes())
        all_rows=[];summary={};g2=json.loads((ROOT/G2).read_bytes())
        if not exp_upper(F(37,4))<10560:raise ArithmeticError('log lower proof')
        laws=json.loads((ROOT/LAWS).read_bytes())
        for xs,t in table['tables'].items():
            cols={c['id']:c for c in t['columns']};recs={}
            for axis,readout in (('T',False),('1Q',False),('1Q',True)):
                label=axis+('_readout' if readout else '_native');rows=[]
                for arm in ('ordinary','PTSC_K0','A'):
                    active=list(weights(xs,arm))
                    for ps in product(PRECISIONS,repeat=len(active)):
                        p=profile(xs,arm,dict(zip(active,ps)),cols,axis,readout);p['x']=xs;rows.append(p);all_rows.append(p)
                if len(rows)!=63:raise ValueError('fixed 63 profile coverage')
                rlo=min(F(p['ratio']['lo']) for p in rows);rhi=min(F(p['ratio']['hi']) for p in rows)
                slack=min(rlo*(1-2*F(e['native_cost']['strict_event_error_upper']))-sqrt_enclosure(F(e['native_cost'][axis])+(F(5,2) if readout else 0)).hi for c in cols.values() for e in c['events'])
                if slack<0:raise ArithmeticError('digitalization bridge condition not certified')
                lower=37*rlo*rlo
                recs[label]={'profiles':63,'r_enclosure':Interval(rlo,rhi).json(),'budget_policy_cost_lower':str(lower),
                    'digital_bridge_min_slack':str(slack),'bridge_class':'nonnegative digital event coefficients with e>=L1 to some ideal B2 and bias e+d^T ctilde; not old tolerance-only K2/K3',
                    'global_arbitrary_full_support_IS_for_stated_class':True,'physical_shot_lower_bound':False}
                if not readout:
                    ref=g2[xs]['axes'][axis]['same_IS_original_vertices']['value'] if xs in g2 else g2['summary'][xs]['axes'][axis]['same_IS_original_vertices']['value']
                    recs[label]['saved_G2_interval_overlap']=not (rhi*rhi<F(ref['lo']) or F(ref['hi'])<rlo*rlo)
                    if not recs[label]['saved_G2_interval_overlap']:raise ArithmeticError('independent vs saved G2 interval inconsistent')
            chosen=next(c for c in laws if c['x']==xs and c['arm']=='J1' and c['optimized_axis']=='1Q')
            proof=finite_law_certificate(chosen,cols)
            recs['J1_finite_certificate']=proof
            recs['strict_T_separation']=F(proof['resource_totals']['T'])<F(recs['T_native']['budget_policy_cost_lower'])
            recs['strict_1Q_separation']=F(proof['resource_totals']['1Q'])<F(recs['1Q_readout']['budget_policy_cost_lower'])
            recs['T_gap']=str(F(recs['T_native']['budget_policy_cost_lower'])-F(proof['resource_totals']['T']))
            recs['1Q_gap']=str(F(recs['1Q_readout']['budget_policy_cost_lower'])-F(proof['resource_totals']['1Q']))
            summary[xs]=recs
        gate=summary['1/4']['strict_T_separation'] and summary['1/4']['strict_1Q_separation']
        write(OUT/'independent_profiles.json',all_rows)
        write(OUT/'result_A.json',{'status':'G4_A_STATED_CLASS_SEPARATION_CERTIFIED' if gate else 'G4_A_SEPARATION_NOT_CLOSED',
            'source_commit':head,'profiles':126,'priced_profile_evaluations':len(all_rows),'source_events_bound':verified,
            'summary':summary,'G4_B_conditional_gate':gate,'ln10560_lower':'37/4','log_lower_exp_upper':str(exp_upper(F(37,4))),
            'wall':time.monotonic()-start,'CPU':time.process_time()-cpu,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'new_LP_sampling_synthesis_circuit_matrix_DF_GPU_calls':0,'retries':0,'runs':1,
            'independent_code_not_external_replication':True,'old_K2_K3_or_original_D0_witness':False})
        print(json.dumps({'gate':gate,'profiles':126,'priced_evaluations':len(all_rows)}))
    except Exception as e:
        write(OUT/'failure_A.json',{'status':'G4_A_TECHNICAL_INCONCLUSIVE','error':str(e),'retries':0,'prefix_not_final':True,'mandatory_STOP':True});raise
    finally:signal.alarm(0)

if __name__=='__main__':run()
