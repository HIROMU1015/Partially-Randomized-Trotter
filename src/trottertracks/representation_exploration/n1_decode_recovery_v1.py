"""N1-only decode recovery. Numerical kernels copied unchanged from bbcb0176.

Only saved-input adapter/schema/preflight differs; no N3 or N2 entry points.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
import gzip
import hashlib
import itertools as it
import json
import math
from pathlib import Path
import resource
import time

import mpmath as mp
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from .algorithm_design import gaussian_circuit, fock_frame, native_action, second_quantize_one_body, sector_indices
from .construction_comparison import apply_pauli

SEED = 20261011
P = 128
D = 1 << P
mp.iv.prec = P
INPUT_SHA = '70b2631b114983572cfc92527b2d569a4b48b03a6d88790b75dcf89f13614438'
CONTEXT_KEYS = {'context','modes','particles','df_rank','weights','one_body','scalar','input_factors','time','model_budget','q_grid','generator_candidates'}
CANDIDATE_KEYS = {'id','family','eta','approx_eta','frame','ungauged_frame','ungauged_eta','g','groups'}
T, EPS = F(.6), F(.04)
GOAL = F(1,250)
METRICS = ('rz','cx','sx','x','size','depth')


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def fj(x):
    x = F(x)
    if max(x.numerator.bit_length(), x.denominator.bit_length()) > 4096:
        raise ArithmeticError('UNRESOLVED_ARITHMETIC_CAP')
    return [x.numerator, x.denominator]


def fs(x):
    return F(*x)


def mat(x):
    return np.asarray(x['real']) + 1j*np.asarray(x['imag'])


def mj(x):
    return {'real': np.asarray(x).real.tolist(), 'imag': np.asarray(x).imag.tolist()}


def exact_iv(x):
    x = F(x)
    return mp.iv.mpf(x.numerator) / x.denominator


def mpf_fraction(t):
    sign, man, exponent, _ = t
    return (-1 if sign else 1)*F(man)*(F(2)**exponent)


def endpoints(x):
    return tuple(mpf_fraction(t) for t in x._mpi_)


def upper(x):
    return endpoints(x)[1]


def iabs(x):
    if hasattr(x, '_mpci_'):
        return abs(x.real) + abs(x.imag)
    return abs(x)


def ivmatrix(x):
    x = np.asarray(x)
    return [[mp.iv.mpc(exact_iv(float(z.real)), exact_iv(float(z.imag))) for z in row] for row in x]


def imm(a,b):
    return [[sum((a[i][k]*b[k][j] for k in range(len(b))),mp.iv.mpc(0)) for j in range(len(b[0]))] for i in range(len(a))]


def iadj(a):
    return [[iconj(a[j][i]) for j in range(len(a))] for i in range(len(a))]


def iconj(z):
    # mpmath 1.3.0 iv.mpc.conjugate incorrectly calls scalar mpf_neg.
    return mp.iv.mpc(z.real,-z.imag)


def iid(n):
    return [[mp.iv.mpc(int(i==j)) for j in range(n)] for i in range(n)]


def gaussian_onebody(qc):
    """Vacuum-normalized one-body action from gate coefficients, never Fock truth."""
    n=qc.num_qubits; out=iid(n)
    for item in qc.data:
        name=item.operation.name; qs=[qc.find_bit(q).index for q in item.qubits]
        par=[exact_iv(float(x)) for x in item.operation.params]; g=iid(n)
        if name in ('rz','p'):
            a=par[0]; g[qs[0]][qs[0]]=mp.iv.mpc(mp.iv.cos(a),mp.iv.sin(a))
        elif name=='xx_plus_yy':
            a,b=par; c,s=mp.iv.cos(a/2),mp.iv.sin(a/2)
            i,j=qs; g[i][i]=g[j][j]=c
            g[i][j]=-mp.iv.mpc(0,1)*s*mp.iv.mpc(mp.iv.cos(b),-mp.iv.sin(b))
            g[j][i]=-mp.iv.mpc(0,1)*s*mp.iv.mpc(mp.iv.cos(b),mp.iv.sin(b))
        else:
            raise ValueError(('STOP_N1_ROUNDING_CERTIFICATE_UNAVAILABLE',name))
        out=imm(g,out)
    return out


def conjugated(q,eta):
    return [[sum((q[i][k]*exact_iv(eta[k])*iconj(q[j][k]) for k in range(len(q))),mp.iv.mpc(0)) for j in range(len(q))] for i in range(len(q))]


def ivmid(a):
    return np.array([[complex(float(sum(endpoints(z.real))/2),float(sum(endpoints(z.imag))/2)) for z in row] for row in a])


def residual(a,b):
    return sum((upper(iabs(a[i][j]-b[i][j])) for i in range(len(a)) for j in range(len(a))),F(0))


@lru_cache(None)
def pproduct(a,b):
    lut={'XY':('Z',1j),'YX':('Z',-1j),'YZ':('X',1j),'ZY':('X',-1j),'ZX':('Y',1j),'XZ':('Y',-1j)}
    phase=1; chars=[]
    for x,y in zip(a,b,strict=True):
        if x=='I': chars.append(y)
        elif y=='I': chars.append(x)
        elif x==y: chars.append('I')
        else:
            z,p=lut[x+y]; chars.append(z); phase*=p
    return ''.join(chars),complex(phase)


def pdadd(a,b):
    out=a.copy()
    for k,v in b.items(): out[k]=out.get(k,0)+v
    return out


def pdmul(a,b):
    out={}
    for ka,va in a.items():
        for kb,vb in b.items():
            k,p=pproduct(ka,kb); out[k]=out.get(k,0)+va*vb*p
    if len(out)>256: raise ValueError('Pauli term cap')
    return out


def annihilator(n,j):
    # Qiskit label order: highest qubit first.
    x=['I']*n; y=x.copy()
    for k in range(j): x[n-1-k]=y[n-1-k]='Z'
    x[n-1-j]='X'; y[n-1-j]='Y'
    return {''.join(x):F(1,2),''.join(y):.5j}


def orbital_interval(g):
    n=len(g); out={}
    for i in range(n):
        ai={k:complex(v).conjugate() for k,v in annihilator(n,i).items()}
        for j in range(n):
            terms=pdmul(ai,annihilator(n,j))
            out=pdadd(out,{k:g[i][j]*v for k,v in terms.items()})
    return out


def squared_interval(g,w):
    f=orbital_interval(g)
    return {k:exact_iv(w)*v for k,v in pdmul(f,f).items()}


def coefficient_certificate(terms):
    """Real Pauli coefficients of a Hermitian operator, outward dyadic enclosure."""
    out={}
    for k,v in terms.items():
        real=v.real if hasattr(v,'_mpci_') else mp.iv.mpf(v)
        lo,hi=endpoints(real)
        l=(lo*D).__floor__(); h=(hi*D).__ceil__()
        out[k]=[l,h]
    return out


def centres(cert):
    return {k:((lo+hi)//2,0) for k,(lo,hi) in cert.items() if k.strip('I') and (lo+hi)//2}


def cradius(cert):
    return sum((F(max((lo+hi)//2-lo,hi-(lo+hi)//2),D) for k,(lo,hi) in cert.items() if k.strip('I')),F(0))


def cmul(a,b):
    return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])


def cadd(a,b):
    return (a[0]+b[0],a[1]+b[1])


def comm(a,b):
    out={}
    for ka,va in a.items():
        for kb,vb in b.items():
            k,p=pproduct(ka,kb)
            if not p.imag: continue
            v=cmul(va,vb); v=cmul(v,(0,2 if p.imag>0 else -2))
            out[k]=cadd(out.get(k,(0,0)),v)
    return {k:v for k,v in out.items() if v!=(0,0)}


def cl1(a,degree=1):
    # Commutators of Hermitian centres are exactly purely real or imaginary.
    return F(sum(abs(x)+abs(y) for x,y in a.values()),D**degree)


def pf_certificate(certs):
    cs=[centres(c) for c in certs]; ds=[cradius(c) for c in certs]; value=F(0); records=[]
    for j,h in enumerate(cs[:-1]):
        r={}
        for other in cs[j+1:]:
            for k,v in other.items(): r[k]=cadd(r.get(k,(0,0)),v)
        dh,dr=ds[j],sum(ds[j+1:],F(0)); hn,rn=cl1(h),cl1(r)
        nr=cl1(comm(r,comm(r,h)),3); nh=cl1(comm(h,comm(h,r)),3)
        er=4*((rn+dr)**2*(hn+dh)-rn**2*hn)
        eh=4*((hn+dh)**2*(rn+dr)-hn**2*rn)
        term=(nr+er)/12+(nh+eh)/24; value+=term
        records.append({'j':j,'R_nested_L1':fj(nr),'H_nested_L1':fj(nh),'R_rounding':fj(er),'H_rounding':fj(eh),'term':fj(term)})
    return {'alpha2':fj(value),'terms':records,'dyadic_bits':P,'method':'exact_combined_Pauli_centres_plus_Lipschitz_interval_radius'}


def exact_diagonal_terms(eta,w):
    # Number operators commute: exact rational arithmetic, including identity.
    n=len(eta); out={}
    for i,e in enumerate(eta):
        z=['I']*n; z[n-1-i]='Z'
        out=pdadd(out,{'I'*n:F(e)/2,''.join(z):-F(e)/2})
    ans={}
    for a,x in out.items():
        for b,y in out.items():
            k,p=pproduct(a,b)
            assert p==1
            ans[k]=ans.get(k,F(0))+F(w)*x*y
    return {k:v for k,v in sorted(ans.items()) if v}


def onebody_terms(h):
    return {k:sum(endpoints(v.real))/2 for k,v in sorted(orbital_interval(ivmatrix(h)).items()) if sum(endpoints(v.real))}


def safe_shots(bias):
    # Upper integer, outward constants. No unknown PF/model error is consulted.
    b=F(bias); lo,_=endpoints(exact_iv(EPS)/mp.iv.sqrt(2)); loghi=upper(mp.iv.ln(80))
    if b>=lo: return None
    return (2*loghi/(lo-b)**2).__ceil__()


def scalar(value):
    if type(value) not in (int,float):raise ValueError('finite numeric scalar; bool refused')
    try:finite=math.isfinite(value)
    except OverflowError:finite=False
    if not finite:raise ValueError('finite numeric scalar')
    return value


def decode_vector(value,n=4,real_spectrum=True):
    if isinstance(value,dict):
        if set(value)!={'real','imag'}:raise ValueError('unknown vector field')
        real,imag=value['real'],value['imag']
    elif isinstance(value,list):real,imag=value,[0.0]*n
    else:raise ValueError('vector schema')
    if not isinstance(real,list) or not isinstance(imag,list) or len(real)!=n or len(imag)!=n:raise ValueError('vector shape')
    real=[scalar(x) for x in real];imag=[scalar(x) for x in imag]
    if real_spectrum and any(x!=0 for x in imag):raise ValueError('nonzero spectrum imaginary part')
    if real_spectrum:return real.copy()
    return np.asarray(real,dtype=float)+1j*np.asarray(imag,dtype=float)


def decode_matrix(value,n=4):
    if not isinstance(value,dict) or set(value)!={'real','imag'}:raise ValueError('matrix schema')
    for component in ('real','imag'):
        a=value[component]
        if not isinstance(a,list) or len(a)!=n or any(not isinstance(row,list) or len(row)!=n for row in a):raise ValueError('matrix shape')
        for row in a:
            for v in row:scalar(v)
    return np.asarray(value['real'],dtype=float)+1j*np.asarray(value['imag'],dtype=float)


def validate_context(c,expected_id=None,expected_counts=None):
    if not isinstance(c,dict) or set(c)!=CONTEXT_KEYS:raise ValueError('oracle/unknown context field refused')
    if expected_id is not None and c['context']!=expected_id:raise ValueError('context identity/order')
    if not isinstance(c['context'],str):raise ValueError('context name')
    if any(type(c[k])!=int or c[k]!=v for k,v in [('modes',4),('particles',2),('df_rank',2)]):raise ValueError('fixed sector/rank')
    for key in ('scalar','time','model_budget'):scalar(c[key])
    if c['time']!=.6 or c['model_budget']!=.02 or c['q_grid']!=[1,2,4,8] or any(type(q)!=int for q in c['q_grid']):raise ValueError('fixed time/q/error')
    if not isinstance(c['weights'],list) or len(c['weights'])!=2:raise ValueError('weight shape')
    for w in c['weights']:scalar(w)
    decode_matrix(c['one_body'])
    if not isinstance(c['input_factors'],list) or len(c['input_factors'])!=2:raise ValueError('factor matrix count')
    for g in c['input_factors']:decode_matrix(g)
    if not isinstance(c['generator_candidates'],list) or len(c['generator_candidates'])!=2:raise ValueError('factor list count')
    decoded=[];ids=set()
    for j,ls in enumerate(c['generator_candidates']):
        if not isinstance(ls,list) or not ls or (expected_counts is not None and len(ls)!=expected_counts[j]):raise ValueError('candidate counts')
        if sum(x.get('family')=='exact' for x in ls if isinstance(x,dict))!=1:raise ValueError('exact fallback inventory')
        group=[]
        for i,x in enumerate(ls):
            if not isinstance(x,dict) or set(x)!=CANDIDATE_KEYS:raise ValueError('oracle/unknown candidate field refused')
            if not isinstance(x['id'],str) or x['id'] in ids:raise ValueError('unique candidate ID')
            if expected_id is not None and x['id']!=f'{expected_id}/factor{j}/candidate{i}':raise ValueError('candidate identity/order')
            ids.add(x['id'])
            if x['family'] not in {'exact','cluster','zero_cutoff','shifted_cutoff'}:raise ValueError('candidate family')
            for key in ('eta','approx_eta','ungauged_eta'):decode_vector(x[key])
            for key in ('frame','ungauged_frame','g'):decode_matrix(x[key])
            groups=x['groups']
            if not isinstance(groups,list) or not groups or any(not isinstance(g,list) or not g for g in groups):raise ValueError('groups partition')
            flat=[v for g in groups for v in g]
            if any(type(v)!=int for v in flat) or sorted(flat)!=list(range(4)):raise ValueError('groups partition')
            # Only runtime vector representation changes. Matrix components retain every saved bit.
            y=x.copy();y['ungauged_eta']=decode_vector(x['ungauged_eta']);group.append(y)
        decoded.append(group)
    result=c.copy();result['generator_candidates']=decoded;return result


def load_contexts(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INPUT_SHA:raise ValueError('selector input hash')
    p=json.loads(raw)
    top={'contexts','id','schema_version','scientific_evaluation_performed','source_path','source_sha256'}
    if not isinstance(p,dict) or set(p)!=top:raise ValueError('oracle/unknown pack field refused')
    if p['schema_version']!=1 or type(p['schema_version'])!=int or p['scientific_evaluation_performed'] is not False:raise ValueError('input pack metadata')
    contexts=p['contexts'];names=('planted_local_gauge','dense_group_frame','separated_spectrum');counts=((9,9),(9,9),(4,4))
    if not isinstance(contexts,list) or len(contexts)!=3:raise ValueError('context inventory')
    decoded=[validate_context(c,name,count) for c,name,count in zip(contexts,names,counts,strict=True)]
    for c in decoded:
        if c['weights']!=[.7,-.4] or c['scalar']!=.11:raise ValueError('fixed weights/scalar')
        if c['one_body']!=mj(np.diag([.08,-.04,.03,-.02])):raise ValueError('fixed one-body')
    return decoded


def preflight(path):
    raw=json.loads(Path(path).read_bytes());contexts=load_contexts(path);ratios=0
    def identity(a,b):
        nonlocal ratios
        if isinstance(a,dict):
            if isinstance(b,list) and set(a)=={'real','imag'}:
                if len(a['real'])!=len(b) or len(a['imag'])!=len(b):raise ValueError('runtime vector shape identity')
                for x,y in zip(a['real'],b,strict=True):identity(x,y)
                for x in a['imag']:
                    if float(x).as_integer_ratio()!=(0,1):raise ValueError('spectrum imaginary identity')
                    ratios+=1
            else:
                if not isinstance(b,dict) or set(a)!=set(b):raise ValueError('runtime field identity')
                for key in a:identity(a[key],b[key])
        elif isinstance(a,list):
            if not isinstance(b,list) or len(a)!=len(b):raise ValueError('runtime list identity')
            for x,y in zip(a,b,strict=True):identity(x,y)
        elif type(a) in (int,float):
            if type(b) not in (int,float) or float(a).as_integer_ratio()!=float(b).as_integer_ratio():raise ValueError('decoded scalar binary64 identity')
            ratios+=1
        elif a!=b:raise ValueError('runtime metadata identity')
    for original,c in zip(raw['contexts'],contexts,strict=True):identity(original,c)
    factors=sum(len(ls) for c in contexts for ls in c['generator_candidates']);pairs=sum(len(c['generator_candidates'][0])*len(c['generator_candidates'][1]) for c in contexts)
    if factors!=44 or pairs!=178:raise ValueError('frozen domain counts')
    return {'status':'PASS','input_sha256':INPUT_SHA,'contexts':[c['context'] for c in contexts],'factor_candidates':factors,'pair_combinations':pairs,'gauged_ungauged_constructions':2*pairs,'binary64_ratio_checks':ratios,'PF_calls':0,'compilations':0,'truth_diagnostics':0,'input_rewrite':False,'raw_factor_symmetrization':False}


def anti_growth(c):
    # Raw saved g are almost Hermitian, but are not silently symmetrized.
    # ||anti(dGamma(g)^2)|| <= 2||dGamma(g)|| ||anti(dGamma(g))||.
    gamma=F(0)
    for gj,w in zip(c['input_factors'],c['weights'],strict=True):
        g=mat(gj); n=len(g); s=F(0); rho=F(0)
        for i in range(n):
            for j in range(n):
                s+=abs(F(float(g[i,j].real)))+abs(F(float(g[i,j].imag)))
                rho+=abs((F(float(g[i,j].real))-F(float(g[j,i].real)))/2)
                rho+=abs((F(float(g[i,j].imag))+F(float(g[j,i].imag)))/2)
        gamma+=abs(F(w))*2*(2*s)*(2*rho)
    return gamma


def prepare_factors(c):
    result=[]
    for j,ls in enumerate(c['generator_candidates']):
        exact=next(x for x in ls if x['family']=='exact'); ref_frame=exact['ungauged_frame']
        if any(x['ungauged_frame']!=ref_frame for x in ls): raise ValueError('ungauged reference identity')
        basis_ref=gaussian_circuit(mat(ref_frame)); Q=gaussian_onebody(basis_ref)
        eta=exact['ungauged_eta']; g0=ivmatrix(mat(c['input_factors'][j])); ref0=conjugated(Q,eta)
        r0=residual(g0,ref0); s0=max(abs(sum((F(eta[i]) for i in a),F(0))) for a in it.combinations(range(4),2))
        factors=[]
        for x in ls:
            et=x['ungauged_eta']; ref1=conjugated(Q,et)
            e=abs(F(c['weights'][j]))*max(abs(sum((F(eta[i]) for i in a),F(0))**2-sum((F(et[i]) for i in a),F(0))**2) for a in it.combinations(range(4),2))
            s1=max(abs(sum((F(et[i]) for i in a),F(0))) for a in it.combinations(range(4),2))
            variants={}
            for gauge,frame,values in [('gauged',x['frame'],x['approx_eta']),('ungauged',x['ungauged_frame'],x['ungauged_eta'])]:
                basis=gaussian_circuit(mat(frame)); qmat=gaussian_onebody(basis); g=conjugated(qmat,values); r1=residual(g,ref1)
                d0,d1=2*r0,2*r1
                bound=e+abs(F(c['weights'][j]))*(d0*(2*s0+d0)+d1*(2*s1+d1))
                cert=coefficient_certificate(squared_interval(g,c['weights'][j]))
                record={'id':x['id'],'family':x['family'],'gauge':gauge,'spectrum':values,
                        'commuting_bound':fj(e),'r0':fj(r0),'r1':fj(r1),'s0':fj(s0),'s1':fj(s1),'weight':fj(F(c['weights'][j])),
                        'model_bound':fj(bound),'pauli_coefficients':cert,'ideal_onebody_midpoint':mj(ivmid(g)),
                        'onebody_interval':[[{'real':[fj(t) for t in endpoints(z.real)],'imag':[fj(t) for t in endpoints(z.imag)]} for z in row] for row in g]}
                variants[gauge]={'record':record,'basis':basis,'terms':exact_diagonal_terms(values,c['weights'][j])}
            factors.append(variants)
        result.append(factors)
    return result


def build_product(blocks,scalar,q,axis):
    n=4; anc=4; qc=QuantumCircuit(5); qc.x(0);qc.x(1);qc.h(anc)
    angle_error=F(0); dt=T/(2*q)
    for _ in range(q):
        for basis,terms in blocks+list(reversed(blocks)):
            qc.compose(basis.inverse(),qubits=range(n),inplace=True)
            for label,coef in terms.items():
                exact=dt*coef; angle=float(exact); angle_error+=abs(F(angle)-exact)
                apply_pauli(qc,label,angle=angle,control=anc)
            qc.compose(basis,qubits=range(n),inplace=True)
    a=float(-T*F(scalar)); angle_error+=abs(F(a)+T*F(scalar)); qc.p(a,anc)
    if axis=='Y':qc.sdg(anc)
    qc.h(anc)
    return qc,angle_error


class Compiler:
    """Cached by the complete emitted circuit and fixed compiler settings."""
    def __init__(self,out):
        self.out=Path(out);self.cache={};self.calls=0;self.hits=0;self.records=[]

    def compile(self,qc,label,reference=None):
        key=digest({'global_phase':float(qc.global_phase),'qubits':qc.num_qubits,'operations':[
            [x.operation.name,[qc.find_bit(q).index for q in x.qubits],[float(v) for v in x.operation.params]] for x in qc.data],
            'seed':SEED,'optimization':1,'basis':['rz','sx','x','cx'],'qubits_initially_zero':False})
        if key in self.cache:
            self.hits+=1;return self.cache[key].copy()
        if self.calls>=1400:raise ValueError('per-context compile cap')
        self.calls+=1
        built=Operator(qc).data
        reference=built if reference is None else reference
        be=float(np.linalg.norm(built-reference,2))
        if be>3e-9:raise AssertionError(('builder absolute semantics',label,be))
        nc=transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=SEED,num_processes=1,qubits_initially_zero=False)
        if nc.size()>20000 or nc.num_qubits>5: raise ValueError('native cap')
        ir={'qubits':nc.num_qubits,'qubits_initially_zero':False,'global_phase':float(nc.global_phase),
            'operations':[{'name':x.operation.name,'qubits':[nc.find_bit(q).index for q in x.qubits],'parameters':[float(p) for p in x.operation.params]} for x in nc.data],
            'logical_reference':mj(reference),'emitted_reference':mj(built),'builder_sha256':key,'label':label}
        eye=np.eye(len(built),dtype=complex);raw=native_action(ir,eye);overlap=np.vdot(built,raw)/len(built)
        re=float(np.linalg.norm(raw-reference,2));sr=float(np.linalg.norm(raw-overlap*built,2));fix=0.
        if re>3e-9:
            if abs(abs(overlap)-1)>3e-9 or sr>3e-9:raise AssertionError(('non-scalar compiler error',label,re))
            fix=-float(np.angle(overlap));ir['global_phase']+=fix
        ne=float(np.linalg.norm(native_action(ir,eye)-reference,2))
        if ne>3e-9:raise AssertionError(('absolute native error',label,ne))
        ir['compiler_phase_audit']={'raw_absolute_error':re,'scalar_residual':sr,'global_phase_correction':fix}
        name='ir_'+key+'.json.gz'; data=gzip.compress(canonical(ir),mtime=0)
        if len(data)>32*1024**2:raise ValueError('IR shard cap')
        (self.out/name).write_bytes(data)
        counts=nc.count_ops(); record={'ir_file':name,'ir_sha256':hashlib.sha256(data).hexdigest(),'builder_sha256':key,
            **{m:int(counts.get(m,0)) for m in METRICS[:4]},'size':nc.size(),'depth':nc.depth(),'native_residual':ne,'builder_residual':be,'phase_correction':fix}
        self.cache[key]=record;self.records.append(record)
        return record.copy()


def select_rows(rows,families,metric):
    eligible=[r for r in rows if all(f in families for f in r['families']) and r['shots'] is not None and fs(r['model_bound'])<=F(.02)]
    secondary='cx' if metric=='rz' else 'rz'
    return min(eligible,key=lambda r:(r['objective'][metric],r['objective'][secondary],fs(r['model_bound']),r['q'],r['id'])) if eligible else None


def close_selection(rows):
    sels={}
    for name,families in [('exact',{'exact'}),('N1',{'exact','cluster'}),('shifted',{'exact','shifted_cutoff'}),('zero',{'exact','zero_cutoff'}),('all_mixed',{'exact','cluster','shifted_cutoff','zero_cutoff'})]:
        sels[name]={m:(r['id'] if (r:=select_rows(rows,families,m)) else None) for m in ('rz','cx')}
    payload={'selections':sels,'rows_sha256':digest(rows),'closed_before_truth':True,'selector_information':'coefficients,certificates,compiler_costs_only'}
    return payload


def ideal_operator(blocks,scalar,q):
    # Post-selection/circuit-reference path: never used by error selector.
    u=np.eye(16,dtype=complex);dt=float(T/(2*q))
    for _ in range(q):
        for basis,terms in blocks+list(reversed(blocks)):
            b=Operator(basis).data
            if all(set(k)<=set('IZ') for k in terms):
                diagonal=np.zeros(16)
                for k,v in terms.items():
                    for s in range(16):diagonal[s]+=float(v)*(-1)**sum((s>>(3-i)&1) for i,z in enumerate(k) if z=='Z')
                step=np.diag(np.exp(-1j*dt*diagonal))
            else:
                if len(terms)!=1:raise ValueError('noncommuting terms within ideal block')
                k,v=next(iter(terms.items()));angle=dt*float(v)
                step=math.cos(angle)*np.eye(16)-1j*math.sin(angle)*pauli_matrix(k)
            u=(b@step@b.conj().T)@u
    return np.exp(-1j*float(T*F(scalar)))*u


def pauli_matrix(label):
    n=len(label);out=np.zeros((1<<n,1<<n),complex)
    for s in range(1<<n):
        t=s;v=1
        for i,p in enumerate(reversed(label)):
            bit=s>>i&1
            if p=='Z':v*=(-1)**bit
            elif p=='X':t^=1<<i
            elif p=='Y':t^=1<<i;v*=1j*(-1)**bit
        out[t,s]=v
    return out


def wrapper_reference(u,axis):
    h=np.array([[1,1],[1,-1]])/math.sqrt(2);had=np.kron(h,np.eye(16));ctrl=np.eye(32,dtype=complex);ctrl[16:,16:]=u
    read=np.diag([1]*16+([-1j]*16 if axis=='Y' else [1]*16));prep=np.zeros((32,32))
    for s in range(32):prep[s^3,s]=1
    return had@read@ctrl@had@prep


def truth_n1(c,rows,factors,closed):
    if closed['rows_sha256']!=digest(rows):raise ValueError('selection changed before truth')
    sec=sector_indices(4,2);h1=second_quantize_one_body(mat(c['one_body'])); original=h1+c['scalar']*np.eye(16)
    for gj,w in zip(c['input_factors'],c['weights']):
        f=second_quantize_one_body(mat(gj));original+=w*f@f
    exact=expm(-1j*float(T)*original[np.ix_(sec,sec)]);initial=sec.index(3);out=[]
    lookup={r['id']:r for r in rows}
    # Audit every closed selector row, using its actual constructed factor Hamiltonian.
    for r in rows:
        a,b=r['factor_indices'];gauge=r['gauge'];vs=[factors[0][a][gauge],factors[1][b][gauge]]
        approx=h1+c['scalar']*np.eye(16)
        for v,w in zip(vs,c['weights']):
            f=second_quantize_one_body(mat(v['record']['ideal_onebody_midpoint']));approx+=w*f@f
        blocks=[(QuantumCircuit(4),onebody_terms(mat(c['one_body'])))]+[(v['basis'],v['terms']) for v in vs]
        pf=ideal_operator(blocks,c['scalar'],r['q'])[np.ix_(sec,sec)]
        ah=approx[np.ix_(sec,sec)];oh=original[np.ix_(sec,sec)];ae=expm(-1j*float(T)*ah)
        model=float(np.linalg.norm(ah-oh,2));pe=float(np.linalg.norm(pf-ae,2));total=float(np.linalg.norm(pf-exact,2))
        if model>float(fs(r['model_bound']))+3e-11 or pe>float(fs(r['pf_bound']))+3e-11 or total>float(fs(r['bias_bound']))+3e-11:
            raise AssertionError(('STOP_CERTIFICATE_SOUNDNESS_FAILURE',r['id'],model,pe,total))
        shots=safe_shots(F(total));out.append({'id':r['id'],'model_error':model,'pf_error':pe,'total_error':total,
           'complex_signal_error':float(abs(pf[initial,initial]-exact[initial,initial])),'diagnostic_shots':shots,
           'diagnostic_work':{m:shots*r['xy_cost'][m] if shots else None for m in ('rz','cx')}})
    return {'rows':out,'original_sector':mj(original[np.ix_(sec,sec)]),'truth_access':'after_selection_hash_closed','closed_selection_sha256':digest(closed)}


def run_n1(task):
    c,outpath=task;start=time.monotonic();out=Path(outpath);out.mkdir()
    resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
    factors=prepare_factors(c);compiler=Compiler(out);rows=[];blocks0=[(QuantumCircuit(4),onebody_terms(mat(c['one_body'])))];gamma=anti_growth(c)
    (out/'factor_certificate_checkpoint.json').write_bytes(canonical([[{g:v['record'] for g,v in x.items()} for x in ls] for ls in factors]))
    basis_costs=[[compiler.compile(x['gauged']['basis'],f'basis/{j}/{i}') for i,x in enumerate(ls)] for j,ls in enumerate(factors)]
    amplification=upper(mp.iv.exp(exact_iv(T*gamma)))
    hcert=coefficient_certificate(orbital_interval(ivmatrix(mat(c['one_body']))))
    for a,b in it.product(range(len(factors[0])),range(len(factors[1]))):
        for gauge in ('gauged','ungauged'):
            vs=[factors[0][a][gauge],factors[1][b][gauge]];certs=[hcert]+[v['record']['pauli_coefficients'] for v in vs]
            pf=pf_certificate(certs);model=sum((fs(v['record']['model_bound']) for v in vs),F(0))
            blocks=blocks0+[(v['basis'],v['terms']) for v in vs]
            for q in (1,2,4,8):
                ident=f'{c["context"]}/{a}/{b}/{gauge}/q{q}'; costs={};angle=None
                # Numerical circuit correctness is permitted information, distinct from H/PF oracle.
                ideal=ideal_operator(blocks,c['scalar'],q)
                for axis in ('X','Y'):
                    qc,err=build_product(blocks,c['scalar'],q,axis);angle=err if angle is None else max(angle,err)
                    costs[axis]=compiler.compile(qc,ident+'/'+axis,wrapper_reference(ideal,axis))
                bpf=T**3*fs(pf['alpha2'])/q**2;bias=T*model*amplification+bpf+angle;shots=safe_shots(bias)
                xy={m:costs['X'][m]+costs['Y'][m] for m in METRICS}
                rows.append({'id':ident,'factor_indices':[a,b],'families':[v['record']['family'] for v in vs],
                    'gauge':gauge,'q':q,'model_bound':fj(model),'pf_certificate':pf,'pf_bound':fj(bpf),'angle_bound':fj(angle),
                    'raw_input_growth_extra':fj(T*model*(amplification-1)),'bias_bound':fj(bias),'shots':shots,'costs':costs,'xy_cost':xy,
                    'objective':{m:shots*xy[m] if shots else None for m in METRICS},
                    'certificate_scope':'ideal_gate_certificate_with_numerical_compile_validation'})
        if (a*len(factors[1])+b)%16==0:
            (out/'rows_checkpoint.json').write_bytes(canonical({'completed_pair':[a,b],'rows':rows}))
    closed=close_selection(rows)
    closed['basis_only']={}
    pairs=[(a,b) for a,b in it.product(range(len(factors[0])),range(len(factors[1]))) if all(v['record']['family'] in ('exact','cluster') for v in (factors[0][a]['gauged'],factors[1][b]['gauged']))]
    for metric in ('rz','cx'):
        a,b=min(pairs,key=lambda ab:(basis_costs[0][ab[0]][metric]+basis_costs[1][ab[1]][metric],ab))
        closed['basis_only'][metric]={'factor_indices':[a,b],'gauge':'gauged','basis_cost':basis_costs[0][a][metric]+basis_costs[1][b][metric]}
    (out/'selection.json').write_bytes(canonical(closed))
    diagnostics=truth_n1(c,rows,factors,closed)
    # Whole-Pauli original and closed N1 RZ/CX selections, no new global candidate scan.
    whole=[]; targets=[('original',None)]
    byid={r['id']:r for r in rows}
    for metric,id_ in closed['selections']['N1'].items():
        if id_: targets.append(('N1_'+metric,byid[id_]))
    for target,r in targets:
        if r is None:
            ts=orbital_interval(ivmatrix(mat(c['one_body'])))
            for gj,w in zip(c['input_factors'],c['weights']):ts=pdadd(ts,squared_interval(ivmatrix(mat(gj)),w))
            ts=pdadd(ts,{'IIII':exact_iv(c['scalar'])});model=gamma
        else:
            a,b=r['factor_indices'];gauge=r['gauge'];ts=orbital_interval(ivmatrix(mat(c['one_body'])))
            for v in (factors[0][a][gauge],factors[1][b][gauge]):
                for k,(lo,hi) in v['record']['pauli_coefficients'].items():
                    ts[k]=ts.get(k,0)+mp.iv.mpf([lo,hi])/D
            ts=pdadd(ts,{'IIII':exact_iv(c['scalar'])});model=fs(r['model_bound'])
        # Exact dyadic midpoint terms are the separately implemented Pauli comparator.
        cert=coefficient_certificate(ts);terms={k:F((lo+hi)//2,D) for k,(lo,hi) in sorted(cert.items()) if (lo+hi)//2}
        midpoint_delta=sum((F(max((lo+hi)//2-lo,hi-(lo+hi)//2),D) for lo,hi in cert.values()),F(0));model+=midpoint_delta
        singleton=[{k:[int(v*D),int(v*D)]} for k,v in terms.items()]
        pfc=pf_certificate(singleton);blocks=[(QuantumCircuit(4),{k:v}) for k,v in terms.items()]
        sec=sector_indices(4,2);orig=mat(diagnostics['original_sector']);wh=sum((float(v)*pauli_matrix(k) for k,v in terms.items()),np.zeros((16,16),complex))[np.ix_(sec,sec)]
        wh_evolution=expm(-1j*float(T)*wh);orig_evolution=expm(-1j*float(T)*orig)
        model_truth=float(np.linalg.norm(wh-orig,2))
        for q in (1,2,4,8):
            costs={};angle=F(0)
            ideal=ideal_operator(blocks,0,q)
            for axis in ('X','Y'):
                qc,ae=build_product(blocks,0,q,axis);angle=max(angle,ae);costs[axis]=compiler.compile(qc,target+f'/whole/q{q}/'+axis,wrapper_reference(ideal,axis))
            bias=T*model*amplification+T**3*fs(pfc['alpha2'])/q**2+angle;shots=safe_shots(bias);xy={m:sum(costs[ax][m] for ax in ('X','Y')) for m in METRICS}
            pf_unitary=ideal[np.ix_(sec,sec)];pf_truth=float(np.linalg.norm(pf_unitary-wh_evolution,2));total_truth=float(np.linalg.norm(pf_unitary-orig_evolution,2))
            if model_truth>float(model)+3e-11 or pf_truth>float(T**3*fs(pfc['alpha2'])/q**2)+3e-11 or total_truth>float(bias)+3e-11:raise AssertionError('whole-Pauli certificate containment')
            whole.append({'target':target,'q':q,'model_bound':fj(model),'pf_certificate':pfc,'pauli_coefficients':{k:fj(v) for k,v in terms.items()},
                  'bias_bound':fj(bias),'angle_bound':fj(angle),'shots':shots,'xy_cost':xy,'costs':costs,'objective':{m:shots*xy[m] if shots else None for m in METRICS},
                  'diagnostic':{'model_error':model_truth,'pf_error':pf_truth,'total_error':total_truth,'complex_signal_error':float(abs(pf_unitary[sec.index(3),sec.index(3)]-orig_evolution[sec.index(3),sec.index(3)]))}})
    result={'context':c['context'],'basis_costs':basis_costs,'factors':[[{g:v['record'] for g,v in x.items()} for x in ls] for ls in factors],
      'rows':rows,'selection':closed,'diagnostics':diagnostics,'whole_pauli':whole,'raw_input_antihermitian_growth':fj(gamma),
      'status':'UNRESOLVED_PF_CERTIFICATE' if any(x is None for x in closed['selections']['exact'].values()) else 'LIMITED_N1_COMPLETE',
      'performance':{'wall_seconds':time.monotonic()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     'transpile_calls':compiler.calls,'compile_cache_hits':compiler.hits,'wrapper_requests':sum(2 for _ in rows)+2*len(whole)},
      'mandatory_stop':True,'next_scientific_stage_authorized':False,'central_theme':None}
    (out/'result.json').write_bytes(canonical(result));return {'context':c['context'],'path':str(out),'status':result['status'],'performance':result['performance']}

