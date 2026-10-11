"""Frozen finite N1/N3 redesign. Certificates never consume spectral diagnostics.

N1 uses 128-bit outward intervals, then exact dyadic Pauli commutators.
N3 uses rational coefficient operations only. Dense matrices belong to the
post-selection diagnostic functions and circuit semantics verification.
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
from scipy.optimize import linprog

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


def load_contexts(path):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=INPUT_SHA: raise ValueError('selector input hash')
    p=json.loads(raw)
    for c in p['contexts']:
        if set(c)!=CONTEXT_KEYS: raise ValueError('oracle/unknown context field refused')
        for ls in c['generator_candidates']:
            for x in ls:
                if set(x)!=CANDIDATE_KEYS: raise ValueError('oracle/unknown candidate field refused')
    return p['contexts']


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


def rational_ldl(m):
    n=len(m);l=[[F(int(i==j)) for j in range(n)] for i in range(n)];d=[]
    for i in range(n):
        di=m[i][i]-sum((l[i][k]**2*d[k] for k in range(i)),F(0))
        if di<=0:return None
        fj(di);d.append(di)
        for j in range(i+1,n):
            l[j][i]=(m[j][i]-sum((l[j][k]*l[i][k]*d[k] for k in range(i)),F(0)))/di;fj(l[j][i])
    return l,d


def quadratic_inverse(ldl,v):
    l,d=ldl;y=[]
    for i in range(len(v)):y.append(v[i]-sum((l[i][j]*y[j] for j in range(i)),F(0)))
    return sum((y[i]**2/d[i] for i in range(len(v))),F(0))


def n3_inputs():
    out=[]
    for name,tail in [('original_gapped',[6,8]),('original_small_gap',['.4','.6']),('strong_external_coupling',['.4','.6']),('near_external_energy',['.4','-.1'])]:
        es=[F(float(e)) for e in ['-.2','-.1','.1','.25',*tail]];h=[[F(0) for _ in es] for _ in es]
        for i,e in enumerate(es):h[i][i]=e
        for i,j,v in [(1,2,'.2'),(0,3,'.1'),(2,4,'.08'),(3,5,'.16' if name=='strong_external_coupling' else '.04'),(4,5,'.20' if name=='strong_external_coupling' else '.05')]:h[i][j]=h[j][i]=F(float(v))
        out.append({'name':name,'h':h,'density':{(0,1):F(.3)},'particles':2})
    return out


def coefficient_trial(h,density,active,nu):
    dets=list(it.combinations(active,nu));diag=[]
    for d in dets:diag.append(sum((h[i][i] for i in d),F(0))+sum((v for (i,j),v in density.items() if i in d and j in d),F(0)))
    trials=[{'kind':'determinant','occupation':list(d),'value':fj(v)} for d,v in zip(dets,diag)]
    for a,b in it.combinations(range(len(dets)),2):
        i=set(dets[a])-set(dets[b]);j=set(dets[b])-set(dets[a])
        if len(i)==len(j)==1:
            p,q=next(iter(i)),next(iter(j));v=h[p][q]
            if v:
                # Sign follows removal q and insertion p in ascending occupations.
                sg=(-1)**(list(dets[b]).index(q)+list(dets[a]).index(p))
                val=(diag[a]+diag[b])/2-abs(v)
                trials.append({'kind':'connected_equal_pair','occupations':[list(dets[a]),list(dets[b])],'fermionic_matrix_element':fj(sg*v),'value':fj(val)})
    return min(fs(t['value']) for t in trials),trials


def legacy_certificate(h,density,active,nu,delta):
    n=len(h);ext=[i for i in range(n) if i not in active];trial=sorted(active)[:nu]
    U=sum((h[i][i] for i in trial),F(0))+sum((v for (i,j),v in density.items() if i in trial and j in trial),F(0))
    vnorm=sum((abs(h[i][j]) for i in range(n) for j in range(i+1,n)),F(0))+sum(map(abs,density.values()),F(0))
    off=sum((abs(h[i][j]) for i in range(n) for j in range(i+1,n) if i in ext or j in ext),F(0));classes=[];width=F(0);ok=True
    for pos,orb in enumerate(ext):
        free=[i for i in range(n) if i not in ext[:pos+1]]
        lower=h[orb][orb]+sum(sorted(h[i][i] for i in free)[:nu-1],F(0))-vnorm
        d=lower-(len(ext)-1)*off;beta=sum((abs(h[orb][i]) for i in active),F(0));gap=d-U
        if gap<=0:ok=False
        else:width+=beta**2/(gap+delta)
        classes.append({'orbital':orb,'class_lower':fj(lower),'discarded_lower':fj(d),'beta':fj(beta),'gap':fj(gap)})
    return {'method':'legacy_scalar','U':fj(U),'vnorm':fj(vnorm),'off_bound':fj(off),'classes':classes,'rhs_at_goal':fj(width),'accepted':ok and width<=delta,
            'proof':'first-occupied-external classes; triangle hopping norm, block row bound, weighted Schur'}


def direction_certificate(h,density,active,nu,delta,method):
    if any(x<0 for x in density.values()):raise ValueError('attractive density outside class')
    ext=[i for i in range(len(h)) if i not in active]
    if len(ext)!=1:raise ValueError('one external orbital required')
    d=ext[0];m=[[h[i][j] for j in active] for i in active];v=[h[i][d] for i in active]
    U,trials=coefficient_trial(h,density,active,nu)
    r=min(m[i][i]-sum((abs(m[i][j]) for j in range(len(m)) if j!=i),F(0)) for i in range(len(m)))
    s=max(m[i][i] for i in range(len(m)))-min(m[i][i] for i in range(len(m)))
    candidates=[]
    for lam in sorted({r+rho*s for rho in [F(0),F(1,20),F(1,10),F(3,20),F(1,5)]}):
        M=[[m[i][j]-int(i==j)*lam for j in range(len(m))] for i in range(len(m))];ldl=rational_ldl(M)
        if ldl is None:candidates.append({'lambda':fj(lam),'spd':False});continue
        k=quadratic_inverse(ldl,v);candidates.append({'lambda':fj(lam),'spd':True,'kappa':fj(k),'LDL_L':[[fj(x) for x in row] for row in ldl[0]],'LDL_D':[fj(x) for x in ldl[1]]})
    # Gershgorin r is a valid non-strict lower bound even if LDL at r fails.
    ell=max([r]+[fs(x['lambda']) for x in candidates if x['spd']]);c=h[d][d]+(nu-1)*ell;beta2=sum((x*x for x in v),F(0))
    accepted=False;chosen=None
    if not beta2:
        accepted=c>=U;chosen={'zero_coupling':True,'delta':fj(F(0))} if accepted else None
    elif method=='norm_only':accepted=c>U and delta>=beta2/(c-U+delta)
    else:
        for x in candidates:
            if not x['spd']:continue
            L=nu*fs(x['lambda']);k=fs(x['kappa']);t=k/(c-U+delta) if c>U else None
            x.update({'L':fj(L),'t':fj(t) if t is not None else None,'rhs':fj(t*(U-L)) if t is not None else None,
                      'accepted':t is not None and t<=1 and delta>=t*(U-L)})
            if x['accepted'] and chosen is None:chosen=x;accepted=True
    return {'method':method,'U':fj(U),'c':fj(c),'ell':fj(ell),'beta2':fj(beta2),'v':[fj(x) for x in v],'trials':trials,'lambda_candidates':candidates,'accepted':accepted,'chosen':chosen}


def run_n3():
    outputs=[]; closed=[]
    for inp in n3_inputs():
        h,density=inp['h'],inp['density'];methods=[]
        for method in ('legacy_scalar','norm_only','direction'):
            active=[0,1,2,3];records=[]
            for stage in range(3):
                ext=[i for i in range(6) if i not in active]
                if not ext:cert={'method':method,'accepted':True,'exact_full_space':True,'delta':fj(F(0))}
                elif len(ext)==2 or method=='legacy_scalar':cert=legacy_certificate(h,density,active,2,GOAL)
                else:cert=direction_certificate(h,density,active,2,GOAL,method)
                cert.update({'active':active.copy(),'external':ext,'stage':stage,'delta_goal':fj(GOAL)});records.append(cert)
                if cert['accepted']:break
                promote=max(ext,key=lambda j:(sum((abs(h[i][j]) for i in active),F(0)),-j));active=sorted(active+[promote])
            methods.append({'method':method,'records':records,'selected_active':active})
        closed.append({'input':inp['name'],'methods':methods,'h':[[fj(x) for x in row] for row in h],'density':[[i,j,fj(v)] for (i,j),v in density.items()]})
    seal=digest(closed)
    # Only now access many-body spectra for saved numerical diagnostics.
    for item in closed:
        h=np.array([[float(fs(x)) for x in row] for row in item['h']]);H=second_quantize_one_body(h)
        for i,j,v in item['density']:
            for state in range(64):H[state,state]+=float(fs(v))*((state>>i)&1)*((state>>j)&1)
        sec=sector_indices(6,2);Hs=H[np.ix_(sec,sec)];E=float(np.linalg.eigvalsh(Hs)[0]);diags=[]
        for meth in item['methods']:
            for cert in meth['records']:
                active=cert['active'];aids=[i for i,s in enumerate(sec) if all(not(s>>j&1) for j in range(6) if j not in active)];qids=[i for i in range(15) if i not in aids]
                A=Hs[np.ix_(aids,aids)];a=float(np.linalg.eigvalsh(A)[0]);width=a-E;C=Hs[np.ix_(qids,qids)]
                cd=float(np.linalg.eigvalsh(C)[0]) if qids else None
                diag={'method':meth['method'],'stage':cert['stage'],'active_ground_diagnostic':a,'full_ground_diagnostic':E,'width_diagnostic':width,'discarded_ground_diagnostic':cd}
                if cert['accepted'] and width>float(GOAL)+3e-12:raise AssertionError(('STOP_CERTIFICATE_SOUNDNESS_FAILURE',item['input'],diag))
                if 'c' in cert:
                    if cd+3e-12<float(fs(cert['c'])) or a>float(fs(cert['U']))+3e-12:raise AssertionError('coefficient U/c soundness')
                    B=Hs[np.ix_(aids,qids)]
                    for x in cert['lambda_candidates']:
                        if not x['spd']:continue
                        L=2*float(fs(x['lambda']));k=float(fs(x['kappa']));gap=float(np.linalg.eigvalsh(k*(A-L*np.eye(len(A)))-B@B.conj().T)[0])
                        if gap<-3e-12:raise AssertionError('direction matrix inequality')
                        diag.setdefault('dominance_checks',[]).append({'lambda':x['lambda'],'minimum_eigenvalue_diagnostic':gap})
                diags.append(diag)
        outputs.append({'input':item['input'],'diagnostics':diags})
    return {'selector':closed,'closed_selection_sha256':seal,'diagnostics':outputs,'status':'LIMITED_N3_COMPLETE','truth_access':'after_selection_hash_closed'}


def run_n2():
    S=np.array([[1,0],[1,1],[0,1],[-1,0]],int);K=np.array([[.3,-.1],[-.1,.2]]);J=S@K@S.T;J[0,3]+=.001;J[3,0]+=.001
    pairs=list(it.combinations_with_replacement(range(4),2));design=np.array([[S[i,0]*S[j,0],S[i,0]*S[j,1]+S[i,1]*S[j,0],S[i,1]*S[j,1]] for i,j in pairs],float)
    weights=np.array([1 if i==j else 2 for i,j in pairs]);target=np.array([J[i,j] for i,j in pairs]);n=len(pairs)
    Aub=np.block([[design,-np.eye(n)],[-design,-np.eye(n)]]);bub=np.r_[target,-target];cost=np.r_[np.zeros(3),weights]
    lp=linprog(cost,A_ub=Aub,b_ub=bub,bounds=[(None,None)]*3+[(0,None)]*n,method='highs')
    if not lp.success:raise AssertionError(('fixed-S LP',lp.message))
    # Rational dual from returned signs: validate feasibility exactly for both interpretations.
    dual=[F(float(x)).limit_denominator(1000000) for x in lp.ineqlin.marginals]
    for j in range(3):
        if sum((dual[i]*F(Aub[i,j]) for i in range(2*n)),F(0))!=0:raise AssertionError('dual equality')
    if any(v>0 for v in dual):raise AssertionError('dual sign')
    for j in range(n):
        if -dual[j]-dual[n+j]>int(weights[j]):raise AssertionError('dual residual variable feasibility')
    results=[]
    rationalK=[F(3,10),F(-1,10),F(1,5)]
    for name,js in [('intended_rational',[sum((F(design[i,j])*rationalK[j] for j in range(3)),F(0))+(F(1,1000) if pairs[i]==(0,3) else 0) for i in range(n)]),('stored_binary64',[F(x) for x in target])]:
        ks=[F(float(x)).limit_denominator(1000000) for x in lp.x[:3]]
        residuals=[js[i]-sum((F(design[i,j])*ks[j] for j in range(3)),F(0)) for i in range(n)]
        primal=sum((int(weights[i])*abs(residuals[i]) for i in range(n)),F(0));dualobj=sum((dual[i]*js[i] for i in range(n)),F(0))-sum((dual[n+i]*js[i] for i in range(n)),F(0))
        # Exact rational incumbent and dual sandwich; binary optimum not copied from decimal.
        ls=np.linalg.lstsq(design*np.sqrt(weights[:,None]),np.asarray(list(map(float,js)))*np.sqrt(weights),rcond=None)[0]
        l1=float(np.sum(weights*abs(np.asarray(list(map(float,js)))-design@ls)))
        results.append({'interpretation':name,'J_upper':[fj(x) for x in js],'K_primal':[fj(x) for x in ks],'residuals':[fj(x) for x in residuals],
             'primal_L1':fj(primal),'dual_lower':fj(dualobj),'optimality_gap':fj(primal-dualobj),'dual':[fj(x) for x in dual],
             'L1_cap':fj(F(21,10000)),'L1_accepted':primal<=F(21,10000),'Frobenius_fit_L1_diagnostic':l1,'Frobenius_fit_accepted':l1<=.0021})
        if primal<dualobj:raise AssertionError('weak duality')
    return {'status':'FIXED_S_L1_REPAIR_COMPLETE','S':S.tolist(),'upper_pairs':[list(x) for x in pairs],'weights':weights.tolist(),'design':design.tolist(),
            'LP_calls':1,'fixtures':results,'new_S_scan':0,'circuit_compilations':0,'conditions':'Sparse fixed S and small charge count are necessary; signed charge bit width, compute/uncompute, diagonal phases, workspace and precision must also beat direct implementation. No FT comparison performed.'}
