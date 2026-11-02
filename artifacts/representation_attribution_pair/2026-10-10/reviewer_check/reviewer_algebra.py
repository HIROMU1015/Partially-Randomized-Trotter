"""Reviewer algebra on fixed batch-2 inputs, not the Codex runner.
Inputs were transcribed from source ce99b57166a9f422f151f56fa17ed799cd9909dd.
No Qiskit/project import, sampling, synthesis, molecular data, or new parameter sweep.
A uses exact rational arithmetic; B uses binary64 to check a derived identity.
"""
from __future__ import annotations
from itertools import product, combinations
from pathlib import Path
import hashlib, json, math, platform
import numpy as np
import sympy as sp

def jw(n: int):
    out=[]
    for j in range(n):
        a=sp.zeros(2**n)
        for k in range(2**n):
            if (k >> j)&1:
                a[k ^ (1 << j),k]=(-1)**((k & ((1 << j)-1)).bit_count())
        out.append(a)
    return out

def lift(g,a):
    return sum((g[p,q]*a[p].H*a[q] for p in range(len(a)) for q in range(len(a))),sp.zeros(a[0].rows))

PAULI={'I':sp.eye(2),'X':sp.Matrix([[0,1],[1,0]]),'Y':sp.Matrix([[0,-sp.I],[sp.I,0]]),'Z':sp.diag(1,-1)}
def coeffs(h,n):
    c={}
    for word in product('IXYZ',repeat=n):
        p=sp.kronecker_product(*(PAULI[x] for x in word))
        v=sp.simplify(sp.trace(p*h)/2**n)
        if v!=0:c[''.join(word)]=v
    return c

def rational_dict(c):return {k:str(v) for k,v in c.items()}
def l1(c,identity=True):return sum((abs(v) for k,v in c.items() if identity or k!='I'*len(k)),sp.Integer(0))

def check_a():
    rat=sp.Rational;a=jw(3);ns=[x.H*x for x in a]
    g0=sp.Matrix([[rat(8,10),rat(9,100),0],[rat(9,100),rat(-35,100),0],[0,0,rat(15,100)]])
    g1=sp.Matrix([[rat(-2,10),0,0],[0,rat(6,10),rat(7,100)],[0,rat(7,100),rat(-45,100)]])
    gs=[g0,g1];h=sum((lift(g,a)**2 for g in gs),sp.zeros(8))
    ds=[sp.diag(*g.diagonal()) for g in gs]
    d0=sum((lift(d,a)**2 for d in ds),sp.zeros(8))
    dstar=sp.diag(*h.diagonal());r0=h-d0;rstar=h-dstar
    t01=a[0].H*a[1]+a[1].H*a[0];t12=a[1].H*a[2]+a[2].H*a[1]
    extra=rat(81,10000)*(ns[0]+ns[1]-2*ns[0]*ns[1])+rat(49,10000)*(ns[1]+ns[2]-2*ns[1]*ns[2])
    roff=(rat(405,10000)*sp.eye(8)+rat(27,1000)*ns[2])*t01+(rat(105,10000)*sp.eye(8)-rat(28,1000)*ns[0])*t12
    assert dstar-d0==extra
    assert rstar==roff
    assert dstar+rstar==h
    assert dstar==sp.diag(*h.diagonal())
    spectra=coeffs(h,3);old=coeffs(r0,3);new=coeffs(rstar,3)
    # Particle number preservation applies to the sum, not each individual Pauli.
    N=sum(ns,sp.zeros(8));assert h*N-N*h==sp.zeros(8)
    hc=np.array(h,dtype=complex)
    return {'scope':'exact rational identities on original fixed A input; no costs recalculated',
        'full_H_paulis':rational_dict(spectra),'old_core_paulis':rational_dict(coeffs(d0,3)),
        'complete_diagonal_core_paulis':rational_dict(coeffs(dstar,3)),
        'old_residual_paulis':rational_dict(old),'new_residual_paulis':rational_dict(new),
        'old_residual_nonidentity_l1':str(l1(old,False)),
        'new_residual_nonidentity_l1':str(l1(new,False)),
        'full_H_collected_nonidentity_l1':str(l1(spectra,False)),
        'full_H_identity':str(spectra['III']),
        'extra_diagonal_absorbed_nonidentity_l1':str(l1(coeffs(extra,3),False)),
        'new_diagonal_supports':sorted(set(coeffs(dstar,3))-set(coeffs(d0,3))),
        'full_H_operator_norm_diagnostic':float(np.linalg.norm(hc,2)),
        'one_particle_effective_matrix':[[str(x) for x in row] for row in (g0*g0+g1*g1).tolist()],
        'one_particle_scope_note':'The fixed prepared state has one particle; an effective one-body evolution is task-equivalent on that sector only, not full Fock operator-equivalent.',
        'all_identity_checks':True}

def fock(u):
    n=len(u);d=2**n;out=np.zeros((d,d),complex)
    for i in range(d):
        r=[p for p in range(n) if (i>>p)&1]
        for j in range(d):
            c=[p for p in range(n) if (j>>p)&1]
            if len(r)==len(c):out[i,j]=np.linalg.det(u[np.ix_(r,c)]) if r else 1
    return out

def check_b():
    n=3;m=4;v=np.eye(m)
    for i,j,angle in [(0,1,math.pi/8),(2,3,math.pi/6),(1,2,math.pi/10)]:
        g=np.eye(m);g[i,i]=g[j,j]=math.cos(angle);g[i,j]=-math.sin(angle);g[j,i]=math.sin(angle)
        v=v@g  # right multiplication as in the frozen source
    u=v[:n,:];edges=[(0,1,.7),(1,2,.4),(2,3,.25),(0,3,-.2)]
    a=[np.array(x,dtype=complex) for x in jw(n)];N4=[np.array(x.H*x,dtype=complex) for x in jw(m)]
    diagonal=sum((w*N4[i]@N4[j] for i,j,w in edges),np.zeros((16,16),complex))
    gamma=fock(v);h=(gamma@diagonal@gamma.conj().T)[:8,:8]
    target=np.zeros_like(h);pair_total=np.zeros_like(h);pairs=[];q_reconstruction=np.zeros_like(h)
    for i,j,w in edges:
        x=u[:,i];y=u[:,j];cx=sum((x[p]*a[p] for p in range(n)),np.zeros((8,8),complex));cy=sum((y[p]*a[p] for p in range(n)),np.zeros((8,8),complex))
        term=cx.conj().T@cy.conj().T@cy@cx
        det=float((np.vdot(x,x)*np.vdot(y,y)-abs(np.vdot(x,y))**2).real)
        e=x/np.linalg.norm(x);z=y-e*np.vdot(e,y);f=z/np.linalg.norm(z)
        b1=sum((e[p]*a[p] for p in range(n)),np.zeros((8,8),complex));b2=sum((f[p]*a[p] for p in range(n)),np.zeros((8,8),complex))
        nt1=b1.conj().T@b1;nt2=b2.conj().T@b2;pair=det*nt1@nt2
        residual=float(np.linalg.norm(term-pair,2))
        assert residual<1e-13
        target+=w*term;pair_total+=w*pair
        projector=nt1@nt2; q=np.eye(8)-2*projector
        assert np.linalg.norm(q@q-np.eye(8),2)<1e-13
        q_reconstruction+=(w*det/2)*(np.eye(8)-q)
        pairs.append({'indices':[i,j],'w':w,'gram_determinant':det,'weighted_pair':w*det,'pair_identity_residual':residual})
    assert np.linalg.norm(target-h,2)<1e-13
    abs_sum=sum(abs(p['weighted_pair']) for p in pairs)
    return {'scope':'same fixed isometry, real orbital entries, binary64 verification of derived exact pair identity',
        'isometry_residual':float(np.linalg.norm(u@u.conj().T-np.eye(n),2)),
        'quartic_projection_residual':float(np.linalg.norm(target-h,2)),
        'physical_pair_reconstruction_residual':float(np.linalg.norm(pair_total-h,2)),
        'projector_involution_reconstruction_residual':float(np.linalg.norm(q_reconstruction-h,2)),
        'projector_involution_unmerged_l1':.5*abs_sum,
        'projector_involution_scalar_offset':.5*sum(p['weighted_pair'] for p in pairs),
        'pair_rows':pairs,'pair_coefficient_l1':abs_sum,
        'unmerged_pair_pauli_nonidentity_l1_upper':.75*abs_sum,
        'unmerged_pair_identity_sum':.25*sum(p['weighted_pair'] for p in pairs),
        'compiled_cost_evaluated':False,'dominance_claim':False}

def stats():
    lam=.9004121714431494;t=.4;tau=lam*t
    b0=math.sqrt(1+tau*tau);b2=tau*tau/2*math.sqrt(1+(tau/3)**2);b=b0+b2;p2=b2/b
    floor={str(e):math.ceil(2/(e/math.sqrt(2))**2*math.log(80)) for e in (.05,.02)}
    # Saved report numbers; no claim of a new independent cost estimate.
    l0={'rz':603525,'cx':194985,'N':9285};identity={'rz':514872,'cx':582806.5,'N':7151}
    refs=[(363426,285040,1024628,729872),(2327946,1825840,6412320,4567680),
          (370617,290680,1027402,731848),(2446980,1919200,6455974,4598776),
          (608839,537676,5090400,4072320),(4623619,4083196,32214960,25771968),
          (718179,634236,5220000,4176000),(6291516,5615568,34335360,27468288)]
    return {'rare_event':{'tau':tau,'normalization':b,'order2_probability':p2,'probability_no_order2_in_8':(1-p2)**8,
                         'expected_order2_in_8':8*p2,'unmerged_single_step_event_count':12+12**3},
            'shot_policy_bias0_normalization1_floor':floor,
            'A_report_factorization':{'RZ_ratio_identity_to_allR':identity['rz']/l0['rz'],
                'CX_ratio_identity_to_allR':identity['cx']/l0['cx'],'N_ratio':7151/9285,
                'RZ_cost_pair_allR':603525/9285,'RZ_cost_pair_identity':514872/7151,
                'CX_cost_pair_allR':194985/9285,'CX_cost_pair_identity':582806.5/7151,
                'warning':'Same seeds are reused across candidates; no between-candidate independence or significance assumed'},
            'C_THRIFT_to_S2_ratios':[{'rz':c/a,'cx':d/b} for a,b,c,d in refs],
            'B_report_ratios':{'rz':711022/145217,'cx':226320/57322.5,'shots':7544/7643},
            'fixed_policy_headroom':{
                'A_frame_identity_max_relative_shot_saving':1-floor['0.05']/7151,
                'B_reflection_max_relative_shot_saving':1-floor['0.05']/7544,
                'B_report_mean_pair_rz':711022/7544,
                'B_report_mean_pair_cx':226320/7544,
                'A_factor0_report_mean_pair_rz':891450/7075},
            'new_quantum_shots_or_compilations':0}

def main():
    out={'provenance':{'source_commit':'ce99b57166a9f422f151f56fa17ed799cd9909dd',
            'result_commit':'c24dcede10ed9726e1e0b1030eca749ba7bc5cba',
            'inputs':'manual exact transcription from fetched source and report; not a downloaded full-result clone',
            'scope':'independent reviewer checks, not preregistered Codex results'},
         'A':check_a(),'B':check_b(),'accounting':stats(),
         'review_runtime':{'python':platform.python_version(),'numpy':np.__version__,'sympy':sp.__version__},
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    p=Path(__file__).with_suffix('.json');p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
