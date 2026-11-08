"""Artificial operator checks, not evidence of molecular resource performance."""
from fractions import Fraction as F
import itertools
import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, XXPlusYYGate
from qiskit.quantum_info import Operator
from trottertracks.track_c_df_truncation import coefficients, one_particle, exterior, diagonal, candidates


def test_exterior_matches_full_gaussian_including_complex_givens():
    ops=[(RZGate(.27),(0,)),(XXPlusYYGate(.73,-.21),(0,1)),(XXPlusYYGate(-.4,.6),(1,2)),(RZGate(-.8),(3,))]
    q=QuantumCircuit(4)
    for g,qs in ops:q.append(g,qs)
    full=Operator(q).data;occ=list(itertools.combinations(range(4),2));ids=[sum(1<<i for i in s) for s in occ]
    actual=full[np.ix_(ids,ids)]/full[0,0]
    assert np.linalg.norm(exterior(one_particle(ops,4),occ)-actual)<1e-13


def test_exact_polynomial_and_controlled_identity_phase():
    K=np.array([[.3,-.2],[-.2,.7]]);co=coefficients({'K':K})
    for bits in itertools.product((0,1),repeat=2):
        exact=sum(F.from_float(float(K[i,j]))*bits[i]*bits[j] for i in range(2) for j in range(2))
        polynomial=sum(v*math.prod(1-2*bits[i] for i in s) for s,v in co.items())
        assert polynomial==exact
    for mode in ('term_parity','network'):
        q=QuantumCircuit(3);error=diagonal(q,co,-.137,2,mode);a=Operator(q).data
        expected=np.ones(8,dtype=complex)
        for b in range(4):
            energy=sum(float(v)*np.prod([1-2*((b>>i)&1) for i in s]) for s,v in co.items())
            expected[4+b]=np.exp(1j*.137*energy)
        assert np.linalg.norm(a-np.diag(expected),2)<1e-14+error


def test_menu_keeps_constants_and_has_nested_small_coefficient_sets():
    c=dict(blocks=[dict(coeff={():F(3),(0,):F(1,10**8),(1,):F(1,10**6)}),dict(coeff={():F(2),(0,1):F(1,10000)})])
    menu=candidates(c);small=[r for r in menu if r['name'].startswith('coeff')]
    assert len(small)>=2
    assert all(all(s for _,s in x['drop']) for x in small)
    assert all(set(a['drop'])<=set(b['drop']) for a,b in zip(small,small[1:]))
    assert any(x['name']=='full' and not x['drop'] for x in menu)


def test_closed_p3_adapter_preserves_noncommuting_signed_mean(monkeypatch):
    from trottertracks import track_c_df_truncation_pr as pr
    from trottertracks import track_c_closed_p3 as cp
    ops=[[(XXPlusYYGate(.71,.23),(0,1))],[(RZGate(.4),(0,)),(XXPlusYYGate(-.52,.1),(0,1))]]
    blocks=[];Qs=[]
    for j,seq in enumerate(ops):
        q=QuantumCircuit(2)
        for gate,wires in seq:q.append(gate,wires)
        B=Operator(q).data;support=(j,);z=np.diag([1-2*((b>>j)&1) for b in range(4)])
        blocks.append(dict(ops=seq));Qs.append((-1 if j==0 else 1)*B@z@B.conj().T)
    symbols=[dict(block=j,support=(j,),sign=-1 if j==0 else 1) for j in range(2)]
    monkeypatch.setattr(pr.d,'model',lambda n:dict(modes=2,blocks=blocks))
    monkeypatch.setattr(pr,'partition',lambda n,ld:dict(symbols=symbols))
    monkeypatch.setattr(pr,'isometries',lambda:dict(entries={}))
    g=cp.CachedCanonical((F(1,3),F(2,3)),F(1,10),3,'closed_P3_tail',root_bits=256,probability_bits=256,eta=cp.ETA,rho=cp.RHO)
    actual=np.zeros((4,4),complex)
    for gi,group in enumerate(g.groups):
        if group.mode=='root_collected':pairs=[((),j) for j in range(2)]
        else:
            ids,_,_=g.conditional(group.first);pairs=[((group.first,k),j) for k in ids for j in ids]
        for word,child in pairs:
            rec=g.event(gi,word,child);q,_=pr.event_circuit(2,0,[rec],'network');a=Operator(q).data
            assert np.linalg.norm(a[:4,:4]-np.eye(4))<1e-13
            actual+=float(rec['coefficient'])*a[4:,4:]
    A=(Qs[0]+2*Qs[1])/3;X=-.1j*A;expected=np.eye(4)+X+X@X/2+X@X@X/6
    assert np.linalg.norm(actual-expected)<1e-13


def test_vectorized_frobenius_encloses_directed_reference():
    from trottertracks.track_c_df_truncation import fast_fro,sg
    rng=np.random.default_rng(317)
    for scale in (0.,1e-200,1.,1e100):
        a=scale*(rng.normal(size=(7,11))+1j*rng.normal(size=(7,11)))
        assert fast_fro(a)>=sg.fro(a)


def test_weighted_readout_range_and_finite_arithmetic():
    from trottertracks.track_c_df_truncation_pr import normalized_measurement
    from trottertracks.track_c_df_truncation import sg
    import mpmath as mp
    records=[dict(weight=F(10000001+j,10000000)) for j in range(64)]
    gamma=sg.up(sg.gamma(2*len(records)+4));W=max(float(r['weight']) for r in records)
    log_range=sg.up(len(records)*mp.iv.log(sg.I(W))+mp.iv.log(1+sg.I(gamma)))
    range_upper=sg.up(mp.iv.exp(sg.I(log_range)))
    value=normalized_measurement(records,range_upper,-1)
    exact=math.prod(r['weight'] for r in records)
    target=-mp.mpf(exact.numerator)/exact.denominator/mp.mpf(range_upper)
    assert -1<=value<=0 and abs(mp.mpf(value)-target)<gamma


def test_exact_rare_conditioning_preserves_bernoulli_moment():
    from trottertracks.track_c_df_truncation_pr import first_nonroot_cdf,cp
    D=1<<cp.BITS;root=D-17;r=3;cdf=first_nonroot_cdf(root,r);previous=0;mass=[]
    for value in cdf:mass.append(value-previous);previous=value
    q=F(D-root,D);conditional=sum((F(m,cdf[-1])*(1+(r-1-j)*q) for j,m in enumerate(mass)),F(0))
    assert conditional==r*q/(1-F(root,D)**r)
    assert cdf[-1]>0 and q<float(1e-70)


def test_reordered_negative_time_modules_equal_repeated_controlled_evolution():
    from trottertracks import track_c_df_truncation as d
    blocks=[]
    for j in range(4):
        blocks.append(dict(index=j,ops=[(XXPlusYYGate(.13*(j+1),.17*j),(0,1))],coeff={():F(j+1,7),(j%2,):F((-1)**j,3),(0,1):F(j+2,11)}))
    c=dict(modes=2)
    seq=d.schedule(dict(kind='full_Y4',order=[2,0,3,1]),.07)
    first=seq[:1];last=seq[1:];middle=seq[1:-1]+[(seq[0][0],seq[0][1],2*seq[0][2])]
    a,_=d.circuit_piece(c,blocks,seq*3,'term_parity');b=QuantumCircuit(3)
    counts=np.zeros(4)
    for part in (first,middle,middle,last):
        q,_=d.circuit_piece(c,blocks,part,'network');native=d.compile_reference(q);counts+=d.metric(native);b.compose(native,inplace=True)
    # Compiler A promises the already-controlled joint circuit up to ONE
    # common global phase. Align that phase on both control branches together.
    actual=Operator(b).data;expected=Operator(a).data;phase=np.trace(expected.conj().T@actual);phase/=abs(phase)
    assert np.linalg.norm(expected-actual/phase,2)<2e-13
    assert d.metric(b)[0]==counts[0] and d.metric(b)[1]==counts[1] and d.metric(b)[3]==counts[3]
    assert d.metric(b)[2]<=counts[2]
