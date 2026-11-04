"""Synthetic operators/laws only. Default real-data checks never evaluate new bounds."""
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import math
import sys

import numpy as np
import pytest
from flint import acb_mat,ctx

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT/'src')]
import sharpness_adapter as s
import run_weighted_transfer as runner
from integration_audit import Unknown
from trotterlib.research_direction_sequence_policy import source_basis_run_lengths


def events_for_orders(orders, choices, colors):
    """Application order is products then rotation; only metadata needed by SOURCE."""
    events=[];position=0
    for n in orders:
        applications=[]
        for _ in range(n+1):
            color=colors[choices[position]];position+=1
            applications.append(SimpleNamespace(is_identity=False,basis_id=str(color),basis_hash='hash-'+str(color)))
        events.append(SimpleNamespace(application_sequence=tuple(applications)))
    return tuple(events)


def all_words(r, p, weights, colors, sine):
    """Exact finite enumeration; independent orders/components."""
    for orders in product((0,2),repeat=r):
        coefficients=tuple(c for n in orders for c in ((sine[0],) if n==0 else (Q(1),Q(1),sine[1])))
        order_probability=math.prod(weights[n//2] for n in orders)
        for choices in product(range(len(p)),repeat=len(coefficients)):
            probability=order_probability*math.prod(p[i] for i in choices)
            runs=source_basis_run_lengths(events_for_orders(orders,choices,colors))
            yield orders,choices,coefficients,runs,probability


@pytest.mark.parametrize('r,count',[(1,3),(2,3),(3,2)])
def test_closed_form_matches_actual_policy_enumeration(r,count):
    p=([Q(1,2),Q(1,3),Q(1,6)] if count==3 else [Q(1,4),Q(3,4)])
    colors=(['A','A','B'] if count==3 else ['A','B'])
    errors=[Q(i+1,100) for i in range(count)]
    weights=[Q(3,4),Q(1,4)];sine=[Q(1,8),Q(1,20)]
    enumerated=Q(0)
    for _,choices,coeff,runs,probability in all_words(r,p,weights,colors,sine):
        cost=sum((c*errors[i] for i,c,length in zip(choices,coeff,runs,strict=True) if length==1),Q(0))
        enumerated+=probability*cost
    moments=s.grouped_moments(p,errors,colors)
    closed=s.singleton_mean_tail(r,weights,sine,moments)
    assert closed==enumerated
    assert closed<=s.unrestricted_mean_tail(r,weights,sine,moments[0])<=3*r*max(errors)


@pytest.mark.parametrize('r',[1,2,3])
def test_word_dp_matches_exhaustive_certificate_maximum(r):
    p=[Q(1,2)]*2;errors=[Q(1,50),Q(1,100)];colors=['A','B']
    sine=[Q(1,8),Q(1,20)]
    exhaustive=max(sum((c*errors[i] for i,c,length in zip(choices,coeff,runs,strict=True) if length==1),Q(0))
        for _,choices,coeff,runs,_ in all_words(r,p,[Q(1,2)]*2,colors,sine))
    assert s.singleton_word_tail(r,dict(zip(colors,errors)),sine)==exhaustive
    assert exhaustive<=r*(2+sine[1])*max(errors)


def test_one_basis_policy_endpoints_and_rare_orders():
    moments=s.grouped_moments([Q(1,2)]*2,[Q(1,10),Q(1,5)],['A','A'])
    assert moments==(Q(3,20),Q(0),Q(0))
    assert s.singleton_mean_tail(1,[Q(3,4),Q(1,4)],[Q(1,8),Q(1,20)],moments)==Q(9,640)
    assert s.singleton_mean_tail(2,[Q(3,4),Q(1,4)],[Q(1,8),Q(1,20)],moments)==0
    assert s.singleton_word_tail(32,{'A':Q(1,5)},[Q(1,8),Q(1,20)])==0


def test_unequal_component_masses_improve_max_without_cancellation():
    moments=s.grouped_moments([Q(999,1000),Q(1,1000)],[Q(1,1000),Q(1)],['A','B'])
    assert moments[0]<Q(1,100)
    value=s.singleton_mean_tail(2,[Q(999,1000),Q(1,1000)],[Q(1,100),Q(1,200)],moments)
    assert 0<=value<s.unrestricted_mean_tail(2,[Q(999,1000),Q(1,1000)],[Q(1,100),Q(1,200)],moments[0])<Q(1,100)
    assert s.mean_extra(8,value,Q(1,10000),0,Q(1))<=Q(1)


def synthetic_operator_means(r=2):
    # Noncommuting full/supported signed involutions; no chemistry or saved input.
    x=np.array([[0,1],[1,0]],complex);z=np.diag([1.,-1.]);eye=np.eye(2)
    full=[z,-x]
    selected=[math.cos(.02)*z+math.sin(.02)*x,
              -(math.cos(.04)*x+math.sin(.04)*z)]
    p=[Q(1,4),Q(3,4)];weights=[Q(3,4),Q(1,4)];angles=[.13,.08]
    ref=np.zeros((2,2),complex);actual=ref.copy()
    for orders,choices,_,runs,prob in all_words(r,p,weights,['A','B'],[Q(1),Q(1)]):
        a=eye.copy();b=eye.copy();pos=0
        for n in orders:
            for role in range(n+1):
                i=choices[pos];target=selected[i] if runs[pos]==1 else full[i]
                if role==n:
                    theta=angles[n//2]
                    a=(math.cos(theta)*eye-1j*math.sin(theta)*full[i])@a
                    b=(math.cos(theta)*eye-1j*math.sin(theta)*target)@b
                else:
                    a=full[i]@a;b=target@b
                pos+=1
            a*=(-1)**(n//2);b*=(-1)**(n//2)
        ref+=float(prob)*a;actual+=float(prob)*b
    return ref,actual,p,weights,angles


def test_noncommuting_mean_bound_and_event_factorization_counterexample():
    ref,actual,p,weights,angles=synthetic_operator_means(2)
    with ctx.workprec(96):
        sine=[s.sine_upper(t.hex()) for t in angles]
    moment=s.grouped_moments(p,[Q(1,32),Q(1,16)],['A','B'])
    eta=s.singleton_mean_tail(2,weights,sine,moment)
    assert np.linalg.norm(actual-ref,2)<=float(eta)
    individual_ref,individual_actual,*_=synthetic_operator_means(1)
    assert np.linalg.norm(actual-individual_actual@individual_actual,2)>1e-5
    np.testing.assert_allclose(ref,individual_ref@individual_ref,atol=2e-14)
    x=np.array([[0,1],[1,0]],complex);z=np.diag([1.,-1.]);eye=np.eye(2)
    f=math.cos(.2)*eye-1j*math.sin(.2)*z
    rev=math.cos(.15)*eye-1j*math.sin(.15)*x
    for q in (1,2,7):
        a=np.linalg.matrix_power(rev@actual@f,q)
        b=np.linalg.matrix_power(rev@ref@f,q)
        assert np.linalg.norm(a-b,2)<=float(s.mean_extra(q,eta,0,0,2))


def operation(name,wires,params):
    return dict(name=name,qubits=list(wires),params=[float(x).hex() for x in params])


@pytest.mark.parametrize('support,ops',[
    ((0,),[operation('rz',(0,),(1.1,))]),
    ((0,1),[operation('xx_plus_yy',(0,1),(.7,.2))])])
def test_projector_removes_phase_and_support_frame_gauge(support,ops):
    b,_=s.old.modules()
    with ctx.workprec(96):
        full=b.unit(b.identity(3));selected=s.old.one_particle(ops,3)
        old=s.old.component_gap(full,selected,support)
        new=s.projector_bound(full,selected,support)
    assert old>Q(1,10) and 0<=new<Q(1,10**20)


@pytest.mark.parametrize('support',[(0,),(0,1)])
def test_projector_bound_covers_nontrivial_synthetic_parity(support):
    b,_=s.old.modules()
    ops=[operation('xx_plus_yy',(1,2),(.14,.21)),operation('xx_plus_yy',(0,1),(.22,-.1))]
    with ctx.workprec(96):
        full=b.unit(b.identity(3));selected=s.old.one_particle(ops,3)
        new=s.projector_bound(full,selected,support)
    # Independent Qiskit full-Fock reference, synthetic 3 qubits only.
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import XXPlusYYGate
    from qiskit.quantum_info import Operator
    circuit=QuantumCircuit(3)
    for op in ops: circuit.append(XXPlusYYGate(*(float.fromhex(p) for p in op['params'])),op['qubits'])
    u=Operator(circuit).data
    parity=np.diag([(-1)**sum((i>>j)&1 for j in support) for i in range(8)])
    assert .01<np.linalg.norm(parity-u@parity@u.conj().T,2)<=float(new)+2e-14


def test_projector_retains_radius_and_full_space_parity():
    b,_=s.old.modules()
    with ctx.workprec(96):
        a=b.recenter(b.identity(3),error=Q(1,10000),norm=1)
        c=b.recenter(b.identity(3),error=Q(1,1000),norm=1)
        assert 0<s.projector_bound(a,a,(0,))<s.projector_bound(a,c,(0,))
        n2=b.unit(b.identity(2))
        assert s.projector_bound(n2,n2,(0,1))==0


@pytest.mark.parametrize('failure',[
    lambda:s.grouped_moments([Q(1,2)],[Q(1)],['A']),
    lambda:s.grouped_moments([Q(1)],[None],['A']),
    lambda:s.singleton_mean_tail(0,[Q(1),Q(0)],[Q(1),Q(1)],[Q(1)]*3),
    lambda:s.singleton_mean_tail(1,[Q(1),Q(0)],[Q(2),Q(1)],[Q(1)]*3),
    lambda:s.singleton_mean_tail(1,[Q(1),Q(0)],[Q(1)]*2,[Q(1),Q(2),Q(1)]),
    lambda:s.singleton_word_tail(1,{None:Q(1)},[Q(1)]*2),
    lambda:s.mean_extra(1,Q(-1),0,0,2)])
def test_invalid_or_missing_evidence_is_unknown(failure):
    with pytest.raises((Unknown,ValueError)):failure()


def test_none_never_becomes_zero_and_strict_equality_is_not_pass():
    assert s.mean_extra(1,None,0,0,2) is None
    axis=dict(U_cat_upper='1/10',shot_half_width=dict(lower='1/10',upper='1/10'),
              interval=dict(lower='-1/5',upper='1/5'))
    from integration_audit import classify_axis
    assert classify_axis(axis,'1/5',0)['decision']=='UNDETERMINED'
    assert classify_axis(axis,'1/5',None)['decision']=='UNKNOWN'


def toy_bundle():
    schedule=[1]*8+[2,2,4,8,8,16,16,32,32,32]
    rows=[dict(m=m,q=1<<m,r=r,K=2,axes={a:dict(epsilon='.2') for a in ('cosine','sine')})
          for m,r in enumerate(schedule)]
    axes={m:{a:dict(U_cat_upper='0',shot_half_width=dict(lower='.01',upper='.01'),
                    interval=dict(lower='-.01',upper='.01')) for a in ('cosine','sine')} for m in range(18)}
    return dict(probabilities=[Q(1,2)]*2,errors=[Q(1,10**12),Q(2,10**12)],colors=['A','B'],
                order_masses={r:[Q(3,4),Q(1,4)] for r in set(schedule)},
                original=dict(rows=rows,saved_axes=axes,theta_by_r={r:[(.1).hex(),(.03).hex()] for r in set(schedule)}),
                result=dict(rounds=[dict(m=m,q=1<<m,r=r,K=2,boundary_upper='0',lowering_upper='0',extra_upper='1/10')
                                   for m,r in enumerate(schedule)]))


def test_synthetic_end_to_end_and_partial_unknown(tmp_path,monkeypatch):
    bundle=toy_bundle()
    result=runner.evaluate(tmp_path/'complete',bundle)
    assert result['decision'].startswith('CONDITIONAL_'),result.get('error')
    assert result['counts']==dict(moment_passes_started=1,sine_bounds_started=12,tail_bounds_started=6,
                                  round_transfers_started=18,strict_comparisons_started=36)
    assert result['RPE_tolerance_certified'] is False and result['U_exec_upper'] is result['U_virtual_IID_upper'] is None
    assert result['bound_scope']=='IID_CATEGORICAL_MEAN_ONLY_NOT_POINTWISE'
    assert not result['pi64_added_again']
    with pytest.raises(FileExistsError):runner.evaluate(tmp_path/'complete',bundle)
    bad=deepcopy(bundle);bad['errors'][0]=None
    stopped=runner.evaluate(tmp_path/'unknown',bad)
    assert stopped['status']=='UNKNOWN_MANDATORY_STOP' and not stopped['rounds']
    import json
    assert json.loads((tmp_path/'unknown/RESULT.json').read_text())==stopped


def test_real_default_is_metadata_only(monkeypatch,capsys):
    def forbidden(*args,**kwargs):raise AssertionError('new H4 numerical evaluation')
    for name in ('grouped_moments','projector_bound','singleton_mean_tail','singleton_word_tail','sine_upper','mean_extra'):
        monkeypatch.setattr(s,name,forbidden)
    monkeypatch.setattr(runner,'run_science',forbidden)
    assert runner.main([])==0
    assert 'PREPARED_NOT_SCIENTIFICALLY_EXECUTED' in capsys.readouterr().out


def test_coherent_error_prevents_generic_sublinear_q_claim():
    # Two-mode one-particle Gaussian S2 example, forced n=0/r=1 per tail.
    # The supported involution is a conjugate, but F/R spoil a common global
    # conjugation of the whole evolution. q-tail policy resets allow coherence.
    x=np.array([[0,1],[1,0]],complex);z=np.diag([1.,-1.]);eye=np.eye(2)
    eps,theta,phi=.001,.002,.003
    half=math.cos(phi/2)*eye-1j*math.sin(phi/2)*x
    changed=math.cos(eps)*z+math.sin(eps)*x
    reference=half@(math.cos(theta)*eye-1j*math.sin(theta)*z)@half
    actual=half@(math.cos(theta)*eye-1j*math.sin(theta)*changed)@half
    local=np.linalg.norm(actual-reference,2)
    for q in (1,2,8):
        gap=np.linalg.norm(np.linalg.matrix_power(actual,q)-np.linalg.matrix_power(reference,q),2)
        assert .99*q*local<=gap<=q*local+2e-14


def test_correlated_neighbors_are_outside_the_IID_formula():
    # Same one-position marginals, but always alternating colors: the expected
    # additive singleton cost is larger than the IID expression. No PCG claim.
    p=[Q(1,2)]*2;d=[Q(1,4)]*2;colors=['A','B'];sine=[Q(1,2),Q(1,2)]
    iid=s.singleton_mean_tail(2,[Q(1),Q(0)],sine,s.grouped_moments(p,d,colors))
    correlated=Q(0)
    for word in ((0,1),(1,0)):
        runs=source_basis_run_lengths(events_for_orders((0,0),word,colors))
        correlated+=Q(1,2)*sum((sine[0]*d[i] for i,length in zip(word,runs,strict=True) if length==1),Q(0))
    assert correlated>iid
