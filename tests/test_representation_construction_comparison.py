"""Meaningful small semantic tests; no production runner or molecular data."""
import math
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Pauli
from scipy.linalg import expm

from trottertracks.representation_exploration import construction_comparison as c
from trottertracks.representation_exploration.mechanisms import second_quantize_one_body
from trotterlib.rte import (enumerate_rte_events,event_unitary,exact_enumerated_event_mean_operator,
                           finite_rte_distribution,finite_taylor_operator)


@pytest.mark.parametrize("n",[2,3])
def test_polynomial_jw_matches_independent_fock_reference(n):
    rng=np.random.default_rng(8+n);a=rng.normal(size=(n,n))+1j*rng.normal(size=(n,n));g=(a+a.conj().T)/2
    assert c.norm(c.pdense(c.orbital_paulis(g),n)-second_quantize_one_body(g))<1e-12


@pytest.mark.parametrize("label",["XYZ","YIY","ZZI","III"])
@pytest.mark.parametrize("time",[.17,-.21])
def test_control_only_central_pauli_preserves_absolute_branches(label,time):
    qc=QuantumCircuit(4);c.apply_pauli(qc,label,angle=time,control=3)
    ref=np.eye(16,dtype=complex);ref[8:,8:]=expm(-1j*time*Pauli(label).to_matrix())
    assert c.norm(Operator(qc).data-ref)<1e-12


def test_fixed_frame_projection_is_invariant_to_factor_label_rotation():
    factors=[np.array([[.7,.1,0],[.1,-.2,.05],[0,.05,.3]]),
             np.array([[-.1,0,.08],[0,.5,0],[.08,0,-.4]])]
    angle=.31;rot=np.array([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])
    mixed=[sum(rot[i,j]*g for j,g in enumerate(factors)) for i in range(2)]
    original=c.frame_construct(factors);changed=c.frame_construct(mixed)
    assert c.norm(original.blocks[0].dense()-changed.blocks[0].dense())<1e-12
    assert c.norm(original.dense()-changed.dense())<1e-12


@pytest.mark.parametrize("anchor",[None,0,1])
def test_frame_and_native_constructor_reconstruct_input(anchor):
    factors=[np.array([[.7,.11],[.11,-.2]]),np.array([[-.4,-.07],[-.07,.3]])]
    h=sum(second_quantize_one_body(g)@second_quantize_one_body(g) for g in factors)
    assert c.norm(c.frame_construct(factors,anchor=anchor).dense()-h)<1e-12
    for ld in (0,1,2):assert c.norm(c.native_df_construct(factors,ld).dense()-h)<1e-12


def test_actual_rte_events_and_corrected_mean_match_independent_taylor():
    factors=[np.array([[.3,.05],[.05,-.2]]),np.array([[-.1,.04],[.04,.25]])]
    con=c.frame_construct(factors);tail=con.tail();time=.18
    dist=finite_rte_distribution(time*tail.lambda_r,2)
    events=enumerate_rte_events(tail.components,dist,max_events=2000)
    ops={x.component_id:y for x,y in zip(tail.components,tail.operators)}
    mean=exact_enumerated_event_mean_operator(events,ops)
    polynomial=finite_taylor_operator(tail.normalized_hamiltonian,time*tail.lambda_r,2)
    assert c.norm(dist.exact_finite_distribution*mean-polynomial)<1e-12
    for seed in (1,12,131):
        qc,u,_=c.trajectory(con,time,2,seed)
        ref=np.eye(8,dtype=complex);ref[4:,4:]=u
        assert c.norm(Operator(qc).data-ref)<1e-12


@pytest.mark.parametrize("target",[0,1,2])
@pytest.mark.parametrize("time",[.2,-.13])
def test_degree_two_mixed_oracle_and_control_zero_identity(target,time):
    fields=[.7,-.4,.2];edges=[(0,1,.12),(1,2,-.08)];alpha=.09
    qc=c.mixed_controlled(fields,edges,target,alpha,time)
    a=c.pdense(c.ising_terms(fields,edges),3);p=["I"]*3;p[2-target]="X"
    ref=np.eye(16,dtype=complex);ref[8:,8:]=expm(-1j*time*(a+alpha*Pauli("".join(p)).to_matrix()))
    assert c.norm(Operator(qc).data-ref)<1e-12


def test_dense_graph_degree_limit_is_explicit_and_not_a_no_go():
    meta=c.mixed_access_metadata([1.]*4,[(0,1,.1),(0,2,.1),(0,3,.1)])
    assert not meta['accepted'] and meta['branch_counts']==[8,2,2,2]
    with pytest.raises(ValueError,match='degree cap'):
        c.mixed_controlled([1.]*4,[(0,1,.1),(0,2,.1),(0,3,.1)],0,.1,.2)


@pytest.mark.parametrize("edges",[[(0,1,.1),(1,0,.2)],[(0,0,.1)],[(0,1,float('nan'))]])
def test_ambiguous_or_invalid_graphs_are_rejected(edges):
    with pytest.raises(ValueError):c.mixed_access_metadata([1.,.7],edges)


@pytest.mark.parametrize("method",["ordinary_first","symmetric_s2","thrift"])
def test_repeated_comparators_preserve_product_order_and_boundary_merge(method):
    qc,u,_=c.c_evolution([.8,.3],[(0,1,-.11)],.13,.24,3,method)
    ref=np.eye(8,dtype=complex);ref[4:,4:]=u
    assert c.norm(Operator(qc).data-ref)<1e-12


def test_isometry_fixture_is_projected_quartic_not_pauli_aux_toy():
    enlarged,direct,meta=c.enlarged_connection()
    hb=enlarged.dense();hp=direct.dense()
    assert meta['encoding_residual']<1e-12
    assert c.norm(hb[:8,:8]-hp)<1e-12
    assert c.norm(hb[8:,:8])<1e-12
    assert c.norm(np.array(meta['enlarged_hamiltonian']['real'])[8:,:8])>1e-3
    qc,u,_=c.trajectory(enlarged,.1,1,28)
    ref=np.eye(32,dtype=complex);ref[16:,16:]=u
    assert c.norm(Operator(qc).data-ref)<1e-12


def test_bias_reserve_and_paired_axis_cost_uncertainty():
    epsilon=.05;b=epsilon/math.sqrt(2)
    assert c.shot_budget(1,b,epsilon) is None
    assert c.shot_budget(1,0,epsilon)<c.shot_budget(1,b/2,epsilon)
    rows=[{'axis':axis,'cost':{m:float(t) for m in c.METRICS}} for t in (2,4)
          for axis in ('X','Y')]
    rec=c.summarize_cost(rows,1,0)[0];n=rec['shots_per_axis']
    assert rec['expected_work']['rz']==6*n
    assert rec['expected_work_se']['rz']==pytest.approx(2*n)


def test_saved_native_ir_keeps_phase_and_reconstructs_absolute_operator():
    qc=QuantumCircuit(2);qc.h(1);qc.crx(.2,1,0);qc.h(1)
    audit=c.NativeAudit(cap=1);ref=Operator(qc).data;cost=audit.compile(qc,ref,'test')
    ir=audit.records[0]
    assert c.norm(Operator(c.reconstruct_ir(ir)).data-ref)<c.ATOL
    assert cost['size']==len(ir['operations'])
    with pytest.raises(ValueError,match='cap'):audit.compile(qc,ref,'over cap')
