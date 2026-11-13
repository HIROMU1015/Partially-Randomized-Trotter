"""Artificial algebra and compiler counterexamples, never saved H4 analysis."""
from fractions import Fraction as F
import itertools

import pytest
from trottertracks import outer_window_residual_design as d


def test_finite_interval_identity_for_an_arbitrary_nonlocal_function():
    # No locality is presumed: a deliberately nonlinear interval functional.
    letters = [2, 7, -3, 5, 11]
    def cost(left, right):
        return sum(letters[left:right])**3 + 13 if left < right else 13
    total = F(13)
    for i in range(len(letters)):
        for j in range(i+1, len(letters)+1):
            if j == i+1:
                inc = cost(i,j)-13
            else:
                inc = cost(i,j)-cost(i+1,j)-cost(i,j-1)+cost(i+1,j-1)
            total += inc
    assert total == cost(0,len(letters))


@pytest.mark.parametrize('q',[1,2,4,17])
def test_stationary_telescoping_and_collapsed_stencil(q):
    values = {n: F(n**3 + 3*n*n - 4*n + 7) for n in range(q+1)}
    for h in range(1,q+1):
        expanded = values[0]
        for k in range(1,h+1):
            delta = sum(v*values[j] for j,v in d.connected_coefficients(k).items())
            expanded += (q-k+1)*delta
        result = d.supplied_combination(q,h,values)
        assert result['value'] == expanded
        assert not result['qualified_for_candidate']
        if h == q:
            assert result['value'] == values[q]
        elif q > 1:
            assert result['value'] != values[q]  # short windows do not prove the tail


def test_common_caps_and_constant_boundary_cancel_algebraically():
    first = d.supplied_combination(20,2,{1:F(3),2:F(8)})['value']
    second = d.supplied_combination(20,2,{1:F(3+17),2:F(8+17)})['value']
    assert second-first == 17  # wrapper/scalar is present once, never q times


def test_missing_tail_and_covariance_are_not_zero():
    assert d.supplied_combination(10,2,{1:3})['value'] is None
    assert d.supplied_combination(10,2,{1:3,2:8})['compiler_influence_tail_bound'] is None
    assert d.conditional_geometric_tail(10,2,amplitude=None,decay=None) is None
    assert d.paired_mean_variance(10,2,variance_hi=4,variance_lo=4,covariance=None,n=4) is None
    # Perfectly correlated windows: a+b=1, so variance is 4/n rather than (a²+b²)4/n.
    assert d.paired_mean_variance(10,2,variance_hi=4,variance_lo=4,covariance=4,n=4) == 1


def test_conditional_tail_sum_matches_finite_expansion():
    for q in (2,5,12):
        for h in range(1,q+1):
            for rho in (F(0),F(1,3),F(7,8)):
                expected = sum((q-k+1)*F(3)*rho**(k-h-1) for k in range(h+1,q+1))
                assert d.conditional_geometric_tail(q,h,amplitude=3,decay=rho) == expected


def test_invalid_domain_and_covariance_are_rejected():
    for q,h in ((True,1),(0,1),(3,0),(2,3)):
        with pytest.raises(ValueError):d.truncation_coefficients(q,h)
    with pytest.raises(ValueError):d.conditional_geometric_tail(5,2,amplitude=1,decay=1)
    with pytest.raises(ValueError):d.paired_mean_variance(5,2,variance_hi=1,variance_lo=1,covariance=2,n=4)
    with pytest.raises(ValueError):d.supplied_combination(5,2,{1:float('nan'),2:3})


def test_existing_mean_kernel_handles_a_basis_pair_reward_without_new_states():
    from trottertracks.mean_compiled_price import MeanKernel
    p = (F(1,4),F(3,4));states=('entry','a','b');letters=('a','b')
    # Count a->b only; this needs previous basis, not an independent-state assumption.
    kernel = MeanKernel.from_events(states,letters,[float(x) for x in p],
        lambda state, token:(token,int(state=='a' and token=='b')))
    value,_ = kernel.repeat(4).expected([1.,0.,0.],[0.,0.,0.])
    exact = F(0)
    for seq in itertools.product(range(2),repeat=4):
        probability = F(1)
        for x in seq:probability*=p[x]
        exact += probability*sum(a==0 and b==1 for a,b in zip(seq,seq[1:]))
    assert value == float(exact)


def _compile_native(qc):
    import qiskit
    from functools import partial
    from trotterlib import rte_compiled_cost as cost
    from trotterlib.rte import CompilerSettings
    old = cost.transpile
    cost.transpile = partial(qiskit.transpile,num_processes=1)
    try:
        return cost.transpile_and_measure_cost(qc,CompilerSettings(
            ('rz','sx','x','cx'),None,None,1,None,None,17,'1.3.0'))
    finally:
        cost.transpile = old


def test_shorter_single_qubit_sequence_can_increase_rz_count():
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(1);qc.sx(0);qc.rz(.31,0)
    for _ in range(3):qc.sx(0)
    qc.rz(.47,0);qc.sx(0)
    compiled=_compile_native(qc)
    assert len(qc.data)==7 and compiled.circuit_size==5
    assert compiled.rz_count==3  # input has only two RZ gates


@pytest.mark.parametrize('length',[2,8,32])
def test_arbitrarily_deep_mirror_cancellation_breaks_fixed_radius_assumption(length):
    from qiskit import QuantumCircuit
    from trottertracks.h4_df10_price_v1 import Accounting
    from trottertracks.compiled_price_fast import price_accounting
    qc = QuantumCircuit(3);a=price_accounting(Accounting)()
    qc.rz(.417,0);a.gate('rz',(0,),(.417,))
    word=[(0,1),(1,2)]*length
    for wires in word+list(reversed(word)):
        qc.cx(*wires);a.gate('cx',wires)
    qc.rz(-.417,0);a.gate('rz',(0,),(-.417,))
    assert a.finish()['structural_rz'] == 2
    compiled=_compile_native(qc)
    assert compiled.rz_count == 0 and compiled.cx_count == 0
    # A SOURCE-level synthetic counterexample, not a legal-H4 probability claim.
