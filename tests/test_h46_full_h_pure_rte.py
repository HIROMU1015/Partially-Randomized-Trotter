"""Small synthetic checks; no molecule loading, native compilation or pricing."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_full_h_pure_rte as full
full.setup()
from trotterlib.rte import (RTEComponent,finite_rte_distribution,enumerate_rte_events,
    exact_enumerated_event_mean_operator,finite_taylor_operator)


@pytest.mark.parametrize('K',[4,6])
def test_component_event_mean_normalization(K):
    z=np.diag([1.,-1.]);x=np.array([[0.,1.],[1.,0.]])
    # An explicit one-body-like Z and a noncommuting fragment-like X.
    components=(RTEComponent('one',.3,.3,-1),RTEComponent('two',.7,.7,1))
    ops={'one':-z,'two':x}
    for tau in (.17,-.17):
        dist=finite_rte_distribution(tau,K)
        events=enumerate_rte_events(components,dist)
        raw=exact_enumerated_event_mean_operator(events,ops)
        corrected=finite_taylor_operator(-.3*z+.7*x,tau,K)
        assert np.linalg.norm(raw*dist.exact_finite_distribution-corrected)<2e-14


def test_full_preparation_has_one_body_in_random_tail():
    from trotterlib.df_hamiltonian import DFHamiltonian
    ham=DFHamiltonian(.23,np.array([[.6,.1j],[-.1j,-.2]]),np.array([.4]),
        (np.array([[.2,.03],[.03,.1]]),),{})
    prep=full.prepare_full_h(ham)
    assert prep.deterministic_blocks==()
    assert prep.deterministic_block_order==()
    assert prep.randomized_block_indices==(0,)
    components=prep.rte_preparation.symbolic_tail.components
    one=[c for c in components if c.df_fragment_id=='full-h-one-body']
    assert len(one)==2 and sum(c.probability for c in one)>0
    assert abs(sum(c.probability for c in components)-1)<1e-14
    assert prep.constant_coefficient==.23
