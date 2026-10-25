"""Finite-bit categorical law and cohort separation, using artificial data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h46_long_time_native as n
n.study.install()


def test_cdf_discretization_preserves_mass_and_small_probability():
    value=n.cdf_law([.75,.25-2**-40,2**-40])
    assert sum(value['probabilities'])==1.
    assert value['probabilities'][-1]>0
    assert value['L1_roundoff']<1e-14


def test_tiny_ideal_probability_is_distinguished_from_finite_grid():
    value=n.cdf_law([1.,1e-25])
    assert value['probabilities']==[1.,0.]
    assert value['L1_roundoff']>0


def test_confirmation_uses_disjoint_seed_namespace():
    row=dict(id='synthetic')
    a={n.trial_seed(row,'EXPLORATION',i,o,p) for i in range(9) for o in range(3) for p in range(11)}
    b={n.trial_seed(row,'CONFIRMATION',i,o,p) for i in range(33) for o in range(3) for p in range(11)}
    assert len(a)==297 and len(b)==1089 and not a&b
