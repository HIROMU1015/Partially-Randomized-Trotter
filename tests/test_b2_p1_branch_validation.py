"""Synthetic complex numbers only. No saved molecular signal reanalysis."""
import copy
import math

import numpy as np
import pytest

from trotterlib import b2_p1_branch_validation as m


@pytest.fixture(autouse=True)
def no_private_or_scientific_entry(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('No molecular inputs, simulation or shots in these synthetic tests')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(m,'git',forbidden)


@pytest.mark.parametrize('phase',[.72,-.61])
def test_raw_only_reconstruction_crosses_wrap_with_attenuation(phase):
    qs=[1,2,4,8,16]
    raw=[(.7**j)*np.exp(1j*q*phase) for j,q in enumerate(qs)]
    decoded=m.decode_raw_means(raw,qs)
    assert decoded['status']=='RAW_MEAN_DECODE_PASS'
    assert decoded['final_base_phase']==pytest.approx(phase,abs=1e-14)
    assert decoded['reference_energy_used_in_decoder'] is False
    original_choices=[r['selected_branch'] for r in decoded['rounds']]
    # The five-round toy has nominal resolution 1.25 Ha; use a declared toy
    # target here, without changing the fixed P1 runner's precision condition.
    after=m.posthoc_correctness(decoded,[np.angle(z) for z in raw],-phase/m.DELTA,epsilon=2.)
    assert after['status']=='P1_CONDITIONAL_BRANCH_ENERGY_PASS'
    assert after['final_energy_absolute_error_Ha']<1e-12
    assert [r['selected_branch'] for r in decoded['rounds']]==original_choices


def test_alias_is_from_prior_and_beta_only():
    a=m.initial_alias(.01)
    assert a['status']=='PASS' and a['reference_energy_used'] is False
    assert a['prior_base_phase_interval']==[-2.56,2.56]
    assert a['interval_endpoint_phase_margin_after_beta_rad']==pytest.approx(math.pi-2.56-.4)
    assert m.initial_alias(.01,energy_interval=(-200,200))['status']=='STOP_INITIAL_ALIAS'
    with pytest.raises(ValueError,match='INITIAL_ALIAS'):
        m.decode_raw_means([1+0j,1+0j],[1,2],energy_interval=(-200,200))


def test_existing_tie_rule_stops_and_prevents_posthoc_reference():
    decoded=m.decode_raw_means([1+0j,-1+0j],[1,2])
    assert decoded['status']=='STOP_BRANCH_AMBIGUITY_OR_MISMATCH'
    assert decoded['rounds'][1]['ambiguity_within_principal_window']
    with pytest.raises(ValueError,match='DECODE_BEFORE_REFERENCE'):
        m.posthoc_correctness(decoded,[0,math.pi],0.)


def test_incomplete_grid_and_nonfinite_raw_rejected():
    with pytest.raises(ValueError,match='consecutive powers'):
        m.decode_raw_means([1+0j,1+0j],[1,4])
    with pytest.raises(ValueError,match='FINITE_NONZERO_RAW_MEANS'):
        m.decode_raw_means([0j,1+0j],[1,2])
    with pytest.raises(ValueError,match='FINITE_NONZERO_RAW_MEANS'):
        m.decode_raw_means([complex(float('nan'),0),1+0j],[1,2])


def test_posthoc_oracle_never_repairs_a_saved_choice():
    qs=[1,2,4,8,16]
    raw=[np.exp(1j*q*.12) for q in qs]
    decoded=m.decode_raw_means(raw,qs)
    before=[r['selected_branch'] for r in decoded['rounds']]
    # An intentionally wrong reference fails correctness, without changing the
    # reconstructor input or returning an oracle-repaired energy estimate.
    after=m.posthoc_correctness(decoded,[np.angle(z) for z in raw],40.)
    assert after['status']=='STOP_POSTHOC_CORRECTNESS'
    assert decoded['final_energy_estimate_Ha']==pytest.approx(-6.)
    assert [r['selected_branch'] for r in decoded['rounds']]==before
