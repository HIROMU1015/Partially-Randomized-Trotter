"""Invented scalar/schema cases only; no stored molecular result is opened."""
from copy import deepcopy
from fractions import Fraction
import pytest

from trottertracks.o1_m2_core_only_qualification import assess_model_omission_budget


def toy():
    return dict(reference='toy mathematical B',sector=[1,0],
        energy_bound_reference_H_H_vs_C_H=dict(upper_bound=.125,exact_fraction='1/8'))


def assess(result=None, **changes):
    kwargs=dict(physical_reference='toy physical H',model_b_reference='toy mathematical B',
        model_sector=(1,0),model_error_upper_bound=.125,model_omission_allocation=.25)
    kwargs.update(changes)
    return assess_model_omission_budget(toy() if result is None else result,**kwargs)


def test_conditional_partial_budget_does_not_qualify_physics_or_rpe():
    out=assess()
    assert out['budget_status']=='CONDITIONALLY_WITHIN_DECLARED_MODEL_OMISSION_ALLOCATION'
    assert out['combined_upper_bound_exact_fraction']=='1/4'
    assert out['physical_model_status']==out['native_core_status']==out['full_rpe_status']=='UNASSESSED'
    assert out['residual_randomization_needed'] is None


@pytest.mark.parametrize('key',['physical_reference','model_b_reference','model_sector',
    'model_error_upper_bound','model_omission_allocation'])
def test_missing_condition_stays_unassessed_without_sum(key):
    out=assess(**{key:None})
    assert out['budget_status']=='UNASSESSED'
    assert out['missing'] and out['combined_upper_bound_exact_fraction'] is None


def test_missing_omission_is_not_default_zero():
    out=assess(result={})
    assert 'omission_upper_bound' in out['missing']
    assert out['budget_status']=='UNASSESSED'


def test_explicit_zero_model_bound_is_not_missing():
    out=assess(model_error_upper_bound=0.,model_omission_allocation=.125)
    assert not out['missing']
    assert out['budget_status']=='CONDITIONALLY_WITHIN_DECLARED_MODEL_OMISSION_ALLOCATION'


def test_formed_reference_cannot_be_added_to_mathematical_b_omission():
    out=assess(model_b_reference='toy formed B packet')
    assert out['incompatible']==['B_reference_mismatch']
    assert out['budget_status']=='UNASSESSED'
    assert out['combined_upper_bound_exact_fraction'] is None


def test_sector_mismatch_is_unassessed():
    out=assess(model_sector=(0,1))
    assert out['incompatible']==['sector_mismatch']
    assert out['budget_status']=='UNASSESSED'


def test_large_bound_is_inconclusive_not_actual_error_failure():
    out=assess(model_omission_allocation=.125)
    assert out['budget_status']=='BOUND_INCONCLUSIVE'
    assert out['residual_randomization_needed'] is None


def test_exact_binary64_sum_does_not_round_down_to_fit_budget():
    result=toy();result['energy_bound_reference_H_H_vs_C_H']['upper_bound']=.2
    out=assess(result,model_error_upper_bound=.1,model_omission_allocation=.3)
    assert out['budget_status']=='BOUND_INCONCLUSIVE'


@pytest.mark.parametrize('bound,allocation',[(2**54+1,2**54),(Fraction(1,7),float(Fraction(1,7)))])
def test_exact_input_bound_is_not_first_rounded_down_to_float(bound,allocation):
    result=toy();result['energy_bound_reference_H_H_vs_C_H']['upper_bound']=0.
    out=assess(result,model_error_upper_bound=bound,model_omission_allocation=allocation)
    assert out['budget_status']=='BOUND_INCONCLUSIVE'


@pytest.mark.parametrize('key,value',[
    ('model_error_upper_bound',-1.),('model_error_upper_bound',float('nan')),
    ('model_error_upper_bound',float('inf')),('model_error_upper_bound',True),
    ('model_omission_allocation',0.),('model_omission_allocation',-1.)])
def test_invalid_explicit_bounds_rejected(key,value):
    with pytest.raises(ValueError):assess(**{key:value})


def test_pure_no_file_io_and_does_not_mutate_supplied_result(monkeypatch):
    result=toy();before=deepcopy(result)
    def forbidden(*args,**kwargs):raise AssertionError('No result or input I/O allowed')
    monkeypatch.setattr('builtins.open',forbidden)
    assert assess(result)['budget_status']=='CONDITIONALLY_WITHIN_DECLARED_MODEL_OMISSION_ALLOCATION'
    assert result==before
