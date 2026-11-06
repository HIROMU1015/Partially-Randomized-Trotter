"""Only invented scalar records; never load molecular input."""
import importlib.util
from pathlib import Path
import math
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/run_o1_m1_c0_pilot.py'
spec=importlib.util.spec_from_file_location('c0_driver_synthetic',path)
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_C0_allowance_is_fixed_program_scale_with_no_discrepancy_fit():
    payload=dict(tolerance=dict(operator=1e-8,basis=2e-8,implementation=3e-8),
                 conditions=dict(full_dimension=4))
    record=dict(PF_signal=dict(radius=.7))
    value=driver.c0_margins(payload,record)
    eta=4*3e-8
    assert value['radius_allowance']==eta
    assert value['complex_signal_allowance']==2*eta
    assert value['phase_allowance']==math.asin(2*eta/(.7-eta))
    record['unrelated_observed_error']=1e8
    assert driver.c0_margins(payload,record)==value


@pytest.mark.parametrize('radius',[0.,1e-8,1e-7])
def test_C0_invalid_radius_phase_domain_fails_without_clamping(radius):
    payload=dict(tolerance=dict(operator=1e-8,basis=2e-8,implementation=3e-8),
                 conditions=dict(full_dimension=4))
    with pytest.raises(ValueError,match='domain'):
        driver.c0_margins(payload,dict(PF_signal=dict(radius=radius)))


def test_registered_conditions_are_the_fixed_C0_scope():
    assert driver.Q==(1,2,4,8,16,32,64)
    assert (driver.DELTA,driver.R,driver.K)==(.05,2,2)
    assert (driver.MODEL_BUDGET,driver.IMPLEMENTATION_BUDGET)==(.03,.01)
