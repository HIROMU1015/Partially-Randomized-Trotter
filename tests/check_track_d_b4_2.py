"""B4-2 wiring tests: artificial data only, no saved H4 input evaluation."""
import importlib.util
import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('b4_2_runner', ROOT/'scripts/run_track_d_b4_2.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_input_hash_rejects_wrong_binding_before_decode():
    data = b'{"artificial": true}'
    records = []
    with patch.object(runner, 'git', return_value=data):
        with pytest.raises(ValueError, match='INPUT_BINDING'):
            runner.read_fixed('TOY', 'TOY.json', records, 'WRONG')
        assert records == []
        assert runner.read_fixed('TOY', 'TOY.json', records, runner.sha(data)) == {'artificial': True}


def test_synthetic_operator_rows_flow_to_existing_adapter_without_science_calls():
    from trottertracks.track_d_b1_c1 import transport_round
    from trottertracks.c1_stage3_accuracy import _finite
    from trottertracks import c1_stage3_accuracy as old
    old._legacy('rpe_target_round_horizon_validation')
    old._legacy('rpe_resource_accounting')
    # The SAME fixed target is tested with invented operator/radius inputs.
    # No B1 saved molecular row or physical state is read by this test.
    rows = [transport_round(dict(attenuation=.9,pf_operator_error=.001,
                rte_corrected_operator_error=0.), m=m,df_error=0.,base_time=.1,
                historical_target=dict(beta_total=math.pi/6,beta_stat=math.pi/12),finite=_finite)
            for m in range(13)]
    toy = dict(rows=rows,norm_qualification='ARTIFICIAL_ZERO_DF_TOY_BOUND')
    physical = dict(status='PASS',energy_Ha=0.,residual=0.,normalization_error=0.,
                    alias_margin=math.pi-(1+math.pi/6),state='TOY_ONLY')
    with patch.object(runner,'read_fixed',side_effect=AssertionError('saved input read')), \
         patch('numpy.linalg.eigh',side_effect=AssertionError('ground solve')), \
         patch('numpy.random.default_rng',side_effect=AssertionError('sampling')):
        result = runner.evaluate(toy,physical,dict(ideal_law='ARTIFICIAL_IDEAL_IID'))
    assert result['status'] == 'CONDITIONAL_IDEAL_13_ROUND_PASS'
    assert result['total_shots'] == sum(r['N_X']+r['N_Y'] for r in result['rows'])
    assert result['actual_lane']['N'] is None
    assert result['Stage_C_handoff']['total_circuit_resources'] is None
    assert not result['decoder']['success_probability_evaluated']
    assert result['energy']['df_counting'] == 'already_in_signal_once'


def test_default_cli_does_not_execute_or_make_science_output():
    import subprocess,sys
    with patch.object(runner,'prepare',side_effect=AssertionError('preflight unexpectedly called')):
        result = subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_track_d_b4_2.py')],
                                capture_output=True,text=True)
    assert result.returncode == 2
    assert 'explicit --execute required' in result.stderr
