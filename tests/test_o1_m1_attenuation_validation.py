"""Invented arrays/scalars and mocked I/O only; no saved H4 data is decoded."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from test_o1_m1_matched_split import PACKAGE, adapter, toy_inputs

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_o1_m1_attenuation_validation as runner
import run_o1_m1_h4_operator_correctness as reference
bridge=importlib.import_module(PACKAGE+'.o1_m1_resource_bridge')
signals=importlib.import_module(PACKAGE+'.o1_m1_circuit_signal_validation')
rte=importlib.import_module(PACKAGE+'.rte')


def forbidden(*args,**kwargs):
    raise AssertionError('No saved-input decode, shot, price, compile or sampling in preparation tests')


@pytest.fixture(autouse=True)
def no_saved_science_or_pricing(monkeypatch):
    import qiskit
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(qiskit,'transpile',forbidden)
    monkeypatch.setattr(rte,'iter_sample_rte_events',forbidden)
    monkeypatch.setattr(rte,'sample_rte_events',forbidden)
    monkeypatch.setattr(bridge,'account_radius_round',forbidden)
    monkeypatch.setattr(bridge,'MatchedNoPrepCostProvider',forbidden)
    monkeypatch.setattr(bridge,'transpile_and_measure_cost',forbidden)


def test_existing_toy_validator_composes_r4_r8_at_q64_without_price_or_samples():
    split=adapter.prepare_matched_a(toy_inputs())
    preparations={name:bridge.prepare_matched_resources(split,dictionary=name) for name in ('P','DF')}
    outputs=[]
    for r in (4,8):
        payload=signals.validate_matched_circuit_signals(split,reference,
            step_times=(.05,),r=r,K_values=(2,),q=64)
        assert payload['status']=='PASS'
        assert all(c['passed'] for c in payload['checks'] if 'relative_phase' in c['name'])
        laws={name:bridge.configure_law(p,delta_time=.05,r=r,K=2) for name,p in preparations.items()}
        row=runner.assess(payload,r=r,preparations=preparations,laws=laws,bridge=bridge)
        for name in ('P','DF'):
            record=payload['records'][0]['dictionaries'][name]
            config=laws[name][0]
            # Independently account for q*r repeated normalization factors.
            assert record['attenuation']==pytest.approx(config.distribution_normalization**(-64*r),rel=2e-14)
            assert config.dimensionless_step_time==pytest.approx(preparations[name].tail.lambda_r*.05/r)
            assert record['raw_signal']['radius']*record['normalization']==pytest.approx(record['corrected_signal']['radius'])
            # This deliberately mismatched artificial original H is rejected;
            # a validator PASS is not automatically a physical-budget PASS.
            assert 'model_phase_budget' in row['arms'][name]['reasons']
        outputs.append((payload,row,laws))
    assert outputs[1][0]['tolerance']['implementation']>outputs[0][0]['tolerance']['implementation']
    assert outputs[1][1]['numerical_allowances']['radius_allowance']>outputs[0][1]['numerical_allowances']['radius_allowance']
    for name in ('P','DF'):
        assert outputs[1][2][name][0].dimensionless_step_time==pytest.approx(outputs[0][2][name][0].dimensionless_step_time/2)


def mock_payload(r):
    return dict(status='PASS',checks=[dict(passed=True)],
        tolerance=dict(operator=1e-16,basis=1e-16,implementation=1e-16),
        conditions=dict(q=64,r=r,step_times=(.05,),K_values=(2,),
            identity_policy='faithful_identity_in_tail',original_state_field='state',
            sector_projection=False,full_dimension=256),
        records=[dict(q=64,r=r,K=2,delta=.05,PF_signal=dict(radius=.99),
            dictionaries={name:dict(attenuation=.8,normalization=1.25) for name in ('P','DF')})])


def good_bounds():
    return dict(reference_radius_lower_bound=.99,corrected_signal_error_bound=.001,
        pf_phase_bound=.001,model_phase_bound=.001,implementation_phase_bound=.001,
        radius_provenance='invented scalar fixture')


@pytest.mark.parametrize('changes,reason',[
    (dict(pf_phase_bound=.061),'PF_phase_budget'),
    (dict(model_phase_bound=.031),'model_phase_budget'),
    (dict(implementation_phase_bound=.011),'implementation_phase_budget'),
    (dict(corrected_signal_error_bound=.04),'RTE_phase_budget'),
    (dict(corrected_signal_error_bound=.99),'nonpositive_radius'),
])
def test_fixed_phase_and_radius_failures_are_not_shot_calculations(changes,reason):
    bounds=good_bounds()|changes
    fake=SimpleNamespace(observed_signal_bounds=lambda *a,**k:bounds,account_radius_round=forbidden)
    row=runner.assess(mock_payload(4),r=4,preparations={n:None for n in ('P','DF')},
        laws={n:(None,None) for n in ('P','DF')},bridge=fake)
    assert all(not arm['feasible'] and reason in arm['reasons'] for arm in row['arms'].values())


def test_fixed_scope_and_allowance_domain_are_rejected_before_bounds():
    payload=mock_payload(4);payload['conditions']['q']=32
    options=dict(r=4,preparations={},laws={},bridge=SimpleNamespace(observed_signal_bounds=forbidden))
    with pytest.raises(ValueError,match='conditions'):
        runner.assess(payload,**options)
    payload=mock_payload(4);payload['tolerance']['implementation']=1.
    with pytest.raises(ValueError,match='domain'):
        runner.assess(payload,**options)


@pytest.mark.parametrize('fail_first',[False,True])
def test_mocked_runner_has_only_two_fixed_validator_calls_and_stops(monkeypatch,tmp_path,fail_first):
    source=tmp_path/'invented.bin';source.write_bytes(b'invented non-molecular test data')
    monkeypatch.setattr(runner,'INPUT_SHA',hashlib.sha256(source.read_bytes()).hexdigest())
    calls=[]
    inputs=SimpleNamespace(num_qubits=8,df_lambdas=tuple(range(10)))
    fake_adapter=SimpleNamespace(load_h4_a_snapshot=lambda p:inputs,prepare_matched_a=lambda data:data)
    def validate(split,ref,**kwargs):
        calls.append(kwargs);payload=mock_payload(kwargs['r'])
        if fail_first:payload['status']='FAIL'
        return payload
    fake_signals=SimpleNamespace(validate_matched_circuit_signals=validate)
    fake_bridge=SimpleNamespace(prepare_matched_resources=lambda split,**k:k['dictionary'],
        configure_law=lambda *a,**k:(SimpleNamespace(r=k['r']),None),
        observed_signal_bounds=lambda *a,**k:good_bounds(),
        account_radius_round=forbidden,MatchedNoPrepCostProvider=forbidden)
    monkeypatch.setattr(runner,'modules',lambda:(fake_adapter,fake_signals,fake_bridge,object()))
    output=tmp_path/'new_result'
    result=runner.run(source,output)
    assert result['mandatory_stop'] is True
    assert [c['r'] for c in calls]==([4] if fail_first else [4,8])
    assert all(c==dict(step_times=(.05,),r=c['r'],K_values=(2,),q=64) for c in calls)
    assert result['status']==('LIMITED_PROBE_NOT_QUALIFIED' if fail_first else 'LIMITED_SIGNAL_PROBE_COMPLETE')
    assert result['c0_control']['use']=='quotation only; no saved-result load/reanalysis'
    with pytest.raises(FileExistsError):runner.run(source,output)


def test_wrong_fixed_input_is_rejected_before_scientific_import(monkeypatch,tmp_path):
    p=tmp_path/'invented.bin';p.write_bytes(b'not the fixed snapshot')
    monkeypatch.setattr(runner,'modules',forbidden)
    with pytest.raises(ValueError,match='input hash'):
        runner.run(p,tmp_path/'new_result')


def test_unexpected_error_stops_only_own_fresh_result_and_preserves_it_on_reentry(monkeypatch,tmp_path):
    p=tmp_path/'invented.bin';p.write_bytes(b'invented non-molecular input')
    monkeypatch.setattr(runner,'INPUT_SHA',hashlib.sha256(p.read_bytes()).hexdigest())
    def failure():
        raise RuntimeError('invented runtime failure before any input decode')
    monkeypatch.setattr(runner,'modules',failure)
    output=tmp_path/'new_result'
    with pytest.raises(RuntimeError,match='invented runtime failure'):
        runner.run(p,output)
    checkpoint=(output/'result.json').read_bytes()
    value=json.loads(checkpoint)
    assert value['status']=='ERROR_STOP' and value['mandatory_stop'] is True
    assert value['error_type']=='RuntimeError' and value['rows']==[]
    with pytest.raises(FileExistsError):
        runner.run(p,output)
    assert (output/'result.json').read_bytes()==checkpoint
