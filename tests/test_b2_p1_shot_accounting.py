"""Artificial bounds/integer ledgers only; no saved physical inputs or prices."""
from types import SimpleNamespace
import csv
import json

import pytest

from trotterlib import b2_p1_shot_accounting as m
from trotterlib.o1_m1_resource_bridge import radius_phase_diagnostics, _hoeffding_shots
from trotterlib.rte import finite_rte_attenuation
from trotterlib.rpe_resource_accounting import RPERoundSpecification


@pytest.fixture(autouse=True)
def no_scientific_inputs_decoder_or_circuits(monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('No saved inputs, simulation, branch, circuits or price in synthetic checks')
    monkeypatch.setattr(m.saved,'git',forbidden)
    monkeypatch.setattr(m.saved,'reconstruct_rpe_phase',forbidden)
    from qiskit import QuantumCircuit
    monkeypatch.setattr(QuantumCircuit,'__init__',forbidden)


def toy_round(index=0):
    config=SimpleNamespace(evolution_time=.02,rte_steps=4,distribution_normalization=1.001)
    bounds=dict(reference_radius_lower_bound=.8,corrected_signal_error_bound=.001,
        pf_phase_bound=.01,model_phase_bound=.001,implementation_phase_bound=.001,
        radius_provenance='artificial PF radius; not raw')
    diagnostic=radius_phase_diagnostics(specification=RPERoundSpecification(index,.02),
        allocation=m.ALLOCATION,beta_rpe=.4,config=config,
        **{k:v for k,v in bounds.items() if k!='radius_provenance'})
    row=dict(m=index,Q=1<<index,T_Ha_inverse=(1<<index)*.02,
        normalization=1/diagnostic['attenuation'],attenuation=diagnostic['attenuation'],
        qualification=dict(diagnostic,phase_total_with_reserved_statistical=.362))
    return row,bounds,config


def test_actual_shot_API_uses_corrected_radius_once_and_no_price():
    row,bounds,config=toy_round(2)
    result=m.account_saved_round(row,bounds,config,context_hash='toy_context',arm_hash='toy_arm')
    A=finite_rte_attenuation(config,tail_evolutions=4)
    assert result['raw_radius_lower']==A*(.8-.001)
    expected=_hoeffding_shots(attenuation=A,reference_radius=.8,signal_error_bound=.001,
        beta_stat_budget=.24,alpha_axis=.05/36)
    assert result['N_X']==result['N_Y']==expected
    assert type(result['N_X']) is int and result['N_round']==2*expected
    assert result['price'] is None and result['total_circuit_cost'] is None
    assert result['guarantee_status']=='not_certified'


def test_nonpositive_radius_and_budget_failure_rejected():
    row,bounds,config=toy_round()
    bounds['corrected_signal_error_bound']=.9
    with pytest.raises(ValueError,match='ROUND_SHOT_INELIGIBLE'):
        m.account_saved_round(row,bounds,config,context_hash='toy_context',arm_hash='toy_arm')
    row,bounds,config=toy_round();bounds['pf_phase_bound']=.061
    with pytest.raises(ValueError,match='PF_phase_budget'):
        m.account_saved_round(row,bounds,config,context_hash='toy_context',arm_hash='toy_arm')


def integer_rows():
    # Deliberately beyond both float's exact integer range and default Decimal
    # precision. Counts must remain exact through aggregation/serialization.
    return [dict(m=i,N_X=10**40+i,N_Y=10**40+2*i,N_round=2*10**40+3*i,
                 alpha_X=.05/36,alpha_Y=.05/36,status='CONDITIONAL_SUFFICIENT_SHOTS') for i in range(18)]


def test_huge_integer_totals_and_serialized_independent_sum(tmp_path):
    rows=integer_rows();total=m.aggregate_shots(rows)
    assert total['N_all_rounds']==36*10**40+3*sum(range(18))
    assert total['exact_failure_budget_sum']=='1/20'
    (tmp_path/'result.json').write_text(json.dumps(dict(rounds=rows,totals=total)))
    with (tmp_path/'rounds.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    verified=m.verify_serialized_totals(tmp_path,total)
    assert verified['JSON_CSV_integer_match']
    assert int(verified['CSV_all_sum'])==total['N_all_rounds']


@pytest.mark.parametrize('mutation',['duplicate','missing','total','float','alpha'])
def test_aggregate_refuses_inconsistent_counts_or_inventory(mutation):
    rows=integer_rows()
    if mutation=='duplicate':rows[3]['m']=2
    elif mutation=='missing':rows.pop()
    elif mutation=='total':rows[0]['N_round']+=1
    elif mutation=='float':rows[0]['N_X']=float(rows[0]['N_X'])
    else:rows[0]['alpha_X']=.01
    with pytest.raises(ValueError):
        m.aggregate_shots(rows)
