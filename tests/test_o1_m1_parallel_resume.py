"""Continuation inventory and duplicate prevention, invented data only."""
from copy import deepcopy
from dataclasses import asdict,replace
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
resume=importlib.import_module('run_o1_m1_c0_parallel_resume')


def invented_prior(count=16):
    rows=[]
    for m in range(7):
        for arm in resume.ARMS:
            rows.append(dict(round_index=m,arm=arm,account=dict(feasible=True,round_total_cost=1.),
                             CX_relative_sample_SE=dict(cosine=.02,sine=.02)))
    return dict(status='RUNNING',configuration=deepcopy(resume.C0),provenance=dict(base=resume.serial.BASE),
        signal_rounds=[dict(round_index=m,q=1<<m,arms={a:dict(feasible=True) for a in ('P','DF')}) for m in range(7)],
        baseline_rounds=[dict(account=dict(feasible=True)) for _ in range(7)],priced_rounds=rows[:count])


@pytest.mark.parametrize('completed',[15,16])
def test_latest_inventory_reuses_finished_points_without_mutation(completed):
    prior=invented_prior(completed);before=deepcopy(prior)
    carried,pending=resume.validate_prior(prior)
    all_points=[(m,arm) for m in range(7) for arm in resume.ARMS]
    assert list(carried)==all_points[:completed] and pending==all_points[completed:]
    assert prior==before


@pytest.mark.parametrize('bad',['conditions','duplicate','stopped','SE','signals'])
def test_incompatible_or_failed_prior_cannot_be_reopened(bad):
    prior=invented_prior()
    if bad=='conditions': prior['configuration']['r']=3
    elif bad=='duplicate': prior['priced_rounds'].append(prior['priced_rounds'][0])
    elif bad=='stopped': prior['mandatory_stop']=True
    elif bad=='SE': prior['priced_rounds'][0]['CX_relative_sample_SE']['cosine']=.11
    else: prior['signal_rounds'].pop()
    with pytest.raises(ValueError): resume.validate_prior(prior)


def test_live_previous_worker_rejects_before_input_read(tmp_path):
    with pytest.raises(RuntimeError,match='still exists'):
        resume.run(tmp_path/'no_input',tmp_path/'no_prior',tmp_path/'new',previous_pid=os.getpid())
    assert not (tmp_path/'new').exists()


def test_dead_worker_check_has_no_stop_action(monkeypatch):
    calls=[]
    def nonexistent(pid,signal):
        calls.append((pid,signal));raise ProcessLookupError
    monkeypatch.setattr(os,'kill',nonexistent)
    resume.require_stopped(123456)
    assert calls==[(123456,0)]


def test_saved_baseline_shots_reproduced_and_drift_rejected():
    from test_o1_m1_parallel_price import bridge,prepared
    accounting=importlib.import_module('trotterlib.rpe_resource_accounting')
    p,_,_=prepared('DF')
    bounds=dict(reference_radius_lower_bound=.9,corrected_signal_error_bound=0.,pf_phase_bound=.01,
                model_phase_bound=.002,implementation_phase_bound=.0001,radius_provenance='invented')
    old=bridge.account_radius_round(context_hash=p.context_hash,arm_hash=resume.STRONG_ARM,
        specification=accounting.RPERoundSpecification(2,.05),
        allocation=accounting.RPEErrorAllocation(.06,.03,.15,.05/14,.05/14),beta_rpe=.30,
        cost_metric='cx_count',config=None,**bounds)
    prior=dict(baseline_rounds=[dict(account=old) for _ in range(7)])
    assert resume.saved_bounds(prior,None,2,'strong_DF_PF',p,None,bridge,accounting)==bounds
    prior=deepcopy(prior);prior['baseline_rounds'][2]['account']['shots']['cosine']+=1
    with pytest.raises(ValueError,match='shot conditions'):
        resume.saved_bounds(prior,None,2,'strong_DF_PF',p,None,bridge,accounting)


def artificial_cohort():
    """Invented qualified records; not numerical evidence about the toy model."""
    from test_o1_m1_parallel_price import bridge,prepared
    from types import SimpleNamespace
    accounting=importlib.import_module('trotterlib.rpe_resource_accounting')
    preps={a:prepared(a)[0] for a in ('P','DF')}
    laws={a:bridge.configure_law(p,delta_time=.05,r=2,K=2) for a,p in preps.items()}
    compiler=importlib.import_module('trotterlib.rte').CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,7,'1.3.0')
    provider=bridge.MatchedNoPrepCostProvider(compiler)
    prior=invented_prior(0);payloads={};all_rows=[]
    allocation=accounting.RPEErrorAllocation(.06,.03,.15,.05/14,.05/14)
    for m in range(7):
        margins=dict(radius_allowance=1e-6,complex_signal_allowance=2e-6,phase_allowance=1e-6)
        record=dict(q=1<<m,delta=.05,r=2,K=2,PF_signal=dict(radius=.9),
            PF_signal_error_vs_saved_DF_exact=dict(wrapped_phase_absolute=.01),
            original_vs_DF_exact_signal_difference=dict(wrapped_phase_absolute=.001),dictionaries={})
        for name in ('P','DF'):
            config=laws[name][0];attenuation=bridge.finite_rte_attenuation(config,tail_evolutions=1<<m)
            record['dictionaries'][name]=dict(lambda_r=preps[name].tail.lambda_r,attenuation=attenuation,
                normalization=1/attenuation,implementation_signal_error=dict(wrapped_phase_absolute=.0001),
                finite_RTE_signal_error_vs_PF=dict(complex_absolute=.001))
        payloads[m]=dict(status='PASS',records=[record])
        signal_row=dict(round_index=m,q=1<<m,numerical_allowances=margins,arms={})
        for name in resume.ARMS:
            p=preps['DF' if name=='strong_DF_PF' else name]
            config=None if name=='strong_DF_PF' else laws[name][0]
            arm=resume.STRONG_ARM if name=='strong_DF_PF' else p.preparation_hash
            bounds=(dict(reference_radius_lower_bound=.9,corrected_signal_error_bound=0.,pf_phase_bound=.01,
                model_phase_bound=.001,implementation_phase_bound=.0001,radius_provenance='invented')
                if config is None else bridge.observed_signal_bounds(p,config,accounting.RPERoundSpecification(m,.05),record,**margins))
            kwargs=dict(context_hash=p.context_hash,arm_hash=arm,specification=accounting.RPERoundSpecification(m,.05),
                allocation=allocation,beta_rpe=.30,cost_metric='cx_count',config=config,**bounds)
            bare=bridge.account_radius_round(**kwargs)
            if config is None: prior['baseline_rounds'][m]=dict(account=bare)
            else: signal_row['arms'][name]=bare
            accumulators={a:bridge.CompiledMetricAccumulator(weighted=True) for a in ('cosine','sine')}
            for acc in accumulators.values(): acc.update(SimpleNamespace(**{k:100. for k in bridge.RPE_COST_METRICS}),weight=1.)
            cost=provider._result(accumulators,context_hash=p.context_hash,arm_hash=arm,q=1<<m,
                delta_time=.05,law_hash=bridge._law_hash(config),method='invented_prior_price')
            all_rows.append(dict(round_index=m,arm=name,cost=asdict(cost),
                account=bridge.account_radius_round(**kwargs,cost=cost),CX_relative_sample_SE=dict(cosine=0.,sine=0.)))
        prior['signal_rounds'][m]=signal_row
    return prior,payloads,all_rows,preps,bridge


@pytest.mark.parametrize('remaining',['P','DF','strong_DF_PF','failed_P'])
def test_full_continuation_only_calls_missing_points_without_signal_replay(tmp_path,monkeypatch,remaining):
    prior,payloads,all_rows,preps,bridge=artificial_cohort()
    missing=[(0,'P'),(0,'DF')] if remaining=='failed_P' else [(0,remaining)]
    prior['priced_rounds']=[row for row in all_rows if (row['round_index'],row['arm']) not in missing]
    prior=json.loads(json.dumps(prior))  # actual checkpoints contain JSON lists, not Python tuples
    before=deepcopy(prior['priced_rounds'])
    previous=tmp_path/'previous';previous.mkdir()
    for m,payload in payloads.items(): (previous/f'signal_m{m}.json').write_text(json.dumps(payload))
    old=previous/'result.json';old.write_text(json.dumps(prior))
    input_path=tmp_path/'invented_input';input_path.write_bytes(b'not a molecular snapshot')
    monkeypatch.setattr(resume.serial,'INPUT_SHA',hashlib.sha256(input_path.read_bytes()).hexdigest())
    adapter=importlib.import_module('trotterlib.o1_m1_matched_split')
    monkeypatch.setattr(adapter,'load_h4_a_snapshot',lambda path:preps['P'].split.inputs)
    monkeypatch.setattr(resume,'require_stopped',lambda pid:None)
    signals=importlib.import_module('trotterlib.o1_m1_circuit_signal_validation')
    monkeypatch.setattr(signals,'validate_matched_circuit_signals',lambda *a,**k:pytest.fail('No signal rerun'))
    from qiskit import QuantumCircuit
    def baseline(split,**kwargs):
        assert kwargs==dict(delta_time=.05,q=1,substeps=4,basis_policy='exact_spin_lift_spatial_eigh_v1')
        return QuantumCircuit(3)
    monkeypatch.setattr(bridge,'build_strong_df_baseline',baseline)
    calls=[]
    class FakeProvider:
        def __init__(self,compiler,**kwargs):
            self.compiler=compiler;self.last_execution=dict(task_count=1,worker_pids=[123])
            from trotterlib.rte import CircuitCost
            self.serial=bridge.MatchedNoPrepCostProvider(compiler,compile_function=lambda circuit,settings,**k:
                CircuitCost(100,100,100,100,100,100,settings,5,'empirical_compiled_estimate'))
        def monte_carlo(self,p,c,d,**kwargs):
            calls.append((p.dictionary,kwargs));cost=self.serial.monte_carlo(p,c,d,**kwargs)
            if remaining=='failed_P':
                cost=replace(cost,cosine_standard_error=replace(cost.cosine_standard_error,cx_count=20.))
            return cost
        def deterministic(self,circuit,**kwargs):
            calls.append(('strong_DF_PF',kwargs));return self.serial.deterministic(circuit,**kwargs)
    monkeypatch.setattr(resume,'ParallelMatchedNoPrepCostProvider',FakeProvider)
    output=tmp_path/'new'
    result=resume.run(input_path,old,output,previous_pid=123,max_workers=2)
    assert result['priced_rounds'][:len(before)]==before
    assert len(calls)==1 and calls[0][0]==('P' if remaining=='failed_P' else remaining)
    if remaining!='strong_DF_PF':
        assert calls[0][1]==dict(q=1,sample_count=32,seed=2026110600+(100 if remaining=='DF' else 0))
    assert result['mandatory_stop']
    assert result['status']==('C0_NOT_QUALIFIED' if remaining=='failed_P' else 'C0_CONDITIONAL_PILOT_COMPLETE')
    assert json.loads(old.read_text())==prior  # historical raw was not changed
