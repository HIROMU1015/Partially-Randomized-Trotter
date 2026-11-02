"""Artificial production-call, RNG/circuit equivalence and corruption tests."""
from dataclasses import asdict
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Operator

from trotterlib.fixed_signal_evaluation import FixedSignalConfig,evaluate_fixed_signal,prepare_signal_adapter,default_compiler
from trotterlib.h4_audited_evaluation import evaluate_audited_signal
from trotterlib.h4_pilot_science import artificial_inputs
from trotterlib.rte_event_audit import TrajectoryAudit,audited_adapter,verify_join
from trotterlib.isolated_qiskit_execution import digest


@pytest.fixture(scope='module')
def models():return artificial_inputs()


@pytest.mark.parametrize('rep,method',[(r,m) for r in ['pauli_jw','df_native'] for m in ['PF','PR']])
def test_actual_production_events_no_extra_draw_and_circuit_equal(models,rep,method,monkeypatch,tmp_path):
    import trotterlib.rte as rte
    from trotterlib.rte_compiled_cost import canonical_qiskit_circuit_fingerprint
    p,d,state,receipt=models;h=p if rep=='pauli_jw' else d
    c=FixedSignalConfig(rep,method=method,T=.2,q=2,r=2,K=4,
        L_D=receipt['prefixes'][rep]['L_D'] if method=='PR' else None,cost_samples=1)
    original=rte.iter_sample_rte_events;calls=[];returned=[]
    def observed(*a,**kw):
        calls.append(dict(kw))
        for e in original(*a,**kw):returned.append(e);yield e
    monkeypatch.setattr(rte,'iter_sample_rte_events',observed)
    old=prepare_signal_adapter(h,c)
    kwargs=dict(q=c.q,seed=1701)
    if rep=='df_native':kwargs['controlled']=True
    baseline=old.trajectory(**kwargs);baseline_calls=list(calls);baseline_objects=list(returned)
    calls.clear();returned.clear()
    audit=TrajectoryAudit(rep+'_'+method,dict(SOURCE='ARTIFICIAL'),directory=tmp_path)
    audit.begin(1701,0);new=audited_adapter(h,c,audit);actual=new.trajectory(**kwargs)
    assert calls==baseline_calls
    assert [e.to_dict() for e in returned]==[e.to_dict() for e in baseline_objects]
    assert [e.to_dict() for ss in audit.objects for e in ss]==[e.to_dict() for e in returned]
    assert canonical_qiskit_circuit_fingerprint(actual)==canonical_qiskit_circuit_fingerprint(baseline)
    assert float(actual.global_phase)==float(baseline.global_phase)
    assert audit.current['event_count']==(c.q*c.r if method=='PR' else 0)
    path=tmp_path/audit.current['trajectory_id']/'events.json';assert path.exists()
    assert not (path.parent/'price_join.json').exists()
    row=audit.join({'cosine':17,'sine':19},default_compiler(),{'cosine':'a','sine':'b'})
    assert verify_join(row)
    assert row['compiled_RZ']=={'cosine':17,'sine':19}
    assert not row['event_costs_additive']


@pytest.mark.parametrize('rep',['pauli_jw','df_native'])
def test_audit_on_off_compiled_prices_and_signals_equal(models,rep,tmp_path):
    from trotterlib.h4_execution_contract import completed_contract
    p,d,state,receipt=models;h=p if rep=='pauli_jw' else d
    c=FixedSignalConfig(rep,method='PR',T=.1,q=1,r=2,K=4,L_D=receipt['prefixes'][rep]['L_D'],cost_samples=2)
    kw=dict(epsilon_sig=.05,numerical_allowance={'cosine':1e-9,'sine':1e-9},
        evidence_kind='EMPIRICAL',evidence_ref='ARTIFICIAL',log_B_margin=1e-12,verify_compiled=True)
    old=evaluate_fixed_signal(h,state,c,**kw)
    audit=TrajectoryAudit(rep+'_PR',dict(SOURCE='ARTIFICIAL'),directory=tmp_path)
    new=evaluate_audited_signal(h,state,c,**kw,event_audit=audit)
    off=evaluate_audited_signal(h,state,c,**kw)
    for k in ['reference','signal','normalization','shots','total_RZ','compiler']:
        assert old[k]==new[k]==off[k]
    assert [(r['seed'],r['RZ']) for r in old['circuit_cost']['samples']]==[(r['seed'],r['RZ']) for r in new['circuit_cost']['samples']]
    assert old['circuit_cost']['axes']==new['circuit_cost']['axes']==off['circuit_cost']['axes']
    assert all(verify_join(r) for r in audit.rows)
    assert new['shots']['usable'] and new['total_RZ']['mean']>0


@pytest.mark.parametrize('order',[0,2,4])
@pytest.mark.parametrize('rep',['pauli_jw','df_native'])
def test_forced_event_serialization_preserves_application_and_phase(models,order,rep):
    from trotterlib.h2_non_df_pilot_validation import event_context
    from trotterlib.rte import _make_event,event_unitary
    from trotterlib.pauli_signal_adapter import append_pauli_event
    from qiskit import QuantumCircuit
    p,d,state,receipt=models;h=p if rep=='pauli_jw' else d
    c=FixedSignalConfig(rep,method='PR',T=.1,q=1,r=2,K=4,L_D=receipt['prefixes'][rep]['L_D'])
    a=audited_adapter(h,c);components,matrices=event_context(a)
    event=_make_event(tuple(i%len(components) for i in range(order+1)),components,a.distribution,order//2)
    audit=TrajectoryAudit(rep+'_PR',{'SOURCE':'ARTIFICIAL_FORCED_NOT_MC'});audit.begin(17,0)
    audit.observe(a,[(event,event)],[17]);r=audit.join({'cosine':1,'sine':1},default_compiler(),{})
    e=r['events'][0]
    assert e['taylor_order']==order and tuple(e['selected_component_ids'])==event.selected_component_ids
    assert e['phase']=={'real':float(event.phase.real),'imag':float(event.phase.imag)}
    assert [s['component_id'] for s in e['application_sequence']]==list(event.selected_component_ids)
    if rep=='pauli_jw':
        circuit=QuantumCircuit(8);append_pauli_event(circuit,event,a.labels)
    else:
        from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
        prep=a.preparation.rte_preparation
        circuit=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.basis_registry).build_event(prep.request_for_event(event)).circuit
    assert np.linalg.norm(Operator(circuit).data-event_unitary(event,matrices),2)<1e-9


@pytest.mark.parametrize('mutation',['missing_event','duplicate_position','price','event_phase','trajectory_id','binding'])
def test_corrupt_or_missing_join_rejected(models,mutation):
    p,d,state,receipt=models;c=FixedSignalConfig('pauli_jw',method='PR',q=1,r=2,K=4,L_D=2)
    audit=TrajectoryAudit('pauli_jw_PR',{'SOURCE':'ARTIFICIAL'});audit.begin(17,0)
    a=audited_adapter(p,c,audit);a.trajectory(q=1,seed=17)
    r=audit.join({'cosine':3,'sine':4},default_compiler(),{})
    if mutation=='missing_event':r['events'].pop()
    if mutation=='duplicate_position':r['events'][1]['event_position']=0
    if mutation=='price':r['compiled_RZ']['cosine']+=1
    if mutation=='event_phase':r['events'][0]['phase']['real']=42
    if mutation=='trajectory_id':r['trajectory_id']='bad'
    if mutation=='binding':r['binding']={'SOURCE':'OTHER'}
    with pytest.raises(ValueError):verify_join(r)


def test_duplicate_trajectory_and_unjoined_record_refused(models):
    audit=TrajectoryAudit('x',{});audit.begin(1,0)
    with pytest.raises(ValueError,match='PREVIOUS'):audit.begin(2,1)
    p,d,state,receipt=models;c=FixedSignalConfig('pauli_jw',method='PR',q=1,r=2,K=4,L_D=2)
    a=audited_adapter(p,c,audit);a.trajectory(q=1,seed=1)
    audit.join({'cosine':1,'sine':1},default_compiler(),{})
    with pytest.raises(ValueError,match='DUPLICATE'):audit.begin(1,0)


def test_trajectory_budget_fails_before_production_sampling():
    audit=TrajectoryAudit('x',{},max_trajectories=0)
    with pytest.raises(ValueError,match='BUDGET'):audit.begin(17,0)


def test_opt_in_evaluator_numeric_ast_matches_frozen_kernel():
    """Remove observation-only statements and compare every remaining AST node."""
    import trotterlib.h4_audited_evaluation as new
    old=ast.parse(inspect.getsource(evaluate_fixed_signal)).body[0]
    node=ast.parse(inspect.getsource(new.evaluate_audited_signal)).body[0]
    node.name=old.name
    node.args.kwonlyargs=node.args.kwonlyargs[:-1];node.args.kw_defaults=node.args.kw_defaults[:-1]
    class Strip(ast.NodeTransformer):
        def visit_If(self,n):
            if ast.unparse(n.test)=='event_audit is not None':return None
            return self.generic_visit(n)
        def visit_Assign(self,n):
            text=ast.unparse(n)
            if text=='fingerprints = {}' or text.startswith('fingerprints[axis] ='):return None
            return self.generic_visit(n)
        def visit_For(self,n):
            if ast.unparse(n.target)=='(trajectory_index, seed)':
                n.target=ast.Name(id='seed',ctx=ast.Store());n.iter=ast.Name(id='seeds',ctx=ast.Load())
            return self.generic_visit(n)
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id=='audited_adapter':
                n.func.id='prepare_signal_adapter';n.args=n.args[:2]
            return self.generic_visit(n)
    node=Strip().visit(node)
    assert ast.dump(node)==ast.dump(old)
