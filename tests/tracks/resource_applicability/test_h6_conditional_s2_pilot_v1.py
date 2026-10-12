"""Artificial scalar/2-system-qubit circuits and mocked workers only.

No H6 trajectory, NPZ array, scientific runner or Qiskit transpile is executed.
"""
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import Operator
from trottertracks.resource_applicability import h6_conditional_s2_sampling_v1 as s
from trottertracks.resource_applicability import h6_conditional_s2_contract_v1 as c
from trottertracks.resource_applicability import h6_conditional_s2_execution_v1 as e
from trottertracks.resource_applicability.ax2a_preparation import digest

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def prohibit_real_compile_and_array_input(monkeypatch):
    import qiskit
    import trotterlib.rte_compiled_cost as compiled
    def forbidden(*a,**k):raise AssertionError('PREPARATION_FORBIDS_TRANSPILE_AND_NPZ')
    monkeypatch.setattr(qiskit,'transpile',forbidden);monkeypatch.setattr(compiled,'transpile',forbidden)
    monkeypatch.setattr(np,'load',forbidden)


def manifest(root):
    carry=dict(known_calls={'compile':1700,'control_probe':360,'trajectory':834,'occurrence':3136,'primitive':1470,'reference_matvec':400},
        reservations={'compile':20,'control_probe':4,'trajectory':10,'occurrence':64},historical_unknown_actual_totals=None,
        reservations_are_not_actual_totals=True,budget_reset=False,failed_calls_refunded=False)
    base=dict(profile={'sha256':'a'*64},input_identity={},plan={'caps_proposed':{
        'compile':1720,'trajectory':844,'occurrence':6656,'control_probe':1024,'primitive':10000,'reference_matvec':20000,
        'output_bytes':2*2**30,'log_bytes':2**20}})
    prior=dict(carry=carry,finite_distribution={'orders':[0,2],'order_probabilities':[.8,.2]})
    return dict(schema=c.SCHEMA,kind=c.KIND,repository_root=str(root),source_commit='b'*40,source_hashes={},
        base_manifest=base,prior_binding=prior,cumulative_accounting=c.accounting(prior,base),
        roundoff_preparation_binding={},registrations=c.registrations(),target_cell=deepcopy(c.CELL),phase=c.PHASE,
        conditioning=dict(stratum='s2',scope='whole trajectory',R=16,K=2,
            count_law='Binomial(16,p2) conditional M>=1; uniform M-subset',
            component_law='original independent component probabilities',quantum_weights_changed=False),
        proposed_resources=c.proposed_resources(),observation_policy=c.observation_policy(),hash_reuse_policy=c.hash_policy(),
        exclusive_output=dict(repository_path=c.NAMESPACE+'launch_pilot_v1',absolute_path=str(root/c.NAMESPACE/'launch_pilot_v1')),
        one_shot_lock=c.LOCK,execution_plan_sealed=True,approved=False,runtime_authorization=False,allowed_cpus=[],
        retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False,H8_authorized=False,absolute_launch_command=None)


@pytest.mark.parametrize('R,p2',[(1,.2),(2,.2),(4,.01),(16,.001),(16,1e-18)])
def test_conditional_binomial_matches_direct_small_law(R,p2):
    d=s.conditional_binomial(R,p2)
    assert math.isclose(math.fsum(d['conditional_probabilities']),1.,abs_tol=2e-15)
    for m,p in zip(d['counts'],d['conditional_probabilities']):
        expected=math.comb(R,m)*p2**m*(1-p2)**(R-m)/d['s2_probability']
        assert math.isclose(p,expected,rel_tol=2e-13,abs_tol=1e-300)
    if R>1:assert d['conditional_probabilities'][1]>0


def test_order_vectors_have_the_exact_conditioned_joint_law():
    R=3;p=.2;mass=1-(1-p)**R
    probabilities=[s.conditional_order_vector_probability(list(v),p) for v in product((0,2),repeat=R) if 2 in v]
    assert math.isclose(sum(probabilities),1.,abs_tol=2e-15)
    assert math.isclose(s.conditional_order_vector_probability([0,2,2],p),(1-p)*p*p/mass)


def test_multiple_nonzero_positions_are_not_forced_to_one():
    class RNG:
        def choice(self,values,**kwargs):
            if isinstance(values,list):return 2
            assert kwargs==dict(size=2,replace=False);return [0,3]
    assert s.draw_order_vector(4,.2,RNG())==(2,0,0,2)


@pytest.mark.parametrize('R,p2',[(0,.2),(True,.2),(65,.2),(2,0.),(2,1.),(2,-.1),(2,float('nan')),(2,float('inf'))])
def test_invalid_law_stops(R,p2):
    with pytest.raises(ValueError):s.conditional_binomial(R,p2)


def toy_preparation():
    from trotterlib.df_hamiltonian import DFHamiltonian
    from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
    from trotterlib.df_partial_s2 import prepare_df_partial_s2
    h=DFHamiltonian(constant=.13,one_body=np.asarray([[.2,.03],[.03,-.1]],dtype=complex),
        lambdas=np.asarray([.2,-.3]),g_matrices=(np.diag([1.,.4]).astype(complex),np.diag([.2,2.]).astype(complex)),
        metadata={'name':'CONDITIONAL_ARTIFICIAL_TWO_SYSTEM_QUBITS'})
    return prepare_df_partial_s2(h,split_df_hamiltonian_by_ld(h,1),identity_policy='extract_identity_phase')


def test_artificial_steps_retain_config_normalization_seed_and_paired_semantics():
    from trottertracks.resource_applicability.ax2b_native_df_v5 import partial_native_from_step_requests,build_native_hadamard_wrapper
    prep=toy_preparation();steps,receipt=s.make_conditioned_steps(prep,T=.12,q=2,r=2,seed=17,ancilla_qubit=2)
    assert receipt['order2_count']>=1 and receipt['R']==4
    assert all(x.rte_distribution==steps[0].rte_distribution for x in steps)
    assert all(x.rte_config==steps[0].rte_config and x.rte_config.seed==17 for x in steps)
    assert all(x.rte_basis_plan is None and x.rte_occurrence.cancel_adjacent_equal_bases for x in steps)
    assert not receipt['quantum_event_weights_changed'] and not receipt['normalization_changed']
    ordinary=partial_native_from_step_requests(steps,control_policy='ordinary',max_instructions=10000)
    directional=partial_native_from_step_requests(steps,control_policy='symmetric_directional',max_instructions=10000)
    assert np.allclose(Operator(ordinary.circuit).data,Operator(directional.circuit).data,atol=2e-12)
    for axis in ('cosine','sine'):
        a=build_native_hadamard_wrapper(ordinary,axis=axis,include_measurement=False,max_instructions=10000)
        b=build_native_hadamard_wrapper(directional,axis=axis,include_measurement=False,max_instructions=10000)
        assert np.allclose(Operator(a).data,Operator(b).data,atol=2e-12)


def test_artificial_events_keep_unconditional_weights_signs_phases_and_component_probabilities():
    from trotterlib.df_rte_tail import extraction_to_normalized_rte_tail
    from trotterlib.rte import event_unitary
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
    prep=toy_preparation();steps,receipt=s.make_conditioned_steps(prep,T=.12,q=2,r=2,seed=31,ancilla_qubit=2)
    dense=extraction_to_normalized_rte_tail(prep.tail_extraction);operators=dict(zip((x.component_id for x in dense.components),dense.operators))
    components={x.component_id:x for x in prep.rte_preparation.symbolic_tail.components}
    builder=QiskitDFRTEEventCircuitBuilder(basis_registry=prep.rte_preparation.basis_registry)
    for step in steps:
        for event in step.rte_occurrence.events:
            idx=step.rte_distribution.orders.index(event.taylor_order)
            expected=step.rte_distribution.order_probabilities[idx]*math.prod(components[i].probability for i in event.selected_component_ids)
            assert math.isclose(event.event_probability,expected,rel_tol=1e-13)
            assert event.phase==(-1)**(event.taylor_order//2)
            request=prep.rte_preparation.request_for_event(event,controlled=True,ancilla_qubit=2)
            actual=Operator(builder.build_event(request).circuit).data;unitary=event_unitary(event,operators)
            reference=np.block([[np.eye(4),np.zeros((4,4))],[np.zeros((4,4)),unitary]])
            assert np.allclose(actual,reference,atol=2e-12)
    assert receipt['conditional_event_sequence_probability']==receipt['original_event_sequence_probability']/receipt['s2_probability']


def test_wrong_finite_law_rejected_before_any_sample():
    prep=toy_preparation()
    with pytest.raises(ValueError,match='FROZEN_FINITE_LAW'):
        s.make_conditioned_steps(prep,T=.12,q=2,r=2,seed=17,ancilla_qubit=2,expected_distribution={'wrong':True})


@pytest.mark.parametrize('change',['valid','old_kind','draft','cap','seed','bool_approval','cpu','output','conditioning','input'])
def test_grant_is_new_exact_and_cannot_reuse_old_approval(tmp_path,change):
    m=manifest(tmp_path);g=c.grant_draft(m);g.update(approved_by_user=True,server_usage_approved_by_user=True,
        conditional_pilot_authorized=True,cumulative_cap_amendments_approved=True)
    if change=='old_kind':g['kind']='H6_INDEPENDENT32_COMPLETE_ONLY_HASH_REUSE_EIGHT_WORKER_V1'
    if change=='draft':g['conditional_pilot_authorized']=False
    if change=='cap':g['proposed_cumulative_caps']['compile']=1720
    if change=='seed':g['task_list_digest']='0'*64
    if change=='bool_approval':g['approved_by_user']=1
    if change=='cpu':g['resources']['cost_cpu_sets']=[[21]]
    if change=='output':g['exclusive_output']+='/other'
    if change=='conditioning':g['conditioning_digest']='0'*64
    if change=='input':g['input_identity_digest']='0'*64
    if change=='valid':c.validate_grant(m,g)
    else:
        with pytest.raises(ValueError):c.validate_grant(m,g)


@pytest.mark.parametrize('change',['valid','workers','seed','count','phase','approval','output','index_bool','wall'])
def test_scope_remains_fixed_eight_and_unapproved(tmp_path,change):
    m=manifest(tmp_path)
    if change=='workers':m['proposed_resources']['cost_workers']=8
    if change=='seed':m['registrations'][0]['task']['seed']+=1
    if change=='count':m['registrations'].pop()
    if change=='phase':m['registrations'][0]['task']['phase']='confirmation'
    if change=='approval':m['runtime_authorization']=True
    if change=='output':m['exclusive_output']['repository_path']='outside'
    if change=='index_bool':m['registrations'][0]['index']=False
    if change=='wall':m['absolute_launch_command']=['run']
    if change=='valid':c.validate_scope(tmp_path,m)
    else:
        with pytest.raises(ValueError):c.validate_scope(tmp_path,m)


def test_carry_caps_and_stop_reservations_are_conservative(tmp_path):
    m=manifest(tmp_path);a=m['cumulative_accounting'];assert a['known_plus_reserved_plus_new']['compile']==1736
    assert a['known_plus_reserved_plus_new']['trajectory']==852
    group={'calls_attempted':dict(compile=2,control_probe=0,trajectory=1,occurrence=8,primitive=0,reference_matvec=0)}
    complete=e.calls_after_run(m,{i:group for i in range(8)});assert complete['known_calls']['compile']==1716
    assert complete['known_calls']['trajectory']==842 and complete['reservations']['compile']==20
    stopped=e.calls_after_run(m,{0:group});assert stopped['reservations']['compile']==34 and stopped['reservations']['trajectory']==17
    assert stopped['historical_unknown_actual_totals'] is None and not stopped['failed_calls_refunded'] and not stopped['budget_reset']
    assert stopped['known_calls']['compile']+stopped['reservations']['compile']==1736


def test_new_seeds_namespace_and_output_are_disjoint(tmp_path):
    from trottertracks.resource_applicability.h6_matched_accounting_v1 import seed
    new={x['task']['seed'] for x in c.registrations()};assert len(new)==8
    old={seed(c.CELL['id'],phase,i) for phase in ('exploration','confirmation') for i in range(32)}
    assert not new&old and len({s.order_seed(x) for x in new})==8
    assert c.PHASE not in ('exploration','confirmation') and 'conditional_s2' in c.NAMESPACE
    assert c.slots(c.proposed_resources())==1


def test_default_cli_does_not_import_science_or_consume_lock(tmp_path):
    script=ROOT/c.RUNNER
    code="import runpy,sys;sys.argv=[%r]\ntry: runpy.run_path(%r,run_name='__main__')\nexcept SystemExit as exc: assert exc.code==0\nassert not {'numpy','scipy','qiskit','pyscf','cupy'}&set(sys.modules)"%(str(script),str(script))
    r=subprocess.run([sys.executable,'-I','-P','-B','-c',code],capture_output=True,text=True,timeout=20)
    assert r.returncode==0 and 'PREPARATION_ONLY' in r.stdout
    assert not (ROOT/c.LOCK).exists()


def test_draft_gate_precedes_profile_input_and_fresh_checks(tmp_path,monkeypatch):
    m=manifest(tmp_path);g=c.grant_draft(m)
    def forbidden(*args,**kw):raise AssertionError('PROFILE_INPUT_OR_FRESH_BEFORE_GRANT')
    monkeypatch.setattr(c,'validate_plan',forbidden);monkeypatch.setattr(c,'fresh_resources',forbidden)
    with pytest.raises(ValueError,match='NEW_APPROVED'):c.launch_gate(tmp_path,m,g,Path(m['exclusive_output']['absolute_path']))


def test_scoped_port_is_restored_on_exception(monkeypatch):
    from trottertracks.resource_applicability import h6_roundoff_cost_port_v1 as port
    original=port.MatchedPort
    with pytest.raises(RuntimeError):
        with e.conditional_port_scope(c.registrations()[0]['task'],{}):
            assert port.MatchedPort is not original;raise RuntimeError('ARTIFICIAL')
    assert port.MatchedPort is original


def mock_group(path,m,index):
    path.mkdir(parents=True);task=m['registrations'][index]['task'];source=m['source_commit']
    events=[[dict(taylor_order=2 if i==0 and j==0 else 0,event_probability=.2 if i==0 and j==0 else .8) for j in range(2)] for i in range(8)]
    orders=[x['taylor_order'] for step in events for x in step];ed=digest(events);law=m['prior_binding']['finite_distribution']
    rows=[dict(cell_id=task['cell_id'],phase=task['phase'],replica=index,seed=task['seed'],control='symmetric_directional',axis=axis,
        event_digest=ed,compiler_hash=e.COMPILER_HASH,fingerprint='f'*64,
        metrics=dict(RZ=10+index,CX=20,RZ_depth=10,CX_depth=20,depth=30,size=40)) for axis in ('cosine','sine')]
    mass=s.conditional_binomial(16,.2)['s2_probability'];p=math.prod(x['event_probability'] for step in events for x in step)
    data={'worker_terminal.json':dict(status='TASK_COMPLETE',reason=None,source_commit=source,task=task),
        'worker_claim.json':dict(source_commit=source,task_digest=digest(task),resource_limits={'rlimit_as':[32*2**30]*2}),
        'task_binding.json':dict(task=task),'cost_result.json':dict(task=task,rows=rows,calls_attempted=dict(compile=2,control_probe=0,trajectory=1,occurrence=8,primitive=0,reference_matvec=0)),
        'trajectory.json':dict(task=task,events=events,event_digest=ed,observed_orders=orders,finite_distribution=law),
        'conditional_sampling_receipt.json':dict(task=task,stratum='s2',whole_trajectory=True,master_seed=task['seed'],
            event_digest=ed,observed_orders=orders,order2_count=1,original_finite_distribution=law,s2_probability=mass,
            conditional_event_sequence_probability=p/mass,quantum_event_weights_changed=False,normalization_changed=False,
            component_selection_probability_changed=False)}
    data.update({f'wrapper_symmetric_directional_{r["axis"]}.json':r for r in rows})
    for name,obj in data.items():(path/name).write_text(json.dumps(obj))
    return task


@pytest.mark.parametrize('change',['valid','source','seed','phase','no_s2','law','normalization','counter','counter_bool','axis','compiler','event_digest','conditional_probability','missing_receipt'])
def test_admission_rejects_partial_or_changed_mock_group(tmp_path,change):
    m=manifest(tmp_path);directory=tmp_path/'ARTIFICIAL_GROUP';task=mock_group(directory,m,0)
    file=None;mutate=None
    if change=='source':file='worker_terminal.json';mutate=lambda d:d.update(source_commit='0'*40)
    if change=='seed':task=deepcopy(task);task['seed']+=1
    if change=='phase':task=deepcopy(task);task['phase']='exploration'
    if change=='no_s2':file='trajectory.json';mutate=lambda d:d.update(observed_orders=[0]*16)
    if change=='law':file='trajectory.json';mutate=lambda d:d.update(finite_distribution={'wrong':True})
    if change=='normalization':file='conditional_sampling_receipt.json';mutate=lambda d:d.update(normalization_changed=True)
    if change=='counter':file='cost_result.json';mutate=lambda d:d['calls_attempted'].update(compile=3)
    if change=='counter_bool':file='cost_result.json';mutate=lambda d:d['calls_attempted'].update(trajectory=True)
    if change=='axis':file='cost_result.json';mutate=lambda d:d['rows'].pop()
    if change=='compiler':file='cost_result.json';mutate=lambda d:d['rows'][0].update(compiler_hash='0'*64)
    if change=='event_digest':file='trajectory.json';mutate=lambda d:d.update(event_digest='0'*64)
    if change=='conditional_probability':file='conditional_sampling_receipt.json';mutate=lambda d:d.update(conditional_event_sequence_probability=.9)
    if change=='missing_receipt':(directory/'conditional_sampling_receipt.json').unlink()
    if file:
        target=directory/file;d=json.loads(target.read_text());mutate(d);target.write_text(json.dumps(d))
    if change=='valid':assert e.admit_group(directory,task,source=m['source_commit'],finite_distribution=m['prior_binding']['finite_distribution'])['calls_attempted']['compile']==2
    else:
        with pytest.raises((ValueError,FileNotFoundError)):e.admit_group(directory,task,source=m['source_commit'],finite_distribution=m['prior_binding']['finite_distribution'])


@pytest.mark.parametrize('stop',[False,True])
def test_mock_supervisor_integration_complete_or_first_stop_no_retry(tmp_path,monkeypatch,stop):
    m=manifest(tmp_path);out=Path(m['exclusive_output']['absolute_path']);out.mkdir(parents=True)
    for i in range(8):mock_group(e.group_path(out,i),m,i)
    monkeypatch.setattr(e,'install_limits',lambda *a,**k:{'ARTIFICIAL':True})
    monkeypatch.setattr(e,'process_identity',lambda p:{'pid':p,'start_ticks':1})
    monkeypatch.setattr(c,'validate_plan',lambda *a,**k:None)
    calls=[]
    def supervisor(root,directory,manifest,*,command,admission):
        calls.append('ONE_MOCK_SUPERVISOR');argv=command(m['registrations'][0],e.group_path(out,0),0)
        assert argv[:4]==[sys.executable,'-I','-P','-B'] and c.RUNNER in argv[4]
        assert manifest['proposed_resources']['cost_workers']==1
        if not stop:admission(e.group_path(out,0),m['registrations'][0]['task'])
        return dict(completed_indices=[] if stop else list(range(8)),first_stop_reason='ARTIFICIAL_MEMORY_PRESSURE' if stop else None,
            all_dispatched_reaped=True,ownership_cleanup_verified=True,cleanup_errors=[],log_and_journal_closed=True)
    monkeypatch.setattr(e,'supervise',supervisor)
    code=e.orchestrate(tmp_path,out,m,e.AtomicWriter(out,byte_cap=2**30),{'ARTIFICIAL':True})
    terminal=json.loads((out/'execution_terminal.json').read_text());account=json.loads((out/'aggregate_calls.json').read_text())
    assert calls==['ONE_MOCK_SUPERVISOR'] and not terminal['retry'] and not terminal['next_stage_authorized']
    assert terminal['mandatory_stop'] and terminal['resource_summary']['all_dispatched_reaped']
    if stop:
        assert code==1 and 'ARTIFICIAL_MEMORY_PRESSURE' in terminal['reason']
        assert account['known_calls']['compile']==1700 and account['reservations']['compile']==36
    else:
        assert code==0 and terminal['reason'] is None and account['known_calls']['compile']==1716
        assert json.loads((out/'pilot_complete_receipt.json').read_text())['wrapper_count']==16
