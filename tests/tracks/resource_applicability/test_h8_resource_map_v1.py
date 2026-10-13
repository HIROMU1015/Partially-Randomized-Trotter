"""Artificial/mock verification only; no real input or transpile."""
import hashlib,json
from pathlib import Path
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate,Instruction
from trotterlib.df_trotter.circuit import simulate_statevector
from trotterlib.rte_compiled_cost import _circuit_payload
from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as old_hash
from trottertracks.resource_applicability.ax2b_native_df_v5 import NativeEvolution
from trottertracks.resource_applicability.ax2b_limits import CallBudget
from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import hadamard_expectation
from trottertracks.resource_applicability.h8_control_reuse_v1 import validate_control as old_control
from trottertracks.resource_applicability import h8_resource_map_contract_v1 as c
from trottertracks.resource_applicability import h8_resource_map_optimization_v1 as fast
from trottertracks.resource_applicability.h8_resource_map_execution_v1 import RotatingJournal,demand,rss_reservation

@pytest.fixture(autouse=True)
def artificial_only(monkeypatch):
 def forbidden(*args,**kwargs):raise AssertionError('REAL_NPZ_OR_TRANSPILE_FORBIDDEN')
 monkeypatch.setattr(np,'load',forbidden)
 import qiskit
 from trotterlib import rte_compiled_cost
 monkeypatch.setattr(qiskit,'transpile',forbidden);monkeypatch.setattr(rte_compiled_cost,'transpile',forbidden)

def signals():
 return [dict(cell=x,signals={'corrected':{'real':1.,'imag':0.}},precision_rows={str(e):{'1.0':dict(cost_acquisition_eligible=True,N_total=20,axes={'real':{'shots':10},'imag':{'shots':10}})} for e in (.05,.01,.005,.001)}) for x in c.cells()]

def test_full_budget_and_registry():
 cs=c.cells();assert len(cs)==92 and len({x['id'] for x in cs})==92
 assert [sum(x['method']==m for x in cs) for m in ('B0','B1','B2')]==[12,8,72]
 assert {x['q'] for x in cs}=={1,2,4,8};b=c.call_budget();ds=[demand(x['task']) for x in c.signal_groups()]
 assert b['compile']==1708 and b['trajectory']==832 and b['occurrence']==4208
 for k in ('signal','normalization','primitive','reference_matvec'):assert sum(x[k] for x in ds)==b[k]

def test_exploration_confirmation_independent():
 ss=signals();a=c.cost_tasks(ss,'1'*40,'exploration');assert len(a)==596
 assert sum(demand(x['task'])['compile'] for x in a)==1196
 assert sum(demand(x['task'])['control_probe'] for x in a)==376
 ids=[x['id'] for x in c.cells() if x['R']][:8];b=c.cost_tasks(ss,'1'*40,'confirmation',ids)
 assert len(b)==256 and len({x['task']['seed'] for x in a+b})==852
 assert all(not x['task']['validate'] and not x['task']['ordinary'] for x in b)

def test_seed_not_schedule_dependent():
 ss=signals();a=c.cost_tasks(ss,'2'*40,'exploration');assert a==c.cost_tasks(list(reversed(ss)),'2'*40,'exploration')
 for n in (1,3,8):assert sorted(x['task']['seed'] for i in range(n) for x in a[i::n])==sorted(x['task']['seed'] for x in a)

def test_ineligible_missing_not_filled():
 ss=signals()
 for x in ss:
  for p in x['precision_rows'].values():p['1.0']['cost_acquisition_eligible']=False
 assert c.cost_tasks(ss,'3'*40,'exploration')==[]
 with pytest.raises(ValueError,match='ALL_SIGNALS'):c.cost_tasks(ss[:-1],'3'*40,'exploration')

def test_tie_rule_and_selected_before_confirmation():
 ss=signals();rows=[dict(cell_id=s['cell']['id'],replica=i,axis=a,control='symmetric_directional',metrics={'RZ':1}) for s in ss if s['cell']['method']=='B2' for i in range(8) for a in ('cosine','sine')]
 sel=c.selection(ss,rows,'4'*40)
 assert sel['selected_ids']==sorted(s['cell']['id'] for s in ss if s['cell']['method']=='B2')[:2]
 assert len(sel['registrations'])==64 and sel['fixed_before_confirmation'] and not sel['exploration_confirmation_pooled']

@pytest.mark.parametrize('n',[2,4,16])
@pytest.mark.parametrize('open_control',[False,True])
def test_joint_actual_vectors(n,open_control):
 from qiskit.circuit.library import RZGate
 q=QuantumCircuit(n+1);q.crx(.29,n,0);q.cp(-.37,n,1);q.p(.13,n)
 if open_control:q.append(RZGate(.43).control(ctrl_state=0),[n,0])
 e=NativeEvolution(q,n,.8,1,'fixture','ordinary',1000,old_hash(q))
 rng=np.random.default_rng(1907);psi=rng.normal(size=1<<n)+1j*rng.normal(size=1<<n);psi/=np.linalg.norm(psi)
 v,b=old_control(e,psi,calls=CallBudget(control_probe=4),simulate=simulate_statevector,expectation=hadamard_expectation)
 count=[]
 def run(q,x):count.append(1);return simulate_statevector(q,x)
 z,new=fast.validate_control(e,psi,calls=CallBudget(control_probe=4),simulate=run,expectation=hadamard_expectation,expected_signal=v['signal'],previous_high=b)
 assert z['joint_branch_structure_proved'] and len(count)==1 and np.linalg.norm(new-b)<2e-12
 # Open-control toy has a nonidentity low branch: preserve the observed error.
 assert abs(z['max_error']-v['max_error'])<2e-12

@pytest.mark.parametrize('name',['h','rx','custom'])
def test_unsafe_falls_back(name):
 q=QuantumCircuit(3)
 if name=='h':q.h(2)
 elif name=='rx':q.rx(.29,2)
 else:
  d=QuantumCircuit(1);d.rx(.19,0);g=Gate('rz',1,[]);g.definition=d;q.append(g,[2])
 e=NativeEvolution(q,2,.8,1,'fixture','ordinary',1000,old_hash(q));psi=np.zeros(4,complex);psi[0]=1;count=[]
 def run(q,x):count.append(1);return simulate_statevector(q,x)
 v,b=fast.validate_control(e,psi,calls=CallBudget(control_probe=4),simulate=run,expectation=hadamard_expectation)
 assert not v['joint_branch_structure_proved'] and len(count)==2

def test_low_error_is_observed():
 q=QuantumCircuit(3);q.x(0);e=NativeEvolution(q,2,.8,1,'fixture','ordinary',1000,old_hash(q));psi=np.zeros(4,complex);psi[0]=1
 v,_=fast.validate_control(e,psi,calls=CallBudget(control_probe=4),simulate=simulate_statevector,expectation=hadamard_expectation);assert v['max_error']>1

@pytest.mark.parametrize('cap',[0,2**20,32*2**20])
def test_lru_bytes_and_reservations(cap):
 q=QuantumCircuit(17,1);q.global_phase=-.17
 for i in range(30):q.rz(i*.071,0);q.cx(0,1)
 a=np.zeros((256,256),complex);a[0,0]=complex(-0.,0.);q.append(Instruction('fixture',8,0,[a]),range(8))
 cache=fast.LRUFingerprint(byte_cap=cap);data=bytearray();cache.write_bytes(q,data.extend)
 assert data==json.dumps(_circuit_payload(q),sort_keys=True,separators=(',',':')).encode()
 for _ in range(2):
  assert cache.fingerprint(q)==old_hash(q) and cache.snapshot()['payload_peak_reserved_bytes']<=cap and cache.reserved_bytes==0

@pytest.mark.parametrize('mutation',['phase','array','dtype','shape','zero','definition','control','order'])
def test_lru_mutations(mutation):
 from qiskit.circuit.library import RZGate
 q=QuantumCircuit(9);a=np.zeros((256,256),complex);a[0,0]=complex(-0.,0.);q.append(Instruction('fixture',8,0,[a]),range(8));q.append(RZGate(.27).control(ctrl_state=0),[0,8])
 g=Gate('custom',1,[]);d=QuantumCircuit(1);d.rz(.19,0);g.definition=d;q.append(g,[1]);cache=fast.LRUFingerprint();old=cache.fingerprint(q)
 if mutation=='phase':q.global_phase=.3
 elif mutation=='array':q.data[0].operation.params[0][1,1]=.02j
 elif mutation=='dtype':q.data[0].operation.params=[a.astype(np.complex64)]
 elif mutation=='shape':q.data[0].operation.params=[a.reshape(128,512)]
 elif mutation=='zero':q.data[0].operation.params[0][0,0]=complex(0.,0.)
 elif mutation=='definition':q.data[-1].operation.definition.x(0)
 elif mutation=='control':q.data[1].operation.ctrl_state=1
 else:q.data=list(reversed(q.data))
 assert cache.fingerprint(q)==old_hash(q)!=old

@pytest.mark.parametrize('bad',['nan','object'])
def test_lru_nonfinite(bad):
 q=QuantumCircuit(1);a=np.array([np.nan]) if bad=='nan' else np.array(['x'],dtype=object);q.append(Instruction('bad',1,0,[a]),[0])
 with pytest.raises((ValueError,TypeError)):fast.LRUFingerprint().fingerprint(q)

def test_restore_hash_aliases():
 from trottertracks.resource_applicability import ax2b_native_df_v5 as native
 with pytest.raises(RuntimeError):
  with fast.hashing():
   assert native.canonical_qiskit_circuit_fingerprint!=old_hash
   raise RuntimeError('fixture')
 assert native.canonical_qiskit_circuit_fingerprint is old_hash

def test_unknown_stays_heavy():
 for p in (14,21):
  cell=next(x for x in c.cells() if x['method']=='B2' and x['prefix']==p);assert rss_reservation({'kind':'COST','cell_id':cell['id']})==24*2**30
 assert rss_reservation({'kind':'SIGNAL'})==24*2**30 and set(c.CPUS).isdisjoint([22,23])

def test_rotated_journal_exact(home_tmp):
 j=RotatingJournal(home_tmp/'j',segment_bytes=80,total_bytes=240)
 for i in range(5):j.write({'i':i,'s':'x'*10})
 j.close();assert len(j.sealed)>1
 for x in j.sealed:
  p=j.directory/f"segment_{x['segment']:05d}.jsonl";assert len(p.read_bytes())==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 tiny=RotatingJournal(home_tmp/'tiny',segment_bytes=10,total_bytes=20)
 with pytest.raises(RuntimeError,match='JOURNAL'):tiny.write({'large':'x'*100})
 tiny.close()

@pytest.fixture
def home_tmp(tmp_path_factory):
 p=tmp_path_factory.mktemp('full_fixture');assert p.is_relative_to('/home/AbeHiromu');return p

def test_grant_and_partial_resume_rejected():
 m={'source_commit':'a'*40,'output':'/home/AbeHiromu/artificial'};g=c.grant(m,True)
 assert g!=c.grant(dict(m,source_commit='b'*40),True) and c.grant(m,False)['allowed_cpus']==[]
 assert c.carry()['refund']==0 and c.carry()['historical_unknown_actual'] is None
 with pytest.raises(ValueError,match='TERMINAL'):c.checkpoint(Path('/home/AbeHiromu/unused'),{},'a'*40,{'exit_code':1,'reaped':True})
 with pytest.raises(ValueError,match='REAP'):c.admission(Path('/home/AbeHiromu/unused'),{},'a'*40,reaped=False)

@pytest.mark.parametrize('random',[False,True])
def test_cost_port_old_new_mock_compiler_equivalence(monkeypatch,random):
 from types import SimpleNamespace
 from contextlib import nullcontext
 from unittest.mock import patch
 from trottertracks.resource_applicability import h8_fast_cost_port_v1 as old
 from trotterlib import rte_compiled_cost as compilation
 q=QuantumCircuit(3);q.cx(2,0);q.crz(.17,2,1);q.p(.13,2);builds=[]
 def build(*a,**k):
  builds.append(k['control_policy']);qc=q.copy();return NativeEvolution(qc,2,.8,1,'artificial',k['control_policy'],1000,old_hash(qc))
 prep=SimpleNamespace(exact_rte_lambda_r=.1,constant_coefficient=.13,deterministic_blocks=())
 psi=np.array([1.,1j,.2,.1j],complex);psi/=np.linalg.norm(psi)
 high=np.concatenate((np.zeros_like(psi),psi));b=simulate_statevector(q,high);z=complex(np.vdot(psi,b[4:]))
 event=SimpleNamespace(taylor_order=0,to_dict=lambda:{'taylor_order':0,'fixture':1});steps=(SimpleNamespace(rte_occurrence=SimpleNamespace(events=[event])),)
 class Port:
  def __init__(self,*a,**kw):self.plan=a[1]['plan'];self.caps=self.plan['caps_proposed'];self.preps={};self.calls=CallBudget(compile=4,trajectory=int(random),occurrence=int(random),control_probe=8)
  def sampled_steps(self,cell,seed):
   assert seed==173;self.calls.take('trajectory');self.calls.take('occurrence');return steps
 class Writer:
  def __init__(self):self.rows={}
  def write(self,name,value):self.rows[name]=value
 class Trace:
  def stage(self,*a,**k):return nullcontext()
 m={'plan':c.plan(),'input_identity':{'snapshot_path':'fixture','snapshot_receipt_path':'fixture','df_receipt_path':'fixture'}}
 task=dict(cell={'id':'fixture','method':'B2' if random else 'B1','prefix':1,'q':1,'r':1 if random else None,'R':1 if random else None,'K':2 if random else None,'formula':'2nd'},cell_id='fixture',seed=173,phase='exploration',replica=0,ordinary=True,validate=True,expected_signal={'real':z.real,'imag':z.imag})
 for name,value in [('MatchedPort',Port),('load_h8_snapshot',lambda *a:(SimpleNamespace(n_qubits=2),None,None,None,{},{})),('private_path',lambda *a:Path('fixture')),('checked_basis_bridge',lambda *a:(None,psi)),('_prepare',lambda *a:prep),('_prepare_discard',lambda *a:prep),('actual_bounds',lambda *a,**k:{'cells':[{'wrapper_instruction_upper_bounds':{'x':1000}}]}),('build_deterministic_native',build),('partial_native_from_step_requests',build)]:monkeypatch.setattr(old,name,value)
 captures=[]
 def fake_transpile(qc,**kw):captures.append((old_hash(qc),kw));return qc.copy()
 monkeypatch.setattr(compilation,'transpile',fake_transpile)
 with patch.object(Path,'read_text',return_value='{}'):
  a=Writer();old.cost_port(Path('/home/AbeHiromu'),m,task,a,None,Trace());previous=captures[:];old_builds=builds[:];captures.clear();builds.clear()
  b=Writer();fast.cost_port(Path('/home/AbeHiromu'),m,task,b,None,Trace())
 assert captures==previous and len(captures)==4 and len(old_builds)==4 and len(builds)==2
 assert a.rows['trajectory.json']==b.rows['trajectory.json'] and a.rows['cost_result.json']['calls_attempted']==b.rows['cost_result.json']['calls_attempted']
 for x,y in zip(a.rows['cost_result.json']['rows'],b.rows['cost_result.json']['rows']):
  for k in ('metrics','fingerprint','compiler_hash','seed','event_digest','phase','replica','axis','control'):assert x[k]==y[k]
 assert b.rows['cost_result.json']['full_evolution_simulations']==2

@pytest.mark.parametrize('failure',[False,True])
def test_supervisor_native_exit_cleanup_and_reservations(monkeypatch,home_tmp,failure):
 from types import SimpleNamespace
 from trottertracks.resource_applicability import h8_resource_map_execution_v1 as ex
 run=home_tmp/'run';run.mkdir();host={'host_available_bytes':500*2**30,'effective_available_bytes':500*2**30,'oom':{'fixture':{'oom':0,'oom_kill':0}},'full_psi_avg10':{'host':0,'fixture':0}}
 (run/'fresh_resources.json').write_text(json.dumps({'resources':{'memory':host}}))
 task=next(x['task'] for x in c.cost_tasks(signals(),'a'*40,'exploration') if x['task']['cell']['method']=='B0');rs=[{'index':i,'task':dict(task,cell_id=task['cell_id'])} for i in range(2)]
 pending=[];reaped=[]
 def fake_spawn(*args,**kw):
  p=SimpleNamespace(pid=900001+len(pending),returncode=None);pending.append(p);return p
 def retire(p,identity):p.returncode=1 if failure else 0;reaped.append(p.pid);return p.returncode
 monkeypatch.setattr(ex.subprocess,'Popen',fake_spawn);monkeypatch.setattr(ex,'stop_owned_group',retire)
 monkeypatch.setattr(ex,'process_identity',lambda pid:{'pid':pid,'start_ticks':100+pid,'pgrp':pid,'session':pid,'state':'R'})
 monkeypatch.setattr(ex.os,'waitid',lambda *a:SimpleNamespace(si_status=0))
 monkeypatch.setattr(ex,'host_memory',lambda:host);monkeypatch.setattr(ex,'process_memory',lambda pid:{'VmRSS_bytes':20*2**20,'VmSize_bytes':30*2**20})
 result={'calls_attempted':{k:demand(task)[k] for k in ('compile','trajectory','occurrence','control_probe','primitive','reference_matvec')},'rows':[]}
 monkeypatch.setattr(ex,'admission',lambda *a,**kw:result);monkeypatch.setattr(ex,'checkpoint',lambda *a,**kw:{'complete_only':True})
 m={'source_commit':'a'*40,'proposed_resources':c.resources(),'observation_policy':c.observation(),'call_budget':c.call_budget()}
 ledger={'known':{k:0 for k in demand(task)},'reserved':{k:0 for k in demand(task)}}
 if failure:
  with pytest.raises(RuntimeError,match='WORKER_EXIT:1'):ex.supervisor(home_tmp,run,'exploration',rs,m,{},ledger)
  assert ledger['reserved']['compile']==4 and ledger['known']['compile']==0
 else:
  done,report=ex.supervisor(home_tmp,run,'exploration',rs,m,{},ledger);assert len(done)==2 and report['all_dispatched_reaped'] and not report['cleanup_errors'];assert ledger['known']['compile']==4 and ledger['reserved']['compile']==0
 assert len(pending)==len(reaped)==2

def test_actual_rare_order_not_promoted():
 from trottertracks.resource_applicability.h8_resource_map_execution_v1 import zero_order_measured_class
 task=next(x['task'] for x in c.cost_tasks(signals(),'a'*40,'exploration') if x['task']['cell_id']=='B2_p7_q1_2nd_r4_K2')
 events=[[{'taylor_order':0} for _ in range(4)]];tr={'task':task,'events':events,'event_digest':c.digest(events),'observed_orders':[0]*4}
 assert rss_reservation(task)==24*2**30 and zero_order_measured_class(task,tr)
 tr['observed_orders']=[2,0,0,0];assert not zero_order_measured_class(task,tr)

def test_B0_truncated_reference_against_artificial_dense():
 from scipy.linalg import expm
 from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
 from trotterlib.pr2_matched_accuracy_m1_execution import _prepare_discard
 from trottertracks.resource_applicability.h8_resource_map_port_v1 import H8CoreSignalPort
 from trottertracks.resource_applicability.h8_sparse_reference_v1 import OccupationCSR
 from trottertracks.resource_applicability.ax2b_independent_reference import occupation_df_matrix
 from trottertracks.resource_applicability.ax2b_h6_pilot_port_v1 import differences
 spatial=np.array([[.3,.1,.03j],[.1,-.2,.02],[-.03j,.02,.4]])
 one=np.zeros((6,6),complex);one[::2,::2]=spatial;one[1::2,1::2]=spatial
 ham=DFHamiltonian(.17,one,np.array([.31,-.27]),(one.copy(),2*one.copy()),{})
 sector=PhysicalSector.spin_sector(n_qubits=6,nelec_alpha=1,nelec_beta=1)
 state=np.arange(1,sector.dimension+1,dtype=complex);state/=np.linalg.norm(state)
 cell=dict(id='toy',method='B0',prefix=1,q=2,formula='2nd',r=None,R=None,K=None)
 p=H8CoreSignalPort.__new__(H8CoreSignalPort);p.ham=ham;p.state=state;p.sector=sector;p.plan={'T':.8}
 p.preps={'toy':_prepare_discard(ham,1)};p.calls=CallBudget(reference_matvec=100000)
 p.spin={};p.primitive_ops={};p.oracle_tails={};p.truncated_references={};p.caps={'tail_matvecs_corrected_and_raw':{'B2':512}}
 topology=OccupationCSR(sector.basis_indices,6);p.one=topology.one(one);p.blocks=tuple(topology.one(g) for g in ham.g_matrices)
 result=p.oracle_signal(cell)
 matrix=np.asarray(occupation_df_matrix(ham.constant,one,ham.lambdas[:1],ham.g_matrices[:1],sector.basis_indices))
 full=np.asarray(occupation_df_matrix(ham.constant,one,ham.lambdas,ham.g_matrices,sector.basis_indices))
 exact=np.vdot(state,expm(-.8j*matrix)@state);reference=np.vdot(state,expm(-.8j*full)@state)
 assert abs(result['signals']['exact_truncated']-exact)<1e-12 and abs(exact-reference)>1e-4
 delta=differences(result['signals'],reference,'B0');assert delta['complex_closure_discrepancy']<1e-12

def test_complete_checkpoint_reuse_tamper(home_tmp):
 source='a'*40;ss=signals();cid=next(x['id'] for x in c.cells() if x['method']=='B2')
 task=c.cost_tasks(ss,source,'confirmation',[cid])[0]['task'];d=home_tmp/'completed';d.mkdir()
 events=[[{'taylor_order':0} for _ in range(task['cell']['r'])] for _ in range(task['cell']['q'])]
 tr={'task':task,'events':events,'event_digest':c.digest(events),'observed_orders':[0]*task['cell']['R']}
 rows=[dict(cell_id=cid,phase='confirmation',replica=0,seed=task['seed'],axis=a,control='symmetric_directional',event_digest=tr['event_digest'],compiler_hash=c.COMPILER,fingerprint='b'*64,metrics={k:1 for k in ['RZ','CX','RZ_depth','CX_depth','depth','size']}) for a in ('cosine','sine')]
 values={'worker_terminal.json':{'status':'COMPLETE','reason':None,'source_commit':source},'worker_claim.json':{'source_commit':source,'task_digest':c.digest(task)},'trajectory.json':tr,'cost_result.json':{'task':task,'rows':rows,'calls_attempted':{'compile':2}}}
 for row in rows:values[f"wrapper_symmetric_directional_{row['axis']}.json"]=row
 for name,value in values.items():(d/name).write_text(json.dumps(value))
 receipt=c.checkpoint(d,task,source,{'exit_code':0,'reaped':True})
 assert c.resume_result(d,task,source,receipt)['rows']==rows
 values['cost_result.json']['rows'][0]['metrics']['RZ']=2;(d/'cost_result.json').write_text(json.dumps(values['cost_result.json']))
 with pytest.raises(ValueError,match='RESUME_BYTES'):c.resume_result(d,task,source,receipt)

def test_all_six_cost_metrics_and_ineligible_status():
 from trottertracks.resource_applicability.h8_resource_map_execution_v1 import full_resource_maps
 ss=signals();s=ss[0];rows=[dict(cell_id=s['cell']['id'],replica=0,axis=a,control='symmetric_directional',metrics={k:1 for k in ['RZ','CX','RZ_depth','CX_depth','depth','size']}) for a in ('cosine','sine')]
 # Add the declared allowance sensitivity variant without producing signals.
 for p in s['precision_rows'].values():p['10.0']=p['1.0'].copy()
 result=full_resource_maps([s],rows,c.plan(),'exploration')
 assert {r['metric'] for r in result['rows']}=={'RZ','CX','RZ_depth','CX_depth','depth','size'}

def test_serialized_manifest_binding_and_plan_tamper(monkeypatch,home_tmp):
 monkeypatch.setattr(c,'profile',lambda:{'fixture':'pinned'})
 monkeypatch.setattr(c,'input_identity',lambda d:{'directory':d,'fixture':'input'})
 monkeypatch.setattr(c,'source_paths',lambda root:[])
 monkeypatch.setattr(c,'file_hash',lambda p:'b'*64)
 m=c.make_manifest(home_tmp,'a'*40,str(home_tmp/'input'),home_tmp/'output')
 m=json.loads(json.dumps(m));g=c.grant(m,True);c.verify(home_tmp,m,g,committed=False)
 m['plan']['cells'][0]['q']=16
 with pytest.raises(ValueError,match='FIXED_CONTRACT'):c.verify(home_tmp,m,c.grant(m,True),committed=False)
