"""Artificial/mock H7 readiness checks; no molecular or transpile calls."""
import sys,types,json,math
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
if 'trotterlib' not in sys.modules:
 p=types.ModuleType('trotterlib');p.__path__=[str(ROOT/'src/trotterlib')];sys.modules['trotterlib']=p
import numpy as np
from scipy.linalg import expm
from trotterlib.df_hamiltonian import DFHamiltonian,df_linear_operator
from trottertracks.resource_applicability.h7_open_shell import contract as c, input as inp
from trottertracks.resource_applicability.h7_open_shell.reference import SparseReference,bounded_solver,sector_for,SpinPrimitive
from trottertracks.resource_applicability.h5_open_shell.sector import independent_df_matrix,spin_squared


def spin(a):return np.kron(a,np.eye(2))
def toy(m=3):
 a=np.diag(np.arange(m)*.3-.5)
 return DFHamiltonian(.137,spin(a),np.array([.02,-.03]),
   (spin(np.diag(np.arange(m)*.07+.1)),spin(np.diag(np.arange(m)*.02+.03))),{})


class ContractTests(unittest.TestCase):
 def test_target_and_scope(self):
  t=c.target();self.assertEqual((t['electrons'],t['alpha'],t['beta'],t['sector_dimension']),(7,4,3,1225))
  self.assertEqual(len(t['geometry']),7);self.assertEqual((t['charge'],t['spin'],t['scf']),(0,1,'ROHF'))
  d=c.draft();self.assertFalse(d['approved']);self.assertFalse(d['runtime_authorization']);self.assertIsNone(d['absolute_launch_command'])
 def test_registry_caps_and_rank_unknown(self):
  for rank in (1,14,28,49):
   first=c.tasks(rank,'1'*40,'2'*64);eligible=[r['id'] for r in c.registry(rank) if r['R']]
   extra=c.tasks(rank,'1'*40,'2'*64,extension=True,eligible=eligible)
   self.assertEqual((len(first),len(extra)),(6,14));self.assertEqual(c.registry(rank)[0]['prefix'],(rank+3)//4)
   seeds=[r['seed'] for r in first+extra if r['seed'] is not None];self.assertEqual(len(set(seeds)),16)
  self.assertIsNone(c.draft()['actual_rank']);self.assertEqual(c.call_limits(28)['action_check_matvec'],1225)
  self.assertEqual(c.call_limits(28)['compile'],40)
 def test_namespace_and_no_eligible_extension(self):
  from trottertracks.resource_applicability.h5_open_shell.contract import seed as h5seed
  self.assertNotEqual(c.seed('1'*40,'2'*64,'same',0),h5seed('1'*40,'2'*64,'same',0))
  self.assertEqual(c.tasks(28,'1'*40,'2'*64,extension=True),[])
 def test_reject_old_grant_and_full_new_binding(self):
  m=dict(SOURCE='1'*40,approved=True,runtime_authorization=True,profile_digest='2'*64,carry_digest='3'*64,
    target_digest=c.digest(c.target()),contract_digest=c.digest(c.draft()),compiler_hash=c.COMPILER_HASH,output='/home/AbeHiromu/H7_TEST_UNUSED')
  g=dict(kind='H7_NEUTRAL_OPEN_SHELL_ONE_SHOT_V1',SOURCE='1'*40,approved=True,runtime_authorization=True,
    manifest_digest=c.digest(m),maximum_compile=40,one_shot=True,automatic_retry=False,
    **{k:m[k] for k in ('profile_digest','carry_digest','target_digest','output')})
  self.assertEqual(str(c.verify_grant(m,g,'1'*40)),m['output'])
  for key,value in [('kind','H5_NEUTRAL_OPEN_SHELL_ONE_SHOT_V1'),('approved',False),('maximum_compile',42),('SOURCE','4'*40),('carry_digest','5'*64)]:
   with self.assertRaises(ValueError):c.verify_grant(m,dict(g,**{key:value}),'1'*40)


class InputTests(unittest.TestCase):
 def test_neutral_H7_ROHF_common_MO_injected_only(self):
  seen={}
  class Mol:
   nelectron=7;nelec=(4,3)
   def build(self):seen.update(charge=self.charge,spin=self.spin,geometry=self.atom,basis=self.basis)
   def nao_nr(self):return 7
   def intor_symmetric(self,_):return np.eye(7)
   def energy_nuc(self):return .2
  class MF:
   mo_coeff=np.eye(7);converged=True;e_tot=-1.
   def __init__(self,mol):pass
   def kernel(self):seen['kernels']=seen.get('kernels',0)+1
  provider=(Mol,MF,lambda *args:(np.eye(7),np.zeros((7,)*4)),lambda a,b:(spin(a),np.zeros((14,)*4)))
  arrays,receipt=inp.rohf_integrals(ROOT,c.Calls(c.call_limits(2)),provider=provider)
  self.assertEqual((seen['charge'],seen['spin'],seen['kernels']),(0,1,1));self.assertEqual(len(seen['geometry']),7)
  self.assertEqual(arrays['one_body'].shape,(14,14));self.assertFalse(receipt['state_is_ROHF_determinant'])
 def test_projection_reuses_signed_tol_only_rule(self):
  ham,receipt,_=inp.weighted_df(0.,np.zeros((14,14)),np.zeros((14,)*4),c.Calls(c.call_limits(2)),
   decomposer=lambda a,**k:(np.array([.1,-.3]),np.stack([np.eye(14)*.1,np.eye(14)*.2]),np.zeros((14,14)),0.))
  self.assertEqual(receipt['actual_rank'],2);self.assertEqual(receipt['primary_N'],14)
  np.testing.assert_array_equal(ham.lambdas,[.1,-.3]);self.assertTrue(receipt['all_returned_fragments'])


class SolverReferenceTests(unittest.TestCase):
 def test_solver_1225_budget_interface_and_output_copy(self):
  calls=c.Calls({'solver_matvec':2});workspace=np.empty(1225,dtype=complex)
  def action(x):workspace[:]=x;return workspace
  def fake(op,**kw):
   self.assertEqual((kw['k'],kw['which'],kw['ncv']),(1,'SA',40))
   a=op@kw['v0'];b=op@(kw['v0']*2);np.testing.assert_allclose(a,kw['v0'])
   return np.array([1.]),kw['v0'][:,None]
  x=np.zeros(1225,dtype=complex);x[0]=1
  _,op=bounded_solver(action,1225,x,calls,solver=fake)
  with self.assertRaises(RuntimeError):op@x
 def test_solver_scope_nonfinite_and_shape_fail_closed(self):
  for n in (True,0,1226):
   with self.assertRaises(ValueError):bounded_solver(lambda x:x,n,np.ones(3),c.Calls({'solver_matvec':1}))
  def fake(op,**kw):op@kw['v0'];return np.array([1.]),kw['v0'][:,None]
  x=np.zeros(9);x[0]=1
  for output in (np.full(9,np.nan),np.zeros(8)):
   with self.assertRaises(ValueError):bounded_solver(lambda x:output,9,x,c.Calls({'solver_matvec':1}),solver=fake)
 def test_sparse_actions_trace_and_tail_against_independent_dense(self):
  ham=toy();sector=sector_for(3,2,1);calls=c.Calls(c.call_limits(2));ref=SparseReference(ham,sector,calls)
  dense=independent_df_matrix(ham,sector);trunc=independent_df_matrix(ham.select_blocks((0,)),sector)
  rng=np.random.default_rng(14);x=rng.normal(size=9)+1j*rng.normal(size=9)
  np.testing.assert_allclose(ref.raw_action(x),dense@x,atol=1e-13)
  np.testing.assert_allclose(ref.raw_action(x,prefix=1),trunc@x,atol=1e-13)
  np.testing.assert_allclose(ref.raw_action(x,tail_from=1,identity=.17,scale=.4),(dense-trunc-.17*np.eye(9))@x/.4,atol=1e-13)
  _,trace=ref.operator();self.assertAlmostEqual(trace,np.trace(dense).real)
  self.assertFalse(ref.one.data.flags.writeable)
 def test_reference_state_spin_and_full_column_agreement(self):
  ham=toy();sector=sector_for(3,2,1);H=independent_df_matrix(ham,sector);values,v=np.linalg.eigh(H);x=v[:,0]
  calls=c.Calls(c.call_limits(2));ref=SparseReference(ham,sector,calls)
  production=df_linear_operator(ham,sector,backend='python')[0]
  checks=ref.validate(x,production)
  self.assertEqual(checks['action_columns_checked'],9);self.assertAlmostEqual(checks['S_squared'],.75)
  self.assertAlmostEqual(checks['reference'],np.vdot(x,expm(-.8j*H)@x))
  self.assertEqual(calls.used['action_check_matvec'],9);self.assertEqual(calls.used['reference_exponential'],3)
 def test_reference_does_not_accept_wrong_production_or_state(self):
  ham=toy();sector=sector_for(3,2,1);H=independent_df_matrix(ham,sector);_,v=np.linalg.eigh(H);x=v[:,0]
  from scipy.sparse.linalg import aslinearoperator
  with self.assertRaisesRegex(ValueError,'INDEPENDENT_DF_ACTION'):
   SparseReference(ham,sector,c.Calls(c.call_limits(2))).validate(x,aslinearoperator(H+1e-3*np.eye(9)))
  with self.assertRaisesRegex(ValueError,'NORMALIZATION'):
   SparseReference(ham,sector,c.Calls(c.call_limits(2))).validate(x*2,aslinearoperator(H))
 def test_high_spin_not_silently_replaced(self):
  ham=DFHamiltonian(0.,np.zeros((6,6)),np.array([0.]),(np.eye(6),),{})
  sector=sector_for(3,2,1);ss=spin_squared(sector).toarray();w,v=np.linalg.eigh(ss);x=v[:,-1]
  from scipy.sparse.linalg import aslinearoperator
  def fake(op,**kw):return np.array([0.]),x[:,None]
  with self.assertRaisesRegex(ValueError,'NOT_DOUBLET'):
   SparseReference(ham,sector,c.Calls(c.call_limits(1))).validate(x,aslinearoperator(np.zeros((9,9))),solver=fake)
 def test_numba_thread1_action_against_python_and_CSR(self):
  ham=toy();sector=sector_for(3,2,1);calls=c.Calls(c.call_limits(2));ref=SparseReference(ham,sector,calls)
  op=df_linear_operator(ham,sector,backend='numba',num_threads=1,block_chunk_size=1)[0]
  x=np.arange(9,dtype=complex)*(1+.2j)
  np.testing.assert_allclose(op@x,ref.raw_action(x),atol=1e-13)
  status=Path('/proc/self/status').read_text();threads=int(next(v.split(':')[1] for v in status.splitlines() if v.startswith('Threads:')))
  self.assertEqual(threads,1)
 def test_1225_unequal_spin_primitive(self):
  sector=sector_for(7,4,3);a=SpinPrimitive(spin(np.diag(np.arange(7)*.1)),sector)
  self.assertEqual(a.sides,(35,35));self.assertNotEqual(a.alpha_basis,a.beta_basis)
  rng=np.random.default_rng(707);x=rng.normal(size=1225)+1j*rng.normal(size=1225);x/=np.linalg.norm(x)
  y=a(x,.8);np.testing.assert_allclose(a(y,-.8),x,atol=1e-12)
 def test_1225_numba_chunked_action_against_independent_CSR(self):
  rng=np.random.default_rng(70751);m=7;sector=sector_for(7,4,3)
  def matrix():
   a=rng.normal(size=(m,m))*.03;return spin(a+a.T)
  ham=DFHamiltonian(.17,matrix(),np.array([.1,-.08,.02,-.01,.04,.05]),tuple(matrix() for _ in range(6)),{})
  calls=c.Calls(c.call_limits(6));ref=SparseReference(ham,sector,calls)
  op=df_linear_operator(ham,sector,backend='numba',num_threads=1,block_chunk_size=1)[0]
  for _ in range(3):
   x=rng.normal(size=1225)+1j*rng.normal(size=1225);x/=np.linalg.norm(x)
   np.testing.assert_allclose(op@x,ref.raw_action(x),atol=2e-12,rtol=1e-12)


class AdapterTests(unittest.TestCase):
 def setUp(self):
  from trottertracks.resource_applicability.h7_open_shell import science
  self.sci=science
  def forbidden(*a,**k):raise AssertionError('REAL_INPUT_OR_TRANSPILE_FORBIDDEN')
  self.patches=[patch('numpy.load',forbidden),patch('qiskit.transpile',forbidden),patch('trotterlib.rte_compiled_cost.transpile',forbidden)]
  for p in self.patches:p.start();self.addCleanup(p.stop)
 def test_sparse_signal_matches_preserved_H5_dense_path(self):
  from trottertracks.resource_applicability.h5_open_shell.science import panel_signals as dense_signal
  from trottertracks.resource_applicability.h5_open_shell.contract import call_limits as h5limits
  from trottertracks.resource_applicability.h5_open_shell.sector import validate_state
  ham=toy();sector=sector_for(3,2,1);H=independent_df_matrix(ham,sector);_,v=np.linalg.eigh(H);x=v[:,0]
  checks,_=validate_state(ham,sector,x);prep=self.sci.preparations(ham)
  old=dense_signal(ham,sector,x,H,checks,prep,c.Calls(h5limits(2)))
  calls=c.Calls(c.call_limits(2));ref=SparseReference(ham,sector,calls)
  new=self.sci.panel_signals(ham,sector,x,ref,checks,prep,calls)
  self.assertEqual(len(new),6)
  for a,b in zip(old,new):
   self.assertEqual(a['cell'],b['cell']);self.assertEqual(a['reference'],b['reference'])
   for key in a['signals']:
    for axis in ('real','imag'):self.assertAlmostEqual(a['signals'][key][axis],b['signals'][key][axis],places=12)
   self.assertAlmostEqual(a['log_B'],b['log_B'],places=14)
  self.assertEqual(calls.used['reference_exponential'],3)
 def test_14_mode_and_ancilla_phases_without_dense_unitary(self):
  from trottertracks.resource_applicability.ax2b_h4_science_v5 import checked_basis_bridge
  from trottertracks.resource_applicability.ax2b_native_df_v5 import build_native_hadamard_wrapper
  from trotterlib.df_trotter.circuit import simulate_statevector
  from trottertracks.resource_applicability.ax2b_molecular_ports_v3 import hadamard_expectation
  ham=toy(7);sector=sector_for(7,4,3);x=np.zeros(1225,dtype=complex);x[0]=1
  full=np.zeros(1<<14,dtype=complex);full[sector.basis_indices]=x
  _,qstate=checked_basis_bridge(ham,sector,full,x,1e-12)
  prep=self.sci.preparations(ham);cell=c.registry(2)[3];task=dict(cell=cell,replica=0,seed=None,validate=True)
  evolution,_=self.sci.build_group(ham,prep,task,c.Calls(c.call_limits(2)))
  ref=SparseReference(ham,sector,c.Calls(c.call_limits(2)));energy=np.vdot(x,ref.raw_action(x)).real;z=np.exp(-.8j*energy)
  for axis,expected in [('cosine',z.real),('sine',z.imag)]:
   wrapper=build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,max_instructions=4000000)
   observed=hadamard_expectation(simulate_statevector(wrapper,np.r_[qstate,np.zeros_like(qstate)]),14)
   self.assertAlmostEqual(observed,expected,places=10)
 def test_mock_input_stage_uses_H7_adapter_and_call_journal(self):
  from trottertracks.resource_applicability.h7_open_shell.execution import input_worker
  import tempfile
  private=Path('/home/AbeHiromu/projects/h7-handoff-evidence/20261016/open-shell-readiness')
  ham=toy(7);sector=sector_for(7,4,3);x=np.zeros(1225,dtype=complex);x[0]=1
  full=np.zeros(16384,dtype=complex);full[sector.basis_indices]=x
  def provider(scratch,calls):
   calls.take('scf');calls.take('integral_build')
   return dict(constant=np.array(.1),one_body=np.zeros((14,14)),two_body=np.zeros((14,)*4)),dict(mock=True)
  def df(*a,**kw):a[-1].take('df_decomposition');return ham,dict(actual_rank=2),{'weights':ham.lambdas}
  checks=dict(reference=1+0j)
  with tempfile.TemporaryDirectory(dir=private,prefix='synthetic-input-') as d:
   with patch('trottertracks.resource_applicability.h7_open_shell.input.rohf_integrals',provider),patch('trottertracks.resource_applicability.h7_open_shell.input.weighted_df',df),patch.object(self.sci,'solve_state',return_value=(sector,x,full,checks,None)),patch.object(self.sci,'preparations',return_value={}),patch.object(self.sci,'validate_primitives',return_value={'max_error':0.}),patch.object(self.sci,'panel_signals',return_value=[]):
    result=input_worker(Path(d),dict(stage='input',SOURCE='1'*40))
   self.assertEqual(result['snapshot']['arrays']['state']['shape'],[1225]);self.assertEqual(result['snapshot']['arrays']['full']['shape'],[16384])
   self.assertTrue((Path(d)/'integrals.npz').exists());self.assertTrue((Path(d)/'call_reservations.jsonl').exists())
 def test_mock_coordinator_full_registry_binding_and_AS_inheritance(self):
  from trottertracks.resource_applicability.h7_open_shell import execution as ex
  from trottertracks.resource_applicability.h7_open_shell.input import array_identity
  import tempfile,contextlib
  private=Path('/home/AbeHiromu/projects/h7-handoff-evidence/20261016/open-shell-readiness')
  ham=toy(7);sector=sector_for(7,4,3);x=np.zeros(1225,dtype=complex);x[0]=1
  full=np.zeros(16384,dtype=complex);full[sector.basis_indices]=x
  data=dict(constant=np.array(ham.constant),one_body=ham.one_body,lambdas=ham.lambdas,
    g_matrices=np.array(ham.g_matrices),sector_basis=sector.basis_indices,state=x,full=full)
  class FakeNPZ:
   files=list(data)
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def __getitem__(self,k):return data[k]
  with tempfile.TemporaryDirectory(dir=private,prefix='mock-dispatch-') as d:
   parent=Path(d);output=parent/'output';snapshot=parent/'synthetic-snapshot';snapshot.write_bytes(b'artificial fixture, no real NPZ')
   expected_tasks=[];limits=c.resource_proposal();m=dict(SOURCE='1'*40,output=str(output),worker_cpus=[0,1,2,3],coordinator_cpu=4,
      limits=limits,compiler_hash=c.COMPILER_HASH,profile_digest='2'*64,carry_digest='3'*64)
   saved=dict(snapshot=dict(path=str(snapshot),sha256=ex.file_hash(snapshot),arrays={k:array_identity(v) for k,v in data.items()}),
      DF=dict(actual_rank=2),calls_attempted=dict.fromkeys(c.call_limits(2),0),
      signal_records=[dict(cell=cell,signals=dict(corrected=dict(real=1.,imag=0.)),
         precision_rows={'0.05':{'1.0':{'cost_acquisition_eligible':True}}}) for cell in c.registry(2)])
   def supervise(directory,registered,fn,assigned,limits,baseline,**kw):
    self.assertIn('observe',kw);expected_tasks.append((directory.name,len(registered),len(assigned)))
    if directory.name=='input':completed=[saved]
    else:
     completed=[]
     for task in registered:
      used=dict.fromkeys(c.call_limits(2),0);used.update(compile=2,trajectory=bool(task['cell']['R']),sampling_call=bool(task['cell']['R']),occurrence=task['cell']['R'] or 0)
      completed.append(dict(task=task,rows=[dict(axis=a) for a in ('cosine','sine')],calls_attempted=used))
    result=dict(completed=completed,retired=[],cleanup_errors=[],first_stop=None)
    ex.write_json(directory/'summary.json',result);return result
   original=Path.read_text;stat_calls=[0]
   def read_text(path,*a,**kw):
    if str(path)=='/proc/stat':
     stat_calls[0]+=100;return '\n'.join(f'cpu{i} 0 0 0 {stat_calls[0]} 0 0 0 0' for i in range(5))
    if '/topology/' in str(path):return str(path.parent.parent.name)
    return original(path,*a,**kw)
   patches=[patch.object(ex,'verify_grant',return_value=output),patch.object(ex,'verify_assets'),
    patch.object(ex,'supervise',supervise),patch.object(ex.os,'sched_getaffinity',return_value=set(range(5))),
    patch.object(ex.os,'sched_setaffinity'),patch.object(ex.resource,'getrlimit',return_value=(-1,-1)),
    patch.object(ex.resource,'setrlimit'),patch.object(ex.time,'sleep'),patch.object(Path,'read_text',read_text),
    patch('trottertracks.resource_applicability.h6_server_quota_v1.quota_readonly',return_value={'status':'KNOWN','items':[]}),
    patch('trottertracks.resource_applicability.h6_server_resources_v1.host_memory',return_value={'effective_available_bytes':2**40,'full_psi_avg10':{'host':0.}}),
    patch.object(np,'load',return_value=FakeNPZ()),patch.object(self.sci,'preparations',return_value={})]
   with contextlib.ExitStack() as stack:
    for p in patches:stack.enter_context(p)
    ex.execute(ROOT,m,{'MOCK_ONLY':True})
    ex.resource.setrlimit.assert_called_once_with(ex.resource.RLIMIT_AS,(8*2**30,-1))
   self.assertEqual(expected_tasks,[('input',1,1),('initial_heavy',1,1),('initial_normal',5,4),('extension',14,4)])
   terminal=json.loads((output/'terminal.json').read_bytes());self.assertEqual((terminal['groups'],terminal['compiled']),(20,40))
   totals=json.loads((output/'call_totals.json').read_bytes());self.assertEqual(totals['new_actual']['trajectory'],16)


class ObserverTests(unittest.TestCase):
 def test_wall_role_memory_and_output_fail_closed(self):
  from trottertracks.resource_applicability.h7_open_shell.execution import runtime_observer
  import tempfile,time
  private=Path('/home/AbeHiromu/projects/h7-handoff-evidence/20261016/open-shell-readiness');limits=c.resource_proposal()
  with tempfile.TemporaryDirectory(dir=private,prefix='mock-monitor-') as d:
   healthy=lambda:dict(VmRSS_bytes=1,VmSize_bytes=1)
   observe=runtime_observer(d,limits,time.monotonic()+10,host=lambda:{'mock':True},self_usage=healthy)
   self.assertIn('coordinator_memory',observe())
   with self.assertRaisesRegex(RuntimeError,'WALL_LIMIT'):
    runtime_observer(d,limits,0,host=lambda:{},self_usage=healthy)()
   with self.assertRaisesRegex(RuntimeError,'COORDINATOR_RSS'):
    runtime_observer(d,limits,time.monotonic()+10,host=lambda:{},self_usage=lambda:dict(VmRSS_bytes=3*2**30,VmSize_bytes=1))()
   (Path(d)/'mock').write_text('synthetic')
   with self.assertRaisesRegex(RuntimeError,'OUTPUT_INODES'):
    runtime_observer(d,dict(limits,output_inodes=0),time.monotonic()+10,host=lambda:{},self_usage=healthy)()


if __name__=='__main__':unittest.main()
