"""Synthetic serial admission, immutable gates and exact 2% failure tests."""
from copy import deepcopy
from fractions import Fraction
import importlib,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import h8_phase_b2_contract as c
import h8_phase_b2_runtime as r
import h8_phase_b2_adapter as a
import launch_h8_phase_b2 as launcher
import track_a_phase_a_parallel as pool


def gate_fixture():
 p=c.plan();base='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-h8-B2'
 cfg=dict(SOURCE='a'*40,execution_sealed=True,plan_sha256=c.digest(p),resources=p['resources'],output=base+'/output',control=base+'/control',lock=base+'/lock')
 g=dict(schema='H8_PHASE_B2_NEW_SCOPE_GRANT_V1',scope=c.SCOPE,approved_by_user=True,runtime_authorization=True,approval_text='artificial',SOURCE=cfg['SOURCE'],config_sha256='cfg',plan_sha256=c.digest(p),resources=p['resources'],call_caps=p['call_caps'],one_shot=True,retry=False,resume=False,refit=False,C_authorized=False,next_stage_authorized=False,driver_cpu=0,monitor_cpu=1,worker_cpus=[2],allowed_cpus=[0,1,2])
 fresh=dict(source=cfg['SOURCE'],config_sha256='cfg',grant_sha256=c.digest(g),allowed_cpus=g['allowed_cpus'],monotonic_seconds=100)
 for k in ('profile_verified','source_input_bytes_verified','physical_cores_verified','resources_idle_verified','protected_jobs_unchanged','quota_verified','external_RSS_monitor_ready'):fresh[k]=True
 return cfg,g,fresh


class Contract(unittest.TestCase):
 def test_preregistered_axes_predictions_and_resource_sum(self):
  p=c.plan();self.assertEqual(len(p['axes']),4);self.assertEqual(p['compile_concurrency'],1)
  self.assertEqual(p['axes'][0]['exact_prediction'],'1175933/7');self.assertEqual(p['axes'][3]['exact_prediction'],'1852640')
  self.assertEqual(p['resources']['aggregate_RSS_bytes']+p['resources']['admission_reserve_bytes'],int(80.25*2**30))
  self.assertEqual(p['call_caps']['cost_fit'],0);self.assertEqual(p['call_caps']['compile'],4)
 def test_exact_boundary_measured_denominator(self):
  self.assertFalse(c.axis_check(102,100)['PASS']);self.assertFalse(c.axis_check(98,100)['PASS'])
  self.assertTrue(c.axis_check(Fraction(10199,100),100)['PASS']);self.assertTrue(c.axis_check(100,102)['PASS'])
  for value in (0,-1,True,100.):
   with self.assertRaises(ValueError):c.axis_check(100,value)
 def test_complete_receipt_order_identity_and_exact_caps(self):
  rows=[]
  for x in c.AXES:
   measured=round(Fraction(x['exact_prediction']))
   rows.append(dict(case_id=x['case_id'],axis=x['axis'],model_sha256=c.MODEL_SHA,profile_sha256=c.PROFILE,
    compiler_hash=c.COMPILER,native_exit_code=0,reaped=True,fingerprint_before='x',fingerprint_after='x',max_control_error=1e-12,
    measured_RZ=measured,model_check=c.axis_check(x['exact_prediction'],measured)))
  self.assertFalse(c.validate_complete(rows,c.CAPS)['full_grid_guaranteed'])
  for mutate in (lambda x:x.reverse(),lambda x:x.pop(),lambda x:x[0].update(reaped=False),lambda x:x[0].update(model_sha256='changed'),lambda x:x[0].update(fingerprint_after='bad'),lambda x:x[0].update(max_control_error=1e-8)):
   d=deepcopy(rows);mutate(d)
   with self.assertRaises(ValueError):c.validate_complete(d,c.CAPS)
 def test_no_scientific_import_before_new_permit(self):
  deny=NS(require=Mock(side_effect=ValueError('DENIED')))
  with patch.object(a.importlib,'import_module') as imported:
   with self.assertRaisesRegex(ValueError,'DENIED'):a.evaluate({},deny,None)
   imported.assert_not_called()
 def test_scalar_B0_and_B0I_once_distinct(self):
  saved=dict(policy='REUSED_SAVED_P21_FULL_FOCK_SCALAR',receipt=dict(C=.125,full_Fock_identity=True,new_scientific_certificate=False))
  self.assertEqual(a.scalar_for('B0',(.5).hex(),dict(scalar_receipt=None)),.5)
  self.assertEqual(a.scalar_for('B0I',(.5).hex(),dict(scalar_receipt=saved)),.625)
  with self.assertRaises(ValueError):a.scalar_for('B0',(.5).hex(),dict(scalar_receipt=saved))
  saved['receipt']['full_Fock_identity']=False
  with self.assertRaises(ValueError):a.scalar_for('B0I',(.5).hex(),dict(scalar_receipt=saved))
 def test_replay_requires_new_B2_scope_not_legacy_grant(self):
  permit=NS(require=Mock(side_effect=ValueError('DENIED')),take=Mock())
  with self.assertRaisesRegex(ValueError,'DENIED'):a.restore({},permit)
  permit.require.assert_called_once_with(c.SCOPE);permit.take.assert_not_called()


class Gates(unittest.TestCase):
 def test_fresh_CPU_failure_is_recorded_before_fork_and_attempt_is_spent(self):
  with tempfile.TemporaryDirectory() as td:
   base=Path(td);cfg,g,_=gate_fixture();cfg.update(output=str(base/'output'),control=str(base/'control'),lock=str(base/'lock'))
   config_path=base/'config.json';config_path.write_text(json.dumps(cfg));g['config_sha256']=r.sha(config_path)
   grant_path=base/'grant.json';grant_path.write_text(json.dumps(g));grant_hash=r.sha(grant_path)
   busy=dict(status='STOP',CPUs=[dict(cpu=0,status='BUSY')],failing_CPUs=[0])
   with patch.object(r,'private_path',side_effect=Path),patch.object(r,'verify_files'),patch.object(r,'protected'),\
    patch.object(r,'topology',return_value={0,1,2}),patch.object(r,'measure_idle',return_value=busy) as sample,\
    patch.object(os,'sched_getaffinity',return_value={0,1,2}),patch.object(os,'sched_setaffinity'),patch.object(launcher.observer,'supervise') as supervise:
    with self.assertRaises(r.CPUIdleError):launcher.launch(config_path,grant_path,base/'fresh_gate.json')
    receipt=json.loads((base/'launcher_admission01/terminal.json').read_text())
    self.assertFalse(receipt['driver_started']);self.assertEqual(receipt['stage'],'INITIAL_FRESH_CPU_RESOURCE_GATE')
    self.assertEqual(r.sha(grant_path),grant_hash);supervise.assert_not_called()
    with self.assertRaisesRegex(ValueError,'ATTEMPT_ALREADY_SPENT'):launcher.launch(config_path,grant_path,base/'fresh_gate.json')
    self.assertEqual(sample.call_count,1)
 def test_refresh_failure_retains_CPU_counters_and_launch_affinity(self):
  busy=dict(status='STOP',CPUs=[dict(cpu=2,status='BUSY')],failing_CPUs=[2]);saved=[]
  with patch.object(r,'verify_authorization',return_value=[0,1,2]) as authorize,patch.object(r,'sha',return_value='cfg'),patch.object(r,'protected'),patch.object(r,'topology',return_value={0,1,2}),patch.object(r,'measure_idle',return_value=busy):
   with self.assertRaises(r.CPUIdleError):launcher.fresh({}, {},Path('/tmp/cfg'),launch_affinity={0,1,2},cpu_record=saved.append)
   self.assertEqual(saved,[busy]);self.assertEqual(authorize.call_args.kwargs['affinity'],{0,1,2})
 def test_duplicate_physical_core_rejected_before_idle_sample(self):
  with self.assertRaisesRegex(ValueError,'THREE_PHYSICAL'):r.topology([0,0,0])
 def test_new_scope_grant_valid_only_with_all_three_cpus_and_current_policy(self):
  cfg,g,f=gate_fixture();r.verify_grant(cfg,g,f,config_sha='cfg',now=101,affinity={0,1,2})
  for key,value in [('schema','H8_B0_ANCHOR_NEW_SCOPE_GRANT_V1'),('approved_by_user',False),('retry',True),('refit',True),('C_authorized',True),('worker_cpus',[2,3]),('scope','compile_validation')]:
   cfg,g,f=gate_fixture();g[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):r.verify_grant(cfg,g,f,config_sha='cfg',now=101,affinity={0,1,2,3})
 def test_stale_freshness_or_unsealed_source_rejected(self):
  for key,value in [('monotonic_seconds',50),('grant_sha256','wrong'),('quota_verified',False)]:
   cfg,g,f=gate_fixture();f[key]=value
   with self.assertRaises(ValueError):r.verify_grant(cfg,g,f,config_sha='cfg',now=101,affinity={0,1,2})
  cfg,g,f=gate_fixture();cfg['execution_sealed']=False
  with self.assertRaises(ValueError):r.verify_grant(cfg,g,f,config_sha='cfg',now=101,affinity={0,1,2})
 def test_four_irrevocable_reservations_no_refund_or_duplicate(self):
  cfg,g,_=gate_fixture()
  with tempfile.TemporaryDirectory() as td:
   permit=r.Permit(cfg,g,td)
   for i in range(4):permit.reserve(dict(id=str(i),counts={'compile':1}))
   with self.assertRaises(ValueError):permit.reserve(dict(id='4',counts={'compile':1}))
   permit.settle('0');self.assertEqual(permit.actual()['compile'],1)
   with self.assertRaises(ValueError):permit.settle('0')
   with self.assertRaises(ValueError):permit.take('cost_fit')
 def test_draft_denied_before_files_and_supervisor(self):
  cfg,g,_=gate_fixture();g['approved_by_user']=False
  with patch.object(r,'read',side_effect=[cfg,g]),patch.object(r,'sha',return_value='cfg'),patch.object(r,'verify_files') as files,patch.object(launcher.observer,'supervise') as launch:
   with self.assertRaises(ValueError):launcher.launch(Path('/tmp/c'),Path('/tmp/g'),Path('/tmp/f'))
   files.assert_not_called();launch.assert_not_called()
 def test_cli_is_plan_only_no_scientific_modules(self):
  for script in ('run_h8_phase_b2.py','launch_h8_phase_b2.py'):
   code='import runpy,sys;sys.argv=["'+script+'"];runpy.run_path("scripts/'+script+'",run_name="__main__");assert not any(x in sys.modules for x in ("numpy","scipy","qiskit"))'
   result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=10)
   self.assertEqual(result.returncode,0,result.stderr)


class Serial(unittest.TestCase):
 def fixture(self,td,*,fail_at=None,build_fail=False,worker_fail=False):
  cfg,g,_=gate_fixture();permit=r.Permit(cfg,g,td);events=[]
  def build(case):
   events.append('build:'+case['id'])
   if build_fail:raise ValueError('CONTROL_FAILURE')
   return case
  def compile_axis(task,built,directory):
   number=len([e for e in events if e.startswith('compile:')]);prediction=c.AXES[number-1]['exact_prediction']
   return dict(case_id=task['case_id'],axis=task['axis'],measured_RZ=1 if number==fail_at else round(Fraction(prediction)))
  def fake_map(tasks,callback,*,directory,reserve,**kwargs):
   self.assertEqual(len(tasks),1);self.assertEqual(len(kwargs['cpus']),1);task=tasks[0]
   events.append('compile:'+task['id']);reserve(task);directory.mkdir()
   if worker_fail:raise ValueError('WORKER_FAIL')
   work=directory/task['id'];work.mkdir();(work/'task_result.json').write_text(json.dumps(callback(task,work)))
   return dict(status='PASS',all_reaped=True,cleanup_errors=[],workers=[dict(task_id=task['id'],directory=str(work),exit_code=0,reaped=True,identity=[1,2,3,4])],child_CPU_seconds=1.)
  try:rows=a.serial_axes(c.plan(),permit,r.Writer(td),build,compile_axis,map_owned=fake_map)
  except ValueError as exc:return events,None,str(exc),permit
  return events,rows,None,permit
 def test_success_order_and_all_checks(self):
  with tempfile.TemporaryDirectory() as td:
   events,rows,error,permit=self.fixture(td)
   self.assertEqual(len(rows),4);self.assertEqual(permit.actual()['model_error_check'],4)
   self.assertEqual(events,['build:B0_p15_q1_4th','compile:B0_p15_q1_4th_cosine','compile:B0_p15_q1_4th_sine','build:B0I_p21_q8_4th','compile:B0I_p21_q8_4th_cosine','compile:B0I_p21_q8_4th_sine'])
 def test_failure_at_every_axis_blocks_all_later_axes(self):
  for i in range(1,5):
   with tempfile.TemporaryDirectory() as td:
    events,rows,error,permit=self.fixture(td,fail_at=i)
    self.assertEqual(len([e for e in events if e.startswith('compile:')]),i);self.assertEqual(error,'AXIS_MODEL_ERROR_GE_2_PERCENT')
    self.assertEqual(permit.actual()['compile'],i);self.assertIsNone(rows)
 def test_control_failure_admits_zero_compile(self):
  with tempfile.TemporaryDirectory() as td:
   events,rows,error,permit=self.fixture(td,build_fail=True);self.assertEqual(permit.actual()['compile'],0)
   self.assertFalse(any(e.startswith('compile:') for e in events))
 def test_worker_failure_is_not_settled_and_no_next_axis(self):
  with tempfile.TemporaryDirectory() as td:
   events,rows,error,permit=self.fixture(td,worker_fail=True)
   self.assertEqual(len(permit.reserved),1);self.assertEqual(len(permit.settled),0);self.assertEqual(len(events),2)
 def test_real_one_worker_success_and_failure_reap(self):
  cpu=min(os.sched_getaffinity(0));limits=dict(max_workers=1,AS_bytes=512*2**20,RSS_bytes=256*2**20,worker_CPU_seconds=10,aggregate_CPU_seconds=20)
  for fail in (False,True):
   with tempfile.TemporaryDirectory() as td:
    directory=Path(td)/'one'
    def callback(task,path):
     if fail:raise ValueError('ARTIFICIAL_COMPILE_FAILURE')
     return dict(artificial=True)
    if fail:
     with self.assertRaises(RuntimeError):pool.map_owned([dict(id='one')],callback,cpus=[cpu],limits=limits,directory=directory,deadline=time.monotonic()+10)
    else:pool.map_owned([dict(id='one')],callback,cpus=[cpu],limits=limits,directory=directory,deadline=time.monotonic()+10)
    receipt=json.loads((directory/'parallel_receipt.json').read_text());self.assertTrue(receipt['all_reaped']);self.assertFalse(receipt['automatic_retry'])


if __name__=='__main__':unittest.main()
