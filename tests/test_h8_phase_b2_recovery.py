"""Synthetic recovery gates, real two-worker cleanup and durable native-wait tests."""
from copy import deepcopy
import io,json,os,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch,Mock
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import h8_phase_b2_recovery_contract as c
import h8_phase_b2_recovery_runtime as r
import h8_phase_b2_recovery_adapter as a
import h8_phase_b2_recovery_parallel as parallel
import h8_phase_b2_recovery_observer as observer
import h8_phase_b2_recovery_witness as witness
import launch_h8_phase_b2_recovery as launcher
import track_a_phase_a_parallel as pool


def fixture():
 p=c.plan();cfg=dict(SOURCE='a'*40,execution_sealed=True,plan_sha256=c.digest(p),resources=c.RESOURCES,output='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-recovery/output',control='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-recovery/control',lock='/home/AbeHiromu/projects/h6-handoff-evidence/artificial-recovery/lock')
 g=dict(schema='H8_PHASE_B2_RECOVERY_NEW_SCOPE_GRANT_V1',scope=c.SCOPE,approved_by_user=True,runtime_authorization=True,approval_text='artificial',SOURCE=cfg['SOURCE'],config_sha256='cfg',plan_sha256=c.digest(p),resources=c.RESOURCES,call_caps=c.CAPS,one_shot=True,retry=False,resume=False,refit=False,C_authorized=False,next_stage_authorized=False,accept_saved_B0_2_axes=True,driver_cpu=0,monitor_cpu=1,worker_cpus=[2,3],witness_cpu=4,allowed_cpus=[0,1,2,3,4])
 return cfg,g


class Contract(unittest.TestCase):
 def test_remaining_two_and_parallel_preparation_not_axis_parallel(self):
  p=c.plan();self.assertEqual(len(p['axes']),2);self.assertEqual(p['axes'][0]['case_id'],'B0I_p21_q8_4th');self.assertEqual(p['compile_concurrency'],1);self.assertEqual(p['preparation_concurrency'],2)
  self.assertEqual(c.RESOURCES['physical_cores'],5);self.assertEqual((c.RESOURCES['aggregate_RSS_bytes']+c.RESOURCES['admission_reserve_bytes'])/2**30,112.3125)
  self.assertEqual(c.CAPS['control_probe'],4);self.assertEqual(c.CAPS['compile'],2);self.assertEqual(c.CAPS['cost_fit'],0)
 def test_saved_B0_receipts_require_independent_future_grant(self):
  v=c.saved_B0();self.assertEqual(v['original_execution_status'],'STOP');self.assertEqual(len(v['axes']),2)
  cfg,g=fixture();r.verify_authorization(cfg,g,config_sha='cfg',affinity=set(range(5)))
  for key,value in [('accept_saved_B0_2_axes',False),('approved_by_user',False),('scope',c.original.SCOPE),('retry',True),('worker_cpus',[2]),('witness_cpu',3)]:
   bad=deepcopy(g);bad[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):r.verify_authorization(cfg,bad,config_sha='cfg',affinity=set(range(5)))
 def test_new_draft_denied_before_any_witness_or_science(self):
  cfg,g=fixture();g['runtime_authorization']=False
  with patch.object(r,'read',side_effect=[cfg,g]),patch.object(r,'sha',return_value='cfg'),patch.object(subprocess,'Popen') as spawn:
   with self.assertRaises(ValueError):witness.dispatch(Path('/tmp/cfg'),Path('/tmp/grant'))
   spawn.assert_not_called()
 def test_no_scientific_import_before_scope(self):
  deny=Mock();deny.require.side_effect=ValueError('DENIED')
  with patch.object(a.importlib,'import_module') as imported:
   with self.assertRaisesRegex(ValueError,'DENIED'):a.evaluate({},deny,None)
   imported.assert_not_called()
 def test_cli_defaults_do_not_import_science(self):
  for name in ('run_h8_phase_b2_recovery','launch_h8_phase_b2_recovery','h8_phase_b2_recovery_witness'):
   code='import runpy,sys;sys.argv=["'+name+'"];runpy.run_path("scripts/'+name+'.py",run_name="__main__");assert not any(x in sys.modules for x in ("numpy","scipy","qiskit"))'
   result=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True,timeout=10)
   self.assertEqual(result.returncode,0,result.stderr)
 def test_nonrefundable_compile_and_qpy_reservations(self):
  cfg,g=fixture()
  with tempfile.TemporaryDirectory() as td:
   permit=r.Permit(cfg,g,td)
   for i in range(2):permit.reserve(dict(id=str(i),counts={'compile':1,'qpy_load':1}))
   with self.assertRaises(ValueError):permit.reserve(dict(id='extra',counts={'compile':1,'qpy_load':1}))
   permit.settle('0');self.assertEqual(permit.actual()['compile'],1);self.assertEqual(permit.actual()['qpy_load'],1)
   self.assertEqual(len(permit.reserved)-len(permit.settled),1)


class Pressure(unittest.TestCase):
 def snapshot(self,host,cgroup):return dict(effective_available_bytes=10**9,events={'cg':{'oom':'0','oom_kill':'0'}},pressure={'host':'full avg10='+str(host)+' avg60=0 avg300=0 total=0','cg':'full avg10='+str(cgroup)+' avg60=0 avg300=0 total=0'})
 def test_unchanged_pressure_boundary_and_raw_evidence(self):
  baseline=self.snapshot(0,0)
  for h,cg,group in [(1,0,'host'),(.5,.001,'cg')]:
   current=self.snapshot(h,cg);reason=observer.resource_reason(current,baseline,1);self.assertEqual(reason,'MEMORY_PRESSURE')
   v=observer.resource_failure(current,baseline,1,reason);self.assertEqual(v['raw_current'],current);self.assertEqual(v['pressure_hits'][0]['group'],group)
  self.assertIsNone(observer.resource_reason(self.snapshot(.999,0),baseline,1))
 def test_OOM_and_headroom_keep_actual_snapshot(self):
  base=self.snapshot(0,0);current=deepcopy(base);current['events']['cg']['oom_kill']='1'
  reason=observer.resource_reason(current,base,1);self.assertEqual(reason,'OOM_INCREMENT');self.assertEqual(observer.resource_failure(current,base,1,reason)['raw_current']['events']['cg']['oom_kill'],'1')
 def test_fresh_resource_failure_keeps_CPU_and_PSI_in_separate_records(self):
  cfg,g=fixture();CPU=[];resources=[];sample=self.snapshot(1,0);sample['effective_available_bytes']=10**15
  with patch.object(r,'verify_authorization',return_value=list(range(5))),patch.object(r,'sha',return_value='cfg'),patch.object(r,'protected'),patch.object(r,'topology',return_value=set(range(5))),patch.object(r,'measure_idle',return_value={'status':'PASS'}),patch.object(launcher,'snapshot',return_value=sample):
   with self.assertRaisesRegex(ValueError,'FRESH_RESOURCE:MEMORY_PRESSURE'):launcher.fresh(cfg,g,Path('/tmp/cfg'),cpu_record=CPU.append,resource_record=resources.append)
  self.assertEqual(len(CPU),1);self.assertEqual(len(resources),1);self.assertEqual(resources[0]['raw_current'],sample)
 def test_real_supervisor_writes_failure_snapshot_before_reaping(self):
  old=os.sched_getaffinity(0);cpu=min(old);os.sched_setaffinity(0,{cpu})
  try:
   with tempfile.TemporaryDirectory() as td:
    root=Path(td);(root/'output').mkdir();identity=list(pool.identity(os.getpid()))
    (root/'witness_started.json').write_text(json.dumps(dict(witness=dict(pid=os.getpid(),identity=identity),witness_cpu=cpu)))
    limits=deepcopy(c.RESOURCES);limits.pop('observer_AS_bytes');limits.update(AS_bytes=2**30,RSS_bytes=512*2**20,witness_AS_bytes=2**30,witness_RSS_bytes=512*2**20,admission_reserve_bytes=1,aggregate_RSS_bytes=512*2**20)
    count=[0]
    def probe():
     count[0]+=1;return self.snapshot(0 if count[0]==1 else 1,0)
    with self.assertRaisesRegex(RuntimeError,'MEMORY_PRESSURE'):observer.supervise([sys.executable,'-B','-c','import time;time.sleep(10)'],env=dict(os.environ),control=root/'control',output=root/'output',limits=limits,allowed_cpus=[cpu],probe=probe)
    receipt=json.loads((root/'control/resource_stop.json').read_text());terminal=json.loads((root/'control/launcher_terminal.json').read_text())
    self.assertEqual(receipt['pressure_hits'][0]['group'],'host');self.assertTrue(terminal['all_reaped']);self.assertFalse(terminal['cleanup_errors'])
  finally:os.sched_setaffinity(0,old)
 def test_marker_glob_reports_actual_axes(self):
  code=(ROOT/'scripts/run_h8_phase_b2_recovery.py').read_text();self.assertIn("glob('axis_*/*/compile_entered.json')",code);self.assertNotIn("glob('wave_*",code)


class SpoolAndParallel(unittest.TestCase):
 def test_QPY_write_limit_and_tamper_path_rejection(self):
  f=io.BytesIO();limited=parallel.LimitedFile(f,3);limited.write(b'abc')
  with self.assertRaises(ValueError):limited.write(b'd')
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'one.qpy';p.write_bytes(b'abc');receipt=dict(path=str(p),bytes=3,sha256=parallel.file_sha(p));qpy=Mock();qpy.load.return_value=['one']
   self.assertEqual(parallel.load_wrapper(receipt,td,qpy,3),'one');p.write_bytes(b'bad')
   with self.assertRaises(ValueError):parallel.load_wrapper(receipt,td,qpy,3)
   with self.assertRaises(ValueError):parallel.load_wrapper(receipt,str(Path(td)/'other'),qpy,3)
 def test_real_two_workers_overlap_and_cleanup_on_failure(self):
  cpus=sorted(os.sched_getaffinity(0))[:2]
  if len(cpus)<2:self.skipTest('two CPUs unavailable')
  limits=dict(max_workers=2,AS_bytes=512*2**20,RSS_bytes=256*2**20,worker_CPU_seconds=10,aggregate_CPU_seconds=30)
  for fail in (False,True):
   with tempfile.TemporaryDirectory() as td:
    root=Path(td)
    def task(value,path):
     (root/value['id']).touch();deadline=time.monotonic()+5
     while not all((root/x).exists() for x in ('a','b')):
      if time.monotonic()>deadline:raise ValueError('NOT_PARALLEL')
      time.sleep(.01)
     if fail and value['id']=='a':raise ValueError('ARTIFICIAL_FAILURE')
     return dict(both_alive=True)
    if fail:
     with self.assertRaises(RuntimeError):pool.map_owned([dict(id=x) for x in ('a','b')],task,cpus=cpus,limits=limits,directory=root/'pool',deadline=time.monotonic()+10)
    else:pool.map_owned([dict(id=x) for x in ('a','b')],task,cpus=cpus,limits=limits,directory=root/'pool',deadline=time.monotonic()+10)
    v=json.loads((root/'pool/parallel_receipt.json').read_text());self.assertTrue(v['all_reaped']);self.assertEqual(len(v['workers']),2);self.assertFalse(v['automatic_retry'])
 def test_first_model_failure_never_admits_sine(self):
  for fail in (False,True):
   with tempfile.TemporaryDirectory() as td:
    cfg,g=fixture();permit=r.Permit(cfg,g,td);permit.child_CPU_seconds=0.;calls=[]
    def compile_axis(task,built,path):return dict(case_id=task['case_id'],axis=task['axis'],measured_RZ=1 if fail else 1852640)
    def fake(tasks,callback,*,directory,reserve,**kwargs):
     self.assertEqual(len(tasks),1);self.assertEqual(len(kwargs['cpus']),1);task=tasks[0];calls.append(task['axis']);reserve(task);directory.mkdir();work=directory/task['id'];work.mkdir();p=work/'task_result.json';p.write_text(json.dumps(callback(task,work)))
     return dict(status='PASS',all_reaped=True,cleanup_errors=[],child_CPU_seconds=1,workers=[dict(task_id=task['id'],directory=str(work),exit_code=0,reaped=True,result_sha256=parallel.file_sha(p),identity=[1,2,3,4])])
    if fail:
     with self.assertRaisesRegex(ValueError,'AXIS_MODEL'):a.serial_axes(c.plan(),permit,r.Writer(td),lambda case:{},compile_axis,map_owned=fake)
     self.assertEqual(calls,['cosine'])
    else:
     rows=a.serial_axes(c.plan(),permit,r.Writer(td),lambda case:{},compile_axis,map_owned=fake);self.assertEqual(len(rows),2);self.assertEqual(calls,['cosine','sine'])


class DurableWitness(unittest.TestCase):
 def test_native_wait_survives_dispatch_parent_exit_and_signal_is_not_fabricated(self):
  cpu=min(os.sched_getaffinity(0))
  for child in ('import time;time.sleep(.3);raise SystemExit(7)','import os,signal;os.kill(os.getpid(),signal.SIGKILL)'):
   with tempfile.TemporaryDirectory() as td:
    code='import sys,os;sys.path.insert(0,'+repr(str(ROOT/'scripts'))+');from h8_phase_b2_recovery_witness import wait_once;wait_once([sys.executable,"-c",'+repr(child)+'],dict(os.environ),'+repr(td)+',limits={"apply_OS_limits":False,"wall_seconds":10},witness_cpu='+str(cpu)+',config_sha="artificial",grant_sha="artificial")'
    dispatcher='import subprocess,sys;subprocess.Popen([sys.executable,"-B","-c",'+repr(code)+'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)'
    subprocess.run([sys.executable,'-B','-c',dispatcher],check=True);deadline=time.monotonic()+10;p=Path(td)/'native_exit.json'
    while not p.exists() and time.monotonic()<deadline:time.sleep(.05)
    self.assertTrue(p.exists());value=json.loads(p.read_text());self.assertTrue(value['wait_completed']);self.assertEqual(value['native_exit_code'],7 if 'SystemExit' in child else -9)


if __name__=='__main__':unittest.main()
