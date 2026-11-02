"""Owned-process startup and fixed-window accounting; no molecular calculation."""
import json,os,signal,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import h8_phase_b2_recovery_runtime as r
import h8_phase_b2_recovery_witness as w

class Accounting(unittest.TestCase):
 def test_PID_reuse_or_missing_sample_never_claims_zero_usage(self):
  a=dict(pid=1,start_ticks=2,uid=3,user_ticks=10,system_ticks=4,available=True)
  b=dict(a,user_ticks=17,system_ticks=6)
  v=r.process_cpu_delta(a,b);self.assertEqual(v['user_ticks_delta'],7);self.assertEqual(v['system_ticks_delta'],2)
  for bad in (dict(b,start_ticks=9),dict(pid=1,available=False)):
   v=r.process_cpu_delta(a,bad);self.assertFalse(v['identity_unchanged']);self.assertNotIn('user_ticks_delta',v)
 def test_busy_owned_process_is_recorded_but_never_subtracted(self):
  before={1:[0,0,0,0,0,0,0,0]};after={1:[6,0,0,94,0,0,0,0]}
  with patch.object(r,'cpu_counters',side_effect=[before,after]),patch.object(r.time,'sleep'):
   v=r.measure_idle({1},process_pids=[os.getpid(),os.getppid()])
  self.assertEqual(v['status'],'STOP');self.assertFalse(v['own_activity_subtracted']);self.assertFalse(v['process_accounting_used_to_relax_gate']);self.assertEqual(len(v['owned_processes']),2)
 def test_exact_five_percent_boundary_is_preserved(self):
  for busy,expected in ((5,'PASS'),(6,'STOP')):
   v=r.idle_receipt({1:[0]*8},{1:[busy,0,0,100-busy,0,0,0,0]},{1});self.assertEqual(v['status'],expected)

class Startup(unittest.TestCase):
 def test_observer_pinned_before_first_child_instruction_and_witness_idle(self):
  cpus=sorted(os.sched_getaffinity(0))[:2]
  if len(cpus)<2:self.skipTest('two CPUs unavailable')
  monitor,witness=cpus
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   child='''import os,sys,time,json
from pathlib import Path
sys.path[:0]=[sys.argv[1],sys.argv[2]]
root=Path(sys.argv[3]);assert os.sched_getaffinity(0)=={int(sys.argv[4])}
from h8_phase_b2_recovery_witness import wait_witness_idle
from h8_phase_b2_recovery_runtime import measure_idle
wait_witness_idle({'control':str(root/'control')},{'witness_cpu':int(sys.argv[5])})
v=measure_idle(set(map(int,sys.argv[4:6])),process_pids=[os.getpid(),os.getppid()])
(root/'sample.json').write_text(json.dumps(v))
raise SystemExit(7)
'''
   code='import os,sys;sys.path[:0]='+repr([str(ROOT/'scripts'),str(ROOT/'src/trotterlib')])+';from h8_phase_b2_recovery_witness import wait_once;wait_once('+repr([sys.executable,'-B','-c',child,str(ROOT/'scripts'),str(ROOT/'src/trotterlib'),td,str(monitor),str(witness)])+',dict(os.environ),'+repr(td)+',limits={"apply_OS_limits":False,"wall_seconds":10},witness_cpu='+str(witness)+',observer_cpu='+str(monitor)+',allowed_cpus='+repr(cpus)+',config_sha="artificial",grant_sha="artificial")'
   subprocess.run([sys.executable,'-B','-c',code],check=True)
   n=json.loads((root/'native_exit.json').read_text());self.assertEqual(n['native_exit_code'],7);self.assertTrue(n['wait_completed'])
   v=json.loads((root/'sample.json').read_text());self.assertFalse(v['own_activity_subtracted']);self.assertEqual(len(v['owned_processes']),2)
   parent=[x for x in v['owned_processes'] if x['before']['pid']!=n['pid']][0]
   self.assertTrue(parent['identity_unchanged']);self.assertLessEqual(parent['user_ticks_delta']+parent['system_ticks_delta'],5)
 def test_signal_interrupts_blocking_wait_and_reaps_owned_child(self):
  cpu=min(os.sched_getaffinity(0))
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);code='import os,sys;sys.path[:0]='+repr([str(ROOT/'scripts'),str(ROOT/'src/trotterlib')])+';from h8_phase_b2_recovery_witness import wait_once;wait_once([sys.executable,"-c","import time;time.sleep(60)"],dict(os.environ),'+repr(td)+',limits={"apply_OS_limits":False,"wall_seconds":10},witness_cpu='+str(cpu)+',config_sha="artificial",grant_sha="artificial")'
   p=subprocess.Popen([sys.executable,'-B','-c',code]);deadline=time.monotonic()+10
   try:
    while not (root/'witness_idle_ready.json').exists() and p.poll() is None:
     if time.monotonic()>deadline:self.fail('ready timeout')
     time.sleep(.01)
    p.send_signal(signal.SIGTERM);p.wait(timeout=15)
    v=json.loads((root/'native_exit.json').read_text());self.assertTrue(v['wait_completed']);self.assertTrue(v['witness_stop_requested']);self.assertEqual(v['native_exit_code'],-15)
   finally:
    if p.poll() is None:p.kill();p.wait()
 def test_handshake_missing_is_technical_STOP(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaisesRegex(ValueError,'TIMEOUT'):w.wait_witness_idle({'control':str(Path(td)/'control')},{'witness_cpu':0},timeout=0)

if __name__=='__main__':unittest.main()
