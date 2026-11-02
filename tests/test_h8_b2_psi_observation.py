"""New resource policy: diagnostic PSI, fail-closed actual memory and raw monitoring."""
from copy import deepcopy
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import h8_phase_b2_recovery_observer as o
import h8_phase_b2_recovery_contract as c

def snapshot(host=0.,local=0.):
 return dict(effective_available_bytes=1000,events={'local':{'oom':'0','oom_kill':'0','oom_group_kill':'0'}},pressure={'host':f'full avg10={host} avg60=0 avg300=0 total=100','local':f'full avg10={local} avg60=0 avg300=0 total=100'})

class PSI(unittest.TestCase):
 def test_nonzero_PSI_does_not_restart_or_change_OS_controls(self):
  baseline=snapshot()
  for host,local in ((0.,.36),(.18,.36),(2.,5.)):
   current=snapshot(host,local);self.assertIsNone(o.resource_reason(current,baseline,100))
   receipt=o.resource_failure(current,baseline,100,None);self.assertFalse(receipt['PSI_is_STOP_gate']);self.assertEqual(receipt['raw_current'],current);self.assertFalse(receipt['thresholds_unchanged']);self.assertTrue(receipt['OOM_and_headroom_STOPS_preserved'])
 def test_headroom_and_OOM_still_STOP(self):
  baseline=snapshot();current=snapshot();current['effective_available_bytes']=99
  self.assertEqual(o.resource_reason(current,baseline,100),'MEMORY_HEADROOM')
  for key in ('oom','oom_kill','oom_group_kill'):
   current=snapshot();current['events']['local'][key]='1';self.assertEqual(o.resource_reason(current,baseline,100),'OOM_INCREMENT')
 def test_invalid_or_missing_observations_fail_closed(self):
  baseline=snapshot()
  for raw in ('full avg10=nan total=0','full avg10=-1 total=0','some avg10=0 total=0','full avg10=101 total=0'):
   current=snapshot();current['pressure']['local']=raw;self.assertEqual(o.resource_reason(current,baseline,100),'PSI_OBSERVATION_UNKNOWN')
  current=snapshot();del current['events']['local'];self.assertEqual(o.resource_reason(current,baseline,100),'CGROUP_OBSERVATION_CHANGED')
 def test_live_owned_supervisor_records_PSI_without_suppressed_OOM_guard(self):
  import json,os,tempfile
  import track_a_phase_a_parallel as owned
  cpu=min(os.sched_getaffinity(0));old=os.sched_getaffinity(0);os.sched_setaffinity(0,{cpu})
  try:
   with tempfile.TemporaryDirectory() as td:
    root=Path(td);output=root/'output';output.mkdir()
    (root/'witness_started.json').write_text(json.dumps(dict(witness=dict(pid=os.getpid(),identity=list(owned.identity(os.getpid()))),witness_cpu=cpu)))
    limits=deepcopy(c.RESOURCES);limits.pop('observer_AS_bytes');limits.update(witness_AS_bytes=2**31,witness_RSS_bytes=2**30)
    def probe():
     value=snapshot(.18,.36);value['effective_available_bytes']=10**15;return value
    command=[sys.executable,'-B','-c','import json,time;from pathlib import Path;time.sleep(2.2);Path('+repr(str(output/'terminal.json'))+').write_text(json.dumps(dict(status="COMPLETE_MANDATORY_STOP")))']
    o.supervise(command,env=dict(os.environ),control=root/'control',output=output,limits=limits,allowed_cpus=[cpu],probe=probe)
    rows=[json.loads(s) for s in (root/'control/observations.jsonl').read_text().splitlines()];self.assertGreaterEqual(len(rows),2)
    self.assertTrue(all(row['pressure_observation'][1]['full_avg10']==.36 for row in rows));self.assertTrue(all('raw_resource_snapshot' in row for row in rows))
    terminal=json.loads((root/'control/launcher_terminal.json').read_text());self.assertEqual(terminal['status'],'COMPLETE_MANDATORY_STOP');self.assertTrue(terminal['all_reaped'])
  finally:os.sched_setaffinity(0,old)
 def test_policy_and_scientific_error_bound_are_sealed_separately(self):
  self.assertEqual(c.RESOURCES['memory_pressure_policy'],'PSI_OBSERVATION_ONLY_OOM_HEADROOM_AND_OS_LIMITS_ENFORCED');self.assertEqual(len(c.AXES),1);self.assertEqual(c.AXES[0]['axis'],'sine')
  self.assertFalse(c.axis_check('1852640',1)['PASS']);self.assertEqual(c.CAPS['compile'],1);self.assertFalse(c.plan()['shared_OS_cgroup_scheduler_changes'])

if __name__=='__main__':unittest.main()
