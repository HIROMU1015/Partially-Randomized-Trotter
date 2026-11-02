"""Exit races and immutable complete-only receipt. No molecular/transpile work."""
import json,sys,types,tempfile,copy
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks.resource_applicability.h7_open_shell import monitor as mon, resume as res, contract as c
from trottertracks.resource_applicability.h5_open_shell.execution import write_json,file_hash
PRIVATE=Path('/home/AbeHiromu/projects/h7-handoff-evidence/20261016/resume01-preparation')


class MonitorTests(unittest.TestCase):
 def child(self,exited):return types.SimpleNamespace(pid=42,identity=dict(pid=42,start_ticks=99),exited=exited)
 def identity(self,pid):return dict(pid=pid,start_ticks=99)
 def run_missing(self,reason,exit_at=None):
  ticks=[0.];waits=[]
  def pause(dt):ticks[0]+=dt;waits.append(dt)
  def missing(pid):raise ValueError(reason)
  child=self.child(lambda:exit_at is not None and ticks[0]>=exit_at)
  return child,dict(identity=self.identity,snapshot=missing,clock=lambda:ticks[0],pause=pause),ticks,waits
 def test_missing_memory_before_waitable_exit(self):
  child,kw,t,w=self.run_missing('PROCESS_MEMORY_MISSING',.03)
  self.assertIsNone(mon.owned_snapshot(child,**kw));self.assertGreaterEqual(t[0],.03);self.assertLess(t[0],.2)
 def test_threads_disappear_before_waitable_exit(self):
  child,kw,t,w=self.run_missing('PROCESS_THREADS_MISSING',.02)
  self.assertIsNone(mon.owned_snapshot(child,**kw))
 def test_live_missing_still_fails_closed(self):
  child,kw,t,w=self.run_missing('PROCESS_MEMORY_MISSING')
  with self.assertRaisesRegex(ValueError,'PROCESS_MEMORY_MISSING'):mon.owned_snapshot(child,**kw)
  self.assertGreaterEqual(t[0],.2);self.assertLessEqual(t[0],.21)
 def test_other_error_is_not_hidden(self):
  child,kw,t,w=self.run_missing('CORRUPT_STATUS')
  with self.assertRaisesRegex(ValueError,'CORRUPT_STATUS'):mon.owned_snapshot(child,**kw)
  self.assertEqual(w,[])
 def test_identity_mismatch_is_not_success(self):
  child=self.child(lambda:False)
  with self.assertRaisesRegex(RuntimeError,'OWN_IDENTITY'):
   mon.owned_snapshot(child,identity=lambda p:dict(pid=p,start_ticks=100),snapshot=lambda p:{})
 def test_valid_observation_and_proven_exit(self):
  result=dict(memory=dict(VmRSS_bytes=12,VmSize_bytes=24),threads=1)
  self.assertEqual(mon.owned_snapshot(self.child(lambda:False),identity=self.identity,snapshot=lambda p:result),result)
  self.assertIsNone(mon.owned_snapshot(self.child(lambda:True),snapshot=lambda p:self.fail('must not sample exited child')))
 def test_supervisor_completion_stop_and_reap_all_slots(self):
  memory=dict(host_available_bytes=1000,effective_available_bytes=1000,oom={},full_psi_avg10={'host':0})
  limits=dict(headroom_bytes=1,worker_RSS_bytes=100,AS_bytes=100,aggregate_RSS_bytes=400,output_bytes=2**20,minimum_free_bytes=1)
  class Fake:
   def __init__(self,pid,directory,task,code):
    self.pid=pid;self.identity=dict(pid=pid,start_ticks=pid);self.directory=directory;self.task=task
    self.reaped=False;self.exit_code=None;self.code=code;self.closed=False
   def exited(self):return True
   def reap(self):self.reaped=True;self.closed=True;self.exit_code=self.code;return self.code
   def terminate(self):pass
   def close(self):self.closed=True
  for fail in (False,True):
   with tempfile.TemporaryDirectory(dir=PRIVATE,prefix='mock-supervisor-') as d:
    children=[]
    def spawn(directory,task,fn,cpu,limits):
     directory.mkdir();code=1 if fail and not children else 0
     write_json(directory/'terminal.json',dict(status='STOP' if code else 'COMPLETE'))
     write_json(directory/'result.json',dict(task=task))
     child=Fake(9000000+len(children),directory,task,code);children.append(child);return child
    with patch.object(mon.time,'sleep'):
     if fail:
      with self.assertRaisesRegex(RuntimeError,'H7_FAIL_CLOSED'):mon.supervise(d,[dict(id=i) for i in range(8)],None,[1,2,3,4],limits,memory,spawn_fn=spawn,observe=lambda:memory)
     else:
      result=mon.supervise(d,[dict(id=i) for i in range(8)],None,[1,2,3,4],limits,memory,spawn_fn=spawn,observe=lambda:memory)
      self.assertEqual(len(result['completed']),8)
    self.assertTrue(all(ch.reaped and ch.closed for ch in children))
    summary=res.read(Path(d)/'summary.json');self.assertEqual(summary['cleanup_errors'],[])
    if fail:self.assertIn('WORKER_STOP',summary['first_stop'])


def make_parent(parent):
 out=parent/'output01';out.mkdir();rank=2
 snapshot=parent/'synthetic-snapshot';snapshot.write_bytes(b'artificial frozen bytes, no science arrays')
 projection=dict(actual_rank=rank);snap=dict(path=str(snapshot),sha256=file_hash(snapshot),arrays={})
 original=c.digest(dict(snapshot=snap['sha256'],projection=projection,source=res.ORIGIN))
 initial=c.tasks(rank,res.ORIGIN,original);eligible=[t['cell']['id'] for t in initial if t['cell']['R']]
 extension=c.tasks(rank,res.ORIGIN,original,extension=True,eligible=eligible)
 profile={'fixture':'metadata only','files':{}}
 manifest=dict(SOURCE=res.ORIGIN,approved=True,runtime_authorization=True,profile=profile,profile_digest=c.digest(profile),
   carry_digest='3'*64,target_digest=c.digest(c.target()),compiler_hash=c.COMPILER_HASH,contract_digest=c.digest(c.draft()),
   output=str(out),source_hashes={'src/trottertracks/resource_applicability/h7_open_shell/science.py':file_hash(ROOT/'src/trottertracks/resource_applicability/h7_open_shell/science.py')})
 grant=dict(kind='H7_NEUTRAL_OPEN_SHELL_ONE_SHOT_V1',SOURCE=res.ORIGIN,approved=True,runtime_authorization=True,
   manifest_digest=c.digest(manifest),maximum_compile=40,one_shot=True,automatic_retry=False,
   **{k:manifest[k] for k in ('profile_digest','carry_digest','target_digest','output')})
 write_json(parent/'manifest.json',manifest);write_json(parent/'authorization.json',grant)
 retired=[];next_id=[9100000]
 def marker(task,code=0):
  next_id[0]+=1;identity=dict(pid=next_id[0],start_ticks=next_id[0],session=9100000)
  retired.append(dict(pid=identity['pid'],identity=identity,reaped=True,exit_code=code))
  return dict(status='COMPLETE',task=task,identity=identity)
 directory=out/'input/group_000';directory.mkdir(parents=True)
 input_task=dict(stage='input',SOURCE=res.ORIGIN)
 saved=dict(task=input_task,snapshot=snap,DF=projection,state_checks=dict(doublet_engineering_check=True,S_squared_target_residual=0.),
  calls_attempted=dict.fromkeys(c.call_limits(rank),0),signal_records=[dict(cell=t['cell'],precision_rows={'0.05':{'1.0':{'cost_acquisition_eligible':True}}}) for t in initial])
 write_json(directory/'result.json',saved);write_json(directory/'terminal.json',marker(input_task))
 write_json(out/'realized_input_binding.json',dict(SOURCE=res.ORIGIN,snapshot=snap,projection_digest=c.digest(projection),input_digest=original,
  profile_digest=manifest['profile_digest'],compiler_hash=c.COMPILER_HASH,initial_tasks=initial))
 for stage,registered in [('initial_heavy',[t for t in initial if t['heavy']]),('initial_normal',[t for t in initial if not t['heavy']]),('extension',extension)]:
  directory=out/stage;directory.mkdir();write_json(directory/'reservations.json',dict(tasks=registered))
  completed=[]
  for i,task in enumerate(registered):
   if stage=='extension' and i>=9:continue
   group=directory/f'group_{i:03d}';group.mkdir();task=dict(task,_cpu=0)
   if stage=='extension' and i>=6:marker(task,-9);continue
   used=dict.fromkeys(c.call_limits(rank),0);used.update(compile=2,trajectory=1 if task['cell']['R'] else 0,
    sampling_call=1 if task['cell']['R'] else 0,occurrence=task['cell']['R'] or 0,control_probe=4 if task['validate'] else 0)
   result=dict(task=task,events=None,event_digest=c.digest(None),calls_attempted=used,validation={'max_error':0.} if task['validate'] else None,
    rows=[dict(axis=axis,compiler_hash=c.COMPILER_HASH,event_digest=c.digest(None),fingerprint='a'*64,
      metrics=dict(RZ=1,CX=1,RZ_depth=1,CX_depth=1,depth=1,size=1)) for axis in ('cosine','sine')])
   write_json(group/'result.json',result);write_json(group/'terminal.json',marker(task))
   if stage!='extension' or i<5:completed.append(result)
  write_json(directory/'summary.json',dict(completed=completed))
 terminal=dict(SOURCE=res.ORIGIN,status='STOP',first_stop='PROCESS_MEMORY_MISSING',all_workers_reaped=True,cleanup_errors=[],retired=retired)
 write_json(out/'terminal.json',terminal)
 write_json(parent/'native_wait_v1.json',dict(SOURCE=res.ORIGIN,wait_completed=True,exit_code=1,wall_seconds=100.,identity=dict(pid=9199999,start_ticks=9199999,session=9199999)))
 return manifest


class ReceiptTests(unittest.TestCase):
 def fixture(self):
  temp=tempfile.TemporaryDirectory(dir=PRIVATE,prefix='mock-parent-');self.addCleanup(temp.cleanup);parent=Path(temp.name);make_parent(parent);return parent
 def test_late_complete_accepted_and_exact_original_remaining_seeds(self):
  d=res.admit_parent(ROOT,self.fixture());self.assertEqual((len(d['accepted']),len(d['remaining_tasks']),len(d['failed_dispatched'])),(12,8,3))
  self.assertEqual(sum(r['late_completion'] for r in d['accepted_references']),1)
  self.assertEqual(d['cumulative_caps'],dict(compile=46,trajectory=19,sampling_call=19,occurrence=38))
  self.assertEqual(d['seed_source'],res.ORIGIN);self.assertEqual(d['remaining_tasks'][0]['replica'],7)
 def test_failed_native_complete_marker_is_not_reused(self):
  parent=self.fixture();p=parent/'output01/terminal.json';d=res.read(p);d['retired'][1]['exit_code']=-9;p.write_text(json.dumps(d))
  with self.assertRaisesRegex(ValueError,'GROUP_NATIVE_EXIT'):res.admit_parent(ROOT,parent)
 def test_wrong_event_bytes_rejected(self):
  parent=self.fixture();p=parent/'output01/initial_heavy/group_000/result.json';d=res.read(p);d['event_digest']='b'*64;p.write_text(json.dumps(d))
  with self.assertRaisesRegex(ValueError,'EVENT_BYTES'):res.admit_parent(ROOT,parent)
 def test_snapshot_mutation_rejected(self):
  parent=self.fixture();(parent/'synthetic-snapshot').write_bytes(b'changed')
  with self.assertRaisesRegex(ValueError,'FROZEN_SNAPSHOT'):res.admit_parent(ROOT,parent)
 def test_old_cleanup_failure_rejected(self):
  parent=self.fixture();p=parent/'output01/terminal.json';d=res.read(p);d['cleanup_errors']=['synthetic denied'];p.write_text(json.dumps(d))
  with self.assertRaisesRegex(ValueError,'PARENT_STOP_OR_CLEANUP'):res.admit_parent(ROOT,parent)
 def test_new_resume_grant_and_mutation_gates(self):
  parent=self.fixture();scope=res.admit_parent(ROOT,parent)
  m=dict(SOURCE='1'*40,approved=True,runtime_authorization=True,profile_digest=scope['profile_digest'],carry_digest='2'*64,
    target_digest=c.digest(c.target()),compiler_hash=c.COMPILER_HASH,contract_digest=c.digest(c.draft()),output=str(parent/'fresh'),resume=scope)
  g=dict(kind='H7_NEUTRAL_OPEN_SHELL_RESUME_ONE_SHOT_V1',SOURCE=m['SOURCE'],approved=True,runtime_authorization=True,
    manifest_digest=c.digest(m),resume_digest=c.digest(scope),maximum_compile=16,cumulative_caps=scope['cumulative_caps'],one_shot=True,automatic_retry=False,
    **{k:m[k] for k in ('profile_digest','carry_digest','target_digest','output')})
  self.assertEqual(c.verify_grant(m,g,m['SOURCE']),Path(m['output']))
  for key,value in [('maximum_compile',18),('automatic_retry',True),('kind','H7_NEUTRAL_OPEN_SHELL_ONE_SHOT_V1'),('resume_digest','3'*64)]:
   with self.assertRaises(ValueError):c.verify_grant(m,dict(g,**{key:value}),m['SOURCE'])
  scope['remaining_tasks'][0]['seed']+=1
  with self.assertRaisesRegex(ValueError,'RESUME_PARENT_CHANGED'):res.load_resume(ROOT,m)


if __name__=='__main__':unittest.main()
