"""Execution receipt and read-only profiling around one unchanged reviewed runner."""
import pathlib,json,sys,os,time,resource,contextlib,runpy,traceback,hashlib,datetime,importlib.abc
root=pathlib.Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/pr2-v4-s2-parallelization-20260928')
out=pathlib.Path(pathlib.Path('/tmp/ax1b_latest_launch_directory.txt').read_text().strip())
assert not (out/'launch_receipt.json').exists(), 'One launch already recorded; no retry'
assert json.loads((out/'final_binding_audit.json').read_text())['status']=='FINAL_BINDING_PASS_READY_FOR_ONE_EXPLICIT_LAUNCH'
hash_value=(out/'authorization_canonical_sha256.txt').read_text().strip()
runner=root/'scripts/resource_applicability/run_track_a_ax1b.py'
args=[str(runner),'--execute-saved-analysis','--authorization',str(out/'execution_authorization_v1.json'),'--launch-authorization-sha256',hash_value,
 '--preparation-manifest',str(root/'artifacts/resource_applicability/track_a_ax1b_identity_compatibility/2026-10-09/identity_compatibility_manifest_v1.json')]
receipt=dict(schema_version='track_a_ax1b_one_launch_receipt_v1',launch_count=1,source_commit='fc297cd9ab840018c4f35b2764d0b8be07c57285',
 authorization_canonical_sha256=hash_value,runner_arguments=args,python_executable=sys.executable,wrapper_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),instrumentation_scope='Read-only function-entry/status/path counts; no source or gate replacement',
 mandatory_stop=True,next_stage_authorized=False)
(out/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
counters={};successful_inputs=[];fit_statuses=[];events=[]
def profile(frame,event,arg):
 module=frame.f_globals.get('__name__','');name=frame.f_code.co_name
 if not module.startswith('trottertracks.resource_applicability.ax1b_'):return
 tracked={ 'execute','authorize','_limits','read','project_saved','analyze','fit_cost','finite_normalization','paired_statistics','selection','common_support_selection' }
 if name not in tracked:return
 key=module.rsplit('.',1)[-1]+'.'+name
 if event=='call':
  counters[key]=counters.get(key,0)+1
  if name in {'execute','_limits','project_saved','analyze','fit_cost'}:events.append(dict(function=key,event='entered'))
 elif event=='return':
  if module.endswith('.ax1b_data') and name=='read' and arg is not None:successful_inputs.append(frame.f_locals.get('path'))
  if name=='fit_cost' and arg is not None:fit_statuses.append(dict(model_id=frame.f_locals.get('model_id'),status=getattr(arg,'status',None)))
class ScienceGenerationBarrier(importlib.abc.MetaPathFinder):
 def __init__(self):self.attempts=0
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'trotterlib','qiskit','openfermion','openfermionpyscf','pyscf','cupy'}:
   self.attempts+=1;raise RuntimeError('New science/GPU import forbidden in AX1b saved analysis')
barrier=ScienceGenerationBarrier();sys.meta_path.insert(0,barrier)
started=time.monotonic();code=1;unhandled=None
os.chdir(root);sys.argv=args
with (out/'runner_stdout.txt').open('x') as std,(out/'runner_stderr.txt').open('x') as err:
 with contextlib.redirect_stdout(std),contextlib.redirect_stderr(err):
  try:
   sys.setprofile(profile)
   runpy.run_path(str(runner),run_name='__main__')
   code=0
  except SystemExit as exc:code=exc.code if type(exc.code) is int else 1
  except BaseException as exc:unhandled=repr(exc);traceback.print_exc();code=1
  finally:sys.setprofile(None)
result=dict(schema_version='track_a_ax1b_one_launch_process_audit_v1',launch_count=1,exit_code=code,wall_seconds_including_authorization=time.monotonic()-started,
 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,processes=1,function_entry_counts=counters,successful_allowlisted_reads=successful_inputs,
 completed_fit_statuses=fit_statuses,stage_events=events,forbidden_science_generation_import_attempts=barrier.attempts,unhandled_exception=unhandled,
 finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),mandatory_stop=True,next_stage_authorized=False)
(out/'process_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(exit_code=code,launch_count=1,function_entry_counts=counters,max_rss_bytes=result['max_rss_bytes'],wall_seconds=result['wall_seconds_including_authorization'],runner_stdout=(out/'runner_stdout.txt').read_text())))
sys.exit(code)
