from pathlib import Path
import json,sys,hashlib
B=Path('/home/abe/Project/Partially Randomized Trotter/.rpe-central');ROOT=B/'releases/95a74e448-pbs-current';sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks.rpe_operation_client import operate
from trottertracks.rpe_q3_store import JobStore
from check_rpe_distributed_q3 import launch_markers
old=B/'acceptance/price-fleet-v2-e917165ba';state=old/'gateway/state';index=json.loads((state/'RUNS.json').read_text());assert len(index)==1
run_id=next(iter(index));run=Path(index[run_id]['run_directory']);config=json.loads((run/'CONTROLLER_CONFIG.json').read_text());assert config['release']['id']=='romutail-price-fleet-e917165ba'
assert json.loads((run/'EXIT.json').read_text())['returncode']==0 and(old/'CONTROLLER_COMPLETE.json').exists()
store=JobStore(old/'run/distributed/QUEUE.sqlite3')
def snapshot():
 return dict(events=store.events(),jobs=[(r['job_id'],r['attempt_id'],r['state'],r['completion_sha256'])for r in store.rows()],launches={k:hashlib.sha256(v).hexdigest()for k,v in launch_markers(old).items()},frozen_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in(run/'CONTROLLER_CONFIG.json',run/'PROBLEM.json',run/'ROUND.json',run/'REQUEST.json')})
before=snapshot();bootstrap=B/'validation/pbs-current-integration-20261118/OLD_ARTIFICIAL_BOOTSTRAP.json';assert not bootstrap.exists();bootstrap.write_text(json.dumps(dict(release_registry=str(B/'RELEASES.json'),state_directory=str(state)),indent=2)+'\n')
client=dict(transport='local',host_id='local',python=str(B.parent/'.venv311/bin/python'),remote_entry=str(ROOT/'tools/local0-search/remote.py'),server_config=str(bootstrap))
status=operate(client,'status',run_id=run_id);resume=operate(client,'resume',run_id=run_id)
assert status['release']['id']==resume['release']['id']=='romutail-price-fleet-e917165ba'
assert status['controller_source']==resume['controller_source']=='e917165ba1f36b7627614cd25e88b5108d9b20b9'
assert resume['status']=='REUSED_COMPLETED'and snapshot()==before
result=dict(status='PASS',global_current=json.loads((B/'RELEASES.json').read_text())['current'],existing_artificial_run=run_id,old_release=resume['release']['id'],old_controller_source=resume['controller_source'],resume_status=resume['status'],same_scientific_source=resume['computation_source'],queue_events_attempts_results_frozen_inputs_and_launches_unchanged=True,additional_launches=0,additional_adoptions=0,additional_pbs_submissions=0,new_scientific_calls=0,bootstrap=str(bootstrap),note='New bootstrap references adopted global registry and existing completed artificial state; original SERVICE/run files unchanged')
out=Path('/home/abe/Project/Partially Randomized Trotter/.worktrees/rpe-romutail-controller-migration/docs/research/rpe_pbs_current_integration_20261118/checks/OLD_PINNED_RESUME.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
