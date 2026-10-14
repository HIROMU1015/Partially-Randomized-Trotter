"""Read-only identity/stage audit. No runner, guard, synthesis or forecast."""
from pathlib import Path
import json,hashlib,subprocess,sys
root=Path.cwd();sys.path.insert(0,str(root/'src'))
from trottertracks.algorithm_codesign.parent_experiment.launch import read_contract,check_identities
from trottertracks.algorithm_codesign.parent_allocation_native import checked_entry
from trottertracks.algorithm_codesign.rte_development.evidence import digest
from trottertracks.algorithm_codesign.parent_experiment.runner import COMPLETE,INCONCLUSIVE
c=read_contract(root);manifest=check_identities(root,c);d=root/c['execution']['result_directory'];a=json.loads((root/c['execution']['authorization_path']).read_text())
def read(name):return json.loads((d/name).read_text())
t=read('terminal.json');inner=read('inner_terminal.json');outer=read('outer_process_receipt.json');stop=read('STOP.json');marker=read('one_shot_consumed.json');job=read('ledger_job.json');start=read('ledger_job.started.json');inventory=read('inventory.json');freeze=read('candidate_freeze.json')
source=a['source_commit'];auth=t['execution_HEAD'];checks=[]
def require(name,value):
 checks.append(dict(check=name,passed=bool(value)))
 if not value:raise AssertionError(name)
require('execution_HEAD=A',subprocess.check_output(['git','rev-parse','HEAD']).decode().strip()==auth)
require('A sole parent S',subprocess.check_output(['git','show','-s','--format=%P',auth]).decode().split()==[source])
changed=subprocess.check_output(['git','diff','--name-only',source,auth]).decode().splitlines()
require('A authorization-only diff',set(changed)=={c['execution']['authorization_path'],c['execution']['optional_receipt_path']})
require('one source/contract/auth/manifest identity',all(v['source_commit']==source and v['execution_HEAD']==auth and v['contract_sha256']==digest(root/'artifacts/track_b_parent_allocation_experiment_source/2026-10-15/v1/execution_contract_v1.json') and v['authorization_sha256']==digest(root/c['execution']['authorization_path']) and v['source_manifest_sha256']==digest(root/c['execution']['source_manifest']) for v in (t,inner,outer,marker,job,start)))
require('marker hash and all runtime receipts',all(v['marker_sha256']==digest(d/'one_shot_consumed.json') for v in (t,inner,outer,job,start)))
require('STOP binds terminal',stop['terminal_sha256']==digest(d/'terminal.json'))
require('terminal binds outer',t['outer_process_receipt_sha256']==digest(d/'outer_process_receipt.json'))
require('outer binds inner',outer['inner_terminal_sha256']==digest(d/'inner_terminal.json'))
require('inner/outer/terminal technical status agree',t['status']==inner['status']==outer['inner_status']==outer['effective_status']==INCONCLUSIVE)
require('process exit1 consistent',outer['exit_code']==1)
require('retry0 and STOP',all(v.get('retries')==0 and v.get('mandatory_STOP') is True for v in (t,inner,outer,stop,marker)))
require('no next science',t['next_science_authorized'] is False and stop['next_science_authorized'] is False)
require('no successful prefix use',t['prefix_scientific_use'] is False and inner['prefix_scientific_use'] is False and t['coverage'] is None)
require('no resource cap failure',outer['resource_failure'] is None)
require('candidates frozen before new prices',freeze['frozen_before_new_price'] is True and inventory['complete_before_new_prices'] is True)
require('new key and root scope within contract',inventory['new_keys']<=c['caps']['new_synthesis_keys'] and inventory['used_old_keys']<=c['caps']['old_unique_guards'] and inventory['new_root_events']<=c['caps']['new_root_events'] and inventory['planned_root_matrix_checks']<=c['caps']['new_root_matrix_checks'])
guards=[]
for p in sorted((d/'old_guards').glob('*.json')):
 v=json.loads(p.read_text());checked_entry(v)
 guards.append(dict(path=str(p.relative_to(d)),key=v['key'],ratio=v['spec']['ratio'],sequence_sha256=v['sequence_sha256'],operator_error_upper=v['operator_error_upper'],guard_binding_sha256=v['guard_binding_sha256']))
require('old guards have unique stored bindings',len({v['key'] for v in guards})==len(guards)==inventory['used_old_keys'])
require('no synthesis worker job/result/reservation',not list((d/'jobs').glob('*.json')) and outer['backend_call_tokens']==0)
require('no comparison/event/law outputs',not (d/'resource_map.json').exists() and not (d/'rows').exists() and not (d/'laws').exists() and not (d/'events').exists())
require('outer PID carried into ledger identity','parent_pid' in job and inner['parent_pid']==job['parent_pid'])
try:dict(**{'parent_pid':17},parent_pid=23)
except TypeError as exc:reproduction=type(exc).__name__+': '+str(exc)
else:raise AssertionError('artificial duplicate-key reproduction did not fail')
require('observed reason matches duplicate-key semantics',t['technical_reason']==inner['reason']==reproduction)
require('no code/contract/auth changes after A',not subprocess.check_output(['git','diff','--name-only',auth]).decode().splitlines())
parent=Path('/home/abe/Project/Partially Randomized Trotter');snapshot=json.loads(Path('/tmp/parent-allocation-execution-parent-protection.json').read_text());changes=[]
for v in snapshot['files']:
 p=parent/v['path'];current=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
 if current!=v['sha256']:changes.append(v['path'])
require('parent HEAD/index/files preserved',not changes and subprocess.check_output(['git','rev-parse','HEAD'],cwd=parent).decode().strip()==snapshot['HEAD'] and hashlib.sha256((parent/'.git/index').read_bytes()).hexdigest()==snapshot['index_sha256'])
raw=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(d.rglob('*')) if p.is_file() and 'saved_output_audit' not in p.parts]
record=dict(status='SAVED_FAILURE_PROVENANCE_AUDIT_PASS',final_classification=INCONCLUSIVE,checks=checks,checks_pass=len(checks),source_S=source,authorization_A=auth,
 contract_sha256=t['contract_sha256'],source_manifest_sha256=t['source_manifest_sha256'],authorization_sha256=t['authorization_sha256'],marker_sha256=digest(d/'one_shot_consumed.json'),
 old_guard_receipts=len(guards),old_guard_checks_are_metadata_only=True,old_guard_receipt_index=guards,
 candidate_constructions_saved=len(freeze['configurations']),planned_inventory_unique_keys=len(inventory['keys']),planned_new_synthesis_keys=inventory['new_keys'],planned_root_matrix_checks=inventory['planned_root_matrix_checks'],
 actual_key_jobs=0,actual_synthesis_calls=0,actual_new_root_matrix_checks=0,completed_comparison_laws=0,quantum_shots_executed=0,
 coefficient_candidates_were_generated_only_in_authorized_run=True,prefix_does_not_support_scientific_win_or_loss=True,
 technical_reason=t['technical_reason'],failure_location='pipeline.acquire: dict(**identity,...,parent_pid=os.getpid(),...) before first key job write/Popen/backend',
 cause_confidence='HIGH: exact saved exception + parent_pid propagation + zero first job + ordinary artificial Python reproduction; no traceback was saved',
 new_source_needed_for_repair=True,old_one_shot_marker_consumed=True,no_retry=True,repair_applied=False,new_run_authorized=False,
 raw_result_files=raw,raw_file_count=len(raw),raw_bytes=sum(v['bytes'] for v in raw),all_raw_files_at_most_512KiB=all(v['bytes']<=524288 for v in raw),
 outer_process_resource=outer,protected_historical_files=len(json.loads((root/c['execution']['protected_ledger']).read_text())['sha256']),source_critical_files=len(manifest['critical_sha256']),
 parent_existing_entries_protected=len(snapshot['files']),scientific_adoption=False,mandatory_STOP=True)
print(json.dumps(record,ensure_ascii=False,indent=2))
