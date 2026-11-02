"""Static STOP/fix evidence validation. No numerical molecular processing."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

A=Path(__file__).resolve().parent;ROOT=A.parents[3]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,value):
    assert value,name
    checks.append(dict(name=name,status='PASS'))

s=read(A/'summary.json');completion=read(A/'completion_witness.json');exit=read(A/'external_exit_witness.json')
check('run_STOP',completion['status']=='STOP' and s['actually_completed_candidate_signal']==s['candidate_ledger_entries']==0)
check('workers_reaped_and_STOP_not_hidden_by_exit0',all(x['exit']==0 and x['reaped'] for x in completion['workers'].values()) and set(completion['errors'])=={'H4','H6'})
check('launcher_exit1_reaped',exit['native_exit']==1 and exit['reaped'])
check('no_external_resource_stop',not exit['forwarded_signals'] and all(read(A/(x+'_STOP.json'))['reason']=='INVALID_GENERATION_PREFIX' for x in ('H4','H6')))
check('fresh2_systems',read(A/'fresh_gate.json')['system_workers']==2)
check('approval_one_shot',read(A/'execution_approval_projection.json')['grant_consumed'] and read(A/'execution_approval_projection.json')['approval']['execution_authorized'])
check('no_retry_next_phase',not completion['retry'] and not completion['next_phase_started'] and not s['retry_started'])
check('no_compile_sampling_reference',s['new_compile']==s['new_sampling']==s['new_reference']==0)
check('no_cost_ranking',not s['cost_accounting'] and not s['ranking'])
p=read(A/'fix_plan_UNAPPROVED_projection.json');grant=read(A/'fix_grant_INVALID.json')
check('fix_grant_invalid',not grant['valid'] and not p['authorized'] and not s['fix_execution_authorized'])
check('same88_scope',p['scope']==read(A/'execution_approval_projection.json')['approval']['scope'] and p['scope']['signal_targets']==88 and p['scope']['maximum_new_candidate_signal']==82 and p['scope']['reused_candidate_signal']==6)
check('empirical_reference_null',p['scope']['allowance_policy']['reference_absolute_error'] is None and not p['scope']['allowance_policy']['certified'])
check('fix_SOURCE',p['SOURCE']==s['fix_SOURCE']==read(A/'fix_source_closure.json')['SOURCE'])
check('SOURCE_git_bytes',all((ROOT/r['path']).read_bytes()==subprocess.check_output(['git','show',s['fix_SOURCE']+':'+r['path']],cwd=ROOT) and sha(ROOT/r['path'])==r['sha256'] for r in read(A/'fix_source_closure.json')['files']))
check('fix_failure_reproduced',all(x in (A/'regression_before_fix_expected_failure.txt').read_text() for x in ['FAILED (failures=1, errors=1)','INVALID_GENERATION_PREFIX']))
check('fix18_PASS','Ran 18 tests' in (A/'artificial_18_after_fix.txt').read_text() and (A/'artificial_18_after_fix.txt').read_text().rstrip().endswith('OK'))
audit=read(A/'metadata_audit.json')
check('metadata37_PASS',len(audit['checks'])==37 and all(x['status']=='PASS' for x in audit['checks']))
preserve=read(A/'preservation.json')
check('preservation_2899_18',preserve['prior_protected_files']==2899 and preserve['stopped_run_files']==18 and not preserve['changed_files'] and preserve['old_SOURCE_untouched'])
manifest=read(ROOT/'artifacts/validation_manifest.json');check('new_manifest',manifest['track_a_h46_m5_signal_run01_stop_and_fix_20261020']==s)
old=json.loads(subprocess.check_output(['git','show',s['base_commit']+':artifacts/validation_manifest.json'],cwd=ROOT));check('old_manifest_preserved',all(manifest.get(k)==v for k,v in old.items()))
report=ROOT/s['report'];body=report.read_text()
check('report_links',all((report.parent/x).resolve().exists() for x in re.findall(r'\]\(([^)]+)\)',body) if '://' not in x))
check('indexes',all(report.name in (ROOT/x).read_text() for x in ['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md','docs/research/研究ノート/2026-10-20_h46_m5_signal_readiness.md']))
check('STOP_AD03_no_push',s['mandatory_STOP'] and s['AD03']=='DEFERRED' and not s['push_performed'])
if (A/'inventory.json').exists():
    check('artifact_inventory',all(sha(A/r['path'])==r['sha256'] and (A/r['path']).stat().st_size==r['bytes'] for r in read(A/'inventory.json')))
print(json.dumps(dict(status='STRUCTURE_PASS_NOT_SCIENTIFIC_RERUN',checks=checks),indent=2))
