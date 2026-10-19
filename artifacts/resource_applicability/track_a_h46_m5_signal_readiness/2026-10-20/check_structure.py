"""Static readiness checks only; no molecular arrays or scientific imports."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

A=Path(__file__).resolve().parent
ROOT=A.parents[3]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,value):
    assert value,name
    checks.append(dict(name=name,status='PASS'))

s=read(A/'summary.json');p=read(A/'scope_FIXED_execution_UNAPPROVED.json');scope=p['scope']
domain=read(ROOT/'references/h46_m5_signal_domain_v1.json')['rows']
check('88_unique_primary',len(domain)==len({r['id'] for r in domain})==88 and all(r['role']=='PRIMARY' for r in domain))
check('systems_40_48',sum(r['system']=='H4' for r in domain)==40 and sum(r['system']=='H6' for r in domain)==48)
check('PR24_DET64',sum(r['method']=='PR' for r in domain)==24 and sum(r['method']!='PR' for r in domain)==64)
check('q_fixed_1_2_4_8',{r['q'] for r in domain}=={1,2,4,8})
check('K4_outer32',all(r['K']==4 and r['r_policy']=='ABS_TIME_PER_OUTER' and r['r_value']==32 for r in domain if r['method']=='PR'))
check('domain_SHA',sha(ROOT/'references/h46_m5_signal_domain_v1.json')==scope['domain_sha256'])
check('new82_reuse6',scope['maximum_new_candidate_signal']==82 and scope['reused_candidate_signal']==6)
check('no_downstream_scope',all(scope[k]==0 for k in ['new_sampling','new_compile','new_circuit_state_action','SCF_DF_state_generation','new_full_H_reference']) and not any(scope[k] for k in ['cost_accounting','ranking','fit','optional20']))
check('no_execution_or_grant',s['scientific_execution_authorized'] is False and read(A/'grant_INVALID.json')['valid'] is False)
check('empirical_reference_unknown',scope['allowance_policy']['certified'] is False and scope['allowance_policy']['reference_absolute_error'] is None and not scope['allowance_policy']['compile_tolerance_used_for_shots'])
check('five_eps',scope['epsilons']==[.05,.01,.005,.001,.0005])
check('SOURCE_parent_plan',subprocess.check_output(['git','rev-parse',s['SOURCE']+'^'],cwd=ROOT,text=True).strip()==s['base_commit'])
closure=read(A/'source_closure.json')
check('SOURCE_closure_bytes',all((ROOT/r['path']).read_bytes()==subprocess.check_output(['git','show',s['SOURCE']+':'+r['path']],cwd=ROOT) and sha(ROOT/r['path'])==r['sha256'] for r in closure['files']))
audit=read(A/'metadata_audit.json')
check('metadata_17_PASS_negative7',len(audit['checks'])==17 and all(r['status']=='PASS' for r in audit['checks']) and len(audit['negative_cases'])==7 and all(r['status']=='REJECTED' for r in audit['negative_cases']))
check('artificial14_regression16',all('Ran '+str(n)+' tests' in (A/f).read_text() and (A/f).read_text().rstrip().endswith('OK') for f,n in [('artificial_14.txt',14),('regression_16.txt',16)]))
bindings=read(A/'saved_input_and_signal_bindings.json')
check('six_reuse_original_evidence',sum(len(x['reuse']) for x in bindings.values())==6 and all(len(x['references'])==2 for x in bindings.values()))
check('tol_policy_observed_rank',all(x['binding']['rank_policy']=='tol_only_1e-8' and x['binding']['rank_is_saved_observation_not_generation_argument'] is True for x in bindings.values()) and [bindings[x]['binding']['rank'] for x in ['H4','H6']]==[10,19])
resources=read(A/'resources_and_STOP.json')
check('independent_STOP_no_PSI_kill',resources['no_global_worktree_freeze'] and resources['no_host_PSI_kill'] and resources['no_retry'] and resources['no_next_phase'] and not resources['resources']['runtime_caps'])
manifest=read(ROOT/'artifacts/validation_manifest.json')
check('manifest_entry',manifest['track_a_h46_m5_signal_readiness_20261020']==s)
old=json.loads(subprocess.check_output(['git','show',s['base_commit']+':artifacts/validation_manifest.json'],cwd=ROOT))
check('old_manifest_preserved',all(manifest.get(k)==v for k,v in old.items()))
report=ROOT/s['report'];body=report.read_text()
links=re.findall(r'\]\(([^)]+)\)',body)
check('report_links_resolve',all((report.parent/link).resolve().exists() for link in links if '://' not in link))
check('indexes_and_note',all(report.name in (ROOT/p).read_text() for p in ['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md','docs/research/研究ノート/2026-10-20_h46_m5_signal_readiness.md']))
check('mandatory_STOP_and_AD03',s['mandatory_STOP'] is True and s['AD03']=='DEFERRED' and s['valid_grants_issued']==0 and s['push_performed'] is False)
if (A/'inventory.json').exists():
    check('artifact_inventory',all(sha(A/r['path'])==r['sha256'] and (A/r['path']).stat().st_size==r['bytes'] for r in read(A/'inventory.json')))
print(json.dumps(dict(status='STRUCTURE_PASS_NOT_SCIENTIFIC_VALIDATION',checks=checks),indent=2))
