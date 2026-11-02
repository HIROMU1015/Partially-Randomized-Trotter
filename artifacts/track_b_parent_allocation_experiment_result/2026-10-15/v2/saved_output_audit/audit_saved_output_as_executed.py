"""Read-only audit of saved v2 values/provenance. No science functions called."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,subprocess,sys

if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(65536)
root=Path.cwd();sys.path.insert(0,str(root/'src'))
from trottertracks.algorithm_codesign.parent_experiment.launch import read_contract,check_identities,CONTRACT
from trottertracks.algorithm_codesign.parent_allocation_native import checked_entry
from trottertracks.algorithm_codesign.rte_development.evidence import digest,verify_chunks
c=read_contract(root);m=check_identities(root,c);d=root/c['execution']['result_directory']
def read(path):return json.loads(Path(path).read_text())
def git(*args,cwd=root):return subprocess.check_output(['git',*args],cwd=cwd)
a=read(root/c['execution']['authorization_path'])
t=read(d/'terminal.json');inner=read(d/'inner_terminal.json') if (d/'inner_terminal.json').exists() else None;outer=read(d/'outer_process_receipt.json')
stop=read(d/'STOP.json');marker=read(d/'one_shot_consumed.json');job=read(d/'ledger_job.json')
start=read(d/'ledger_job.started.json');inv=read(d/'inventory.json');freeze=read(d/'candidate_freeze.json')
source=a['source_commit'];auth=t['execution_HEAD'];checks=[]
def require(name,value):
 checks.append(dict(check=name,passed=bool(value)))
 if not value:raise AssertionError(name)
complete=t['status']=='PARENT_ALLOCATION_RESOURCE_MAP_COMPLETE_AWAITING_GPT_REVIEW'
identity={k:t[k] for k in ('source_commit','execution_HEAD','contract_sha256','authorization_sha256','source_manifest_sha256','review_sha256','tool_identity_sha256')}
require('execution HEAD is A2',git('rev-parse','HEAD').decode().strip()==auth)
require('A2 sole parent S2',git('show','-s','--format=%P',auth).decode().split()==[source])
require('A2 authorization-only direct child',set(git('diff','--name-only',source,auth).decode().splitlines())=={c['execution']['authorization_path'],c['execution']['optional_receipt_path']})
require('remote A2 matches HEAD',git('ls-remote','--heads','origin','refs/heads/'+git('branch','--show-current').decode().strip()).decode().split()[0]==auth)
require('no tracked source/contract/auth changes after A2',not git('diff','--name-only',auth).decode().strip())
require('saved source/contract/auth/manifest identity',all(all(v.get(k)==val for k,val in identity.items()) for v in (inner,outer,marker,job,start,read(d/'job_start_receipt.json'),read(d/'runtime_receipt.json')) if v is not None))
require('identity matches frozen files',t['contract_sha256']==digest(root/CONTRACT) and t['authorization_sha256']==digest(root/c['execution']['authorization_path']) and t['source_manifest_sha256']==digest(root/c['execution']['source_manifest']))
marker_hash=digest(d/'one_shot_consumed.json')
require('all marker bindings match',all(v.get('marker_sha256')==marker_hash for v in (t,inner,outer,job,start) if v is not None))
require('STOP-terminal-outer-inner hash chain',stop['terminal_sha256']==digest(d/'terminal.json') and t['outer_process_receipt_sha256']==digest(d/'outer_process_receipt.json') and outer['inner_terminal_sha256']==(digest(d/'inner_terminal.json') if inner else None))
require('inner/outer/terminal status agrees',t['status']==outer['effective_status'] and (inner['status']==outer['inner_status']==t['status'] if inner else outer['inner_status'] is None and not complete))
require('process exit consistent',outer['exit_code']==0 if complete else outer['exit_code']!=0 or bool(t['technical_reason']))
require('retry0 and mandatory STOP',all(v.get('retries')==0 and v.get('mandatory_STOP') is True for v in (t,inner,outer,stop,marker) if v is not None))
require('no next science and no prefix claim',t['next_science_authorized'] is False and stop['next_science_authorized'] is False and t['prefix_scientific_use'] is False and (inner is None or inner['prefix_scientific_use'] is False))
require('outer-ledger ownership saved',start['parent_pid']==job['parent_pid']==read(d/'job_start_receipt.json')['owner_pid'])
require('candidate and inventory freeze exists',freeze['frozen_before_new_price'] is True and inv['complete_before_new_prices'] is True)
require('planned scope within fixed caps',len(freeze['configurations'])<=c['caps']['configurations'] and inv['new_keys']<=c['caps']['new_synthesis_keys'] and inv['used_old_keys']<=c['caps']['old_unique_guards'] and inv['planned_root_matrix_checks']<=c['caps']['new_root_matrix_checks'])
oldguards=[]
for p in sorted((d/'old_guards').glob('*.json')):
 v=read(p);checked_entry(v)
 oldguards.append(dict(path=str(p.relative_to(d)),key=v['key'],ratio=v['spec']['ratio'],sequence_sha256=v['sequence_sha256'],operator_error_upper=v['operator_error_upper']))
require('old strict guard bindings unique and complete',len({v['key'] for v in oldguards})==len(oldguards)==inv['used_old_keys'])
fresh=[v for v in inv['keys'] if v['acquisition']=='NEW_ONCE'];worker_rows=[]
for path in sorted((d/'jobs').glob('key_[0-9][0-9][0-9].json')):
 j=read(path);s=read(path.with_suffix('.started.json'));p=read(path.with_suffix('.process_receipt.json'));r=read(path.with_suffix('.result.json'))
 token=path.with_suffix('.backend_call_consumed.json')
 require('worker '+path.stem+' job/source/freeze identity',all(j.get(k)==v for k,v in identity.items()) and j['parent_pid']==start['pid'] and j['inventory_sha256']==digest(d/'inventory.json') and j['candidate_freeze_sha256']==digest(d/'candidate_freeze.json') and j['marker_sha256']==marker_hash and fresh[j['job_index']]['key']==j['key'])
 require('worker '+path.stem+' raw result receipt identity',all(r.get(k)==v for k,v in j.items()) and p['key']==j['key'] and p['pid']==s['pid'] and p['result_sha256']==digest(path.with_suffix('.result.json')))
 require('worker '+path.stem+' one completed call/no retry',token.exists() and r['status']=='KEY_COMPLETE' and r['synthesis_calls']==1 and r['retries']==0 and p['exit_code']==0)
 v=r['saved'];require('worker '+path.stem+' sequence identity/strict guard',hashlib.sha256(v['sequence'].encode()).hexdigest()==v['sequence_sha256'] and Fraction(v['strict_operator_error_upper'])<=Fraction(c['primitive_error']) and v['error_pass'] is True and c['acquisition']['options']['up_to_phase'] is False)
 require('worker '+path.stem+' resource caps',p['peak_RSS_KiB']<=c['caps']['worker_RSS_MiB']*1024 and p['CPU_seconds']<=c['caps']['per_key_CPU_seconds'] and path.with_suffix('.stdout.log').stat().st_size<=c['caps']['per_key_log_bytes'] and path.with_suffix('.stderr.log').stat().st_size<=c['caps']['per_key_log_bytes'])
 worker_rows.append(dict(key=j['key'],ratio=j['ratio'],T_count=v['T_count'],Tdagger_count=v['Tdagger_count'],one_qubit_count=v['one_qubit_count'],global_W_count=v['global_W_count'],sequence_sha256=v['sequence_sha256'],strict_operator_error_upper=v['strict_operator_error_upper'],CPU_seconds=p['CPU_seconds'],peak_RSS_KiB=p['peak_RSS_KiB']))
require('fresh inventory complete once each',len(worker_rows)==len(fresh)==outer['backend_call_tokens'] and len({v['key'] for v in worker_rows})==len(worker_rows))
cache=read(d/'guarded_cache.json')
for v in cache.values():checked_entry(v)
require('guarded cache covers all inventory',set(cache)=={v['ratio'] for v in inv['keys']})
chunks_checked=0;evidence=None
if (d/'evidence_manifest.json').exists():
 evidence=read(d/'evidence_manifest.json')
 for entry in evidence['files']:
  path=d/entry['path'];require('evidence hash '+entry['path'],path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'])
  if entry['chunks']:
   require('lossless UTF8 chunks '+entry['path'],all(v['bytes']<=c['caps']['chunk_bytes'] for v in entry['chunks']) and verify_chunks(d,entry));chunks_checked+=len(entry['chunks'])
 if complete:require('completed evidence source identity',all(evidence.get(k)==v for k,v in identity.items()) and evidence['status']==t['status'])
root_checks=0;event_count=0;bindings_by_file={}
for path in sorted((d/'events').glob('config_*.json')):
 bindings=read(path);bindings_by_file[path.name]=bindings;event_count+=len(bindings)
 for item in bindings:
  if item.get('new_root') and 'new_validation' in item:
   root_checks+=2;v=item['new_validation']
   require('stored root phase/error diagnostic '+path.name+':'+str(item['index']),v['up_to_phase'] is False and v['independent_target'] is True and v['bias_certificate'] is False and v['ideal_error_diagnostic']<=c['matrix_tolerance'] and v['actual_error_diagnostic']<=float(Fraction(item['native_cost']['strict_native_error_upper']))+c['matrix_tolerance'])
require('saved event scope within caps',event_count<=c['caps']['event_bindings'] and root_checks<=c['caps']['new_root_matrix_checks'])
law_paths=sorted((d/'laws').glob('law_*.json'));row_paths=sorted((d/'rows').glob('row_*.json'))
require('saved law scope within caps',len(law_paths)<=c['caps']['laws'] and len(row_paths)<=c['caps']['laws'])
summary=None
if complete:
 maps=read(d/'resource_map.json');rows=maps['rows']
 require('complete law/event/terminal coverage',len(rows)==len(law_paths)==len(row_paths)==inner['laws'] and len(bindings_by_file)==inner['configurations']==len(freeze['configurations']) and inner['old_guards']==len(oldguards) and inner['new_synthesis_calls']==len(worker_rows) and inner['new_root_matrix_checks']==root_checks==inv['planned_root_matrix_checks'])
 require('complete map no retry/scientific adoption',maps['retries']==0 and maps['mandatory_STOP'] is True and maps['quantum_shots_executed']==0 and maps['scientific_adoption'] is False and maps['future_science_authorized'] is False)
 for ri,row in enumerate(rows):
  events=read(d/row['law']['path']);bindings=bindings_by_file[Path(row['saved_events']['path']).name];info=row['ledger']
  pairs=list(zip(events,bindings))
  require('law '+row['id']+' bound row/files',len(events)==len(bindings) and row==read(d/('rows/row_'+str(ri).zfill(2)+'.json')) and digest(d/row['law']['path'])==row['law']['sha256'] and digest(d/row['saved_events']['path'])==row['saved_events']['sha256'])
  q=[Fraction(e['proposal']) for e in events];cf=[Fraction(e['coefficient']) for e in events];w=[Fraction(e['weight']) for e in events]
  require('law '+row['id']+' actual dyadic support/weight/mean',sum(q)==1 and all(qi>0 and (qi.denominator & (qi.denominator-1))==0 and qi.denominator<=2**c['precision']['K'] and qi*wi==ci and ci==Fraction(item['event']['coefficient']) for qi,wi,ci,(_,item) in zip(q,w,cf,pairs)))
  require('law '+row['id']+' stored moment/range/normalization',Fraction(info['acceptance'])==1 and Fraction(info['coefficient_mass'])==sum(cf) and Fraction(info['weight_second_moment'])==sum(qi*wi*wi for qi,wi in zip(q,w)) and Fraction(info['range'])==max(abs(wi) for wi in w))
  per={key:sum(qi*item['native_cost'][key] for qi,(_,item) in zip(q,pairs)) for key in ('T','CX','1Q')}
  require('law '+row['id']+' stored native unit price',all(per[key]==Fraction(info['per_trial_native'][key]) for key in per))
  require('law '+row['id']+' stored native error no re-guard',sum(ci*Fraction(item['native_cost']['strict_native_error_upper']) for ci,(_,item) in zip(cf,pairs))==Fraction(info['weighted_native_error_upper']))
  if info['eligibility']=='ELIGIBLE':
   require('law '+row['id']+' stored sufficient N cost identity',type(info['N_per_axis']) is int and 0<info['N_per_axis']<=c['caps']['shot_cap_per_axis'] and info['prep_readout_slope']==2*info['N_per_axis'] and all(Fraction(info['expected_native_intercept'][key])==2*info['N_per_axis']*per[key] for key in per))
 summary=dict(comparisons={k:v for k,v in maps['comparisons'].items() if k!='affine_relations'},dropped_proposals=maps['dropped'],affine_relations=len(maps['comparisons']['affine_relations']),C_star=maps['C_star'])
else:
 require('incomplete output not used scientifically',t['coverage'] is None)
chunk_groups=[]
for path in sorted(d.rglob('*.json')):
 chunks=sorted(path.parent.glob(path.name+'.part*.txt'))
 if not chunks:continue
 entry=dict(path=str(path.relative_to(d)),bytes=path.stat().st_size,sha256=digest(path),chunks=[dict(path=str(p.relative_to(d)),bytes=p.stat().st_size,sha256=digest(p)) for p in chunks])
 require('all complete saved chunk identities '+entry['path'],all(v['bytes']<=c['caps']['chunk_bytes'] for v in entry['chunks']) and verify_chunks(d,entry))
 chunk_groups.append(entry)
partials=[str(p.relative_to(d)) for p in sorted(d.rglob('*.partial'))]
raw=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(d.rglob('*')) if p.is_file() and 'saved_output_audit' not in p.parts]
if complete:require('complete run resource limits',outer['resource_failure'] is None and sum(v['bytes'] for v in raw)<=c['caps']['output_bytes'] and outer['completed_tree_CPU_seconds']+outer['outer_process_CPU_seconds']<=c['caps']['total_CPU_seconds'] and outer['observed_peak_aggregate_RSS_bytes']<=c['caps']['aggregate_RSS_MiB']*1048576)
old_marker=root/'artifacts/track_b_parent_allocation_experiment_result/2026-10-15/v1/one_shot_consumed.json'
require('old v1 marker unchanged',digest(old_marker)=='257112f17667e42550b434ced6aea4cb926ebad939adbbba0e0c806d4e01ad7c')
parent=Path('/home/abe/Project/Partially Randomized Trotter');snap=read('/tmp/parent-allocation-v2-execution-parent-snapshot.json')
changed=[]
for path,v in snap['files'].items():
 p=parent/path
 if not p.is_file() or p.stat().st_size!=v['bytes'] or digest(p)!=v['sha256']:changed.append(path)
require('parent HEAD/index/status/files preserved',not changed and git('rev-parse','HEAD',cwd=parent).decode().strip()==snap['HEAD'] and digest(parent/'.git/index')==snap['index_sha256'] and git('status','--porcelain=v1','-z','--untracked-files=normal','--','.',':!.worktrees',cwd=parent).hex()==snap['status_hex'])
record=dict(status='SAVED_OUTPUT_PROVENANCE_AUDIT_PASS',classification=t['status'],technical_reason=t['technical_reason'],source_S2=source,authorization_A2=auth,branch=git('branch','--show-current').decode().strip(),contract_sha256=t['contract_sha256'],authorization_sha256=t['authorization_sha256'],source_manifest_sha256=t['source_manifest_sha256'],marker_sha256=marker_hash,checks=checks,checks_pass=len(checks),protected_files=len(read(root/c['execution']['protected_ledger'])['sha256']),critical_source_files=len(m['critical_sha256']),old_guard_receipts=oldguards,new_worker_receipts=worker_rows,new_synthesis_calls=len(worker_rows),planned_configurations=len(freeze['configurations']),saved_event_configurations=len(bindings_by_file),saved_event_bindings=event_count,saved_new_root_matrix_checks=root_checks,completed_laws=len(row_paths),partial_files=partials,resource_map_saved=(d/'resource_map.json').exists(),summary=summary,raw_files=raw,raw_file_count=len(raw),raw_total_bytes=sum(v['bytes'] for v in raw),complete_chunk_groups=chunk_groups,lossless_chunks_checked=chunks_checked,outer_process_resource=outer,parent_status_entries_protected=snap['status_entries'],parent_files_protected=len(snap['files']),operator_guard_reruns_in_audit=0,science_calls_in_audit=0,additional_forecasts_in_audit=0,quantum_shots_executed=0,retries=0,mandatory_STOP=True,scientific_adoption=False,next_science_authorized=False)
print(json.dumps(record,ensure_ascii=False,indent=2))
