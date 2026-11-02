"""Static saved-result transcription checks; no signal/eligibility recalculation."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

A=Path(__file__).resolve().parent;ROOT=A.parents[3]
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
def check(name,value):
    assert value,name
    checks.append(dict(name=name,status='PASS'))
def csvread(name):
    with (A/name).open(newline='') as f:return list(csv.DictReader(f))
def csvvalues(rows):return [{k:'' if v is None else str(v) for k,v in row.items()} for row in rows]

s=read(A/'summary.json');collection=read(A/'signal_records.json');records=collection['records']
domain=read(ROOT/'references/h46_m5_signal_domain_v1.json')['rows']
expected={x['id']:x for x in domain}
check('normal_completion',s['status']=='SIGNAL_RUN02_COMPLETE_MANDATORY_STOP')
check('88_unique_records',len(records)==len({r['candidate']['id'] for r in records})==88)
check('exact_domain_order',[r['candidate'] for r in records]==domain)
check('new82_reuse6',sum(r['new_signal'] for r in records)==82 and sum(r['new_signal']==0 for r in records)==6)
check('H4_40_H6_48',Counter(r['candidate']['system'] for r in records)=={'H4':40,'H6':48})
check('PR24_DET64',Counter(r['candidate']['kind'] for r in records)=={'PR_SIGNAL':24,'DET_SIGNAL':64})
check('all_candidate_technical_PASS',all(r['status']=='SIGNAL_DIAGNOSTIC_PASS' for r in records))
check('SOURCE_plan_identity',collection['SOURCE']==s['SOURCE'] and all(r['SOURCE']==s['SOURCE'] and r['plan_sha256']==collection['plan_sha256'] for r in records))
check('saved_scope_zero_compile_sampling_cost_ranking',all(r['new_compile']==r['new_sampling']==0 and r['cost_accounting'] is False and r['ranking'] is False for r in records))
check('not_formally_adopted_no_strict_certificate',all(r['shot_allowance_formally_adopted'] is False and r['strict_bias_certificate'] is None for r in records))
axes=[];pairs=[];signals=[]
for data in records:
    row=data['candidate'];sig=data['signal'];e=data['empirical_eligibility']
    metadata={k:row[k] for k in ('id','system','input_series','formula','method','prefix','q','K','r_policy','r_value')}
    signals.append(dict(metadata,signal_origin=data['signal_origin'],new_signal=data['new_signal'],preparation_hash=data['preparation_hash'],source_signal_sha256=data['source_signal_sha256']))
    for key in ('reference','exact_tail','corrected','raw','total_signed_bias','outer_PF_or_discard_bias','finite_RTE_bias'):
        for axis in ('real','imag'):signals[-1][key+'_'+axis]=sig[key][axis]
    signals[-1].update(log_B=sig['log_B'],raw_corrected_vector_closure=sig['allowance_evidence']['raw_corrected_vector_closure'],maximum_intermediate_norm=sig['allowance_evidence']['maximum_intermediate_norm'],u=e['u'],log_B_margin=e['log_B_margin'])
    axes.extend(dict(metadata,**entry,signal_origin=data['signal_origin']) for entry in e['rows'])
    for epsilon in (.05,.01,.005,.001,.0005):
        statuses={r['axis']:r['status_under_proposed_empirical_policy'] for r in e['rows'] if r['epsilon']==epsilon}
        pairs.append(dict(metadata,epsilon=epsilon,real_status=statuses['real'],imag_status=statuses['imag'],conditional_on_saved_reference=True))
check('signal_csv_exact_saved_values',csvread('signals_88.csv')==csvvalues(signals))
check('880_axis_rows_exact_saved_values',len(axes)==880 and csvread('axis_eligibility_880.csv')==csvvalues(axes))
check('440_both_axis_labels_exact_saved_values',len(pairs)==440 and csvread('candidate_precision_status_440.csv')==csvvalues(pairs))
check('all_axes_shots_and_costs_null',all(x['shots'] is None and x['cost'] is None for x in axes))
check('all_axes_conditional_reference_null',all(x['conditional_on_saved_reference'] is True and x['reference_absolute_error'] is None and x['strict_bias_certificate'] is None for x in axes))
counts=Counter((r['system'],r['epsilon'],r['status_under_proposed_empirical_policy']) for r in axes)
countrows=[dict(system=k[0],epsilon=k[1],status=k[2],axis_rows=v) for k,v in sorted(counts.items())]
check('status_csv_counts_saved_labels',csvread('saved_status_counts.csv')==csvvalues(countrows))
check('end_audit47_PASS',len(read(A/'end_audit.json')['checks'])==47 and all(x['status']=='PASS' for x in read(A/'end_audit.json')['checks']))
completion=read(A/'completion_witness.json');external=read(A/'external_exit_witness.json')
check('completion_no_errors',completion['status']=='SIGNAL_COMPLETE_MANDATORY_STOP' and completion['errors']=={})
check('both_worker_exit0_reaped',set(completion['workers'])=={'H4','H6'} and all(x['exit']==0 and x['reaped'] for x in completion['workers'].values()))
check('launcher_exit0_reaped',external['native_exit']==external['wait4_status']==0 and external['reaped'])
check('no_next_phase_retry',not completion['retry'] and not completion['next_phase_started'] and not external['forwarded_signals'])
check('two_processes30_threads',read(A/'fresh_gate.json')['system_workers']==2 and all(read(A/(x+'_context_receipt.json'))['threads']==read(A/(x+'_start.json'))['threads']==30 for x in ('H4','H6')))
inventory=read(A/'private_raw_inventory.json');by_path={r['path']:r for r in inventory}
check('unique_private_inventory196',len(inventory)==len(by_path)==196 and all(not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts for r in inventory))
results_by_id={x['candidate']['id']:x for x in records}
for system,new_count in [('H4',37),('H6',45)]:
    report=read(A/(system+'_complete.json'));context=read(A/(system+'_context_receipt.json'))
    check(system+'_complete_new_reuse_counts',report['status']=='SIGNAL_SYSTEM_COMPLETE_MANDATORY_STOP' and report['new_signal']==new_count and report['reused_signal']==3)
    check(system+'_context_no_new_reference',context['saved_snapshot_restores']==context['saved_reference_reads']==1 and context['new_full_H_reference']==0)
    check(system+'_candidate_result_SHA_inventory',all(x['file']==x['id']+'.json' and by_path['run02/output/'+x['file']]['sha256']==x['sha256'] for x in report['results']))
    check(system+'_prefix_identity',all(results_by_id[x['id']]['preparation_hash']==context['prefixes'][str(expected[x['id']]['prefix'])]['preparation'] and results_by_id[x['id']]['prefix_context_hash']==context['prefixes'][str(expected[x['id']]['prefix'])]['context_hash'] for x in report['results']))
closure=read(A/'source_closure.json')
check('scientific_SOURCE_bytes',all((ROOT/r['path']).read_bytes()==subprocess.check_output(['git','show',s['SOURCE']+':'+r['path']],cwd=ROOT) and sha(ROOT/r['path'])==r['sha256'] for r in closure['files']))
check('input_rank_policy_not_fixed_rank',all(x['rank_policy']=='tol_only_1e-8' and x['rank_is_saved_observation_not_generation_argument'] for x in closure['input_bindings'].values()))
preserve=read(A/'preservation.json')
check('old_preservation_and_STOP',preserve['prior_protected_files']==2899 and preserve['run01_STOP_protected_files']==18 and not preserve['changed_files'] and preserve['original_run01_STOP_remains_STOP'])
approval=read(A/'execution_scope_and_approval_projection.json')
check('scoped_new_user_approval',approval['approval']['user_instruction']=='修正を行って再実行して' and approval['approval']['execution_authorized'] and approval['approval']['allowance_policy_accepted_for_this_run'] and approval['one_shot_ended'])
manifest=read(ROOT/'artifacts/validation_manifest.json')
check('manifest_current',manifest['track_a_h46_m5_signal_run02_results_20261020']==s)
old=json.loads(subprocess.check_output(['git','show',s['base_commit']+':artifacts/validation_manifest.json'],cwd=ROOT))
check('previous_manifest_entries_preserved',all(manifest.get(k)==v for k,v in old.items()))
report=ROOT/s['report'];body=report.read_text()
check('report_links_resolve',all((report.parent/x).resolve().exists() for x in re.findall(r'\]\(([^)]+)\)',body) if '://' not in x))
check('indexes_and_note',all(report.name in (ROOT/x).read_text() for x in ['PROJECT_MAP.md','VALIDATION_STATUS.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','docs/research/研究概要・現状.md','docs/research/研究ノート/2026-10-20_h46_m5_signal_readiness.md']))
check('STOP_no_AD03_adoption_no_push',s['mandatory_STOP'] and s['AD03']=='DEFERRED' and s['push_performed'] is False)
if (A/'inventory.json').exists():
    check('public_artifact_inventory',all(sha(A/r['path'])==r['sha256'] and (A/r['path']).stat().st_size==r['bytes'] for r in read(A/'inventory.json')))
print(json.dumps(dict(status='STRUCTURE_PASS_SAVED_TRANSCRIPTION_NOT_NEW_SCIENTIFIC_ANALYSIS',checks=checks),indent=2))
