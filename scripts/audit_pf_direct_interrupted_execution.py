"""Read-only audit of interrupted PF acquisition; never run or retry science."""
from collections import Counter
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import subprocess

import pf_direct_compile_contract as c
import pf_direct_compile_adapter as a
import pf_direct_execution as e

OUT=e.ROOT/'artifacts/resource_applicability/track_a_pf_direct_compile_interrupted_audit/2026-10-20'
SOURCE='220b9644df6e114116737fd041973973fcf86cf5'


def records(path):
    return [json.loads(row) for row in Path(path).read_text().splitlines()]


def write_gzip(path,value):
    raw=json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw,mtime=0))


def table(path,rows,fields):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def audit():
    run=e.PRIVATE/'run01'
    c.require(not (run/'terminal.json').exists(),'USE_NORMAL_AUDIT_IF_COMPLETE')
    handoff=c.read_json(e.PRIVATE/'RUNNING_CHAT_HANDOFF.json')
    proc=Path('/proc')/str(handoff['coordinator_pid'])
    if proc.exists():
        stat=proc.joinpath('stat').read_text();parts=stat[stat.rfind(')')+2:].split()
        c.require(parts[19]!=handoff['coordinator_process_start_ticks'],'COORDINATOR_STILL_ALIVE')
    search=subprocess.run(['pgrep','-af',str(e.ROOT/'scripts/pf_direct_production_worker.py')],capture_output=True,text=True)
    c.require(search.returncode==1,'OWNED_WORKERS_STILL_PRESENT_OR_SEARCH_FAILED')
    OUT.mkdir(parents=True,exist_ok=False)
    accepted=[];coverage=[];workers=[];contracts={};preserved={};native_witnesses=[]
    dependency_files={}
    for group in e.GROUPS:
        specpath=e.PRIVATE/'sealed01'/(group+'_spec.json');grantpath=e.PRIVATE/'sealed01'/(group+'_grant.json')
        spec,grant=c.read_json(specpath),c.read_json(grantpath)
        _,selected=e.authorize(spec,grant);root=Path(spec['output_root'])
        c.require(spec['SOURCE']==SOURCE,'UNCHANGED_EXECUTION_SOURCE')
        gate=e.gate(spec,grant)
        for path,h in spec['bound_files'].items():
            c.require(path not in dependency_files or dependency_files[path]==h,'DEPENDENCY_HASH_CONFLICT')
            dependency_files[path]=h
        parent=records(root/'ledger.jsonl');reserved={k:0 for k in spec['call_caps']};actual={k:0 for k in reserved}
        reserve_tasks=[];parent_exits={}
        for row in parent:
            if row['event']=='BEFORE_WORKER_RESERVED':
                reserve_tasks.extend(row['task_ids'])
                for key,value in row['calls'].items():reserved[key]+=value
            elif row['event']=='NATIVE_REAP_AND_ACCEPTANCE':
                w=row['witness'];key=(w['system'],w['candidate_id'])
                c.require(key not in parent_exits,'DUPLICATE_PARENT_EXIT');parent_exits[key]=w
            else:raise ValueError('UNEXPECTED_PARENT_EVENT')
        c.require(len(reserve_tasks)==len(set(reserve_tasks)) and set(reserve_tasks)==set(spec['selected_task_ids']), 'ALL_SCOPED_PAIRS_RESERVED_ONCE')
        c.require(reserved==spec['call_caps'],'CONTRACT_RESERVED_CAPS')
        counts=Counter();bundle=[]
        for (system,cid),pair in e.candidate_pairs(selected).items():
            directory=root/(system+'__'+cid);c.require(directory.is_dir(),'MISSING_RESERVED_TASK')
            ledger=records(directory/'ledger.jsonl');calls={k:0 for k in actual}
            for entry in ledger:
                c.require(entry['before_call'] is True and entry['nonrefundable'] is True,'BEFORE_CALL_RESERVATION')
                calls[entry['call']]+=entry['count']
            c.require(all(calls[k]<=v for k,v in c.call_caps(pair).items()),'PAIR_CALL_SCOPE')
            for key,value in calls.items():actual[key]+=value
            files={p.name:c.read_json(p) for p in directory.glob('*.json')}
            for p in directory.iterdir():
                if p.is_file():preserved[str(p)]=e.sha(p)
            accepted_path=directory/'accepted_measurements.json'
            native=files.get('owned_worker_exit.json')
            if accepted_path.exists():
                doc=files['accepted_measurements.json'];original=files['result.json']
                c.require(native is not None and native['status']=='PASS' and native['exit_code']==0
                    and native['reaped'] is True and native['cleanup_errors']==[],'CONFIRMED_NATIVE_EXIT')
                c.require(parent_exits[(system,cid)]==native,'PARENT_CHILD_EXIT_MATCH')
                c.require(e.sha(directory/'result.json')==native['result_sha256'],'BOUND_NATIVE_RESULT_HASH')
                c.require(all(native[key]==value for key,value in doc['witness'].items()),'SAVED_WITNESS_IDENTITY')
                c.require(len(doc['rows'])==len(original['rows'])==2 and original['calls']==calls==c.call_caps(pair),'EXACT_COMPLETED_PAIR_CALLS')
                c.require({r['axis'] for r in doc['rows']}==set(c.AXES),'COMPLETE_COSINE_SINE')
                for raw,row in zip(original['rows'],doc['rows']):
                    expected=system+':'+cid+':'+row['axis']
                    c.require(row['task_id']==expected and row['SOURCE']==SOURCE,'ROW_IDENTITY')
                    restored=a.accept_owned_measurement(raw,dict(native,task_id=expected,result_digest=c.digest(raw)),expected)
                    c.require(row==restored,'SAVED_MEASUREMENT_CHANGED')
                    a.validate_measurement(row,c.COMPILER,row['axis'])
                    accepted.append(dict(contract_id=group,**row,peak_worker_RSS_bytes=native['peak_RSS_bytes']))
                status='SAVED_TASK_ACCEPTANCE';reason='';counts[status]+=2
            elif 'STOP.json' in files:
                c.require(native is not None and native['status']=='STOP_TASK' and native['exit_code']!=0
                    and native['reaped'] is True,'EXPLICIT_STOP_NATIVE_WITNESS')
                c.require(parent_exits[(system,cid)]==native,'STOP_PARENT_EXIT')
                c.require(calls['wrapper_compile']==0,'PRECOMPILE_CLASSIFICATION_ONLY')
                status='EXPLICIT_PRECOMPILE_STOP';reason=files['STOP.json']['reason'];counts[status]+=2
            else:
                c.require(native is None and (system,cid) not in parent_exits,'UNCONFIRMED_EXIT_NOT_SYNTHESIZED')
                status='INTERRUPTED_NATIVE_COMPLETION_UNCONFIRMED';reason='No native exit/reap/result witness; coordinator and workers absent'
                counts[status]+=2
            if native is not None:native_witnesses.append(native)
            for task in pair:
                raw=files.get('raw_'+task['axis']+'.json')
                coverage.append(dict(contract_id=group,task_id=task['task_id'],system=system,candidate_id=cid,axis=task['axis'],
                    status=status,reason=reason,raw_axis_file_present=raw is not None,
                    raw_compile_metrics_accepted=status=='SAVED_TASK_ACCEPTANCE'))
            bundle.append(dict(system=system,candidate_id=cid,status=status,calls=calls,files=files,ledger=ledger))
        c.require(sum(counts.values())==len(selected),'SCOPE_CLASSIFICATION_COVERAGE')
        c.require(all(actual[k]<=reserved[k] for k in actual),'ACTUAL_WITHIN_RESERVED')
        contracts[group]=dict(authorized_axes=len(selected),axis_counts=dict(counts),SOURCE=SOURCE,
            original_spec_digest=c.digest(spec),original_grant_digest=c.digest(grant),approval_receipt_sha256=grant['approval_sha256'],
            original_marker=c.read_json(root/'marker.json'),original_fresh_gate=c.read_json(root/'fresh_gate.json'),
            original_final_gate_present=False,original_final_STOP_present=(root/'STOP.json').exists(),
            post_interruption_byte_gate=gate,post_interruption_gate_is_not_missing_final_gate=True,
            confirmed_exit_witnesses=len(parent_exits),all_native_exits_confirmed=len(parent_exits)==len(bundle),
            parent_nonrefundable_reservations=reserved,worker_before_call_reservations=actual,
            systems={key:dict(binding=value['config']['binding'],profile=value['profile'],provider_SOURCE=value['provider_SOURCE'],input_hashes=value['input_hashes']) for key,value in spec['systems'].items()})
        workers.append(dict(contract_id=group,pairs=bundle,parent_ledger=parent))
        for path in (specpath,grantpath,root/'marker.json',root/'ledger.jsonl',root/'fresh_gate.json'):
            preserved[str(path)]=e.sha(path)
    c.require(len({r['task_id'] for r in accepted})==len(accepted),'NO_DUPLICATE_ACCEPTANCE')
    c.require(len({r['task_id'] for r in coverage})==len(coverage)==788,'EXACT_788_COVERAGE')
    logs=records(run/'resource_observations.jsonl')
    resource=dict(first_observation=logs[0]['time_UTC'],last_observation=logs[-1]['time_UTC'],
        observation_span_seconds=logs[-1]['monotonic']-logs[0]['monotonic'],
        exact_run_end_time=None,full_run_wall_seconds=None,
        confirmed_worker_CPU_seconds=sum(w['user_CPU_seconds']+w['system_CPU_seconds'] for w in native_witnesses),
        confirmed_worker_peak_RSS_max_bytes=max(w['peak_RSS_bytes'] for w in native_witnesses),
        peak_sampled_owned_RSS_sum_bytes=max(row['sampled_owned_RSS_sum'] for row in logs),
        peak_observed_worker_count=max(row['owned_workers'] for row in logs),
        usage_of_eight_unreaped_workers_incomplete=True)
    total=Counter(row['status'] for row in coverage)
    summary=dict(status='INTERRUPTED_MANDATORY_STOP_NOT_FULL_RUN_PASS',SOURCE=SOURCE,authorized_axes=788,
        axis_counts=dict(total),per_series={system:dict(Counter(row['status'] for row in coverage if row['system']==system)) for system in c.SYSTEM_LAYOUTS},
        confirmed_native_exit_witnesses=len(native_witnesses),unconfirmed_native_exit_pairs=394-len(native_witnesses),
        active_owned_workers_observed=0,coordinator_present=False,coordinator_tool_session='Unknown process id 25019',
        interruption_cause='UNCONFIRMED; no terminal/exit witness, no leftover worker log diagnosis',
        final_terminal_missing=True,original_final_gates_missing=True,
        complete_native_reaping_proven=False,confirmed_compile_failure_axes=0,
        resource=resource,new_science_in_this_audit=False,new_retry=False,
        detailed_scientific_review=False,final_evidence_freeze=False)
    e.save(OUT/'interruption_audit.json',summary)
    e.save(OUT/'execution_contracts_and_call_audit.json',contracts)
    e.save(OUT/'accepted_measurements.json',dict(qualification='Saved task-level accepted observations only; original global final gates absent',rows=accepted,
        metric='compiled RZ of full Hadamard wrapper, excludes state preparation, no shot weighting',model_evaluated=False))
    table(OUT/'axis_status.csv',coverage,list(coverage[0]))
    table(OUT/'accepted_gate_counts.csv',accepted,['contract_id','task_id','system','candidate_id','axis','RZ','CX','RZ_depth','CX_depth','depth','size','compiler_hash','SOURCE','compile_seconds','peak_worker_RSS_bytes'])
    write_gzip(OUT/'worker_evidence.json.gz',dict(contracts=workers))
    write_gzip(OUT/'bound_dependency_hashes.json.gz',dependency_files)
    (OUT/'resource_observations.jsonl.gz').write_bytes(gzip.compress((run/'resource_observations.jsonl').read_bytes(),mtime=0))
    for name,source in [('loader_failure_diagnosis.json','loader_failure_diagnosis01.json'),('scalar_schema_diagnosis.json','scalar_schema_diagnosis01.json')]:
        e.save(OUT/name,c.read_json(e.PRIVATE/source))
    old=c.read_json(e.PRIVATE/'old_packet_preservation01.json')
    for path,h in old['files'].items():c.require(e.sha(e.ROOT/path)==h,'PRIOR_PACKET_CHANGED')
    e.save(OUT/'prior_packet_preservation.json',old)
    e.save(OUT/'original_run_preservation.json',dict(files=preserved,
        original_terminal_or_native_exit_not_created=True,original_SOURCE_grants_ledgers_markers_STOP_not_modified=True))
    e.save(OUT/'MANDATORY_STOP.json',dict(status='MANDATORY_STOP_INTERRUPTION_AUDITED',axis_counts=dict(total),
        retry_authorized=False,new_sampling=0,new_fit=0,new_shot_recalculation=0,new_cost_reranking=0,next_stage_authorized=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':audit()
