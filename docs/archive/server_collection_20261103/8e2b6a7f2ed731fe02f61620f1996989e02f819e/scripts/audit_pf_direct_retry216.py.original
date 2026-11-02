"""Automatic saved-result audit after coordinator native exit; no science calls."""
import argparse
import csv
import gzip
import json
from pathlib import Path

import pf_direct_compile_contract as c
import pf_direct_compile_adapter as a
import pf_direct_retry_contract as r


def write_csv(path,rows,fields):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def audit(seal,*,api=r):
    r=api
    expected_new=getattr(r,'EXPECTED_NEW_AXES',216);expected_old=getattr(r,'EXPECTED_OLD_AXES',572)
    spec=c.read_json(seal/(r.GROUPS[0]+'_spec.json'));grant=c.read_json(seal/(r.GROUPS[0]+'_grant.json'))
    _,selected=r.authorize(spec,grant)
    observer=r.PRIVATE/'observer01';native=c.read_json(observer/'coordinator_native_exit.json')
    c.require(native['exit_code']==0 and native['reaped'] and native['cleanup_errors']==[],'INDEPENDENT_COORDINATOR_EXIT')
    terminal=r.PRIVATE/'run01/terminal.json';binding=c.read_json(observer/'coordinator_terminal_binding.json')
    c.require(r.sha(terminal)==binding['sha256'] and r.sha(observer/'coordinator_native_exit.json')==binding['native_exit_witness_sha256'],
        'NATIVE_COORDINATOR_TERMINAL_BINDING')
    term=c.read_json(terminal);root=Path(spec['output_root']);summary=c.read_json(root/'summary.json')
    c.require(term['active_owned_workers']==0 and summary['all_workers_reaped'],'ALL_OWNED_WORKERS_REAPED')
    gate=r.gate(spec,grant)
    rows=[];coverage=[];bundles=[];native_ids=set();used={k:0 for k in spec['call_caps']}
    ledger=[json.loads(v) for v in (root/'ledger.jsonl').read_text().splitlines()]
    reserved=[tid for event in ledger if event['event']=='BEFORE_WORKER_RESERVED' for tid in event['task_ids']]
    c.require(len(reserved)==len(set(reserved)) and set(reserved)==set(spec['selected_task_ids']),'ONE_RESERVATION_PER_RETRY_AXIS')
    for (system,cid),pair in r.candidate_pairs(selected).items():
        directory=root/(system+'__'+cid);exitpath=directory/'owned_worker_exit.json'
        c.require(exitpath.is_file(),'EXACT_WORKER_NATIVE_WITNESS')
        witness=c.read_json(exitpath);c.require(witness['reaped'] and witness['cleanup_errors']==[],'WORKER_CLEANUP')
        identity=(witness['pid'],witness['start_ticks'])
        c.require(identity not in native_ids,'UNIQUE_NATIVE_WORKER_IDENTITY');native_ids.add(identity)
        worker_ledger=[json.loads(v) for v in (directory/'ledger.jsonl').read_text().splitlines()]
        for entry in worker_ledger:used[entry['call']]+=entry['count']
        files={path.name:c.read_json(path) for path in directory.glob('*.json')}
        good=(directory/'accepted_measurements.json').exists();reason=''
        if good:
            doc=files['accepted_measurements.json'];original=files['result.json']
            if hasattr(r,'validate_saved_worker'):r.validate_saved_worker(pair,files,witness,spec)
            c.require(witness['exit_code']==0 and witness['status']=='PASS' and r.sha(directory/'result.json')==witness['result_sha256'],
                'SUCCESSFUL_NATIVE_RESULT')
            c.require({v['axis'] for v in doc['rows']}==set(c.AXES) and len(doc['rows'])==2,'FULL_PAIR')
            for raw,row in zip(original['rows'],doc['rows']):
                expected=system+':'+cid+':'+row['axis']
                accepted=a.accept_owned_measurement(raw,dict(witness,task_id=expected,result_digest=c.digest(raw)),expected)
                c.require(accepted==row and row['SOURCE']==spec['SOURCE'],'RETRY_ROW_PROVENANCE')
                a.validate_measurement(row,c.COMPILER,row['axis']);rows.append(dict(row,provenance=getattr(r,'NEW_PROVENANCE','NEW_RETRY216_DIRECT_OBSERVATION')))
        else:reason=files.get('STOP.json',{}).get('reason',witness.get('reason','UNKNOWN_STOP'))
        for task in pair:coverage.append(dict(task_id=task['task_id'],status='ACCEPTED_RETRY' if good else 'STOP_RETRY',reason=reason))
        bundles.append(dict(system=system,candidate_id=cid,files=files,worker_ledger=worker_ledger))
    c.require(len(rows)==summary['accepted_axes'],'COORDINATOR_ACCEPTED_COUNT')
    c.require(summary['final_gate']['status']=='PASS' and all(used[k]<=spec['call_caps'][k] for k in used),'FINAL_GATE_AND_CALL_SCOPE')
    reuse=c.read_json(spec['reuse_receipt']);old=c.read_json(getattr(r,'OLD_MEASUREMENTS_PATH',r.ROOT/r.OLD_PACKET/'accepted_measurements.json'))['rows']
    c.require(len(old)==expected_old and {v['task_id'] for v in old}==set(reuse['accepted_task_ids']),'SAME_SAVED_REUSE')
    combined=([dict(row,reused_by_SOURCE=spec['SOURCE']) if getattr(r,'PRESERVE_REUSED_ROWS',False)
        else dict(row,provenance='REUSED_OLD_TASK_ACCEPTANCE_WITH_GLOBAL_GATE_LIMIT') for row in old]+rows)
    c.require(len({v['task_id'] for v in combined})==len(combined),'NO_DUPLICATE_NEW_OLD_OBSERVATIONS')
    output=r.PUBLIC/'results01';output.mkdir(parents=True,exist_ok=False)
    result=dict(status=getattr(r,'SUCCESS_STATUS','PASS_RETRY216_COMPLETE') if len(rows)==expected_new else 'PARTIAL_RETRY_MANDATORY_STOP',
        SOURCE=spec['SOURCE'],new_retry_axes=len(rows),old_saved_task_acceptance_axes=expected_old,combined_saved_axes=len(combined),
        old_global_final_gate_not_retroactively_recreated=True,all_new_worker_exits_confirmed=True,
        independent_coordinator_native_exit=native,coordinator_summary=summary,coordinator_terminal=term,
        post_execution_byte_gate=gate,actual_worker_calls=used,parent_reserved_calls=summary['calls_reserved'],new_sampling=0,new_fit=0,new_shot_recalculation=0,
        new_cost_reranking=0,next_stage_authorized=False,detailed_scientific_review=False,final_evidence_freeze=False)
    r.save(output/'completion_audit.json',result)
    r.save(output/'new_measurements.json',dict(rows=rows,metric='full Hadamard wrapper compiled integer gates; excludes state preparation, no shot weighting'))
    r.save(output/'combined_observation_index.json',dict(rows=combined,old_global_gate_limitation_preserved=True,fit=False))
    write_csv(output/'new_gate_counts.csv',rows,['task_id','system','candidate_id','axis','RZ','CX','RZ_depth','CX_depth','depth','size','compiler_hash','SOURCE','compile_seconds','requires_saved_classical_signal_reconciliation'])
    write_csv(output/'retry_axis_status.csv',coverage,['task_id','status','reason'])
    write_csv(output/'combined_gate_counts.csv',combined,['task_id','system','candidate_id','axis','RZ','CX','compiler_hash','SOURCE','provenance','requires_saved_classical_signal_reconciliation'])
    (output/'worker_evidence.json.gz').write_bytes(gzip.compress(json.dumps(dict(pairs=bundles,parent_ledger=ledger),sort_keys=True).encode(),mtime=0))
    (output/getattr(r,'REUSE_FILENAME','reuse_572_receipt.json.gz')).write_bytes(gzip.compress(json.dumps(reuse,sort_keys=True).encode(),mtime=0))
    (output/'resource_observations.jsonl.gz').write_bytes(gzip.compress((r.PRIVATE/'run01/resource_observations.jsonl').read_bytes(),mtime=0))
    r.save(output/'MANDATORY_STOP.json',dict(status=result['status'],automatic_retry=False,next_stage_authorized=False))
    print(json.dumps(dict(status=result['status'],new_axes=len(rows),combined_axes=len(combined))))
    return 0 if len(rows)==expected_new else 2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sealed-directory',type=Path,required=True)
    raise SystemExit(audit(p.parse_args().sealed_directory))
