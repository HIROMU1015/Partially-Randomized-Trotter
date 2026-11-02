"""Audit saved PF direct results; no scientific imports, fit or ranking."""
import csv
import io
from pathlib import Path
import pf_direct_compile_contract as c
import pf_direct_compile_adapter as a
import pf_direct_execution as e


def audit():
    term=c.read_json(e.PRIVATE/'run01/terminal.json');rows=[];coverage=[]
    for group in e.GROUPS:
        spec=c.read_json(e.PRIVATE/'sealed01'/(group+'_spec.json'))
        _,selected=e.expected_tasks(group);root=Path(spec['output_root'])
        observed={}
        for path in sorted(root.glob('*/accepted_measurements.json')):
            doc=c.read_json(path);w=doc['witness']
            c.require(w['exit_code']==0 and w['reaped'] and w['cleanup_errors']==[],'OWNED_NATIVE_EXIT')
            original=c.read_json(path.parent/'result.json')
            c.require(e.sha(path.parent/'result.json')==w['result_sha256'],'RESULT_BYTES')
            c.require(len(original['rows'])==len(doc['rows'])==2,'BOTH_AXES')
            for row in doc['rows']:
                a.validate_measurement(row,c.COMPILER,row['axis'])
                c.require(row['task_id'] not in observed,'DUPLICATE_RESULT')
                observed[row['task_id']]=row;rows.append(dict(contract_id=group,**row,
                    native_worker_pid=w['pid'],peak_worker_RSS_bytes=w['peak_RSS_bytes']))
        allowed={r['task_id'] for r in selected};c.require(set(observed)<=allowed,'UNAPPROVED_AXIS')
        for task in selected:
            output=root/(task['system']+'__'+task['candidate_id'])
            reason=''
            if task['task_id'] not in observed:
                p=output/'STOP.json'
                reason=c.read_json(p)['reason'] if p.exists() else 'UNPERFORMED_OR_UNACCEPTED_SEE_CONTRACT_LEDGER'
            coverage.append(dict(contract_id=group,task_id=task['task_id'],status='PASS_DIRECT_COMPILED' if task['task_id'] in observed else 'STOP_OR_UNPERFORMED',reason=reason))
        c.require(len(observed)==term['contracts'][group]['accepted_axes'],'SUMMARY_COUNT')
    target=e.PUBLIC;target.mkdir(parents=True,exist_ok=False)
    e.save(target/'completion_audit.json',dict(status='AUDITED_SAVED_RESULTS',summary=term,
        attempted_scopes_axes=788,accepted_axes=len(rows),stopped_or_unperformed_axes=788-len(rows),
        author_checks_not_independent_scientific_review=True,no_model_evaluation=True))
    e.save(target/'direct_measurements.json',dict(rows=rows,metric='compiled RZ of full Hadamard wrapper; no state preparation; no shot weighting',
        model_values_preserved_in_original_plan=True,no_fit=True,no_reranking=True))
    for name,values in (('axis_coverage.csv',coverage),('gate_counts.csv',rows)):
        fields=list(values[0]) if name=='axis_coverage.csv' else ['contract_id','task_id','system','candidate_id','axis','RZ','CX','RZ_depth','CX_depth','depth','size','compiler_hash','SOURCE','compile_seconds','peak_worker_RSS_bytes']
        with (target/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(values)
    e.save(target/'MANDATORY_STOP.json',dict(status='MANDATORY_STOP',accepted_axes=len(rows),not_accepted_axes=788-len(rows),new_sampling=0,new_fit=0,new_shot_recalculation=0,new_cost_reranking=0,next_stage_authorized=False))
    print('SAVED_RESULT_AUDIT_COMPLETE',len(rows),788-len(rows))


if __name__=='__main__':audit()
