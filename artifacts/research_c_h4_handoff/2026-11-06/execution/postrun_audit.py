"""Read-only post-run integrity/count audit of the permitted experiment."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_h4_relocated_handoff as h
from trottertracks import c_h4_relocated_downstream as bridge
from trottertracks import c_h4_relocated_execution as execution


def main():
    c=json.loads((ROOT/'docs/research/research_c_h4_handoff_contract_20261106.json').read_text())
    base=ROOT/'artifacts/research_c_h4_handoff/2026-11-06';out=ROOT/c['fresh_output'];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    stop=json.loads((out/'STOP.json').read_text());completion=json.loads((base/'execution/completion.json').read_text())
    assert not stop['failed'] and stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    assert completion['failed'] is None and completion['planned_handoffs']==1
    assert not (out/'failure.json').exists()
    execution.preflight_bytes(ROOT,c);h.validate_checkpoint(ROOT,c)
    result=json.loads((out/'comparison.json').read_text());plan=json.loads((out/'frozen_selection.json').read_text())
    assert result['status']=='REGISTERED_RELOCATED_DOWNSTREAM_COMPLETE'
    assert result['selection']==plan and result['mandatory_stop']
    tasks=[]
    for label,row in c['candidates'].items():
        identity=f"{row['file_sha256']}:{bridge.VERSION}:LD2"
        selection=json.loads((out/label/'selection'/'selection.json').read_text())
        assert sorted(x['r'] for x in selection)==c['r_grid']
        for entry in selection:
            if entry['eligible']:
                tasks.append((label,identity,'selection',entry['r'],out/label/'selection'/f"r{entry['r']}"/'selection',1))
        d=json.loads((out/label/'selection/D/result.json').read_text())
        assert d['candidate']==identity
        if d['native_admitted']:
            for kind in ('raw','compiled','raw_adjoint','compiled_adjoint'):
                assert (out/label/'selection/D/circuits'/f'{kind}.qpy').is_file()
    for label,r in plan['PR_final_pairs']:
        identity=f"{c['candidates'][label]['file_sha256']}:{bridge.VERSION}:LD2"
        folder=out/label/f'final_r{r}'
        summary=json.loads((folder/'summary.json').read_text())
        assert len(summary['rows'])==64
        tasks.append((label,identity,'final',r,folder,2))
    observations=set();reused=0;computed=0;cohorts=[];timings=[]
    old=Path(c['handoff']['prior_output'])
    snap=json.loads((ROOT/c['handoff']['checkpoint']).read_text())
    for label,identity,split,r,folder,batches in tasks:
        value=json.loads((folder/'result.json').read_text());assert value['samples']==32*batches
        for batch in range(batches):
            p=folder/f'batch{batch:02d}';policy=h.sampling_policy(bridge.cohort_key(identity),split,r,batch)
            rows=h.completed_samples(p,policy,identity,c['sampling_SOURCE_sha256']);assert len(rows)==32
            for i in rows:
                key=(identity,split,r,batch,i);assert key not in observations;observations.add(key)
            old_cohort=(folder/'handoff_reuse.json').exists()
            if old_cohort:
                reused+=32
                for i in range(32):
                    for part in h.sample_paths(i):
                        rel=str((p/part).relative_to(out));assert sha(p/part)==snap['files'][rel]
            else:
                marker=json.loads((p/'handoff_samples.json').read_text())
                reused+=marker['reused'];computed+=marker['computed'];assert marker['reused']+marker['computed']==32
                timing=json.loads((p/'native_timing.json').read_text())
                assert timing['full_input_output_dimension']==512 and timing['actual_inverse_circuits_independently_evaluated']
                timings.append(dict(path=str(p.relative_to(out)),wall_seconds=timing['wall_seconds'],worker_CPU_seconds=timing['worker_CPU_seconds'],workers=timing['resource_plan']['workers']))
            cohorts.append(dict(candidate=label,split=split,r=r,batch=batch,samples=32,reused_complete_cohort=old_cohort))
    assert reused==snap['completed_native_samples']==288
    assert computed+reused==len(observations)<=c['logical_work']['native_PR_max']
    old_batch=old/snap['interrupted_batch'];new_batch=out/snap['interrupted_batch']
    for i in range(32):assert (old_batch/'samples'/f'sample{i:02d}_events.json').read_bytes()==(new_batch/'samples'/f'sample{i:02d}_events.json').read_bytes()
    record=dict(status='FIXED_SCOPE_HANDOFF_AND_RESULTS_AUDIT_PASS',SOURCE_sha256=c['SOURCE_sha256'],sampling_SOURCE_sha256=c['sampling_SOURCE_sha256'],
        prior_verified_samples_reused=reused,new_samples_computed=computed,unique_price_samples=len(observations),
        selection_policies=len([x for x in tasks if x[2]=='selection']),final_unique_pairs=len(plan['PR_final_pairs']),
        correctness_failure=0,old_interrupted_events_byte_identical=32,
        all64_SOURCE_files_and16_inputs_verified=True,checkpoint_files=len(snap['files']),
        all_verified_sample_file_sets_and_original_domain_checked=True,actual_inverse_circuits_independently_evaluated=True,
        original_results_unchanged=True,within_continuation_retry=0,within_continuation_resume=0,planned_handoffs=1,
        mandatory_stop=True,cohorts=cohorts,new_batch_timings=timings,completion=completion)
    (base/'execution/result_audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
