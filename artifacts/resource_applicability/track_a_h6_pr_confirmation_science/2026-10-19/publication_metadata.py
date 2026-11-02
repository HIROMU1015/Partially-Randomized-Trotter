"""Serialize completed receipts only. No sampling, compilation or statistics."""
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys

SCIENCE_ROOT = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-pr-confirmation-science-source-20261019')
REPORT_ROOT = Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/h6-pr-confirmation-science-20261019')
PRIVATE = Path('/home/AbeHiromu/projects/h6-pr-confirmation-preparation/20261019/science-run01')
SEALED = PRIVATE.parent / 'final-sealed02'
sys.path.insert(0,str(SCIENCE_ROOT/'scripts'))
import h6_pr_confirmation as h
import h6_pr_confirmation_seal as final
c = h.c


def main():
    frozen = c.read(SEALED/'preparation_receipt_PRIVATE.json')
    scopes,axes,trajectories,native,controls,grants = [],[],[],[],[],[]
    for item in frozen['specs']:
        spec=c.read(item['path']);scope=spec['scope'];p=PRIVATE/scope;out=Path(spec['output'])
        stop=c.read(p/'SCOPE_MANDATORY_STOP.json')
        audit=c.read(p/'witness/completion_audit.json')
        c.require(stop['status']==audit['status']=='COMPLETE_MANDATORY_STOP','EACH_SCOPE_STOP_REQUIRED')
        c.require(c.sha(item['path'])==item['sha256'],'ORIGINAL_SPEC_BYTES')
        h.verify(spec)
        result=c.read(out/'result.json')
        c.require(result['calls']==spec['calls'] and result['primary']['n']==spec['n'],'FIXED_SCOPE_COUNTS')
        c.require(stop['result_sha256']==c.sha(out/'result.json'),'SCOPE_STOP_RESULT_SHA')
        grant=c.read(p/'grant_ACTIVE_PRIVATE.json');preflight=c.read(p/'preflight_PASS_PRIVATE.json')
        c.require(c.sha(p/'grant_ACTIVE_PRIVATE.json')==preflight['grant_sha256'],'ISSUED_GRANT_UNCHANGED')
        quota=c.read(p/'quota_receipt_PRIVATE.json')
        c.require(quota['policy']=='NO_QUOTA_CONFIRMED','CONFIRMED_QUOTA_POLICY')
        scopes.append(dict(scope=scope,epsilon=spec['target']['epsilon'],cell=spec['target']['cell'],
            fixed_PF_id=spec['target']['fixed_PF_id'],binding=spec['binding'],shots=spec['shots'],
            primary=result['primary'],event_order_counts=result['event_order_counts'],calls=result['calls'],
            result_sha256=c.sha(out/'result.json'),terminal_sha256=c.sha(out/'terminal.json'),
            technical_digest=spec['technical_digest'],scope_STOP_sha256=c.sha(p/'SCOPE_MANDATORY_STOP.json'),
            completion_audit_sha256=c.sha(p/'witness/completion_audit.json')))
        maps=c.read(out/'native_receipt.json')['maps']
        workers=[w for m in maps for w in m['workers']]
        for worker in workers:
            directory=Path(worker['directory']);task=c.read(directory/'task_result.json')
            trajectories.append(dict(scope=scope,cell_id=task['cell_id'],replica=task['replica'],cohort=task['cohort'],
                event_digest=task['event_digest'],trajectory_sha256=c.sha(directory/'trajectory.json'),
                result_sha256=c.sha(directory/'task_result.json'),exit_code=worker['exit_code'],reaped=worker['reaped']))
            for axis,v in task['axes'].items():
                axes.append(dict(scope=scope,cell_id=task['cell_id'],replica=task['replica'],axis=axis,
                    saved_shots=spec['shots'][axis],**{k:v[k] for k in
                    ('RZ','CX','depth','RZ_depth','CX_depth','size','compiler_hash','fingerprint','event_digest')}))
            if task['replica']==0:
                controls.append(dict(scope=scope,**c.read(directory/'control.json')))
        start=c.read(out/'startup_gate.json')['fresh']['time']
        end=(p/'witness/native_completion.json').stat().st_mtime
        native.append(dict(scope=scope,driver=audit['native'],raw_audit=audit['raw_audit'],
            trajectory_worker_count=len(workers),all_maps_reaped=all(m['all_reaped'] for m in maps),
            cleanup_errors=[e for m in maps for e in m['cleanup_errors']],
            elapsed_from_startup_to_native_receipt_seconds=end-start,
            peak_RSS_is_single_process_not_pool_total=True,
            allowed_CPU_count=len(grant['allowed_cpus']),trajectory_groups=(len(grant['allowed_cpus'])+1)//2))
        grants.append(dict(scope=scope,SOURCE=spec['SOURCE'],technical_digest=spec['technical_digest'],
            grant_sha256=c.sha(p/'grant_ACTIVE_PRIVATE.json'),
            approval_receipt_sha256=grant['approval_receipt_sha256'],
            consumed_scope_marker_sha256=c.sha(spec['lock']),
            witness_marker_sha256=c.sha(p/'witness/WITNESS_ONE_SHOT.json'),
            quota_policy=quota['policy'],quota_queries=quota['observations'],
            seed_recheck_sha256=c.sha(p/'seed_recheck_PRIVATE.json'),valid_grant_contents_omitted=True))
    total={k:sum(s['calls'][k] for s in scopes) for k in scopes[0]['calls']}
    c.require(total==frozen['total_calls'] and len(axes)==304 and len(trajectories)==152,'EXACT_CAMPAIGN_TOTALS')
    c.verify_byte_rows(c.read(SEALED/'seed_origin_preservation_PRIVATE.json'))
    c.require(not subprocess.check_output(['git','status','--porcelain'],cwd=SCIENCE_ROOT,text=True),'CLEAN_SCIENCE_CHECKOUT')
    bundle=REPORT_ROOT/'artifacts/resource_applicability/track_a_h6_pr_confirmation_science/2026-10-19'
    bundle.mkdir(parents=True,exist_ok=False)
    def save(name,value):c.save(bundle/name,value)
    save('result.json',dict(status='COMPLETE_MANDATORY_STOP',SOURCE=frozen['SOURCE'],
        final_seal_commit='f79366f68caf8585cc36d2f2b0fefcb1a4bd0d1f',scopes=scopes,total_calls=total,
        old_samples_pooled=False,automatic_retry=False,next_stage_authorized=False,
        scientific_statistics_recomputed_for_publication=False,raw_seed_values_omitted=True))
    for name,rows in [('axis_costs.csv',axes),('trajectory_receipts.csv',trajectories)]:
        with (bundle/name).open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save('native_completion.json',native);save('control_checks.json',controls);save('grant_consumption_receipts.json',grants)
    save('source_and_preservation.json',dict(status='PASS_AFTER_BOTH_SCOPES',SOURCE=frozen['SOURCE'],
        seed_registry_sha256=frozen['seed_registry_sha256'],source_hashes=c.read(frozen['specs'][0]['path'])['source_hashes'],
        runtime_receipt_sha256=frozen['runtime_receipt_sha256'],runtime_leaves=34273,provider_files=877,input_rows=114,
        two_final_SHA_gates_passed=True,seed_origin_metadata_preserved=True,
        scientific_SOURCE_checkout_clean=True,old_grants_markers_STOP_not_modified=True,
        quota_policy='NO_QUOTA_CONFIRMED',quota_or_shared_settings_changed=False,AI_subagents=0,independent_review=False,
        publication_helper_sha256=c.sha(__file__)))
    save('scope_boundary.json',c.read(PRIVATE/'Q2_STOP_TO_Q8_EXPLICIT_DISPATCH_PRIVATE.json'))
    save('STOP.json',dict(status='FINAL_MANDATORY_STOP',SOURCE=frozen['SOURCE'],
        both_scope_audits_complete=True,new_sampling_or_compile_authorized=False,
        next_stage_authorized=False,public_push=False,new_review_or_H4_H8_integration=False,
        timestamp_UTC=datetime.now(timezone.utc).isoformat()))
    status=dict(status='TWO_FIXED_H6_SCOPES_COMPLETE_MANDATORY_STOP',SOURCE=frozen['SOURCE'],
        report='docs/research/track_a_h6_pr_confirmation_science_results_20261019.md',
        bundle=str(bundle.relative_to(REPORT_ROOT)),n_by_scope=final.FINAL_N,
        total_calls=total,point_direction_reproductions={s['scope']:s['primary']['reproduction_status'] for s in scopes},
        native_exit0=True,all_workers_reaped=True,quota_policy='NO_QUOTA_CONFIRMED',
        new_signal=0,new_PF_compile=0,additional_trajectory=0,automatic_retry=False,
        new_review_or_H4_H8_integration=False,public_push=False,next_stage_authorized=False,
        independent_review=False,AI_subagents=0)
    save('validation_status_entry.json',status)
    shutil.copyfile(__file__,bundle/'publication_metadata.py')
    for s in scopes:
        p=s['primary'];print(s['scope'], 'mean',float(Fraction(p['mean_exact'])),
            'SE',p['paired_SE'],'PR_reduction_percent',float(Fraction(p['PR_reduction_percent_exact'])),
            'direction',p['reproduction_status'],'events',s['event_order_counts'])
    print('NATIVE',native)
    print('BUNDLE_SAVED',bundle)


if __name__=='__main__':main()
