"""Collect completed DIAGNOSTICS bytes/metadata only; never rerun science."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[4]
sys.path.insert(0,str(ROOT/'scripts'))
import h2_diagnostics_binding as b
import h2_common_production_contract as c


def copy_once(source,target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():b.require(b.sha(source)==b.sha(target),'EXISTING_RECEIPT_COPY_CHANGED')
    else:shutil.copyfile(source,target)


def main():
    assets=b.ASSETS;cfg=b.read(assets/'DIAGNOSTICS_contract.json');run=Path(cfg['run_root'])
    auth=assets/'DIAGNOSTICS_authorization01';authorization=b.read(OUT/'authorization_record.json')
    b.require(b.digest(cfg)==authorization['contract_sha256'],'AUTHORIZED_CONTRACT')
    b.require(b.sha(auth/'approval.json')==authorization['approval_file_sha256'],'APPROVAL_BYTES')
    b.require(b.sha(auth/'grant.json')==authorization['grant_file_sha256'],'ORIGINAL_GRANT_BYTES')
    completion_sha=b.sha(run/'completion.json')
    accepted=b.accepted_diagnostics_receipt(run,cfg,completion_sha256=completion_sha)
    completion=b.read(run/'completion.json');marker=b.read(run/'CONSUMED.json')
    b.require(marker['grant']['id']==authorization['grant_id'],'EXACT_GRANTED_ATTEMPT')
    b.require(b.read(OUT/'process_exit.json')['returncode']==0,'NORMAL_PROCESS_EXIT')
    b.require(completion['used']==c.counts('DIAGNOSTICS'),'EXACT_FINISHED_BUDGET')
    c.seal_check(cfg);c.source_check(cfg,{Path(r['path']) for r in cfg['files']})
    copy_once(run/'completion.json',OUT/'completion.json')
    for rel in accepted['files']:copy_once(run/rel,OUT/rel)
    ledger=[]
    for path in sorted((run/'ledger').glob('*.json')):
        copy_once(path,OUT/'ledger'/path.name)
        ledger.append(dict(path=str(path.resolve()),sha256=b.sha(path),record=b.read(path)))
    for name in ('stdout.log','stderr.log'):copy_once(auth/name,OUT/name)
    candidates=[]
    for candidate in cfg['candidates']:
        row=b.read(run/'results'/(candidate['id']+'_diagnostics.json'));inputs=row['inputs']
        prep=b.read(run/'results'/('preparation_'+str(candidate['prefix'])+'.json'))
        b.require(row['preparation_sha256']==prep['preparation_sha256'] and
                  row['operations_sha256']==prep['operations_sha256']==b.digest(prep['operation_stream']),
                  'PREPARATION_OPERATION_METADATA_JOIN')
        candidates.append(dict(candidate=candidate,
            primary_signals=row['primary_signals'],independent_signals=row['independent_signals'],
            signal_columns=row['signal_columns'],log_B=row['log_B'],schedule=row['schedule'],
            path_signal_errors=inputs['path_signal_errors'],path_vector_errors=inputs['path_vector_errors'],
            corrected_signal_closure=inputs['corrected_signal_closure'],
            log_B_difference=inputs['log_B_discrepancy'],maximum_norm=inputs['maximum_norm'],
            work_primary=inputs['work_primary'],work_independent=inputs['work_independent'],
            reference_binding=row['reference_binding'],preparation_sha256=row['preparation_sha256'],
            operations_sha256=row['operations_sha256'],vector_archive=row['vector_archive'],
            scalar=row['scalar'],scalar_applications=row['scalar_applications'],
            strict_diagnostic_limit=row['strict_diagnostic_limit'],
            phase_alignment=row['phase_alignment'],postcompile_correction=row['postcompile_correction']))
    b.write(OUT/'diagnostic_projection.json',dict(status='COMPLETED_DIAGNOSTICS_VALUES_COPIED_ONLY',
        physical=cfg['physical'],compiler=cfg['compiler'],binding=cfg['binding'],
        source_commit=cfg['source_commit'],contract_sha256=b.digest(cfg),candidates=candidates,
        scientific_values_recomputed=False,new_allowance_shots_cost_or_ranking=False))
    prep=ROOT/'artifacts/resource_applicability/track_a_h2_diagnostics_binding/2026-10-23'
    baseline=b.read(prep/'preservation_observation.json');historical=b.read(prep/'historical_completed_preservation.json')
    checked=[]
    for row in baseline['originals']+baseline['known_control_plane_files']+historical['files']:
        expected=row.get('historical_sha256',row.get('sha256'));actual=b.sha(row['path'])
        b.require(actual==expected,'OLD_ORIGINAL_PRESERVATION')
        checked.append(dict(path=row['path'],sha256=actual,matches_saved_hash=True))
    preparation_files=[]
    for row in b.read(prep/'file_inventory.json')['files']:
        p=prep/row['path'];b.require(b.sha(p)==row['sha256'],'PREPARATION_RECORD_PRESERVATION')
        preparation_files.append(dict(path=str(p.relative_to(ROOT)),sha256=row['sha256']))
    b.require(not Path(cfg['runtime']['cache_prefix']).exists(),'NO_NEW_BYTECODE_CACHE')
    b.write(OUT/'preservation_audit.json',dict(status='PASS',checked_at_utc=datetime.now(timezone.utc).isoformat(),
        old_known_originals=checked,old_preparation_artifact_files=preparation_files,
        sealed_SOURCE_input_runtime_unchanged=True,sealed_file_count=len(cfg['files']),
        SOURCE_git_byte_matches=True,original_grant_approval_bytes_unchanged=True,
        old_STOPs_retained=True,new_grant_consumed_once=True,retry=False,
        shared_environment_modified=False,other_worktrees_frozen=False,
        new_scientific_evaluation_during_collection=False))
    b.write(OUT/'acceptance_audit.json',dict(status='DIAGNOSTICS_CONTRACT_COMPLETE',
        completion_sha256=completion_sha,marker_sha256=b.sha(run/'CONSUMED.json'),
        original_marker_private_locator=str(run/'CONSUMED.json'),grant_id=authorization['grant_id'],
        approval_sha256=marker['approval_sha256'],grant_contents_published=False,
        metadata_acceptance=accepted,used=completion['used'],budget=cfg['budget'],ledger=ledger,
        ledger_sha256=b.digest(ledger),process_reaped=True,strict_diagnostic_gates_completed=4,
        source_runtime_and_loaded_closure_guards_completed=True,
        per_module_actual_import_list_saved=False,
        scope='Primary process/source guards passed. Metadata/hash checks only after execution; not an independent numerical review.',
        scientific_GO=False,compare_authorized=False,mandatory_STOP=True))
    b.write(OUT/'STOP.json',dict(status='MANDATORY_STOP_AFTER_COMPLETED_DIAGNOSTICS',
        contract_sha256=b.digest(cfg),source_commit=cfg['source_commit'],completion_sha256=completion_sha,
        grant_id=authorization['grant_id'],grant_state='CONSUMED_NO_RETRY_NO_REFUND',
        completed_candidates=4,used=completion['used'],retry=False,
        compare_status='BLOCKED_PR_C_IDENTITY_AND_NO_LIVE_CONTINUATION_LOADER',
        further_scientific_execution_authorized=False,scientific_GO=False,
        push=False,pull_request_created=False,merge=False,mandatory_STOP=True))
    print(json.dumps(dict(status='DIAGNOSTICS_ACCEPTANCE_METADATA_PASS',candidates=4,
        scientific_reexecution=False,old_originals_preserved=len(checked),completion_sha256=completion_sha)))


if __name__=='__main__':main()
