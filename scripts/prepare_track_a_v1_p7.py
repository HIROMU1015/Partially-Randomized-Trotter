"""Prepare unapproved p7 config/grant draft using saved bytes and metadata only."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import track_a_v1_p7_runner as run

ROOT=run.ROOT
base=run.base
c=run.contract
ART=ROOT/'artifacts/resource_applicability'


def put(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def protected_files(exclude):
    rows={}
    for name in ('h5-handoff-evidence','h6-handoff-evidence','h7-handoff-evidence','h8-handoff-evidence'):
        tree=ROOT.parent.parent/name
        if not tree.exists():continue
        for directory,folders,names in os.walk(tree):
            parent=Path(directory)
            if parent==exclude:folders[:]=[];continue
            folders[:]=[x for x in folders if x not in ('scratch','cache','numba','matplotlib','__pycache__')]
            for name in names:
                if any(x in name.lower() for x in ('grant','ledger','carry','one_shot')):
                    p=parent/name
                    if p.is_file():rows[str(p)]=dict(bytes=p.stat().st_size,sha256=base.sha(p))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase-b-config',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--private-directory',type=Path,required=True);p.add_argument('--public-directory',type=Path,required=True)
    a=p.parse_args();start=time.monotonic()
    for path in (a.private_directory,a.public_directory):
        if not path.is_absolute() or path.resolve()!=path or path.exists() or not path.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('NEW_ABSOLUTE_DESTINATION_REQUIRED')
    original=base.read_json(a.phase_b_config);protected=protected_files(a.private_directory)
    phase_a=ART/'track_a_v1_phase_a_run02/2026-10-16';phase_b=ART/'track_a_v1_phase_b_run01/2026-10-16'
    export=base.read_json(phase_a/'signal_export_receipt_v1.json');blob=(phase_a/'signal_results_v1.json.gz').read_bytes()
    raw=gzip.decompress(blob)
    if (hashlib.sha256(blob).hexdigest()!=export['compressed_sha256'] or len(raw)!=export['uncompressed_bytes']
            or hashlib.sha256(raw).hexdigest()!=export['uncompressed_sha256']):raise ValueError('PUBLIC_PHASE_A_EXPORT_IDENTITY')
    saved=base.accounting.strict_json(raw)
    selected=[x for x in saved['signals'] if x['cell']['id']==c.CELL_ID]
    if len(selected)!=1:raise ValueError('UNIQUE_SAVED_P7_REQUIRED')
    signal=selected[0]
    if signal['cell']!=c.cell():raise ValueError('REGISTERED_P7_CELL')
    frozen_path=ART/'track_a_v1_phase_b_readiness/2026-10-16/frozen_model.json'
    if base.sha(frozen_path)!=c.MODEL_SHA:raise ValueError('FROZEN_MODEL_HASH')
    frozen=base.read_json(frozen_path);predictions=c.predictions(frozen)
    validation=base.read_json(phase_b/'validation_complete.json')
    base.contract.validate_predictions(validation['rows'],expected_axes=frozen['validation_axes'],frozen_model_sha256=c.MODEL_SHA)
    for path in (phase_a/'execution_terminal_v1.json',phase_b/'execution_terminal.json',
                 ART/'track_a_v1_reaccount_result/2026-10-16/execution_terminal.json'):
        terminal=base.read_json(path)
        if terminal.get('status')!='COMPLETE_MANDATORY_STOP' or terminal.get('exit_code')!=0:
            raise ValueError('PREVIOUS_PHASE_NOT_COMPLETE')
    accounting=[signal['precision_rows']['0.05'][str(f)] for f in (1.,10.)]
    if any(x['N_total']!=24572 or x['axes']['real']['shots']!=10216 or x['axes']['imag']['shots']!=14356
            or not x['cost_acquisition_eligible'] for x in accounting):raise ValueError('SAVED_SHOTS_CHANGED')
    comparisons=base.read_json(ART/'track_a_v1_reaccount_result/2026-10-16/comparisons.json')['rows']
    for factor in (1.,10.):
        row=next(x for x in comparisons if x['epsilon']==.05 and x['factor']==factor
            and x['bias_rule']=='signed_signal' and x['scope']=='fine_grid_S2_plus_truncated_S4')
        if (row['deterministic']['best']['cell_id']!=c.CELL_ID or
                row['confirmed_PR']['best']['cell_id']!='B2_p5_q1_2nd_r4_K4'
                or row['confirmed_PR']['best']['G_exact']!='403423200'):
            raise ValueError('SAVED_COMPARISON_CHANGED')
    binding=deepcopy(original['binding']);archive=base.read_json(original['archive']['path'])
    if base.sha(original['archive']['path'])!=original['archive']['sha256']:raise ValueError('ARCHIVE_HASH')
    codec=archive['codec'];node=lambda ref:codec['nodes'][ref['ref']]
    full=node(codec['roots']['faithful:p19'])['fields'];blocks=node(full['deterministic_blocks'])['items']
    if full['hamiltonian_hash']!=binding['hamiltonian'] or len(blocks)!=20:raise ValueError('PREPARATION_BINDING')
    bound=1
    for ref in blocks[:8]:
        item=node(ref);fields=item['fields'];n=fields['num_system_qubits']
        pairs=0 if item['type']=='DFDeterministicOneBodySpec' else n*(n-1)//2
        bound+=4*len(node(fields['runtime_basis_operations'])['items'])+4*n+6*pairs+1
    if bound+4>c.plan()['native_instruction_cap']:raise ValueError('STATIC_INSTRUCTION_ADMISSION')
    sources=list(original['source_hashes'])+['src/trotterlib/track_a_v1_p7_contract.py',
        'scripts/track_a_v1_p7_runner.py','scripts/track_a_v1_p7_compile.py','scripts/prepare_track_a_v1_p7.py']
    receipts=[deepcopy(x) for x in original['files'] if x['logical_name'] in
        ('h6_input_snapshot.npz','snapshot_receipt.json','df_receipt.json','old_preparation_archive','font_seed')]
    if len(receipts)!=5:raise ValueError('FIVE_SCIENTIFIC_INPUT_RECEIPTS_REQUIRED')
    mappings=deepcopy({k:v for k,v in original['input_mapping'].items() if k in
        ('h6_input_snapshot.npz','snapshot_receipt.json','df_receipt.json')})
    files={'frozen_model.json':frozen_path,'phase_a_terminal.json':phase_a/'execution_terminal_v1.json',
        'phase_a_cleanup.json':phase_a/'resource_and_cleanup_v1.json','phase_b_validation.json':phase_b/'validation_complete.json',
        'phase_b_terminal.json':phase_b/'execution_terminal.json','phase_b_cleanup.json':phase_b/'launcher_terminal.json',
        'reaccount_terminal.json':ART/'track_a_v1_reaccount_result/2026-10-16/execution_terminal.json',
        'original_phase_a_signal_gzip':phase_a/'signal_results_v1.json.gz'}
    config=dict(schema='track_a_v1_p7_unapproved_config_v1',plan_sha256=c.digest(c.plan()),
        provider_root=original['provider_root'],provider_source=original['provider_source'],
        provider_hashes=original['provider_hashes'],python=original['python'],runtime_file_hashes=original['runtime_file_hashes'],
        source_hashes={name:base.sha(ROOT/name) for name in sources},font_seed=original['font_seed'],
        archive=original['archive'],compiler_options=deepcopy(c.OPTIONS),binding=binding,profile_reference=original['profile_reference'],
        resources=c.plan()['resources'],frozen_model_sha256=c.MODEL_SHA,frozen_model_digest=c.digest(frozen),
        p7_signal_digest=c.digest(signal),input_mapping=mappings,files=receipts,
        output=str(a.private_directory/'output01'),lock=str(a.private_directory/'p7_compile_ONE_SHOT'),control=str(a.private_directory/'control01'),
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None)
    # Verify historical SOURCE against the base commit, and pinned provider/runtime,
    # before writing any new draft. New-source files are not Git objects until commit.
    for name,h in original['source_hashes'].items():
        if base.sha(ROOT/name)!=h or hashlib.sha256(subprocess.check_output(['git','show',c.BASE+':'+name],cwd=ROOT)).hexdigest()!=h:
            raise ValueError('OLD_EXECUTION_SOURCE_CHANGED:'+name)
    run.verify_bytes(config,c.BASE)
    for name,h in config['provider_hashes'].items():
        if hashlib.sha256(subprocess.check_output(['git','show',c.PROVIDER+':'+name],cwd=config['provider_root'])).hexdigest()!=h:
            raise ValueError('PROVIDER_GIT_BLOB_CHANGED')
    a.private_directory.mkdir(mode=0o700);a.public_directory.mkdir(parents=True)
    put(a.private_directory/'p7_signal.json',signal)
    saved_binding=dict(schema='track_a_v1_p7_saved_binding_v1',base_commit=c.BASE,
        p7_signal_digest=c.digest(signal),original_phase_a_sha256=export['uncompressed_sha256'],
        predictions=predictions,full_H_reference=signal['reference'],scalar_receipt=signal['scalar_receipt'],
        allowance_digest=c.digest(signal['allowance']),runtime_closure_digest=c.digest(config['runtime_file_hashes']),
        profile_reference=original['profile_reference'],provider_closure_digest=c.digest(config['provider_hashes']),
        instruction_upper_bound=bound,wrapper_instruction_upper_bound=bound+4,
        prediction_status='FROZEN_MODEL_ESTIMATE_NOT_MEASURED',approved=False)
    put(a.public_directory/'saved_binding.json',saved_binding)
    files.update({'p7_signal.json':a.private_directory/'p7_signal.json','p7_saved_binding.json':a.public_directory/'saved_binding.json'})
    for name,path in files.items():
        config['input_mapping'][name]=str(path)
        config['files'].append(dict(logical_name=name,path=str(path),bytes=path.stat().st_size,sha256=base.sha(path)))
    put(a.private_directory/'config_UNAPPROVED_DRAFT.json',config)
    grant=dict(schema='track_a_v1_p7_compile_grant_v1',approved_by_user=False,runtime_authorization=False,
        scope=c.plan()['scope'],SOURCE=None,source_commit_binding='final committed HEAD, after explicit execution approval',
        config_sha256=base.sha(a.private_directory/'config_UNAPPROVED_DRAFT.json'),plan_sha256=c.digest(c.plan()),
        call_caps=c.plan()['call_caps'],resources=c.plan()['resources'],frozen_model_sha256=c.MODEL_SHA,
        p7_signal_digest=config['p7_signal_digest'],one_shot=True,retry=False,resume=False,
        allowed_cpus=[],driver_cpu=None,monitor_cpu=None,absolute_launch_command=None,next_stage_authorized=False)
    put(a.private_directory/'grant_UNAPPROVED_DRAFT.json',grant);put(a.public_directory/'grant_UNAPPROVED_DRAFT.json',grant)
    put(a.public_directory/'plan.json',c.plan())
    put(a.public_directory/'future_accounting_plan.json',dict(status='PLAN_ONLY_SEPARATE_EXECUTION_APPROVAL_REQUIRED',
        cell_id=c.CELL_ID,epsilon=.05,factors=[1.,10.],shots=dict(cosine=10216,sine=14356,total=24572),
        formula='G_det = 10216 * measured_RZ_cosine + 14356 * measured_RZ_sine',
        saved_PR=dict(cell_id='B2_p5_q1_2nd_r4_K4',G_exact='403423200',scope='CONFIRMATION_UNION8_N32',
                      kind='SAVED_PAIRED_SAMPLE_MEAN_NOT_POPULATION_MEAN'),
        reduction='100*(1-403423200/G_det); negative means PR is more expensive',
        measurement_gate='both registered axes complete, immutable bindings, per-axis relative error < 1/50',
        new_shot_calculation=False,new_signal=False,candidate_reselection=False,approved=False,
        compile_phase_auto_reaccount=False,conservative_bias_certificate=None,next_stage_authorized=False))
    for name,row in protected.items():
        path=Path(name)
        if path.stat().st_size!=row['bytes'] or base.sha(path)!=row['sha256']:raise ValueError('PROTECTED_FILE_CHANGED:'+name)
    put(a.private_directory/'protected_before.json',protected)
    from launch_track_a_v1_phase_a import snapshot,resource_reason
    resources=snapshot()
    put(a.public_directory/'metadata_audit.json',dict(status='READ_ONLY_METADATA_AUDIT_PASS',base_commit=c.BASE,
        review_receipt=dict(logical_name=a.review.name,bytes=a.review.stat().st_size,sha256=base.sha(a.review)),
        old_source_files_unchanged=len(original['source_hashes']),provider_files_unchanged=len(config['provider_hashes']),
        runtime_files_unchanged=len(config['runtime_file_hashes']),protected_files_unchanged=len(protected),
        protected_inventory_digest=c.digest(protected),input_receipts=len(config['files']),
        private_config_sha256=base.sha(a.private_directory/'config_UNAPPROVED_DRAFT.json'),
        source_hashes=config['source_hashes'],input_binding=binding,
        input_receipts_public=[{k:v for k,v in x.items() if k!='path'} for x in config['files']],
        effective_available_bytes_at_snapshot=resources['effective_available_bytes'],
        snapshot_resource_reason=resource_reason(resources,resources,c.plan()['resources']['aggregate_RSS_bytes']+16*2**30),
        allowed_cpu_count_at_snapshot=len(os.sched_getaffinity(0)),fresh_execution_gate_issued=False,
        new_scientific_imports=0,new_snapshot_decode=0,new_native_build=0,new_control_probe=0,new_compile=0,
        new_fit=0,new_grants_issued=0,unapproved_grant_drafts=1,wall_seconds=time.monotonic()-start,
        next_stage_authorized=False,mandatory_stop=True))
    print(json.dumps(dict(status='PREPARED_NOT_AUTHORIZED',prediction=c.PREDICTION,axes=2,
        protected_files=len(protected),runtime_files=len(config['runtime_file_hashes']),input_receipts=len(config['files']),
        new_compile=0,new_grants_issued=0),indent=2))


if __name__=='__main__':main()
