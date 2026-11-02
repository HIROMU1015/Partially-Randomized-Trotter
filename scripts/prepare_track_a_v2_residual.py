"""Seal an UNAPPROVED config and provenance audit; never evaluate real costs."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess

import track_a_v2_residual_runner as run

ROOT=run.ROOT
EXECUTION_SOURCES=('scripts/track_a_v2_residual_runner.py','src/trotterlib/track_a_v2_residual.py',
    'scripts/prepare_track_a_v2_residual.py','scripts/launch_track_a_v1_phase_a.py',
    'scripts/track_a_phase_a_parallel.py','scripts/track_a_v1_runner.py',
    'src/trotterlib/track_a_v1_contract.py','scripts/d0_saved_value_reanalysis_v1.py',
    'scripts/d0_frozen_shot_rule_v1.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('theory-root','runtime-config','private','public'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for path in (a.private,a.public):
        if not path.is_absolute() or path.resolve()!=path or path.exists() or not path.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('NEW_BOUND_PREPARATION_DIRECTORY_REQUIRED')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT):raise ValueError('COMMIT_SOURCE_FIRST')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.theory_root,text=True).strip()!=run.model.THEORY_SOURCE:
        raise ValueError('FIXED_THEORY_RESULT_REQUIRED')
    manifest=run.base.read_json(ROOT/'artifacts/resource_applicability/track_a_v2_residual_readiness/2026-10-16/input_manifest.json')
    receipts={x['path']:x for x in manifest['files'] if x['commit']==run.model.THEORY_SOURCE}
    art='artifacts/resource_applicability/track_a_native_df_theory_evaluation/2026-10-16/'
    names=dict(theory=art+'theoretical_wrapper_costs_v1.json',connections=art+'same_candidate_saved_cost_connection_v1.json.gz',
        index='artifacts/resource_applicability/track_a_native_df_metadata_completion/2026-10-16/saved_candidate_connection_index_v1.json',
        binding=art+'execution_binding_v1.json')
    inputs={}
    for name,relative in names.items():
        item=receipts[relative];path=a.theory_root/relative
        if path.stat().st_size!=item['bytes'] or run.base.sha(path)!=item['sha256']:raise ValueError('INPUT_CHANGED')
        import hashlib
        if hashlib.sha256(subprocess.check_output(['git','show',run.model.THEORY_SOURCE+':'+relative],cwd=a.theory_root)).hexdigest()!=item['sha256']:
            raise ValueError('THEORY_GIT_BLOB_CHANGED')
        inputs[name]=dict(path=str(path),bytes=item['bytes'],sha256=item['sha256'],commit=run.model.THEORY_SOURCE,repository_path=relative)
    # Only the saved identity index and independent execution-binding metadata.
    # No theory rows/compiled counts, probabilities, arrays or regression are evaluated.
    index=run.base.read_json(inputs['index']['path']);binding=run.base.read_json(inputs['binding']['path'])
    registry=[]
    for system,entries in index.items():
        fixed=binding[system+'_binding']
        for entry in entries:
            values={k:None for k in (*run.model.CRITICAL,*run.model.PROOF)}
            for name in ('input','hamiltonian','state','df','fragment_order'):values[name]=fixed[name]
            values.update(compiler_hash=entry['compiler_hash'],sample_scope=entry['scope'],
                origin_sources=entry.get('execution_sources'),wrapper_identity=entry.get('wrapper_identity_digest') or entry.get('axis_fingerprints'),
                dataset_seed_source=entry.get('dataset_seed_source') if entry['cell_id'].startswith('B2_') else 'NOT_APPLICABLE_DETERMINISTIC')
            registry.append(dict(key=[system,entry['scope'],entry['cell_id']],observed={k:entry[k] for k in
                ('cell_id','scope','cell','T','execution_sources','compiler_hash','paired_groups','wrapper_phase')},
                values=values,metadata_resolved=False,evidence_refs=[dict(input='index',sha256=inputs['index']['sha256']),
                    dict(input='binding',sha256=inputs['binding']['sha256'])],
                interpretation='EXISTING_METADATA_NOT_FULL_NUMERIC_ENVELOPE_NO_UNKNOWN_FIELDS_INFERRED'))
    audit=run.model.metadata_audit(index,registry)
    legacy=run.base.read_json(a.runtime_config)
    for name,expected in legacy['runtime_file_hashes'].items():
        if run.base.sha(name)!=expected:raise ValueError('PINNED_RUNTIME_CHANGED')
    sources={name:run.base.sha(ROOT/name) for name in EXECUTION_SOURCES}
    for name,h in sources.items():
        import hashlib
        if hashlib.sha256(subprocess.check_output(['git','show',source+':'+name],cwd=ROOT)).hexdigest()!=h:
            raise ValueError('SOURCE_NOT_COMMITTED')
    os.umask(0o077);a.private.mkdir(mode=0o700);a.public.mkdir(parents=True)
    run.write(a.private/'provenance_registry.json',registry)
    rpath=a.private/'provenance_registry.json'
    inputs['registry']=dict(path=str(rpath),bytes=rpath.stat().st_size,sha256=run.base.sha(rpath))
    export=run.base.read_json(a.theory_root/(art+'saved_connection_export_receipt_v1.json'))
    config=dict(schema='track_a_v2_residual_config_v1',source_freeze=source,policy_sha256=run.model.digest(run.model.policy()),
        source_hashes=sources,inputs=inputs,python=legacy['python'],runtime_file_hashes=legacy['runtime_file_hashes'],
        decompressed_connection=dict(bytes=export['uncompressed_bytes'],sha256=export['uncompressed_sha256']),
        output=str(a.private/'output01'),control=str(a.private/'control01'),lock=str(a.private/'residual_ONE_SHOT'),
        approved=False,runtime_authorization=False)
    run.write(a.private/'config_UNAPPROVED_DRAFT.json',config)
    grant=dict(schema='track_a_v2_residual_grant_v1',approved_by_user=False,runtime_authorization=False,
        metadata_conditional_interpretation_approved=False,SOURCE=None,scope=run.model.SCOPE,
        config_sha256=run.base.sha(a.private/'config_UNAPPROVED_DRAFT.json'),policy_sha256=config['policy_sha256'],
        limits=run.model.LIMITS,caps=run.model.CAPS,allowed_cpus=[],driver_cpu=None,monitor_cpu=None,
        one_shot=True,retry=False,resume=False,next_stage_authorized=False,absolute_launch_command=None)
    run.write(a.private/'grant_UNAPPROVED_DRAFT.json',grant);run.write(a.public/'grant_UNAPPROVED_DRAFT.json',grant)
    run.write(a.public/'provenance_audit.json',audit)
    run.write(a.public/'policy.json',run.model.policy())
    run.write(a.public/'source_input_binding.json',dict(source_freeze=source,source_hashes=sources,
        config_sha256=grant['config_sha256'],policy_sha256=grant['policy_sha256'],
        runtime_closure_digest=run.model.digest(config['runtime_file_hashes']),runtime_files=len(config['runtime_file_hashes']),
        python_executable_sha256=run.base.sha(legacy['python']),
        inputs={name:{k:v for k,v in x.items() if k!='path'} for name,x in inputs.items()},
        decompressed_connection=config['decompressed_connection'],
        provenance_counts=audit['status_counts'],reason_counts=dict(Counter(reason for row in audit['rows'] for reason in row['reasons'])),
        scientific_imports=0,new_fit=0,new_residual=0,new_ratio=0,new_compile=0,new_grants_issued=0,
        fresh_gate_issued=False,one_shot_lock_created=False,runtime_authorization=False))
    print(json.dumps(dict(status='PREPARED_NOT_AUTHORIZED',source_freeze=source,provenance=audit['status_counts'],new_numerical_analysis=0)))


if __name__=='__main__':main()
