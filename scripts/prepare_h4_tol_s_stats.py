"""Prepare only DF10 S_STATS metadata and an INVALID grant. Never run numerics."""
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import statistics  # Freeze the future stdlib NormalDist runtime, without calling it.
import subprocess
import sys

import h4_tol_s_stats_contract as s
c=s.c
PRIVATE=s.ROOT/'.h4_tol_s_stats_private'
PUBLIC=s.ROOT/'artifacts/resource_applicability/track_a_h4_tol_s_stats_preparation/2026-10-19'
OLD=s.ROOT.parent/'h4-tol-only-main-phase0-20261019/.h4_tol_main_private'


def preserve_prior():
    rows={}
    for path in (OLD/'preservation_manifest_PRIVATE.json',
        OLD/'D_seal01_20261019/completed_D_preservation_manifest_PRIVATE.json',
        OLD/'A_seal01_20261019/seal_preservation_manifest_PRIVATE.json',
        OLD/'A_seal01_20261019/completed_A_preservation_manifest_PRIVATE.json'):
        for row in c.read(path):
            for name in (row['path'],row.get('copy',row['path'])):
                c.require(name not in rows or rows[name]==row['sha256'],'CONSISTENT_OLD_HASHES')
                rows[name]=row['sha256']
        rows[str(path)]=c.sha(path)
    for directory in (OLD,OLD/'D_seal01_20261019',OLD/'A_seal01_20261019'):
        for path in directory.glob('*.json'):rows[str(path)]=c.sha(path)
    for path,h in rows.items():c.require(c.sha(path)==h,'PRIOR_RESULT_GRANT_LEDGER_MARKER_STOP_CHANGED')
    return [dict(path=p,sha256=h) for p,h in sorted(rows.items())]


def refusal(spec_path,grant_path,directory):
    guard=directory/'refusal_guard';guard.mkdir()
    (guard/'sitecustomize.py').write_text("import sys\ndef guard(event,args):\n if event=='import' and (args[0]=='h4_tol_s_stats_numeric' or args[0].startswith(('numpy','scipy','qiskit','trottertracks'))): raise AssertionError('NUMERICS_IMPORT_BEFORE_APPROVAL')\nsys.addaudithook(guard)\n")
    rows=[]
    for witness in (False,True):
        command=[sys.executable,str(s.ROOT/'scripts/run_h4_tol_s_stats.py'),'--spec',str(spec_path),
            '--grant',str(grant_path)]+(['--witness'] if witness else [])
        result=subprocess.run(command,capture_output=True,text=True,
            env=dict(os.environ,PYTHONPATH=str(guard),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1'))
        c.require(result.returncode!=0 and 'SEPARATE_EXPLICIT_S_STATS_APPROVAL_REQUIRED' in result.stderr
            and 'NUMERICS_IMPORT_BEFORE_APPROVAL' not in result.stderr,'REFUSED_BEFORE_NUMERIC_IMPORT')
        path=directory/('witness.stderr' if witness else 'direct.stderr');path.write_text(result.stderr)
        rows.append(dict(witness=witness,exit_code=result.returncode,stderr_sha256=c.sha(path),
            refused_before_numeric_import=True,input_summary_loaded=False,native_driver_started=False))
    return rows


def runtime_binding():
    paths={Path(sys.executable).resolve()}
    for module in list(sys.modules.values()):
        for field in ('__file__','__cached__'):
            name=getattr(module,field,None)
            if name:
                path=Path(name).resolve()
                if path.is_file() and s.ROOT not in path.parents:paths.add(path)
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split()
        if len(fields)>=6 and fields[-1].startswith('/'):
            path=Path(fields[-1]).resolve()
            if path.is_file() and s.ROOT not in path.parents:paths.add(path)
    c.require(not any(x in sys.modules for x in ('numpy','scipy','qiskit','h4_tol_s_stats_numeric')),
        'METADATA_ONLY_STDLIB_RUNTIME')
    return {str(path):c.sha(path) for path in sorted(paths)}


def prepare(source):
    c.require(not PRIVATE.exists() and not PUBLIC.exists(),'NEW_STATS_PREPARATION_NAMESPACE')
    prior=preserve_prior();PRIVATE.mkdir();store=PRIVATE/'inputs';store.mkdir();inputs=[]
    def fixed(role,commit,name):
        raw=subprocess.check_output(['git','show',commit+':'+name],cwd=s.ROOT)
        h=hashlib.sha256(raw).hexdigest();c.require(c.sha(s.ROOT/name)==h,'UNCHANGED_FIXED_INPUT')
        path=store/(role+'.json');path.write_bytes(raw)
        row=dict(role=role,commit=commit,repo_path=name,path=str(path),sha256=h);inputs.append(row)
        return c.read(path)
    stages={}
    for stage,commit,folder in [('A',s.A_COMMIT,'track_a_h4_tol_a_completed'),('D',s.D_COMMIT,'track_a_h4_tol_d_completed')]:
        base='artifacts/resource_applicability/'+folder+'/2026-10-19/'
        stages[stage]=fixed(stage+'_result',commit,base+'result.json')
        fixed(stage+'_audit',commit,base+'completion_audit.json')
        fixed(stage+'_native_exit',commit,base+'native_exit.json')
        execution=fixed(stage+'_execution_binding',commit,base+'execution_binding.json')
        c.require(execution['SOURCE']==s.SCIENCE_SOURCE,'SAME_FIXED_D_A_SCIENCE_SOURCE')
        manifest=fixed(stage+'_artifact_manifest',commit,base+'artifact_manifest.json')
        for name,h in manifest['files'].items():
            c.require(c.sha(s.ROOT/base/name)==h,'ORIGINAL_COMPLETION_ARTIFACT_SHA')
    draft=fixed('rank10_S_STATS_draft','80652ab6b42cf0c11ee97485214027f507b1bf0a',
        'artifacts/resource_applicability/track_a_h4_tol_main_phase0/2026-10-19/S_STATS_contract_DRAFT.json')
    c.require(draft['plan']['calls']==s.main.budgets()['S_STATS'],'INHERITED_SENSITIVITY_CONTRACT')
    summary=s.selected_summary(stages['A'],stages['D'])
    summary_path=PRIVATE/'selected_saved_summaries_PRIVATE.json';c.save(summary_path,summary)
    receipt=dict(D_formally_accepted=True,A_formally_accepted=True,fixed_D_commit=s.D_COMMIT,fixed_A_commit=s.A_COMMIT,
        instruction='H4 tol-only DF10 S_STATS最終準備',technical_preparation_authorized=True,
        S_STATS_numerical_execution_authorized=False,P_scientific_execution_authorized=False,
        sampling_compile_authorized=False,raw_statistics_authorized=False,
        independent_local_commit_authorized=True,new_public_push_explicitly_authorized=False,mandatory_STOP=True)
    c.save(PRIVATE/'preparation_authorization_PRIVATE.json',receipt)
    c.save(PRIVATE/'prior_preservation_manifest_PRIVATE.json',prior)
    for role,path in [('selected_summaries',summary_path),('preparation_receipt',PRIVATE/'preparation_authorization_PRIVATE.json')]:
        inputs.append(dict(role=role,path=str(path),sha256=c.sha(path)))
    source=subprocess.check_output(['git','rev-parse',source+'^{commit}'],cwd=s.ROOT,text=True).strip()
    hashes=s.source_closure()
    for name,h in hashes.items():
        raw=subprocess.check_output(['git','show',source+':'+name],cwd=s.ROOT)
        c.require(hashlib.sha256(raw).hexdigest()==h,'FIXED_S_STATS_SOURCE_REQUIRED')
    # Verify upstream code identities without loading their scientific adapters.
    upstream=stages['A']['SOURCE'];source_rows={}
    binding=c.read(store/'A_execution_binding.json')
    for name,h in binding['source_hashes'].items():
        raw=subprocess.check_output(['git','show',upstream+':'+name],cwd=s.ROOT)
        c.require(hashlib.sha256(raw).hexdigest()==h,'ACCEPTED_A_SOURCE_SHA')
        source_rows[name]=h
    c.save(PRIVATE/'upstream_scientific_SOURCE_PRIVATE.json',dict(SOURCE=upstream,files=source_rows))
    spec=dict(stage='S_STATS',scope=s.SCOPE,status='TECHNICALLY_SEALED_PENDING_EXPLICIT_STATS_APPROVAL',
        SOURCE=source,plan=s.plan(),source_hashes=hashes,inputs=inputs,summary=str(summary_path),
        preparation_receipt=str(PRIVATE/'preparation_authorization_PRIVATE.json'),runtime_hashes=runtime_binding(),
        python=sys.executable,python_sha256=c.sha(sys.executable),
        output=str(PRIVATE/'run01/output'),lock=str(PRIVATE/'run01/ONE_SHOT.json'),
        fresh_resources=dict(admitted=False,receipt=None,sha256=None),
        pending=['SEPARATE_EXPLICIT_S_STATS_NUMERICAL_APPROVAL','FRESH_RESOURCE_RECEIPT_AT_ACTIVATION'],
        approved_by_user=False,runtime_authorization=False,actual_numerical_passes=0,
        automatic_retry=False,next_stage_authorized=False,mandatory_STOP=True)
    spec_path=PRIVATE/'S_STATS_contract_SEALED_PENDING_APPROVAL_PRIVATE.json';c.save(spec_path,spec)
    grant=s.invalid_grant(spec);grant_path=PRIVATE/'S_STATS_grant_INVALID_PRIVATE.json';c.save(grant_path,grant)
    c.require(s.verify_bytes(spec)==summary,'TECHNICAL_BYTE_SUMMARY_CHECK')
    refused=refusal(spec_path,grant_path,PRIVATE)
    c.require(not Path(spec['output']).parent.exists(),'NO_NUMERIC_ATTEMPT_CREATED')
    for row in prior:c.require(c.sha(row['path'])==row['sha256'],'OLD_PRESERVATION_FINAL_CHECK')
    PUBLIC.mkdir(parents=True)
    c.save(PUBLIC/'S_STATS_contract_SEALED_PENDING_APPROVAL.json',dict(stage=spec['stage'],status=spec['status'],SOURCE=source,
        plan=spec['plan'],spec_digest=c.digest(spec),source_hashes=hashes,input_count=len(inputs),
        input_manifest=[{k:v for k,v in r.items() if k!='path'} for r in inputs],summary_sha256=c.sha(summary_path),
        runtime_file_count=len(spec['runtime_hashes']),runtime_binding_digest=c.digest(spec['runtime_hashes']),
        python_sha256=spec['python_sha256'],pending=spec['pending'],approved_by_user=False,runtime_authorization=False,
        actual_numerical_passes=0,output_created=False,one_shot_created=False,mandatory_STOP=True))
    c.save(PUBLIC/'selected_saved_summaries.json',summary)
    c.save(PUBLIC/'S_STATS_grant_INVALID.json',grant)
    c.save(PUBLIC/'upstream_SOURCE_provenance.json',dict(SOURCE=upstream,source_hashes=source_rows,
        A_result_sha256=next(x['sha256'] for x in inputs if x['role']=='A_result'),
        D_result_sha256=next(x['sha256'] for x in inputs if x['role']=='D_result')))
    c.save(PUBLIC/'refusal_report.json',dict(cases=2,rows=refused,all_refused_before_numeric_import=True,
        real_numerical_passes=0,native_driver_started=False,output_created=False,one_shot_created=False))
    c.save(PUBLIC/'approval_boundary.json',dict(receipt,launch_command=None,valid_grant_issued=False,
        future_activation='After separate numerical approval create new receipt/spec/grant; retain this seal and INVALID grant',
        P_n=None,P_SOURCE=None,P_seeds=None,P_budget=None,
        proposed_S_STATS_calls=s.plan()['calls'],rare_tail_inputs_missing='p2/p4, visibility, costs of unseen strata',
        no_extra_analysis_implicitly_authorized=True))
    resources=dict(observed_UTC=datetime.now(timezone.utc).isoformat(),allowed_CPUs=len(os.sched_getaffinity(0)),
        process_plan='one scalar pass over 24 scenarios, no heavy worker pool',agent_imposed_CPU_RAM_worker_wall_caps=None,
        fresh_resource_receipt_required_at_activation=True,admitted=False,shared_settings_modified=False)
    c.save(PUBLIC/'resource_observation.json',resources)
    audit=dict(status='S_STATS_TECHNICAL_PREPARATION_COMPLETE_PENDING_NUMERICAL_APPROVAL',SOURCE=source,
        source_files=len(hashes),input_files=len(inputs),runtime_files=len(spec['runtime_hashes']),
        fixed_primary_endpoints=2,normal_scenarios_reserved=24,sensitivity_settings=12,
        actual_numerical_passes=0,new_sampling_compile=0,P_science=0,new_seed_values=0,
        saved_values_reaggregated=False,old_FP_or_legacy_pooled=False,prior_records_preserved=len(prior),
        prior_manifest_sha256=c.sha(PRIVATE/'prior_preservation_manifest_PRIVATE.json'),
        source_input_runtime_SHA='PASS',n_or_power_computed_from_real_data=False,
        P_n=None,P_budget=None,mandatory_STOP=True,AI_subagents_used=False,new_public_push=False)
    c.save(PUBLIC/'completion_audit.json',audit)
    c.save(PUBLIC/'artifact_manifest.json',dict(files={p.name:c.sha(p) for p in sorted(PUBLIC.glob('*.json'))
        if p.name!='artifact_manifest.json'},scientific_results_generated=False))
    c.save(PRIVATE/'technical_seal_preservation_manifest_PRIVATE.json',[
        dict(path=str(p),sha256=c.sha(p)) for p in sorted(PRIVATE.rglob('*')) if p.is_file()])
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);args=parser.parse_args()
    import json
    print(json.dumps(prepare(args.source)))
