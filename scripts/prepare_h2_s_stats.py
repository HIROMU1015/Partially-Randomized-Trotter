"""Freeze H2 S_STATS metadata, then bind the user's preparation-through-run approval."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import h2_s_stats_contract as s
from prepare_h2_panels import preservation, row
from prepare_h4_tol_s_stats import runtime_binding
c=s.c


def preserve_prior():
    preserved={}
    manifests=[s.PRIVATE/'phase0_preparation01/preparation_preservation_manifest_PRIVATE.json',
        s.PRIVATE/'activation01/F_completed_preservation_manifest_PRIVATE.json',
        s.PRIVATE/'FP_activation01/FP_completed_preservation_manifest_PRIVATE.json',
        s.PRIVATE/'D_interruption_audit01/original_D_preservation_manifest_PRIVATE.json',
        s.PRIVATE/'D_recovery_activation01/D_completed_preservation_manifest_PRIVATE.json',
        s.PRIVATE/'A_seal01/seal_preservation_manifest_PRIVATE.json',
        s.PRIVATE/'A_activation01/A_completed_preservation_manifest_PRIVATE.json']
    for manifest in manifests:
        for r in c.read(manifest):
            path=r['path'];c.require(path not in preserved or preserved[path]==r['sha256'], 'CONSISTENT_PRIOR_HASHES')
            preserved[path]=r['sha256']
        preserved[str(manifest)]=c.sha(manifest)
    for path,h in preserved.items():c.require(c.sha(path)==h, 'PRIOR_EVIDENCE_PRESERVED')
    return [dict(path=p,sha256=h) for p,h in sorted(preserved.items())]


def refusal(spec_path,grant_path,directory):
    results=[]
    guard="import sys,runpy;from pathlib import Path\ndef hook(event,args):\n if event=='import' and (args[0] in ('h2_s_stats_numeric','h4_tol_s_stats_numeric') or args[0].startswith(('numpy','scipy','qiskit','pyscf'))):raise AssertionError('NUMERIC_IMPORT_BEFORE_APPROVAL')\nsys.addaudithook(hook);sys.path.insert(0,str(Path(sys.argv[1]).parent));sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    for witness in (False,True):
        command=[sys.executable,'-B','-S','-c',guard,str(s.ROOT/'scripts/run_h2_s_stats.py'),
            '--spec',str(spec_path),'--grant',str(grant_path)]+(['--witness'] if witness else [])
        proc=subprocess.run(command,capture_output=True,text=True)
        c.require(proc.returncode!=0 and 'EXPLICIT_H2_S_STATS_APPROVAL_REQUIRED' in proc.stderr
            and 'NUMERIC_IMPORT_BEFORE_APPROVAL' not in proc.stderr, 'INVALID_STATS_GRANT_REJECTED_FIRST')
        target=directory/('refusal_witness.txt' if witness else 'refusal_direct.txt');target.write_text(proc.stderr)
        results.append(dict(witness=witness,returncode=proc.returncode,refused_before_numeric_import=True,
            stderr_sha256=c.sha(target),output_created=False,one_shot_created=False))
    return results


def prepare(source):
    hashes=s.source_closure()
    c.require(len(source)==40, 'FULL_COMMITTED_STATS_SOURCE')
    for rel,h in hashes.items():
        blob=subprocess.check_output(['git','--no-optional-locks','show',source+':'+rel],cwd=s.ROOT)
        c.require(hashlib.sha256(blob).hexdigest()==h, 'COMMITTED_STATS_SOURCE_BYTES')
    rel='scripts/h4_tol_s_stats_numeric.py'
    original=subprocess.check_output(['git','--no-optional-locks','show',s.SHARED_KERNEL_SOURCE+':'+rel],cwd=s.ROOT)
    c.require(hashlib.sha256(original).hexdigest()==hashes[rel], 'UNMODIFIED_VALIDATED_H4_NUMERIC_KERNEL')
    c.require(not (s.PRIVATE/'attempts/S_STATS_run01').exists(), 'S_STATS_UNSTARTED')
    prior=preserve_prior();s.STORE.mkdir(exist_ok=False);s.PUBLIC.mkdir(exist_ok=False)
    store=s.STORE/'inputs';store.mkdir();inputs=[]
    def fixed(role,commit,relative):
        blob=subprocess.check_output(['git','--no-optional-locks','show',commit+':'+relative],cwd=s.ROOT)
        h=hashlib.sha256(blob).hexdigest();c.require(c.sha(s.ROOT/relative)==h, 'FIXED_RESULT_COMMIT_BYTES')
        target=store/(role+'.json');target.write_bytes(blob)
        inputs.append(dict(role=role,path=str(target),sha256=h,commit=commit,repo_path=relative))
        return c.read(target)
    saved={}
    for stage,commit,folder in [('A',s.A_COMMIT,'A_completed_run01'),('D',s.D_COMMIT,'D_completed_run02')]:
        base=str((s.main.PUBLIC/folder).relative_to(s.ROOT))+'/'
        saved[stage]=fixed(stage+'_result',commit,base+'result.json')
        saved[stage+'_audit']=fixed(stage+'_audit',commit,base+'native_completion_audit.json')
        fixed(stage+'_native_exit',commit,base+'native_exit.json')
        fixed(stage+'_execution_binding',commit,base+'execution_binding.json')
        manifest=fixed(stage+'_artifact_manifest',commit,base+'artifact_manifest.json')
        for r in manifest['files']:c.require(c.sha(s.ROOT/base/r['path'])==r['sha256'], 'UPSTREAM_PUBLIC_ARTIFACTS')
    summary=s.selected_summary(saved['A'],saved['D'],saved['A_audit'])
    summary_path=s.STORE/'selected_saved_summaries.json';c.save(summary_path,summary)
    receipt=dict(role='user',verbatim_user_message=s.MESSAGE,selected_response_text='S_STATS契約を技術準備',
        A_formally_accepted=True,D_formally_accepted=True,A_commit=s.A_COMMIT,D_commit=s.D_COMMIT,
        authorized_scope='Prepare and freeze H2 S_STATS, execute its sole numerical design pass, audit, then STOP',
        numerical_execution_authorized_after_fixed_SOURCE_inputs_budget=True,
        raw_aggregation_bootstrap_sampling_compile_law_reconstruction_authorized=False,
        P_scientific_execution_authorized=False,new_public_push_authorized=False)
    c.save(s.STORE/'preparation_authorization.json',receipt)
    c.save(s.STORE/'prior_preservation_manifest_PRIVATE.json',prior)
    for role,path in [('selected_summaries',summary_path),('preparation_receipt',s.STORE/'preparation_authorization.json'),
                      ('prior_preservation',s.STORE/'prior_preservation_manifest_PRIVATE.json')]:
        inputs.append(row(path,role))
    c.require(not any(name in sys.modules for name in ('h2_s_stats_numeric','h4_tol_s_stats_numeric','numpy','scipy','qiskit')),
        'METADATA_PREPARATION_NO_NUMERIC_IMPORT')
    spec=dict(scope=s.SCOPE,stage='S_STATS',status='SEALED_PENDING_ACTIVATION',SOURCE=source,plan=s.plan(),
        source_hashes=hashes,inputs=inputs,summary=str(summary_path),
        preparation_receipt=str(s.STORE/'preparation_authorization.json'),
        prior_manifest=str(s.STORE/'prior_preservation_manifest_PRIVATE.json'),
        runtime_hashes=runtime_binding(),python=sys.executable,python_sha256=c.sha(sys.executable),
        attempt=str(s.PRIVATE/'attempts/S_STATS_run01'),output=str(s.PRIVATE/'attempts/S_STATS_run01/output'),
        lock=str(s.PRIVATE/'attempts/S_STATS_run01/ONE_SHOT.json'),fresh_resources=dict(admitted=False,receipt=None,sha256=None),
        pending=['BIND_EXISTING_USER_APPROVAL_TO_FIXED_SOURCE_AND_BUDGET','FRESH_RESOURCE_ADMISSION'],
        actual_numerical_passes=0,automatic_retry=False,next_stage_authorized=False)
    path=s.STORE/'S_STATS_spec_PENDING.json';c.save(path,spec)
    grant=s.invalid_grant(spec);grant_path=s.STORE/'S_STATS_grant_INVALID.json';c.save(grant_path,grant)
    c.require(s.verify_bytes(spec)==summary, 'FIXED_STATS_SUMMARY_AND_BYTES')
    refused=refusal(path,grant_path,s.STORE)
    c.require(not Path(spec['attempt']).exists(), 'METADATA_PREPARATION_NO_ATTEMPT')
    for name,value in [('selected_saved_summaries.json',summary),('preparation_authorization.json',receipt),
        ('S_STATS_grant_INVALID.json',grant),('refusal_report.json',dict(status='PASS',cases=refused,
            actual_numerical_passes=0,output_created=False,one_shot_created=False))]:c.save(s.PUBLIC/name,value)
    c.save(s.PUBLIC/'preregistration.json',dict(SOURCE=source,plan=spec['plan'],source_hashes=hashes,
        pending_spec_sha256=c.sha(path),spec_digest=c.digest(spec),inputs=inputs,
        runtime_files=len(spec['runtime_hashes']),runtime_digest=c.digest(spec['runtime_hashes']),
        python_sha256=spec['python_sha256'],summary_sha256=c.sha(summary_path),
        existing_user_approval_received=True,actual_numerical_passes=0,valid_grant_created=False,
        no_SD_n_power_computation_in_preparation=True))
    c.save(s.STORE/'technical_seal_preservation_manifest_PRIVATE.json',[
        row(p,str(p.relative_to(s.STORE))) for p in sorted(s.STORE.rglob('*')) if p.is_file()])
    return dict(status='TECHNICAL_SEAL_READY_FOR_AUTHORIZED_ACTIVATION',SOURCE=source,
        source_files=len(hashes),input_files=len(inputs),runtime_files=len(spec['runtime_hashes']),
        prior_records_preserved=len(prior),calls=spec['plan']['calls'],actual_numerical_passes=0)


def activate():
    pending=s.STORE/'S_STATS_spec_PENDING.json';spec=c.read(pending);s.verify_bytes(spec)
    c.require(not Path(spec['attempt']).exists(), 'UNUSED_S_STATS_ATTEMPT')
    s.ACTIVATION.mkdir(exist_ok=False)
    observed=s.main.f.observation()
    c.require(observed['allowed_cpus'] and observed['memory_bytes']['MemAvailable']>0, 'AVAILABLE_STATS_RESOURCES')
    resource=s.ACTIVATION/'resources.json'
    c.save(resource,dict(admitted=True,checked_at_UTC=datetime.now(timezone.utc).isoformat(),
        observations=observed,processes=1,reason='24 scalar scenarios; no heavy work requiring a worker pool',
        agent_imposed_CPU_RAM_worker_wall_caps=None,affinity_modified=False,shared_settings_modified=False,
        PSI_diagnostic_only=True,other_processes_modified=False))
    approval=s.ACTIVATION/'approval_receipt.json'
    c.save(approval,dict(role='user',stage='S_STATS',SOURCE=spec['SOURCE'],plan_digest=c.digest(s.plan()),
        verbatim_user_message=s.MESSAGE,selected_response_text='S_STATS契約を技術準備',
        scope='Technical preparation followed by one sealed H2 S_STATS numerical pass and audit; mandatory STOP',
        pending_spec_sha256=c.sha(pending),call_budget=s.plan()['calls'],
        P_scientific_execution_authorized=False,next_stage_authorized=False,new_public_push_authorized=False))
    spec=deepcopy(spec);spec.update(status='SEALED_FOR_EXECUTION',pending=[],pending_spec=str(pending),
        pending_spec_sha256=c.sha(pending),fresh_resources=dict(admitted=True,receipt=str(resource),sha256=c.sha(resource)))
    grant=dict(scope=s.SCOPE,stage='S_STATS',SOURCE=spec['SOURCE'],spec_digest=c.digest(spec),
        approved_by_user=True,runtime_authorization=True,one_shot=True,automatic_retry=False,
        next_stage_authorized=False,approval_receipt=str(approval),approval_receipt_sha256=c.sha(approval))
    c.save(s.ACTIVATION/'S_STATS_spec_EXECUTION.json',spec)
    c.save(s.ACTIVATION/'S_STATS_grant_APPROVED_PRIVATE.json',grant);s.authorize(spec,grant)
    c.save(s.ACTIVATION/'preflight_identity_gate.json',dict(status='PASS',SOURCE=spec['SOURCE'],
        spec_digest=c.digest(spec),grant_digest=c.digest(grant),scientific_execution_started=False))
    c.save(s.ACTIVATION/'activation_preservation_manifest.json',[
        row(p,str(p.relative_to(s.ACTIVATION))) for p in sorted(s.ACTIVATION.iterdir()) if p.is_file()])
    return dict(status='S_STATS_PREFLIGHT_PASS',SOURCE=spec['SOURCE'],numeric_passes_authorized=1,
        normal_scenarios=24,fixed_directions=2,sensitivity_settings=12,scientific_execution_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source');parser.add_argument('--activate',action='store_true');args=parser.parse_args()
    if args.activate:result=activate()
    else:
        c.require(args.source, 'COMMITTED_S_STATS_SOURCE_REQUIRED');result=prepare(args.source)
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
