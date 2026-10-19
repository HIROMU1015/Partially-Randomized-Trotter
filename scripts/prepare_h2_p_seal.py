"""H2 P metadata seal only. Never activates a grant or runs a trajectory."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import h2_p_contract as c
import h4_p_seed_inventory as inventory
from prepare_h2_panels import row

MESSAGE='S_STATSの結果を正式採用し、P最終sealを実施してください。'


def preserve_prior():
    previous=c.PRIVATE/'S_STATS_preparation01/prior_preservation_manifest_PRIVATE.json'
    rows={r['path']:r['sha256'] for r in c.read(previous)}
    rows[str(previous)]=c.sha(previous)
    for directory in ('S_STATS_preparation01','S_STATS_activation01','attempts/S_STATS_run01'):
        for p in (c.PRIVATE/directory).rglob('*'):
            if p.is_file():rows[str(p)]=c.sha(p)
    for name in ('S_STATS_tests01.json','S_STATS_tests01.txt','S_STATS_seal_stdout01.txt',
                 'S_STATS_seal_stderr01.txt','S_STATS_audit_stdout01.txt','S_STATS_audit_stderr01.txt',
                 'S_STATS_reporting_verification01.json'):
        p=c.PRIVATE/name;rows[str(p)]=c.sha(p)
    # Recheck original completed-stage manifests, rather than regenerate evidence.
    for manifest in (c.PRIVATE/'S_STATS_preparation01/technical_seal_preservation_manifest_PRIVATE.json',
                     c.PRIVATE/'S_STATS_activation01/activation_preservation_manifest.json',
                     c.PRIVATE/'S_STATS_preparation01/completed_preservation_manifest_PRIVATE.json'):
        for r in c.read(manifest):c.require(c.sha(r['path'])==r['sha256'],'PREVIOUS_S_STATS_EVIDENCE_CHANGED')
    for path,h in rows.items():c.require(c.sha(path)==h,'PRIOR_H2_SOURCE_RESULT_GRANT_LEDGER_STOP')
    return [dict(path=p,sha256=h) for p,h in sorted(rows.items())]


def seed_metadata_values(value):
    """Conservative metadata projection; never draw Taylor orders/components.

    All explicit integer fields are excluded as a superset, and every declared
    ordinary task with cell.q also contributes its original derived step seeds.
    The inherited exclusion ledger retains previously derived conditional seeds.
    """
    found=set()
    def walk(v):
        if type(v) is int:
            if 0<=v<2**64:found.add(v)
        elif isinstance(v,list):
            for x in v:walk(x)
        elif isinstance(v,dict):
            if type(v.get('seed')) is int and isinstance(v.get('cell'),dict) and type(v['cell'].get('q')) is int:
                found.update(c.base.stats.seed_hierarchy(v['seed'],v['cell']['q']))
            for x in v.values():walk(x)
    walk(value)
    return found


def freeze_seed_inventory():
    inherited=c.read(c.base.STORE/'old_seed_values.json')['values']
    excluded=set(inherited);origins=[];destination=c.STORE/'seed_metadata';destination.mkdir()
    known=c.read(c.PRIVATE/'FP_activation01/seed_origins.json')
    for origin in known:
        frozen=Path(origin['snapshot']);c.require(c.sha(frozen)==origin['sha256'],'PREVIOUS_SEED_SNAPSHOT')
        current=Path(origin['origin']);data=current.read_bytes();h=hashlib.sha256(data).hexdigest()
        # The inherited derived-seed proof applies to this known inventory.
        # A changed input needs a new derivation audit, never a blanket worktree freeze.
        c.require(h==origin['sha256'],'KNOWN_SEED_INPUT_CHANGED_NEEDS_DERIVATION_AUDIT:'+origin['role'])
        snapshot=destination/h
        if not snapshot.exists():snapshot.write_bytes(data)
        values=seed_metadata_values(c.read(snapshot));excluded.update(values)
        origins.append(dict(role=origin['role'],origin=str(current),snapshot=str(snapshot),sha256=h,
            metadata_integer_superset=len(values),changed_since_previous_seal=False))
    for stage in ('FP','A'):
        path=c.base.STORE/(stage+'_seeds.json');r=c.read(path);excluded.update(c.base.seed_values(r['tasks']))
        origins.append(dict(role='H2_'+stage+'_RESERVED',origin=str(path),snapshot=str(path),sha256=c.sha(path)))
    geometry=c.read(c.base.STORE/'seed_origins.json')[-1]
    data=Path(geometry['origin']).read_bytes();h=hashlib.sha256(data).hexdigest()
    c.require(h==geometry['sha256'],'REGISTERED_GEOMETRY_SEED_CASES_CHANGED')
    path=destination/h;path.write_bytes(data)
    origins.append(dict(role='GEOMETRY_RESERVED_SEED_CASES',origin=geometry['origin'],snapshot=str(path),sha256=h))
    c.save(c.STORE/'seed_origins.json',origins)
    c.save(c.STORE/'seed_exclusions_PRIVATE.json',dict(values=sorted(excluded),
        inherited_explicit_values=len(inherited),H2_FP_A_reserved_values=3892,
        explicit_metadata_integer_superset=True,actual_RTE_event_sampling=False,
        future_unregistered_campaigns_covered=False))
    return excluded,c.read(path)


def refusal(spec_path,grant_path):
    entry=c.ROOT/'scripts/run_h2_p.py';results=[]
    guard="import sys,runpy;from pathlib import Path\ndef hook(event,args):\n if event=='import' and (args[0] in ('h2_p_science','h2_panels_science') or args[0].startswith(('numpy','scipy','qiskit','pyscf','trotterlib','trottertracks'))):raise AssertionError('SCIENCE_BEFORE_APPROVAL')\nsys.addaudithook(hook);sys.path.insert(0,str(Path(sys.argv[1]).parent));sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')"
    for label,flags in [('driver',[]),('worker',['--worker'])]:
        proc=subprocess.run([sys.executable,'-B','-S','-c',guard,str(entry),'--spec',str(spec_path),
            '--grant',str(grant_path),*flags],capture_output=True,text=True)
        c.require(proc.returncode==1 and 'SEPARATE_EXPLICIT_P_SCIENCE_APPROVAL_REQUIRED' in proc.stderr
            and 'SCIENCE_BEFORE_APPROVAL' not in proc.stderr,'INVALID_P_GRANT_REJECTED_FIRST')
        path=c.STORE/('refusal_'+label+'.txt');path.write_text(proc.stderr)
        results.append(dict(entry=label,exit_code=proc.returncode,refused_before_scientific_import=True,
            stderr_sha256=c.sha(path),attempt_created=False,marker_created=False,output_created=False))
    command="import sys;sys.path.insert(0,sys.argv[1]);import h2_p_science as p;import h2_p_contract as c;p.execute(c.read(sys.argv[2]),c.read(sys.argv[3]),None)"
    proc=subprocess.run([sys.executable,'-B','-S','-c',command,str(c.ROOT/'scripts'),str(spec_path),str(grant_path)],
        capture_output=True,text=True)
    c.require(proc.returncode==1 and 'SEPARATE_EXPLICIT_P_SCIENCE_APPROVAL_REQUIRED' in proc.stderr,
        'P_ADAPTER_ALSO_REQUIRES_APPROVAL')
    path=c.STORE/'refusal_adapter.txt';path.write_text(proc.stderr)
    results.append(dict(entry='adapter',exit_code=1,refused_before_scientific_package_import=True,
        stderr_sha256=c.sha(path),attempt_created=False,marker_created=False,output_created=False))
    c.require(not (c.PRIVATE/'attempts/P_run01').exists(),'P_ATTEMPT_NOT_CREATED_BY_REFUSAL')
    return dict(status='PASS',cases=results,P_scientific_execution=0,valid_grant_created=False)


def verify_seal():
    spec=c.read(c.STORE/'P_spec_FINAL_PENDING.json');c.verify(spec)
    c.require(spec['status']=='SEALED_TECHNICAL_PENDING_USER_APPROVAL' and spec['pending']==c.PENDING
        and spec['worker_cpus']==[] and spec['resource_receipt'] is None,'UNACTIVATED_P_SEAL')
    grant=c.read(c.STORE/'P_grant_INVALID.json');c.require(grant==c.invalid_grant(),'P_GRANT_REMAINS_INVALID')
    for r in c.read(c.STORE/'seal_preservation_manifest_PRIVATE.json'):
        c.require(c.sha(r['path'])==r['sha256'],'H2_P_SEAL_PRESERVED')
    c.require(not (c.PRIVATE/'attempts/P_run01').exists() and not (c.PRIVATE/'P_activation01').exists(),
        'NO_P_ACTIVATION_ATTEMPT_SAMPLING_COMPILE')
    return dict(status='FINAL_SEAL_MANDATORY_STOP',SOURCE=spec['SOURCE'],n_fixed=c.N_FIXED,calls=c.CALLS,
        seed_registry_sha256=c.sha(spec['seed_registry']),seed_plan_digest=spec['seed_plan_digest'],
        spec_digest=c.digest(spec),source_files=len(spec['source_hashes']),input_files=len(spec['inputs']),
        provider_files=len(spec['provider_files']),kernel_files=len(spec['kernel_files']),
        runtime_files=len(c.read(spec['runtime_manifest'])),prior_records=len(c.read(spec['prior_manifest'])),
        seal_records=len(c.read(c.STORE/'seal_preservation_manifest_PRIVATE.json')),
        scientific_execution=0,sampling=0,compile=0,new_n_SD_power_analysis=0,
        valid_grant_created=False,P_attempt_created=False,automatic_retry=False,next_stage_authorized=False)


def seal(source):
    closure=c.source_closure()
    c.require(len(source)==40,'COMMITTED_P_SOURCE_REQUIRED')
    for rel,h in closure.items():
        blob=subprocess.check_output(['git','--no-optional-locks','show',source+':'+rel],cwd=c.ROOT)
        c.require(hashlib.sha256(blob).hexdigest()==h,'P_SOURCE_MUST_BE_COMMITTED')
    c.require(not (c.PRIVATE/'attempts/P_run01').exists() and not (c.PRIVATE/'P_activation01').exists(), 'P_UNSTARTED')
    prior=preserve_prior();c.STORE.mkdir(exist_ok=False);c.EVIDENCE.mkdir(exist_ok=False)
    inputs=[];folder=c.STORE/'inputs';folder.mkdir();dependencies={};results={}
    for stage,(commit,directory,scientific_source) in c.DEPENDENCIES.items():
        dep=dict(commit=commit,folder=directory,SOURCE=scientific_source)
        for key,name in [('result','result.json'),('audit','completion_audit.json' if stage=='S_STATS' else 'native_completion_audit.json'),
                         ('native','native_exit.json'),('binding','execution_binding.json'),('manifest','artifact_manifest.json')]:
            public=c.PUBLIC/directory/name;rel=str(public.relative_to(c.ROOT))
            blob=subprocess.check_output(['git','--no-optional-locks','show',commit+':'+rel],cwd=c.ROOT)
            h=hashlib.sha256(blob).hexdigest();c.require(h==c.sha(public),'COMMITTED_STAGE_INPUT_BYTES')
            dest=folder/(stage+'_'+key+'.json');dest.write_bytes(blob)
            inputs.append(dict(role=stage+'_'+key,path=str(dest),sha256=h,bytes=len(blob),commit=commit,repo_path=rel))
            dep[key]=str(dest)
        manifest=c.read(dep['manifest'])
        for r in manifest['files']:c.require(c.sha(c.PUBLIC/directory/r['path'])==r['sha256'],'UPSTREAM_PUBLIC_EVIDENCE')
        dependencies[stage]=dep;results[stage]=c.read(dep['result'])
    endpoints=c.selected_endpoints(results['A'],results['D'],results['S_STATS'])
    adoption=dict(role='user',verbatim_user_message=MESSAGE,accepted_S_STATS_commit=c.STATS_COMMIT,
        adopted_n=c.N_FIXED,adopted_trajectory=92,adopted_compiled_axes=184,adopted_RTE_event=256,
        purpose='H2 P final technical seal only',P_scientific_execution_authorized=False,
        rare_event_cost=None,rare_event_variance=None,population_mean_certified=False,new_public_push_authorized=False)
    c.save(c.STORE/'adoption_receipt.json',adoption);c.save(c.STORE/'prior_preservation_manifest_PRIVATE.json',prior)
    excluded,cases=freeze_seed_inventory()
    seedplan=c.seed_plan(source,endpoints,dependencies,c.sha(c.STORE/'seed_exclusions_PRIVATE.json'))
    pd=c.digest(seedplan);registry=dict(SOURCE=source,seed_plan_digest=pd,namespace=c.SCOPE,
        tasks=c.tasks(source,pd),actual_RTE_event_sampling=False,redraw_on_collision=False)
    values=c.check_tasks(registry,source,pd,excluded);registry_path=c.STORE/'seed_registry_PRIVATE.json';c.save(registry_path,registry)
    start=time.monotonic();checks,workers=inventory.check_geometry(cases,values)
    c.require(all(not r['collisions'] for r in checks),'P_GEOMETRY_SEED_COLLISION_NO_REDRAW')
    receipt=dict(status='COLLISION_ZERO',checked_at_UTC=datetime.now(timezone.utc).isoformat(),
        new_values_digest=c.digest(sorted(values)),seed_registry_sha256=c.sha(registry_path),
        master_seeds=92,seed_slots=140,old_exclusion_values=len(excluded),geometry_checks=checks,
        geometry_workers=workers,wall_seconds=time.monotonic()-start,seed_redrawn=False,new_RTE_sampling=0,
        activation_refresh_required=True,future_unregistered_campaigns_covered=False,
        zero_collision_is_independence_proof=False,metadata_integer_superset=True)
    c.save(c.STORE/'seed_check.json',receipt)
    original=c.read(c.PRIVATE/'A_activation01/A_spec_EXECUTION.json')
    for r in original['inputs']:
        if r['path'] in original['native_config']['mapping'].values():inputs.append(deepcopy(r))
    # Binding the original accepted A spec also retains its complete dependency chain.
    inputs.append(row(c.PRIVATE/'A_activation01/A_spec_EXECUTION.json','original_A_execution_spec'))
    for path,role in [(c.STORE/'adoption_receipt.json','S_STATS_user_adoption'),
        (c.STORE/'prior_preservation_manifest_PRIVATE.json','prior_preservation'),
        (registry_path,'P_seed_registry'),(c.STORE/'seed_check.json','P_seed_check'),
        (c.STORE/'seed_exclusions_PRIVATE.json','seed_exclusions'),(c.STORE/'seed_origins.json','seed_origins')]:
        inputs.append(row(path,role))
    for p in sorted((c.STORE/'seed_metadata').iterdir()):inputs.append(row(p,'seed_snapshot_'+p.name))
    spec=dict(scope=c.SCOPE,stage='P',status='SEALED_TECHNICAL_PENDING_USER_APPROVAL',SOURCE=source,
        plan=c.plan(),calls=dict(c.CALLS),source_hashes=closure,inputs=inputs,dependencies=dependencies,
        endpoints=endpoints,seed_plan=seedplan,seed_plan_digest=pd,seed_registry=str(registry_path),
        seed_check=str(c.STORE/'seed_check.json'),seed_exclusions=str(c.STORE/'seed_exclusions_PRIVATE.json'),
        seed_origins=str(c.STORE/'seed_origins.json'),prior_manifest=str(c.STORE/'prior_preservation_manifest_PRIVATE.json'),
        adoption_receipt=str(c.STORE/'adoption_receipt.json'),attempt=str(c.PRIVATE/'attempts/P_run01'),
        worker_cpus=[],resource_receipt=None,resource_receipt_sha256=None,pending=list(c.PENDING),
        activation_rule='Only separate P science approval may add approval/resource/fresh-seed receipts; all other sealed fields immutable')
    for k in ('native_config','provider_roots','provider_files','kernel_files','A_kernel_SOURCE','A_kernel_scripts',
              'runtime_manifest','runtime_manifest_sha256','python','python_sha256'):spec[k]=deepcopy(original[k])
    c.verify(spec)
    path=c.STORE/'P_spec_FINAL_PENDING.json';c.save(path,spec);grant=c.STORE/'P_grant_INVALID.json';c.save(grant,c.invalid_grant())
    refused=refusal(path,grant);c.save(c.STORE/'refusal_report.json',refused)
    resource=c.base.f.observation();c.save(c.STORE/'resource_observation.json',dict(observation=resource,
        science_admitted=False,worker_cpus_assigned=False,PSI_diagnostic_only=True,
        agent_imposed_CPU_RAM_worker_wall_caps=None,shared_settings_modified=False))
    for name in ('adoption_receipt.json','seed_check.json','refusal_report.json','resource_observation.json','P_grant_INVALID.json'):
        (c.EVIDENCE/name).write_bytes((c.STORE/name).read_bytes())
    c.save(c.EVIDENCE/'P_contract_TECHNICALLY_SEALED.json',dict(status=spec['status'],SOURCE=source,
        private_spec_sha256=c.sha(path),spec_digest=c.digest(spec),plan=spec['plan'],calls=spec['calls'],
        source_hashes=closure,inputs=inputs,dependencies=dependencies,
        endpoints=[{k:v for k,v in ep.items() if k!='signal'} for ep in endpoints],
        seed_plan_digest=pd,private_seed_registry_sha256=c.sha(registry_path),
        master_seeds=92,seed_slots=140,runtime_files=len(c.read(spec['runtime_manifest'])),
        runtime_manifest_sha256=spec['runtime_manifest_sha256'],provider_files=len(spec['provider_files']),
        kernel_files=len(spec['kernel_files']),python_sha256=spec['python_sha256'],pending=c.PENDING,
        P_scientific_execution=0,P_grant_valid=False,mandatory_STOP=True))
    c.save(c.EVIDENCE/'P_execution_approval_request.json',dict(stage='P',SOURCE=source,
        adopted_n=c.N_FIXED,calls=c.CALLS,seed_registry_sha256=c.sha(registry_path),seed_plan_digest=pd,
        separate_user_science_approval_required=True,approval_received=False,
        fresh_resources_required=True,fresh_seed_inventory_check_required=True,
        additional_sampling=False,automatic_retry=False,candidate_reselection=False,
        rare_event_cost=None,rare_event_variance=None,population_mean_certified=False,mandatory_STOP=True))
    c.save(c.STORE/'seal_preservation_manifest_PRIVATE.json',[row(p,str(p.relative_to(c.STORE)))
        for p in sorted(c.STORE.rglob('*')) if p.is_file()])
    audit=verify_seal();c.save(c.EVIDENCE/'completion_audit.json',audit)
    c.save(c.EVIDENCE/'artifact_manifest.json',dict(files=[dict(path=p.name,sha256=c.sha(p),bytes=p.stat().st_size)
        for p in sorted(c.EVIDENCE.iterdir()) if p.is_file()],manifest_self_excluded=True))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--seal',action='store_true');group.add_argument('--verify',action='store_true')
    parser.add_argument('--source');args=parser.parse_args()
    if args.seal:c.require(args.source,'P_SOURCE_REQUIRED');result=seal(args.source)
    else:result=verify_seal()
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
