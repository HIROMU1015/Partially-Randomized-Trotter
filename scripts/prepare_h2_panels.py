"""Metadata-only H2 FP/D/A seals. Never import science or deserialize F arrays."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys

import h2_panels_contract as c


def row(path,role):
    path=Path(path)
    return dict(path=str(path),role=role,sha256=c.sha(path),bytes=path.stat().st_size)


def freeze_kernel(directory):
    """Copy the original A steps/control implementation and its stdlib closure."""
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    pending=['scripts/pr_cost_native.py','scripts/pr_cost_contract.py']; copied={}
    while pending:
        rel=pending.pop()
        if rel in copied: continue
        data=subprocess.check_output(['git','show',c.A_SOURCE+':'+rel],cwd=c.ROOT)
        dest=directory/rel; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(data)
        copied[rel]=hashlib.sha256(data).hexdigest()
        for node in ast.walk(ast.parse(data)):
            mods=[x.name for x in node.names] if isinstance(node,ast.Import) else (
                [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            for mod in mods:
                name=mod.split('.')[0]
                if name not in ('pr_cost_contract','pr_cost_integration','h8_six_axis_reaccount','h8_phase_c_accounting'): continue
                path=('scripts/' if name=='pr_cost_contract' else 'src/trotterlib/')+name+'.py'
                pending.append(path)
    return copied


def primitive_count(cells,provider):
    # This provider module is stdlib-only and loads a pure PF schedule with runpy.
    sys.path.insert(0,str(Path(provider)/'src'))
    from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
    return 3*len({('one' if i==0 else str(i-1),t) for x in cells if x['method']!='B0I'
        for i,t in primitive_time_schedule(x,T=.8)['unique_primitive_times']})


def preservation():
    roots=[c.PRIVATE/'phase0_preparation01/preparation_preservation_manifest_PRIVATE.json',
           c.PRIVATE/'activation01/F_completed_preservation_manifest_PRIVATE.json']
    records=[]
    for path in roots:
        for r in c.read(path):
            c.require(c.sha(r['path'])==r['sha256'],'PRESERVED_F_OR_PREPARATION_CHANGED')
            records.append(r)
    return dict(status='PASS',records_checked=len(records),manifests=[row(x,'F_preservation') for x in roots])


def seal(source):
    c.require(len(source)==40 and subprocess.check_output(['git','rev-parse',source],cwd=c.ROOT).decode().strip()==source,
              'FULL_COMMITTED_SOURCE_REQUIRED')
    c.STORE.mkdir(parents=True,exist_ok=False); c.EVIDENCE.mkdir(parents=True,exist_ok=False)
    preserved=preservation(); c.save(c.STORE/'F_preservation_before.json',preserved)
    f=c.read(c.PRIVATE/'activation01/F_spec_EXECUTION.json')
    f_output=c.PRIVATE/'attempts/F_run01/output'
    f_result=c.read(f_output/'result.json')
    c.require(f_result['hamiltonian_hash']==c.BINDING['hamiltonian'] and
              f_result['state_hash']==c.BINDING['state'] and f_result['rank']==3,'ACCEPTED_F_IDENTITY')
    f_public=c.PUBLIC/'F_completed_run01/result.json'
    blob=subprocess.check_output(['git','show',c.F_COMMIT+':'+str(f_public.relative_to(c.ROOT))],cwd=c.ROOT)
    c.require(hashlib.sha256(blob).hexdigest()==c.sha(f_public)==c.sha(f_output/'result.json'), 'F_COMPLETION_COMMIT_BYTES')
    acceptance=dict(role='user',accepted_by_user=True,stage='F',completion_commit=c.F_COMMIT,
        result_sha256=c.sha(f_public),verbatim_user_message='H2のF結果を正式に受け入れ、FP・D・Aの技術準備をまとめて進めてください。',
        FP_D_A_science_authorized=False)
    c.save(c.STORE/'F_acceptance.json',acceptance); c.save(c.EVIDENCE/'F_acceptance.json',acceptance)
    kernel=c.STORE/'A_kernel'; kernel_hashes=freeze_kernel(kernel)
    own=c.source_closure()
    for rel,h in own.items():
        c.require(hashlib.sha256(subprocess.check_output(['git','show',source+':'+rel],cwd=c.ROOT)).hexdigest()==h,
                  'SOURCE_MUST_INCLUDE_CURRENT_CODE:'+rel)
    # Only known sealed seed metadata is read, never another job's mutable output.
    other=c.ROOT.parent
    old_root=other/'h4-tol-p-seal-20261019/.h4_tol_p_private'
    sources=[old_root/'old_seed_exclusions_PRIVATE.json',old_root/'seed_registry_PRIVATE.json',
             old_root/'seed_inventory_receipt_PRIVATE.json',
             other/'h4-p-seal-20261019/.h4_p_seal_private/geometry_seed_cases_PRIVATE.json']
    origins=[]
    for i,path in enumerate(sources):
        dest=c.STORE/('seed_origin_%d.json'%i); dest.write_bytes(path.read_bytes())
        origins.append(dict(origin=str(path),**row(dest,'seed_origin_%d'%i)))
    excluded=set(c.read(c.STORE/'seed_origin_0.json')['values'])
    old_registry=c.read(c.STORE/'seed_origin_1.json')
    excluded.update(old_registry['values'])
    previous=c.read(c.STORE/'seed_origin_2.json')
    cases=c.read(c.STORE/'seed_origin_3.json')
    c.save(c.STORE/'old_seed_values.json',dict(values=sorted(excluded)))
    registries={stage:dict(SOURCE=source,plan_digest=c.digest(c.plan(stage)),
        tasks=c.seed_tasks(stage,source),actual_RTE_event_sampling=False) for stage in ('FP','A')}
    values={stage:c.seed_values(r['tasks']) for stage,r in registries.items()}
    c.require(not values['FP']&values['A'] and not (values['FP']|values['A'])&excluded, 'OLD_FP_A_SEED_COLLISION')
    import h4_p_seed_inventory as inventory
    # Six independent geometries, existing CPU-affinity-aware process pool.
    checks,geometry_workers=inventory.check_geometry(cases,values['FP']|values['A'])
    c.require(all(not r['collisions'] for r in checks),'GEOMETRY_RESERVED_SEED_COLLISION')
    now=datetime.now(timezone.utc).isoformat()
    for stage in ('FP','A'):
        c.save(c.STORE/(stage+'_seeds.json'),registries[stage])
        c.save(c.STORE/(stage+'_seed_check.json'),dict(status='COLLISION_ZERO',
            new_values_digest=c.digest(sorted(values[stage])),new_seed_slots=len(values[stage]),
            joint_FP_A_values_digest=c.digest(sorted(values['FP']|values['A'])),
            old_explicit_values=len(excluded),geometry_checks=checks,geometry_workers=geometry_workers,checked_at_UTC=now,
            old_explicit_exclusions_sha256=c.sha(c.STORE/'old_seed_values.json'),
            inherited_inventory_cutoff=previous['cutoff_UTC'],inherited_coverage=previous['coverage'],
            known_campaigns_only=True,future_unregistered_campaigns_covered=False,
            activation_refresh_required=True,actual_RTE_event_sampling=False,
            zero_collision_is_independence_proof=False))
    c.save(c.STORE/'seed_origins.json',origins)
    providers=[row(Path(f['provider_root'])/rel,rel) for rel,h in f['provider_hashes'].items()]
    c.require(all(r['sha256']==f['provider_hashes'][r['role']] for r in providers),'F_PROVIDER_PRESERVED')
    kernels=[row(kernel/rel,rel) for rel,h in kernel_hashes.items()]
    runtime=c.read(f['runtime_manifest'])
    for path,h in runtime.items(): c.require(c.sha(path)==h,'F_RUNTIME_CHANGED:'+path)
    config=dict(provider_root=f['provider_root'],binding=c.BINDING,compiler_options=c.COMPILER,
        mapping=dict(tol_snapshot=str(f_output/'tol_only_snapshot.npz'),tol_receipt=str(f_output/'tol_only_receipt.json')))
    specs={}
    for stage in ('FP','D','A'):
        count=primitive_count(c.registrations()[stage],f['provider_root'])
        inputs=[row(f_output/'tol_only_snapshot.npz','F_snapshot'),row(f_output/'tol_only_receipt.json','F_receipt'),
                row(c.STORE/'F_acceptance.json','F_acceptance'),row(c.STORE/'old_seed_values.json','old_seed_values'),
                row(c.STORE/'seed_origins.json','seed_provenance')]
        if stage in registries:
            inputs += [row(c.STORE/(stage+'_seeds.json'),'seeds'),row(c.STORE/(stage+'_seed_check.json'),'seed_check')]
        spec=dict(scope=c.SCOPE,stage=stage,status='SEALED_PENDING_APPROVAL',SOURCE=source,plan=c.plan(stage),
            source_hashes=own,inputs=inputs,native_config=config,provider_roots=[f['provider_root']],provider_files=providers,
            kernel_files=kernels,A_kernel_SOURCE=c.A_SOURCE,A_kernel_scripts=str(kernel/'scripts'),
            runtime_manifest=f['runtime_manifest'],runtime_manifest_sha256=f['runtime_manifest_sha256'],
            python=f['python'],python_sha256=f['python_sha256'],attempt=str(c.PRIVATE/('attempts/'+stage+'_run01')),
            primitive_probes=count,calls=c.budget(stage,count),
            pending=['SEPARATE_EXPLICIT_'+stage+'_APPROVAL','FRESH_RESOURCE_ADMISSION']+
                (['ACCEPTED_FP_RESULT_SHA'] if stage=='D' else ['ACCEPTED_FP_RESULT_SHA','ACCEPTED_D_RESULT_SHA'] if stage=='A' else [])+
                (['SEED_INVENTORY_ACTIVATION_REFRESH'] if stage in registries else []),
            dependency_binding_rule='add accepted same-binding completed result/audit/acceptance byte hashes; preserve this pending seal',
            dependencies={},worker_cpus=[],resource_receipt=None,resource_receipt_sha256=None)
        if stage in registries:
            spec.update(seed_registry=str(c.STORE/(stage+'_seeds.json')),seed_check=str(c.STORE/(stage+'_seed_check.json')))
        c.validate_metadata(spec)
        c.save(c.STORE/(stage+'_spec_PENDING.json'),spec)
        c.save(c.EVIDENCE/(stage+'_INVALID_GRANT.json'),c.invalid_grant(stage))
        c.save(c.EVIDENCE/(stage+'_preregistration.json'),dict(SOURCE=source,plan=spec['plan'],calls=spec['calls'],
            contract_sha256=c.sha(c.STORE/(stage+'_spec_PENDING.json')),source_hashes=own,
            input_rows=[{k:v for k,v in r.items() if k!='path'} for r in inputs],
            provider_SOURCE=f['provider_SOURCE'],provider_files=len(providers),A_kernel_SOURCE=c.A_SOURCE,
            A_kernel_hashes=kernel_hashes,runtime_manifest_sha256=f['runtime_manifest_sha256'],runtime_files=len(runtime),
            pending=spec['pending'],scientific_execution=False,mandatory_STOP=True))
        specs[stage]=spec
    c.save(c.EVIDENCE/'seed_audit_summary.json',dict(checked_at_UTC=now,
        master_seeds={k:len(v['tasks']) for k,v in registries.items()},
        seed_slots={k:len(v) for k,v in values.items()},old_explicit_values=len(excluded),geometry_checks=checks,
        RTE_sampling=0,known_campaigns_only=True,activation_refresh_required=True))
    c.save(c.EVIDENCE/'F_preservation_after.json',preservation())
    c.save(c.STORE/'preparation_preservation.json',[row(p,str(p.relative_to(c.STORE)))
        for p in sorted(c.STORE.rglob('*')) if p.is_file()])
    return dict(status='TECHNICAL_PREPARATION_MANDATORY_STOP',SOURCE=source,
        stages={s:dict(calls=x['calls'],pending=x['pending']) for s,x in specs.items()},
        new_scientific_execution=0,next_stage_authorized=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source',required=True)
    a=p.parse_args(argv); print(__import__('json').dumps(seal(a.source),indent=2))


if __name__=='__main__': main()
