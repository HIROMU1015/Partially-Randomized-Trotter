"""Seal an INVALID four-lane proposal using file bytes/seed metadata only."""
import argparse
import json
from pathlib import Path
import subprocess

import h46_m5_contract as c


def seal(path,role):
    path=Path(path).resolve()
    return dict(path=str(path),role=role,bytes=path.stat().st_size,sha256=c.sha(path))


def prepare(old_config,confirmation_config,confirmation_registry,destination,source):
    old=c.read(old_config);conf=c.read(confirmation_config)
    destination=Path(destination).resolve();destination.mkdir(parents=True,exist_ok=False)
    for relative in c.SOURCE_PATHS:
        original=subprocess.check_output(['git','show',source+':'+relative],cwd=c.ROOT)
        c.require(original==(c.ROOT/relative).read_bytes(),'COMMITTED_SCIENTIFIC_SOURCE')
    # Existing immutable snapshot bytes are reused, never opened as molecular arrays.
    inputs={**old['inputs'],**conf['inputs']}
    for name,row in inputs.items():c.require(c.sha(row['path'])==row['sha256'],'ORIGINAL_INPUT_SHA:'+name)
    binding=c.read(c.ROOT/'artifacts/resource_applicability/track_a_h46_m5_best_parallel_readiness/2026-10-19/main_input_binding.json')['systems']
    for system in ('H4','H6'):
        c.require(inputs[system+'_snapshot.npz']['sha256']==binding[system]['snapshot'],'MAIN_SNAPSHOT_BINDING')
    from prepare_h46_confirmation_native import previous_seeds
    older_registry=c.read(old['seed_registry_path'])
    excluded,audit=previous_seeds(c.ROOT,older_registry)
    confirm=c.read(confirmation_registry)
    values={v for task in confirm['tasks'].values() for v in task.values()}
    c.require(not excluded&values,'CONFIRMATION_EXPLORATION_COLLISION')
    excluded|=values
    exclusion_path=destination/'excluded_seed_PRIVATE.json'
    c.save(exclusion_path,dict(values=sorted(excluded),scope='KNOWN_HIGHER_PF_AND_ACCESSIBLE_REFERENCE_METADATA',
        confirmation_seed_registry_sha256=c.sha(confirmation_registry),audit=audit,
        unknown_external_registry_coverage_claimed=False))
    reg_path=destination/'seed_registry_PROVISIONAL_PRIVATE.json'
    seeds=c.registry(source,excluded);c.save(reg_path,dict(SOURCE=source,purpose='TECHNICAL_PILOT_NOT_CONFIRMATION',lanes=seeds))
    seed_count=sum(len(v) for lane in seeds.values() for v in lane.values())
    # All provider files and runtime objects already sealed by the successful run.
    provider=Path(old['provider_root']).resolve()
    runtime_files=[e for e in old['sealed_files'] if Path(e['path']).is_relative_to(provider)
        or '/site-packages/' in e['path'] or e['path']=='/usr/bin/python3.12']
    c.require(runtime_files,'EXISTING_PROVIDER_RUNTIME_CLOSURE')
    runtime=dict(provider_root=str(provider),provider_SOURCE=old['provider_SOURCE'],python=old['python'],
        python_target=str(Path(old['python']).resolve()),qiskit='1.3.0',
        packages='EXACT_PREVIOUS_SEALED_RUNTIME_BYTES',CPU_only=True)
    common=[seal(e['path'],'provider/runtime') for e in runtime_files]
    for previous,now in zip(runtime_files,common):c.require(previous['sha256']==now['sha256'],'PREVIOUS_RUNTIME_UNCHANGED')
    common += [seal(c.ROOT/p,'scientific_SOURCE:'+p) for p in c.SOURCE_PATHS]
    common += [seal(reg_path,'seed_registry'),seal(exclusion_path,'seed_exclusion'),
        seal(c.ROOT/'artifacts/resource_applicability/track_a_h46_m5_best_parallel_readiness/2026-10-19/main_input_binding.json','main_input_binding')]
    root=destination/'future_run01';configs={};reports={}
    for lane in c.LANES:
        system=lane[:2];inp={k:v for k,v in inputs.items() if k.startswith(system+'_')}
        files=common+[seal(row['path'],'saved_input:'+name) for name,row in inp.items()]
        plan=dict(status='DRAFT_UNAPPROVED',SOURCE=source,scope=c.scope(lane),
            closure_digest=c.core.digest(files),input_digest=c.core.digest(inp),binding_digest=c.core.digest(binding[system]),
            runtime_digest=c.core.digest(runtime),seed_registry_sha256=c.sha(reg_path),
            native_policy=c.POLICY,phase3_authorized=False)
        plan_path=destination/(lane+'_plan_DRAFT_PRIVATE.json');c.save(plan_path,plan)
        paths=dict(output=str(root/(lane+'_output')),ledger=str(root/(lane+'_ledger')),marker=str(root/(lane+'_CONSUMED.json')))
        grant=dict(valid=False,grant_id=None,approval_reference=None,lane=lane,SOURCE=source,
            plan_sha256=c.sha(plan_path),maximum_axes=c.scope(lane)['compile_axes'],
            maximum_trajectories=6 if lane.endswith('PR') else 0,
            maximum_reference_jobs=c.scope(lane)['reference_jobs'],retry=False,phase3_authorized=False,**paths)
        config=dict(authorized=False,lane=lane,system=system,SOURCE=source,provider_root=str(provider),
            native_policy=c.POLICY,runtime=runtime,inputs=inp,binding=binding[system],sealed_files=files,
            plan=plan,plan_path=str(plan_path),plan_sha256=c.sha(plan_path),
            registry_path=str(reg_path),excluded_seed_path=str(exclusion_path),grant=grant,run_root=str(root),
            reference_root=str(root/(system+'_reference')),
            # Planning proxy, not a hard RAM cap: old Y4 peak * stage ratio + dense oracle workspace.
            RSS_proxy_bytes=math_proxy(system),**paths)
        config_path=destination/(lane+'_config_DRAFT_PRIVATE.json');c.save(config_path,config)
        c.save(destination/(lane+'_grant_INVALID.json'),grant);configs[lane]=str(config_path)
        c.verify_seals(config)
        reports[lane]=dict(plan_sha256=c.sha(plan_path),closure_files=len(files),axes=grant['maximum_axes'],
            grant_valid=False,scientific_SOURCE=source)
    c.save(destination/'bundle_DRAFT_PRIVATE.json',dict(configs=configs))
    result=dict(status='FOUR_NATIVE_CONTRACTS_SEALED_INVALID_GRANTS',SOURCE=source,lanes=reports,
        fresh_seed_values=seed_count,known_excluded_seed_values=len(excluded),collisions=0,
        registry_sha256=c.sha(reg_path),closure_files_common=len(common),valid_grants=0,
        sampling=0,compile=0,signal=0,mandatory_STOP=True)
    c.save(destination/'preparation_summary.json',result);return result


def math_proxy(system):
    # Exposed scaling estimate until real pilot RSS is observed, no historical worker ceiling.
    return (1723486208*11//3)+(512 if system=='H6' else 32)*1024**2


def main():
    p=argparse.ArgumentParser();p.add_argument('--old-config',required=True);p.add_argument('--confirmation-config',required=True)
    p.add_argument('--confirmation-registry',required=True);p.add_argument('--destination',required=True);p.add_argument('--source',required=True)
    a=p.parse_args();print(json.dumps(prepare(a.old_config,a.confirmation_config,a.confirmation_registry,a.destination,a.source)))


if __name__=='__main__':main()
