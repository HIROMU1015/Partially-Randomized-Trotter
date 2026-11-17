#!/usr/bin/env python3
"""Describe one selected release and map its fixed components onto its hosts.

Reads existing verified profiles and adds only the saved price-worker binding.
It creates configuration only: no host connection, controller or scientific job.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_host_profiles import profile_core
from trottertracks.rpe_release_registry import fingerprint


def read(path):return json.loads(Path(path).read_text())
def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def prepare(source,release_id,service_path,price_profiles,hosts,output):
    if len(source)!=40 or any(c not in '0123456789abcdef'for c in source):
        raise ValueError('Controller SOURCE must be a full commit')
    output=Path(output).resolve()
    if output.exists():raise ValueError('Use a new fleet output directory')
    service=read(service_path);round_value=read(service['round'])
    profile_paths=round_value['executor']['distributed']['preparation_profiles']
    worker_source=hosts['local']['worker_source']
    preparation_source=service['migration']['candidate_completion_source']
    price_source=round_value['price_validation']['source']
    components={
        'optimizer':dict(source=source,allow_metadata_commits=True,
            required_paths=['scripts/run_rpe_schedule_search.py','scripts/run_rpe_search.py','scripts/manage_rpe_release_deployment.py','src/trottertracks/rpe_release_deployment.py']),
        'worker':dict(source=worker_source,required_paths=['scripts/run_rpe_q3_worker.py']),
        'preparation':dict(source=preparation_source,required_paths=['scripts/run_rpe_preparation_job.py']),
        'q3':dict(source=service['computation_source'],required_paths=['scripts/run_rpe_optimizer_q3_job.py']),
        'price_worker':dict(source=price_source,required_paths=['scripts/run_rpe_price_validation_job.py'])}
    stage_components={stage:set()for stage in ['signal','compile','audit','q3']};stage_components['q3'].add('q3')
    profiles={};source_roots={};bindings={};runtime_layouts={};profile_preserved={};participation={}
    for host,defaults in hosts.items():
        base_profile=Path(defaults.get('base_profile')or profile_paths[host])
        price_profile=Path(defaults.get('price_profile')or Path(price_profiles)/(host+'.json'))
        profile=read(base_profile);before=profile_core(profile)
        price=read(price_profile)
        extras=[r for r in price.get('preparation_profiles',[])if r['provider']=='PRICE_MODEL_VALIDATION'and r['source_commit']==price_source]
        if len(extras)!=1:raise ValueError('Expected one fixed price-worker rule for '+host)
        replacements={(r['stage'],r['provider'],r['source_commit']):r for r in extras}
        profile['preparation_profiles']=[deepcopy(replacements.pop((r['stage'],r['provider'],r['source_commit']),r))
            for r in profile.get('preparation_profiles',[])]+deepcopy(list(replacements.values()))
        if profile_core(profile)!=before:raise ValueError('Fleet price integration changed scientific Q3 profile')
        profile_preserved[host]=dict(q3_profile_core_unchanged=True,
            previous_profile_sha256=hashlib.sha256(base_profile.read_bytes()).hexdigest(),
            price_profile_sha256=hashlib.sha256(price_profile.read_bytes()).hexdigest())
        roots={service['computation_source']:profile['root'],worker_source:defaults['worker_root']}
        adapter=profile.get('adapter_manifest')
        if adapter:
            if 'q3_adapter'in components and components['q3_adapter']['source']!=adapter['commit']:
                raise ValueError('Hosts disagree about fixed Q3 adapter')
            components['q3_adapter']=dict(source=adapter['commit'],required_paths=['scripts/run_rpe_optimizer_q3_job.py'])
            roots[adapter['commit']]=profile['adapter_root']
        contracts=[dict(stage='controller',component='optimizer',runtime='optimizer')]
        runtimes={'optimizer':dict(python=defaults['python'],modules={},components=['optimizer'])}
        if defaults.get('pbs'):
            pbs=defaults['pbs'];transport_source=pbs.get('transport_source',source)
            for name,path in [('pbs_gateway','scripts/rpe_pbs_gateway.py'),('pbs_worker','scripts/run_rpe_pbs_worker.py')]:
                if name in components and components[name]['source']!=transport_source:
                    raise ValueError('PBS hosts disagree about the fixed transport SOURCE')
                components.setdefault(name,dict(source=transport_source,required_paths=[path]))
            participation[host]=['optimizer','pbs_worker','pbs_gateway']
            runtimes['pbs_worker']=dict(python=pbs.get('worker_python',profile['interpreter']),modules={},components=['pbs_worker'])
            # A gateway runtime is explicit: system Python is not substituted
            # during collection or after the PBS allocation has ended.
            runtimes['pbs_gateway']=dict(python=pbs['gateway_python'],modules={},components=['pbs_gateway'])
            contracts += [dict(stage='worker',component='pbs_worker',runtime='pbs_worker'),
                dict(stage='gateway',component='pbs_gateway',runtime='pbs_gateway',operations=['collect','harvest'])]
            transport_root=pbs.get('transport_root')or defaults.get('optimizer_root')or str(Path(defaults['install_root'])/source[:12])
            def bind_transport(value):
                if isinstance(value,dict):
                    if isinstance(value.get('pbs'),dict):value['pbs'].update(worker_source=transport_source,worker_root=transport_root)
                    for item in value.values():bind_transport(item)
                elif isinstance(value,list):
                    for item in value:bind_transport(item)
            bind_transport(profile)
        else:
            participation[host]=['optimizer','worker']
            runtimes['worker']=dict(python=defaults.get('worker_python',profile['interpreter']),modules={},components=['worker'])
            contracts.append(dict(stage='worker',component='worker',runtime='worker'))
        def views(p):
            for key in ('route_views','preparation_sources'):
                for commit,path in p.get(key,{}).items():roots.setdefault(commit,path)
            manifest=p.get('preparation_adapter_manifest')
            if manifest and p.get('preparation_adapter_root'):roots.setdefault(manifest['commit'],p['preparation_adapter_root'])
        views(profile)
        for index,rule in enumerate(profile.get('route_profiles',[])):
            p=rule['profile'];views(p);name='q3_route_'+str(index)
            runtimes[name]=dict(python=p['interpreter'],modules={'numpy':None,'scipy':None},components=['q3'])
            contracts.append(dict(stage='q3',route_kind=rule['route_kind'],view_commit=rule.get('view_commit'),
                runtime=name,fixed_runtime_checks=deepcopy(p.get('runtime_checks',[]))))
        for index,rule in enumerate(profile.get('preparation_profiles',[])):
            p=rule['profile'];views(p);commit=rule['source_commit'];name='preparation_'+str(index)
            if rule in extras:
                roots[price_source]=p['preparation_sources'][price_source]
            modules={'numpy':None,'scipy':None}if rule['stage']=='signal'else{'qiskit':'1.3.0'}
            runtimes[name]=dict(python=p['interpreter'],modules=modules,components=[])
            contracts.append(dict(stage=rule['stage'],provider=rule['provider'],source=commit,
                runtime=name,request_runtime_contract='preparation.runtime_requirements',
                compiler_binding='preparation.compiler_bindings'))
        observations=defaults.get('runtime_observations',{})
        for contract in contracts:
            observed=observations.get(contract['runtime'])
            if observed:
                # These facts describe the deployed release. Existing worker
                # REQUEST/compiler/AST contracts remain the execution authority.
                contract['runtime_observation']=deepcopy(observed)
            else:
                contract['runtime_observation_status']='DEFERRED_OFFLINE'if host=='server1'else'NOT_RECORDED'
        profiles[host]=profile;source_roots[host]=roots;bindings[host]=contracts;runtime_layouts[host]=runtimes
    names={spec['source']:name for name,spec in components.items()}
    for roots in source_roots.values():
        for commit in roots:
            if commit not in names:
                name='provider_'+commit[:12];names[commit]=name
                components[name]=dict(source=commit,required_paths=[])
    # The preparation wrapper and worker may intentionally share one SOURCE.
    names[preparation_source]='preparation';names[price_source]='price_worker'
    for name,specification in components.items():
        if name=='optimizer':specification['hosts']=list(hosts)
        elif name in ('pbs_worker','pbs_gateway'):specification['hosts']=[host for host,defaults in hosts.items()if defaults.get('pbs')]
        else:specification['hosts']=[host for host,roots in source_roots.items()if roots.get(specification['source'])
            and(name!='worker'or not hosts[host].get('pbs'))]
    for profile in profiles.values():
        for rule in profile['preparation_profiles']:
            stage_components[rule['stage']].add(names[rule['source_commit']])
    manifest=dict(schema='RPE_RELEASE_COMPONENTS_V1',release_id=release_id,controller_host='RomuTail',
        components=components,stage_components={k:sorted(v)for k,v in stage_components.items()},
        participation_components=['optimizer','worker'],participation_components_by_host=participation,
        hosts=list(hosts),runtime_contracts=bindings,
        scientific_execution_permission='UNCHANGED_FIXED_CONTRACTS_AND_EXPLICIT_USER_SCOPE')
    put(output/'RELEASE_COMPONENTS.json',manifest)
    layouts={}
    for host,defaults in hosts.items():
        metadata=output if host=='local'else Path(defaults['metadata_root'])/release_id
        optimizer=defaults.get('optimizer_root')or str(Path(defaults['install_root'])/source[:12])
        roots={name:(optimizer if name=='optimizer'else source_roots[host].get(spec['source']))for name,spec in components.items()}
        if defaults.get('pbs'):
            transport=defaults['pbs'].get('transport_root')or optimizer
            roots.update(pbs_worker=transport,pbs_gateway=transport)
            roots['worker']=None
        # All host-local paths are kept out of the public component manifest.
        profile_path=output/'profiles'/f'{host}.json'if host=='local'else metadata/'PROFILE.json'
        layout_path=output/'hosts'/f'{host}.json'if host=='local'else metadata/'HOST.json'
        manifest_path=output/'RELEASE_COMPONENTS.json'if host=='local'else metadata/'RELEASE_COMPONENTS.json'
        for index,rule in enumerate(profiles[host]['preparation_profiles']):
            runtime_layouts[host]['preparation_'+str(index)]['components']=[names[rule['source_commit']]]
        layout=dict(host_id=host,repository=defaults['repository'],seed_repository=defaults['seed_repository'],
            component_roots=roots,runtimes=runtime_layouts[host],input_files={},
            participation_components=participation[host],
            manifest_path=str(manifest_path),layout_path=str(layout_path),profile_path=str(profile_path),
            availability='DEFERRED_OFFLINE'if host=='server1'else'AVAILABLE',
            intake='OFF'if host in('server1','local0-wsl')else'UNCHANGED_EXISTING_POLICY')
        if defaults.get('bundle_files'):layout['bundle_files']=defaults['bundle_files']
        layouts[host]=layout
        put(output/'profiles'/f'{host}.json',profiles[host]);put(output/'hosts'/f'{host}.json',layout)
    report=dict(schema='RPE_RELEASE_FLEET_CONFIGURATION_V1',release_id=release_id,source=source,
        manifest=str(output/'RELEASE_COMPONENTS.json'),manifest_sha256=fingerprint(manifest),
        host_layouts={host:str(output/'hosts'/f'{host}.json')for host in hosts},
        profiles={host:str(output/'profiles'/f'{host}.json')for host in hosts},
        profile_preservation=profile_preserved,server1='DEFERRED_OFFLINE',local0_wsl_accepting_new_jobs=False,
        hosts_contacted=[],scientific_calls=0)
    put(output/'FLEET.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--release',required=True)
    parser.add_argument('--service',type=Path,required=True);parser.add_argument('--price-profiles',type=Path,required=True)
    parser.add_argument('--hosts',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=prepare(args.source,args.release,args.service,args.price_profiles,read(args.hosts),args.output)
    print(json.dumps(result))


if __name__=='__main__':main()
