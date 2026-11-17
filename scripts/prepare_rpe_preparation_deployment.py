#!/usr/bin/env python3
"""Freeze per-stage profiles and the existing local0 SSH deployment procedure.

Produces inspection-only saved-input examples unless --tasks supplies exact task
requests. This command creates no worker, signal, compiler or scientific queue.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.rpe_preparation_queue import normalize_preparation_request


def read(path):return json.loads(Path(path).read_text())
def raw(root,commit,path):return subprocess.check_output(['git','show',commit+':'+path],cwd=root)
def sha(data):return hashlib.sha256(data).hexdigest()
def encode(value):return(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode()
def put(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(encode(value))


def source_manifest(commit,names):return dict(commit=commit,files={name:sha(raw(ROOT,commit,name))for name in names})


def saved_examples(source):
    pr='981c5efd3154014ef623eaaa5e6376c3b7d75321'
    pa='docs/research/h4_pr_multiwidth_resume_20261117/run01/completion/candidates/15e782d4b0e09c9be929/'
    df='docs/research/h4_pr_candidate_continuity_20261117/run01/completion/candidates/6fd6cc805556fafcf685/'
    full='docs/research/d_pf_on_demand_evidence_20261116/selected/9e96ed75f7f10b5f993ba9e2841684e52ad97b7d4e9df07b7b0070f02c852981/'
    b0i='docs/research/d_pf_on_demand_evidence_20261116/selected/dd8bfc248fdc67d42a6ae75df7eb0fbfcf34edc5e6ab3f3632e8784b47a7dbe0/'
    pf='36ad751d4ebf0ec160532615da6425b268a0c18a'
    provider=subprocess.check_output(['git','rev-parse','d8780090f'],cwd=ROOT,text=True).strip()
    pf_files=['scripts/prepare_pf_candidate.py','src/trottertracks/pf_evidence_preparation.py','scripts/run_pf_m0_prices.py','scripts/run_pf_p7_h_prices.py','scripts/run_truncated_p7_prices.py']
    tasks=[]
    for method in ['FULL_PF','TRUNCATED_PF','PR_PAULI','PR_DF']:
        for stage in ['signal','compile','audit']:
            if method in ('FULL_PF','TRUNCATED_PF'):
                commit=pf;path=full+'PLAN.json'if method=='FULL_PF'else b0i+'SAVED_Q3_INPUTS.json'
                source_record=source_manifest(pf,pf_files)
                command=['@PYTHON@','@TRANSPORT@/scripts/run_pf_preparation_stage.py','--provider-root','@SOURCE@','--input','@INPUTS@/SAVED_INPUT.json','--output','@WORK@/RESULT.json']
                name='PF_'+{'signal':'SIGNALS','compile':'COMPILE','audit':'AUDIT'}[stage]
            elif method=='PR_PAULI':
                commit=pr;path=pa+('PAULI_SIGNAL_REQUEST.json'if stage=='signal'else'PAULI_PRICE_REQUEST.json')
                script='scripts/run_pr_pauli_candidate_signals.py'if stage=='signal'else'scripts/run_pr_pauli_candidate_prices.py'
                source_record=source_manifest(source,[script,'src/trottertracks/rpe_preparation_paths.py'])
                command=['@PYTHON@','@SOURCE@/'+script,'--request','@INPUTS@/SAVED_INPUT.json','--output','@WORK@/RESULT.json','--execute','--no-push']
                if stage!='signal':command+=['--phase',stage]
                name='PAULI_'+{'signal':'SIGNALS','compile':'COMPILE','audit':'AUDIT'}[stage]
            else:
                commit=pr;path=df+('DF_SIGNAL_REQUEST.json'if stage=='signal'else'df_price_groups/98621104fb2fde4e40d8/df_prices/INPUTS.json')
                script='scripts/run_pr_df_candidate_signal_jobs.py'if stage=='signal'else'scripts/run_pr_df_candidate_prices.py'
                source_record=source_manifest(source,[script,'src/trottertracks/rpe_preparation_paths.py'])
                command=['@PYTHON@','@SOURCE@/'+script,'--input','@INPUTS@/SAVED_INPUT.json','--output','@WORK@/RESULT.json']
                name='DF_'+{'signal':'SIGNALS','compile':'CALIBRATION','audit':'MODEL_FREEZE'}[stage]
            data=raw(ROOT,commit,path)
            from trottertracks.rpe_preparation_bridge import collect_fixed_inputs,collect_required_views,collect_required_paths
            documents=[json.loads(data)];nested_inputs=[];working_inputs=[];required_views={};required_paths={};blockers=[]
            try:nested_inputs,working_inputs=collect_fixed_inputs(ROOT,documents,{'SAVED_INPUT.json':sha(data)})
            except Exception as error:blockers.append('NESTED_INPUTS:'+str(error))
            try:required_views=collect_required_views(documents)
            except Exception as error:blockers.append('OPERATIONAL_SOURCE_VIEW:'+str(error))
            try:required_paths=collect_required_paths(documents)
            except Exception as error:blockers.append('OPERATIONAL_RUNTIME_PATH:'+str(error))
            compiler_versions=set();compiler_bindings=set()
            def bindings(value):
                if isinstance(value,dict):
                    for key,item in value.items():
                        if key=='qiskit_version'and isinstance(item,str):compiler_versions.add(item)
                        if key=='compiler'and isinstance(item,str):compiler_bindings.add(item)
                        bindings(item)
                elif isinstance(value,list):
                    for item in value:bindings(item)
            bindings(json.loads(data))
            if len(compiler_versions)>1:raise ValueError('Saved example mixes compiler versions')
            packages={'numpy':None,'scipy':None}if stage=='signal'else{'qiskit':next(iter(compiler_versions),'1.3.0')}
            task=dict(schema='RPE_PREPARATION_REQUEST_V1',method=method,route=dict(kind='preparation'),
                preparation=dict(stage=stage,provider=name,source=source_record,command=command,outputs=['RESULT.json'],inputs={'SAVED_INPUT.json':sha(data)},
                    input_hashes=list({(item['commit'],item['path']):item for item in [dict(commit=commit,path=path,sha256=sha(data))]+nested_inputs}.values()),
                    working_input_hashes=working_inputs,required_views=required_views,required_paths=required_paths,inspection_blockers=blockers,
                    runtime_requirements=dict(version_info_prefix=[3],packages=packages),
                    compiler_bindings=sorted(compiler_bindings),dry_check_only=True,inspection_scope='SAVED_REAL_INPUT_SOURCE_PATH_PYTHON_COMPILER_METADATA_ONLY'),
                original_contract=dict(saved_input=dict(commit=commit,path=path),scientific_body_evaluation=False))
            tasks.append(normalize_preparation_request(task))
    return tasks


def create(tasks,source,output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    old=ROOT/'docs/research/rpe_q3_auto_allocation/deployment_20261117'
    config=read(old/'local0_tools/PACKAGE_CONFIG.json')
    adapter_files=['scripts/run_rpe_preparation_job.py','src/trottertracks/rpe_preparation_worker.py',
        'src/trottertracks/rpe_preparation_artifacts.py','src/trottertracks/rpe_q3_timing.py',
        'src/trottertracks/rpe_preparation_queue.py','src/trottertracks/rpe_preparation_paths.py',
        'scripts/run_pf_preparation_stage.py','scripts/pauli_split_cal_audit_request.py',
        'scripts/run_pr_pauli_candidate_signals.py','scripts/run_pr_pauli_candidate_prices.py',
        'scripts/run_pr_df_candidate_signal_jobs.py','scripts/run_pr_df_candidate_prices.py',
        'scripts/derive_pr_df_candidate_descriptor.py']
    adapter=source_manifest(source,adapter_files)
    all_sources={}
    for task in tasks:
        all_sources[task['preparation']['source']['commit']]=task['preparation']['source']
        for workspace in task['preparation'].get('workspaces',{}).values():all_sources[workspace['source']['commit']]=workspace['source']
        for required in task['preparation'].get('required_views',{}).values():all_sources[required['source']['commit']]=required['source']
    # Actual selected PF completion pins its provider entry HEAD (36ad),
    # while the original native acquisition implementation is d878. Keep
    # both immutable source views available; do not assign either by host.
    for revision in ['36ad751d4ebf0ec160532615da6425b268a0c18a','d8780090f']:
        commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
        all_sources.setdefault(commit,dict(commit=commit))
    input_commits=sorted({loc['commit']for task in tasks for loc in task['preparation'].get('input_hashes',[])})
    hosts=[];profiles={};source_name=source[:12]
    physical={h['host_id']:h for h in config['hosts']}
    physical['server2']=dict(host_id='server2',repository=str(ROOT),transport_python='python3')
    for host_id in ['server2','server1','local','local0-wsl']:
        h=deepcopy(physical[host_id]);repo=Path(h['repository'])
        owner=(repo/'.preparation-distributed'/source_name)
        transport=owner/'transport';views={commit:str(transport if commit==source else owner/'views'/commit)for commit in all_sources}
        base=read(old/'profiles'/f'{host_id}.json')
        base['work_dir']=str(owner/'spool');base['preparation_profiles']=[]
        for task in tasks:
            method=task.get('method');spec=task['preparation']
            # Reuse the actual verified execution Python, not the checker launcher.
            q3=read(old/'assignments'/f'{method}.json')
            from trottertracks.rpe_host_profiles import select_host_profile
            selected=select_host_profile(read(old/'profiles'/f'{host_id}.json'),q3)
            profile={k:deepcopy(v)for k,v in selected.items()if k not in ('route_profiles','runtime_checks','adapter_root','adapter_manifest','runtime')}
            if spec['stage'] in ('compile','audit'):
                # Compiler tooling uses the existing project environment;
                # the special Q3 AST interpreter is a separate route contract.
                profile['interpreter']=base['interpreter']
                if host_id=='server2':
                    profile['interpreter']='/home/AbeHiromu/projects/Evaluation-of-gate-numbers-for-ground-state-energy-calculations-using-higher-order-product-formulae/venv/bin/python'
            profile.update(preparation_sources=views,preparation_adapter_root=str(transport),preparation_adapter_manifest=adapter)
            if host_id=='server2':
                profile['preparation_paths']={alias:required['controller_path'] for example in tasks for alias,required in example['preparation'].get('required_paths',{}).items()}
            from trottertracks.rpe_preparation_bridge import STAGES
            prefix='PF_'if method in ('FULL_PF','TRUNCATED_PF')else'PAULI_'if method=='PR_PAULI'else'DF_'
            providers=[name for name,value in STAGES.items()if name.startswith(prefix)and value==spec['stage']]
            if method=='PR_PAULI'and spec['stage']=='signal':providers.append('A_BASE_P0')
            for provider in providers:
                rule=dict(stage=spec['stage'],provider=provider,source_commit=spec['source']['commit'],profile=profile)
                if rule not in base['preparation_profiles']:base['preparation_profiles'].append(rule)
        profiles[host_id]=base
        h.update(transport_root=str(transport),worker_profile=str(owner/'PROFILE.json'),profile='profiles/'+host_id+'.json',
            run_directory=str(owner/'metadata-dry'),source_views=views,input_commits=input_commits)
        hosts.append(h)
        put(output/'profiles'/f'{host_id}.json',base)
    put(output/'TASKS.json',dict(schema='RPE_PREPARATION_INSPECTION_TASKS_V1',tasks=tasks,scientific_calls=0))
    files={'TASKS.json':(output/'TASKS.json').read_bytes(),
        'run_rpe_preparation_deployment.py':(ROOT/'scripts/run_rpe_preparation_deployment.py').read_bytes(),
        'legacy_auto_deployment.py':(old/'local0_tools/run_auto_deployment.py').read_bytes()}
    files.update({'profiles/'+host+'.json':encode(profile)for host,profile in profiles.items()})
    relay_files=['scripts/run_rpe_q3_relay.py','src/trottertracks/__init__.py',
                 'src/trottertracks/rpe_q3_coordinator.py','src/trottertracks/rpe_q3_timing.py']
    for name in relay_files:files['relay_payload/'+name]=raw(ROOT,source,name)
    relay_workers=[]
    for host in hosts:
        if host['host_id']=='server2':continue
        worker=dict(host_id=host['host_id'],root=host['transport_root'],profile=host['worker_profile'],
                    python=host['transport_python'],priority_group=1 if host['host_id']=='local0-wsl' else 0)
        if host.get('transport')=='wsl':
            worker.update(transport='wsl',distribution=host['distribution'],participation_file=config['participation_file'])
        else:worker['ssh_alias']=host['ssh_alias']
        relay_workers.append(worker)
    relay_template=dict(template_only=True,note='Copy for a separately adopted new run. Fill the controller ports from that run ENDPOINT; never use an old artificial controller. This package starts no relay.',
        state_directory='__SET_NEW_LOCAL0_RELAY_STATE_DIRECTORY__',controller=dict(ssh_alias='qg2',local_port='__SET_NEW_LOCAL_FORWARD_PORT__',remote_port='__FROM_ADOPTED_RUN_ENDPOINT__'),
        workers=relay_workers,poll_seconds=1,retry_backoff_max_seconds=30)
    options=dict(template_only=True,implementation=source,note='Settings only; adopt a new scientific round and bind its exact Q3 readiness policy, source, budget and limits before starting.',
        executor=dict(scheduler='concurrent',max_concurrent_jobs='auto',distributed=dict(
            bind='127.0.0.1',port=0,computation_root=profiles['server2']['root'],computation_source='28d976b99cdf8329adcc1b79327e329e947ef774',
            local_worker=profiles['server2'],preparation_profiles={host:str(output/'profiles'/f'{host}.json')for host in profiles},
            host_eligibility='__GENERATED_FOR_THE_ADOPTED_ROUND__',required_hosts=[],
            worker_script=next(h['transport_root']for h in hosts if h['host_id']=='server2')+'/scripts/run_rpe_q3_worker.py',
            auto_allocation=dict(priority_groups=[['server2','server1','local'],['local0-wsl']],offer_ttl_seconds=3,discovery_grace_seconds=1),
            host_admission=dict(schema='RPE_Q3_HOST_ADMISSION_V1',hosts={'local0-wsl':dict(accepting=True,optional=True)}))))
    put(output/'RELAY_TEMPLATE.json',relay_template);put(output/'DISTRIBUTED_OPTIONS.json',options)
    files['RELAY_TEMPLATE.json']=encode(relay_template);files['DISTRIBUTED_OPTIONS.json']=encode(options)
    manifest=dict(schema='RPE_PREPARATION_DEPLOYMENT_PACKAGE_V1',mode='dry-check',source=source,hosts=[h for h in hosts if h['host_id']!='server2'],
        server2=next(h for h in hosts if h['host_id']=='server2'),participation_file=config['participation_file'],
        return_directory=str(ROOT/'docs/research/rpe_distributed_preparation/remote_results'),
        return_transport=dict(ssh_alias='qg2',transport_python='python3'),files={name:sha(raw)for name,raw in files.items()},scientific_calls=0,compile_calls=0,queue_operations=0)
    files['MANIFEST.json']=encode(manifest)
    package=output/'PREPARATION_DEPLOYMENT.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED)as archive:
        for name,data in sorted(files.items()):archive.writestr(zipfile.ZipInfo(name,date_time=(2026,11,17,0,0,0)),data)
    put(output/'MANIFEST.json',manifest)
    for name in ['run_rpe_preparation_deployment.py','legacy_auto_deployment.py']:(output/name).write_bytes(files[name])
    digest=sha(package.read_bytes())
    ps="""$ErrorActionPreference = 'Stop'
$D = Join-Path $env:USERPROFILE 'rpe-preparation-deploy-SOURCE'
New-Item -ItemType Directory -Force -Path $D | Out-Null
$Z = Join-Path $D 'PREPARATION_DEPLOYMENT.zip'
scp 'qg2:REMOTEZIP' $Z
if ($LASTEXITCODE -ne 0) { throw 'Package download failed' }
if ((Get-FileHash -LiteralPath $Z -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ZIPHASH') { throw 'Package hash differs' }
Expand-Archive -LiteralPath $Z -DestinationPath $D -Force
py -3 -B (Join-Path $D 'run_rpe_preparation_deployment.py') --bundle $Z --bundle-sha256 ZIPHASH --output (Join-Path $D 'results')
""".replace('SOURCE',source_name).replace('REMOTEZIP',str(package)).replace('ZIPHASH',digest)
    (output/'RUN_PREPARATION_DEPLOYMENT.ps1').write_text(ps)
    put(output/'PACKAGE.json',dict(source=source,zip_sha256=digest,zip=str(package),tasks=len(tasks),target_hosts=3,server2_profile=str(output/'profiles/server2.json')))
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True)
    parser.add_argument('--tasks',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();tasks=read(args.tasks)if args.tasks else saved_examples(args.source)
    if isinstance(tasks,dict):tasks=tasks['tasks']
    manifest=create(tasks,args.source,args.output)
    print(json.dumps(dict(source=args.source,tasks=len(tasks),hosts=len(manifest['hosts']),output=str(args.output))))


if __name__=='__main__':main()
