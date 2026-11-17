"""Source-bound signal/compile/audit adapter used by the ordinary pull worker.

Only the command supplied by the central planner is executed. Paths, Python and
compiler resolve from a stage profile; output workspaces and Git publications
are owned by one attempt. Dry-check probes identities, never scientific leaves.
"""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .rpe_preparation_artifacts import ArtifactCache, artifact_path, artifact_hash, input_manifest, worker_cache
from .rpe_q3_timing import append as timing_append

STAGES = ('signal', 'compile', 'audit')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def put(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def is_preparation(value):return (value.get('request',value).get('route')or{}).get('kind')=='preparation'


def preparation_profile_core(profile):
    excluded={'work_dir','slots','slot_policy','route_profiles','preparation_profiles','artifact_cache_directory'}
    return {key:deepcopy(value) for key,value in profile.items() if key not in excluded}


def select_preparation_profile(base, assignment):
    request=assignment.get('request',assignment);spec=request['preparation']
    matches=[rule for rule in base.get('preparation_profiles',[]) if
        rule.get('stage')==spec.get('stage') and rule.get('provider')==spec.get('provider') and
        (not rule.get('source_commit') or rule['source_commit']==spec['source']['commit'])]
    if len(matches)>1:raise ValueError('Ambiguous preparation stage/provider profile')
    if matches:selected=deepcopy(matches[0]['profile'])
    elif base.get('preparation_stage')==spec.get('stage') and base.get('preparation_provider')==spec.get('provider'):selected=deepcopy(base)
    elif spec.get('artificial') is True:selected=deepcopy(base)
    else:raise ValueError('No verified preparation profile for stage/provider')
    if selected.get('host_id')!=base.get('host_id'):raise ValueError('Preparation profile belongs to another host')
    for name in ('work_dir','slots','slot_policy','artifact_cache_directory'):
        if name in base:selected[name]=deepcopy(base[name])
    selected.pop('route_profiles',None);selected.pop('preparation_profiles',None)
    selected['preparation_stage']=spec.get('stage');selected['preparation_provider']=spec.get('provider')
    bound=assignment.get('host_execution_profile')
    if bound is not None:
        if digest(bound)!=assignment.get('host_execution_profile_sha256') or preparation_profile_core(selected)!=bound:
            raise ValueError('Preparation requires another verified stage profile; fallback forbidden')
    return selected


def _head(root):return subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()


def _source(profile, source):
    root=(profile.get('preparation_sources')or{}).get(source['commit'])
    if isinstance(root,dict):root=root.get('root')
    if root is None and _head(profile['root'])==source['commit']:root=profile['root']
    if root is None:raise ValueError('Preparation SOURCE view missing: '+source['commit'])
    root=Path(root).resolve()
    if _head(root)!=source['commit']:raise ValueError('Preparation SOURCE HEAD differs')
    if not isinstance(source.get('files'),dict) or not source['files']:
        raise ValueError('Preparation requires fixed scientific SOURCE files')
    for name,sha in source['files'].items():
        path=root.joinpath(*artifact_path(name).parts)
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('Preparation SOURCE hash differs: '+name)
    return root


def _runtime(profile, requirements):
    python=profile['interpreter']
    if not Path(python).is_absolute() or not Path(python).is_file():raise ValueError('Preparation Python missing: '+python)
    probe='import sys,json,platform,importlib.metadata;print(json.dumps(dict(executable=sys.executable,version=sys.version,version_info=list(sys.version_info),platform=platform.platform(),packages={name:importlib.metadata.version(name)for name in json.load(sys.stdin)})))'
    result=subprocess.run([python,'-B','-c',probe],input=json.dumps(list(requirements.get('packages',{}))),text=True,capture_output=True)
    if result.returncode:raise ValueError('Preparation runtime/package probe failed: '+result.stderr.strip()[-1500:])
    runtime=json.loads(result.stdout)
    if runtime['executable']!=python:
        actual=Path(runtime['executable']).resolve()
        runtime.update(interpreter_invocation=python,launcher_sha256=hashlib.sha256(Path(python).read_bytes()).hexdigest(),actual_executable_realpath=str(actual),actual_binary_sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
    prefix=requirements.get('version_info_prefix',[])
    if runtime['version_info'][:len(prefix)]!=prefix:raise ValueError('Preparation Python version differs')
    for name,expected in requirements.get('packages',{}).items():
        if expected is not None and runtime['packages'][name]!=expected:raise ValueError('Preparation package version differs: '+name)
    expected=profile.get('runtime')or{}
    for name,value in expected.items():
        if runtime.get(name)!=value:raise ValueError('Preparation runtime contract differs: '+name)
    tools={}
    for name,requirement in requirements.get('tools',{}).items():
        configured=(profile.get('preparation_tools')or{}).get(name)
        configured={'path':configured}if isinstance(configured,str)else configured
        if not configured or not Path(configured['path']).is_file():raise ValueError('Preparation compiler/tool missing: '+name)
        path=Path(configured['path']).resolve();version=subprocess.check_output([str(path),*configured.get('version_args',['--version'])],text=True,stderr=subprocess.STDOUT)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        needed=requirement if isinstance(requirement,dict)else {'version_contains':requirement}
        if needed.get('sha256') and sha!=needed['sha256']:raise ValueError('Preparation compiler/tool hash differs: '+name)
        if configured.get('sha256') and sha!=configured['sha256']:raise ValueError('Profile compiler/tool hash differs: '+name)
        if needed.get('version_contains') and needed['version_contains']not in version:raise ValueError('Preparation compiler/tool version differs: '+name)
        tools[name]=dict(path=str(path),sha256=sha,version=version.strip())
    return runtime,tools


def current_preparation_fingerprint(assignment, base_profile):
    request=assignment['request'];spec=request['preparation'];profile=select_preparation_profile(base_profile,assignment)
    if spec.get('stage')not in STAGES:raise ValueError('Unknown preparation stage')
    requirements=spec.get('runtime_requirements',{})
    if spec.get('stage')=='compile'and spec.get('artificial')is not True and not requirements.get('packages',{}).get('qiskit') and not requirements.get('tools'):
        raise ValueError('Compile requires the fixed compiler version/package or tool contract')
    if request['request_id']!=assignment['job_id'] or digest(request)!=assignment['request_sha256']:raise ValueError('Preparation task identity differs')
    source=spec['source']
    if assignment.get('source_manifest')not in (None,source):raise ValueError('Preparation assignment SOURCE differs')
    root=_source(profile,source)
    views={source['commit']:str(root)}
    for alias,workspace in spec.get('workspaces',{}).items():views[alias]=str(_source(profile,workspace['source']))
    for locator in spec.get('input_hashes',[]):
        artifact_path(locator['path'])
        raw=subprocess.check_output(['git','show',locator['commit']+':'+locator['path']],cwd=root)
        if hashlib.sha256(raw).hexdigest()!=locator['sha256']:raise ValueError('Preparation fixed Git input differs: '+locator['path'])
    adapter = profile.get('preparation_adapter_manifest')
    if spec.get('artificial') is not True and not adapter:
        raise ValueError('Fixed preparation adapter manifest missing for this stage')
    if adapter:
        adapter_root=Path(profile.get('preparation_adapter_root') or Path(__file__).resolve().parents[2]).resolve()
        if _head(adapter_root)!=adapter['commit']:raise ValueError('Preparation adapter SOURCE HEAD differs')
        for name,sha in adapter['files'].items():
            path=adapter_root.joinpath(*artifact_path(name).parts)
            if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Preparation adapter SOURCE hash differs: '+name)
    runtime,tools=_runtime(profile,spec.get('runtime_requirements',{}))
    inputs=input_manifest(request)
    for bundle in spec.get('input_git_bundles',[]):artifact_hash(bundle['sha256'])
    command=spec.get('command')
    if not isinstance(command,list)or not command or any(not isinstance(v,str)for v in command):raise ValueError('Preparation command must be explicit argv')
    if not spec.get('outputs'):raise ValueError('Declared preparation outputs required')
    for name in spec['outputs']:artifact_path(name)
    return dict(schema='RPE_PREPARATION_CURRENT_HOST_FINGERPRINT_V1',stage=spec['stage'],provider=spec.get('provider'),request_sha256=assignment['request_sha256'],source_manifest=source,input_hashes=spec.get('input_hashes',[]),input_artifacts=inputs,profile=preparation_profile_core(profile),views=views,runtime=runtime,tools=tools)


def dry_check(request, profile, source_manifest=None):
    assignment=dict(request=request,request_sha256=digest(request),job_id=request['request_id'],source_manifest=source_manifest or request['preparation']['source'])
    fingerprint=current_preparation_fingerprint(assignment,profile)
    return dict(status='DRY_CHECK_PASS',mode='dry-check',request_id=request['request_id'],route='preparation',stage=request['preparation']['stage'],fingerprint=fingerprint,scientific_calls=0)


def assigned_artifacts(assignment):
    manifest=input_manifest(assignment['request'])
    for name,sha in assignment.get('artifact_manifest',{}).items():
        name=str(artifact_path(name));artifact_hash(sha)
        if name in manifest and manifest[name]!=sha:raise ValueError('Assignment changed an explicit input artifact')
        manifest[name]=sha
    return manifest


def _guard_owned_source(root, source):
    for name,sha in source['files'].items():
        path=Path(root).joinpath(*artifact_path(name).parts)
        if not path.is_file()or path.is_symlink()or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('Imported workspace changed fixed scientific SOURCE: '+name)


def _clone(source, destination, bundles, cache, checkout_input_head=True):
    destination=Path(destination)
    if destination.exists():raise ValueError('Preparation workspace already exists; never retry science')
    subprocess.run(['git','clone','--shared','--no-checkout',str(source),str(destination)],check=True,capture_output=True)
    commit=_head(source)
    subprocess.run(['git','checkout','-b','preparation-owned',commit],cwd=destination,check=True,capture_output=True)
    subprocess.run(['git','remote','remove','origin'],cwd=destination,check=True,capture_output=True)
    subprocess.run(['git','config','user.name','HIROMU1015'],cwd=destination,check=True)
    subprocess.run(['git','config','user.email','u411612g@ecs.osaka-u.ac.jp'],cwd=destination,check=True)
    for bundle in bundles:
        sha=bundle['sha256'];path=cache.path(sha)
        if cache.missing([sha]):raise ValueError('Missing source Git bundle')
        subprocess.run(['git','bundle','verify',str(path)],cwd=destination,check=True,capture_output=True)
        subprocess.run(['git','fetch',str(path),bundle.get('ref','HEAD')],cwd=destination,check=True,capture_output=True)
        if checkout_input_head:
            subprocess.run(['git','reset','--hard','FETCH_HEAD'],cwd=destination,check=True,capture_output=True)
    return destination


def _export_bundle(root, source, output):
    new=subprocess.check_output(['git','rev-list','HEAD','^'+source['commit']],cwd=root,text=True).strip()
    if not new:return None
    subprocess.run(['git','bundle','create',str(output),'HEAD','^'+source['commit']],cwd=root,check=True,capture_output=True)
    return dict(path=output.name,ref='HEAD',head=_head(root),base=source['commit'])


def execute(assignment, base_profile, directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    request=assignment['request'];spec=request['preparation'];profile=select_preparation_profile(base_profile,assignment)
    if spec.get('dry_check_only') is True:raise ValueError('Inspection-only saved-input task cannot execute a scientific leaf')
    fingerprint=current_preparation_fingerprint(assignment,base_profile)
    marker=directory/'PREPARATION_STARTED.json'
    with marker.open('x')as stream:json.dump(dict(request_sha256=assignment['request_sha256'],automatic_retry=False),stream)
    cache=worker_cache(profile);artifact_inputs=assigned_artifacts(assignment);cache.materialize(artifact_inputs,directory/'inputs')
    source=Path(fingerprint['views'][spec['source']['commit']]);owned={}
    if spec.get('source_workspace'):
        source=_clone(source,directory/'workspace',spec.get('input_git_bundles',[]),cache);owned['source']=(source,spec['source'])
    replacements={'@PYTHON@':profile['interpreter'],'@SOURCE@':str(source),'@WORK@':str(directory),'@INPUTS@':str(directory/'inputs'),'@TRANSPORT@':str(Path(__file__).resolve().parents[2])}
    for alias,workspace in spec.get('workspaces',{}).items():
        if not alias.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe preparation workspace alias')
        target=_clone(fingerprint['views'][alias],directory/'workspaces'/alias,workspace.get('input_git_bundles',[]),cache,workspace.get('checkout_input_head',True))
        replacements['@WORKSPACE:'+alias+'@']=str(target);owned[alias]=(target,workspace['source'])
    for name,tool in fingerprint['tools'].items():replacements['@TOOL:'+name+'@']=tool['path']
    def resolve(value):
        for token,path in replacements.items():value=value.replace(token,path)
        return value
    bindings={str(old):resolve(value)for old,value in spec.get('path_bindings',{}).items()}
    command=[]
    for argument in spec['command']:
        for old,new in sorted(bindings.items(),key=lambda pair:-len(pair[0])):argument=argument.replace(old,new)
        command.append(resolve(argument))
    if command[0]!=profile['interpreter'] and command[0]not in {t['path']for t in fingerprint['tools'].values()}:raise ValueError('Command does not use verified Python/compiler')
    # Inputs may contain prior owned workspace outputs. Materialize after clone.
    target_inputs=spec.get('materialize_inputs')
    if target_inputs is None:
        target_inputs={name:name for name in artifact_inputs if name.startswith(('workspace/','workspaces/','private/'))}
    for input_name,destination in target_inputs.items():
        path=directory.joinpath(*artifact_path(destination).parts)
        path.parent.mkdir(parents=True,exist_ok=True)
        data=(directory/'inputs').joinpath(*artifact_path(input_name).parts).read_bytes()
        if path.exists()and path.read_bytes()!=data:raise ValueError('Input would replace different fixed workspace bytes')
        if not path.exists():path.write_bytes(data)
    _guard_owned_source(source,spec['source'])
    for alias,(workspace,manifest)in owned.items():_guard_owned_source(workspace,manifest)
    start=time.monotonic();timing_append(directory/'JOB_STAGES.jsonl','PREPARATION_BODY_START',stage=spec['stage'])
    with(directory/'PREPARATION_BODY.log').open('xb')as log:
        process=subprocess.run(command,cwd=source,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RPE_PREPARATION_PATH_MAP=json.dumps(bindings),RPE_PREPARATION_PYTHON=profile['interpreter']),stdout=log,stderr=subprocess.STDOUT)
    timing_append(directory/'JOB_STAGES.jsonl','PREPARATION_BODY_END',stage=spec['stage'],returncode=process.returncode)
    receipt=dict(command=command,source=spec['source'],runtime=fingerprint['runtime'],tools=fingerprint['tools'],returncode=process.returncode if process.returncode>=0 else None,signal=-process.returncode if process.returncode<0 else None,wall_seconds=time.monotonic()-start,automatic_retry=False)
    put(directory/'PREPARATION_RECEIPT.json',receipt)
    outputs=list(spec['outputs']);bundles=[]
    for name in spec.get('optional_outputs',[]):
        artifact_path(name)
        if directory.joinpath(*Path(name).parts).exists():outputs.append(name)
    if process.returncode==0:
        for alias,(workspace,manifest)in owned.items():
            path=directory/(alias+'-output.bundle');bundle=_export_bundle(workspace,manifest,path)
            if bundle:
                bundle['workspace']='workspace'if alias=='source'else'workspaces/'+alias
                outputs.append(path.name);bundles.append(bundle)
        artifacts=cache.capture(directory,outputs)
    else:artifacts={}
    outcome=dict(status='PREPARATION_COMPLETE'if process.returncode==0 else'PREPARATION_FAILED',route='preparation',request_id=request['request_id'],task_id=request['request_id'],request_sha256=assignment['request_sha256'],stage=spec['stage'],artifacts=artifacts,git_bundles=bundles,source=spec['source'],source_manifest=spec['source'],execution_profile_sha256=digest(preparation_profile_core(profile)),resolved_path_bindings=bindings,automatic_retry=False)
    put(directory/'ARTIFACT_MANIFEST.json',artifacts);put(directory/'OUTCOME.json',outcome)
    return outcome


def collect(assignment, profile, directory):
    directory=Path(directory);outcome=read(directory/'OUTCOME.json')
    if outcome.get('request_id')!=assignment['job_id']or outcome.get('request_sha256')!=assignment['request_sha256']:raise ValueError('Preparation outcome identity differs')
    if outcome.get('status')not in ('PREPARATION_COMPLETE','PREPARATION_FAILED'):raise ValueError('No established preparation outcome')
    cache=worker_cache(profile)
    if cache.missing(outcome.get('artifacts',{}).values()):raise ValueError('Completed preparation artifact missing')
    for path,sha in outcome.get('artifacts',{}).items():
        actual=directory.joinpath(*artifact_path(path).parts)
        if not actual.is_file()or hashlib.sha256(actual.read_bytes()).hexdigest()!=sha:raise ValueError('Completed preparation output differs: '+path)
    return outcome


def validate_preparation_fingerprint(assignment, base_profile, fingerprint):
    """Bind a fresh metadata-only host report to the central stage contract.

    The host's existing source/runtime probes establish actual facts; this pure
    check compares those facts with the configured profile and exact REQUEST.
    It never starts a provider or treats Q3 readiness as preparation readiness.
    """
    request=assignment['request'];spec=request['preparation']
    selected=select_preparation_profile(base_profile,assignment)
    expected=dict(schema='RPE_PREPARATION_CURRENT_HOST_FINGERPRINT_V1',stage=spec['stage'],
        provider=spec.get('provider'),request_sha256=assignment['request_sha256'],source_manifest=spec['source'],
        input_hashes=spec.get('input_hashes',[]),input_artifacts=input_manifest(request),profile=preparation_profile_core(selected))
    for key,value in expected.items():
        if fingerprint.get(key)!=value:raise ValueError('Current preparation fingerprint differs: '+key)
    if digest(request)!=assignment['request_sha256']or request['request_id']!=assignment['job_id']:
        raise ValueError('Current preparation REQUEST identity differs')
    for alias,source in [(spec['source']['commit'],spec['source'])]+[(name,item['source'])for name,item in spec.get('workspaces',{}).items()]:
        configured=(selected.get('preparation_sources')or{}).get(source['commit'],selected.get('root'))
        if isinstance(configured,dict):configured=configured.get('root')
        if fingerprint.get('views',{}).get(alias)!=str(configured):
            raise ValueError('Current preparation SOURCE view differs: '+alias)
    if spec.get('artificial')is not True and not selected.get('preparation_adapter_manifest'):
        raise ValueError('Fixed preparation adapter manifest missing')
    runtime=fingerprint.get('runtime')or{}
    python=selected['interpreter']
    if runtime.get('executable')!=python:
        if runtime.get('interpreter_invocation')!=python or not runtime.get('actual_executable_realpath'):
            raise ValueError('Current preparation actual interpreter differs')
        artifact_hash(runtime.get('launcher_sha256'));artifact_hash(runtime.get('actual_binary_sha256'))
    if not runtime.get('version')or not isinstance(runtime.get('version_info'),list):raise ValueError('Current preparation runtime observation missing')
    requirements=spec.get('runtime_requirements',{})
    prefix=requirements.get('version_info_prefix',[])
    if runtime['version_info'][:len(prefix)]!=prefix:raise ValueError('Current preparation Python version differs')
    for name,version in requirements.get('packages',{}).items():
        observed=runtime.get('packages',{}).get(name)
        if not isinstance(observed,str)or(version is not None and observed!=version):raise ValueError('Current preparation package differs: '+name)
    for name,value in (selected.get('runtime')or{}).items():
        if runtime.get(name)!=value:raise ValueError('Current preparation runtime contract differs: '+name)
    tools=fingerprint.get('tools')or{}
    if set(tools)!=set(requirements.get('tools',{})):raise ValueError('Current preparation compiler/tool set differs')
    for name,requirement in requirements.get('tools',{}).items():
        observed=tools[name];configured=(selected.get('preparation_tools')or{}).get(name)
        configured={'path':configured}if isinstance(configured,str)else configured
        if not configured or observed.get('path')!=configured.get('path'):raise ValueError('Current preparation compiler/tool path differs: '+name)
        artifact_hash(observed.get('sha256'))
        requirement=requirement if isinstance(requirement,dict)else{'version_contains':requirement}
        for item in (requirement,configured):
            if item.get('sha256')and observed['sha256']!=item['sha256']:raise ValueError('Current preparation compiler/tool hash differs: '+name)
        if requirement.get('version_contains')and requirement['version_contains']not in observed.get('version',''):
            raise ValueError('Current preparation compiler/tool version differs: '+name)
    return preparation_profile_core(selected)
