"""Source-bound signal/compile/audit adapter used by the ordinary pull worker.

Only the command supplied by the central planner is executed. Paths, Python and
compiler resolve from a stage profile; output workspaces and Git publications
are owned by one attempt. Dry-check probes identities, never scientific leaves.
"""
from copy import deepcopy
import hashlib
import fcntl
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
PAULI_BINDING_ORIGIN = 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def put(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value,indent=2,default=str)+'\n');temporary.replace(path)


def is_preparation(value):return (value.get('request',value).get('route')or{}).get('kind')=='preparation'


def bind_df_signal_workers(command, workers):
    if (type(workers) is not int or workers < 1 or len(command) < 3
            or Path(command[2]).name != 'run_pr_df_candidate_signal_jobs.py'):
        raise ValueError('DF signal reservation requires a fixed runner and positive CPU count')
    if '--workers' in command:
        index = command.index('--workers')
        if command[index+1:index+2] != [str(workers)]:
            raise ValueError('DF signal worker count differs from its reserved CPUs')
        return command
    return [*command, '--workers', str(workers)]


def preparation_profile_core(profile):
    excluded={'work_dir','slots','slot_policy','route_profiles','preparation_profiles','artifact_cache_directory','preparation_metadata_cache_directory','pbs'}
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


def _required_path_facts(profile, requirements):
    facts={};configured=profile.get('preparation_paths')or{}
    for alias,required in requirements.items():
        if not alias.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe required runtime path alias')
        target=configured.get(alias,configured.get(required['controller_path']))
        if isinstance(target,dict):target=target.get('path')
        if not target:raise ValueError('Required runtime path is not mapped in this host profile: '+required['controller_path'])
        root=Path(target).resolve();kind=required.get('kind')
        if kind=='file':
            if not root.is_file():raise ValueError('Required runtime file missing: '+str(root))
            actual=hashlib.sha256(root.read_bytes()).hexdigest()
            if actual!=required['sha256']:raise ValueError('Required runtime file hash differs: '+required['controller_path'])
            facts[alias]=dict(path=str(root),kind=kind,sha256=actual)
        elif kind=='directory':
            if not root.is_dir():raise ValueError('Required runtime provider directory missing: '+str(root))
            if not required.get('files'):raise ValueError('Required runtime provider source manifest missing: '+required['controller_path'])
            for name,sha in required['files'].items():
                path=root.joinpath(*artifact_path(name).parts)
                if not path.is_file()or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
                    raise ValueError('Required runtime provider SOURCE differs: '+name)
            facts[alias]=dict(path=str(root),kind=kind,files=deepcopy(required['files']))
        else:raise ValueError('Unknown required runtime path type')
    return facts


def metadata_bundle_hashes(assignment):
    spec=assignment.get('request',assignment)['preparation']
    bundles=list(spec.get('input_git_bundles',[]))
    for workspace in spec.get('workspaces',{}).values():bundles.extend(workspace.get('input_git_bundles',[]))
    return list(dict.fromkeys(artifact_hash(bundle['sha256'])for bundle in bundles))


def _metadata_repository_path(profile, root):
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=root,text=True).strip())
    if not common.is_absolute():common=Path(root)/common
    directory=Path(profile.get('preparation_metadata_cache_directory')or Path(profile['work_dir']).parent/'preparation-metadata-cache')
    return directory.resolve()/digest(str(common.resolve()))[:16]/'objects.git'


def _metadata_input_repository(profile, root, spec):
    bundles=list(spec.get('input_git_bundles',[]))
    for workspace in spec.get('workspaces',{}).values():bundles.extend(workspace.get('input_git_bundles',[]))
    if not bundles:return root
    cache=worker_cache(profile)
    hashes=[artifact_hash(bundle['sha256'])for bundle in bundles]
    if cache.missing(hashes):raise ValueError('Preparation metadata Git bundle missing before preflight')
    target=_metadata_repository_path(profile,root)
    directory=target.parent;directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'IMPORTED.json'
    with(directory/'IMPORT.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():subprocess.run(['git','clone','--shared','--bare',str(root),str(target)],check=True,capture_output=True)
        imported=read(marker)if marker.exists()else[]
        for bundle in bundles:
            sha=bundle['sha256']
            if sha in imported:continue
            path=cache.path(sha)
            subprocess.run(['git','bundle','verify',str(path)],cwd=target,check=True,capture_output=True)
            ref = bundle.get('ref','HEAD')
            heads=subprocess.check_output(['git','bundle','list-heads',str(path)],cwd=target,text=True).splitlines()
            if not any(line.split(maxsplit=1)[1] == ref for line in heads):
                raise ValueError('Preparation metadata bundle lacks its declared ref: '+ref)
            # Import this verified local bundle directly. A fetch checks the
            # connectivity of unrelated history, which can be sparse locally.
            # The fingerprint below still reads and hashes every declared input.
            subprocess.run(['git','bundle','unbundle',str(path),ref],cwd=target,check=True,capture_output=True)
            for index,line in enumerate(heads):
                commit=line.split()[0]
                subprocess.run(['git','update-ref','refs/rpe-metadata/'+sha+'/'+str(index),commit],cwd=target,check=True,capture_output=True)
            imported.append(sha);put(marker,imported)
    return target


def current_preparation_fingerprint(assignment, base_profile):
    request=assignment['request'];spec=request['preparation'];profile=select_preparation_profile(base_profile,assignment)
    pauli_calibration_manifest(spec)
    if spec.get('stage')not in STAGES:raise ValueError('Unknown preparation stage')
    if spec.get('publish_pauli_binding') is True:
        publication = profile.get('pauli_binding_publication') or {}
        reference = publication.get('base_ref', '')
        if (spec.get('stage') != 'compile' or spec.get('provider') != 'PAULI_COMPILE'
                or publication.get('origin') != PAULI_BINDING_ORIGIN
                or not isinstance(reference, str) or not reference.startswith('refs/heads/codex/')
                or subprocess.run(['git', 'check-ref-format', reference], capture_output=True).returncode):
            raise ValueError('Fixed Pauli binding publication profile is missing')
    if spec.get('parallel_df_price_axes') is True:
        if spec.get('provider') not in ('DF_CALIBRATION', 'DF_HOLDOUT'):
            raise ValueError('Parallel DF price execution requires a calibration or holdout request')
        if type(profile.get('task_cpu_count')) is not int or profile['task_cpu_count'] < 1:
            raise ValueError('Parallel DF price execution requires a fixed CPU reservation')
    if spec.get('inspection_blockers'):raise ValueError('Saved inspection input dependencies unresolved: '+'; '.join(spec['inspection_blockers']))
    requirements=spec.get('runtime_requirements',{})
    if spec.get('stage')=='compile'and spec.get('artificial')is not True and not requirements.get('packages',{}).get('qiskit') and not requirements.get('tools'):
        raise ValueError('Compile requires the fixed compiler version/package or tool contract')
    if request['request_id']!=assignment['job_id'] or digest(request)!=assignment['request_sha256']:raise ValueError('Preparation task identity differs')
    source=spec['source']
    if assignment.get('source_manifest')not in (None,source):raise ValueError('Preparation assignment SOURCE differs')
    root=_source(profile,source)
    views={source['commit']:str(root)}
    for alias,workspace in spec.get('workspaces',{}).items():views[alias]=str(_source(profile,workspace['source']))
    for alias,required in spec.get('required_views',{}).items():
        if not alias.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe required SOURCE alias')
        views['view:'+alias]=str(_source(profile,required['source']))
    path_facts=_required_path_facts(profile,spec.get('required_paths',{}))
    input_root=_metadata_input_repository(profile,root,spec)
    for locator in spec.get('input_hashes',[]):
        artifact_path(locator['path'])
        raw=subprocess.check_output(['git','show',locator['commit']+':'+locator['path']],cwd=input_root)
        if hashlib.sha256(raw).hexdigest()!=locator['sha256']:raise ValueError('Preparation fixed Git input differs: '+locator['path'])
    declared=input_manifest(request)
    for locator in spec.get('working_input_hashes',[]):
        if declared.get(locator['path'])!=locator['sha256']:raise ValueError('Preparation WORKING input is not bound to its immutable artifact')
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
    return dict(schema='RPE_PREPARATION_CURRENT_HOST_FINGERPRINT_V1',stage=spec['stage'],provider=spec.get('provider'),request_sha256=assignment['request_sha256'],source_manifest=source,input_hashes=spec.get('input_hashes',[]),working_input_hashes=spec.get('working_input_hashes',[]),required_views=spec.get('required_views',{}),required_paths=spec.get('required_paths',{}),runtime_paths=path_facts,input_artifacts=inputs,profile=preparation_profile_core(profile),views=views,runtime=runtime,tools=tools)


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


def checkout_directories(source_files):
    directories={'scripts','src','configs','schemas','references'}
    for name in source_files:
        relative=artifact_path(name)
        if len(relative.parts)>1 and relative.parts[0]not in directories:
            directories.add(relative.parent.as_posix())
    return directories


def output_directories(outputs, prefix, workspace):
    directories=set()
    for name in outputs:
        relative=Path(*artifact_path(name).parts)
        try:relative=relative.relative_to(prefix)
        except ValueError:continue
        if not relative.parts:continue
        target=workspace/relative
        cone=relative if target.is_dir()or (not target.exists()and not relative.suffix) else relative.parent
        if cone.parts:directories.add(cone.as_posix())
    return directories


def _clone(source, destination, bundles, cache, checkout_input_head=True, source_files=(), *, metadata_repository=None):
    destination=Path(destination)
    if destination.exists():raise ValueError('Preparation workspace already exists; never retry science')
    clone=['git','clone','--shared','--no-checkout']
    if bundles and metadata_repository is not None:
        # Preflight already imported and pinned these objects on this host.
        # The working tree and new commits still belong to this attempt.
        clone.extend(['--reference',str(metadata_repository)])
    subprocess.run(clone+[str(source),str(destination)],check=True,capture_output=True)
    commit=_head(source)
    directories=checkout_directories(source_files)
    # Older supported Git versions interpret --cone after set as a path.
    # Keep untouched directories as sparse tree entries. A preparation commit
    # should not require the blobs beneath every unrelated archive subtree.
    subprocess.run(['git','sparse-checkout','init','--cone','--sparse-index'],
                   cwd=destination,check=True,capture_output=True)
    subprocess.run(['git','sparse-checkout','set','--stdin'],input='\n'.join(sorted(directories))+'\n',
                   text=True,cwd=destination,check=True,capture_output=True)
    subprocess.run(['git','remote','remove','origin'],cwd=destination,check=True,capture_output=True)
    subprocess.run(['git','config','user.name','HIROMU1015'],cwd=destination,check=True)
    subprocess.run(['git','config','user.email','u411612g@ecs.osaka-u.ac.jp'],cwd=destination,check=True)
    for bundle in bundles:
        sha=bundle['sha256'];path=cache.path(sha)
        if cache.missing([sha]):raise ValueError('Missing source Git bundle')
        subprocess.run(['git','bundle','verify',str(path)],cwd=destination,check=True,capture_output=True)
        ref=bundle.get('ref','HEAD')
        heads=subprocess.check_output(['git','bundle','list-heads',str(path),ref],cwd=destination,text=True).splitlines()
        if len(heads)!=1:raise ValueError('Preparation bundle does not declare its unique checkout ref: '+ref)
        if metadata_repository is None:
            subprocess.run(['git','bundle','unbundle',str(path),ref],cwd=destination,check=True,capture_output=True)
        tip=heads[0].split()[0]
        subprocess.run(['git','cat-file','-e',tip+'^{commit}'],cwd=destination,check=True,capture_output=True)
        if checkout_input_head and bundle.get('checkout_input_head',True):
            commit=tip
    subprocess.run(['git','checkout','-b','preparation-owned',commit],cwd=destination,check=True,capture_output=True)
    return destination


def _include_declared_outputs(owned, directory, outputs):
    """Permit the provider's ordinary git add only inside declared output cones."""
    for alias,(workspace,source)in owned.items():
        directories=output_directories(outputs,workspace.relative_to(directory),workspace)
        if directories:
            subprocess.run(['git','sparse-checkout','add','--stdin'],input='\n'.join(sorted(directories))+'\n',
                           text=True,cwd=workspace,check=True,capture_output=True)


def _export_bundle(root, source, output, input_head=None):
    base=source['commit']
    result=dict(ref='HEAD',head=_head(root),base=base)
    if input_head and input_head!=base:
        ancestry=subprocess.run(['git','merge-base','--is-ancestor',input_head,result['head']],
                                cwd=root,capture_output=True)
        if ancestry.returncode not in (0,1):ancestry.check_returncode()
        if ancestry.returncode==0:
            result['input_head']=input_head
            if input_head==result['head']:return result
            base=input_head
    if not subprocess.check_output(['git','rev-list','-1','HEAD','^'+base],cwd=root,text=True).strip():
        return None
    subprocess.run(['git','bundle','create',str(output),'HEAD','^'+base],cwd=root,check=True,capture_output=True)
    return dict(result,path=output.name)


def pauli_calibration_manifest(spec):
    from .rpe_preparation_paths import (
        PAULI_CALIBRATION_FILES, PAULI_CALIBRATION_SOURCE, PAULI_CALIBRATION_VIEW,
    )
    if spec.get('artificial') is True or spec.get('provider') not in ('PAULI_COMPILE', 'PAULI_AUDIT'):
        return None
    manifest = spec.get('required_views', {}).get(PAULI_CALIBRATION_VIEW, {}).get('source', {})
    if (manifest.get('commit') != PAULI_CALIBRATION_SOURCE
            or set(manifest.get('files', {})) != set(PAULI_CALIBRATION_FILES)
            or (not spec.get('dry_check_only') and 'provider' not in spec.get('workspaces', {}))):
        raise ValueError('Fixed Pauli calibration SOURCE view declaration missing')
    return manifest


def materialize_pauli_calibration_view(spec, fingerprint, directory):
    from .rpe_preparation_paths import PAULI_CALIBRATION_VIEW

    manifest = pauli_calibration_manifest(spec)
    if manifest is None:
        return
    expected = Path(fingerprint['views']['view:' + PAULI_CALIBRATION_VIEW]).resolve(strict=True)
    _guard_owned_source(expected, manifest)
    if not (directory/'workspaces'/'provider').is_dir():
        raise ValueError('Fixed Pauli calibration requires an owned provider workspace')
    alias = directory/'workspaces'/PAULI_CALIBRATION_VIEW
    alias.symlink_to(expected, target_is_directory=True)


def prepare_pauli_binding_publication(profile, assignment, workspace, base):
    """Give the legacy binding check an independently published, scoped ref."""
    publication = profile['pauli_binding_publication']
    origin, reference = publication['origin'], publication['base_ref']
    if _head(workspace) != base:
        raise ValueError('Pauli publication workspace does not start at its fixed public base')
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_SSH_COMMAND='ssh -o BatchMode=yes')
    result = subprocess.check_output(['git', 'ls-remote', '--exit-code', '--refs', origin, reference],
                                     cwd=workspace, env=env, text=True, timeout=25).splitlines()
    if len(result) != 1 or result[0].split()[1] != reference:
        raise ValueError('Pauli publication base ref is unavailable')
    published = result[0].split()[0]
    if published != base:
        raise ValueError('Pauli publication base ref differs from the fixed provider')
    ids = (assignment['job_id'], assignment['attempt_id'])
    if any(not v or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in v) for v in ids):
        raise ValueError('Invalid Pauli publication attempt identity')
    branch = 'codex/rpe-pauli-binding-' + '-'.join(ids)
    subprocess.run(['git', 'branch', '-m', branch], cwd=workspace, check=True)
    subprocess.run(['git', 'remote', 'add', 'origin', origin], cwd=workspace, check=True)
    return dict(RPE_PAULI_BINDING_PUBLIC_BASE=base, GIT_SSH_COMMAND=env['GIT_SSH_COMMAND'],
                GIT_TERMINAL_PROMPT=env['GIT_TERMINAL_PROMPT'])


def execute(assignment, base_profile, directory):
    directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    request=assignment['request'];spec=request['preparation'];profile=select_preparation_profile(base_profile,assignment)
    if spec.get('dry_check_only') is True:raise ValueError('Inspection-only saved-input task cannot execute a scientific leaf')
    fingerprint=current_preparation_fingerprint(assignment,base_profile)
    marker=directory/'PREPARATION_STARTED.json'
    with marker.open('x')as stream:json.dump(dict(request_sha256=assignment['request_sha256'],automatic_retry=False),stream)
    cache=worker_cache(profile);artifact_inputs=assigned_artifacts(assignment)
    stage_inputs=spec.get('stage_inputs',True)
    if stage_inputs:
        cache.materialize(artifact_inputs,directory/'inputs')
    elif cache.missing(artifact_inputs.values()):
        raise ValueError('Missing fixed preparation input artifact')
    source=Path(fingerprint['views'][spec['source']['commit']]);owned={}
    metadata_repository=_metadata_repository_path(profile,source) if metadata_bundle_hashes(assignment) else None
    if spec.get('source_workspace'):
        source=_clone(source,directory/'workspace',spec.get('input_git_bundles',[]),cache,
                      source_files=spec['source']['files'],metadata_repository=metadata_repository)
        owned['source']=(source,spec['source'])
    # Transport releases may change without changing the verified provider
    # wrapper that a preparation command actually imports or executes.
    provider_transport=profile.get('preparation_adapter_root') or Path(__file__).resolve().parents[2]
    replacements={'@PYTHON@':profile['interpreter'],'@SOURCE@':str(source),'@WORK@':str(directory),'@INPUTS@':str(directory/'inputs'),'@TRANSPORT@':str(provider_transport)}
    for alias,workspace in spec.get('workspaces',{}).items():
        if not alias.replace('_','').replace('-','').isalnum():raise ValueError('Unsafe preparation workspace alias')
        target=_clone(fingerprint['views'][alias],directory/'workspaces'/alias,workspace.get('input_git_bundles',[]),cache,
                      workspace.get('checkout_input_head',True),workspace['source']['files'],
                      metadata_repository=metadata_repository)
        replacements['@WORKSPACE:'+alias+'@']=str(target);owned[alias]=(target,workspace['source'])
    materialize_pauli_calibration_view(spec, fingerprint, directory)
    for alias in spec.get('required_views',{}):replacements['@VIEW:'+alias+'@']=fingerprint['views']['view:'+alias]
    for alias,facts in fingerprint.get('runtime_paths',{}).items():replacements['@PATH:'+alias+'@']=facts['path']
    for name,tool in fingerprint['tools'].items():replacements['@TOOL:'+name+'@']=tool['path']
    def resolve(value):
        for token,path in replacements.items():value=value.replace(token,path)
        return value
    bindings={str(old):resolve(value)for old,value in spec.get('path_bindings',{}).items()}
    for alias,required in spec.get('required_views',{}).items():
        if required.get('controller_path'):bindings[required['controller_path']]=replacements['@VIEW:'+alias+'@']
    for alias,required in spec.get('required_paths',{}).items():
        bindings[required['controller_path']]=replacements['@PATH:'+alias+'@']
    command=[]
    for argument in spec['command']:
        for old,new in sorted(bindings.items(),key=lambda pair:-len(pair[0])):argument=argument.replace(old,new)
        command.append(resolve(argument))
    if spec.get('provider') == 'DF_SIGNALS' and profile.get('task_cpu_count') is not None:
        command = bind_df_signal_workers(command, profile['task_cpu_count'])
    if command[0]!=profile['interpreter'] and command[0]not in {t['path']for t in fingerprint['tools'].values()}:raise ValueError('Command does not use verified Python/compiler')
    outcome=dict(route='preparation',request_id=request['request_id'],task_id=request['request_id'],
                 request_sha256=assignment['request_sha256'],stage=spec['stage'],source=spec['source'],
                 source_manifest=spec['source'],execution_profile_sha256=digest(preparation_profile_core(profile)),
                 resolved_path_bindings=bindings,automatic_retry=False)
    try:
        _include_declared_outputs(owned,directory,list(spec['outputs'])+list(spec.get('optional_outputs',[])))
    except subprocess.CalledProcessError as error:
        # Git has already exited and the provider has not started. Return the
        # failure with the original attempt so recovery need not infer it.
        outcome.update(status='PREPARATION_FAILED',artifacts={},git_bundles=[],
                       workspace_error=dict(step='git sparse-checkout add',returncode=error.returncode,
                                            scientific_body_started=False))
        print(str(error),error.stderr or '',file=sys.stderr)
        put(directory/'ARTIFACT_MANIFEST.json',{})
        put(directory/'OUTCOME.json',outcome)
        return outcome
    # Inputs may contain prior owned workspace outputs. Materialize after clone.
    target_inputs=spec.get('materialize_inputs')
    if target_inputs is None:
        target_inputs={name:name for name in artifact_inputs if name.startswith(('workspace/','workspaces/','private/'))}
    for input_name,destination in target_inputs.items():
        path=directory.joinpath(*artifact_path(destination).parts)
        path.parent.mkdir(parents=True,exist_ok=True)
        original=((directory/'inputs').joinpath(*artifact_path(input_name).parts) if stage_inputs
                  else cache.path(artifact_inputs[input_name]))
        data=original.read_bytes()
        if path.exists()and path.read_bytes()!=data:raise ValueError('Input would replace different fixed workspace bytes')
        if not path.exists():path.write_bytes(data)
    _guard_owned_source(source,spec['source'])
    for alias,(workspace,manifest)in owned.items():_guard_owned_source(workspace,manifest)
    input_heads={}
    if spec.get('return_git_delta') is True:
        input_heads={alias:_head(workspace) for alias,(workspace,_) in owned.items()}
    start=time.monotonic();timing_append(directory/'JOB_STAGES.jsonl','PREPARATION_BODY_START',stage=spec['stage'])
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',RPE_PREPARATION_PATH_MAP=json.dumps(bindings),
             RPE_PREPARATION_PYTHON=profile['interpreter'])
    if spec.get('publish_pauli_binding') is True:
        provider, manifest = owned['provider']
        if ('--no-push' not in command or '--phase' not in command
                or command[command.index('--phase')+1:command.index('--phase')+2] != ['compile']):
            raise ValueError('Pauli binding publication requires the fixed compile command')
        env.update(prepare_pauli_binding_publication(profile, assignment, provider, manifest['commit']))
        command.append('--publish-binding')
    if spec.get('parallel_df_price_axes') is True:
        env.update(RPE_DF_PRICE_AXIS_WORKERS=str(profile['task_cpu_count']),
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   RAYON_NUM_THREADS='1', QISKIT_NUM_PROCS='1')
    with(directory/'PREPARATION_BODY.log').open('xb')as log:
        process=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
    timing_append(directory/'JOB_STAGES.jsonl','PREPARATION_BODY_END',stage=spec['stage'],returncode=process.returncode)
    receipt=dict(command=command,source=spec['source'],runtime=fingerprint['runtime'],tools=fingerprint['tools'],returncode=process.returncode if process.returncode>=0 else None,signal=-process.returncode if process.returncode<0 else None,wall_seconds=time.monotonic()-start,automatic_retry=False)
    put(directory/'PREPARATION_RECEIPT.json',receipt)
    outputs=list(spec['outputs']);bundles=[]
    for name in spec.get('optional_outputs',[]):
        artifact_path(name)
        if directory.joinpath(*Path(name).parts).exists():outputs.append(name)
    if process.returncode==0:
        for alias,(workspace,manifest)in owned.items():
            path=directory/(alias+'-output.bundle');bundle=_export_bundle(workspace,manifest,path,input_heads.get(alias))
            if bundle:
                bundle['workspace']='workspace'if alias=='source'else'workspaces/'+alias
                if 'path' in bundle:outputs.append(path.name)
                bundles.append(bundle)
        artifacts=cache.capture(directory,outputs)
    else:artifacts={}
    outcome.update(status='PREPARATION_COMPLETE'if process.returncode==0 else'PREPARATION_FAILED',
                   artifacts=artifacts,git_bundles=bundles)
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
        input_hashes=spec.get('input_hashes',[]),working_input_hashes=spec.get('working_input_hashes',[]),
        required_views=spec.get('required_views',{}),required_paths=spec.get('required_paths',{}),
        input_artifacts=input_manifest(request),profile=preparation_profile_core(selected))
    for key,value in expected.items():
        if fingerprint.get(key)!=value:raise ValueError('Current preparation fingerprint differs: '+key)
    if digest(request)!=assignment['request_sha256']or request['request_id']!=assignment['job_id']:
        raise ValueError('Current preparation REQUEST identity differs')
    expected_views={}
    sources=[(spec['source']['commit'],spec['source'])]+[(name,item['source'])for name,item in spec.get('workspaces',{}).items()]
    sources += [('view:'+name,item['source'])for name,item in spec.get('required_views',{}).items()]
    for alias,source in sources:
        configured=(selected.get('preparation_sources')or{}).get(source['commit'],selected.get('root'))
        if isinstance(configured,dict):configured=configured.get('root')
        expected_views[alias]=str(configured)
    if fingerprint.get('views')!=expected_views:raise ValueError('Current preparation SOURCE views differ')
    declared=input_manifest(request)
    for locator in spec.get('working_input_hashes',[]):
        name=str(artifact_path(locator['path']));sha=artifact_hash(locator['sha256'])
        if declared.get(name)!=sha:raise ValueError('Current preparation WORKING input is not bound to its immutable artifact')
    runtime_paths={};configured_paths=selected.get('preparation_paths')or{}
    for alias,required in spec.get('required_paths',{}).items():
        configured=configured_paths.get(alias,configured_paths.get(required['controller_path']))
        if isinstance(configured,dict):configured=configured.get('path')
        if not isinstance(configured,str)or not configured:raise ValueError('Required runtime path is not configured: '+alias)
        kind=required.get('kind')
        if kind=='file':
            runtime_paths[alias]=dict(path=configured,kind=kind,sha256=artifact_hash(required['sha256']))
        elif kind=='directory':
            files=required.get('files')
            if not isinstance(files,dict)or not files:raise ValueError('Required runtime provider SOURCE manifest missing: '+alias)
            files={str(artifact_path(name)):artifact_hash(sha)for name,sha in files.items()}
            runtime_paths[alias]=dict(path=configured,kind=kind,files=files)
        else:raise ValueError('Unknown required runtime path type')
    if fingerprint.get('runtime_paths')!=runtime_paths:raise ValueError('Current preparation runtime paths differ')
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
    if spec.get('stage')=='compile'and spec.get('artificial')is not True and not requirements.get('packages',{}).get('qiskit')and not requirements.get('tools'):
        raise ValueError('Compile requires the fixed compiler version/package or tool contract')
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
