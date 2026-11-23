"""Place a selected release's components without changing scientific ownership.

The central release registry selects the version. This module only reuses or
installs its immutable assets and runs the worker's existing metadata guards.
Installed assets never change a host's participation/Pause setting.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile

from .rpe_release_registry import fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def manifest_reference(manifest):
    """Canonical identity used outside scientific REQUEST/ID/hash fields."""
    if manifest.get('schema') != 'RPE_RELEASE_COMPONENTS_V1':
        raise ValueError('Unsupported release component manifest')
    return dict(id=manifest['release_id'], manifest_sha256=fingerprint(manifest))


def prepare_worker_assignment(profile, assignment, deployment):
    """Prepare from the host's existing declarations, before any claim."""
    reference=assignment.get('required_release')
    if not deployment or not reference:return
    selected=deployment.get('task_deployments',{}).get(reference['id'],deployment.get('deployment'))
    if selected:
        manifest,layout=(selected[key] if isinstance(selected[key],dict) else read(selected[key])
                         for key in ('manifest','layout'))
        ensure_assignment_components(manifest,layout,assignment,profile,install=True)


def participation_components(manifest, host_layout):
    """Resolve the selected host's transport roles, including PBS's worker."""
    host=host_layout['host_id']
    selected=manifest.get('participation_components_by_host',{}).get(host,
        manifest.get('participation_components',['optimizer','worker']))
    for name in selected:
        component=manifest['components'].get(name)
        if component is None or(component.get('hosts')and host not in component['hosts']):
            raise ValueError('Participation component unavailable on '+host+': '+name)
    return list(selected)


def _git(root, *arguments):
    return subprocess.check_output(['git', *arguments], cwd=root, text=True,
                                   stderr=subprocess.PIPE).strip()


def _source_present(repository, source):
    return subprocess.run(['git','cat-file','-e',source+'^{commit}'],cwd=repository,
                          stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0


def _repository(layout, install):
    repository=Path(layout['repository']).resolve()
    if not repository.exists() and install and layout.get('seed_repository'):
        repository.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.rpe-repository-',dir=repository.parent) as temporary:
            staged=Path(temporary)/'git'
            subprocess.run(['git','clone','--bare','--shared',str(layout['seed_repository']),str(staged)],
                           check=True,capture_output=True)
            try:
                staged.rename(repository)
            except OSError:
                if not repository.is_dir():raise
    return repository


def _ensure_source(repository, source, layout, install):
    if not repository.exists():raise ValueError('SOURCE_REPOSITORY_MISSING: '+str(repository))
    if _source_present(repository, source):return
    if install:
        for bundle in layout.get('bundle_files',[]):
            if not Path(bundle).is_file():continue
            result=subprocess.run(['git','fetch','--no-tags','--no-write-fetch-head',str(bundle),
                                   source+':refs/rpe-components/'+source],cwd=repository,
                                  stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            if result.returncode==0 and _source_present(repository,source):return
    raise ValueError('SOURCE_OBJECT_MISSING: '+source)


def _inspect_component(specification, root):
    source=specification['source']
    head=_git(root,'rev-parse','HEAD')
    if not specification.get('allow_metadata_commits') and head != source:
        raise ValueError('EXISTING_CHECKOUT_SOURCE_DIFFERS: '+str(root))
    # This is the same executable/settings boundary used by the normal Gateway.
    from .rpe_local0_search import fixed_implementation
    fixed_implementation(dict(code_root=str(root),code_source=source))
    for name in specification.get('required_paths',[]):
        path=root/name
        if not path.resolve().is_relative_to(root):raise ValueError('Unsafe component path')
        if not path.exists():raise ValueError('COMPONENT_FILE_MISSING: '+name)
    return head


def _runtime(specification):
    python=Path(specification['python'])
    if not python.is_absolute() or not python.is_file():
        raise ValueError('RUNTIME_INTERPRETER_MISSING: '+str(python))
    program='''import importlib.util,importlib.metadata,json,sys
v=json.loads(sys.argv[1]);missing=[];versions={}
for name,expected in v.items():
 if importlib.util.find_spec(name) is None:missing.append(name);continue
 try:version=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:version=None
 versions[name]=version
 if expected is not None and version!=expected:missing.append(name+'=='+expected)
print(json.dumps(dict(version=list(sys.version_info),version_string=sys.version,modules=versions,missing=missing)))
'''
    result=json.loads(subprocess.check_output([str(python),'-B','-c',program,
        json.dumps(specification.get('modules',{}))],text=True))
    if result['missing']:raise ValueError('RUNTIME_MODULES_MISSING: '+','.join(result['missing']))
    prefix=specification.get('version_info_prefix')
    if prefix and result['version'][:len(prefix)]!=prefix:
        raise ValueError('RUNTIME_VERSION_DIFFERS: '+str(result['version']))
    return result


def _ensure_runtime(specification, install):
    python=Path(specification['python'])
    offline=specification.get('offline_install')
    if not install or not offline or python.exists():
        return _runtime(specification)
    if not python.is_absolute() or python.parts[-2:]!=('bin','python'):
        raise ValueError('Offline runtime must use an absolute dedicated bin/python')
    root=python.parent.parent
    if root.exists() or root.is_symlink():
        return _runtime(specification)
    base,lock,wheels=(Path(offline[key]) for key in ('base_python','requirements','wheelhouse'))
    if not all(p.is_absolute() for p in (base,lock,wheels)) or not base.is_file() or not wheels.is_dir():
        raise ValueError('Offline runtime requires a fixed bootstrap Python, lockfile and local wheels')
    raw=lock.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=offline['requirements_sha256']:
        raise ValueError('Offline runtime lockfile hash differs')
    # Restrict this lockfile to pinned wheels. pip --no-index alone still
    # permits direct URLs and recursive requirements files.
    lines=[line for line in raw.decode().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not all(re.fullmatch(r'[\w.-]+==[\w.+!-]+(?: +--hash=sha256:[0-9a-f]{64})+',line.strip()) for line in lines):
        raise ValueError('Offline runtime lockfile requires pinned hashed wheel entries')
    root.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.rpe-runtime-',dir=root.parent) as temporary:
        staged=Path(temporary)/root.name
        fixed_lock=Path(temporary)/'requirements.txt'
        fixed_lock.write_bytes(raw)
        subprocess.run([str(base),'-B','-m','venv',str(staged)],check=True,capture_output=True)
        executable=staged/'bin/python'
        subprocess.run([str(executable),'-B','-m','pip','--isolated','--disable-pip-version-check','install',
            '--no-index','--no-deps','--only-binary=:all:','--require-hashes','--find-links',str(wheels),
            '-r',str(fixed_lock)],check=True,capture_output=True)
        subprocess.run([str(executable),'-B','-m','pip','--isolated','check'],check=True,capture_output=True)
        _runtime(dict(specification,python=str(executable)))
        # pip entry points and activation scripts embed the installation path.
        for script in (staged/'bin').iterdir():
            if script.is_symlink() or not script.is_file():continue
            content=script.read_bytes()
            if b'\x00' not in content and os.fsencode(staged) in content:
                script.write_bytes(content.replace(os.fsencode(staged),os.fsencode(root)))
        try:
            staged.rename(root)
        except OSError:
            if not root.is_dir():raise
    return _runtime(specification)


def ensure_components(manifest, host_layout, *, install=False, components=None, repository=None, runtimes=None):
    """Reuse verified assets; acquire only missing roots/objects/input files.

    Code comes from the host's object repository or explicit local Git bundles.
    A missing dedicated runtime can use an explicitly registered offline lock.
    """
    reference=manifest_reference(manifest)
    layout=dict(host_layout)
    if repository is not None:layout['repository']=str(repository)
    host=layout['host_id']
    if manifest.get('hosts') and host not in manifest['hosts']:raise ValueError('Host absent from selected release')
    selected=list(components)if components is not None else [name for name,spec in manifest['components'].items()
        if not spec.get('hosts')or host in spec['hosts']]
    unknown=set(selected)-set(manifest['components'])
    if unknown:raise ValueError('Unknown release components: '+','.join(sorted(unknown)))
    report=dict(status='PASS',release_id=reference['id'],manifest_sha256=reference['manifest_sha256'],
        host_id=host,components={},runtimes={},inputs={},holds=[],scientific_calls=0,
        participation_modified=False,not_applicable_components=[name for name,spec in manifest['components'].items()
            if spec.get('hosts')and host not in spec['hosts']])
    object_repository=None
    for name in selected:
        specification=manifest['components'][name];source=specification['source']
        if not isinstance(source,str)or len(source)!=40 or any(c not in '0123456789abcdef'for c in source):
            raise ValueError('Component SOURCE must be a full commit: '+name)
        root_value=layout.get('component_roots',{}).get(name)
        item=dict(source=source,root=root_value,status='HOLD')
        try:
            if not root_value or not Path(root_value).is_absolute():raise ValueError('COMPONENT_ROOT_MISSING: '+name)
            root=Path(root_value).resolve();item['root']=str(root)
            created=False
            if not root.exists():
                if not install:raise ValueError('COMPONENT_NOT_INSTALLED: '+name)
                if object_repository is None:object_repository=_repository(layout,install)
                _ensure_source(object_repository,source,layout,install)
                root.parent.mkdir(parents=True,exist_ok=True)
                # Publish a checkout only after source verification. Interrupted
                # transfers leave the previous installed version untouched.
                with tempfile.TemporaryDirectory(prefix='.rpe-install-',dir=root.parent)as temporary:
                    staged=Path(temporary)/'checkout'
                    subprocess.run(['git','clone','--shared','--no-checkout','--sparse',str(object_repository),str(staged)],
                                   check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    subprocess.run(['git','sparse-checkout','set',*specification.get('checkout_paths',
                        ['scripts','src','tools','configs','schemas','tests'])],cwd=staged,
                        check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    subprocess.run(['git','checkout','--detach',source],cwd=staged,check=True,
                                   stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    _inspect_component(specification,staged)
                    try:
                        staged.rename(root)
                        created=True
                    except OSError:
                        if not root.is_dir():raise
            item.update(observed_head=_inspect_component(specification,root),
                        status='INSTALLED'if created else'REUSED')
        except (OSError,ValueError,subprocess.CalledProcessError)as error:
            item['reason']=str(error);report['holds'].append(dict(component=name,reason=str(error)))
        report['components'][name]=item
    runtime_cache={}
    for name,specification in layout.get('runtimes',{}).items():
        if runtimes is not None and name not in runtimes:continue
        required=specification.get('components',[])
        if required and not set(required)&set(selected):continue
        try:
            key=fingerprint({k:v for k,v in specification.items()if k!='components'})
            if key not in runtime_cache:runtime_cache[key]=_ensure_runtime(specification,install)
            report['runtimes'][name]=dict(status='REUSED',python=specification['python'],**runtime_cache[key])
        except (OSError,ValueError,subprocess.CalledProcessError)as error:
            report['runtimes'][name]=dict(status='HOLD',reason=str(error))
            report['holds'].append(dict(runtime=name,reason=str(error)))
    for name,specification in layout.get('input_files',{}).items():
        required=specification.get('components',[])
        if required and not set(required)&set(selected):continue
        try:
            path=Path(specification['path']);created=False
            if not path.is_absolute():raise ValueError('Input path must be absolute')
            if not path.exists() and install and specification.get('copy_from'):
                source=Path(specification['copy_from'])
                if hashlib.sha256(source.read_bytes()).hexdigest()!=specification['sha256']:
                    raise ValueError('Saved input source hash differs: '+name)
                path.parent.mkdir(parents=True,exist_ok=True)
                with tempfile.TemporaryDirectory(prefix='.rpe-input-',dir=path.parent) as temporary:
                    staged=Path(temporary)/'input'
                    shutil.copyfile(source,staged)
                    if hashlib.sha256(staged.read_bytes()).hexdigest()!=specification['sha256']:
                        raise ValueError('Saved input changed during copy: '+name)
                    try:
                        os.link(staged,path)
                        created=True
                    except FileExistsError:
                        pass
            if hashlib.sha256(path.read_bytes()).hexdigest()!=specification['sha256']:
                raise ValueError('Installed input hash differs: '+name)
            report['inputs'][name]=dict(status='INSTALLED'if created else'REUSED',path=str(path),sha256=specification['sha256'])
        except (OSError,ValueError)as error:
            report['inputs'][name]=dict(status='HOLD',reason=str(error))
            report['holds'].append(dict(input=name,reason=str(error)))
    if report['holds']:report['status']='HOLD'
    return report


def ensure_assignment_components(manifest, host_layout, assignment, profile, *, install=False):
    """Prepare the fixed stage and the registered assets of its selected host profile."""
    from .rpe_q3_worker import select_host_profile

    reference=manifest_reference(manifest)
    if assignment.get('required_release') != reference:
        raise ValueError('Assigned release/component manifest differs')
    preparation=assignment.get('request',{}).get('preparation')
    stage=preparation['stage']if preparation else'q3'
    source=(preparation['source']if preparation else assignment['source_manifest'])['commit']
    host=host_layout['host_id']
    if profile.get('host_id') != host:
        raise ValueError('Worker profile and deployment host differ')
    components=manifest['components']
    roots=host_layout['component_roots']
    def applies(name):
        return not components[name].get('hosts') or host in components[name]['hosts']
    allowed=manifest.get('stage_components',{}).get(stage,[])
    matches=[name for name in allowed if applies(name) and components[name]['source']==source]
    if not matches:raise ValueError('Task SOURCE is not a selected component for '+stage)
    selected=select_host_profile(profile,assignment)
    request=assignment['request'];route=request['route']
    if preparation:
        view_commits={source}
        for key in ('required_views','workspaces'):
            view_commits.update(view['source']['commit'] for view in preparation.get(key,{}).values())
        views=selected.get('preparation_sources',{})
        adapter=selected.get('preparation_adapter_manifest')
        adapter_root=selected.get('preparation_adapter_root')
    else:
        view_commits={route.get('view_commit'), (route.get('horizon_risk_source') or {}).get('commit')}
        view_commits.update(row['view_commit'] for row in selected.get('runtime_checks', ()))
        views=selected.get('route_views',{})
        adapter=selected.get('adapter_manifest')
        adapter_root=selected.get('adapter_root')
    bindings=set()
    for commit in view_commits:
        path=views.get(commit)
        if isinstance(path,dict):path=path.get('root')
        if path:bindings.add((commit, str(Path(path).resolve())))
    if adapter and adapter_root:
        bindings.add((adapter['commit'], str(Path(adapter_root).resolve())))
    matches.extend(name for name in components if name not in matches and applies(name) and roots.get(name)
                   and (components[name]['source'], str(Path(roots[name]).resolve())) in bindings)
    contracts=manifest.get('runtime_contracts',{}).get(host)
    runtimes=None
    if contracts is not None:
        criteria=(dict(provider=preparation.get('provider'), source=source) if preparation else
                  dict(route_kind=route['kind'], view_commit=route.get('view_commit'),
                       horizon_risk_commit=(route.get('horizon_risk_source') or {}).get('commit')))
        runtimes={row['runtime'] for row in contracts if row['stage']==stage
                  and all(row.get(key)==value for key,value in criteria.items())}
    inspected=ensure_components(manifest,host_layout,install=install,components=matches,runtimes=runtimes)
    if inspected['status']!='PASS':raise ValueError('Required task component unavailable: '+json.dumps(inspected['holds']))
    return dict(inspected, stage=stage, scientific_source=source, matched_components=matches)


def validate_assignment_release(manifest,host_layout,assignment,profile,*,input_hash_cache=None,
                                request_validator=None):
    """Check the operational envelope, then reuse scientific worker guards."""
    inspected=ensure_assignment_components(manifest,host_layout,assignment,profile)
    from .rpe_q3_worker import validate_assignment,select_host_profile
    validator = request_validator or validate_assignment
    if input_hash_cache is None:
        validator(profile, assignment)
    else:
        validator(profile, assignment, input_hash_cache=input_hash_cache)
    runtime=None
    if inspected['stage']=='q3':
        from .rpe_host_profiles import validate_adapter_profile,check_runtime_profile
        selected=select_host_profile(profile,assignment)
        validate_adapter_profile(selected)
        runtime=check_runtime_profile(selected)
        if runtime['status']!='PASS':raise ValueError('Assigned runtime/AST contract differs')
    return dict(status='PASS',required_release=manifest_reference(manifest),stage=inspected['stage'],
                scientific_source=inspected['scientific_source'],matched_components=inspected['matched_components'],
                runtime=runtime,scientific_calls=0)
