"""Place a selected release's components without changing scientific ownership.

The central release registry selects the version. This module only reuses or
installs its immutable assets and runs the worker's existing metadata guards.
Installed assets never change a host's participation/Pause setting.
"""
from pathlib import Path
import hashlib
import json
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
        subprocess.run(['git','clone','--bare','--shared',str(layout['seed_repository']),
                        str(repository)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    return repository


def _ensure_source(repository, source, layout, install):
    if not repository.exists():raise ValueError('SOURCE_REPOSITORY_MISSING: '+str(repository))
    if _source_present(repository, source):return
    if install:
        for bundle in layout.get('bundle_files',[]):
            if not Path(bundle).is_file():continue
            result=subprocess.run(['git','fetch','--no-tags',str(bundle),source],cwd=repository,
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


def ensure_components(manifest, host_layout, *, install=False, components=None, repository=None):
    """Reuse verified assets; acquire only missing roots/objects/input files.

    Code comes from the host's existing object repository or explicit local Git
    bundles. Dependency shortages are reported for a task-specific environment;
    no shared Python or global package installation is changed here.
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
                    staged.rename(root)
                created=True
            item.update(observed_head=_inspect_component(specification,root),
                        status='INSTALLED'if created else'REUSED')
        except (OSError,ValueError,subprocess.CalledProcessError)as error:
            item['reason']=str(error);report['holds'].append(dict(component=name,reason=str(error)))
        report['components'][name]=item
    runtime_cache={}
    for name,specification in layout.get('runtimes',{}).items():
        required=specification.get('components',[])
        if required and not set(required)&set(selected):continue
        try:
            key=fingerprint({k:v for k,v in specification.items()if k!='components'})
            if key not in runtime_cache:runtime_cache[key]=_runtime(specification)
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
                path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path);created=True
            if hashlib.sha256(path.read_bytes()).hexdigest()!=specification['sha256']:
                raise ValueError('Installed input hash differs: '+name)
            report['inputs'][name]=dict(status='INSTALLED'if created else'REUSED',path=str(path),sha256=specification['sha256'])
        except (OSError,ValueError)as error:
            report['inputs'][name]=dict(status='HOLD',reason=str(error))
            report['holds'].append(dict(input=name,reason=str(error)))
    if report['holds']:report['status']='HOLD'
    return report


def validate_assignment_release(manifest,host_layout,assignment,profile,*,input_hash_cache=None,
                                request_validator=None):
    """Check the operational envelope, then reuse scientific worker guards."""
    reference=manifest_reference(manifest)
    if assignment.get('required_release') != reference:
        raise ValueError('Assigned release/component manifest differs')
    preparation=assignment.get('request',{}).get('preparation')
    stage=preparation['stage']if preparation else'q3'
    source=(preparation['source']if preparation else assignment['source_manifest'])['commit']
    allowed=manifest.get('stage_components',{}).get(stage,[])
    matches=[name for name in allowed if manifest['components'][name]['source']==source]
    if not matches:raise ValueError('Task SOURCE is not a selected component for '+stage)
    inspected=ensure_components(manifest,host_layout,components=matches)
    if inspected['status']!='PASS':raise ValueError('Required task component unavailable: '+json.dumps(inspected['holds']))
    from .rpe_q3_worker import validate_assignment,select_host_profile
    (request_validator or validate_assignment)(profile,assignment,input_hash_cache=input_hash_cache)
    runtime=None
    if not preparation:
        from .rpe_host_profiles import validate_adapter_profile,check_runtime_profile
        selected=select_host_profile(profile,assignment)
        validate_adapter_profile(selected)
        runtime=check_runtime_profile(selected)
        if runtime['status']!='PASS':raise ValueError('Assigned runtime/AST contract differs')
    return dict(status='PASS',required_release=reference,stage=stage,scientific_source=source,
                matched_components=matches,runtime=runtime,scientific_calls=0)
