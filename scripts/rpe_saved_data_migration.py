#!/usr/bin/env python3
"""Prepare verified storage copies and separate execution/evidence views.

No source path, ref, active service or registry is changed. Adoption/removal of
an original is a separate operation, not a side effect of preparing a copy.
"""
import argparse
import hashlib
import json
import lzma
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile

from compact_rpe_git_objects import git_environment
from rpe_romutail_cold_storage import sha256


def git(root, *args, **kwargs):
    return subprocess.check_output(['git','-C',str(root),*args], env=git_environment(), **kwargs)


def new_destination(source, destination):
    source = Path(source).resolve(strict=True)
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise ValueError('Migration destination already exists')
    destination = destination.resolve()
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Migration destination overlaps source')
    destination.parent.mkdir(parents=True, exist_ok=True)
    return source, destination


def inventory(root, *, skip=None):
    """Hash regular files, preserve link text and mode; never follow symlinks."""
    root = Path(root)
    def walk(directory):
        for path in sorted(directory.iterdir()):
            name=path.relative_to(root).as_posix()
            if name == skip:
                continue
            info=path.lstat(); mode=stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                yield name, ('link',os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                yield name, ('dir',mode)
                yield from walk(path)
            elif stat.S_ISREG(info.st_mode):
                yield name, ('file',mode,info.st_size,sha256(path))
            else:
                raise ValueError('Non-regular migration input: '+name)
    return dict(walk(root))


def copy_saved_tree(source, destination):
    """Copy a settled trial/evidence directory to an independently checked copy."""
    source,destination=new_destination(source,destination)
    before=inventory(source)
    with tempfile.TemporaryDirectory(prefix='.rpe-migrate-',dir=destination.parent) as tmp:
        staged=Path(tmp)/'copy'
        shutil.copytree(source,staged,symlinks=True)
        if inventory(staged)!=before or inventory(source)!=before:
            raise ValueError('Saved directory changed or copied bytes differ')
        if destination.exists() or destination.is_symlink():
            raise ValueError('Migration destination appeared')
        staged.rename(destination)
    external_links=[name for name,row in before.items() if row[0]=='link' and
                    (Path(row[1]).is_absolute() or not (source/name).resolve().is_relative_to(source))]
    alternates=[name for name in before if name.endswith('/objects/info/alternates')]
    return dict(status='VERIFIED_COPY',entries=len(before),destination=str(destination),source_preserved=True,
                external_links=external_links,git_alternates=alternates,
                independent=not external_links and not alternates)


def compressed_digest(path):
    with lzma.open(path,'rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def copy_compressed_file(source,destination):
    """Cold-copy raw result bytes without parsing or normalizing JSON."""
    source,destination=new_destination(source,destination)
    if not source.is_file():raise ValueError('Raw result is not a regular file')
    expected=sha256(source)
    with tempfile.TemporaryDirectory(prefix='.rpe-raw-archive-',dir=destination.parent) as tmp:
        staged=Path(tmp)/'raw.xz'
        with source.open('rb') as raw,lzma.open(staged,'wb') as packed:
            shutil.copyfileobj(raw,packed,1024**2)
        if compressed_digest(staged)!=expected or sha256(source)!=expected:raise ValueError('Raw result changed or archive differs')
        os.link(staged,destination)
    return dict(status='VERIFIED_COMPRESSED_COPY',sha256=expected,source_preserved=True,destination=str(destination))


def restore_compressed_file(archive,destination,expected_sha):
    archive,destination=new_destination(archive,destination)
    with tempfile.TemporaryDirectory(prefix='.rpe-raw-restore-',dir=destination.parent) as tmp:
        staged=Path(tmp)/'raw'
        with lzma.open(archive,'rb') as packed,staged.open('wb') as raw:
            shutil.copyfileobj(packed,raw,1024**2)
        if sha256(staged)!=expected_sha:raise ValueError('Restored raw result digest differs')
        os.link(staged,destination)
    return dict(status='VERIFIED_RESTORATION',sha256=expected_sha,destination=str(destination))


def object_list(repo, output):
    """Include unreachable objects: frozen Git locators need not be named refs."""
    with Path(output).open('wb') as stream:
        subprocess.run(['git','-C',str(repo),'cat-file','--batch-all-objects','--batch-check=%(objectname)'],
                       env=git_environment(),stdout=stream,check=True)
    return sha256(Path(output))


def materialize_pack(source, objects, ids):
    with ids.open('rb') as stream:
        subprocess.run(['git','-C',str(source),'pack-objects',str(objects/'pack/pack')],
                       env=git_environment(),stdin=stream,check=True,capture_output=True)


def stage_git_object_store(source, destination):
    """Create an independent bare store from a bare, normal or linked view."""
    source,destination=new_destination(source,destination)
    with tempfile.TemporaryDirectory(prefix='.rpe-object-store-',dir=destination.parent) as tmp:
        tmp=Path(tmp);staged=tmp/'store';ids=tmp/'objects.txt';after=tmp/'after.txt'
        expected=object_list(source,ids)
        if not ids.stat().st_size:raise ValueError('No Git objects to migrate')
        algorithm=git(source,'rev-parse','--show-object-format',text=True).strip()
        subprocess.run(['git','init','--bare','--object-format='+algorithm,str(staged)],
                       env=git_environment(),check=True,capture_output=True)
        materialize_pack(source,staged/'objects',ids)
        if object_list(staged,after)!=expected or object_list(source,after)!=expected:
            raise ValueError('Git object inventory changed')
        git(staged,'fsck','--full','--no-dangling',stderr=subprocess.PIPE)
        if destination.exists() or destination.is_symlink():raise ValueError('Migration destination appeared')
        staged.rename(destination)
    return dict(status='VERIFIED_OBJECT_STORE',source_preserved=True,objects=str(destination/'objects'),
                object_inventory_sha256=expected)


def migrate_git_repository(source, destination, *, target_objects=None):
    """Prepare an isolated clone with no dependency on its original alternates.

    Copy everything except the object directory, materialize *all* visible
    objects, or use a target that supplies them all. Preserve refs, index,
    reflogs, tracked/untracked files and exact historical object IDs.
    """
    source,destination=new_destination(source,destination)
    git_dir=Path(git(source,'rev-parse','--absolute-git-dir',text=True).strip())
    if git_dir!=source/'.git' or git_dir.is_symlink() or (git_dir/'commondir').exists():
        raise ValueError('Migration requires a standalone non-bare checkout')
    if (git_dir/'objects').is_symlink() or (git_dir/'worktrees').exists():
        raise ValueError('Migration does not copy linked Git worktrees')
    if subprocess.run(['git','-C',str(source),'config','--get','core.worktree'],env=git_environment(),capture_output=True).returncode==0:
        raise ValueError('External Git worktree configuration')
    before=inventory(source,skip='.git/objects')
    with tempfile.TemporaryDirectory(prefix='.rpe-git-migrate-',dir=destination.parent) as tmp:
        tmp=Path(tmp);staged=tmp/'checkout';ids=tmp/'objects.txt';after_ids=tmp/'after.txt'
        expected=object_list(source,ids)
        if not ids.stat().st_size:
            raise ValueError('No Git objects to migrate')
        def ignore(directory,names):
            return ['objects'] if Path(directory)==git_dir else []
        shutil.copytree(source,staged,symlinks=True,ignore=ignore)
        if inventory(staged)!=before or inventory(source,skip='.git/objects')!=before:
            raise ValueError('Git checkout changed during migration')
        objects=staged/'.git/objects';(objects/'pack').mkdir(parents=True);(objects/'info').mkdir()
        if target_objects is None:
            materialize_pack(source,objects,ids)
        else:
            target=Path(target_objects).resolve(strict=True)
            if (not target.is_dir() or (target/'info/alternates').exists()
                    or target.is_relative_to(source) or target.is_relative_to(destination)):
                raise ValueError('Target object store is not independent of the source')
            (objects/'info/alternates').write_text(str(target)+'\n')
            # A shared target may have additional objects. Check every source
            # object via the new view rather than require an identical universe.
            missing=0;checked=0
            with ids.open('rb') as stream, subprocess.Popen(
                    ['git','-C',str(staged),'cat-file','--batch-check'],env=git_environment(),
                    stdin=stream,stdout=subprocess.PIPE) as child:
                for row in child.stdout:
                    checked+=1
                    if len(row.split())!=3 or not row.split()[-1].isdigit():
                        missing+=1
                if child.wait()!=0:
                    raise ValueError('Target object check failed')
            with ids.open('rb') as stream:expected_count=sum(1 for _ in stream)
            if missing or checked!=expected_count:raise ValueError('Target object store lacks a source object')
        if target_objects is None and object_list(staged,after_ids)!=expected:
            raise ValueError('Independent Git object inventory differs')
        git(staged,'fsck','--full','--no-dangling',stderr=subprocess.PIPE)
        if object_list(source,after_ids)!=expected or inventory(source,skip='.git/objects')!=before:
            raise ValueError('Original Git repository changed during migration')
        if destination.exists() or destination.is_symlink():
            raise ValueError('Migration destination appeared')
        staged.rename(destination)
    return dict(status='VERIFIED_GIT_COPY',source_preserved=True,destination=str(destination),object_inventory_sha256=expected)


def prepare_evidence_service(config, object_repository, checkout, *, checkout_paths=None):
    """Create a separate fixed controller Git view for *new* runs only.

    The execution view and its search-evidence directory stay together so
    existing Git locator generation needs no changes. The release registry
    and original service are not modified. Register a new service explicitly.
    """
    from copy import deepcopy
    from trottertracks.rpe_local0_search import fixed_implementation
    from rpe_romutail_retention import alternate_relocation_readiness
    release, checkout=new_destination(config['code_root'],checkout)
    repository=Path(object_repository).resolve(strict=True)
    object_root=Path(git(repository,'rev-parse','--path-format=absolute','--git-path','objects',text=True).strip()).resolve(strict=True)
    if repository.is_relative_to(release) or object_root.is_relative_to(release) or (object_root/'info/alternates').exists():
        raise ValueError('Evidence view must not depend on the retired release object directory')
    fixed_implementation(config)
    if alternate_relocation_readiness(release,object_root)['status']!='OBJECTS_AVAILABLE':
        raise ValueError('Independent execution view lacks fixed release objects')
    source=config['code_source']
    if not isinstance(source,str) or len(source)!=40 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('Full fixed implementation SOURCE required')
    # Preserve the old checkout's tracked-file scope. A full old checkout is
    # not silently reduced; explicit paths allow a deliberately scoped view.
    paths=checkout_paths
    sparse_mode='--cone'
    if paths is None:
        sparse=subprocess.run(['git','-C',str(release),'config','--bool','core.sparseCheckout'],env=git_environment(),capture_output=True,text=True)
        if sparse.returncode==0 and sparse.stdout.strip()=='true':
            cone=subprocess.run(['git','-C',str(release),'config','--bool','core.sparseCheckoutCone'],env=git_environment(),capture_output=True,text=True)
            sparse_mode='--cone' if cone.stdout.strip()=='true' else '--no-cone'
            paths=git(release,'sparse-checkout','list',text=True).splitlines()
    selected=deepcopy(config)
    with tempfile.TemporaryDirectory(prefix='.rpe-evidence-view-',dir=checkout.parent) as tmp:
        staged=Path(tmp)/'checkout'
        subprocess.run(['git','clone','--shared','--no-checkout',str(repository),str(staged)],env=dict(git_environment(),GIT_ALLOW_PROTOCOL='file'),check=True,capture_output=True)
        if paths is not None:
            git(staged,'sparse-checkout','set',sparse_mode,'--stdin',input='\n'.join(paths)+'\n',text=True)
        git(staged,'checkout','--detach',source,stderr=subprocess.PIPE)
        fixed_implementation(dict(code_root=str(staged),code_source=source))
        # Fixed service files stored beside the old release can be relocated
        # byte-for-byte; do not modify arbitrary provider/runtime declarations.
        def relocate(value):
            path=Path(value)
            if not path.is_absolute() or not path.resolve().is_relative_to(release):return value
            original=path.resolve(strict=True)
            if not original.is_file():raise ValueError('Fixed service input is not a file')
            relative=Path('service-inputs')/original.relative_to(release)
            target=staged/relative;target.parent.mkdir(parents=True,exist_ok=True)
            expected=sha256(original)
            shutil.copy2(original,target)
            if sha256(target)!=expected or sha256(original)!=expected:raise ValueError('Fixed service input changed')
            return str(checkout/relative)
        for key in ('problem','round','pbs_walltime_policy'):
            if selected.get(key):selected[key]=relocate(selected[key])
        for key in ('problem','round'):
            if selected.get('absolute_kappa_route',{}).get(key):
                selected['absolute_kappa_route'][key]=relocate(selected['absolute_kappa_route'][key])
        if selected.get('fixed_settings'):
            selected['fixed_settings']={relocate(path):value for path,value in selected['fixed_settings'].items()}
        if checkout.exists() or checkout.is_symlink():raise ValueError('Evidence checkout appeared')
        staged.rename(checkout)
    selected.update(code_root=str(checkout),run_evidence_root=str(checkout/'search-evidence'))
    # A service path is assigned by the caller when registering the NEW view.
    selected.pop('config_path',None)
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    for name in ('copy-tree','copy-git','stage-object-store','compress-file','restore-file'):
        command=sub.add_parser(name);command.add_argument('source',type=Path);command.add_argument('destination',type=Path)
        if name=='copy-git':command.add_argument('--target-objects',type=Path)
        if name=='restore-file':command.add_argument('--sha256',required=True)
    view=sub.add_parser('evidence-service');view.add_argument('config',type=Path);view.add_argument('repository',type=Path);view.add_argument('checkout',type=Path);view.add_argument('--checkout-path',action='append')
    args=parser.parse_args()
    if args.action=='copy-tree':result=copy_saved_tree(args.source,args.destination)
    elif args.action=='copy-git':result=migrate_git_repository(args.source,args.destination,target_objects=args.target_objects)
    elif args.action=='stage-object-store':result=stage_git_object_store(args.source,args.destination)
    elif args.action=='compress-file':result=copy_compressed_file(args.source,args.destination)
    elif args.action=='restore-file':result=restore_compressed_file(args.source,args.destination,args.sha256)
    else:result=prepare_evidence_service(json.loads(args.config.read_bytes()),args.repository,args.checkout,checkout_paths=args.checkout_path)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
