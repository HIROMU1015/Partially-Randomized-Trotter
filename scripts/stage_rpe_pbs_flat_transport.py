#!/usr/bin/env python3
"""Stage private fixed PBS checkouts against one independent Git object store."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def git(root, *args):
    result = subprocess.run(['git', *args], cwd=str(root), check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return result.stdout.strip()


def object_directories(root, base, seen, ancestors):
    if root in ancestors or base not in root.parents:
        raise ValueError('Invalid private Git alternate: ' + str(root))
    if root in seen:
        return
    if not root.is_dir():
        raise ValueError('Missing Git object directory: ' + str(root))
    alternate_file = root/'info'/'alternates'
    if alternate_file.is_file():
        for name in alternate_file.read_text().splitlines():
            object_directories((root/name).resolve(), base, seen, ancestors | {root})
    seen.add(root)


def link_objects(directories, target):
    for source in sorted(directories):
        for pack in (source/'pack').glob('pack-*.pack'):
            index = pack.with_suffix('.idx')
            if not index.is_file():
                raise ValueError('Missing Git pack index: ' + str(pack))
            for item in (pack, index):
                path = target/'pack'/item.name
                if not path.exists():
                    os.link(str(item), str(path))
        for item in source.glob('[0-9a-f][0-9a-f]/[0-9a-f]*'):
            path = target/item.relative_to(source)
            if item.is_file() and not path.exists():
                path.parent.mkdir(exist_ok=True)
                os.link(str(item), str(path))


def stage(base, destination, sources):
    base, destination = base.resolve(), destination.resolve()
    if base not in destination.parents or destination.exists():
        raise ValueError('Destination must be a new private PBS deployment')
    heads, stores = {}, set()
    for source in sources:
        source = source.resolve()
        common = (source/git(source, 'rev-parse', '--git-common-dir')).resolve()
        object_directories(common/'objects', base, stores, set())
        # Refuse to copy a checkout with tracked changes.
        git(source, 'diff', '--exit-code', 'HEAD', '--')
        heads[source] = (git(source, 'rev-parse', 'HEAD'), common)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.stage-transport-', dir=str(destination.parent)) as directory:
        staging = Path(directory)/'deployment'
        staging.mkdir()
        git(staging, 'init', '--bare', 'objects.git')
        store = staging/'objects.git'/'objects'
        link_objects(stores, store)
        paths, commits = {}, {}
        for index, (source, (head, common)) in enumerate(heads.items()):
            relative = Path('checkouts')/('{:02}-'.format(index)+source.name)
            checkout = staging/relative
            checkout.mkdir(parents=True)
            git(checkout, 'init')
            metadata = checkout/'.git'
            (metadata/'objects'/'info'/'alternates').write_text(
                os.path.relpath(str(store), str(metadata/'objects'))+'\n')
            shallow = common/'shallow'
            if shallow.is_file():
                shutil.copyfile(str(shallow), str(metadata/'shallow'))
            sparse = Path(git(source, 'rev-parse', '--git-path', 'info/sparse-checkout'))
            sparse = source/sparse
            if sparse.is_file() and git(source, 'config', '--bool', '--get', 'core.sparseCheckout') == 'true':
                git(checkout, 'config', 'core.sparseCheckout', 'true')
                cone = subprocess.run(['git', 'config', '--bool', '--get', 'core.sparseCheckoutCone'],
                                      cwd=str(source), stdout=subprocess.PIPE, universal_newlines=True)
                if cone.stdout.strip() == 'true':
                    git(checkout, 'config', 'core.sparseCheckoutCone', 'true')
                shutil.copyfile(str(sparse), str(metadata/'info'/'sparse-checkout'))
            git(checkout, 'checkout', '--quiet', '--detach', head)
            git(checkout, 'diff', '--exit-code', head, '--')
            paths[str(source)] = str(destination/relative)
            commits[str(source)] = head
        report = dict(paths=paths, commits=commits, stores=len(stores), alternate_depth=1,
                      packs=len(list((store/'pack').glob('*.pack'))))
        (staging/'TRANSPORT.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        staging.rename(destination)
    return report



def profile_sources(profile, base, *, minimum_depth=5):
    """Return declared clean private views with an overly deep alternate graph.

    Memoizing shared ancestors is essential: fleet views often have many
    overlapping alternate branches even when the number of stores is small.
    """
    roots, depths = set(), {}
    def walk(value):
        if isinstance(value, dict):
            for item in value.values():walk(item)
        elif isinstance(value, list):
            for item in value:walk(item)
        elif isinstance(value, str) and value.startswith(str(base)+'/'):
            path = Path(value)
            if (path/'.git').exists():roots.add(path)
    def depth(root, ancestors=frozenset()):
        if root in ancestors:raise ValueError('Cyclic Git alternate: '+str(root))
        if root in depths:return depths[root]
        path=root/'info'/'alternates'
        result=(1+max((depth((root/name).resolve(), ancestors|{root})
                       for name in path.read_text().splitlines() if name), default=-1)) if path.is_file() else 0
        depths[root]=result
        return result
    walk(profile)
    sources=[];issues={}
    for root in sorted(roots):
        try:
            common=(root/git(root,'rev-parse','--git-common-dir')).resolve()
            if depth(common/'objects')<=minimum_depth:continue
            # Check private scope and tracked identity before adding any view.
            object_directories(common/'objects',base,set(),set())
            git(root,'diff','--exit-code','HEAD','--')
            sources.append(root)
        except (OSError,ValueError,subprocess.CalledProcessError) as error:
            issues[str(root)]=str(error)
    return sources,issues


def stage_profile(base, destination, profile_path, *, minimum_depth=5):
    base=base.resolve();profile=json.loads(profile_path.read_text())
    sources,issues=profile_sources(profile,base,minimum_depth=minimum_depth)
    if not sources:return dict(paths={},issues=issues)
    report=stage(base,destination,sources)
    def relocate(value):
        if isinstance(value,dict):return {key:relocate(item) for key,item in value.items()}
        if isinstance(value,list):return [relocate(item) for item in value]
        return report['paths'].get(value,value) if isinstance(value,str) else value
    output=destination/'PROFILE.json'
    output.write_text(json.dumps(relocate(profile),sort_keys=True,indent=2)+'\n')
    return dict(report,profile=str(output),issues=issues)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--minimum-depth', type=int, default=5)
    parser.add_argument('sources', nargs='*', type=Path)
    options = parser.parse_args()
    if bool(options.profile)==bool(options.sources):parser.error('select a profile or explicit source views')
    result=(stage_profile(options.base,options.destination,options.profile,minimum_depth=options.minimum_depth)
            if options.profile else stage(options.base, options.destination, options.sources))
    print(json.dumps(result, sort_keys=True))
