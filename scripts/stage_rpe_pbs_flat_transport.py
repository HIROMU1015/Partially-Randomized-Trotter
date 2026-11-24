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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('sources', nargs='+', type=Path)
    options = parser.parse_args()
    print(json.dumps(stage(options.base, options.destination, options.sources), sort_keys=True))
