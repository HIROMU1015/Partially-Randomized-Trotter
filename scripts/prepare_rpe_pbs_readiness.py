#!/usr/bin/env python3
"""Package exact existing source views and saved inputs for PBS metadata checks.

The sparse object pack retains original commit/tree/blob identities. It omits
unrelated history and result blobs; no scientific request or SOURCE is rewritten.
Only operational paths in the host profile are mapped to the new private host.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT.parents[1] if ROOT.parent.name == '.worktrees' else ROOT
REMOTE = '/home/AbeHiromu/projects/rpe-pbs-20261118/readiness'


def git(*args, binary=False):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=not binary)


def read(path):
    return json.loads(Path(path).read_text())


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def prepare(output, profile_path, python, remote_root=REMOTE, extra_locators=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    profile = read(profile_path)
    tasks = read(MAIN / '.preparation-distributed/f956f5197a5a/metadata-dry/payload/TASKS.json')['tasks']
    tasks += read(MAIN / '.rpe-central/deployment/d887abdca-r2-inputs/TASKS_PR_DF_INTEGRATED.json')['tasks']
    assignments = {method: read(ROOT / 'docs/research/rpe_q3_auto_allocation/deployment_20261117/assignments' / (method + '.json'))
                   for method in ('FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF')}
    roots = {}
    for value in walk(profile):
        for key in ('root', 'adapter_root', 'preparation_adapter_root'):
            if isinstance(value.get(key), str):
                roots[value[key]] = None
        for key in ('route_views', 'preparation_sources'):
            roots.update({path: None for path in value.get(key, {}).values()})
    for path in roots:
        roots[path] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip()
    # Fixed checker transport is part of the package rather than a dirty tree.
    checker_source = git('rev-parse', '14a6102d4^{commit}').strip()
    commits = set(roots.values()) | {checker_source}
    mappings = {old: remote_root + '/views/' + sha for old, sha in roots.items()}
    runtime_files = []
    for value in walk(profile):
        for alias, path in value.get('preparation_paths', {}).items():
            path = path['path'] if isinstance(path, dict) else path
            if path in mappings:
                continue
            source = Path(path)
            destination = remote_root + '/runtime-inputs/' + alias
            if source.is_file():
                destination += '/' + source.name
            mappings[path] = destination
            runtime_files.append((source, destination))

    def remap(value):
        if isinstance(value, dict):
            mapped = {key: remap(item) for key, item in value.items()}
            if 'host_id' in mapped:
                mapped['host_id'] = 'pbs-0'
            if 'interpreter' in mapped:
                mapped['interpreter'] = python
            if 'work_dir' in mapped:
                mapped['work_dir'] = remote_root + '/inspection-spool'
            if 'prefix_cache_directory' in mapped:
                mapped['prefix_cache_directory'] = remote_root + '/cache/dp'
            if 'artifact_cache_directory' in mapped:
                mapped['artifact_cache_directory'] = remote_root + '/cache/artifacts'
            return mapped
        if isinstance(value, list):
            return [remap(item) for item in value]
        if isinstance(value, str):
            return mappings.get(value, value)
        return value

    mapped = remap(profile)
    put(output / 'payload/PROFILE.json', mapped)
    put(output / 'payload/TASKS.json', dict(tasks=tasks, scientific_calls=0, compile_calls=0))
    locators = {}
    for method, assignment in assignments.items():
        put(output / 'payload/assignments' / (method + '.json'), assignment)
    for value in walk([tasks, list(assignments.values())]):
        if isinstance(value.get('commit'), str) and isinstance(value.get('path'), str):
            locators[(value['commit'], value['path'])] = value.get('sha256')
    for locator in extra_locators or []:
        locators[(locator['commit'], locator['path'])] = locator.get('sha256')
    objects = set()
    for commit in sorted(commits):
        objects.add(commit)
        objects.add(git('rev-parse', commit + '^{tree}').strip())
        listing = git('ls-tree', '-r', '-t', commit)
        for line in listing.splitlines():
            meta, path = line.split('\t', 1)
            _, kind, oid = meta.split()
            if kind == 'tree' or path.split('/')[0] in ('scripts', 'src', 'configs', 'schemas'):
                objects.add(oid)
    for (commit, path), expected in locators.items():
        objects.add(commit)
        objects.add(git('rev-parse', commit + '^{tree}').strip())
        blob = git('show', commit + ':' + path, binary=True)
        if expected and hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError('Saved locator hash differs: ' + commit + ':' + path)
        objects.add(git('rev-parse', commit + ':' + path).strip())
        for parent in Path(path).parents:
            if str(parent) != '.':
                objects.add(git('rev-parse', commit + ':' + parent.as_posix()).strip())
    pack = output / 'sources.pack'
    with pack.open('wb') as stream:
        process = subprocess.run(['git', 'pack-objects', '--stdout'], cwd=ROOT,
                                 input=('\n'.join(sorted(objects)) + '\n').encode(), stdout=stream,
                                 stderr=subprocess.PIPE, check=True)
    setup = '''import json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);data=json.loads((root/'PAYLOAD.json').read_text());repo=root/'objects.git'
def run(*args):subprocess.run(list(args),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if not repo.exists():run('git','init','--bare',str(repo))
pack=root/'sources.pack'
with pack.open('rb')as stream:subprocess.run(['git','--git-dir='+str(repo),'index-pack','--stdin'],stdin=stream,check=True,stdout=subprocess.PIPE)
(repo/'shallow').write_text('\\n'.join(data['all_commits'])+'\\n')
run('git','--git-dir='+str(repo),'config','extensions.worktreeConfig','true')
for sha in data['source_commits']:
 view=root/'views'/sha
 if view.exists():
  current=subprocess.check_output(['git','rev-parse','HEAD'],cwd=view,universal_newlines=True).strip()
  if current!=sha:raise ValueError('Existing immutable view differs')
  continue
 view.parent.mkdir(parents=True,exist_ok=True)
 run('git','--git-dir='+str(repo),'worktree','add','--detach','--no-checkout',str(view),sha)
 run('git','-C',str(view),'config','--worktree','core.bare','false')
 run('git','-C',str(view),'sparse-checkout','set','--no-cone','/scripts/','/src/','/configs/','/schemas/')
 run('git','-C',str(view),'checkout','--detach',sha)
print(json.dumps({'status':'SOURCES_PLACED','sources':len(data['source_commits'])}))
'''
    (output / 'setup.py').write_text(setup)
    manifest = dict(schema='RPE_PBS_SAVED_INPUT_PACKAGE_V1', source_commits=sorted(commits),
                    all_commits=sorted(commits | {commit for commit, _ in locators}),
                    checker_source=checker_source, runtime_python=python,
                    remote_root=remote_root, profile_source=str(profile_path),
                    source_object_count=len(objects), source_pack_bytes=pack.stat().st_size,
                    source_pack_sha256=hashlib.sha256(pack.read_bytes()).hexdigest(),
                    input_locator_count=len(locators), q3_methods=list(assignments),
                    preparation_task_count=len(tasks), path_map=mappings,
                    scientific_calls=0, compile_calls=0, build_seconds=time.monotonic()-started)
    put(output / 'PAYLOAD.json', manifest)
    archive = output / 'readiness.tar.gz'
    with tarfile.open(archive, 'w:gz') as target:
        for name in ('payload', 'sources.pack', 'setup.py', 'PAYLOAD.json'):
            target.add(output / name, arcname=name)
        for source, destination in runtime_files:
            relative = Path(destination).relative_to(remote_root)
            target.add(source, arcname=str(relative))
    put(output / 'PACKAGE.json', dict(manifest, archive_bytes=archive.stat().st_size,
                                     archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    return manifest


def stage_commit(commit, destination, alias, output, remote_root=REMOTE, extra_locators=None):
    """Add one exact fixed transport to the private object store, incrementally.

    This is technical code placement only. Existing immutable deployments are
    checked and reused, and never switched to another commit.
    """
    started = time.monotonic()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = git('rev-parse', commit + '^{commit}').strip()
    if not Path(destination).is_absolute() or not str(destination).startswith(str(Path(remote_root).parent) + '/'):
        raise ValueError('Destination must be inside the dedicated private PBS project')
    objects = {commit, git('rev-parse', commit + '^{tree}').strip()}
    for line in git('ls-tree', '-r', '-t', commit).splitlines():
        meta, path = line.split('\t', 1)
        _, kind, oid = meta.split()
        if kind == 'tree' or path.split('/')[0] in ('scripts', 'src', 'configs', 'schemas'):
            objects.add(oid)
    extra_commits = set()
    for locator in extra_locators or []:
        revision = git('rev-parse', locator['commit'] + '^{commit}').strip()
        path = locator['path']
        raw = git('show', revision + ':' + path, binary=True)
        if locator.get('sha256') and hashlib.sha256(raw).hexdigest() != locator['sha256']:
            raise ValueError('Supplemental fixed locator differs: ' + path)
        objects.update((revision, git('rev-parse', revision+'^{tree}').strip(),
                        git('rev-parse', revision+':'+path).strip()))
        extra_commits.add(revision)
        for parent in Path(path).parents:
            if str(parent) != '.':
                objects.add(git('rev-parse', revision+':'+parent.as_posix()).strip())
    pack = output / ('transport-' + commit[:12] + '.pack')
    # Ask the existing object store which exact object IDs are absent. A reuse
    # check transfers only this short list and creates no PBS job or worker.
    repo = remote_root + '/objects.git'
    query = "import json,subprocess,sys; p=subprocess.run(['git','--git-dir=' + sys.argv[1],'cat-file','--batch-check=%(objectname) %(objecttype)'],input=sys.stdin.read(),universal_newlines=True,stdout=subprocess.PIPE,check=True); print(json.dumps([line.split()[0] for line in p.stdout.splitlines() if line.endswith(' missing')]))"
    command = ['ssh', '-T', '-o', 'BatchMode=yes', alias,
               ' '.join(shlex.quote(arg) for arg in ['python3', '-c', query, repo])]
    result = subprocess.run(command, input='\n'.join(sorted(objects))+'\n', text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    missing = json.loads(result.stdout)
    with pack.open('wb') as stream:
        if missing:
            subprocess.run(['git', 'pack-objects', '--stdout'], cwd=ROOT,
                           input=('\n'.join(missing)+'\n').encode(), stdout=stream, check=True)
    remote_pack = remote_root + '/' + pack.name
    if missing:
        subprocess.run(['scp', '-q', str(pack), alias + ':' + remote_pack], check=True)
    install = '''import hashlib,json,subprocess,sys,time
from pathlib import Path
repo,sha,target,pack,expected,extra=sys.argv[1:]
repo=Path(repo);target=Path(target);started=time.monotonic()
def run(*args):subprocess.run(list(args),check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
if pack:
 path=Path(pack)
 if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Transport pack hash differs')
 with path.open('rb')as stream:subprocess.run(['git','--git-dir='+str(repo),'index-pack','--stdin'],stdin=stream,check=True,stdout=subprocess.PIPE)
shallow=repo/'shallow';previous=shallow.read_text().splitlines()if shallow.exists()else[]
shallow.write_text('\\n'.join(sorted(set(previous+[sha]+json.loads(extra))))+'\\n')
if target.exists():
 actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,universal_newlines=True).strip()
 if actual!=sha:raise ValueError('Existing immutable transport differs')
 reused=True
else:
 target.parent.mkdir(parents=True,exist_ok=True)
 run('git','--git-dir='+str(repo),'config','extensions.worktreeConfig','true')
 run('git','--git-dir='+str(repo),'worktree','add','--detach','--no-checkout',str(target),sha)
 run('git','-C',str(target),'config','--worktree','core.bare','false')
 run('git','-C',str(target),'sparse-checkout','set','--no-cone','/scripts/','/src/','/configs/','/schemas/')
 run('git','-C',str(target),'checkout','--detach',sha)
 reused=False
run('git','-C',str(target),'diff','--exit-code','HEAD','--','scripts','src','configs','schemas')
print(json.dumps(dict(commit=sha,destination=str(target),reused=reused,remote_same_clock_seconds=time.monotonic()-started)))
'''
    args = ['python3', '-c', install, repo, commit, destination,
            remote_pack if missing else '', hashlib.sha256(pack.read_bytes()).hexdigest(),
            json.dumps(sorted(extra_commits))]
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', alias,
                             ' '.join(shlex.quote(arg) for arg in args)],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    value = dict(json.loads(result.stdout), missing_object_count=len(missing),
                 transferred_bytes=pack.stat().st_size if missing else 0,
                 local_same_clock_seconds=time.monotonic()-started)
    put(output / ('STAGE_' + commit[:12] + '_' + str(time.time_ns()) + '.json'), value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=MAIN / '.rpe-central/deployment/price-validation-14a6102d4-r3/profiles/local.json')
    parser.add_argument('--python', default='/home/AbeHiromu/projects/rpe-pbs-20261118/runtime/venv311rc1/bin/python')
    parser.add_argument('--remote-root', default=REMOTE)
    parser.add_argument('--stage-commit', help='Incrementally place one new fixed transport instead of preparing saved inputs')
    parser.add_argument('--destination', help='Absolute private cluster transport path for --stage-commit')
    parser.add_argument('--ssh-alias', default='prism')
    parser.add_argument('--locators', type=Path, help='Additional exact historical Git blobs required by source metadata guards')
    args = parser.parse_args()
    default_locators = ROOT / 'docs/research/rpe_pbs_worker_20261118/readiness/HISTORICAL_LOCATORS.json'
    locators = read(args.locators or default_locators) if (args.locators or default_locators.exists()) else []
    if args.stage_commit:
        if not args.destination:
            parser.error('--destination is required with --stage-commit')
        print(json.dumps(stage_commit(args.stage_commit, args.destination, args.ssh_alias, args.output, args.remote_root, locators)))
        return
    value = prepare(args.output, args.profile, args.python, args.remote_root, locators)
    print(json.dumps({key:value[key] for key in ('source_pack_bytes','source_object_count','input_locator_count','preparation_task_count','build_seconds')}))


if __name__ == '__main__':
    main()
