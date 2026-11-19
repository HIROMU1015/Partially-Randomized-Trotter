"""Launch a fixed PF entry with its saved inputs at the recorded runtime paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def fixed_files(config):
    mounts = []
    for entry in config['inputs']:
        original = Path(entry['original'])
        saved = Path(entry['path']).resolve(strict=True)
        if not original.is_relative_to('/home/AbeHiromu/projects') or '..' in original.parts:
            raise ValueError('Invalid fixed PF input path')
        if 'sha256' in entry:
            files = {saved: entry['sha256']}
        else:
            files = {}
            for name, digest in entry['files'].items():
                path = (saved/name).resolve(strict=True)
                if not path.is_relative_to(saved):
                    raise ValueError('Invalid fixed PF provider file')
                files[path] = digest
            if not files:
                raise ValueError('Missing fixed PF provider files')
        for path, digest in files.items():
            with path.open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError('Saved PF runtime input differs: '+str(path))
        mounts.extend(('--ro-bind', str(saved), str(original)))
    return mounts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    trailing = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    python = config['python']
    # Adapter children inherit the already prepared mount namespace. The
    # worker's interpreter identity probe has no provider input to map.
    if os.environ.get('RPE_PF_LEGACY_RUNTIME') == str(args.config) or '-c' in trailing[:2]:
        os.execv(python, [python, *trailing])
    if trailing[:1] != ['-B'] or len(trailing) < 2:
        raise ValueError('Expected a fixed PF script')
    scope = Path(config['write_root']).resolve(strict=True)
    script = Path(trailing[1]).resolve(strict=True)
    entry = script.name
    if (script == Path(__file__).with_name('run_rpe_q3_worker.py').resolve()
            and len(trailing) == 4 and trailing[2] == '--supervise-attempt'):
        attempt = Path(trailing[3]).resolve(strict=True)
        if attempt == scope or not attempt.is_relative_to(scope) or not attempt.is_dir():
            raise ValueError('Fixed PF attempt is outside the owned workspace')
        # The supervisor records its one-shot ownership and starts this
        # launcher again for dry-check and execution inside the input mounts.
        os.execv(python, [python, *trailing])
    if entry == 'run_rpe_preparation_job.py':
        target = Path(trailing[trailing.index('--task-root')+1]).resolve()
    elif entry in ('check_pf_preparation_owner.py', 'run_pf_preparation_stage.py'):
        output = Path(trailing[trailing.index('--output')+1]).resolve()
        target = scope/output.relative_to(scope).parts[0]
    else:
        raise ValueError('Unsupported fixed PF entry: '+entry)
    if target == scope or not target.is_relative_to(scope) or not target.is_dir():
        raise ValueError('Fixed PF write location is outside the owned workspace')
    executable = shutil.which('bwrap')
    if executable is None:
        raise ValueError('Fixed PF runtime requires bubblewrap')
    home = Path.home()
    command = [executable,'--die-with-parent','--ro-bind','/','/','--dev','/dev',
               '--tmpfs','/home','--ro-bind',str(home),str(home)]
    cache_scope = scope
    if config.get('projects_overlay'):
        projects = home/'projects'
        project = Path(config['projects_overlay']).resolve(strict=True)
        if project.parent != projects:
            raise ValueError('Fixed PF project is outside the dedicated PBS location')
        command += ['--tmpfs',str(projects),'--ro-bind',str(project),str(project)]
        cache_scope = project
    caches = [Path(p).resolve(strict=True) for p in config.get('cache_directories', [])]
    for directory in caches:
        if directory == cache_scope or not directory.is_relative_to(cache_scope):
            raise ValueError('Fixed PF cache is outside the configured write root')
        command += ['--bind',str(directory),str(directory)]
    command += ['--bind',str(target),str(target)]
    cache = target/'pf-runtime-cache'
    cache.mkdir(exist_ok=True)
    for name in ('TMPDIR','MPLCONFIGDIR','NUMBA_CACHE_DIR'):
        command += ['--setenv',name,str(cache)]
    command += fixed_files(config)
    command += ['--setenv','RPE_PF_LEGACY_RUNTIME',str(args.config),
                '--chdir',os.getcwd(),'--',python,*trailing]
    os.execv(executable, command)


if __name__ == '__main__':
    main()
