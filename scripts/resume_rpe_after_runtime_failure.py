"""Wait for a selected failed invocation, then use the normal Resume entry."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.rpe_local0_search import (
    RequestError, load, locked, reconcile_runtime_failure, schedule_driver, write,
)


def resume(central, selection_file, previous_file, instruction):
    service = central/'SERVICE.json'
    state = central/'state'
    selection = load(selection_file)
    previous = load(previous_file)
    run_id = selection['run_id']
    kinds = [(kind, selection['resume_'+kind+'_failure']) for kind in ('ast', 'dispatch', 'pf_request')
             if selection.get('resume_'+kind+'_failure')]
    if len(kinds) != 1:
        raise ValueError('Select exactly one failed invocation')
    kind, invocation = kinds[0]
    run = state/'runs'/run_id
    evidence = selection_file.parent

    def record(status, reason=None):
        data = dict(status=status, run_id=run_id, invocation=invocation,
                    controller_source=selection['source'], reason=reason, updated=time.time())
        write(evidence/'DEFERRED_RESUME.json', data)
        print(json.dumps(data, ensure_ascii=False), flush=True)

    def current_binding():
        if ((run/'STOP_REQUESTED.json').exists() or (run/'SEARCH_TERMINATION.json').exists()
                or load(state/'CURRENT_TARGET.json')['run_id'] != run_id
                or load(run/'LAUNCH.json')['invocation_id'] != invocation):
            return None
        config = load(service)
        if config.get('schedule_resumes', {}).get(run_id) in (previous, selection):
            return config
        return None

    record('WAITING_FOR_RUNTIME_OWNERSHIP')
    while current_binding() is not None:
        with (central/'release-update.lock').open('rb') as release_lock:
            fcntl.flock(release_lock, fcntl.LOCK_EX)
            with locked(state):
                config = current_binding()
                if config is None:
                    break
                try:
                    reconcile_runtime_failure(run, load(run/'LAUNCH.json'), load(run/'EXIT.json'), invocation, kind)
                    schedule_driver(load(run/'CONTROLLER_CONFIG.json'), run, selection)
                except RequestError as error:
                    if error.field != 'runtime_ownership':
                        record('RESUME_REJECTED', str(error))
                        return 1
                    record('WAITING_FOR_RUNTIME_OWNERSHIP', error.reason)
                else:
                    if config['schedule_resumes'][run_id] != selection:
                        (evidence/'SERVICE.before-switch.json').write_bytes(service.read_bytes())
                        config['schedule_resumes'][run_id] = selection
                        with tempfile.NamedTemporaryFile(mode='w', dir=central, encoding='utf-8',
                                                         delete=False) as staged:
                            json.dump(config, staged, indent=2, ensure_ascii=False)
                            staged.write('\n')
                            staged.flush()
                            os.fsync(staged.fileno())
                            os.fchmod(staged.fileno(), stat.S_IMODE(service.stat().st_mode))
                        os.replace(staged.name, service)
                    break
        time.sleep(30)

    if current_binding() is None:
        record('SUPERSEDED_OR_STOPPED')
        return 0
    code = Path(selection['root'])
    command = [sys.executable, '-B', str(code/'scripts/run_rpe_search.py'), '--config',
               str(central/'CLIENT.json'), '--action', 'resume', '--run-id', run_id,
               '--instruction', instruction]
    process = subprocess.run(command, cwd=central.parent, capture_output=True, check=False)
    (evidence/'RESUME_RESPONSE.json').write_bytes(process.stdout)
    (evidence/'RESUME_RESPONSE.stderr').write_bytes(process.stderr)
    try:
        result = json.loads(process.stdout)
    except ValueError:
        record('RESUME_REJECTED', 'The normal entry did not return JSON; see RESUME_RESPONSE.stderr')
        return 1
    status = result.get('status')
    if process.returncode or status not in ('STARTED', 'ALREADY_RUNNING', 'REUSED_COMPLETED'):
        record('RESUME_REJECTED', result.get('errors') or result.get('error') or status)
        return 1
    record('RESUME_'+status)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--central', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--instruction', required=True)
    args = parser.parse_args()
    raise SystemExit(resume(args.central.resolve(), args.selection.resolve(), args.previous.resolve(), args.instruction))
