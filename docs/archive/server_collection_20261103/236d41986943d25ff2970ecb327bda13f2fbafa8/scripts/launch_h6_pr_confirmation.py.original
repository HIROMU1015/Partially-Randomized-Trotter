"""Own-session H6 native-exit witness. Phase 0 proposals/invalid grants fail closed."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h6_pr_confirmation as h
import launch_h8_pr_confirmation as witness
from h6_pr_confirmation_seal import audit_completion


def launch(spec, grant, directory):
    h.authorize(spec, grant)
    h.verify(spec)
    h.shared.fresh_gate(spec, grant)
    directory = h.c.home_path(directory)
    directory.mkdir(exist_ok=False)
    h.c.claim_attempt(directory / 'WITNESS_ONE_SHOT.json', spec['SOURCE'], spec['scope'])
    child, owned = None, None
    try:
        witness.check_interruption()
        with (directory / 'driver.log').open('x') as log:
            child = subprocess.Popen([spec['python'], '-P', '-B', str(h.ROOT / 'scripts/h6_pr_confirmation.py'),
                '--spec', grant['spec_path'], '--grant', grant['grant_path']], stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
            try:
                owned = witness.identity(child.pid)
            except (FileNotFoundError, ProcessLookupError):
                receipt = witness.reap(child, block=True)
            else:
                while True:
                    witness.check_interruption()
                    receipt = witness.reap(child)
                    if receipt is not None:
                        break
                    time.sleep(.5)
        h.c.save(directory / 'native_completion.json', receipt)
        h.c.require(receipt['exit_code'] == 0 and receipt['reaped'], 'H6_DRIVER_NATIVE_EXIT0_REQUIRED')
        terminal = h.c.read(Path(spec['output']) / 'terminal.json')
        h.c.require(terminal['status'] == 'COMPLETE_MANDATORY_STOP'
                    and terminal['SOURCE'] == spec['SOURCE']
                    and terminal['result_sha256'] == h.c.sha(Path(spec['output']) / 'result.json'),
                    'H6_COMPLETION_RESULT_SHA')
        h.verify(spec)
        raw_audit = audit_completion(spec, receipt)
        h.c.save(directory / 'completion_audit.json', dict(status='COMPLETE_MANDATORY_STOP',
            SOURCE=spec['SOURCE'], scope=spec['scope'], native=receipt,
            raw_audit=raw_audit, retry=False, next_stage_authorized=False))
    except BaseException as error:
        if child is not None and child.returncode is None:
            receipt = witness.stop(child, owned) if owned is not None else witness.reap(child, block=True)
            h.c.save(directory / 'interruption_native_completion.json', receipt)
        h.c.first_stop(directory, type(error).__name__ + ':' + str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--grant', required=True, type=Path)
    parser.add_argument('--witness-directory', required=True, type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        witness.INTERRUPTED = signum
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    # Bind exactly the bytes inspected by this witness to its child command.
    spec, grant = h.c.read(args.spec), h.c.read(args.grant)
    h.c.require(grant.get('spec_path') == str(args.spec.resolve())
                and grant.get('grant_path') == str(args.grant.resolve()), 'EXACT_LAUNCH_ARGUMENT_PATHS')
    launch(spec, grant, args.witness_directory)
