"""Select a run's transport relay after the ordinary central start/resume.

This changes only mailbox polling. Scientific workers, requests and the run's
fixed controller/source remain owned by their existing execution contracts.
"""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def process_arguments(pid):
    try:
        directory = Path('/proc') / str(int(pid))
        if (directory / 'stat').read_text().split(') ', 1)[1].split()[0] == 'Z':
            return []
        return (directory / 'cmdline').read_bytes().decode().strip('\0').split('\0')
    except (OSError, ValueError):
        return []


def ensure(config, result):
    run_id = result.get('run_id')
    entry = config.get('runs', {}).get(run_id)
    if not entry:
        return dict(status='NOT_CONFIGURED', run_id=run_id)
    state = Path(config['state_directory']).resolve()
    run = Path(read(state / 'RUNS.json')[run_id]['run_directory']).resolve()
    if Path(result['run_directory']).resolve() != run:
        raise ValueError('Confirmed run differs from the registered run directory')
    script = Path(entry['entry']).resolve(strict=True)
    python = config['python']
    directory = run / 'distributed/relay'
    directory.mkdir(parents=True, exist_ok=True)
    relay_config = (directory / 'CONFIG.json').resolve()
    invocation = read(run / 'LAUNCH.json')['invocation_id']
    deadline = time.monotonic() + config.get('wait_seconds', 75)

    def current():
        if read(run / 'LAUNCH.json')['invocation_id'] != invocation:
            raise ValueError('Run invocation changed during relay selection')
        if (run / 'STOP_REQUESTED.json').exists():
            raise ValueError('User Stop is pending; relay selection will not restart it')
        driver = read(run / 'NORMAL_DRIVER.json')
        return driver.get('invocation_id') == invocation and bool(process_arguments(driver['pid']))

    def wait_for(check):
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError('Relay selection is pending; existing workers are preserved')
            if current():
                value = check()
                if value:
                    return value
            time.sleep(.2)

    def active():
        path = directory / 'STATUS.json'
        if not path.exists():
            return None
        status = read(path)
        args = process_arguments(status['pid'])
        if status.get('stopped') or not args:
            return None
        if (str(relay_config) not in args or '--run' not in args
                or Path(status['configuration']).resolve() != relay_config):
            raise ValueError('Relay process does not own the confirmed configuration')
        return status, args

    def command(action):
        remaining = max(.1, deadline - time.monotonic())
        observed = subprocess.run([python, '-B', str(script), '--config', str(relay_config), action],
                                  capture_output=True, text=True, timeout=remaining)
        if observed.returncode:
            raise RuntimeError('Relay '+action+' failed: '+observed.stderr[-1000:])

    # Reuse the normal operation lock so a concurrent Stop cannot be undone
    # by the relay CLI clearing its own transport STOP marker during start.
    with (state / 'REGISTRY.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status, args = wait_for(active)
        previous = status['pid']
        changed = str(script) not in args
        if changed:
            command('--stop')
            wait_for(lambda: not process_arguments(previous))
            if not current():
                raise ValueError('Controller is no longer active; relay will not restart')
            command('--start')
            status, args = wait_for(active)
            if str(script) not in args:
                raise ValueError('Started relay does not use the selected transport')
        record = dict(status='RELAY_TRANSPORT_CONFIRMED', run_id=run_id,
                      invocation_id=invocation, action='REPLACED' if changed else 'ALREADY_CURRENT',
                      previous_pid=previous, pid=status['pid'], entry=str(script))
        # Operational launch evidence only; no scientific SOURCE or result edits.
        (directory / 'TRANSPORT.json').write_text(json.dumps(record, indent=2)+'\n')
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    try:
        answer = ensure(read(args.config), json.load(sys.stdin))
        print(json.dumps(answer)); return 0
    except Exception as error:
        print(json.dumps(dict(status='RELAY_TRANSPORT_PENDING', error=str(error)))); return 1


if __name__ == '__main__':
    raise SystemExit(main())
