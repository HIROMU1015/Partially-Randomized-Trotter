"""Select a run's transport relay after the ordinary central start/resume.

This changes only mailbox polling. Scientific workers, requests and the run's
fixed controller/source remain owned by their existing execution contracts.
"""
import argparse
import fcntl
import json
import os
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


def write_watch(path, value):
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def watch_exit_reason(run, invocation):
    if (run/'STOP_REQUESTED.json').exists():
        return 'USER_STOP_REQUESTED'
    if read(run/'LAUNCH.json')['invocation_id'] != invocation:
        return 'INVOCATION_CHANGED'
    driver = read(run/'NORMAL_DRIVER.json')
    if driver.get('invocation_id') != invocation or not process_arguments(driver['pid']):
        return 'DRIVER_NOT_ACTIVE'
    return None


def watch(configuration, run_id, invocation):
    config = read(configuration)
    run = Path(read(Path(config['state_directory'])/'RUNS.json')[run_id]['run_directory']).resolve()
    directory = run/'distributed/relay'
    script = str(Path(config['runs'][run_id]['entry']).resolve(strict=True))
    result = dict(run_id=run_id, run_directory=str(run))
    record = dict(status='WATCHING', pid=os.getpid(), invocation_id=invocation,
                  entry=str(Path(__file__).resolve()), started_epoch=time.time())
    # One watcher owns this lock for its entire lifetime. Normal operations use
    # REGISTRY.lock independently and never wait for the watcher's lifetime.
    with (directory/'WATCH.lock').open('a') as lock:
        # The parent holds this briefly while publishing this child's PID.
        # Blocking here avoids a child-start race with that publication.
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            while True:
                reason = watch_exit_reason(run, invocation)
                if reason:
                    record.update(status='STOPPED', reason=reason, stopped_epoch=time.time())
                    break
                record.update(status='RELAY_TRANSPORT_PENDING' if record.get('error') else 'WATCHING',
                              heartbeat_epoch=time.time())
                path = directory/'STATUS.json'
                if path.exists():
                    relay = read(path)
                    args = process_arguments(relay['pid'])
                    # An inactive relay between optimizer batches is normal.
                    # Only replace an existing generation; never create demand.
                    if not relay.get('stopped') and args and script not in args:
                        record['status'] = 'SELECTING'
                        write_watch(directory/'WATCH.json', record)
                        try:
                            record['last_selection'] = ensure(config, result)
                            record.pop('error', None)
                            record['status'] = 'WATCHING'
                        except Exception as error:
                            record.update(status='RELAY_TRANSPORT_PENDING', error=str(error)[:1500])
                write_watch(directory/'WATCH.json', record)
                time.sleep(1)
        finally:
            record.setdefault('stopped_epoch', time.time())
            if record['status'] not in ('STOPPED', 'RELAY_TRANSPORT_PENDING'):
                record['status'] = 'STOPPED'
            write_watch(directory/'WATCH.json', record)
    return record


def ensure_watch(configuration, result, selection):
    if selection.get('status') != 'RELAY_TRANSPORT_CONFIRMED':
        return None
    configuration = Path(configuration).resolve()
    config = read(configuration)
    run = Path(result['run_directory']).resolve()
    invocation = selection['invocation_id']
    directory = run/'distributed/relay'
    command = [config['python'], '-B', str(Path(__file__).resolve()), '--config', str(configuration),
               '--watch', '--run-id', result['run_id'], '--invocation', invocation]
    deadline = time.monotonic()+3
    with (directory/'WATCH.lock').open('a') as lock:
        while True:
            path = directory/'WATCH.json'
            existing = read(path) if path.exists() else {}
            args = process_arguments(existing['pid']) if existing.get('pid') else []
            if args and args[1:] == command[1:]:
                return dict(status='ALREADY_CURRENT', pid=existing['pid'], invocation_id=invocation)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise ValueError('Previous relay watcher is still draining')
                time.sleep(.1)
        reason = watch_exit_reason(run, invocation)
        if reason:
            raise ValueError('Relay watcher will not start: '+reason)
        with (directory/'WATCH.log').open('a') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        record = dict(status='STARTING', pid=process.pid, invocation_id=invocation,
                      entry=str(Path(__file__).resolve()), started_epoch=time.time())
        write_watch(directory/'WATCH.json', record)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--invocation')
    args = parser.parse_args()
    if args.watch and not (args.run_id and args.invocation):
        parser.error('--watch requires --run-id and --invocation')
    try:
        if args.watch:
            answer = watch(args.config, args.run_id, args.invocation)
        else:
            result = json.load(sys.stdin)
            answer = ensure(read(args.config), result)
            watcher = ensure_watch(args.config, result, answer)
            if watcher:
                answer['watcher'] = watcher
        print(json.dumps(answer)); return 0
    except Exception as error:
        print(json.dumps(dict(status='RELAY_TRANSPORT_PENDING', error=str(error)))); return 1


if __name__ == '__main__':
    raise SystemExit(main())
