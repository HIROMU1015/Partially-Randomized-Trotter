"""One launcher, one owned stdlib child, one saved-accounting execution.

Preparation/check mode never evaluates scientific costs. No automatic retry.
"""
import argparse
import csv
import errno
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_six_axis_reaccount_contract as c
n = c.n


def csv_file(path, rows):
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def driver(config, grant):
    permit = c.authorize(config, grant)
    c.verify(config)
    control, output = [c.home(config[k]) for k in ('control', 'output')]
    fresh = c.read(control / 'fresh_gate.json')
    lock = c.read(config['lock'])
    c.require(fresh['status'] == 'PASS' and fresh['SOURCE'] == config['SOURCE'] and
        fresh['config_sha256'] == lock['config_sha256'] == c.digest(config) and
        fresh['grant_sha256'] == lock['grant_sha256'] == c.digest(grant) and
        0 <= time.monotonic() - fresh['observation']['monotonic'] < 30 and
        lock['launcher_pid'] == os.getppid() and lock['scientific_dispatches_reserved'] == 1,
        'FRESH_NATIVE_PARENT_ONESHOT_BINDING')
    first_stop = None
    try:
        data, registry, measured = c.load_saved(config, permit)
        result = n.reaccount(data, registry, measured, permit)
        all_rows = result.pop('accounting_rows')
        changes = result.pop('measured_candidate_changes')
        result.update(SOURCE=config['SOURCE'], plan_sha256=config['plan_sha256'],
                      grant_sha256=c.digest(grant), input_manifest_sha256=c.sha(c.ROOT / c.INPUTS))
        c.write(output / 'comparisons.json', result)
        c.write(output / 'measured_candidate_changes.json', changes)
        with (output / 'accounting_rows.jsonl.gz').open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                for row in all_rows:
                    stream.write((json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n').encode())
        summary = []
        for row in result['comparisons']:
            old, new, pr = row['before']['best'], row['after']['best'], row['saved_PR']['best']
            summary.append(dict(bias_kind=row['bias_kind'], epsilon=row['epsilon'], factor=row['factor'],
                scope=row['scope'], winner_before=old['cell_id'], winner_after=new['cell_id'],
                G_before_exact=old['G_exact'], G_after_exact=new['G_exact'],
                winner_cost_status=new['cost_status'], PR_saved_id=pr['cell_id'],
                PR_G_exact=pr['G_exact'], PR_saved_SE=pr.get('saved_paired_SE', pr.get('G_standard_error')),
                reduction_before_percent_exact=row['reduction_before_exact'],
                reduction_after_percent_exact=row['reduction_after_exact'],
                change_percentage_points_exact=row['change_percentage_points_exact'],
                winner_changed=row['winner_changed'], **row['counts']))
        csv_file(output / 'comparison_summary_exact.csv', summary)
        csv_file(output / 'mode_differences_exact.csv', result['mode_differences'])
        csv_file(output / 'accounting_rows_exact.csv', [dict(
            bias_kind=r['bias_kind'], cell_id=r['saved']['cell_id'], epsilon=r['saved']['epsilon'],
            factor=r['saved']['factor'], eligible=r['saved']['eligible'],
            G_before_exact=r['saved']['G_exact'], G_after_exact=r['updated']['G_exact'],
            real_shots=(r['saved_shots'] or {}).get('real'), imag_shots=(r['saved_shots'] or {}).get('imag'),
            cost_status_before=r['saved']['cost_status'], cost_status_after=r['updated']['cost_status']) for r in all_rows])
        c.verify(config)
        receipts = [dict(name=p.name, bytes=p.stat().st_size, sha256=c.sha(p)) for p in output.iterdir()]
        c.write(control / 'driver_terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=config['SOURCE'],
            first_stop=None, scientific_dispatches=1, saved_rows=3800, comparisons=40,
            output_receipts=receipts, retry=False, next_stage_authorized=False))
        return 0
    except BaseException as error:
        first_stop = type(error).__name__ + ':' + str(error)
        c.write(control / 'driver_terminal.json', dict(status='STOP_MANDATORY_STOP', SOURCE=config['SOURCE'],
            first_stop=first_stop, retry=False, next_stage_authorized=False))
        return 1


def wait_owned(child, receipt_path, *, interrupt=None, poll=None):
    """Only an unreaped direct child can be signalled. FD and streams close.

    pidfd pins the task even if it exits between observation and signal. ESRCH
    alone means already exited; other signalling errors remain cleanup errors.
    """
    c.require(child.returncode is None, 'ALREADY_REAPED_CHILD_NO_SECOND_CLEANUP')
    fields = Path('/proc', str(child.pid), 'stat').read_text().rsplit(')',1)[1].split()
    c.require(int(fields[1]) == os.getpid() and Path('/proc', str(child.pid)).stat().st_uid == os.getuid(),
              'DIRECT_CHILD_OWNERSHIP_REQUIRED')
    fd, status, usage, first_stop = None, None, None, None
    errors = []
    start = time.monotonic()
    try:
        fd = os.pidfd_open(child.pid)
        try:
            while status is None:
                pid, s, ru = os.wait4(child.pid, os.WNOHANG)
                if pid:
                    status, usage = s, ru
                    break
                if poll: poll()
                if interrupt: interrupt()
                time.sleep(.05)
        except BaseException as error:
            first_stop = type(error).__name__ + ':' + str(error)
    except ProcessLookupError:
        # Still an unreaped direct child: wait4 must provide the native status.
        pass
    except BaseException as error:
        first_stop = type(error).__name__ + ':' + str(error)
    finally:
        if status is None:
            for sig, seconds in ((signal.SIGTERM, 2), (signal.SIGKILL, 5)):
                if fd is not None:
                    try:
                        signal.pidfd_send_signal(fd, sig)
                    except OSError as error:
                        if error.errno != errno.ESRCH:
                            errors.append('SIGNAL:' + repr(error))
                else:
                    # Popen.pid is reserved by the unreaped direct child.
                    try: os.kill(child.pid, sig)
                    except OSError as error:
                        if error.errno != errno.ESRCH: errors.append('SIGNAL:' + repr(error))
                until = time.monotonic() + seconds
                while time.monotonic() < until:
                    try:
                        pid, s, ru = os.wait4(child.pid, os.WNOHANG)
                    except ChildProcessError as error:
                        errors.append('WAIT:' + repr(error))
                        break
                    if pid:
                        status, usage = s, ru
                        break
                    time.sleep(.02)
                if status is not None: break
        if fd is not None: os.close(fd)
        for stream in (child.stdin, child.stdout, child.stderr):
            if stream is not None: stream.close()
    if status is not None:
        child.returncode = os.waitstatus_to_exitcode(status)
    receipt = dict(pid=child.pid, native_exit_code=child.returncode, reaped=status is not None,
        wait_kind='os.wait4_DIRECT_UNREAPED_CHILD', wall_seconds=time.monotonic()-start,
        first_stop=first_stop, cleanup_errors=errors, automatic_retry=False,
        peak_RSS_bytes=None if usage is None else usage.ru_maxrss*1024,
        CPU_seconds=None if usage is None else usage.ru_utime+usage.ru_stime)
    c.write(receipt_path, receipt)
    return receipt


def launch(config, grant):
    c.authorize(config, grant)
    output, control, lock = c.paths(config)
    fresh = c.fresh(config, grant)
    with lock.open('x') as stream:
        json.dump(dict(config_sha256=c.digest(config), grant_sha256=c.digest(grant),
                       launcher_pid=os.getpid(), scientific_dispatches_reserved=1), stream)
    control.mkdir()
    output.mkdir()
    c.write(control / 'fresh_gate.json', fresh)
    c.write(control / 'execution_binding.json', dict(SOURCE=config['SOURCE'], plan_sha256=config['plan_sha256'],
        config_sha256=c.digest(config), grant_sha256=c.digest(grant), input_manifest_sha256=c.sha(c.ROOT/c.INPUTS),
        source_closure=config['source_hashes'], inputs=config['inputs'], environment=config['environment'],
        one_shot_consumed=True, retry=False, next_stage_authorized=False))
    first_stop, native, child = None, None, None
    handlers = {}
    stop_requested = [None]
    def interrupted(sig, frame):
        if stop_requested[0] is None: stop_requested[0] = 'LAUNCHER_SIGNAL_' + str(sig)
    def check_interrupt():
        request = c.home(config['_stop_request_path'])
        if request.exists():
            saved = c.read(request)
            c.require(saved['config_sha256'] == c.digest(config) and
                      saved['grant_sha256'] == c.digest(grant), 'STOP_REQUEST_BINDING')
            if stop_requested[0] is None: stop_requested[0] = saved['first_stop']
        if stop_requested[0] is not None: raise KeyboardInterrupt(stop_requested[0])
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, interrupted)
        with (control / 'stdout.txt').open('x') as stdout, (control / 'stderr.txt').open('x') as stderr:
            check_interrupt()
            child = subprocess.Popen([c.PYTHON, '-I', '-S', '-P', '-B', str(Path(__file__).resolve()),
                '--driver', '--config', config['_config_path'], '--grant', config['_grant_path']],
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, close_fds=True)
            native = wait_owned(child, control / 'native_driver_exit.json', interrupt=check_interrupt)
        if native['first_stop']: first_stop = native['first_stop']
        c.require(native['native_exit_code'] == 0 and native['reaped'] is True and
                  native['cleanup_errors'] == [], 'NATIVE_CHILD_NOT_COMPLETE')
        terminal = c.read(control / 'driver_terminal.json')
        c.require(terminal['status'] == 'COMPLETE_MANDATORY_STOP', 'DRIVER_TERMINAL_STOP')
        for row in terminal['output_receipts']:
            p = output/row['name']
            c.require(p.stat().st_size == row['bytes'] and c.sha(p) == row['sha256'], 'RESULT_RECEIPT')
        c.verify(config)
        after = c.observe(output.parent)
        c.write(control / 'resources_after.json', after)
        c.write(control / 'final_gate.json', dict(status='PASS_MANDATORY_STOP', SOURCE=config['SOURCE'],
            first_stop=None, native_exit_code=0, all_owned_processes_reaped=True,
            source_input_bytes_unchanged=True, PR_shot_old32_invariance='PASS_IN_ACCOUNTING',
            other_jobs_modified=False, retry=False, next_stage_authorized=False))
        return 0
    except BaseException as error:
        if first_stop is None: first_stop = type(error).__name__ + ':' + str(error)
        # An interrupt immediately after Popen must follow the same own-child
        # cleanup path as one received while waiting.
        if child is not None and child.returncode is None:
            def stop_wait(): raise KeyboardInterrupt(first_stop)
            native = wait_owned(child, control / 'native_driver_exit.json', interrupt=stop_wait)
        terminal_path = control / 'driver_terminal.json'
        if terminal_path.exists():
            first_stop = c.read(terminal_path).get('first_stop') or first_stop
        c.write(control / 'final_gate.json', dict(status='STOP_MANDATORY_STOP', SOURCE=config['SOURCE'],
            first_stop=first_stop, native=native, retry=False, next_stage_authorized=False))
        return 1
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)


def main():
    # Explicitly restore signal delivery in a child started by an outer witness.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--grant')
    parser.add_argument('--check', action='store_true', help='metadata only; zero scientific calls')
    parser.add_argument('--driver', action='store_true')
    args = parser.parse_args()
    config = c.read(args.config)
    if args.check:
        c.verify(config)
        proof, _ = c.proof_metadata(config)
        print(json.dumps(dict(proof=proof, saved_structure=c.check_saved_structure(config))))
        return 0
    c.require(args.grant, 'GRANT_FILE_REQUIRED')
    grant = c.read(args.grant)
    if args.driver: return driver(config, grant)
    # Procedural paths are explicitly part of the sealed config to avoid hidden
    # changes to its canonical digest after authorization.
    c.require(config['_config_path'] == str(c.home(args.config)) and
              config['_grant_path'] == str(c.home(args.grant)), 'CONFIG_GRANT_FILE_BINDING')
    return launch(config, grant)


if __name__ == '__main__':
    sys.exit(main())
