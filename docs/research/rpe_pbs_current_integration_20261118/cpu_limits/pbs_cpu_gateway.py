"""Apply a CPU budget to new PBS allocations and delegate the existing RPCs."""

import argparse
import fcntl
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import re
import sys


def reserved_cpus(gateway, runs, requested_root):
    """Count scheduler allocations and local intents not visible to qstat yet."""
    jobs = gateway.qstat()
    rpe_jobs = [
        job for job in jobs.values()
        if re.fullmatch(r'RPE[0-9a-f]{12}', job.get('Job_Name', ''))
    ]
    names = {job['Job_Name'] for job in rpe_jobs}
    terminal = {'F', 'X'}
    used = sum(
        int(job['Resource_List']['ncpus']) for job in rpe_jobs
        if job['job_state'] not in terminal
    )
    directories = set(runs.glob('*/allocations/*/SPEC.json'))
    directories.update(requested_root.glob('allocations/*/SPEC.json'))
    for path in directories:
        spec = gateway.read(path)
        # Scheduler jobs were already counted, or have finished.
        if 'RPE' + spec['token'][:12] in names:
            continue
        receipt = gateway.read(path.parent / 'QSUB_RETURN.json') or {}
        state = gateway.read(path.parent / 'STATE.json') or {}
        if receipt.get('returncode', 0) != 0 or state.get('state') == 'EXITED':
            continue
        # A lost qsub response or unconfirmed terminal state still reserves CPU.
        used += int(spec['resources']['ncpus'])
    return used


def handle(gateway, policy, root, request):
    if request['action'] != 'ensure':
        return gateway.handle(root, request)
    spec = request['spec']
    token = gateway.safe_id(spec['token'])
    directory = root / 'allocations' / token
    limit = policy['max_cpus']
    requested = spec['resources']['ncpus']
    if type(requested) is not int or not 0 < requested <= limit:
        raise ValueError(f'PBS allocation must request 1–{limit} CPUs')
    runs = Path(policy['runs']).resolve()
    if root.parent != runs:
        raise ValueError('CPU admission requires the configured run namespace')
    with (runs / '.CPU_ADMISSION.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (directory / 'SPEC.json').exists():
            control = gateway.read(directory / 'CONTROL.json') or {}
            if control.get('stop'):
                return dict(token=token, status='STOPPED_BEFORE_QSUB')
            if reserved_cpus(gateway, runs, root) + requested > limit:
                return dict(token=token, status='CPU_BUDGET_WAIT')
        # ensure writes the durable SPEC under the budget lock, before qsub.
        return gateway.handle(root, request)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    root = parser.parse_args().root.resolve()
    policy = json.loads(Path(__file__).with_name('PBS_CPU_POLICY.json').read_text())
    spec = spec_from_file_location('_rpe_fixed_pbs_gateway', policy['gateway_script'])
    gateway = module_from_spec(spec)
    spec.loader.exec_module(gateway)
    for line in sys.stdin:
        try:
            result = handle(gateway, policy, root, json.loads(line))
            reply = dict(ok=True, result=result)
        except Exception as error:
            reply = dict(ok=False, error=f'{type(error).__name__}: {error}')
        print(json.dumps(reply, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
