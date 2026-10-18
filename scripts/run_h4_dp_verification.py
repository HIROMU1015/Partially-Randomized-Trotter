"""Fail-closed future D/P runner. Phase 0 invokes only its rejection path."""
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import os
import signal
from pathlib import Path
import sys

import h4_dp_contract as c

ROOT = Path(__file__).resolve().parents[1]
_CONTEXT = None
_SPEC = None


def worker_context_spec(spec):
    """Send loader context only; the parent retains all authorization manifests."""
    keys = ['stage', 'output', 'native_config']
    if spec['stage'] == 'P':
        keys += ['A_native_adapter', 'cells']
    return {key: spec[key] for key in keys}


def initialize(spec):
    global _CONTEXT, _SPEC
    _SPEC = spec
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    # Own workers must not outlive an abnormal own-driver exit.
    import ctypes
    parent = os.getppid()
    c.require(ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) == 0, 'OWN_PARENT_DEATH_SIGNAL')
    c.require(os.getppid() == parent, 'OWN_PARENT_EXITED_DURING_INITIALIZATION')
    # Process-local caches/threads only. No affinity, cgroup or shared setting changes.
    pid_root = Path(spec['output']) / 'worker_contexts' / str(os.getpid())
    pid_root.mkdir(parents=True)
    for key in ('TMPDIR', 'MPLCONFIGDIR', 'XDG_CACHE_HOME', 'NUMBA_CACHE_DIR'):
        target = pid_root / key.lower(); target.mkdir()
        os.environ[key] = str(target)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    _CONTEXT = None


def work(task):
    global _CONTEXT
    from h4_dp_science import deterministic_task, confirmation_task
    # Loader failure propagates as a task exception rather than causing an
    # initializer respawn loop. A dead native worker breaks the executor.
    if _CONTEXT is None:
        from h4_dp_science import load_context
        _CONTEXT = load_context(_SPEC, Path(_SPEC['output']) / 'worker_contexts' / str(os.getpid()))
    directory = Path(_SPEC['output']) / 'tasks' / task['id']
    directory.mkdir(parents=True, exist_ok=False)
    operation = deterministic_task if _SPEC['stage'] == 'D' else confirmation_task
    result = operation(_CONTEXT, task, directory)
    c.save(directory / 'result.json', result)
    return result


def execute(spec, grant):
    c.authorize(spec, grant, ROOT)
    c.validate_tasks(spec)
    # No science import, mkdir or seed sampling occurs before the approval gate.
    import subprocess
    for name, expected in spec['source_hashes'].items():
        blob = subprocess.check_output(['git', 'show', spec['SOURCE'] + ':' + name], cwd=ROOT)
        c.require(__import__('hashlib').sha256(blob).hexdigest() == expected, 'SOURCE_GIT_BINDING')
    output = Path(spec['output'])
    c.require(output.parent == Path(spec['lock']).parent, 'EXCLUSIVE_RUN_ROOT')
    with Path(spec['lock']).open('x') as stream:
        stream.write(c.encoded(dict(SOURCE=spec['SOURCE'], spec_digest=c.digest(spec))).decode())
    output.mkdir(exist_ok=False)
    c.save(output / 'start.json', dict(SOURCE=spec['SOURCE'], retry=False, next_stage_authorized=False))
    try:
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(spec['workers'], mp_context=context,
                                 initializer=initialize, initargs=(worker_context_spec(spec),)) as pool:
            def batch(tasks):
                futures = [pool.submit(work, task) for task in tasks]
                return [future.result() for future in futures]
            if spec['stage'] == 'P':
                first = [t for t in spec['tasks'] if t['replica'] == 0]
                rest = [t for t in spec['tasks'] if t['replica'] != 0]
                rows = batch(first)
                rows += batch(rest)
            else:
                rows = batch(spec['tasks'])
            processes = list(pool._processes.values())
        c.require(all(p.exitcode == 0 for p in processes), 'NATIVE_WORKER_EXIT')
        c.require(len(rows) == len(spec['tasks']) and len({r['cell_id'] + ':' + str(r.get('replica', 0))
                                                      for r in rows}) == len(rows), 'COMPLETE_FIXED_TASKS')
        actual_calls = {name: sum(r['calls'].get(name, 0) for r in rows)
                        for name in {k for r in rows for k in r['calls']}}
        c.require(all(type(spec['calls'].get(name)) is int and used <= spec['calls'][name]
                      for name, used in actual_calls.items()), 'GLOBAL_SCIENTIFIC_CALL_BUDGET')
        if spec['stage'] == 'D':
            from h4_dp_science import deterministic_account
            measurements = c.read(spec['saved_measurements'])
            measurements.update({r['cell_id']: r['axes'] for r in rows})
            result = deterministic_account(c.read(spec['signal_catalog']), c.read(spec['saved_endpoints']), measurements,
                                           c.read(spec['previous_subset_comparisons']))
        else:
            result = dict(paired_costs=rows, comparisons=[], within_candidate_covariance={})
            for endpoint in spec['evaluation_endpoints']:
                pairs = sorted([r for r in rows if r['cell_id'] == endpoint['cell_id']], key=lambda r: r['replica'])
                c.require(len(pairs) == spec['n_fixed'][endpoint['cell_id']], 'FIXED_N_COVERAGE')
                result['comparisons'].append(dict(endpoint=endpoint, **c.evaluate_pairs(pairs,
                    endpoint['saved_shots'], endpoint['PF_G_exact'], endpoint['expected_direction'])))
            for cid in spec['n_fixed']:
                points = [ep for ep in spec['evaluation_endpoints'] if ep['cell_id'] == cid]
                pairs = sorted([r for r in rows if r['cell_id'] == cid], key=lambda r: r['replica'])
                result['within_candidate_covariance'][cid] = dict(endpoint_order=points,
                    covariance_exact=c.within_candidate_covariance(pairs, [ep['saved_shots'] for ep in points]))
            result['stratum_counts'] = {cid: {label: sum(r['cell_id'] == cid and r['stratum'] == label for r in rows)
                for label in ('S0_ONLY', 'S2_SINGLE', 'S2_MULTI', 'S4_PRESENT')} for cid in spec['n_fixed']}
        # Recheck only the actual execution closure. Unrelated worktrees are never gates.
        for row in spec['inputs']:
            c.require(c.sha(row['path']) == row['sha256'], 'FINAL_INPUT_CHANGED')
        for name, h in spec['source_hashes'].items():
            c.require(c.sha(ROOT / name) == h, 'FINAL_SOURCE_CHANGED')
        for group in ('runtime_hashes', 'provider_hashes'):
            for name, h in spec[group].items():
                c.require(c.sha(name) == h, 'FINAL_EXECUTION_CLOSURE_CHANGED')
        if spec['stage'] == 'P':
            for directory, files in spec['frozen_stage_closure'].items():
                for name, h in files.items():
                    c.require(c.sha(Path(directory)/name) == h, 'FINAL_P_KERNEL_CHANGED')
        result['actual_calls'] = actual_calls
        result['actual_calls']['endpoint_accounting_pass' if spec['stage'] == 'P' else 'deterministic_accounting_pass'] = 1
        c.save(output / 'result.json', result)
        c.save(output / 'terminal.json', dict(status='COMPLETE_MANDATORY_STOP', SOURCE=spec['SOURCE'],
            native_workers_reaped=True, retry=False, next_stage_authorized=False))
        return result
    except BaseException as error:
        c.save(output / 'STOP.json', dict(status='TECHNICAL_STOP', reason=type(error).__name__ + ':' + str(error),
            reservations_retained=True, retry=False, next_stage_authorized=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--grant', type=Path, required=True)
    args = parser.parse_args()
    execute(c.read(args.spec), c.read(args.grant))


if __name__ == '__main__':
    main()
