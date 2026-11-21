"""Execute only missing fixed PR-DF mean jobs; publishable raw, resumable, no retry.

The parent candidate backend owns selection, source/input fixation, commit and
bundle/intake. This worker keeps the existing numerical provider and all gates.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
DESCRIPTOR_EXTENSION = 'src/trottertracks/pr_df_descriptor_extension.py'
sys.path.insert(0, str((Path.cwd() if '--transport-source' in sys.argv else ROOT) / 'src'))
from trottertracks.rpe_preparation_paths import resolve as resolve_runtime_path
CALLER = 'scripts/run_pr_df_candidate_signal_jobs.py'
PROVIDER_COMMIT = 'fefd548ecd56f60d72031b3d385f6f5e6ee95d06'
PROVIDER_PATH = 'src/trottertracks/pr_rpe_signal_provider.py'
B_RELEASE = '437ba4acdb82708b6fc98e3cd119a2a734f7cf64'
LOCAL = 'LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'
IMPLEMENTATION = dict(commit=PROVIDER_COMMIT, path=PROVIDER_PATH,
                      recipe='DIRECT_EXPM_HEAD_PAIRED_TAYLOR_MATRIX_POWER_V1')
STATE = {}
if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def read_git(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def atomic(path, value):
    path = Path(path); temporary = path.with_suffix('.tmp.' + str(os.getpid()))
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def event(output, stage, **values):
    with (Path(output) / 'PROGRESS.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage=stage, epoch=time.time(), pid=os.getpid(), **values), default=str) + '\n')


def fixed_runtime(view):
    """Keep pinned relative imports separate from the worker's own package."""
    view = Path(view).resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=view, text=True).strip() != PROVIDER_COMMIT:
        raise ValueError('Fixed signal view HEAD differs')
    paths = (PROVIDER_PATH, 'src/trottertracks/pr_rpe_signal_plan.py',
             'src/trottertracks/pr_rpe_rmse_s0_v2.py')
    for path in paths:
        if (view / path).read_bytes() != read_git(PROVIDER_COMMIT, path):
            raise ValueError('Fixed signal bytes differ: ' + path)
    # As in the fixed DF planner's FixedRuntime, use a distinct package for
    # the provider and its relative imports, including lazy imports after fork.
    sys.path[:0] = [str(view / 'src'), str(view / 'scripts')]
    package = '_pr_df_signal_worker_fixed'
    directory = view / 'src/trottertracks'
    if package not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package, directory / '__init__.py', submodule_search_locations=[str(directory)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[package] = module
        spec.loader.exec_module(module)
    if list(sys.modules[package].__path__) != [str(directory)]:
        raise ValueError('DF signal worker already bound to a different view')
    modules = [importlib.import_module(package + '.' + Path(path).stem) for path in paths]
    for module, path in zip(modules, paths):
        if Path(module.__file__).resolve() != (view / path).resolve():
            raise ValueError('Fixed signal import came from a foreign package: ' + path)
    provider, metadata, fixed = modules
    node = next(n for n in ast.parse((view / paths[2]).read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'b_metadata')
    namespace = dict(vars(fixed), B_RELEASE=B_RELEASE)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'fixed:DF-B_metadata', 'exec'), namespace)
    b = namespace['b_metadata'](ROOT)
    return provider, metadata, b


def validate_jobs(config, plan, provider, metadata, b):
    b.validate(config)
    if plan['configuration_id'] != config['id'] or plan['provider'] != IMPLEMENTATION:
        raise ValueError('Plan configuration/provider differs')
    if len(plan['rounds']) != len(config['rounds']) or [r['m'] for r in plan['rounds']] != list(range(len(config['rounds']))):
        raise ValueError('Complete ordered plan required')
    jobs = []
    for row in plan['rounds']:
        if row['decision'] != 'REQUEST':
            continue
        m = row['m']; job = row['job']; source = config['rounds'][m]; context = job['physical_context']
        expected = dict(m=m, B_configuration_id=config['id'], B_candidate_id=source['candidate']['id'],
                        B_identity=source['identity'], B_law_binding=source['law_binding'])
        if any(not metadata.common.same(job.get(k), value) for k, value in expected.items()):
            raise ValueError('Request canonical identity differs at m' + str(m))
        if any(not metadata.common.same(context[k], config['spec'][k]) for k in ('binding', 'profile', 'split', 'law')):
            raise ValueError('Request physical contract differs at m' + str(m))
        if (not metadata.common.same(context['physical_candidate'], source['physical_candidate'])
                or not metadata.common.same(context['structure'], source['structure'])
                or context['numerical_recipe'] != provider.numerical_recipe(IMPLEMENTATION)):
            raise ValueError('Request physical schedule/recipe differs at m' + str(m))
        locator = plan.get('terminal_reference')
        if m == 13 and locator is not None:
            path = Path(__file__).resolve().parents[1] / DESCRIPTOR_EXTENSION
            spec = importlib.util.spec_from_file_location('_df_terminal_reference_fixed', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            ref = module.terminal_reference(ROOT, locator, job, provider)
        else:
            ref = provider.saved_h4_reference(ROOT, job)
        if not metadata.common.same(ref, row['request']['reference']):
            raise ValueError('Saved reference differs at m' + str(m))
        jobs.append((job, ref))
    return jobs


def valid_record(record, job):
    if any(record.get(k) != job[k] for k in
           ('m', 'B_configuration_id', 'B_candidate_id', 'B_identity', 'B_law_binding', 'physical_context')):
        return False
    return (record.get('status') == LOCAL and record.get('numerical', {}).get('gate_pass') is True
            and record.get('scalar_once') is True and record.get('source') == IMPLEMENTATION
            and record.get('signal') is not None and bool(record.get('input_source'))
            and bool(record.get('reference_source')))


def one(job_and_reference):
    job, reference = job_and_reference; m = job['m']; output = STATE['output']; start = time.perf_counter()
    event(output, 'ROUND_START', m=m)
    try:
        record = STATE['provider'].evaluate(STATE['context'], job, reference, root=ROOT)
        # Preserve the provider's complete failure record too; never retry it.
        write_new(output / 'raw' / ('m' + str(m) + '.json'), record)
        value = dict(m=m, status='VALID' if valid_record(record, job) else 'NUMERICAL_OR_INPUT_GATE_FAILED',
                     raw_path=str((output / 'raw' / ('m' + str(m) + '.json')).relative_to(ROOT)),
                     provider_status=record.get('status'), elapsed_seconds=time.perf_counter()-start,
                     process_pid=os.getpid(), peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        value = dict(m=m, status='WORKER_ERROR_NO_RETRY', error=traceback.format_exc(),
                     elapsed_seconds=time.perf_counter()-start, process_pid=os.getpid())
    write_new(output / 'terminal' / ('m' + str(m) + '.json'), value)
    event(output, 'ROUND_END', **value)
    return value


def saved_context(config, tail_descriptor, provider):
    extension_path=ROOT/'src/trottertracks/pr_df_descriptor_extension.py'
    extension_spec=importlib.util.spec_from_file_location('_df_signal_descriptor_extension',extension_path)
    extension=importlib.util.module_from_spec(extension_spec);extension_spec.loader.exec_module(extension)
    return extension.context(ROOT,config,tail_descriptor,provider)


def execute(config, plan, output, provider, metadata, b, *, workers=None, tail_descriptor=None):
    """Small testable orchestration layer; real entry supplies the pinned runtime."""
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    (output / 'raw').mkdir(exist_ok=True); (output / 'terminal').mkdir(exist_ok=True); (output / 'started').mkdir(exist_ok=True)
    lock = (output / 'DRIVER.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close(); raise ValueError('Another DF signal driver owns this output')
    start = time.perf_counter()
    rows = []; pending_jobs = []; processes = []; jobs = []
    try:
        jobs = validate_jobs(config, plan, provider, metadata, b)
        event(output, 'DRIVER_START', configuration_id=config['id'], requested_rounds=[j['m'] for j, _ in jobs])
        for job, reference in jobs:
            m = job['m']; terminal = output / 'terminal' / ('m' + str(m) + '.json'); marker = output / 'started' / ('m' + str(m) + '.json')
            raw = output / 'raw' / ('m' + str(m) + '.json')
            if terminal.exists():
                value = json.loads(terminal.read_text())
                if value['status'] == 'VALID' and raw.exists() and valid_record(json.loads(raw.read_text()), job):
                    rows.append(dict(value, disposition='REUSED_TERMINAL')); continue
                rows.append(dict(m=m, status='HELD_PRIOR_FAILURE_NO_RETRY', original_terminal=value)); continue
            if marker.exists() or raw.exists():
                rows.append(dict(m=m, status='HELD_AMBIGUOUS_PRIOR_START_NO_RETRY')); continue
            pending_jobs.append((job, reference))
        if pending_jobs:
            event(output, 'CONTEXT_START')
            ctx=saved_context(config,tail_descriptor,provider)
            event(output, 'CONTEXT_END', scalar_components=dict(
                hamiltonian_constant=ctx.constant,
                tail_identity=ctx.tails[config['spec']['L_D']]['scalar'],
                application='TOTAL_SCALAR_ONCE_AT_TOTAL_T_BY_FIXED_PROVIDER'))
            # Provider's context and cached macros are private copies after fork.
            STATE.update(provider=provider, context=ctx, output=output)
            context = __import__('multiprocessing').get_context('fork')
            count = workers if workers is not None else len(pending_jobs)
            if type(count) is not int or count < 1:
                raise ValueError('Positive worker count required')
            with ProcessPoolExecutor(max_workers=count, mp_context=context) as pool:
                future_jobs = {}
                for pair in pending_jobs:
                    job = pair[0]; m = job['m']
                    # The marker precedes submit; a crash here remains ambiguous, never an implicit retry.
                    write_new(output / 'started' / ('m' + str(m) + '.json'),
                              dict(m=m, job_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
                                   driver_pid=os.getpid(), epoch=time.time(), automatic_retry=False))
                    future_jobs[pool.submit(one, pair)] = m
                processes = list(pool._processes.values())
                while future_jobs:
                    done, _ = wait(future_jobs, timeout=10, return_when=FIRST_COMPLETED)
                    for future in done:
                        m = future_jobs.pop(future)
                        try:
                            value = future.result(); rows.append(dict(value, disposition='NEW_PROVIDER_CALL'))
                        except BaseException:
                            rows.append(dict(m=m, status='PROCESS_FAILURE_NO_RETRY', error=traceback.format_exc()))
                    event(output, 'PROGRESS', finished=len(rows), pending=list(future_jobs.values()))
        complete = len(rows) == len(jobs) and all(r['status'] == 'VALID' for r in rows)
        result = dict(schema='PR_DF_CANDIDATE_SIGNAL_EXECUTION_V1', status='REQUESTED_ROUNDS_COMPLETE' if complete else 'PARTIAL_OR_HELD_NO_RETRY',
                      configuration_id=config['id'], rows=sorted(rows, key=lambda r:r['m']),
                      requested_rounds=[j['m'] for j, _ in jobs],
                      new_provider_calls=sum(r.get('disposition') == 'NEW_PROVIDER_CALL' for r in rows),
                      unknown_provider_calls=sum(r['status'] == 'PROCESS_FAILURE_NO_RETRY' for r in rows),
                      newly_submitted_rounds=len(pending_jobs),
                      reused_terminal_rounds=sum(r.get('disposition') == 'REUSED_TERMINAL' for r in rows),
                      not_requested_rounds=[r['m'] for r in plan['rounds'] if r['decision'] != 'REQUEST'],
                      process_exits=[dict(pid=p.pid, exitcode=p.exitcode) for p in processes],
                      elapsed_seconds=time.perf_counter()-start,
                      peak_driver_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      sampling=0, compile=0, Q3=0, new_reference_actions=0, automatic_retry=False)
        atomic(output / 'RESULT.json', result); event(output, 'DRIVER_END', status=result['status'])
        return result
    except Exception:
        atomic(output / 'STOP.json', dict(status='INPUT_CONTEXT_OR_PERSISTENCE_ERROR_NO_RETRY',
               error=traceback.format_exc(), partial_rows=rows, elapsed_seconds=time.perf_counter()-start))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


def main():
    global ROOT
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--inputs-commit', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--workers', type=int)
    p.add_argument('--transport-source')
    p.add_argument('--check-input-only', action='store_true')
    a = p.parse_args()
    if a.transport_source:
        ROOT = Path.cwd().resolve()
    source = subprocess.check_output(['git', 'rev-parse', a.source], cwd=ROOT, text=True).strip()
    caller_source = a.transport_source or source
    if read_git(caller_source, CALLER) != Path(__file__).read_bytes():
        raise ValueError('DF candidate signal runner differs from fixed SOURCE')
    input_path = a.input.resolve(); relative = str(input_path.relative_to(ROOT))
    raw = read_git(a.inputs_commit, relative)
    if raw != input_path.read_bytes():
        raise ValueError('Input bytes differ from fixed inputs commit')
    request = json.loads(raw)
    if request['execution_source'] != source:
        raise ValueError('Input execution SOURCE differs')
    if request['plan'].get('terminal_reference') is not None:
        helper = Path(__file__).resolve().parents[1] / DESCRIPTOR_EXTENSION
        if read_git(caller_source, DESCRIPTOR_EXTENSION) != helper.read_bytes():
            raise ValueError('DF terminal reference helper differs from fixed SOURCE')
        if [row['m'] for row in request['plan']['rounds'] if row['decision'] == 'REQUEST'] != [13]:
            raise ValueError('Terminal transport may request only the new last round')
    provider, metadata, b = fixed_runtime(resolve_runtime_path(str(request.get('runtime', {}).get('signal_view',
        ROOT.parent / 'd2-pr-df-ld7-signal-source-fefd-20261116'))))
    import numpy, scipy
    if (numpy.__version__, scipy.__version__) != ('1.26.4', '1.14.1'):
        raise ValueError('Fixed NumPy1.26.4/SciPy1.14.1 runtime required')
    config = request['configuration']
    if set(config) >= {'commit', 'path'}:
        config = json.loads(read_git(config['commit'], config['path']))
        for key in request['configuration'].get('selector', []):
            config = config[key]
    if a.check_input_only:
        jobs = validate_jobs(config, request['plan'], provider, metadata, b)
        print(json.dumps(dict(status='FIXED_SIGNAL_INPUTS_READY',
                              requested_rounds=[job['m'] for job, _ in jobs])))
        return 0
    output = (ROOT / request['output_directory']).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must be in the owned candidate worktree')
    result = execute(config, request['plan'], output, provider, metadata, b, workers=a.workers,tail_descriptor=request.get('tail_descriptor'))
    print(json.dumps(dict(status=result['status'], new_provider_calls=result['new_provider_calls'],
                         reused_terminal_rounds=result['reused_terminal_rounds'])))
    return 0 if result['status'] == 'REQUESTED_ROUNDS_COMPLETE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
