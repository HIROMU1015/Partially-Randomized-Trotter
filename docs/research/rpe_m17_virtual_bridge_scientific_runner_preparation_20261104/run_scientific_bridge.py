"""Default metadata only. --runtime-check is artificial; --run is separately permitted m17 science."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'docs/research/rpe_m17_virtual_measurement_bridge_preparation_20261104'))
import run_virtual_bridge as prior
from runtime_policy import Checkpoints, write_json

SOURCE_FILES = ['run_scientific_bridge.py', 'runtime_policy.py', 'runtime_checks.py', 'observe_runtime.py', 'bridge_pipeline.py']
FIXED_HELPERS = prior.SOURCES + [
    ('728c90a2d4d4335ef5b7237808c15b182e694bb9', 'docs/research/rpe_m17_virtual_measurement_bridge_preparation_20261104/'+name)
    for name in ['virtual_bridge_adapter.py', 'run_virtual_bridge.py']
] + [('e2fd35f1dcc73b697ac85661ef91f21082fd2136',
       'docs/research/rpe_m17_openblas_kernel_bound_preflight_20261104/kernel_bounds.py')]
STATUS = {'U_exec_upper': None, 'RPE_tolerance_certified': False, 'scientific_status': 'UNDETERMINED',
          'original_numeric_diagnostic_status': 'INCONSISTENT', 'old_u_num_bound': None,
          'fixed_PCG64_IID_certified': False, 'mandatory_STOP': True}


def source_metadata(require_frozen=True):
    hashes = {}
    for name in SOURCE_FILES:
        path = HERE/name
        raw = path.read_bytes()
        if require_frozen and raw != prior.fixed_bytes('HEAD', path.relative_to(ROOT).as_posix()):
            raise ValueError('Freeze numerical SOURCE before --run')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    for commit, path in FIXED_HELPERS:
        if (ROOT/path).read_bytes() != prior.fixed_bytes(commit, path):
            raise ValueError('Fixed helper changed: '+path)
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'qiskit', 'python-flint')}
    if versions != {'numpy': '1.26.4', 'qiskit': '1.3.0', 'python-flint': '0.9.0'}:
        raise ValueError('Pinned numerical dependency versions required')
    import flint
    if flint.__FLINT_VERSION__ != '3.6.0' or sys.version.split()[0] != '3.11.0rc1':
        raise ValueError('Pinned FLINT/Python required')
    return {'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'code_sha256': hashes, 'versions': versions, 'frozen_required': require_frozen}


def result(output, record):
    write_json(Path(output)/'RESULT.json', record)


def worker(mode, output):
    import bridge_pipeline as pipeline
    import kernel_bounds as kernel
    from runtime_checks import binary_profile, probe
    from flint import ctx
    output = Path(output)
    record = json.loads((output/'RESULT.json').read_text())
    checkpoint = Checkpoints(output)
    try:
        if mode == 'science' and source_metadata() != record['SOURCE']:
            raise ValueError('Worker SOURCE differs from frozen supervisor SOURCE')
        profile = binary_profile()
        if profile != kernel.PROFILE:
            raise ValueError('Current numerical binary/core profile mismatch')
        record['binary_profile'] = profile
        runtime = probe(checkpoint)
        contract = kernel.selected_path_contract(profile, rows=70, columns=70, layout='C',
                                                  mxcsr_values=runtime['observed_mxcsr'])
        if contract['bounds'] is None:
            raise ValueError('Kernel contract unresolved')
        if mode == 'science':
            checkpoint('before_context')
            record['stage'] = 'saved_input_load'
            record['scientific_calls']['saved_input_load_started'] = 1
            result(output, record)
            record['inputs'] = prior.metadata()
            data = pipeline.load_fixed_data()
            record['fixed_literals'] = data['literals']
            record['stage'] = 'numeric_context'
            record['scientific_calls']['numeric_context_started'] = 1
            result(output, record)
            numeric, trig = pipeline.construct_context(data, output)
            record['component_ids'] = [item['id'] for item in data['export']['components']]
            record['trig64_hex'] = [[value.hex() for value in pair] for pair in trig]
            checkpoint('after_context')
            ctx.prec = pipeline.BITS
            ctx.threads = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1
            record['resources'] = {'FLINT_threads': ctx.threads, 'CPU_affinity_count': ctx.threads,
                'BLAS_environment': {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}
            checkpoint('before_bound')
            record['stage'] = 'bound_96'
            record['scientific_calls']['bound_pass_started'] = 1
            result(output, record)
            bound = pipeline.evaluate(data, numeric, trig)
            checkpoint('after_bound')
            record['bound'] = bound
            record['scientific_calls']['bound_pass_completed'] = 1
        checkpoint('complete')
        record['execution_status'] = 'COMPLETED_MANDATORY_STOP' if mode == 'science' else 'COMPLETED_RUNTIME_CHECK_MANDATORY_STOP'
        record['stage'] = 'completed'
    except BaseException as exc:
        record['execution_status'], record['stage'] = 'INTERRUPTED_OR_INVALID_MANDATORY_STOP', 'stopped'
        record['error'] = {'type': type(exc).__name__, 'message': str(exc).replace(str(ROOT), '<repository>')}
        raise
    finally:
        result(output, record)


def cleanup_owned(process, output):
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=10)  # Shutdown timeout, never a scientific wall-time limit.
    except subprocess.TimeoutExpired:
        runtime_file = Path(output)/'runtime.json'
        state = json.loads(runtime_file.read_text()) if runtime_file.exists() else {}
        pid = state.get('child_pid', 0)
        if pid:
            try:
                if os.getsid(pid) == process.pid:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.kill()
        process.wait()


def execute(mode, output):
    output = Path(output).resolve()
    if mode == 'science' and not output.is_relative_to(ROOT/'artifacts/rpe_m17_virtual_measurement_bridge'):
        raise ValueError('Use a fresh directory under artifacts/rpe_m17_virtual_measurement_bridge')
    output.mkdir(parents=True, exist_ok=False)
    started, process = time.monotonic(), None
    record = {'execution_status': 'STARTED', **STATUS, 'mode': mode, 'stage': 'SOURCE',
              'scientific_calls': dict(saved_input_load_started=0, numeric_context_started=0,
                                       bound_pass_started=0, bound_pass_completed=0),
              'sampling_RNG_replay_circuit_build_compile_cost_calls': 0}
    result(output, record)
    try:
        record['SOURCE'] = source_metadata(require_frozen=mode == 'science')
        result(output, record)
        if shutil.which('gdb') is None:
            raise ValueError('GDB runtime observer unavailable')
        environment = dict(os.environ, RPE_BRIDGE_OBSERVER_OUTPUT=str(output))
        with (output/'observer.log').open('wb') as log:
            process = subprocess.Popen(['gdb', '-q', '-batch', '-x', str(HERE/'observe_runtime.py'), '--args',
                sys.executable, str(HERE/'run_scientific_bridge.py'), '--child', mode, '--output-dir', str(output)],
                env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = process.wait()
        record = json.loads((output/'RESULT.json').read_text())
        state = json.loads((output/'runtime.json').read_text()) if (output/'runtime.json').exists() else {}
        record['runtime'] = state
        if code != 0 or state.get('status') != 'RUNTIME_CONDITIONS_CONFIRMED' or state.get('phase') != 'complete':
            raise ValueError('Runtime observation failed; inspect observer.log/runtime.json')
        expected = 'COMPLETED_MANDATORY_STOP' if mode == 'science' else 'COMPLETED_RUNTIME_CHECK_MANDATORY_STOP'
        if record['execution_status'] != expected:
            raise ValueError('Monitored worker incomplete')
    except BaseException as exc:
        if process is not None:
            cleanup_owned(process, output)
        record = json.loads((output/'RESULT.json').read_text())
        record['execution_status'], record['stage'] = 'INTERRUPTED_OR_INVALID_MANDATORY_STOP', 'stopped'
        if 'error' in record:
            record['worker_error'] = record['error']
        record['error'] = {'type': type(exc).__name__, 'message': str(exc).replace(str(ROOT), '<repository>')}
        if 'bound' in record:
            record['bound']['local_model_II_status'] = 'UNDETERMINED_RUNTIME'
            for axis in record['bound']['axes'].values():
                axis.update(U_virtual_IID_upper=None, U_exec_upper=None, decision='UNDETERMINED')
        raise
    finally:
        if (output/'runtime.json').exists():
            record['runtime'] = json.loads((output/'runtime.json').read_text())
        log = output/'observer.log'
        if log.exists() and log.stat().st_size > 65536:
            with log.open('rb') as handle:
                handle.seek(-65536, 2)
                tail = handle.read()
            log.write_bytes(b'[Earlier diagnostic text omitted; runtime.json retains complete checks/counts]\n'+tail)
        record['wall_seconds'] = time.monotonic()-started
        record['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
                                  if p.is_file() and p.name != 'RESULT.json'}
        result(output, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--runtime-check', action='store_true')
    mode.add_argument('--child', choices=('science', 'runtime'), help=argparse.SUPPRESS)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    if args.child:
        if args.output_dir is None: parser.error('--child requires --output-dir')
        worker(args.child, args.output_dir)
    elif args.run or args.runtime_check:
        if args.output_dir is None: parser.error('Execution requires fresh --output-dir')
        execute('science' if args.run else 'runtime', args.output_dir)
    else:
        print(json.dumps({'status': 'PREPARED_NOT_SCIENTIFICALLY_EXECUTED', **STATUS,
                          'inputs': prior.metadata(), 'precision_bits': [96], 'max_bound_passes': 1,
                          'real_numerical_calls': 0, 'runtime_conditions': 'CHECKED_INSIDE_OBSERVED_CHILD_AT_EXECUTION'}, indent=2))


if __name__ == '__main__':
    main()
