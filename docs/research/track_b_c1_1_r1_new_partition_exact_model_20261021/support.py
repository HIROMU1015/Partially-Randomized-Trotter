"""R1 stdlib provenance and primary-failure-preserving logger support."""
from __future__ import annotations

import ast
import dataclasses
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def emit(event_name, details=None):
    # Nested details prevent stage/status keys from colliding with the logger.
    print(json.dumps({'event': event_name, 'details': details or {}}, ensure_ascii=False), flush=True)


class GateFailure(RuntimeError):
    def __init__(self, message, classification):
        super().__init__(message)
        self.classification = classification


def require(value, message, classification="TECHNICAL_INCONCLUSIVE"):
    if not value:
        raise GateFailure(message, classification)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def environment():
    packages = {}
    for name in ['numpy', 'scipy', 'qiskit', 'qiskit-nature', 'qiskit-algorithms',
                 'openfermion', 'openfermionpyscf', 'pyscf', 'numba', 'h5py']:
        dist = importlib.metadata.distribution(name)
        files = {}
        for rel in dist.files or ():
            if str(rel).endswith(('/METADATA', '/RECORD')):
                path = Path(dist.locate_file(rel)).resolve()
                files[str(path)] = digest(path)
        packages[name] = {'version': dist.version, 'metadata_file_sha256': files}
    return {'executable': sys.executable, 'executable_realpath': str(Path(sys.executable).resolve()),
            'executable_sha256': digest(Path(sys.executable).resolve()), 'python_version': sys.version,
            'prefix': sys.prefix, 'packages': packages}


def verify_files(manifest):
    for row in manifest['files']:
        path = ROOT / row['path']
        require(path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                'fixed file changed: ' + row['path'])


def float_record(value, provenance):
    return {'value': value, 'repr': repr(value), 'hex': float(value).hex(),
            'type': type(value).__module__ + '.' + type(value).__name__, 'provenance': provenance}


def run_guarded(action, output, finalizer=lambda: None, event_writer=emit):
    """Preserve primary failures, secondary failures, terminal, and exit mapping."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    wall, cpu = time.monotonic(), time.process_time()
    secondary = []
    state = {'classification': 'TECHNICAL_INCONCLUSIVE', 'diagnosis_complete': False}
    save(output / 'attempt.json', {'attempt': 1, 'retry': 0,
         'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    try:
        state = action(output)
        if state['classification'].startswith('HASH_MISMATCH'):
            save(output / 'failure.json', {'kind': 'HASH_COMPARISON_MISMATCH',
                 'exception_traceback': None, 'diagnostic_outcome': state,
                 'meaning': 'preserved mismatch, not an invented Python exception'})
    except BaseException as exc:
        state = {'classification': getattr(exc, 'classification', 'TECHNICAL_INCONCLUSIVE'), 'diagnosis_complete': False,
                 'primary_error_type': type(exc).__name__, 'primary_error': str(exc)}
        record = {**state, 'traceback': traceback.format_exc()}
        try:
            save(output / 'failure.json', record)
        except BaseException:
            secondary.append({'where': 'primary_failure_save', 'traceback': traceback.format_exc()})
        traceback.print_exc()
    try:
        finalizer()
        state['preservation_after_run'] = 'VERIFIED'
    except BaseException:
        secondary.append({'where': 'preservation', 'traceback': traceback.format_exc()})
        state['original_classification_before_preservation_failure'] = state['classification']
        state['classification'] = 'TECHNICAL_INCONCLUSIVE'
        state['preservation_after_run'] = 'FAILED'
    state.update({'attempts': 1, 'retries': 0, 'mandatory_STOP': True,
                  'wall_seconds': time.monotonic() - wall, 'own_CPU_seconds': time.process_time() - cpu,
                  'peak_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    try:
        event_writer('MANDATORY_STOP', {'terminal': state})
    except BaseException:
        secondary.append({'where': 'terminal_logger', 'traceback': traceback.format_exc()})
    if secondary:
        save(output / 'secondary_failures.json', secondary)
        state['secondary_failure_count'] = len(secondary)
        state['diagnostic_classification_before_secondary_failure'] = state['classification']
        state['classification'] = 'TECHNICAL_INCONCLUSIVE'
    exit_code = (1 if state['classification'] == 'TECHNICAL_INCONCLUSIVE' else
                 0 if state['classification'] == 'R1_LIMITED_REANALYSIS_COMPLETE' else 2)
    state['expected_process_exit'] = exit_code
    save(output / 'terminal.json', state)
    save(output / 'runner_output_inventory.json', [
        {'path': str(p.relative_to(output)), 'bytes': p.stat().st_size, 'sha256': digest(p)}
        for p in sorted(output.rglob('*')) if p.is_file()])
    return exit_code
