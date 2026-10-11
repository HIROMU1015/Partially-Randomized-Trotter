#!/usr/bin/env python3
"""Export exact old-host DF preparations; no signal, circuit build or transpile.

The two-file delivery packet can be used without checking out a new branch in
the old research repository. --codec points to the delivered codec module.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--codec', type=Path)
    parser.add_argument('--export-old-preparations', action='store_true', required=True)
    args = parser.parse_args()
    root = args.repository.resolve(); output = args.output.absolute()
    home = Path.home().resolve()
    if (not output.resolve().is_relative_to(home) or output.exists()
            or any(p.is_symlink() for p in (output, *output.parents))):
        raise ValueError('EXPORT_FRESH_HOME_OUTPUT_REQUIRED')
    # Process-local limits and environment only. The old venv is never modified.
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 2**20, 16 * 2**20))
    signal.alarm(120); os.nice(10)
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS', 'RAYON_NUM_THREADS',
                 'QISKIT_NUM_PROCS', 'BLIS_NUM_THREADS', 'GOTO_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    os.environ['QISKIT_PARALLEL'] = 'FALSE'
    output.mkdir(parents=True, exist_ok=False)
    for name in ('tmp', 'cache', 'mpl', 'numba'):
        (output / name).mkdir()
    for name in ('TMPDIR', 'TEMP', 'TMP'):
        os.environ[name] = str(output / 'tmp')
    os.environ.update(XDG_CACHE_HOME=str(output / 'cache'), MPLCONFIGDIR=str(output / 'mpl'),
                      NUMBA_CACHE_DIR=str(output / 'numba'))
    codec = args.codec or root / 'src/trottertracks/resource_applicability/h6_frozen_preparation_v1.py'
    spec = importlib.util.spec_from_file_location('h6_frozen_preparation_v1', codec)
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    sys.path.insert(0, str(root / 'src'))
    started = time.monotonic(); reason = None; receipt = None
    try:
        old = json.loads((root / frozen.OLD_MANIFEST).read_bytes())
        hashes = old['source_hashes']
        if len(hashes) != 224:
            raise ValueError('EXPORT_SCIENCE_CLOSURE')
        for path, sha in hashes.items():
            blob = subprocess.check_output(['git', '-c', 'maintenance.auto=false', '-c', 'gc.auto=0',
                                            'show', frozen.SCIENCE_SOURCE + ':' + path], cwd=root)
            if frozen.hashlib.sha256(blob).hexdigest() != sha or frozen.file_hash(root / path) != sha:
                raise ValueError('EXPORT_FIXED_SOURCE_CHANGED:' + path)
        from trottertracks.resource_applicability.h6_matched_contract_v1 import environment, verify_parent
        if environment() != old['environment'] or verify_parent(root) != old['input_identity']:
            raise ValueError('EXPORT_OLD_INPUT_ENVIRONMENT')
        from trottertracks.resource_applicability.ax2b_h6_saved_completion_loader_v1 import load_h6_snapshot
        from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
        identity = old['input_identity']
        ham, *_ = load_h6_snapshot(root / identity['snapshot_path'],
                                  json.loads((root / identity['snapshot_receipt_path']).read_bytes()),
                                  json.loads((root / identity['df_receipt_path']).read_bytes()))
        expected = frozen.expected_preparations(root); preparations = {}
        for name in sorted(expected):
            policy, prefix = name.split(':p')
            preparations[name] = (_prepare_discard if policy == 'discard' else _prepare)(ham, int(prefix))
        # A changed old environment cannot produce an accepted archive merely
        # by declaring old hashes. Compare every old preparation before export.
        frozen.verify_preparations(preparations, expected, identity['hamiltonian_hash'])
        codec = frozen.encode_preparations(preparations)
        restored = frozen.decode_preparations(codec)
        frozen.verify_preparations(restored, expected, identity['hamiltonian_hash'])
        partial = frozen.verify_partial_450(root, restored)
        receipt = dict(status='OLD_PREPARATIONS_EXPORTED', all_seven_preparations_exact=True,
                       partial_450=partial, compile_calls=0, signal_calls=0, input_generation=0,
                       package_install_or_change=False, wall_seconds=time.monotonic()-started,
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       old_source_224_checked=True, fixed_input_checked=True)
        receipt.update(exporter_sha256=frozen.file_hash(Path(__file__)),
                       codec_sha256=frozen.file_hash(Path(frozen.__file__)))
        binding = dict(science_source=frozen.SCIENCE_SOURCE, science_source_hashes=hashes,
                       input_identity=identity, compiler=old['plan']['compiler_proposed'],
                       old_prepared_sha256=frozen.file_hash(root / frozen.OLD_RUN / 'signal/prepared_representation.json'),
                       old_partial_450_sha256=frozen.file_hash(root / frozen.OLD_RUN / 'cost_exploration_0450/trajectory.json'),
                       environment=old['environment'])
        archive = dict(schema=frozen.SCHEMA, binding=binding, codec=codec,
                       codec_digest=frozen.digest(codec), export_receipt=receipt)
        payload = frozen.canonical(archive) + b'\n'
        if len(payload) > frozen.MAX_BYTES:
            raise ValueError('EXPORT_ARCHIVE_CAP')
        path = output / 'h6-frozen-preparations-v1.json'
        with path.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        with (output / 'SHA256SUMS').open('x') as stream:
            stream.write(frozen.file_hash(path) + '  ' + path.name + '\n')
        print('EXPORTED:', path, 'bytes=', len(payload), 'sha256=', frozen.file_hash(path))
    except Exception as exc:
        reason = type(exc).__name__ + ':' + str(exc)[:512]
        print('EXPORT_STOP:', reason, file=sys.stderr)
    with (output / 'export_terminal.json').open('x') as stream:
        json.dump(dict(status='EXPORT_COMPLETE' if reason is None else 'EXPORT_STOP', reason=reason,
                       receipt=receipt, scientific_runner_started=False, grant_issued=False,
                       wall_seconds=time.monotonic()-started), stream, indent=2)
        stream.write('\n')
    return 0 if reason is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
