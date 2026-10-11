#!/usr/bin/env python3
"""One bounded exact archive/partial-450 diagnostic; never start a cost runner."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-exact-preparations', action='store_true', required=True)
    args = parser.parse_args(); output = args.output.absolute(); archive = args.archive.absolute()
    for path in (output, archive):
        if (not path.is_relative_to('/home/AbeHiromu')
                or any(p.is_symlink() for p in (path, *path.parents))):
            raise ValueError('VERIFY_PRIVATE_HOME_PATH')
    if output.exists():
        raise ValueError('VERIFY_FRESH_OUTPUT')
    os.sched_setaffinity(0, {20}); os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 2**20, 16 * 2**20))
    signal.alarm(120)
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
    from trottertracks.resource_applicability.h6_parallel_contract_v1 import BASE, sources, phase_a_receipt
    from trottertracks.resource_applicability.h6_matched_reuse_v3 import committed_hashes
    from trottertracks.resource_applicability.h6_server_contract_v1 import verify_bindings
    from trottertracks.resource_applicability import h6_frozen_preparation_v1 as frozen
    begun = time.monotonic(); reason = None; receipt = None
    try:
        base = json.loads((ROOT / BASE).read_bytes())
        source_hashes = {p: frozen.file_hash(ROOT / p) for p in sources(ROOT)}
        committed_hashes(ROOT, args.source_commit, source_hashes)
        verify_bindings(ROOT, base)
        preparations, saved = frozen.verify_archive(ROOT, archive, sha256=args.sha256)
        old_receipt = saved['export_receipt']
        if (old_receipt.get('status') != 'OLD_PREPARATIONS_EXPORTED'
                or old_receipt.get('all_seven_preparations_exact') is not True
                or old_receipt.get('partial_450', {}).get('exact_events_matched') is not True
                or old_receipt.get('exporter_sha256') != frozen.file_hash(ROOT / 'scripts/resource_applicability/export_track_a_h6_frozen_preparations_v1.py')
                or old_receipt.get('codec_sha256') != frozen.file_hash(Path(frozen.__file__))
                or old_receipt['partial_450']['calls'] != {'trajectory': 1, 'occurrence': 8}
                or any(old_receipt.get(k) != 0 for k in ('compile_calls', 'signal_calls', 'input_generation'))):
            raise ValueError('VERIFY_OLD_EXPORT_RECEIPT')
        partial = frozen.verify_partial_450(ROOT, preparations)
        comparison = frozen.verify_phase_a_wrappers(preparations, base)
        with (output / 'phase_a_wrapper_comparison_v1.json').open('x') as stream:
            json.dump(comparison, stream, indent=2); stream.write('\n')
        if not comparison['exact']:
            raise ValueError('VERIFY_PHASE_A_447_WRAPPER_MISMATCH_REMEASUREMENT_REQUIRED')
        if frozen.file_hash(archive) != args.sha256:
            raise ValueError('VERIFY_ARCHIVE_CHANGED')
        committed_hashes(ROOT, args.source_commit, source_hashes)
        receipt = dict(schema='h6_server_frozen_preparation_receipt_v1',
                       status='PASS_EXACT_PREPARATIONS_AND_PARTIAL_450',
                       source_commit=args.source_commit, source_hashes_digest=frozen.digest(source_hashes),
                       archive_sha256=args.sha256, archive_bytes=archive.stat().st_size,
                       profile_sha256=base['profile']['sha256'],
                       input_identity_digest=frozen.digest(base['input_identity']),
                       all_seven_preparations_exact=True, partial_450_exact=True,
                       phase_a_full_wrapper_exact=True,
                       phase_a_receipt_digest=frozen.digest(phase_a_receipt(base, require=True)),
                       partial_450_digest=partial['event_digest'], compile_calls=0, signal_calls=0,
                       input_generation=0, server_diagnostic_calls=partial['calls'],
                       old_host_diagnostic_calls=old_receipt['partial_450']['calls'],
                       preparation_hashes={k: p.preparation_hash for k, p in preparations.items()},
                       wall_seconds=time.monotonic()-begun,
                       peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                       approved=False, runtime_authorization=False, runner_started=False)
        with (output / 'frozen_preparation_receipt_v1.json').open('x') as stream:
            json.dump(receipt, stream, indent=2); stream.write('\n')
        print('PASS_EXACT_PREPARATIONS_AND_PARTIAL_450; no runner started')
    except Exception as exc:
        reason = type(exc).__name__ + ':' + str(exc)[:512]
        print('VERIFY_STOP:' + reason, file=sys.stderr)
    with (output / 'verification_terminal.json').open('x') as stream:
        json.dump(dict(status='VERIFY_COMPLETE' if reason is None else 'VERIFY_STOP', reason=reason,
                       receipt_created=receipt is not None, runner_started=False,
                       wall_seconds=time.monotonic()-begun), stream, indent=2)
        stream.write('\n')
    return 0 if reason is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
