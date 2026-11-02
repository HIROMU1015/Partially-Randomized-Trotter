"""Isolated saved/artificial IO profile, never a G10 science execution.

No scientific inputs are evaluated. Temporary outputs have no production
marker, success STOP or completion token. The frozen caps/runtime are used.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

from g10_json_compatibility_fixtures import large_typed
from trottertracks.algorithm_codesign.g10_saved import serial
from trottertracks.algorithm_codesign.g10_io import IOBudgetGuard, OutputSession, memory_snapshot

ROOT = Path(__file__).resolve().parents[3]
PREP = ROOT/'artifacts/track_b_g10_key_compatibility_preparation/2026-10-10/v3'
SAVED = ROOT/'artifacts/track_b_g10_degree_result/2026-10-10/v1/result_v1.json'
EXPECTED = 'b62695c19964a5a121b965c39142427bfa8048efad9bce05f3221494bb14dfe1'


def run(mode):
    c = json.loads((PREP/'contract_v3.json').read_text())
    guard = IOBudgetGuard(c['caps'])
    snapshots, identity, reason = [], None, None

    def snap(phase):
        guard.check()
        snapshots.append({'phase': phase, **guard.usage(), **memory_snapshot()})

    try:
        with guard:
            snap('start')
            if mode.endswith('saved'):
                raw = SAVED.read_bytes()
                if hashlib.sha256(raw).hexdigest() != EXPECTED:
                    raise PermissionError('saved technical JSON changed')
                value = json.loads(raw)
                if value['status'] != 'G10_TECHNICAL_INCONCLUSIVE' or value['prefix_rows_usable_for_final_research_decision']:
                    raise PermissionError('technical-prefix scope mismatch')
                del raw
            else:
                value = large_typed()
            gc.collect()
            snap('decoded_or_artificial_tree_only')
            with tempfile.TemporaryDirectory(prefix='g10-v3-nonscience-IO-') as directory:
                if mode.startswith('legacy'):
                    copied = serial(value)
                    snap('old_recursive_serial_copy')
                    text = json.dumps(copied, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
                    snap('old_json_list_join_string')
                    encoded = text.encode()
                    snap('old_utf8_string_coexistence')
                    if len(encoded) > c['caps']['output_bytes']:
                        raise RuntimeError('unchanged output cap')
                    with (Path(directory)/'artificial-legacy.json').open('xb') as stream:
                        stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
                    guard.check()
                    identity = {'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}
                else:
                    io = OutputSession(directory, c['caps']['output_bytes'], guard,
                                       {'scope': 'saved/artificial IO only'})
                    identity = io.write_result(value)
                    snap('stream_encode_write_flush_close_disk_verify')
                if (Path(directory)/'COMPLETED.v2').exists() or (Path(directory)/'one_shot_consumed.json').exists():
                    raise AssertionError('IO profile must not create production authority/completion')
                if mode.endswith('saved') and (identity['bytes'] != 66842493 or identity['sha256'] != EXPECTED):
                    raise AssertionError('saved JSON bytes/hash differ')
            snap('done')
    except Exception as exc:
        reason = type(exc).__name__+': '+str(exc)[:1024]
    forbidden = [name for name in sys.modules if name.startswith(('numpy', 'mpmath', 'pygridsynth'))]
    if forbidden:
        raise AssertionError('science module imported in IO-only profile')
    return {'mode': mode, 'passed': reason is None, 'technical_reason': reason,
            'identity': identity, 'snapshots': snapshots, 'resource': guard.usage(),
            'python': platform.python_version(), 'executable': os.path.realpath(sys.executable),
            'frozen_caps': c['caps'], 'caps_modified_for_diagnostic': False,
            'typed_fixture_count': 50000 if mode.endswith('typed') else None,
            'production_runner_invocations': 0, 'production_marker_created': False,
            'science_synthesis_matrix_sampling_LP': 0, 'scientific_imports': forbidden,
            'original_live_G10_heap_recreated': False, 'production_completion_proven': False,
            'retries': 0, 'mandatory_STOP': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('legacy_saved', 'stream_saved', 'legacy_typed', 'stream_typed'), required=True)
    report = run(parser.parse_args().mode)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report['passed'] else 1)
