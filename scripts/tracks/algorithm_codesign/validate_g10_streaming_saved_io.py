"""Non-science subprocess memory check: saved technical JSON or artificial types.

No G10 runner/science module imports. Pure old serial() is extracted via AST.
Each specified mode is run once by the preparation orchestrator; no retries.
The output payload exists only in a TemporaryDirectory and is never scientific.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
import platform
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from trottertracks.algorithm_codesign.g10_io import IOBudgetGuard, OutputSession, memory_snapshot

ROOT = Path(__file__).resolve().parents[3]
RESULT = ROOT/'artifacts/track_b_g10_degree_result/2026-10-10/v1/result_v1.json'
EXPECTED = 'b62695c19964a5a121b965c39142427bfa8048efad9bce05f3221494bb14dfe1'


def old_serial():
    tree = ast.parse((ROOT/'src/trottertracks/algorithm_codesign/g10_saved.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'serial')
    space = {'F': Fraction}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<old-pure-serial>', 'exec'), space)
    return space['serial']


def run(mode):
    contract = json.loads((ROOT/'artifacts/track_b_g10_degree_preparation/2026-10-10/contract_v1.json').read_text())
    caps = dict(contract['caps'], wall_seconds=60, cpu_seconds=60)
    guard = IOBudgetGuard(caps)
    snapshots = []
    def snap(phase):
        guard.check()
        snapshots.append({'phase': phase, **guard.usage(), **memory_snapshot()})
    with guard:
        snap('start')
        if mode.endswith('saved'):
            raw = RESULT.read_bytes()
            if hashlib.sha256(raw).hexdigest() != EXPECTED:
                raise PermissionError('saved technical JSON identity mismatch')
            value = json.loads(raw)
            if value['status'] != 'G10_TECHNICAL_INCONCLUSIVE' or value['prefix_rows_usable_for_final_research_decision']:
                raise PermissionError('technical-prefix scope mismatch')
            del raw
        else:
            # Artificial serialization fixture; no event/provider/circuit method.
            value = {'artificial': [{'fraction': Fraction(i-25000, 7),
                      'tuple': ('😀漢', i, None), 'flags': [True, False]}
                      for i in range(50000)]}
        gc.collect()
        snap('decoded_or_artificial_tree_only')
        with tempfile.TemporaryDirectory(prefix='g10-io-only-') as directory:
            if mode.startswith('legacy'):
                copied = old_serial()(value)
                snap('old_container_copy')
                raw = json.dumps(copied, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
                snap('old_dumps_with_chunk_list_join')
                encoded = raw.encode()
                snap('old_utf8_string_coexistence')
                if len(encoded) > caps['output_bytes']:
                    raise RuntimeError('unchanged output cap')
                identity = {'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}
            else:
                io = OutputSession(directory, caps['output_bytes'], guard, {'scope': 'I/O-only'})
                identity = io.write_result(value)
                if (Path(directory)/'COMPLETED.v2').exists():
                    raise AssertionError('I/O-only check may not commit a scientific result')
                snap('new_stream_write_flush_close_verify')
            if mode.endswith('saved') and (identity['sha256'] != EXPECTED or identity['bytes'] != RESULT.stat().st_size):
                raise AssertionError('saved JSON bytes not identical')
        snap('done')
    print(json.dumps({'mode': mode, 'passed': True, 'identity': identity,
          'snapshots': snapshots, 'resource': guard.usage(),
          'python': platform.python_version(), 'executable': os.path.realpath(sys.executable),
          'diagnostic_caps': {'RSS_MiB': 512, 'AS_MiB': 1536, 'wall_seconds': 60, 'CPU_seconds': 60},
          'scientific_runner_invocations': 0, 'science_synthesis_matrix_sampling_LP': 0,
          'original_live_G10_heap_recreated': False, 'full_G10_completion_proven': False,
          'mandatory_STOP': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('legacy_saved', 'stream_saved', 'legacy_typed', 'stream_typed'), required=True)
    run(parser.parse_args().mode)
