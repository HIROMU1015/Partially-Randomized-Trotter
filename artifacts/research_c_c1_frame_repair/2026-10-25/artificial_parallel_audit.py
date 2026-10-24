"""Artificial subprocess audit only. No snapshot/export/fit/circuit loading."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TEST = ROOT/'tests/management/test_research_c_c1_frame_repair.py'


def main():
    records = []
    started = time.monotonic()
    for legacy in (True, False):
        # The old path is nondeterministic. Seek one reproduced defect within
        # eight fresh artificial processes; fixed checks always cover all eight.
        for trial in range(8):
            child = subprocess.run([sys.executable, str(TEST), '--legacy-probe' if legacy else '--probe'],
                                   capture_output=True, text=True, check=True, cwd=ROOT)
            row = json.loads(child.stdout)
            row['trial'] = trial
            row['bad_frames'] = [r['frame'] for r in row['frames']
                                 if max(r[k] for k in ('raw_vs_loop', 'K_vs_loop', 'raw_vs_serial', 'raw_vs_K')) > 3e-13]
            records.append(row)
            print(json.dumps(row), flush=True)
            if legacy and row['bad_frames']:
                break
            if not legacy and row['bad_frames']:
                raise AssertionError('FIXED_ARTIFICIAL_PARALLEL_MISMATCH')
    old = [row for row in records if row['legacy']]
    fixed = [row for row in records if not row['legacy']]
    result = dict(schema='C1_FRAME_ARTIFICIAL_PARALLEL_AUDIT_v1',records=records,
                  legacy_defect_reproduced=any(row['bad_frames'] for row in old),
                  fixed_processes=len(fixed),fixed_frame_comparisons=8*len(fixed),
                  fixed_all_pass=all(not row['bad_frames'] for row in fixed),
                  check_tolerance=3e-13,check_tolerance_role='Artificial regression only, not a scientific allowance change',
                  elapsed_seconds=time.monotonic()-started,affinity=len(os.sched_getaffinity(0)),
                  molecular_input_or_export_decodes=0,molecular_scientific_calls=0)
    with (HERE/'artificial_parallel_audit.json').open('x') as stream:
        json.dump(result, stream, indent=2);stream.write('\n')
    assert result['fixed_all_pass'] and result['fixed_processes'] == 8


if __name__ == '__main__':
    main()
