#!/usr/bin/env python3
"""Fresh S1 only; S2 must use its recorded PASS, never retry S1."""
import argparse
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
OUTPUT = ROOT/'artifacts/track_d_r16_limited_validation/2026-11-10/run01'
SOURCE_FILES = ('src/trottertracks/track_d_r16_validation.py',
    'scripts/run_track_d_r16_validation.py', 'src/trottertracks/c_pr_aware_df_stage_b.py',
    'src/trottertracks/c1_stage3_connection.py', 'src/trottertracks/c1_stage3_reference.py',
    'src/trotterlib/rte.py', 'src/trottertracks/track_d_b1_c1.py',
    'src/trottertracks/track_d_original_h_accuracy.py',
    'src/trottertracks/c1_stage3_accuracy.py', 'src/trotterlib/rpe_resource_accounting.py')


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-s1', action='store_true', required=True)
    parser.parse_args()
    from trottertracks import track_d_r16_validation as v
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT).decode().strip()
    hashes = {}
    for name in SOURCE_FILES:
        data = (ROOT/name).read_bytes()
        if data != subprocess.check_output(['git','show',source+':'+name],cwd=ROOT):
            raise ValueError('UNCOMMITTED_SOURCE:'+name)
        hashes[name] = v.digest(data)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    save(OUTPUT/'SOURCE_S1.json', dict(execution_source=source, files=hashes))
    save(OUTPUT/'seed_plan.json', v.seed_plan())
    tick = time.monotonic()
    try:
        from threadpoolctl import threadpool_limits
        workers = min(13, len(os.sched_getaffinity(0)))
        # Thirteen independent 36x36 operations; one BLAS thread per task
        # avoids nesting 13 full-host BLAS teams for these tiny matrices.
        with threadpool_limits(limits=1):
            c = v.prepare(ROOT)
            save(OUTPUT/'input_identity.json', c['records'])
            save(OUTPUT/'r16_law.json', dict(law=c['law'],sha256=c['law_sha256'],gates=c['gates']))
            result = v.run_s1(c, source, workers)
        for path, row in c['records'].items():
            data = subprocess.check_output(['git','show',row['commit']+':'+path],cwd=ROOT)
            if v.digest(data) != row['sha256']:
                raise ValueError('POST_INPUT_HASH:'+path)
        if hashes != {n:v.digest((ROOT/n).read_bytes()) for n in hashes}:
            raise ValueError('POST_SOURCE_CHANGED')
        result.update(source_input_preserved=True, wall_seconds=time.monotonic()-tick,
                      round_workers=workers, sampling_calls=0, compile_calls=0)
        save(OUTPUT/'S1_result.json', result)
        print(result['status'], flush=True)
        if not result['S2_allowed']:
            save(OUTPUT/'STOP.json', dict(mandatory_stop=True, reason=result['status'],
                S2_started=False, retry=0, resume=0))
        return 0 if result['S2_allowed'] else 1
    except BaseException as error:
        save(OUTPUT/'S1_failure.json',dict(type=type(error).__name__,error=str(error).replace(str(ROOT),'<task>')))
        save(OUTPUT/'STOP.json', dict(mandatory_stop=True, reason='S1_TECHNICAL_FAILURE',
             S2_started=False, retry=0, resume=0))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
