"""Recheck only failed Q3 metadata cells after placing original guard blobs."""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import check_saved as C

if not os.environ.get('PBS_JOBID'):
    raise ValueError('PBS allocation required')
C.OUTPUT = C.ROOT / 'checks02'
C.OUTPUT.mkdir(exist_ok=False)
started = time.monotonic()
with ThreadPoolExecutor(max_workers=2) as pool:
    rows = list(pool.map(C.execute, ['FULL_PF', 'TRUNCATED_PF', 'PR_PAULI', 'PR_DF']))
summary = dict(schema='RPE_PBS_SAVED_Q3_RECHECK_V1', pbs_job_id=os.environ['PBS_JOBID'],
               hostname=os.uname().nodename, q3_dry_checks=rows,
               same_clock_duration_seconds=time.monotonic()-started,
               scientific_evaluation_calls=0, compile_calls=0, q3_execution_calls=0,
               reason='Place original historical Git blobs required by unchanged metadata guards; no SOURCE or input edit',
               scope='SAVED_REAL_INPUT_METADATA_ONLY')
(C.OUTPUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary), flush=True)
