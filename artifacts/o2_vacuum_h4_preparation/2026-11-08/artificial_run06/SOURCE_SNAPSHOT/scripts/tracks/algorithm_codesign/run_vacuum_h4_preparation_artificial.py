#!/usr/bin/env python3
"""New adapter/qualification only, invented records; NEVER real H4 activation."""
import argparse
from fractions import Fraction as F
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/tracks/algorithm_codesign'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from trottertracks.algorithm_codesign import vacuum_h4_execution as exe
    import numpy,qiskit,mpmath
    own=str(Path(__file__).relative_to(ROOT))
    names=tuple(dict.fromkeys((*exe.SOURCE_FILES,own,
        'tests/tracks/algorithm_codesign/test_vacuum_h4_preparation.py',
        'tests/tracks/algorithm_codesign/test_a1_engine_connection.py')))
    sha={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names}
    # Preserve every new source version on technical failure; old helpers are
    # content-pinned in SOURCE and remain unchanged in the Git parent.
    for name in names:
        if 'vacuum_h4' in name:
            dest=args.output/'SOURCE_SNAPSHOT'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes((ROOT/name).read_bytes())
    source=dict(scope='INVENTED_SAVED_SCHEMA_AND_OPERATOR_QUALIFICATION_ONLY',
        parent='0edc0fa5ad4656fdd492ca6c96ade4f77cadf412',sha256=sha,
        existing_evidence_reused='14 correctness / 128 artificial compiled rows; not rerun',
        environment=dict(python=platform.python_version(),numpy=numpy.__version__,
            qiskit=qiskit.__version__,mpmath=mpmath.__version__,executable=sys.executable),
        resource_start=dict(cpu_count=os.cpu_count(),loadavg=os.getloadavg(),
            MemAvailable=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))),
        scientific_input_files=[],real_N1_loads=0,state_Born_sampling_shots=0)
    (args.output/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    suite=unittest.defaultTestLoader.loadTestsFromName('test_vacuum_h4_preparation.PreparationTests')
    stream=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    import test_vacuum_h4_preparation as tests
    test_result=dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        elapsed_seconds=time.perf_counter()-start,evidence=tests.EVIDENCE,
        prior_suite_rerun=False,uniform_error_method='explicit IEEE operator contraction + interval local matrices')
    (args.output/'TEST_RESULTS.json').write_text(json.dumps(test_result,indent=2)+'\n')
    status='CORRECTNESS_FAILED';rows=[];counts={k:0 for k in exe.CAPS}
    def call(name,function):
        counts[name]+=1
        if counts[name]>exe.CAPS[name]:raise ValueError('artificial finite call exceeded')
        with (args.output/'CALLS.jsonl').open('a') as f:
            f.write(json.dumps(dict(event='CALL_START',call=name,index=counts[name]))+'\n')
        value=function()
        with (args.output/'CALLS.jsonl').open('a') as f:
            f.write(json.dumps(dict(event='CALL_RETURN',call=name,index=counts[name]))+'\n')
        return value
    if result.wasSuccessful():
        try:
            records=tests.fixture()
            (args.output/'ARTIFICIAL_INPUTS.json').write_text(json.dumps(records,indent=2)+'\n')
            report=exe.compare_records(records,args.output,call)
            rows=json.loads((args.output/'RESULT_ROWS.json').read_text())
            for r in rows:
                # Regression threshold on INVENTED examples only, not a new
                # scientific numerical allowance or H4 acceptance threshold.
                if float(F(r['combined_coordinate_difference_upper']))>=1e-7:
                    raise ValueError('artificial operator qualification regression')
            status='PASS';(args.output/'BENCHMARK_RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
        except Exception:
            status='TECHNICAL_BENCHMARK_FAILED';stream.write(traceback.format_exc())
    (args.output/'RUN.txt').write_text(stream.getvalue())
    (args.output/'CALL_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n')
    stop=dict(status='MANDATORY_STOP',technical_status=status,scientific_execution_started=False,
        automatic_next_stage=False,elapsed_seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.output/'STOP.json').write_text(json.dumps(stop,indent=2)+'\n')
    print(json.dumps(dict(status=status,tests=result.testsRun,rows=len(rows),counts=counts,
        elapsed_seconds=stop['elapsed_seconds'],peak_rss_kib=stop['peak_rss_kib'])))
    return 0 if status=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
