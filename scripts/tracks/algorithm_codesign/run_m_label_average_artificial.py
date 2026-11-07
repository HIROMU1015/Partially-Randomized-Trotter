#!/usr/bin/env python3
"""Run only the new M artificial tests and save their evidence to fresh output."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    paths=['src/trottertracks/algorithm_codesign/m_label_average.py',
           'tests/tracks/algorithm_codesign/test_m_label_average.py',
           str(Path(__file__).relative_to(ROOT)),
           'src/trottertracks/algorithm_codesign/a2_lcu_access.py',
           'src/trottertracks/algorithm_codesign/a1_control_variates.py',
           'src/trotterlib/rpe_hadamard_interrogation.py',
           'src/trottertracks/pr_dictionary_validation/gates.py']
    import numpy,qiskit
    source=dict(scope='INVENTED_1_2_QUBIT_EXHAUSTIVE_INSTRUMENT_ONLY',
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        execution_source='content hashes; working changes are published with these artifacts',
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        environment=dict(python=platform.python_version(),numpy=numpy.__version__,qiskit=qiskit.__version__),
        scientific_input_files=[],rng_calls=0,shot_sampling_calls=0,compile_calls=0,
        old_tests_executed=False,scientific_engine_qualified=False)
    (args.output/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    stream=io.StringIO();start=time.perf_counter()
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests/tracks/algorithm_codesign'),
                                            pattern='test_m_label_average.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    module=sys.modules['test_m_label_average']
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        evidence=module.EVIDENCE,scientific_claim='ARTIFICIAL_CORRECTNESS_ONLY')
    (args.output/'TEST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'RUN.txt').write_text(stream.getvalue())
    (args.output/'STOP.json').write_text(json.dumps(dict(status='MANDATORY_STOP',
        reason='Artificial scope finished; H4 metadata/Born/shots/compile not authorized',
        test_status=report['status'],automatic_next_stage=False),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evidence'}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
