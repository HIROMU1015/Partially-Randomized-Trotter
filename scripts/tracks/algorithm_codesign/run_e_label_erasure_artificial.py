#!/usr/bin/env python3
"""Invented E instrument only; reuse the existing M artificial evidence pattern."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'tests/tracks/algorithm_codesign'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    paths=['src/trottertracks/algorithm_codesign/e_label_erasure.py',
           'tests/tracks/algorithm_codesign/test_e_label_erasure.py',
           str(Path(__file__).relative_to(ROOT)),
           'src/trottertracks/algorithm_codesign/a2_lcu_access.py',
           'tests/tracks/algorithm_codesign/test_a2_lcu_access.py',
           'src/trottertracks/pr_dictionary_validation/gates.py',
           'src/trotterlib/rpe_hadamard_interrogation.py']
    import numpy,qiskit
    source=dict(scope='INVENTED_1_2_SYSTEM_QUBIT_TWO_LABEL_EXHAUSTIVE_ONLY',
        design_commit='081959402f54f06048adb038f97911950091ae1c',
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        execution_source='content hashes; these exact files are published with results',
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        environment=dict(python=platform.python_version(),numpy=numpy.__version__,
                         qiskit=qiskit.__version__,executable=sys.executable),
        resource_start=dict(cpu_count=os.cpu_count(),loadavg=os.getloadavg(),
            memory_available=next(x.strip() for x in Path('/proc/meminfo').read_text().splitlines()
                                  if x.startswith('MemAvailable:'))),
        scientific_input_files=[],RNG_sampling_calls=0,shot_calls=0,compile_calls=0,
        H4_calls=0,saved_result_reanalysis=0,scientific_engine_qualified=False)
    (args.output/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    loader=unittest.defaultTestLoader
    suite=loader.loadTestsFromName('test_e_label_erasure')
    # Only shared validation/control contracts needed by the new adapter.
    for name in ('test_nonphysical_or_unspecified_inputs_stop',
                 'test_scalar_phase_is_coherent_and_Y_sign_is_preserved'):
        suite.addTests(loader.loadTestsFromName('test_a2_lcu_access.LCUAccessTests.'+name))
    stream=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        related_A2_regressions=2,failures=len(result.failures),errors=len(result.errors),
        elapsed_seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        evidence=sys.modules['test_e_label_erasure'].EVIDENCE,
        technical_classification='TECHNICALLY_VALID_KNOWN_LCU_APPLICATION' if result.wasSuccessful()
                                 else 'TECHNICAL_CORRECTNESS_FAILED',
        scientific_claim='ARTIFICIAL_CORRECTNESS_ONLY_NO_H4_RESOURCE_CLAIM')
    (args.output/'TEST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'RUN.txt').write_text(stream.getvalue())
    (args.output/'STOP.json').write_text(json.dumps(dict(status='MANDATORY_STOP',
        reason='Artificial technical selection finished; no automatic scientific stage',
        test_status=report['status'],automatic_next_stage=False),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evidence'}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
