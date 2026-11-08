#!/usr/bin/env python3
"""Invented vacuum-reference correctness then finite compile comparison only."""
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
import traceback
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/tracks/algorithm_codesign'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    paths=['src/trottertracks/algorithm_codesign/vacuum_reference_cswap.py',
        'tests/tracks/algorithm_codesign/test_vacuum_reference_cswap.py',
        'tests/tracks/algorithm_codesign/vacuum_cswap_artificial_benchmark.py',
        str(Path(__file__).relative_to(ROOT)),
        'src/trottertracks/algorithm_codesign/a2_lcu_access.py',
        'tests/tracks/algorithm_codesign/test_a2_lcu_access.py',
        'src/trottertracks/pr_dictionary_validation/gates.py',
        'src/trottertracks/pr_dictionary_validation/fixture.py',
        'src/trotterlib/research_direction_structure_pilot.py',
        'src/trotterlib/rpe_hadamard_interrogation.py']
    import numpy,qiskit
    source=dict(scope='INVENTED_1_3_SYSTEM_QUBITS_NO_MOLECULAR_INPUT',
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        execution_source='content hashes; exact source published with results',
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
        environment=dict(python=platform.python_version(),numpy=numpy.__version__,
                         qiskit=qiskit.__version__,executable=sys.executable),
        resource_start=dict(cpu_count=os.cpu_count(),loadavg=os.getloadavg(),
            memory_available=next(x.strip() for x in Path('/proc/meminfo').read_text().splitlines()
                                  if x.startswith('MemAvailable:'))),
        scientific_input_files=[],RNG_sampling_calls=0,shot_calls=0,H4_calls=0,
        saved_result_reanalysis=0,scientific_engine_qualified=False,
        finite_benchmark='7 invented paths x X/Y x 4 methods x 2 topologies; one Fredkin primitive')
    (args.output/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    suite=unittest.defaultTestLoader.loadTestsFromName('test_vacuum_reference_cswap')
    for name in ('test_nonphysical_or_unspecified_inputs_stop',
                 'test_scalar_phase_is_coherent_and_Y_sign_is_preserved'):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromName('test_a2_lcu_access.LCUAccessTests.'+name))
    stream=io.StringIO();start=time.perf_counter()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        related_A2_regressions=2,failures=len(result.failures),errors=len(result.errors),
        elapsed_seconds=time.perf_counter()-start,
        evidence=sys.modules['test_vacuum_reference_cswap'].EVIDENCE,
        tolerance=8e-12,phase_fitting=False,scientific_claim='ARTIFICIAL_ONLY_NO_UNIFORM_NUMERICAL_CERTIFICATE')
    (args.output/'TEST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    status='CORRECTNESS_FAILED';benchmark=None
    if result.wasSuccessful():
        try:
            from vacuum_cswap_artificial_benchmark import run
            benchmark=run(args.output);status='PASS'
        except Exception:
            status='TECHNICAL_BENCHMARK_FAILED';stream.write(traceback.format_exc())
    (args.output/'RUN.txt').write_text(stream.getvalue())
    stop=dict(status='MANDATORY_STOP',reason=status,automatic_next_stage=False,
        elapsed_seconds=time.perf_counter()-start,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        H4_calls=0,RNG_sampling_calls=0,shot_calls=0)
    (args.output/'STOP.json').write_text(json.dumps(stop,indent=2)+'\n')
    print(json.dumps(dict(status=status,tests=report['tests'],
        benchmark_cells=None if benchmark is None else benchmark['cells'],
        elapsed_seconds=stop['elapsed_seconds'],peak_rss_kib=stop['peak_rss_kib'])))
    return 0 if status=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
