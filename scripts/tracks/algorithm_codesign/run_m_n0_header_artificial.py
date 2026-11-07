#!/usr/bin/env python3
"""Only new saved-header mock tests and three relevant old eligibility checks."""
import argparse
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    from trottertracks.algorithm_codesign.m_n0_metadata_execution import SOURCE_FILES,INPUT_BLOBS
    extra=('tests/tracks/algorithm_codesign/test_m_n0_header_binding.py',
           'tests/tracks/algorithm_codesign/test_m_label_average.py',
           str(Path(__file__).relative_to(ROOT)))
    source=dict(scope='ARTIFICIAL_SAVED_HEADERS_AND_MOCK_ENTRY_ONLY',
        base_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCE_FILES+extra},
        environment=dict(python=platform.python_version(),numpy=importlib.metadata.version('numpy'),qiskit=importlib.metadata.version('qiskit')),
        real_N1_input_blobs_pinned_without_payload_evaluation=INPUT_BLOBS,
        real_N1_pass_calls=0,quantum_simulation_calls=0,shots_or_rng=0,compile_calls=0)
    (args.output/'SOURCE.json').write_text(json.dumps(source,indent=2)+'\n')
    start=time.perf_counter();stream=io.StringIO()
    loader=unittest.defaultTestLoader
    suite=loader.discover(str(ROOT/'tests/tracks/algorithm_codesign'),pattern='test_m_n0_header_binding.py')
    import test_m_label_average as old
    old_names=('test_exact_actual_law_signed_support_identity_and_partition',
               'test_noncommon_nearby_or_channel_equivalent_descriptions_are_rejected',
               'test_zero_support_actual_probability_and_invalid_mapping_stop')
    suite.addTests(old.MInstrumentTests(name) for name in old_names)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    new=sys.modules['test_m_n0_header_binding']
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',new_tests=result.testsRun-len(old_names),
        old_eligibility_regressions=len(old_names),failures=len(result.failures),errors=len(result.errors),
        elapsed_seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        evidence=new.EVIDENCE,old_full_instrument_suite_run=False,real_N1_evaluated=False)
    (args.output/'TEST_RESULTS.json').write_text(json.dumps(report,indent=2)+'\n')
    (args.output/'RUN.txt').write_text(stream.getvalue())
    (args.output/'STOP.json').write_text(json.dumps(dict(status='MANDATORY_STOP',
        test_status=report['status'],real_N1_pass_started=False,downstream_started=False),indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='evidence'}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
