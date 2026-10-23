"""Run only supplied artificial regressions, optionally against the held source.

An engine path is Python source, never a saved H4 model or scientific input.
The old source is imported read-only; no source or previous log is modified.
"""
import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expect-failure',action='store_true')
    a=p.parse_args();a.output.mkdir(exist_ok=False)
    sys.path.insert(0,str(ROOT/'tests'))
    import test_h4_ball_evidence as tests
    spec=importlib.util.spec_from_file_location('reference_regression_engine',a.engine)
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    tests.e=engine
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.ReferenceContraction)
    with (a.output/'tests.log').open('x') as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    record=dict(engine=str(a.engine.resolve()),tests=result.testsRun,failures=len(result.failures),
                errors=len(result.errors),skips=len(result.skipped),
                status='PASS' if result.wasSuccessful() else 'FAIL',
                expected_failure=a.expect_failure,
                scope='Hard-coded 1/2-qubit artificial states and models only',real_H4_calls=0)
    with (a.output/'tests.json').open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(record))
    # The held source must exhibit assertion failures without unexpected errors.
    ok=(bool(result.failures) and not result.errors) if a.expect_failure else result.wasSuccessful()
    sys.exit(0 if ok else 1)

if __name__=='__main__':main()
