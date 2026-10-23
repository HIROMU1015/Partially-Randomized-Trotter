#!/usr/bin/env python3
"""Explicit artificial C/B regression suite; no real-H4 execution test discovery."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    modules=['test_h4_precision_bc','test_h4_precision_b_authority','test_h4_ball_evidence','test_h4_certificate_control']
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in modules)
    with (args.output/'tests.log').open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    record=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),modules=modules,real_H4_C_calls=0,real_H4_B_axes_repeated=0,science_grants_issued=0,scope='Artificial and mock tests only; old B saved results never re-evaluated')
    with (args.output/'tests.json').open('x') as stream:json.dump(record,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(record));sys.exit(0 if result.wasSuccessful() else 1)
if __name__=='__main__':main()
