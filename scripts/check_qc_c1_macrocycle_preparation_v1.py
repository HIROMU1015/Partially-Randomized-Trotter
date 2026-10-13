#!/usr/bin/env python3
"""Exact off-domain preparation checks only; zero physics substitutions/eigensolves."""
import json
from pathlib import Path
import sys
import unittest
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from trottertracks.representation_exploration.c1_macrocycle_algebra_v1 import preparation_proofs
if __name__=='__main__':
    suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_qc_c1_macrocycle_preparation_v1.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=preparation_proofs()
    report.update(status='PASS' if result.wasSuccessful() else 'FAIL',tests_run=result.testsRun,
                  numeric_performance_runs=0,registered_input_evaluations=0,compile_runs=0)
    folder=root/'artifacts/qc_c1_macrocycle_preparation_v1/2026-10-13'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'off_domain_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    if not result.wasSuccessful():sys.exit(1)
