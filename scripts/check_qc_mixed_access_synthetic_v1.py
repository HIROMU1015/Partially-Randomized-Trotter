#!/usr/bin/env python3
"""Run this source version's non-molecular tests and save technical evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(); started=time.monotonic()
    suite=unittest.defaultTestLoader.loadTestsFromName('test_qc_mixed_access_feasibility_v1')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    files=sorted(ROOT.glob('src/trottertracks/representation_exploration/qc_mixed_access_*_v1.py'))+[ROOT/'tests/test_qc_mixed_access_feasibility_v1.py']
    report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,
            'errors':len(result.errors),'failures':len(result.failures),'molecular_jobs':0,
            'registered_scientific_probes':0,'quantum_compile':0,'old_synthetic_suite_rerun':False,
            'wall_seconds':time.monotonic()-started,'parent_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'children_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*1024,
            'source_SHA256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'claim':'local non-molecular implementation verification;not molecular feasibility or CI evidence',
            'mandatory_stop':True}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__': raise SystemExit(main())
