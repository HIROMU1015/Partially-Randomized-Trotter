"""Collect new artificial tests and retain their raw observations in a fresh dir."""
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def main():
    sys.dont_write_bytecode=True
    sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
    os.chdir(ROOT)
    target=HERE/'forward_evidence';target.mkdir(exist_ok=False)
    os.environ['B3_PILOT_EVIDENCE']=str(target)
    import test_g6_w2c_b3_pilot_candidate as tests
    suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
    def ids(s):
        for t in s:
            if isinstance(t,unittest.TestSuite):yield from ids(t)
            else:yield t.id()
    collected=list(ids(suite))
    with (HERE/'pilot_tests.log').open('x')as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    bad={t.id()for t,_ in result.failures+result.errors+result.skipped}
    forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1]in tests.C.P.FORBIDDEN]
    report=dict(collected=collected,run=result.testsRun,pass_count=result.testsRun-len(bad),
        failed=[t.id()for t,_ in result.failures+result.errors],skipped=len(result.skipped),
        passed_ids=[x for x in collected if x not in bad],forbidden_science_modules_imported=forbidden,
        successful=result.wasSuccessful()and not forbidden,artificial_only=True,
        scientific_calls=0,real_authorizations=0,real_grants=0,real_claims=0,
        source_sha256=hashlib.sha256(Path(tests.__file__).read_bytes()).hexdigest())
    (HERE/'pilot_test_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in ('collected','passed_ids')}))
    raise SystemExit(0 if report['successful']else 1)


if __name__=='__main__':main()
