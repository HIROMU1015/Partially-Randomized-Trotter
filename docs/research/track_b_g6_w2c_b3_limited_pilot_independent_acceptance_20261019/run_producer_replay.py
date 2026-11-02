"""Separate fixed producer artificial regression, not independent checker proof."""
from pathlib import Path
import sys,os,json,unittest,hashlib,argparse
R=Path('/tmp/prt-b3-limited-pilot-independent-20261019/producer-runtime');T=R.parent
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime',type=Path,default=R);parser.add_argument('--new-output',type=Path);options=parser.parse_args();R=options.runtime.resolve()
 if options.new_output:T=options.new_output.resolve();T.mkdir(parents=True,exist_ok=False)
 sys.dont_write_bytecode=True;sys.path[:0]=[str(R/'src'),str(R/'tests/tracks/algorithm_codesign')]
 os.chdir(R);os.environ['B3_PILOT_EVIDENCE']=str(T/'fresh_producer_evidence')
 import test_g6_w2c_b3_pilot_candidate as tests
 suite=unittest.defaultTestLoader.loadTestsFromModule(tests)
 def ids(s):
  for x in s:
   if isinstance(x,unittest.TestSuite):yield from ids(x)
   else:yield x.id()
 collected=list(ids(suite))
 with (T/'producer_replay.log').open('x') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
 forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1] in tests.C.P.FORBIDDEN]
 report=dict(collected=collected,run=r.testsRun,passed=r.testsRun-len(r.failures)-len(r.errors)-len(r.skipped),fail=len(r.failures),error=len(r.errors),skip=len(r.skipped),failed_ids=[x.id() for x,_ in r.failures+r.errors],science_modules_imported=forbidden,source_sha256=hashlib.sha256(Path(tests.__file__).read_bytes()).hexdigest(),producer_replay_only=True,independent_test_count_addition=False)
 (T/'producer_replay.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));sys.exit(0 if r.wasSuccessful() and not forbidden else 1)
