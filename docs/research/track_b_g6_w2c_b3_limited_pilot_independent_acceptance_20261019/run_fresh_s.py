"""One new fake S projection; same frozen SOURCE, no scientific bridge."""
from pathlib import Path
import sys,os,json,tempfile,tarfile,shutil,argparse
T=Path('/tmp/prt-b3-limited-pilot-independent-20261019');R=T/'producer-runtime'
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--runtime',type=Path,default=R);parser.add_argument('--new-case',type=Path,default=T/'fresh_S_projection');options=parser.parse_args();R=options.runtime.resolve()
 sys.dont_write_bytecode=True;sys.path[:0]=[str(R/'src'),str(R/'tests/tracks/algorithm_codesign')]
 import b3_pilot_artificial_fixture as F
 dest=options.new_case.resolve();dest.mkdir(parents=True,exist_ok=False)
 with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-') as d:
  root=Path(d);c,inner,ledger,auth=F.fixture(root,F.F.P.Settings(profile_n=16,codec='fused',mode='S',workers=1,workload='light'))
  r=F.invoke(root,c,inner,ledger,auth);assert r['complete'] and r['states']['COMPLETED']==768
  for n,obj in [('pilot_contract.json',c),('inner_contract.json',inner),('observations.json',dict(report=r))]:
   (dest/n).write_bytes(F.F.P.safe_json(obj)+b'\n')
  shutil.copyfile(ledger.path,dest/'FAKE_ONLY_claims.sqlite3')
  with tarfile.open(dest/'artificial_raw.tar.gz','w:gz') as f:f.add(root/'artificial-output',arcname='artificial-output')
 forbidden=[n for n in sys.modules if n.rsplit('.',1)[-1] in F.F.P.FORBIDDEN];assert not forbidden
 print(json.dumps(dict(case='fresh_S_projection',complete=True,fake_calls=768,scientific_calls=0,forbidden_imports=forbidden)))
