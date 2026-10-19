"""Small artificial development probe; no science or real authority."""
from pathlib import Path
import sys,tempfile
ROOT=Path(__file__).resolve().parents[3]
sys.dont_write_bytecode=True;sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/tracks/algorithm_codesign')]
import b3_storage_artificial_fixture as F
from trottertracks.algorithm_codesign import g6_w2c_b3_storage_verify_v2 as V

if __name__=='__main__':
    mode=sys.argv[1]if len(sys.argv)>1 else'S'
    fault=F.Fault(sys.argv[2]if len(sys.argv)>2 else'',{'op':sys.argv[3]}if len(sys.argv)>3 else{})
    with tempfile.TemporaryDirectory(prefix='b3-r1-artificial-')as t:
        root=Path(t);c,i,l,a=F.fixture(root,F.F.P.Settings(profile_n=16,codec='fused',workload='light',mode=mode))
        r=F.invoke(root,c,i,l,a,fault)
        print(dict(complete=r['complete'],reserved=r['reserved'],started=r['started_observed'],states=r['states'],injected=fault.events))
        print(V.verify(root/'artificial-output',c,i,r))
