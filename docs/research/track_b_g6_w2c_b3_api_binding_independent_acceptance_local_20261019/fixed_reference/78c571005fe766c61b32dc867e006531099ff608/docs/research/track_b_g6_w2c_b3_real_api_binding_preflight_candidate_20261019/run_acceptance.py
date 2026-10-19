"""Artificial pair port exercises; unchanged Bundle regression is separate."""
from pathlib import Path
import argparse,json,sys,tempfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.dont_write_bytecode=True

def main():
    from trottertracks.algorithm_codesign.g6_w2c_b3_api_binding_v1 import static as S,exercise as E
    p=argparse.ArgumentParser();p.add_argument('label');p.add_argument('--n',type=int,default=256)
    p.add_argument('--fault-mode',choices=['S','P1']);p.add_argument('--fault-stage',default='')
    a=p.parse_args()
    dest=HERE/('runs_'+a.label);dest.mkdir(exist_ok=False)
    c=S.candidate(ROOT);(dest/'trusted_candidate.json').write_bytes(S.encoded(c))
    root=Path(tempfile.mkdtemp(prefix='b3-bundle-artificial-API-'))
    (dest/'location.json').write_text(json.dumps(dict(root=str(root)))+'\n')
    r=E.run(c,root,n=a.n,fault_mode=a.fault_mode,fault_stage=a.fault_stage)
    print(json.dumps(dict(root=str(root),complete=r['complete'],pairs=len(r['rows']),outer=r['outer'])))
if __name__=='__main__':main()
