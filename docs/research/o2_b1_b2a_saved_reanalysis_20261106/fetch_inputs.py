"""Restore public fixed GitHub bytes for this analysis in a dedicated directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    repo=a.out/'.fixed-source-clone'
    if not repo.exists():
        subprocess.run(['git','clone','--no-checkout','--filter=blob:none','--depth=1','https://github.com/HIROMU1015/Partially-Randomized-Trotter.git',str(repo)],check=True)
    refs=json.loads((Path(__file__).parent/'INPUT_INDEX.json').read_text())['references']
    present=set()
    for r in refs:
        sha,path=r['commit'],r['path']
        out=a.out/sha/path
        if out.exists():
            assert hashlib.sha256(out.read_bytes()).hexdigest()==r['sha256']
            continue
        if sha not in present:
            subprocess.run(['git','-C',str(repo),'fetch','--depth=1','origin',sha],check=True)
            present.add(sha)
        data=subprocess.check_output(['git','-C',str(repo),'show',sha+':'+path])
        assert hashlib.sha256(data).hexdigest()==r['sha256']
        blob=subprocess.check_output(['git','-C',str(repo),'hash-object','--stdin'],input=data,text=False).decode().strip()
        assert blob==r['blob']
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(data)
    print('Verified fixed files:',len(refs))


if __name__=='__main__':
    main()
