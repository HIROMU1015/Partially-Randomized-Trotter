"""Fetch public fixed commit bytes and compare SHA256 against local Git."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hashlib import sha256
from urllib.request import urlopen
from urllib.parse import quote
import argparse,datetime,json,subprocess,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BRANCH='track-b-g6-w2c-b3-real-api-adapter-contract-closed-v1-20261020'
BASE='62967269e611b61e73b70abe20e998d9c7eaaf50'
REPO='https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'
RAW='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--content-commit',required=True);p.add_argument('--head',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    remote=git('ls-remote',REPO,'refs/heads/'+BRANCH).decode().split()[0];assert remote==a.head
    paths=git('diff','--name-only',BASE,a.head).decode().splitlines()
    def one(path):
        local=git('show',a.head+':'+path);url=RAW+a.head+'/'+quote(path,safe='/')
        for attempt in range(4):
            try:
                with urlopen(url,timeout=60) as response:public=response.read()
                break
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
        assert public==local,path
        return dict(path=path,bytes=len(public),sha256=sha256(public).hexdigest(),public_equals_git=True)
    with ThreadPoolExecutor(max_workers=12) as pool:rows=list(pool.map(one,paths))
    v=json.loads((HERE/'verification.json').read_text())
    receipt=dict(schema='b3_closed_adapter_publication_receipt_v1',repository='HIROMU1015/Partially-Randomized-Trotter',
        branch=BRANCH,base=BASE,measured_SOURCE=v['measured_SOURCE'],content_commit=a.content_commit,
        remote_HEAD_observed=a.head,status='PASS',files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),
        verified_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),scientific_GO=False,mandatory_STOP=True,
        receipt_self_hash_is_external=True)
    Path(a.output).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','remote_HEAD_observed','file_count','bytes')}))
