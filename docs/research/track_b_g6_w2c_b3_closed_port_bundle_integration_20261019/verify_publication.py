"""Read exact public bytes and branch HEAD; never changes remote state."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hashlib import sha256
from urllib.request import urlopen
from urllib.parse import quote
import argparse,datetime,json,subprocess,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BRANCH='track-b-g6-w2c-b3-closed-port-bundle-integration-20261019'
BASE='d24cccbecdf0928fe977080dd7b28c591df0089e'
SOURCE='deade78a19f5b23859d16816d3a7bfb37765dd80'
REPO='https://github.com/HIROMU1015/Partially-Randomized-Trotter.git'
RAW='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'

def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)

def main():
    p=argparse.ArgumentParser();p.add_argument('--content-commit',required=True);p.add_argument('--final-commit');p.add_argument('--output')
    a=p.parse_args();head=a.final_commit or a.content_commit
    remote=git('ls-remote',REPO,'refs/heads/'+BRANCH).decode().split()[0]
    assert remote==head,(remote,head)
    paths=[x.decode() for x in git('diff','--name-only','-z',BASE,a.content_commit).split(b'\0') if x]
    def one(path):
        local=git('show',a.content_commit+':'+path);url=RAW+head+'/'+quote(path,safe='/')
        for attempt in range(4):
            try:
                with urlopen(url,timeout=120)as r:public=r.read()
                break
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
        assert public==local,path
        return dict(path=path,bytes=len(public),sha256=sha256(public).hexdigest(),public_equals_git=True,url=url)
    with ThreadPoolExecutor(max_workers=12)as pool:files=list(pool.map(one,paths))
    result=dict(schema='b3_global_port_publication_receipt_v1',repository='HIROMU1015/Partially-Randomized-Trotter',branch=BRANCH,
                measured_runtime_SOURCE=SOURCE,content_commit=a.content_commit,remote_HEAD_observed=head,
                file_count=len(files),bytes=sum(x['bytes']for x in files),files=files,status='PASS',
                verified_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                receipt_only_finalization_expected=a.final_commit is None,receipt_self_hash_is_external=True,
                run_ready=False,science_release=None,scientific_GO=False,mandatory_STOP=True)
    if a.final_commit:
        receipt=str((HERE/'publication_receipt.json').relative_to(ROOT));local=git('show',head+':'+receipt)
        with urlopen(RAW+head+'/'+quote(receipt,safe='/'),timeout=120) as r:public=r.read()
        assert public==local
        result['final_receipt_readback']=dict(bytes=len(public),sha256=sha256(public).hexdigest(),commit=head)
    dest=Path(a.output) if a.output else HERE/'publication_receipt.json'
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ('status','remote_HEAD_observed','file_count','bytes')}))
if __name__=='__main__':main()
