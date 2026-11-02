"""Read back exact public commit bytes and branch HEAD; no write to GitHub."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hashlib import sha256
from urllib.request import urlopen
from urllib.parse import quote
import argparse,json,subprocess,time,datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
REPO='https://github.com/HIROMU1015/Partially-Randomized-Trotter.git';RAW='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'
BRANCH='track-b-g6-w2c-b3-v2-production-bundle-bridge-candidate-20261019';BASE='596781daeba253418ba35f4b7964312a54caf233'
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
def main():
    p=argparse.ArgumentParser();p.add_argument('--content-commit',required=True);p.add_argument('--receipt-only-final-commit');p.add_argument('--output',default=str(HERE/'publication_receipt.json'));a=p.parse_args()
    head=a.receipt_only_final_commit or a.content_commit
    remote=git('ls-remote',REPO,'refs/heads/'+BRANCH).decode().split()[0];assert remote==head,(remote,head)
    paths=[x.decode()for x in git('diff','--name-only','-z',BASE,a.content_commit).split(b'\0')if x]
    def one(path):
        local=git('show',a.content_commit+':'+path);url=RAW+head+'/'+quote(path,safe='/')
        for attempt in range(4):
            try:
                with urlopen(url,timeout=120)as r:public=r.read()
                break
            except Exception:
                if attempt==3:raise
                time.sleep(1+attempt)
        assert public==local,path
        return dict(path=path,bytes=len(public),sha256=sha256(public).hexdigest(),public_equals_git=True,url=url)
    with ThreadPoolExecutor(max_workers=12)as pool:files=list(pool.map(one,paths))
    result=dict(schema='b3_bundle_publication_receipt_v1',repository='HIROMU1015/Partially-Randomized-Trotter',branch=BRANCH,content_commit=a.content_commit,remote_HEAD_observed=head,measured_runtime_SOURCE_commit='e099dcc461247e81eff981867dc28c3f3b16df5e',verified_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),file_count=len(files),bytes=sum(x['bytes']for x in files),files=files,status='PASS',receipt_only_finalization_expected=a.receipt_only_final_commit is None,receipt_self_hash_is_external_to_this_file=True,mandatory_STOP=True)
    if a.receipt_only_final_commit:
        path=str((HERE/'publication_receipt.json').relative_to(ROOT));local=git('show',head+':'+path)
        with urlopen(RAW+head+'/'+quote(path,safe='/'),timeout=120)as r:public=r.read()
        assert local==public;result['final_receipt_readback']=dict(bytes=len(public),sha256=sha256(public).hexdigest(),commit=head)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ('status','remote_HEAD_observed','file_count','bytes')}))
if __name__=='__main__':main()
