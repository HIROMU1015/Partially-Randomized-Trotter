"""Read-only local/Git/public byte and remote HEAD check; fresh report only."""
from pathlib import Path
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
import argparse,json,os,subprocess,urllib.request,urllib.parse,time
P=Path(__file__).resolve().parent;W=P.parents[2]
BASE='6bffb5b5819918698b522270ff19b255766d3072'
def git(*args):return subprocess.check_output(['git','-C',str(W),*args])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--head',required=True);ap.add_argument('--new-report',required=True);a=ap.parse_args();out=Path(a.new_report).resolve();assert not out.exists()
    remote=git('remote','get-url','origin').decode().strip();assert remote=='git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
    branch=git('branch','--show-current').decode().strip();head=git('rev-parse','HEAD').decode().strip();assert head==a.head
    remote_head=git('ls-remote','origin','refs/heads/'+branch).decode().split()[0];assert remote_head==head
    paths=git('diff','--name-only','--diff-filter=ACMRT',BASE,head).decode().splitlines();packet=str(P.relative_to(W))+'/'
    exact={'PROJECT_MAP.md','docs/README.md','scripts/README.md','src/trotterlib/README.md','tests/tracks/algorithm_codesign/test_g6_w2c_b3_api_independent_v1.py'}
    assert all(p in exact or p.startswith((packet,'src/trottertracks/algorithm_codesign/g6_w2c_b3_api_independent_v1/')) for p in paths)
    def one(path):
        raw=git('show',head+':'+path);assert (W/path).read_bytes()==raw
        url='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'+head+'/'+urllib.parse.quote(path)
        for attempt in range(4):
            try:
                public=urllib.request.urlopen(url,timeout=60).read();assert public==raw
                return dict(path=path,bytes=len(raw),sha256=sha256(raw).hexdigest(),public_url=url,local_git_public_equal=True)
            except Exception:
                if attempt==3:raise
                time.sleep(1)
    with ThreadPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as pool:rows=list(pool.map(one,paths))
    report=dict(schema='b3_API_independent_publication_receipt_v1',branch=branch,local_HEAD=head,remote_HEAD=remote_head,base_commit=BASE,changed_files=len(rows),all_local_git_public_bytes_equal=True,files=rows,scientific_GO=False,mandatory_STOP=True)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='files'}))
if __name__=='__main__':main()
