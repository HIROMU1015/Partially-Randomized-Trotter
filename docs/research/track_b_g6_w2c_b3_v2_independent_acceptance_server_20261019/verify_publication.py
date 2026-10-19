"""Read back the exact published content commit and record every changed byte."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import argparse,json,subprocess,urllib.request
from urllib.parse import quote
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE='49a0dcc9c8ec6dfc96eba5bb862cfdcb3d0fc866'
BRANCH='track-b-g6-w2c-b3-v2-independent-acceptance-server-20261019'
REPO='HIROMU1015/Partially-Randomized-Trotter'
def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--content-commit',required=True);a=p.parse_args();commit=a.content_commit
    assert len(commit)==40 and git('rev-parse',commit).decode().strip()==commit
    url=git('remote','get-url','origin').decode().strip();assert url.removesuffix('.git')=='https://github.com/'+REPO
    remote=git('ls-remote','origin','refs/heads/'+BRANCH).decode().split()[0];assert remote==commit
    paths=git('diff','--name-only','--diff-filter=ACM',BASE,commit).decode().splitlines()
    def one(path):
        raw_url='https://raw.githubusercontent.com/'+REPO+'/'+commit+'/'+quote(path,safe='/')
        raw=urllib.request.urlopen(raw_url,timeout=90).read();expected=git('show',commit+':'+path);assert raw==expected,path
        return dict(path=path,bytes=len(raw),sha256=sha256(raw).hexdigest(),url=raw_url,git_bytes_equal=True)
    with ThreadPoolExecutor(max_workers=16) as pool:rows=list(pool.map(one,paths))
    receipt=dict(schema='b3_v2_independent_publication_receipt_v1',repository=REPO,branch=BRANCH,content_commit=commit,remote_HEAD_at_content_verification=remote,tree=git('rev-parse',commit+'^{tree}').decode().strip(),checked_UTC=datetime.now(timezone.utc).isoformat(),files_checked=len(rows),bytes_checked=sum(x['bytes'] for x in rows),all_public_bytes_SHA256_equal=True,files=rows,receipt_scope='All content paths changed from fixed base. A following receipt-only commit adds this file; final ref and receipt bytes are checked separately without a self-referential hash claim.',mandatory_STOP=True)
    target=HERE/'publication_receipt.json'
    with target.open('x') as dst:dst.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
if __name__=='__main__':main()
