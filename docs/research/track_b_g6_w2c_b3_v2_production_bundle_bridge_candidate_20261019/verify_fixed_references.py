"""Read-only public-byte verification and fresh measured-Git-object restore."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from hashlib import sha256
import json,subprocess,urllib.request,urllib.error,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];REPO='https://github.com/HIROMU1015/Partially-Randomized-Trotter.git';RAW='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'
P='596781daeba253418ba35f4b7964312a54caf233';I='77126767c00585d0967558c0756907df2d69d1ff';M='45bf46b4a833e71f138903fb511f2741e54043b7';BASE='76ca3c4089b201d55da0fede08e1f77733ac38a5';D='docs/research/track_b_g6_w2c_b3_trace_reservation_accounting_repair_20261019/'
def get(ref,path):
    from urllib.parse import quote
    with urllib.request.urlopen(RAW+ref+'/'+quote(path,safe='/'),timeout=90)as r:return r.read()
def git(*args,cwd=ROOT):return subprocess.check_output(['git',*args],cwd=cwd)
def main():
    records=json.loads((HERE/'fixed_reference_reads.json').read_text());inv=json.loads(git('show',P+':'+D+'source_inventory.json'))
    wanted={(x['commit'],x['path'])for x in records}|{(P,x['path'])for x in inv['files']+inv['protected']}
    inventory=json.loads(git('show',I+':docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/source_inventory.json'))
    wanted|={(I,x['path'])for x in inventory['files']}
    def one(pair):
        ref,path=pair;raw=get(ref,path);local=git('show',ref+':'+path);assert raw==local,(ref,path)
        return dict(commit=ref,path=path,bytes=len(raw),sha256=sha256(raw).hexdigest(),public_bytes_equal_git=True)
    with ThreadPoolExecutor(max_workers=12)as pool:checks=list(pool.map(one,sorted(wanted)))
    branches=['track-b-g6-w2c-b3-trace-reservation-accounting-repair-20261019','track-b-g6-w2c-b3-v2-independent-acceptance-server-20261019'];heads=git('ls-remote',REPO,*['refs/heads/'+b for b in branches]).decode();assert P in heads and I in heads
    bare=HERE/'preflight/measurement_restore.git';bare.mkdir(exist_ok=False);git('init','--bare',str(bare));git('fetch','--no-tags',REPO,BASE,cwd=bare)
    b=get(P,D+'measurement_source_history.bundle');assert len(b)==49475 and sha256(b).hexdigest()=='3f0012af22e4bd07c47de3b166e39b7a8633958728572d76d66f6d1ebd50e14c'
    bundle=HERE/'preflight/measurement_source_history.bundle';bundle.write_bytes(b);git('bundle','verify',str(bundle),cwd=bare);git('fetch',str(bundle),'refs/b3-storage-repair/source-20261019',cwd=bare)
    measured=[]
    for row in inv['files']:
        if row.get('at_measured_SOURCE'):
            original=git('show',M+':'+row['path'],cwd=bare);public=get(P,row['path']);assert original==public and sha256(original).hexdigest()==row['sha256'];measured.append(dict(path=row['path'],bytes=len(original),sha256=sha256(original).hexdigest()))
    assert len(measured)==15
    result=dict(schema='b3_bundle_fixed_public_reference_verification_v1',remote_heads=heads,public_files=checks,measured_restore=dict(measured_commit=M,public_head=P,prerequisite=BASE,bundle_bytes=len(b),bundle_sha256=sha256(b).hexdigest(),measured_source_files=measured,measured_source_count=15,public_head_later_helpers_count=2),independent_fixed_counts=json.loads(git('show',I+':docs/research/track_b_g6_w2c_b3_v2_independent_acceptance_server_20261019/verification.json')))
    (HERE/'fixed_public_reference_verification.json').write_text(json.dumps(result,indent=2)+'\n');print('public files',len(checks),'measured source files',len(measured))
if __name__=='__main__':main()
