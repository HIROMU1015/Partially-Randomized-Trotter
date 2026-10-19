"""Fetch pinned public bytes and independently restore artificial evidence only."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
import io,json,subprocess,tarfile,urllib.request
HERE=Path(__file__).resolve().parent
SHA='596781daeba253418ba35f4b7964312a54caf233'
PREFIX='docs/research/track_b_g6_w2c_b3_trace_reservation_accounting_repair_20261019/'
def git(path,commit=SHA):
    return subprocess.check_output(['git','show',commit+':'+path],cwd=HERE)
def main():
    inventory=json.loads(git(PREFIX+'source_inventory.json'))
    archive=json.loads(git(PREFIX+'archive_parts.json'))['archives'][0]
    names=['README.md','handoff.md','defect_reproduction.md','root_cause_and_fix.md','source_and_contract_delta.md','stop_and_unknown_verification.md','independent_checker_comparison.md','reservation_accounting_verification.json','fault_injection_matrix.json','artificial_regression_results.md','source_inventory.json','verification.json','publication_receipt.json','archive_parts.json','restore_artifacts.py','fault_cases_manifest.json']
    names += ['before_repair/fresh_trace_reserve_write_failure/'+x for x in ('pilot_contract.json','inner_contract.json','observations.json','FAKE_ONLY_claims.sqlite3','artificial_raw.tar.gz')]
    paths={PREFIX+x for x in names}|{PREFIX+x['path'] for x in archive['parts']}|{x['path'] for x in inventory['files']+inventory['protected']}
    root=HERE/'public_bytes';root.mkdir(exist_ok=False)
    def one(path):
        url='https://raw.githubusercontent.com/HIROMU1015/Partially-Randomized-Trotter/'+SHA+'/'+path
        raw=urllib.request.urlopen(url,timeout=90).read()
        assert raw==git(path),path
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        return dict(path=path,url=url,bytes=len(raw),sha256=sha256(raw).hexdigest(),git_bytes_equal=True)
    with ThreadPoolExecutor(max_workers=16) as pool:receipts=list(pool.map(one,sorted(paths)))
    for row in inventory['files']:
        raw=(root/row['path']).read_bytes();assert len(raw)==row['bytes'] and sha256(raw).hexdigest()==row['sha256']
        if row['at_measured_SOURCE']:assert raw==git(row['path'],inventory['measured_SOURCE_commit'])
    for row in inventory['protected']:
        assert sha256((root/row['path']).read_bytes()).hexdigest()==row['sha256']
    raw=bytearray()
    for part in archive['parts']:
        block=(root/PREFIX/part['path']).read_bytes()
        assert len(block)==part['bytes'] and sha256(block).hexdigest()==part['sha256'];raw.extend(block)
    assert len(raw)==archive['bytes'] and sha256(raw).hexdigest()==archive['sha256']
    target=HERE/'restored';target.mkdir(exist_ok=False)
    members=[]
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tf:
        for member in tf:
            p=Path(member.name);assert not p.is_absolute() and '..' not in p.parts
            dest=target/p
            if member.isdir():dest.mkdir(parents=True,exist_ok=True);continue
            assert member.isfile() and not dest.exists(),member.name
            data=tf.extractfile(member).read();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            members.append(dict(path=member.name,bytes=len(data),sha256=sha256(data).hexdigest()))
    expected=json.loads((root/PREFIX/archive['member_manifest']).read_text())
    assert sorted(members,key=lambda x:x['path'])==sorted(expected,key=lambda x:x['path'])
    (HERE/'public_source_and_restore_verification.json').write_text(json.dumps(dict(public_HEAD=SHA,measured_SOURCE=inventory['measured_SOURCE_commit'],public_files=receipts,archive=archive,restored_files=members,all_equal=True),indent=2)+'\n')
    print(json.dumps(dict(public_files=len(receipts),archive_bytes=len(raw),restored_files=len(members),all_equal=True)))
if __name__=='__main__':main()
