"""Read-only packaging of the one completed/stopped run; no science imports.

Copies primary results and archives every raw byte, including cache and logs.
Refuses a live manager or a changed file during capture. Never alters the run.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile

LIVE=Path('/tmp/scv-signed-support-448-preparation-20261107')
REL=Path('artifacts/scv_signed_support_fresh_pilot/2026-11-07')
DEST=Path('artifacts/scv_signed_support_448_results/2026-11-07/run01')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    base=LIVE/REL;out=base/'campaign_run01';manager=base/'manager_run01'
    if not (out/'STOP.json').is_file() or not ((manager/'DISPATCH_EXIT.json').is_file() or (manager/'DISPATCH_STOP.json').is_file()):
        raise RuntimeError('one campaign must have stopped and manager must have exited')
    if not (out/'WORKER_EXIT_CODES.json').is_file() and list(out.glob('WORKER_[0-9]*.json')):
        raise RuntimeError('worker exit evidence missing; inspect before packaging')
    DEST.mkdir(parents=True,exist_ok=False)
    paths=sorted(p for sub in ('campaign_run01','manager_run01') for p in (base/sub).rglob('*') if p.is_file())
    records=[dict(path=str(p.relative_to(base)),bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
    archive=DEST/'RAW_COMPLETE.tar.gz'
    with archive.open('xb') as output:
        with gzip.GzipFile(filename='',mode='wb',fileobj=output,mtime=0,compresslevel=6) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|') as tar:
                for p,row in zip(paths,records):
                    info=tar.gettarinfo(str(p),arcname=row['path']);info.uid=info.gid=0;info.uname=info.gname=''
                    with p.open('rb') as stream:tar.addfile(info,stream)
        output.flush();os.fsync(output.fileno())
    for p,row in zip(paths,records):
        if digest(p)!=row['sha256']:raise RuntimeError('raw file changed during package: '+str(p))
    by={r['path']:r for r in records};seen=[]
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            assert member.isfile()
            data=tar.extractfile(member).read();row=by[member.name]
            if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:raise ValueError('archive body mismatch')
            seen.append(member.name)
    assert set(seen)==set(by) and len(seen)==len(by)
    primary=[]
    for p,row in zip(paths,records):
        # Human/machine primary records stay directly inspectable. The complete
        # original logs/word tapes/cache are in the verified archive.
        if p.parent in (out,manager) and p.suffix in ('.json','.txt'):
            target=DEST/row['path'];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);primary.append(row['path'])
    manifest=dict(schema='scv448_raw_preservation_v1',source_commit='f70e555fbfe08676f530d989192aade75a57ab1a',
        live_root=str(base),archive='RAW_COMPLETE.tar.gz',archive_bytes=archive.stat().st_size,archive_sha256=digest(archive),
        files=records,all_original_bytes_verified=True,primary_copies=primary,original_files_modified=False,
        expected_cost_guarantee='UNKNOWN',science_retry=0)
    (DEST/'RAW_MANIFEST.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(files=len(records),raw_bytes=sum(r['bytes'] for r in records),archive_bytes=archive.stat().st_size,
        archive_sha256=manifest['archive_sha256'],primary_files=len(primary),all_original_bytes_verified=True)))


if __name__=='__main__':main()
