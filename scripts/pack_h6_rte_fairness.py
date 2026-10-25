#!/usr/bin/env python3
"""Store exact bytes using the existing manifest/restorer's identity encoding."""
import hashlib, io, json, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/h6_rte_fairness/2026-10-25'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    assert json.loads((OUT/'STUDY_STOP.json').read_text())['mandatory_STOP']
    archives=[]
    for folder in ['H6','confirmations']:
        p=OUT/(folder+'_evidence.tar.xz');members={}
        with p.open('xb') as output:
            with tarfile.open(fileobj=output,mode='w:xz') as tar:
                for f in sorted((OUT/folder).rglob('*')):
                    if not f.is_file():continue
                    name=str(f.relative_to(OUT));b=f.read_bytes();info=tarfile.TarInfo(name)
                    info.size=len(b);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(b))
                    members[name]=dict(sha256=sha(b),bytes=len(b),archive_member=name,encoding='identity')
        b=p.read_bytes();parts=[]
        for index,start in enumerate(range(0,len(b),8*1024*1024)):
            chunk=b[start:start+8*1024*1024];part=p.with_name(p.name+f'.part{index:03d}')
            with part.open('xb') as f:f.write(chunk)
            parts.append(dict(path=part.name,sha256=sha(chunk),bytes=len(chunk)))
        archives.append(dict(namespace=folder,format='tar.xz; existing restorer identity encoding; original gzip bytes unmodified',
            archive_sha256=sha(b),archive_bytes=len(b),parts=parts,members=members,original_raw_left_untouched=True))
        print(folder,len(b),len(members),flush=True)
    with (OUT/'data_archives.json').open('x') as f:json.dump(archives,f,indent=2);f.write('\n')
if __name__=='__main__':main()
