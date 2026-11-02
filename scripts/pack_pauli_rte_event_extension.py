#!/usr/bin/env python3
"""Reuse the existing reversible QPY-aware archive and generic restore format."""
import gzip,hashlib,importlib.util,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pauli_rte_event_extension/2026-10-26'
s=importlib.util.spec_from_file_location('prior_qpy_pack',ROOT/'scripts/pack_pauli_h2_h6_evidence.py')
P=importlib.util.module_from_spec(s);s.loader.exec_module(P)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    assert json.loads((OUT/'STUDY_STOP.json').read_text())['mandatory_STOP']
    inventory=[]
    for name in ['cases','stages','confirmations','inputs']:
        folder=OUT/name
        if not folder.exists():continue
        archive=OUT/(name+'_evidence.tar.xz');members={}
        with archive.open('xb') as target:
            process=subprocess.Popen(['xz','--threads=0','-6','-c'],stdin=subprocess.PIPE,stdout=target)
            with tarfile.open(fileobj=process.stdin,mode='w|') as tar:
                for f in sorted(folder.rglob('*')):
                    if not f.is_file():continue
                    rel=str(f.relative_to(OUT));raw=f.read_bytes();stored=raw;arc=rel
                    meta=dict(sha256=sha(raw),bytes=len(raw),archive_member=arc,encoding='identity')
                    if rel.endswith('.qpy.gz'):
                        stored=gzip.decompress(raw);assert P.gz(stored)==raw,'GZIP_NOT_REVERSIBLE'
                        arc=rel[:-3];meta.update(archive_member=arc,encoding='gzip-level9-mtime0-os3',stored_sha256=sha(stored),stored_bytes=len(stored))
                    members[rel]=meta;info=tarfile.TarInfo(arc);info.size=len(stored);info.mtime=0;info.mode=0o644
                    tar.addfile(info,io.BytesIO(stored))
            process.stdin.close();rc=process.wait()
        assert rc==0
        parts=[]
        with archive.open('rb') as stream:
            index=0
            while raw:=stream.read(8*1024*1024):
                p=OUT/(name+f'_evidence.tar.xz.part{index:03d}');p.write_bytes(raw)
                parts.append(dict(path=p.name,sha256=sha(raw),bytes=len(raw)));index+=1
        inventory.append(dict(namespace=name,format='prior reversible QPY-aware tar.xz; restore_pauli_h2_h6_evidence.py',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,parts=parts,members=members,xz_native_exit=rc,xz_native_reaped=True,original_raw_left_untouched=True))
        print(name,archive.stat().st_size,len(members),flush=True)
    (OUT/'data_archives.json').write_text(json.dumps(inventory,indent=2)+'\n')
if __name__=='__main__':main()
