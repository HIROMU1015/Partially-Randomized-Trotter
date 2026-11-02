#!/usr/bin/env python3
"""Use the prior reversible QPY-aware tar format for this fresh study."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pauli_hybrid_rte_comparison/2026-10-25'
SPEC=importlib.util.spec_from_file_location('prior_lossless_pack',ROOT/'scripts/pack_pauli_h2_h6_evidence.py')
P=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(P)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def main():
    assert json.loads((OUT/'STUDY_STOP.json').read_text())['mandatory_STOP']
    inventory=[]
    for name in ('H4','H6','stages'):
        folder=OUT/name
        if not folder.exists():continue
        path=OUT/(name+'_evidence.tar.xz');members={}
        with path.open('xb') as target:
            process=subprocess.Popen(['xz','--threads=0','-6','-c'],stdin=subprocess.PIPE,stdout=target)
            with tarfile.open(fileobj=process.stdin,mode='w|') as tar:
                for f in sorted(folder.rglob('*')):
                    if not f.is_file():continue
                    rel=str(f.relative_to(OUT));raw=f.read_bytes();stored=raw;arc=rel
                    meta=dict(sha256=sha(raw),bytes=len(raw),archive_member=arc,encoding='identity')
                    if rel.endswith('.qpy.gz'):
                        import gzip
                        stored=gzip.decompress(raw)
                        assert P.gz(stored)==raw,'GZIP_NOT_REVERSIBLE'
                        arc=rel[:-3];meta.update(archive_member=arc,encoding='gzip-level9-mtime0-os3',
                                               stored_sha256=sha(stored),stored_bytes=len(stored))
                    members[rel]=meta;info=tarfile.TarInfo(arc);info.size=len(stored);info.mtime=0;info.mode=0o644
                    tar.addfile(info,io.BytesIO(stored))
            process.stdin.close();exit_code=process.wait()
        assert exit_code==0
        parts=[]
        with path.open('rb') as stream:
            index=0
            while raw:=stream.read(8*1024*1024):
                part=OUT/(name+f'_evidence.tar.xz.part{index:03d}');part.write_bytes(raw)
                parts.append(dict(path=part.name,sha256=sha(raw),bytes=len(raw)));index+=1
        inventory.append(dict(namespace=name,format='same QPY-aware tar.xz as H2/H6; generic prior restore script',
                              archive_sha256=sha(path.read_bytes()),archive_bytes=path.stat().st_size,
                              parts=parts,members=members,xz_native_exit=exit_code,xz_native_reaped=True,
                              original_raw_left_untouched=True))
        print(name,path.stat().st_size,len(members),flush=True)
    (OUT/'data_archives.json').write_text(json.dumps(inventory,indent=2)+'\n')


if __name__=='__main__':main()
