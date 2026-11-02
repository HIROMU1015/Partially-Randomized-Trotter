#!/usr/bin/env python3
"""Narrow H6 archive connection using the existing QPY/restore format.

Operational logs with private absolute paths stay local; PUBLIC_NATIVE_AUDIT
contains the public exit/reap evidence. Original scientific raw stays untouched.
"""
import gzip,hashlib,importlib.util,io,json,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/h6_hybrid_fairness_extension/2026-10-27'
s=importlib.util.spec_from_file_location('existing_qpy_format',ROOT/'scripts/pack_pauli_h2_h6_evidence.py');P=importlib.util.module_from_spec(s);s.loader.exec_module(P)
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    assert json.loads((OUT/'verification.json').read_text())['status']=='PASS'
    spec=importlib.util.spec_from_file_location('pack_free_cpu',ROOT/'scripts/run_h6_hybrid_fairness_extension.py');W=importlib.util.module_from_spec(spec);spec.loader.exec_module(W)
    capacity=W.D.idle_capacity();threads=capacity['workers_from_actual_free_CPU']
    (OUT/'archive_resources.json').write_text(json.dumps(capacity,indent=2)+'\n')
    path=OUT/'H6_evidence.tar.xz';members={}
    with path.open('xb') as target:
        child=subprocess.Popen(['nice','-n','10','xz',f'--threads={threads}','-3','-c'],stdin=subprocess.PIPE,stdout=target)
        with tarfile.open(fileobj=child.stdin,mode='w|') as tar:
            for f in sorted((OUT/'H6').rglob('*')):
                if not f.is_file():continue
                rel=str(f.relative_to(OUT));raw=f.read_bytes();stored=raw;arc=rel
                meta=dict(sha256=digest(raw),bytes=len(raw),archive_member=arc,encoding='identity')
                if rel.endswith('.qpy.gz'):
                    stored=gzip.decompress(raw);assert P.gz(stored)==raw
                    arc=rel[:-3];meta.update(archive_member=arc,encoding='gzip-level9-mtime0-os3',stored_sha256=digest(stored),stored_bytes=len(stored))
                members[rel]=meta;info=tarfile.TarInfo(arc);info.size=len(stored);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(stored))
        child.stdin.close();rc=child.wait()
    assert rc==0
    parts=[]
    with path.open('rb') as f:
        i=0
        while raw:=f.read(8*1024*1024):
            p=OUT/f'H6_evidence.tar.xz.part{i:03d}';p.write_bytes(raw);parts.append(dict(path=p.name,bytes=len(raw),sha256=digest(raw)));i+=1
    manifest=[dict(namespace='H6',format='existing reversible QPY-aware tar.xz',archive_sha256=digest(path.read_bytes()),archive_bytes=path.stat().st_size,parts=parts,members=members,xz_native_exit=rc,xz_native_reaped=True,original_raw_untouched=True,private_operational_logs_excluded=True)]
    (OUT/'data_archives.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Validate the existing restoration transformation without a second large
    # raw copy: same SHA checks as restore_pauli_h2_h6_evidence.py, in memory.
    by_arc={v['archive_member']:(k,v) for k,v in members.items()};seen=set()
    with tarfile.open(path,'r:xz') as tar:
        for member in tar:
            rel,meta=by_arc[member.name];b=tar.extractfile(member).read()
            if meta['encoding']!='identity':assert digest(b)==meta['stored_sha256'];b=P.gz(b)
            assert digest(b)==meta['sha256'];seen.add(rel)
    assert seen==set(members)
    (OUT/'archive_restore_verification.json').write_text(json.dumps(dict(status='PASS',restored_original_SHA_checks=len(seen),all_original_bytes_reconstructible=True,original_raw_untouched=True,second_full_raw_copy_avoided=True),indent=2)+'\n')
    print('ARCHIVE_RESTORE_PASS',path.stat().st_size,len(members),flush=True)
if __name__=='__main__':main()
