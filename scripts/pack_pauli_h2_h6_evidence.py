#!/usr/bin/env python3
"""Lossless QPY-aware archive: avoid compressing already compressed data.

Original raw files are read only. QPY bytes are stored uncompressed inside
the tar stream so xz can reuse its dictionary across related circuits. The
original deterministic gzip containers are verified reconstructible first.
"""
import gzip,hashlib,io,json,os,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/pauli_h2_h6_comparison/2026-10-25'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def gz(raw):
 a=bytearray(gzip.compress(raw,compresslevel=9,mtime=0));a[9]=3;return bytes(a)
def main():
 inventory=[]
 for n in [2,6]:
  root=OUT/f'H{n}';folders=sorted([*root.glob('batch*'),*root.glob('precision_refinement_*'),root/'confirmation01'])
  path=OUT/f'H{n}_evidence.tar.xz';members={}
  with path.open('wb') as target:
   p=subprocess.Popen(['xz','--threads=0','-6','-c'],stdin=subprocess.PIPE,stdout=target)
   with tarfile.open(fileobj=p.stdin,mode='w|') as tar:
    for folder in folders:
     for f in sorted(folder.rglob('*')):
      if not f.is_file():continue
      rel=str(f.relative_to(OUT));raw=f.read_bytes();stored=raw;arc=rel
      meta=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),archive_member=arc,encoding='identity')
      if rel.endswith('.qpy.gz'):
       stored=gzip.decompress(raw)
       if gz(stored)!=raw:raise ValueError('GZIP_NOT_REVERSIBLE '+rel)
       arc=rel[:-3];meta.update(archive_member=arc,encoding='gzip-level9-mtime0-os3',stored_sha256=hashlib.sha256(stored).hexdigest(),stored_bytes=len(stored))
      members[rel]=meta;info=tarfile.TarInfo(arc);info.size=len(stored);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(stored))
   p.stdin.close();rc=p.wait()
   if rc:raise RuntimeError('XZ_NATIVE_FAILURE')
  parts=[]
  with path.open('rb') as f:
   i=0
   while data:=f.read(8*1024*1024):
    part=OUT/f'H{n}_evidence.tar.xz.part{i:03d}';part.write_bytes(data);parts.append(dict(path=part.name,sha256=sha(part),bytes=len(data)));i+=1
  inventory.append(dict(molecule=n,format='QPY-aware tar.xz; restore with scripts/restore_pauli_h2_h6_evidence.py into a fresh directory',
    archive_sha256=sha(path),archive_bytes=path.stat().st_size,parts=parts,members=members,
    QPY_raw_bytes_and_original_gzip_containers_losslessly_preserved=True,xz_native_exit=rc,xz_native_reaped=True))
  print(n,path.stat().st_size,len(parts),flush=True)
 old=OUT/'data_archives.json';backup=OUT/'superseded_data_archives_gzip.json'
 if old.exists() and not backup.exists():backup.write_bytes(old.read_bytes())
 old.write_text(json.dumps(inventory,indent=2)+'\n')
if __name__=='__main__':main()
