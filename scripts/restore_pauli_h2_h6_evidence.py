#!/usr/bin/env python3
"""Restore the published evidence in an empty directory, checking every SHA."""
import gzip,hashlib,io,json,sys,tarfile
from pathlib import Path
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
 manifest=Path(sys.argv[1]);dest=Path(sys.argv[2]).resolve();dest.mkdir(exist_ok=False)
 for archive in json.loads(manifest.read_text()):
  parts=[]
  for part in archive['parts']:
   b=(manifest.parent/part['path']).read_bytes()
   if digest(b)!=part['sha256']:raise ValueError('PART_SHA')
   parts.append(b)
  stream=b''.join(parts)
  if digest(stream)!=archive['archive_sha256']:raise ValueError('ARCHIVE_SHA')
  by_stored={v['archive_member']:(p,v) for p,v in archive['members'].items()}
  with tarfile.open(fileobj=io.BytesIO(stream),mode='r:xz') as tar:
   seen=set()
   for member in tar:
    original,meta=by_stored[member.name];path=(dest/original).resolve()
    if not member.isfile() or not path.is_relative_to(dest):raise ValueError('ARCHIVE_PATH')
    b=tar.extractfile(member).read()
    if meta['encoding']!='identity':
     if digest(b)!=meta['stored_sha256']:raise ValueError('QPY_SHA')
     b=bytearray(gzip.compress(b,compresslevel=9,mtime=0));b[9]=3;b=bytes(b)
    if digest(b)!=meta['sha256']:raise ValueError('ORIGINAL_SHA '+original)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b);seen.add(original)
   if seen!=set(archive['members']):raise ValueError('MISSING_MEMBERS')
 print('All original evidence bytes restored and verified.')
if __name__=='__main__':main()
