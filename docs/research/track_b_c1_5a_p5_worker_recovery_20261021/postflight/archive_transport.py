"""Byte-preserving transport packaging only, no scientific imports/reaggregation."""
from pathlib import Path
import json,hashlib,tarfile,io,gzip,shutil
w=Path('/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/track-b-c1-5a-p5-worker-recovery-20261021');n=w/'docs/research/track_b_c1_5a_p5_worker_recovery_20261021';r=Path(str(w)+'.runtime');out=n/'result'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def row(p):b=p.read_bytes();return dict(path=str(p.relative_to(n)),bytes=len(b),sha256=sha(b))
assert (out/'terminal.json').exists(),'Job must have terminated before packaging'
terminal=json.loads((out/'terminal.json').read_text());assert terminal['mandatory_STOP'] and terminal['retry']==0
# Root raw logs were redirected outside clean worktree during its entry gate.
for name in ('science.stdout.log','science.stderr.log'):
 target=out/name;assert not target.exists();shutil.copyfile(r/name,target)
keep={'setup.json','summary.json','producer_summary.json','terminal.json','resources_before.json','resources_after.json','producer_progress.json',
      'producer_supervision/summary.json','producer_supervision/cgroup_before.json','producer_supervision/cgroup_after.json','producer_supervision/first_failure.json',
      'worker_supervision/summary.json','worker_supervision/cgroup_before.json','worker_supervision/cgroup_after.json','worker_supervision/first_failure.json'}
allfiles=sorted(p for p in out.rglob('*') if p.is_file());selected=[p for p in allfiles if str(p.relative_to(out)) not in keep]
entries=[row(p) for p in selected];archive=n/'raw_archives';archive.mkdir()
# Deterministic tar metadata and gzip. Existing gz Evidence remain exact byte payloads.
with io.BytesIO() as buffer:
 with tarfile.open(fileobj=buffer,mode='w',format=tarfile.PAX_FORMAT) as tar:
  for p in selected:
   raw=p.read_bytes();info=tarfile.TarInfo(str(p.relative_to(n)));info.size=len(raw);info.mode=0o644;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
   tar.addfile(info,io.BytesIO(raw))
 packed=gzip.compress(buffer.getvalue(),mtime=0)
parts=[];size=4*1024*1024
for start in range(0,len(packed),size):
 p=archive/('P5R_raw.tar.gz.part'+str(start//size).zfill(3));p.write_bytes(packed[start:start+size]);parts.append(row(p))
assert b''.join((n/p['path']).read_bytes() for p in parts)==packed
with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as tar:
 actual={}
 for member in tar.getmembers():
  assert member.isfile() and member.name.startswith('result/') and '..' not in Path(member.name).parts
  raw=tar.extractfile(member).read();actual[member.name]=(len(raw),sha(raw));assert raw==(n/member.name).read_bytes()
 assert actual=={e['path']:(e['bytes'],e['sha256']) for e in entries}
index=dict(schema='P5R_BYTE_EXACT_TRANSPORT_V1',scientific_reanalysis=False,archive={'bytes':len(packed),'sha256':sha(packed)},parts=parts,members=entries,all_members_byte_SHA256_verified=True,old_archives_not_opened=True,
  direct_files=[row(p) for p in allfiles if p not in selected])
(archive/'index.json').write_text(json.dumps(index,indent=2)+'\n')
# Move original new raw into this task's own preservation area, only after verified archive roundtrip.
preserve=r/'raw_preserved'
for p in selected:
 target=preserve/p.relative_to(n);target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();shutil.move(str(p),str(target))
for e in entries:
 p=preserve/e['path'];assert p.stat().st_size==e['bytes'] and sha(p.read_bytes())==e['sha256']
print(json.dumps(dict(raw_members=len(entries),raw_bytes=sum(e['bytes'] for e in entries),archive_bytes=len(packed),parts=len(parts),all_match=True)))
