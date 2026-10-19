"""Restore verified public43 and35 overlays into a NEW namespace only."""
from pathlib import Path
from hashlib import sha256
import json,tarfile,tempfile,os,shutil
import argparse
a=argparse.ArgumentParser();a.add_argument('--new-directory',required=True);args=a.parse_args()
P=Path(__file__).resolve().parent;D=P/'fixed_reference/bridge/docs/research/track_b_g6_w2c_b3_v2_production_bundle_bridge_candidate_20261019';dest=Path(args.new_directory).resolve();assert not dest.is_relative_to(P) and not P.is_relative_to(dest);dest.mkdir(exist_ok=False)
m=json.loads((D/'artificial_raw_manifest.json').read_text());expected={x['path']:x for x in m['members']};seen=set();h=sha256();size=0
assert len(expected)==len(m['members'])==2022
with tempfile.TemporaryFile()as f:
 for part in m['parts']:
  b=(D/part['path']).read_bytes();assert len(b)==part['bytes']and sha256(b).hexdigest()==part['sha256'];h.update(b);size+=len(b);f.write(b)
 assert size==m['archive_bytes']==27677690 and h.hexdigest()==m['archive_sha256']=='2d38d8b0a210b4428a7116a38bb44420c5a709a07861821c3b1444f44106a208';f.seek(0)
 with tarfile.open(fileobj=f,mode='r:gz')as tf:
  for x in tf:
   p=Path(x.name);assert x.isfile()and not p.is_absolute()and '..'not in p.parts and x.name in expected and x.name not in seen
   spec=expected[x.name];b=tf.extractfile(x).read();assert len(b)==spec['bytes']and sha256(b).hexdigest()==spec['sha256'];out=dest/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b);seen.add(x.name)
 assert seen==set(expected)
for p in sorted((dest/'mutations').glob('*/overlay.json')):
 v=json.loads(p.read_text());out=dest/'restored_mutations'/p.parent.name;shutil.copytree(dest/'cases'/v['base_case'],out,copy_function=os.link)
 for n in v['deleted_files']:(out/n).unlink()
 for row in v['changed_files']:
  x=out/row['path'];assert not Path(row['path']).is_absolute()and '..'not in Path(row['path']).parts
  if x.exists():x.unlink()
  x.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p.parent/row['path'],x);assert sha256(x.read_bytes()).hexdigest()==row['sha256']
r=dict(schema='local_independent_bundle_restore_v1',archive_bytes=size,archive_sha256=h.hexdigest(),member_files=len(seen),member_bytes=sum(x['bytes']for x in expected.values()),all_members_equal=True,cases=len(list((dest/'cases').iterdir())),mutation_overlays=len(list((dest/'restored_mutations').iterdir())),original_fixed_bytes_modified=False,unchanged_overlay_files_shared_read_only=True)
(dest/'restore_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
