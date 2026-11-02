"""Restore immutable artificial parts/members and bundle overlays; new directory only."""
from pathlib import Path
from hashlib import sha256
import argparse,json,tarfile,tempfile,shutil,os
HERE=Path(__file__).resolve().parent
H='62967269e611b61e73b70abe20e998d9c7eaaf50'
Q='docs/research/track_b_g6_w2c_b3_closed_port_bundle_integration_20261019'
p=argparse.ArgumentParser();p.add_argument('--new-directory',required=True);a=p.parse_args();dest=Path(a.new_directory).resolve()
assert not dest.exists() and not dest.is_relative_to(HERE) and not HERE.is_relative_to(dest)
D=HERE/'fixed_reference'/H/Q/'evidence';manifest=json.loads((D/'archive_manifest.json').read_text());expected={r['path']:r for r in manifest['members']};assert len(expected)==len(manifest['members']);dest.mkdir();seen=set();h=sha256()
with tempfile.TemporaryFile() as f:
 for part in manifest['parts']:
  b=(D/Path(part['path']).name).read_bytes();assert len(b)==part['bytes'] and sha256(b).hexdigest()==part['sha256'];f.write(b);h.update(b)
 assert f.tell()==manifest['archive_bytes'] and h.hexdigest()==manifest['archive_sha256'];f.seek(0)
 with tarfile.open(fileobj=f,mode='r:gz') as tf:
  for row in tf:
   path=Path(row.name);assert row.isfile() and not path.is_absolute() and '..' not in path.parts and row.name in expected and row.name not in seen
   b=tf.extractfile(row).read();r=expected[row.name];assert len(b)==r['bytes'] and sha256(b).hexdigest()==r['sha256'];out=dest/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b);seen.add(row.name)
 assert seen==set(expected)
for p in (dest/'overlays').glob('*/overlay.json'):
 m=json.loads(p.read_text());out=dest/'restored_overlays'/p.parent.name;shutil.copytree(dest/'cases'/m['base_case'],out,copy_function=os.link)
 for name in m['deleted_files']:
  assert not Path(name).is_absolute() and '..' not in Path(name).parts;(out/name).unlink()
 for row in m['changed_files']:
  name=row['path'];assert not Path(name).is_absolute() and '..' not in Path(name).parts;f=out/name
  if f.exists():f.unlink()
  f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p.parent/name,f);assert sha256(f.read_bytes()).hexdigest()==row['sha256']
receipt=dict(parts=len(manifest['parts']),bytes=manifest['archive_bytes'],sha256=h.hexdigest(),members=len(seen),all_equal=True,real_science=False)
(dest/'restore_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
