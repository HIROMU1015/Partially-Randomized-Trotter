"""Verify new local raw parts/members, optionally materialize18 negative overlays."""
from pathlib import Path
from hashlib import sha256
import argparse,json,tarfile,tempfile,shutil,os
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--new-directory',required=True);p.add_argument('--public-raw');a=p.parse_args();dest=Path(a.new_directory).resolve()
 assert not dest.exists()and not dest.is_relative_to(HERE)and not HERE.is_relative_to(dest);dest.mkdir()
 manifest=json.loads((HERE/'local_artificial_raw_manifest.json').read_text());expected={r['path']:r for r in manifest['members']};seen=set();h=sha256();size=0
 assert len(expected)==len(manifest['members'])
 with tempfile.TemporaryFile()as f:
  for part in manifest['parts']:
   b=(HERE/part['path']).read_bytes();assert len(b)==part['bytes']and sha256(b).hexdigest()==part['sha256'];f.write(b);h.update(b);size+=len(b)
  assert size==manifest['archive_bytes']and h.hexdigest()==manifest['archive_sha256'];f.seek(0)
  with tarfile.open(fileobj=f,mode='r:gz')as tf:
   for row in tf:
    path=Path(row.name);assert row.isfile()and not path.is_absolute()and '..'not in path.parts and row.name in expected and row.name not in seen
    b=tf.extractfile(row).read();r=expected[row.name];assert len(b)==r['bytes']and sha256(b).hexdigest()==r['sha256'];out=dest/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b);seen.add(row.name)
  assert seen==set(expected)
 count=0
 if a.public_raw:
  public=Path(a.public_raw).resolve();assert not dest.is_relative_to(public)and not public.is_relative_to(dest)
  for p in sorted((dest/'new_mutations').glob('*/overlay.json')):
   meta=json.loads(p.read_text());out=dest/'restored_new_mutations'/p.parent.name;shutil.copytree(public/'cases'/meta['base_case'],out,copy_function=os.link)
   for name in meta['deleted_files']:
    assert not Path(name).is_absolute()and '..'not in Path(name).parts;(out/name).unlink()
   for r in meta['changed_files']:
    name=r['path'];assert not Path(name).is_absolute()and '..'not in Path(name).parts;target=out/name
    if target.exists():target.unlink()
    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p.parent/name,target);assert sha256(target.read_bytes()).hexdigest()==r['sha256']
   count+=1
 r=dict(schema='local_port_bundle_artificial_archive_roundtrip_v1',archive_bytes=size,archive_sha256=h.hexdigest(),members=len(seen),all_members_equal=True,new_mutations_materialized=count,real_scientific_calls=0)
 (dest/'restore_receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
