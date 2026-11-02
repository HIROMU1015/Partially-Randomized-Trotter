"""Read-only parts + new destination; safe restore verifies every member byte."""
from pathlib import Path
from hashlib import sha256
import argparse,json,tarfile,tempfile,shutil
HERE=Path(__file__).resolve().parent

def restore(dest):
    dest=Path(dest);dest.mkdir(exist_ok=False);m=json.loads((HERE/'artificial_raw_manifest.json').read_text());h=sha256();size=0
    with tempfile.TemporaryFile()as archive:
        for part in m['parts']:
            p=HERE/part['path'];raw=p.read_bytes();assert len(raw)==part['bytes']and sha256(raw).hexdigest()==part['sha256'];archive.write(raw);h.update(raw);size+=len(raw)
        assert size==m['archive_bytes']and h.hexdigest()==m['archive_sha256'];archive.seek(0);expected={r['path']:r for r in m['members']};seen=set()
        with tarfile.open(fileobj=archive,mode='r:gz')as tar:
            for row in tar:
                assert row.isfile()and row.name in expected and row.name not in seen and not Path(row.name).is_absolute()and'..'not in Path(row.name).parts
                spec=expected[row.name];raw=tar.extractfile(row).read();assert len(raw)==spec['bytes']and sha256(raw).hexdigest()==spec['sha256'];p=dest/row.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);seen.add(row.name)
        assert seen==set(expected)
    # Restore again as negative fixtures by overlaying only onto a NEW normal copy.
    for overlay in (dest/'mutations').glob('*/overlay.json'):
        meta=json.loads(overlay.read_text());target=dest/'restored_mutations'/overlay.parent.name;shutil.copytree(dest/'cases'/meta['base_case'],target)
        for name in meta['deleted_files']:(target/name).unlink()
        for spec in meta['changed_files']:
            p=target/spec['path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(overlay.parent/spec['path'],p);assert sha256(p.read_bytes()).hexdigest()==spec['sha256']
    return dict(status='PASS',archive_sha256=m['archive_sha256'],member_files=len(seen),bytes=sum(r['bytes']for r in expected.values()),mutation_overlays_restored=len(list((dest/'mutations').glob('*/overlay.json'))),new_directory=str(dest))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--new-directory',required=True);a=p.parse_args();print(json.dumps(restore(a.new_directory),indent=2))
