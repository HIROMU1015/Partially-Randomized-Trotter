"""Verify and restore new artificial raw to a fresh directory; execute nothing."""
from hashlib import sha256
from pathlib import Path
import argparse,json,tarfile,tempfile
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--new-directory',type=Path,required=True);args=parser.parse_args()
    manifest=json.loads((HERE/'new_artificial_raw_manifest.json').read_text());dest=args.new_directory.resolve();dest.mkdir(parents=True,exist_ok=False)
    digest=sha256();total=0;actual=[]
    with tempfile.TemporaryFile() as archive:
        for part in manifest['parts']:
            raw=(HERE/part['path']).read_bytes();assert len(raw)==part['bytes'] and sha256(raw).hexdigest()==part['sha256']
            archive.write(raw);digest.update(raw);total+=len(raw)
        assert total==manifest['bytes'] and digest.hexdigest()==manifest['sha256'];archive.seek(0)
        with tarfile.open(fileobj=archive,mode='r:gz') as tf:
            for member in tf:
                name=Path(member.name);assert member.isfile() and not name.is_absolute() and '..' not in name.parts
                path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);raw=tf.extractfile(member).read()
                with path.open('xb') as dst:dst.write(raw)
                actual.append(dict(path=member.name,bytes=len(raw),sha256=sha256(raw).hexdigest()))
    assert sorted(actual,key=lambda x:x['path'])==sorted(manifest['members'],key=lambda x:x['path'])
    print(json.dumps(dict(restored_files=len(actual),all_member_bytes_SHA256_equal=True,new_directory=str(dest))))
if __name__=='__main__':main()
