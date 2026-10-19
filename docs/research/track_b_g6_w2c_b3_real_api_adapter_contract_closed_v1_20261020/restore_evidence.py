"""Verify every archive part/member and restore into a fresh directory."""
from hashlib import sha256
from pathlib import Path
import io,json,sys,tarfile,tempfile
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    dest=Path(sys.argv[1]);dest.mkdir(exist_ok=False)
    m=json.loads((HERE/'evidence/archive_manifest.json').read_text());whole=sha256();count=0
    with tempfile.TemporaryFile() as stream:
        for row in m['parts']:
            raw=(HERE/row['path']).read_bytes();assert len(raw)==row['bytes'] and sha256(raw).hexdigest()==row['sha256']
            whole.update(raw);count+=len(raw);stream.write(raw)
        assert count==m['archive_bytes'] and whole.hexdigest()==m['archive_sha256'];stream.seek(0)
        expected={row['path']:row for row in m['members']};seen=set()
        with tarfile.open(fileobj=stream,mode='r|gz') as tar:
            for member in tar:
                assert member.isfile() and member.name in expected and member.name not in seen
                target=dest/member.name;assert target.resolve().is_relative_to(dest.resolve())
                row=expected[member.name];raw=tar.extractfile(member).read()
                assert len(raw)==row['bytes'] and sha256(raw).hexdigest()==row['sha256']
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);seen.add(member.name)
        assert seen==set(expected)
    print(json.dumps(dict(status='PASS',members=len(seen),archive_sha256=m['archive_sha256'])))
