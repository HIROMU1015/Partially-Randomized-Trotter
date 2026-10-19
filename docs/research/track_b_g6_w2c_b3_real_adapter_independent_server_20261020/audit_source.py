"""Independent, stdlib-only Git/blob and artificial archive audit. No source imports."""
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HEAD = '80b7bf61fe0a735d4031bb719c4b3e788611c3ba'
SOURCE = 'b7ad344401b10395c49e4cfa806909282859bab5'
SCIENCE = '76ca3c4089b201d55da0fede08e1f77733ac38a5'
BASE = '5c1e13fa74b3c5e4f3166babed785274844655d0'
PACKET = 'docs/research/track_b_g6_w2c_b3_real_api_adapter_contract_closed_v1_20261020'
ROOT = Path('/tmp/b3-independent-server-r2-20261020')
OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])

def blob(commit, path):
    return git('show', commit + ':' + path)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n')

def main():
    inventory = json.loads(blob(HEAD, PACKET + '/source_inventory.json'))
    rows = []
    for path, expected in inventory['source_pins'].items():
        data = blob(SOURCE, path)
        rows.append(dict(path=path, bytes=len(data), sha256=sha(data),
                         expected=expected, pass_=sha(data) == expected,
                         public_HEAD_equal=data == blob(HEAD, path)))
    assert len(rows) == 89 and all(r['pass_'] and r['public_HEAD_equal'] for r in rows)
    assert inventory['artifact_identity'] == '024d8812e97c5d541391533680fdbb3aee4cf6d8e51f020d3302a00767af42cb'
    receipt = json.loads(blob(HEAD, PACKET + '/publication_receipt.json'))
    receipt_rows = []
    for row in receipt['files']:
        data = blob(receipt['content_commit'], row['path'])
        receipt_rows.append(dict(path=row['path'], bytes=len(data), sha256=sha(data),
                                 pass_=len(data) == row['bytes'] and sha(data) == row['sha256'],
                                 final_HEAD_equal=data == blob(HEAD, row['path'])))
    assert all(r['pass_'] and r['final_HEAD_equal'] for r in receipt_rows)
    names = ['g7_generator', 'g7_provider', 'return_aggregation', 'g6_f3_contract',
             'g6_f3_precision', 'g6_f3_generator', 'g6_w2c_b3_driver', 'g6_w2c_cost_preparation']
    science = []
    for name in names:
        path = 'src/trottertracks/algorithm_codesign/' + name + '.py'
        data = blob(SCIENCE, path)
        tree = ast.parse(data, filename=path)
        entries = [dict(name=n.name, line=n.lineno, kind=type(n).__name__)
                   for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        science.append(dict(path=path, bytes=len(data), sha256=sha(data),
                            measured_equal=data == blob(SOURCE, path), entries=entries))
    assert all(r['measured_equal'] for r in science)
    manifest_data = blob(HEAD, PACKET + '/evidence/archive_manifest.json')
    manifest = json.loads(manifest_data)
    archive = io.BytesIO()
    parts = []
    for row in manifest['parts']:
        data = blob(HEAD, PACKET + '/' + row['path'])
        assert len(data) == row['bytes'] and sha(data) == row['sha256']
        parts.append(dict(path=row['path'], bytes=len(data), sha256=sha(data)))
        archive.write(data)
    assert archive.tell() == manifest['archive_bytes']
    assert sha(archive.getvalue()) == manifest['archive_sha256']
    archive.seek(0)
    restored = ROOT / 'restored'
    restored.mkdir(exist_ok=False)
    expected = {m['path']:m for m in manifest['members']}
    found = []
    with tarfile.open(fileobj=archive, mode='r:gz') as tar:
        for member in tar:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert member.name in expected and member.name not in found
            data = tar.extractfile(member).read()
            row = expected[member.name]
            assert len(data) == row['bytes'] and sha(data) == row['sha256']
            dest = restored / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            found.append(member.name)
    assert set(found) == set(expected) and len(found) == 994
    # These existing paths are observed read-only, never frozen as a global policy.
    preservation = git('diff', '--name-status', SOURCE, HEAD, '--', 'src', 'tests').decode()
    assert not preservation
    save('source_inventory.json', dict(schema='independent_source_audit_v1', public_HEAD=HEAD,
         measured_SOURCE=SOURCE, independent_base=BASE, content_commit=receipt['content_commit'],
         artifact_identity=inventory['artifact_identity'], source_pins=rows,
         publication_content_files=receipt_rows, science_static_only=science,
         archive=dict(manifest_bytes=len(manifest_data), manifest_sha256=sha(manifest_data),
                      archive_sha256=manifest['archive_sha256'], archive_bytes=manifest['archive_bytes'],
                      parts=parts, member_count=len(found), member_manifest=manifest['members']),
         actual_scientific_module_imports=0))
    print(json.dumps(dict(source_pins=len(rows), public_content_files=len(receipt_rows),
                          static_scientific_modules=len(science), static_entries=sum(len(x['entries']) for x in science),
                          archive_members=len(found), restored=str(restored))))

if __name__ == '__main__':
    main()
