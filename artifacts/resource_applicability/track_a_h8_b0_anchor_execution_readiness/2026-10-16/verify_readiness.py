"""Read-only public readiness checker, no molecular inputs, grants or workers."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def main():
    manifest=json.loads((HERE/'publication_files_v1.json').read_text())
    for row in manifest['files']:
        p=Path(row['path'])
        if p.is_absolute() or '..' in p.parts:raise ValueError('PUBLIC_PATH')
        raw=(ROOT/p).read_bytes()
        if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:
            raise ValueError('PUBLIC_BYTES_CHANGED:'+str(p))
    binding=json.loads((HERE/'source_input_profile_binding_v1.json').read_text())
    g=json.loads((HERE/'grant_INVALID_DRAFT_v1.json').read_text())
    if (binding['SOURCE']!='572a337d0dc3fa78b2fd6b25ace6a37845db384e' or
            binding['execution_source_bound'] is not True or g['SOURCE']!=binding['SOURCE'] or
            g['approved_by_user'] is not False or g['runtime_authorization'] is not False or
            g['valid'] is not False or g['allowed_cpus']!=[] or g['command'] is not None or
            g['config_sha256']!=binding['config_sha256']):raise ValueError('SEALED_BUT_UNAPPROVED_BINDING')
    for name,h in binding['source_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise ValueError('EXECUTION_SOURCE_CHANGED')
    print(json.dumps(dict(status='PASS_PREPARED_NOT_AUTHORIZED',SOURCE=binding['SOURCE'],
        public_files=len(manifest['files']),real_H8_compile=0,new_fit=0,mandatory_stop=True)))


if __name__=='__main__':main()
