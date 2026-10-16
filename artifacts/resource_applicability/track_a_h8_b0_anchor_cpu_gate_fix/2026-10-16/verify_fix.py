"""Verify saved CPU-fix readiness, without probing CPUs or launching science."""
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
    binding=json.loads((HERE/'source_profile_input_binding_v1.json').read_text())
    grant=json.loads((HERE/'grant_INVALID_DRAFT_v2.json').read_text())
    assert binding['SOURCE']==grant['SOURCE']=='db559a518214fd27fa96e2f1451911df05c4f4a8'
    assert binding['execution_source_bound'] is True and binding['runtime_files_verified']==34273
    assert grant['approved_by_user'] is False and grant['runtime_authorization'] is False
    assert grant['valid'] is False and grant['allowed_cpus']==[] and grant['command'] is None
    assert binding['config_sha256']==grant['config_sha256']
    for name,h in binding['source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    contract=json.loads((HERE/'execution_contract_v2.json').read_text())
    assert len(contract['axes'])==8 and contract['CPU_admission']['maximum_busy_fraction']=='1/20'
    assert contract['CPU_admission']['automatic_reassignment'] is False
    assert contract['resources']['aggregate_RSS_bytes']+contract['resources']['admission_reserve_bytes']==int(176.25*2**30)
    print(json.dumps(dict(status='PASS_SAVED_FIX_NOT_AUTHORIZED',SOURCE=binding['SOURCE'],
        public_files=len(manifest['files']),new_compile=0,new_fit=0,mandatory_stop=True)))


if __name__=='__main__':main()
