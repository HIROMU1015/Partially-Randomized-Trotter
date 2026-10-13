"""Public-only source/profile/task/binding audit; never builds or launches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(args.repo/'src'))
    from trottertracks.resource_applicability.h4_b0i.contract import digest,validate_plan,authorize
    from trottertracks.resource_applicability.h4_geometry.identity import Stop,fingerprint
    b=args.bundle;plan=json.loads((b/'plan.json').read_text());validate_plan(plan)
    auth=json.loads((b/'authorization_draft.json').read_text());review=json.loads((b/'review_draft.json').read_text())
    binding=json.loads((b/'binding.json').read_text())
    assert binding['source_commit']==plan['source_commit']
    assert (binding['plan_digest'],binding['authorization_digest'],binding['review_digest'])==(digest(plan),digest(auth),digest(review))
    assert auth['plan_digest']==review['plan_digest']==digest(plan) and review['authorization_digest']==digest(auth)
    for value in (plan,auth):
        assert value['approved'] is False and value['runtime_authorization'] is False
        assert value['allowed_cpus']==[] and value['absolute_launch_command'] is None
        assert value['retry'] is False and value['automatic_scope_expansion'] is False
        assert value['next_stage_authorized'] is False and value['mandatory_stop'] is True
    assert review['approved'] is False and binding['runtime_sealed'] is False
    try:authorize(plan,auth,review)
    except Stop:pass
    else:raise AssertionError('preparation unexpectedly authorizes runtime')
    for item in plan['source_closure']:
        data=subprocess.check_output(['git','show',plan['source_commit']+':'+item['path']],cwd=args.repo)
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        assert data==(args.repo/item['path']).read_bytes()
    for item in plan['bound_public_files']:
        data=(args.repo/item['path']).read_bytes()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    assert json.loads((b/'registry.json').read_text())==plan['tasks']
    arrays=json.loads((b/'frozen_array_identities_v1.json').read_text())
    for task in plan['tasks']:
        ids=arrays['geometries'][task['geometry']];assert task['frozen_array_byte_identities']==ids
        assert fingerprint('h4-H-v1',ids['H'])==task['input_identity']['H']
        assert fingerprint('h4-DF-v1',{k:ids[k] for k in ('lambdas','G','correction','permutation','raw_indices','raw_eigenvalues','raw_eigenvectors','signs')})==task['input_identity']['DF']
        assert fingerprint('h4-state-v1',{k:ids[k] for k in ('sector','sector_state','state','qiskit_state')})==task['input_identity']['state']
    test=json.loads((b/'artificial_tests_v1.json').read_text());assert test['status']=='PASS' and test['tests_run']==44 and not any(test['counts'].values())
    manifest=b/'artifact_manifest_v1.json'
    if manifest.exists():
        for name,item in json.loads(manifest.read_text())['files'].items():
            data=(args.repo/name).read_bytes()
            assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    print(json.dumps({'status':'PASS_UNAPPROVED_PUBLIC_PREPARATION','source_commit':plan['source_commit'],
        'source_files':len(plan['source_closure']),'tasks':6,'test_cases':44,
        'NPZ_arrays_build_transpile_launch_access':0,'scope':'local metadata/hash audit, not scientific review or live H4 compiler validation'}))


if __name__=='__main__':main()
