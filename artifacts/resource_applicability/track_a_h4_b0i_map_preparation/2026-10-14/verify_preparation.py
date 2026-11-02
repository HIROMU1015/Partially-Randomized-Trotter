"""Public byte/binding/domain verifier; no NPZ, native process or scientific import."""
import hashlib
import json
from pathlib import Path
import resource
import signal
import subprocess
import sys

BUNDLE=Path(__file__).absolute().parent;ROOT=BUNDLE.parents[3];sys.path.insert(0,str(ROOT/'src'))


def main():
    resource.setrlimit(resource.RLIMIT_AS,(256*2**20,256*2**20));resource.setrlimit(resource.RLIMIT_CPU,(30,30));signal.alarm(60)
    from trottertracks.resource_applicability.h4_b0i_map import contract as c,runtime as r
    plan=json.loads((BUNDLE/'plan.json').read_text());auth=json.loads((BUNDLE/'authorization_draft.json').read_text());review=json.loads((BUNDLE/'review_draft.json').read_text());binding=json.loads((BUNDLE/'binding.json').read_text())
    c.validate_plan(plan);r.source_gate(ROOT,plan);mapped=r.reuse_gate(ROOT,plan)
    assert plan['approved'] is auth['approved'] is review['approved'] is False
    assert plan['runtime_authorization'] is auth['runtime_authorization'] is False
    assert plan['allowed_cpus']==auth['allowed_cpus']==[] and plan['absolute_launch_command'] is auth['absolute_launch_command'] is None
    assert binding['technical_bound'] is True and binding['runtime_sealed'] is False
    assert c.digest(plan)==auth['plan_digest']==review['plan_digest']==binding['plan_digest']
    assert c.digest(auth)==review['authorization_digest']==binding['authorization_digest']
    assert c.digest(review)==binding['review_digest']
    for item in plan['source_closure']:
        data=subprocess.check_output(['git','show',plan['source_commit']+':'+item['path']],cwd=ROOT)
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    freeze=json.loads((BUNDLE/'frozen_inputs_byte_binding.json').read_text())
    assert set(freeze['inputs'])==set(c.GEOMETRIES) and freeze['scientific_arrays_decoded'] is False
    for t in plan['tasks']:
        for key in ('input_file','input_identity'):assert t[key]==freeze['inputs'][t['geometry']][key]
    budget=json.loads((BUNDLE/'storage_and_budget_projection.json').read_text())
    assert budget['total_actual_upper']==plan['carry']['actual_invocations']+234==73263
    assert budget['observer_logical_bound']<plan['caps']['observer_output_bytes']
    assert budget['task_and_reservation_inodes']+32<plan['caps']['run_inode_cap']
    tests=json.loads((BUNDLE/'limited_contract_tests.json').read_text())
    assert tests['passed'] and tests['tests']==19 and tests['failures']==tests['errors']==tests['skipped']==0
    assert all(tests[k]==0 for k in ('real_compile_calls','artificial_compile_calls','scientific_imports','scientific_array_reads','process_spawns','affinity_changes'))
    if (BUNDLE/'artifact_manifest.json').exists():
        for item in json.loads((BUNDLE/'artifact_manifest.json').read_text())['files']:
            data=(ROOT/item['path']).read_bytes();assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
    print(json.dumps(dict(status='PASS_CLOSED_240_PREPARATION_PUBLIC_BINDING',source=plan['source_commit'],source_files=len(plan['source_closure']),wrappers=240,new_compile_registry=234,reuse_static_proofs=len(mapped),numeric_reuse_check_deferred=True,artificial_mock_tests=tests['tests'],new_compute_calls=0,runtime_authorization=False,independent_external_review=False)))


if __name__=='__main__':main()
