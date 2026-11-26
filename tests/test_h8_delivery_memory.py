import base64
import hashlib
from types import SimpleNamespace

from trottertracks import rpe_pbs as pbs
from trottertracks.rpe_preparation_artifacts import ArtifactCache


def test_inline_plan_inputs_preserve_claim_and_legacy_cache(tmp_path):
    cache = ArtifactCache(tmp_path/'central')
    local = ArtifactCache(tmp_path/'worker')
    manager = object.__new__(pbs.PbsManager)
    manager.executor = SimpleNamespace(artifact_cache=cache)
    raw = b'fixed small PLAN'
    sha = hashlib.sha256(raw).hexdigest()
    cache.put({sha: base64.b64encode(raw).decode()})
    request = {'preparation': {'provider': 'HCHAIN_SCOPED', 'inputs': {'PLAN.json': sha}}}
    claim = dict(job_id='fixed-job', attempt_id='fixed-attempt', request=request,
                 artifact_manifest={'PLAN.json': sha}, request_sha256='original')
    result = {'assignments': [claim], 'stop': False, 'eligibility_challenges': []}
    reply = manager._inline_hchain_plan_inputs({'cached_artifact_sha256': []}, result)
    delivered = dict(reply['assignments'][0])
    # Exactly the legacy worker's acquisition order: pop transient blobs,
    # verify/cache them, then compare the durable assignment.
    local.put(delivered.pop('artifact_blobs'))
    assert delivered == claim
    assert 'artifact_blobs' not in claim
    assert local.missing([sha]) == []
    assert manager._inline_hchain_plan_inputs({'cached_artifact_sha256': [sha]}, result) == result
    assert manager._inline_hchain_plan_inputs({}, result) == result
    assert manager._inline_hchain_plan_inputs({'cached_artifact_sha256': []}, result) == reply
    other = dict(claim, artifact_manifest={'PLAN.json':'invalid'})
    assert manager._inline_hchain_plan_inputs({'cached_artifact_sha256': []},
           {'assignments':[other]})['assignments'] == [other]


def test_inline_large_missing_and_non_hchain_remain_on_get_path(tmp_path):
    cache = ArtifactCache(tmp_path)
    manager = object.__new__(pbs.PbsManager)
    manager.executor = SimpleNamespace(artifact_cache=cache)
    raw = b'z' * (64*1024+1)
    sha = hashlib.sha256(raw).hexdigest()
    cache.put({sha: base64.b64encode(raw).decode()})
    def claim(provider, sha):
        return dict(request={'preparation':{'provider':provider,'inputs':{'PLAN.json':sha}}},
                    artifact_manifest={'PLAN.json':sha})
    result={'assignments':[claim('HCHAIN_SCOPED',sha),claim('HCHAIN_SCOPED','0'*64),
                           claim('PF_PRICE',sha)]}
    assert manager._inline_hchain_plan_inputs({'cached_artifact_sha256':[]},result)==result


def test_reduced_reservation_applies_to_new_claims_only(monkeypatch):
    G=pbs.GIB
    prep={'provider':'HCHAIN_SCOPED','source':{'commit':'fixed'},'stage':'compile','inputs':{'PLAN.json':'p'}}
    table={'source':'fixed','stages':{'compile':{'p':{'memory_bytes':7*G//2,'previous_memory_bytes':4*G}}},
           'preserve_assignments_before':50,'preserve_adjusted_assignments_before':100}
    settings={'hchain_preparation_memory':table}
    monkeypatch.setattr(pbs,'resource_need',lambda request,profile,settings,host:
                        (1, pbs.task_memory_estimate(request,settings) if settings else 315*G))
    row=dict(request={'preparation':prep},state='RUNNING',detail={'pid':1},assigned_epoch=40)
    assert pbs.owned_resource_need(row,{},settings,'pbs')==(1,315*G)
    row['assigned_epoch']=90
    assert pbs.owned_resource_need(row,{},settings,'pbs')==(1,4*G)
    row['assigned_epoch']=101
    assert pbs.owned_resource_need(row,{},settings,'pbs')==(1,7*G//2)
    row['detail']={'pbs_launch_resources':{'cpu':1,'memory_bytes':4*G}}
    assert pbs.owned_resource_need(row,{},settings,'pbs')==(1,4*G)
