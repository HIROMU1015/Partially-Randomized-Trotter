"""Synthetic regression for common memory telemetry and reservation continuity."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from trottertracks import rpe_pbs as P, rpe_q3_memory as M, rpe_host_profiles as H
from trottertracks.rpe_q3_worker import digest


def setup():
    host='pbs-test'; view='b'*40
    runtime=dict(host_id=host,root='/fixed/runtime',interpreter='/fixed/python',route_views={view:'/fixed/view'},
                 dp_workers=3,task_cpu_count=4)
    settings=dict(resources={'ncpus':32,'mem':'64gb'},workers=[dict(host_id=host,profile=runtime)],
                  allocation_sizing={'q3_startup_memory_bytes_per_task':8*P.GIB})
    req=dict(method='PR_PAULI',route=dict(kind='hchain_scoped_q3',view_commit=view),N_X=[4,8],N_Y=[5,8],request_id='j1')
    return host,runtime,settings,req


def accepted(tmp_path, index, runtime, req, peak, *, legacy=False):
    host=runtime['host_id']; req=dict(req,request_id=f'j{index}'); job=req['request_id']; attempt=f'a{index}'
    row=dict(state='ACCEPTED',job_id=job,attempt_id=attempt,host_id=host,request=req,request_sha256=digest(req),
             eligibility={'execution_profiles':{host:runtime}})
    leaf=tmp_path/attempt;leaf.mkdir()
    sample=dict(schema='RPE_Q3_EXECUTION_RESOURCES_V1',key=H.q3_cpu_measurement_key(req,runtime),
      memory_class_key=H.q3_cpu_measurement_key(req,runtime,memory_class=True),job_id=job,attempt_id=attempt,
      host_id=host,policy_host_id=host,request_sha256=row['request_sha256'],method=req['method'],route_kind=req['route']['kind'],
      dp_workers=runtime['dp_workers'],task_cpu_count=runtime['task_cpu_count'],samples=3,peak_rss_bytes=peak,wall_seconds=2.)
    if legacy:
        sample.update(schema='RPE_OWNED_Q3_MEMORY_V1',N_X=req['N_X'],N_Y=req['N_Y'],view_commit=req['route']['view_commit'],
                      runtime_root=runtime['root'],interpreter=runtime['interpreter'],route_view='/fixed/view',adapter_source=None,returncode=0)
    for name,doc in [('OWNED_RESOURCES.json' if legacy else 'Q3_EXECUTION_RESOURCES.json',sample),
                     ('PROCESS_RETURN.json',dict(returncode=0,signal=None,wall_seconds=2.)),
                     ('OUTCOME.json',dict(request_id=job,status='MODEL_BOUND_MEETS_ADOPTED_THRESHOLD'))]:
        (leaf/name).write_text(json.dumps(doc))
    return row,leaf


def test_legacy_and_exact_port(tmp_path):
    host,runtime,settings,req=setup()
    row,leaf=accepted(tmp_path,1,runtime,req,P.GIB,legacy=True)
    sample=M.observation(row,leaf); assert sample and sample['peak_rss_bytes']==P.GIB
    assert M.observation(dict(row,state='RUNNING'),leaf) is None
    row2,leaf2=accepted(tmp_path,2,runtime,dict(req,N_X=[6,8]),2*P.GIB)
    second=M.observation(row2,leaf2);assert second
    assert sample['memory_class_key']==second['memory_class_key']
    assert not M.envelopes([sample,sample],M.learning_policy(True),settings)
    result=M.envelopes([sample,second],M.learning_policy(True),settings)
    assert P.measured_q3_memory(req,runtime,host,result)==3*P.GIB
    assert P.measured_q3_memory(dict(req,N_X=[7,8]),runtime,host,result) is None
    assert P.measured_q3_memory(req,dict(runtime,interpreter='/other/python'),host,result) is None
    assert P.measured_q3_memory(req,dict(runtime,dp_workers=2),host,result) is None
    assert P.measured_q3_memory(req,runtime,'other-host',result) is None
    assert M.observation(dict(row,request_sha256='changed'),leaf) is None


def test_owner_stays_reserved_through_updates_and_resume(tmp_path):
    host,runtime,pbs,req=setup(); M.apply_policy({'pbs':pbs},True)
    path=tmp_path/'learner.json'; learner=M.MemoryLearner(path,pbs)
    owner=dict(state='RUNNING',request=req,host_id=host,job_id='old',attempt_id='old-attempt',request_sha256=digest(req),detail={})
    a,leaf=accepted(tmp_path,3,runtime,req,P.GIB)
    b,leaf2=accepted(tmp_path,4,runtime,req,2*P.GIB)
    with patch('trottertracks.rpe_q3_worker.select_host_profile',return_value=runtime):
        learner.update(M.observation(a,leaf),[owner],pbs)
        assert P.resource_need_with_q3_memory(req, runtime, pbs, host) == (4, 8*P.GIB, False)
        learner.update(M.observation(b,leaf2),[owner],pbs)
        assert P.resource_need(req,runtime,pbs,host)==(4,3*P.GIB)
        assert P.resource_need_with_q3_memory(req, runtime, pbs, host) == (4, 3*P.GIB, True)
        assert P.resource_need_with_q3_memory(dict(req, N_X=[5, 8]), runtime, pbs, host) == (4, 8*P.GIB, False)
        assert P.owned_resource_need(owner,runtime,pbs,host)==(4,8*P.GIB)
        c,leaf3=accepted(tmp_path,5,runtime,req,3*P.GIB)
        learner.update(M.observation(c,leaf3),[owner],pbs)
        assert P.resource_need(req,runtime,pbs,host)==(4,9*P.GIB//2)
        assert P.owned_resource_need(owner,runtime,pbs,host)==(4,8*P.GIB)
        same=M.MemoryLearner(path,pbs);assert len(same.observations)==3
        assert P.owned_resource_need(owner,runtime,pbs,host)==(4,8*P.GIB)
    assert P.owned_resource_need(dict(owner,detail={'pbs_launch_resources':{'cpu':4,'memory_bytes':17}}),runtime,pbs,host)==(4,17)


def test_hchain_no_adapter_exact_table_reference(tmp_path):
    host,runtime,pbs,req=setup(); view=req['route']['view_commit']; target=tmp_path/'run';target.mkdir()
    row=dict(host=host,method=req['method'],route_kind=req['route']['kind'],view_commit=view,horizon_risk_commit=None,
             adapter_source=None,interpreter=runtime['interpreter'],runtime_root=runtime['root'],route_view='/fixed/view',
             dp_workers=3,task_cpu_count=4,N_X=req['N_X'],N_Y=req['N_Y'],rounds=2,
             min_peak_shots=8,max_peak_shots=8,memory_bytes=2*P.GIB)
    table=dict(schema='RPE_PBS_Q3_MEASURED_MEMORY_V1',run_id=target.name,estimates=[row],
               preserve_assignments_before=20,preserve_assignments_before_by_method={req['method']:30},
               measurement_preference={host:[req['method']]})
    path=tmp_path/'external.json';path.write_text(json.dumps(table));reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    d={'pbs':pbs};P.apply_q3_memory_reference(d,target,reference)
    assert P.measured_q3_memory(req,runtime,host,pbs['q3_measured_memory'])==2*P.GIB
    assert P.measured_q3_memory(dict(req,N_X=[5,8]),runtime,host,pbs['q3_measured_memory']) is None
    assert P.measured_q3_memory(req,dict(runtime,root='/other'),host,pbs['q3_measured_memory']) is None
    with pytest.raises(ValueError):P.apply_q3_memory_reference(d,tmp_path/'other-run',reference)


def test_preparation_table_refresh_preserves_previous_and_parallel_classes():
    from build_rpe_hchain_preparation_memory import refresh
    prof=dict(profile=['same-fixed-shape'],memory_bytes=8*P.GIB,task_cpu_count=1)
    table=dict(stages={'compile':{'a':prof,'b':dict(prof,task_cpu_count=3)},'audit':{}},preserve_adjusted_assignments_before=10)
    observations=[dict(job_id=f'j{i}',stage='compile',profile=prof['profile'],task_cpu_count=1,peak_rss_bytes=P.GIB) for i in (1,2)]
    with pytest.raises(ValueError):refresh(table,observations,M.learning_policy(True),20,oldest_owned=5)
    new=refresh(table,observations,M.learning_policy(True),20,oldest_owned=15)
    assert new['stages']['compile']['a']['memory_bytes']==3*P.GIB//2
    assert new['stages']['compile']['a']['previous_memory_bytes']==8*P.GIB
    assert new['stages']['compile']['b']['memory_bytes']==8*P.GIB
    assert table['stages']['compile']['a']['memory_bytes']==8*P.GIB


@pytest.mark.parametrize('legacy',[False,True])
def test_receipt_uses_reserved_claim_cpu_for_legacy_profile(tmp_path,legacy):
    host,runtime,settings,req=setup()
    row,leaf=accepted(tmp_path,11,runtime,req,P.GIB,legacy=legacy)
    old_runtime=dict(runtime);old_runtime.pop('task_cpu_count')
    row['eligibility']={'execution_profiles':{host:old_runtime}}
    row['detail']={'pbs_launch_resources':{'cpu':4}}
    assert M.observation(row,leaf)['task_cpu_count']==4
    assert M.observation(dict(row,detail={'pbs_launch_resources':{'cpu':5}}),leaf) is None
    assert M.observation(dict(row,detail={}),leaf) is None
    row['eligibility']={'execution_profiles':{host:runtime}}
    assert M.observation(dict(row,detail={'pbs_launch_resources':{'cpu':3}}),leaf) is None


def test_resume_bulk_observations_preserve_original_owner_envelope(tmp_path):
    host,runtime,pbs,req=setup(); M.apply_policy({'pbs':pbs},True)
    learner=M.MemoryLearner(tmp_path/'bulk.json',pbs)
    owner=dict(state='RUNNING',request=req,host_id=host,job_id='old',attempt_id='old-attempt',request_sha256=digest(req),detail={})
    a,leaf=accepted(tmp_path,3,runtime,req,P.GIB)
    b,leaf2=accepted(tmp_path,4,runtime,req,2*P.GIB)
    with patch('trottertracks.rpe_q3_worker.select_host_profile',return_value=runtime):
        observations=[M.observation(a,leaf),M.observation(b,leaf2)]
        assert learner.update_many(observations,[owner],pbs)
        assert P.resource_need(req,runtime,pbs,host)==(4,3*P.GIB)
        assert P.owned_resource_need(owner,runtime,pbs,host)==(4,8*P.GIB)
        assert not learner.update_many(observations,[owner],pbs)
