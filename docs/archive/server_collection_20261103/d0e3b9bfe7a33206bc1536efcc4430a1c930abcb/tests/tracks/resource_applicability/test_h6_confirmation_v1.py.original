"""Metadata-only selection/seed/carry/grant/canonical monitor regressions."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
from trottertracks.resource_applicability import h6_confirmation_contract_v1 as c
from trottertracks.resource_applicability import h6_confirmation_execution_v1 as e
from trottertracks.resource_applicability import h6_parallel_execution_v1 as old
from trottertracks.resource_applicability.h6_matched_accounting_v1 import seed
from trottertracks.resource_applicability.ax2a_preparation import digest

ROOT=Path(__file__).resolve().parents[3]


def fixtures():
    signals=[];rows=[]
    for cid in ('B2_A','B2_B','B2_C'):
        cell=dict(id=cid,method='B2',q=1,r=1,R=1,prefix=5,K=2,formula='2nd')
        signals.append(dict(cell=cell,precision_rows={str(eps):{'1.0':dict(cost_acquisition_eligible=True,
            N_total=2,axes={'real':{'shots':1},'imag':{'shots':1}})} for eps in (.05,.01)}))
        for i in range(8):
            for axis in ('cosine','sine'):
                rows.append(dict(cell_id=cid,phase='exploration',replica=i,axis=axis,
                    control='symmetric_directional',metrics={'RZ':1 if cid!='B2_C' else 2}))
    plan=dict(epsilons=[.05,.01],cost_rule=dict(confirmation_top_per_epsilon=2,
        confirmation_union_max=8,confirmation_random_trajectories=32),sensitivity_cells=[])
    return signals,rows,plan


def manifest(tmp_path):
    signals,rows,plan=fixtures();selection=c.select_and_register(signals,rows,plan)
    binding=dict(selection=selection,carry_known_calls=dict(compile=1188,control_probe=360,trajectory=578,occurrence=2176),
        carry_reservations=dict(compile=12,control_probe=4,trajectory=6,occurrence=48))
    return dict(schema=c.SCHEMA,kind=c.KIND,repository_root=str(tmp_path),
        source_commit='a'*40,source_hashes={},exploration_binding=binding,registrations=selection['registrations'],
        base_manifest={'profile':{'sha256':'b'*64}},cumulative_accounting={'proposed_caps':{'trajectory':840}},
        roundoff_preparation_binding={'archive_sha256':'c'*64},proposed_resources=c.proposed_resources(),
        observation_policy=c.observation_policy(),candidate_selection_sealed=True,execution_plan_sealed=True,
        approved=False,runtime_authorization=False,allowed_cpus=[],confirmation_authorized=False,
        H8_authorized=False,retry=False,resume=False,mandatory_stop=True,next_stage_authorized=False,
        absolute_launch_command=None,one_shot_lock=c.LOCK,
        exclusive_output=dict(repository_path=c.NAMESPACE+'launch_confirmation_v1',
            absolute_path=str(tmp_path/c.NAMESPACE/'launch_confirmation_v1')))


def test_original_top2_tie_rule_and_fresh32_namespace():
    ss,rs,p=fixtures();binding=c.select_and_register(ss,rs,p)
    assert binding['selected_ids']==['B2_A','B2_B']
    assert len(binding['registrations'])==64
    assert all([x['cell_id'] for x in r['top2']]==['B2_A','B2_B'] for r in binding['rankings'])
    seeds={x['task']['seed'] for x in binding['registrations']}
    assert len(seeds)==64 and not seeds&{seed(cid,'exploration',i) for cid in ('B2_A','B2_B') for i in range(8)}
    assert all(x['task']['phase']=='confirmation' and not x['task']['validate'] and not x['task']['ordinary'] for x in binding['registrations'])


def test_fixed32_demand_and_smallest_historical_amendment():
    binding=dict(selection={'registrations':[{'task':{'cell':{'q':q}}} for q in (1,2,4,8) for _ in range(64)]},
        carry_known_calls=dict(compile=1188,control_probe=360,trajectory=578,occurrence=2176),
        carry_reservations=dict(compile=12,control_probe=4,trajectory=6,occurrence=48))
    base={'plan':{'caps_proposed':dict(compile=1712,control_probe=1024,trajectory=832,occurrence=6656)}}
    a=c.accounting(binding,base)
    assert a['new_demand']==dict(compile=512,control_probe=0,trajectory=256,occurrence=960)
    assert a['admission_totals']==dict(compile=1712,control_probe=364,trajectory=840,occurrence=3184)
    assert a['amendments']=={'trajectory':{'original':832,'proposed':840}}
    assert base['plan']['caps_proposed']['trajectory']==832 and not a['cap_amendment_approved']
    assert a['unknown_actual_totals'] is None and not a['budget_reset'] and not a['failed_calls_refunded']


@pytest.mark.parametrize('tamper',['draft','old_kind','seed','selection','resources','cap','amendment','float_approval','valid'])
def test_new_phase_and_cap_specific_grant_required(tmp_path,tamper):
    m=manifest(tmp_path);g=c.grant_draft(m)
    g.update(approved_by_user=True,server_usage_approved_by_user=True,trajectory_cap_amendment_approved=True,confirmation_authorized=True)
    if tamper=='draft':g['approved_by_user']=False
    if tamper=='old_kind':g['kind']='H6_PARALLEL_ROUNDOFF_COMPLETE_REUSE_RESUME01_V1'
    if tamper=='seed':g['task_list_digest']='d'*64
    if tamper=='selection':g['candidate_selection_digest']='e'*64
    if tamper=='resources':g['resources']['cost_workers']=8
    if tamper=='cap':g['proposed_cumulative_caps']['trajectory']=832
    if tamper=='amendment':g['trajectory_cap_amendment_approved']=False
    if tamper=='float_approval':g['approved_by_user']=1
    if tamper=='valid':c.validate_grant(m,g)
    else:
        with pytest.raises(ValueError):c.validate_grant(m,g)


@pytest.mark.parametrize('tamper',['scope','worker8','output','lock','execution','flags','command','valid'])
def test_worker_does_not_dispatch_outside_bound_scope(tmp_path,tamper):
    m=manifest(tmp_path);out=Path(m['exclusive_output']['absolute_path'])
    if tamper=='scope':m['registrations'][0]['task']['phase']='exploration';m['registrations']=deepcopy(m['registrations']);m['registrations'][0]['task']['seed']+=1;m['exploration_binding']['selection']['registrations'][0]['task']['seed']-=1
    if tamper=='worker8':m['proposed_resources']['cost_workers']=8
    if tamper=='output':out=tmp_path/'other'
    if tamper=='lock':m['one_shot_lock']='old.lock'
    if tamper=='execution':m['execution_plan_sealed']=False
    if tamper=='flags':m['runtime_authorization']=True
    if tamper=='command':m['absolute_launch_command']='python runner'
    if tamper=='valid':c.worker_scope(tmp_path,out,m)
    else:
        with pytest.raises(ValueError):c.worker_scope(tmp_path,out,m)


def test_monitor_and_worker_keep_frozen_behavior_except_phase_path():
    for name in ('owned_exit_flag','retain_exiting_memory','supervise'):
        assert inspect.getsource(getattr(e,name))==inspect.getsource(getattr(old,name)).replace('cost_exploration_','cost_confirmation_')
    assert inspect.getsource(e.worker)==inspect.getsource(old.worker).replace('cost_exploration_','cost_confirmation_').replace("phase='exploration'","phase='confirmation'")


@pytest.mark.parametrize('case',['partial','one_axis','phase','replica','valid'])
def test_complete_confirmation_rejects_partial_or_pooled_rows(tmp_path,case):
    m=manifest(tmp_path);groups={}
    for r in m['registrations']:
        groups[r['index']]=dict(task=deepcopy(r['task']),rows=[dict(control='symmetric_directional',axis=a) for a in ('cosine','sine')])
    if case=='partial':groups.pop(0)
    if case=='one_axis':groups[0]['rows'].pop()
    if case=='phase':groups[0]['task']['phase']='exploration'
    if case=='replica':groups[0]['task']['replica']=31
    if case=='valid':assert len(e.complete_confirmation(m,groups))==128
    else:
        with pytest.raises(ValueError):e.complete_confirmation(m,groups)


def test_dry_runner_has_no_science_grant_output_or_launch():
    script=str(ROOT/c.RUNNER)
    code=f'import sys,runpy;sys.argv=[{script!r}];\ntry:runpy.run_path({script!r},run_name="__main__")\nexcept SystemExit as exc:assert exc.code==0\nassert not set(sys.modules)&{{"numpy","qiskit","pyscf","numba","cupy"}}'
    r=subprocess.run([sys.executable,'-I','-P','-B','-c',code],capture_output=True,text=True,timeout=20)
    assert r.returncode==0,r.stderr


def test_unapproved_runner_cannot_create_output(tmp_path):
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,'-I','-P','-B',str(ROOT/c.RUNNER),'--execute','--output',str(out)],capture_output=True,text=True,timeout=20)
    assert r.returncode!=0 and not out.exists()
