"""Saved17/new1 orchestration and context reuse using artificial data only."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_integrated_cgfix as m
from trottertracks import c_integrated_plan as plan


def write(path,row):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row))


def tasks():
    return [dict(job_id=f'H4_100_{s}_{a}_g{str(g).replace(".","")}',start=s,arm=a,strength=g,
        objective_spec={'ARTIFICIAL':True}) for s in plan.STARTS for a in plan.ARMS for g in plan.STRENGTHS]


def test_only_failed_cell_is_dispatched_and_saved_results_are_not_recomputed(tmp_path):
    from trottertracks.c_integrated_execution import Run
    run=Run.__new__(Run)
    run.root=tmp_path;run.digest='ARTIFICIAL';run.cpus=[0,1,2,3];run.stages={}
    run.c={'maximum_calls':{'stage_totals':{'development_optimization':3213},'total':118188}}
    run.progress=lambda *a,**k:None
    (tmp_path/'cold').write_bytes(b'ARTIFICIAL_OPAQUE_INPUT_NOT_ARRAY')
    entries=tasks();failed='H4_100_C0_ENTRY_g03'
    reused={t['job_id']:{'SAVED':True} for t in entries if t['job_id']!=failed}
    specs={}
    for t in entries:specs.setdefault((t['start'],t['strength']),{})[t['arm']]=t['objective_spec']
    g=dict(geometry='H4_100',system={'input_sha256':'I'},input_path='input',
        starts={s:{'path':'cold'} for s in plan.STARTS},strengths=None,specs=specs,scale=1.,
        output=tmp_path/'fresh',physical='MOCK',reused_optimization=reused)
    seen=[]
    def fit(task,physical,affinity):
        seen.append(task['job_id']);assert affinity==run.cpus
        return dict(scientific_units=1,summary={'status':'ARTIFICIAL_COMPLETE'},checkpoints={})
    run.fit=fit
    value=run.optimize(g)
    assert seen==[failed] and len(value['optimization'])==18
    assert run.stages['development_optimization']==1
    for key,row in reused.items():assert value['optimization'][key] is row


def test_original_H_context_factory_does_not_rebuild_CAR_or_H(tmp_path,monkeypatch):
    from trottertracks.c_pr_integrated_evaluation import Evaluation
    from trottertracks import c1_molecule_extension as molecule
    def forbidden(*a,**k):raise AssertionError('NO_NEW_CAR_OR_H')
    monkeypatch.setattr(molecule,'SectorCAR',forbidden)
    p={'h':np.eye(2),'g':np.zeros((2,)*4),'constant':np.array(.1)}
    context=SimpleNamespace(sector=SimpleNamespace(basis_indices=np.array([0,1])))
    h=np.array([[.2,.1],[.1,.3]])
    common=dict(p=p,ctx=context,physical_reference=h)
    value=Evaluation.from_context(tmp_path,{'DF_budget_Ha':.00019875},tmp_path,common)
    assert value.ctx is context and value.physical is h and value.packet is p
    assert value.cache=={}


def test_initialization_reuses_saved_starts_specs_and_context_without_fit(tmp_path,monkeypatch):
    from trottertracks import c_integrated_execution_rpc as rpc
    entries=tasks();failed='H4_100_C0_ENTRY_g03';completed=[];missing=[]
    for t in entries:
        task_path='old/tasks/'+t['job_id']+'.json';write(tmp_path/task_path,t)
        item=dict(job_id=t['job_id'],task_path=task_path)
        if t['job_id']==failed:missing.append(item)
        else:
            item['result_path']='old/results/'+t['job_id']+'.json'
            write(tmp_path/item['result_path'],{'SAVED':t['job_id']});completed.append(item)
    initial=dict(system={'input_sha256':'I'},scale=.125,starts={s:{'path':'old/'+s} for s in plan.STARTS})
    write(tmp_path/'old/initial.json',initial)
    write(tmp_path/m.HANDOFF,dict(initialization='old/initial.json',common_pickle='old/context.pkl',
        completed=completed,failed=missing))
    c=dict(environment={'evaluation_python':sys.executable},input_roles={'input':'input.npz'},
        cgfix={'new_development_cells':[failed]},maximum_calls={'stage_totals':{'development_initialization':1},'total':118188})
    write(tmp_path/m.CONTRACT,c);source=tmp_path/'SOURCE.json';source.write_text('{}')
    seen=[]
    class Physical:
        def __init__(self,*a):pass
        def request(self,op,**kw):
            seen.append((op,kw));return {'common_pickle':'new/context.pkl'}
    monkeypatch.setattr(rpc,'RpcProcess',Physical)
    run=m.Run(tmp_path,source)
    run.fit=lambda *a,**k:(_ for _ in ()).throw(AssertionError('NO_RESTORE_OR_INITIAL_FIT'))
    g=run.initialize('H4_100','fresh/run01')
    assert len(g['reused_optimization'])==17
    assert g['starts']==initial['starts'] and len(g['specs'])==9
    assert seen==[('initialize',dict(system=initial['system'],input_rel='input.npz',reuse_common_pickle='old/context.pkl'))]
    assert run.stages['development_initialization']==1
