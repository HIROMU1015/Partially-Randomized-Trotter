"""Execution plumbing tests; invented inputs, optimizer mocked, no real H4."""
from dataclasses import asdict
from pathlib import Path
import json
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_integrated_fit_worker as w
from trottertracks import c_integrated_robustness as fit
from trottertracks import c_near_transfer as transfer
from trottertracks.c1_h6_initialization import reconstruct


def fixture(tmp_path, *, mode='cell', start='P1'):
    rng = np.random.default_rng(421)
    J = rng.normal(size=(3,2,2))*.05
    J = (J+J.transpose(0,2,1))/2
    U = np.tile(np.eye(2), (3,1,1))
    g = reconstruct(J,U)+np.full((2,)*4, .001)
    np.savez(tmp_path/'input.npz', g=g)
    np.savez(tmp_path/'cold.npz', J=J, U=U)
    system = transfer.System(2,3,1,1.,w.io.sha(tmp_path/'input.npz'))
    task = dict(root=str(tmp_path), input_path='input.npz', cold_path='cold.npz',
        input_sha256=system.input_sha256, cold_sha256=w.io.sha(tmp_path/'cold.npz'),
        system=asdict(system), loss_scale=2e-9, start=start, out=mode,
        arm='PR', strength=.1, source_sha256='b'*64)
    return task, (g,J,U)


def mock_optimizer(monkeypatch):
    import scipy.optimize
    seen=[]
    def optimizer(fun,x,**options):
        seen.append(options)
        value,gradient=fun(x)
        return scipy.optimize.OptimizeResult(x=x.copy(),jac=gradient,fun=value,
            success=False,status=1,message='ARTIFICIAL_NO_ITERATIONS',nit=0,nfev=1,njev=1)
    monkeypatch.setattr(scipy.optimize,'minimize',optimizer)
    return seen


@pytest.mark.parametrize('start,physical_ok',[('P1',True),('P2',False),('C0',True)])
def test_real_restore_engine_mock_optimizer_exact_five_units(tmp_path,monkeypatch,start,physical_ok):
    task,(g,J,U)=fixture(tmp_path,mode='restore',start=start)
    seen=mock_optimizer(monkeypatch)
    physical=[]
    def assess(label,a,b):
        physical.append(label)
        return dict(budget_PASS=physical_ok,sector_error=1e-8,numerical_guard=1e-10)
    result=w.run_restore(task,assess,initializer=lambda target,rank:(J.copy(),U.copy()))
    out=tmp_path/'restore'
    stop=json.loads((out/'STOP.json').read_text())
    assert stop['failed'] is False and stop['scientific_units']==5
    assert stop['worker_terminal'] is True and stop['mandatory_stop'] is False
    assert stop['integrated_continuation_allowed'] is True
    assert result['mandatory_stop'] is False and result['worker_terminal'] is True
    assert stop['actual_calls']==w.RESTORE_LIMITS
    assert len(seen)==1 and physical==['restored_'+start]
    assert result['eligible'] is physical_ok and result['optimizer']['nit']==0
    assert result['candidate_sha256']==w.io.sha(out/'restored.npz')
    assert (out/'partial.json').exists() and (out/'resources.json').exists()
    records=[json.loads(line) for line in (out/'trajectory.jsonl').read_text().splitlines()]
    assert {row['kind'] for row in records}=={'start_generation','restore_initial','restore_raw_terminal'}
    assert all((tmp_path/row['array_path']).exists() for row in records)
    if start=='C0':
        assert result['loss_scale']==pytest.approx(float(np.sum((reconstruct(J,U)-g)**2)/2))
    with pytest.raises(FileExistsError):
        w.run_restore(task,assess,initializer=lambda target,rank:(J.copy(),U.copy()))
    assert len(seen)==1


def test_paired_setup_once_and_real_recipe_wrapper_with_mock_trajectory(tmp_path,monkeypatch):
    from trottertracks import c_near_practical as practical
    task,(g,J,U)=fixture(tmp_path)
    specs=w.make_objective_specs(task)
    assert set(specs)=={'PR','ENTRY'}
    task['objective_spec']=specs['PR']
    monkeypatch.setattr(practical.algorithm.control.small,'assess',
        lambda *args,**kwargs: dict(J_numerically_qualified=True))
    def trajectory(target,jcold,ustart,spec,physical,*,step,policy,maximum_total_points,call,record):
        assert step.max_updates==64 and maximum_total_points==513
        p=dict(J=jcold.copy(),U=ustart.copy(),state={'first_order_met':False})
        for count in range(1,65):
            call('inner_J_solve',str(count),lambda:None)
            record('inner_raw',str(count),dict(inner_summary_record=True,status=0,nfev=2,njev=2,
                nit=1,value_gradient_calls_in_block=3,hessian_calls_in_block=1),None)
            p['J']=jcold+count*.001
            row=physical(str(count),p['J'],p['U'])
            record('physical_candidate',str(count),dict(update=count,state=p['state'],physical=row,
                qualification={'admission':'PASS'}),None)
            record('accepted',str(count),dict(physically_qualified=True),dict(J=p['J'],U=p['U']))
        return dict(terminal=p,status='MAXIMUM_ACCEPTED_UPDATES',accepted_U_updates=64)
    recipe=lambda *args,**kwargs:fit.recipe_cell(*args,**kwargs,trajectory=trajectory)
    result=w.run_cell(task,lambda *args:dict(budget_PASS=True,sector_error=1e-8,numerical_guard=1e-10),recipe=recipe)
    assert set(result['checkpoints'])=={'16','64'}
    for budget,checkpoint in result['checkpoints'].items():
        path=tmp_path/checkpoint['candidate_path']
        assert path.exists() and checkpoint['candidate_sha256']==w.io.sha(path)
        assert checkpoint['physical']['candidate_sha256']==checkpoint['candidate_sha256']
        assert checkpoint['physical']['evaluated_array_identity']==checkpoint['array_identity']
        with np.load(path,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['J'],J+int(budget)*.001)
    assert result['checkpoints']['16']['evidence']['qualification']=={'admission':'PASS'}
    stop=json.loads((tmp_path/'cell'/'STOP.json').read_text())
    assert stop['failed'] is False and stop['maximum_logical_units']==3213
    assert stop['mandatory_stop'] is False and stop['integrated_continuation_allowed'] is True
    assert result['worker_terminal'] is True and result['mandatory_stop'] is False
    assert stop['actual_calls']['arm_export']==2
    assert stop['actual_calls']['endpoint_diagnostics']==2


def test_failure_preserves_checkpoint_partial_stop_and_forbids_retry(tmp_path):
    task,(_,J,U)=fixture(tmp_path)
    task['objective_spec']=w.make_objective_specs(task)['PR']
    def recipe(name,g,cold,spec,system,physical,call,record):
        row=physical('invented16',J,U)
        record('checkpoint_export','16',dict(evidence={'physical':row}),dict(J=J,U=U))
        call('inner_J_solve','synthetic_fault',lambda:(_ for _ in ()).throw(ValueError('INVENTED_FAULT')))
    with pytest.raises(ValueError,match='INVENTED_FAULT'):
        w.run_cell(task,lambda *args:dict(budget_PASS=True),recipe=recipe)
    out=tmp_path/'cell'
    stop=json.loads((out/'STOP.json').read_text())
    assert stop['failed'] is True and stop['mandatory_stop'] is True
    assert stop['worker_terminal'] is True and stop['integrated_continuation_allowed'] is False
    partial=json.loads((out/'partial.json').read_text())
    assert set(partial['checkpoints'])=={'16'}
    assert partial['scientific_units']==1 and partial['failure']['error_type']=='ValueError'
    assert not (out/'result.json').exists()
    with pytest.raises(FileExistsError):
        w.run_cell(task,lambda *args:dict(budget_PASS=True),recipe=recipe)


def test_wrong_input_refused_before_any_array_decode(tmp_path):
    task,_=fixture(tmp_path)
    task['objective_spec']={}
    task['input_sha256']='f'*64
    def blocked(*args):
        raise AssertionError('NO_ARRAY_DECODE')
    with pytest.raises(ValueError,match='TASK_SYSTEM_INPUT_IDENTITY'):
        w.run_cell(task,lambda *args:None,loader=blocked)
    stop=json.loads((tmp_path/'cell'/'STOP.json').read_text())
    assert stop['failed'] is True and stop['scientific_units']==0
