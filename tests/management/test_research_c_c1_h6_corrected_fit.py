"""Eval-env artificial Hamiltonian end-to-end and passive scope checks."""
from pathlib import Path
import sys
from unittest.mock import Mock,patch
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_corrected_fit as m


@pytest.mark.parametrize('outcome',['pass','budget','optimizer','initial_fail','worker_exception'])
def test_artificial_parent_end_to_end_gates_decisions_partial_STOP(tmp_path,outcome):
    root=tmp_path;out=root/'fresh';J=np.array([[[.2,.03],[.03,.1]]]);U=np.array([[[.8,-.6],[.6,.8]]])
    packet=dict(h=np.diag([-.2,.1]),g=m.old.reconstruct(J,U),constant=np.asarray(.125))
    np.savez(root/'input.npz',**packet)
    c=dict(entry='UNUSED',original_input='input.npz',rank=1,DF_reserve_Ha=.00159/8)
    stages=[];system_run=m.subprocess.run
    def worker(command,**kwargs):
        if not any(x in command for x in ('--prepare-worker','--fit-worker')):
            return system_run(command,**kwargs)  # e.g. lazy platform.uname during library imports
        # Symlink spelling is preserved for Python venv discovery.
        assert command[0]=='/task/.venv-fit/bin/python'
        mode=command[command.index('--contract')+2];stages.append(mode)
        directory=out/('prepare' if mode=='--prepare-worker' else 'fit');directory.mkdir()
        if mode=='--fit-worker' and outcome=='worker_exception':raise RuntimeError('ARTIFICIAL_WORKER_STOP')
        for state in (('before','initial') if mode=='--prepare-worker' else ('final',)):
            scale=1.2 if ((outcome=='initial_fail' and state=='initial') or (outcome=='budget' and state=='final')) else 1
            a=J*scale;np.savez(directory/(state+'.npz'),J=a,U=U)
            m.io.save(directory/(state+'_tensor.json'),m.old.tensor_metrics(packet['g'],a,U))
        m.io.save(directory/'STOP.json',dict(failed=False,call_counts=m.PREPARE_LIMITS if mode=='--prepare-worker' else m.FIT_LIMITS))
        if mode=='--fit-worker':m.io.save(directory/'optimizer.json',dict(success=outcome!='optimizer',nit=0))
    forbidden=Mock(side_effect=AssertionError('FORBIDDEN_PF_NATIVE_OR_OLD_FIT'))
    with patch.object(m,'preflight',return_value=c),patch.object(m.subprocess,'run',side_effect=worker),patch.object(m.old.molecule,'factor',forbidden),patch.object(m.old.molecule,'PF_rounds',forbidden),patch.object(m.old.molecule,'native_H2',forbidden):
        if outcome in ('initial_fail','worker_exception'):
            with pytest.raises((ValueError,RuntimeError)):m.run(root,root/'contract',out,'/task/.venv-fit/bin/python')
        else:m.run(root,root/'contract',out,'/task/.venv-fit/bin/python')
    forbidden.assert_not_called();stop=m.io.read(out/'STOP.json')
    assert stop['failed']==(outcome!='pass') and stop['scientific_units']<=m.MAX_CALLS
    assert stop['retry']==stop['resume']==0
    if outcome=='initial_fail':assert stages==['--prepare-worker'] and not (out/'fit').exists()
    elif outcome=='worker_exception':assert (out/'failure.json').exists() and not (out/'final_sector.json').exists()
    else:
        assert stop['scientific_units']==m.MAX_CALLS
        result=m.io.read(out/'result.json');assert result['fit_invocations']==1 and result['selected_initial_as_fallback'] is False
        assert result['DF_budget_status']==('NOT_MET' if outcome=='budget' else 'PASS')
        assert result['optimizer_success']==(outcome!='optimizer')
        assert result['status']==dict(pass_='FIT_NUMERICAL_PILOT_PASS',budget='DF_BUDGET_NOT_MET',optimizer='OPTIMIZER_NOT_SUCCESS')['pass_' if outcome=='pass' else outcome]
    before=(out/'STOP.json').read_bytes()
    with patch.object(m,'preflight',return_value=c),pytest.raises(FileExistsError):m.run(root,root/'contract',out,'/venv/python')
    assert (out/'STOP.json').read_bytes()==before


def contract_stub(root):
    (root/'source').write_text('ARTIFICIAL');(root/'input').write_text('NOT_AN_NPZ')
    m.io.save(root/'SOURCE.json',dict(files={'source':m.io.sha(root/'source')}))
    return dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',rank=14,max_new_fit=1,retry=0,resume=0,
        codec=m.codec.VERSION,bridge=m.bridge.VERSION,optimizer_settings=dict(m.bridge.SETTINGS,max_vecs=14),
        DF_reserve_Ha=.00159/8,guard_policy=m.old.molecule.GUARD_POLICY,allowances=m.ALLOWANCES,
        prepare_calls=m.PREPARE_LIMITS,fit_calls=m.FIT_LIMITS,sector_calls=m.SECTOR_LIMITS,
        maximum_logical_units=m.MAX_CALLS,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        saved_inputs={'input':m.io.sha(root/'input')},original_input='input',initial_arrays='input',XDF_full='input',evaluation_packages={})


@pytest.mark.parametrize('change',[None,'rank','options','budget','input','source'])
def test_preflight_passive_no_decode_and_scope_identity_failures(tmp_path,change):
    c=contract_stub(tmp_path)
    if change=='rank':c['rank']=18
    if change=='options':c['optimizer_settings']['options']={}
    if change=='budget':c['DF_reserve_Ha']*=2
    if change=='input':c['initial_arrays']='unlisted'
    if change=='source':(tmp_path/'source').write_text('CHANGED')
    m.io.save(tmp_path/'contract.json',c)
    with patch.object(np,'load',side_effect=AssertionError('MOLECULAR_DECODE_FORBIDDEN')):
        if change:
            with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'contract.json')
        else:assert m.preflight(tmp_path,tmp_path/'contract.json')==c


def test_maximum_and_no_rank_search_or_follow_on_stages():
    assert m.MAX_CALLS==20
    assert m.FIT_LIMITS['optimized_fitter']==1
    assert not set(m.PREPARE_LIMITS|m.FIT_LIMITS|m.SECTOR_LIMITS)&{'rank_search','PF','compile','shots','total_cost'}
