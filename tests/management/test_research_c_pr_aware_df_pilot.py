"""Artificial orchestration/STOP tests; molecular backends are mocked."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_pilot as m


@pytest.mark.parametrize('fault',[None,'initial_budget','representation','optimizer','candidate_budget','no_gain'])
def test_artificial_30_call_orchestration_and_STOP(tmp_path,monkeypatch,fault):
    out=tmp_path/'run01';script=tmp_path/'runner.py';script.write_text('# synthetic dispatch')
    c=dict(fresh_output='run01',entry='runner.py',source_manifest='SOURCE.json')
    m.io.save(tmp_path/'SOURCE.json',dict(files={'synthetic':'only'}))
    monkeypatch.setattr(m,'preflight',lambda *a,**k:c)
    monkeypatch.setattr(m.molecule,'SectorCAR',lambda n:SimpleNamespace(physical=lambda *a:np.zeros((2,2))))
    monkeypatch.setattr(m.old,'resource_plan',lambda **k:dict(workers=2))
    def assessment(ctx,p,J,U,physical,rank):
        error=.001 if fault=='initial_budget' and J[0,0,0]==1 else .00001
        if fault=='candidate_budget' and J[0,0,0]==3:error=.001
        return dict(sector_error=error,numerical_guard=1e-8),[np.zeros((2,2))]
    monkeypatch.setattr(m.molecule,'rank_assessment',assessment)
    def representation(*a):
        if fault=='representation':raise ValueError('ARTIFICIAL_INDEPENDENT_REFERENCE_MISMATCH')
        return dict(status='PASS',max_entry=1e-15)
    monkeypatch.setattr(m.molecule,'representation_gate',representation)
    def worker(args,**kwargs):
        role=args[args.index('--worker')+1];child=out/role;child.mkdir()
        if role=='prepare':
            m.arrays(child/'input.npz',h=np.eye(4),g=np.zeros((4,)*4),constant=np.array(.5))
            for name in ('before','initial'):m.arrays(child/(name+'.npz'),J=np.ones((8,4,4)),U=np.array([np.eye(4)]*8))
            limits=m.PREPARE
        else:
            m.arrays(child/'final.npz',J=np.full((8,4,4),2 if role=='baseline' else 3),U=np.array([np.eye(4)]*8))
            m.io.save(child/'optimizer.json',dict(success=not(fault=='optimizer' and role=='proposed')))
            mass=10. if role=='baseline' or fault=='no_gain' else 8.
            m.io.save(child/'final_diagnostic.json',dict(exact_tail_mass_Ha=mass,CA_achieved=None,total_cost=None))
            m.io.save(child/'tensor.json',dict(objective_from_tensor=1e-8));limits=m.FIT
        m.io.save(child/'STOP.json',dict(failed=False,call_counts=limits,scientific_units=sum(limits.values())))
    monkeypatch.setattr(m.subprocess,'run',worker)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'artificial-SOURCE\n')
    if fault in ('initial_budget','representation'):
        with pytest.raises(ValueError):m.run(tmp_path,tmp_path/'contract.json',out,'input.npz','fit-python')
        assert not (out/'baseline').exists() and m.io.read(out/'STOP.json')['failed']
    else:
        m.run(tmp_path,tmp_path/'contract.json',out,'input.npz','fit-python')
        r=m.io.read(out/'result.json');stop=m.io.read(out/'STOP.json')
        assert stop['scientific_units']==30 and stop['call_counts']==m.LIMITS
        assert r['status']==('STAGE_A_PROXY_AND_DF_PASS' if fault is None else 'STAGE_A_NOT_MET_OR_UNRESOLVED')
        assert r['shots'] is r['total_cost'] is r['CA_achieved'] is r['attenuation'] is None
        assert not r['stages_B_C_authorized']
        before=(out/'STOP.json').read_bytes()
        with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'contract.json',out,'input.npz','fit-python')
        assert before==(out/'STOP.json').read_bytes()


def test_coefficient_mass_does_not_certify_shots():
    from trottertracks.c1_stage3_accuracy import AccuracyTarget,RoundEvidence,assess_round
    import math
    target=AccuracyTarget(.00159,1/8,.1,math.pi/6,math.pi/12,.01/26,.01)
    def evaluate(A,pf):return assess_round(target,RoundEvidence(0,A,pf,0.,0.,0.,'artificial','own_DF_ground_state_oracle',True,True,law_model='ideal_finite_RTE_law'))
    # Better attenuation can lose the shot comparison when PF error changes.
    assert evaluate(.9,.2)['shots']>evaluate(.8,0.)['shots']


def test_passive_import():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_pilot as m;assert m.MAX_CALLS==30;assert not any(x in sys.modules for x in ['numpy','scipy','ffsim','jax','qiskit'])"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_byte_only_preflight_rejects_changed_source_or_input(tmp_path):
    # Opaque invented bytes, deliberately not a decodable molecular export.
    source=tmp_path/'source.py';source.write_text('# artificial source\n')
    initial=tmp_path/'initial.npz';initial.write_bytes(b'opaque artificial bytes')
    original=tmp_path/'input.npz';original.write_bytes(b'opaque original bytes')
    m.io.save(tmp_path/'SOURCE.json',dict(files={'source.py':m.io.sha(source)}))
    c=dict(scope=m.SCOPE,status='UNAPPROVED_PROPOSAL',conditions=m.CONDITIONS,
        call_limits=m.LIMITS,maximum_logical_units=30,stages_B_C_authorized=False,
        objective=m.objective.VERSION,original_input=dict(sha256=m.io.sha(original)),
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(tmp_path/'SOURCE.json'),
        initial_export='initial.npz',saved_inputs={'initial.npz':m.io.sha(initial)},evaluation_packages={})
    path=tmp_path/'contract.json';m.io.save(path,c)
    assert m.preflight(tmp_path,path,original)==c
    source.write_text('# changed\n')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,path,original)
    original.write_bytes(b'changed input')
    with pytest.raises(ValueError,match='ORIGINAL_INPUT'):m.preflight(tmp_path,path,original)


def test_artificial_general_K_to_existing_tail_and_RTE_law():
    from types import SimpleNamespace
    from trottertracks import c1_stage3_connection as connection
    from trottertracks import c1_stage2_native as native
    from trottertracks import c1_general_k_adapter as general
    from trottertracks.c1_stage3_accuracy import _legacy
    rte=_legacy('rte')
    J=np.array([[[.1,-.02],[-.02,.07]], [[.02,.003],[.003,-.01]]])
    U=np.array([np.eye(2)]*2);model=general.from_ffsim_number_arrays(
        one_body_tensor=np.zeros((2,2)),diag_coulomb_mats=J,orbital_rotations=U,constant=.4,
        z_representation=False,frame_ids=['F0','F1'],group_ids=['F0','F1'],
        source_identity='artificial',input_identity='artificial',one_body_role='ffsim_corrected_kappa',enable_general_k=True)
    blocks=[native.NativeBlock('one_body',model.frames[0].orbital_rotation,
        SimpleNamespace(global_phase=0.,rz=(),rzz=()))]
    blocks.extend(native.NativeBlock(f.frame_id,f.orbital_rotation,general.diagonal_primitives(f,1.)) for f in model.frames)
    plan=connection.split_native_model(native.NativeModel('CDF',tuple(blocks),.4,'artificial','artificial'),1)
    tail,config,distribution=connection.configure_rte(plan,'1/40',16,0)
    expected=float(np.sum(np.abs(m.objective.coefficients(J[1:],np))))
    assert tail.lambda_r==pytest.approx(expected,abs=1e-15)
    assert all(a.kind!='IDENTITY' for a in plan.atoms) and plan.extracted_identity!=0
    assert sum(x.probability for x in tail.components)==pytest.approx(1.)
    A=rte.finite_rte_attenuation(config,tail_evolutions=4)
    assert A==pytest.approx((1+(expected*.025/16)**2)**(-16*4/2),rel=1e-12)
