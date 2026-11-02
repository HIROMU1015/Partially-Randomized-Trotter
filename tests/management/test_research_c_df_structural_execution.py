"""Artificial CAR/sector, mock child optimizer, passive preflight and STOP tests."""
from pathlib import Path
import sys,json,subprocess
from collections import Counter
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_df_structural_execution as e
from trottertracks import c_df_structural_support as m


def synthetic():
    from trottertracks.c1_h6_initialization import reconstruct
    rng=np.random.default_rng(50528);J=np.zeros((8,4,4));U=np.array([np.eye(4)]*8)
    for l in range(8):
        for i in range(4):
            for j in range(i,4):
                if l<2 or j-i<=1:J[l,i,j]=J[l,j,i]=rng.normal()*.01
    return dict(h=np.diag([.1,-.2,.3,-.4]),g=reconstruct(J,U),constant=np.array(.31)),dict(J=J,U=U)


def test_artificial_full256_CAR_general_K_scalar_spin_and_sector36():
    from trottertracks import c1_stage1 as s
    from trottertracks.c1_general_k_adapter import from_ffsim_number_arrays
    p,a=synthetic();model=from_ffsim_number_arrays(one_body_tensor=s.corrected_kappa(p['h'],p['g']),diag_coulomb_mats=a['J'],orbital_rotations=a['U'],constant=.31,z_representation=False,frame_ids=[f'F{i}' for i in range(8)],group_ids=[f'F{i}' for i in range(8)],source_identity='ARTIFICIAL',input_identity='ARTIFICIAL',one_body_role='ffsim_corrected_kappa',enable_general_k=True)
    with threadpool_limits(limits=1):
        car=s.car_bilinears(8);raw=s.raw_base(s.corrected_kappa(p['h'],p['g']),.31,car);other=s.candidate_base(model)
        for J,U,frame in zip(a['J'],a['U'],model.frames,strict=True):
            raw+=s.raw_frame(J,U,car)[0];other+=s.candidate_frame(frame)[0]
        np.testing.assert_allclose(raw,other,atol=1e-12,rtol=0)
        assert raw[0,0]==pytest.approx(.31) # scalar retained exactly once incl vacuum phase
        ctx=e.old.molecule.SectorCAR(4);G=ctx.generators(**p,**a)
        e.old.molecule.representation_gate(ctx,p,a['J'],a['U'],G)
        ix=s.bit_reverse_indices(8)[ctx.sector.basis_indices]
        np.testing.assert_allclose(sum(G),raw[np.ix_(ix,ix)],atol=1e-12,rtol=0)
        # controlled full operator target keeps scalar relative phase; no phase fit.
        from scipy.linalg import expm
        evo=expm(-.1j*raw);controlled=np.zeros((512,512),complex);controlled[:256,:256]=np.eye(256);controlled[256:,256:]=evo
        assert controlled[256,256]==pytest.approx(np.exp(-.031j),abs=1e-12)
        # New sparse-J model enters the unchanged frame/signed-RTE interface.
        from fractions import Fraction
        from trottertracks import c1_stage2_native as native,c1_stage3_connection as connection,c_pr_aware_df as objective
        nm=native.prepare_cdf(model);plan=connection.split_native_model(nm,2)
        tail,config,distribution=connection.configure_rte(plan,Fraction(1,10),128,0)
        assert config.rte_steps==128 and len(plan.atoms)<=144
        assert tail.lambda_r==pytest.approx(float(np.sum(abs(objective.coefficients(a['J'][2:],np)))),abs=1e-12)
        assert [frame.original_index for frame in model.frames]==list(range(8))
        assert [frame.group_id for frame in model.frames]==[f'F{i}' for i in range(8)]

        np.testing.assert_allclose(controlled.conj().T@controlled,np.eye(512),atol=1e-12,rtol=0)


@pytest.mark.parametrize('fault',[None,'budget','optimizer','representation'])
def test_mock_stage_DF_end_to_end_21_calls_and_STOP(tmp_path,monkeypatch,fault):
    p,a=synthetic();np.savez(tmp_path/'input.npz',**p);e.io.save(tmp_path/'controls.json',dict(status='STAGE_A_PROXY_AND_DF_PASS'))
    e.io.save(tmp_path/'source.json',dict(files={}))
    c=dict(input='input.npz',controls='controls.json',entry='entry.py',source_manifest='source.json')
    monkeypatch.setattr(e,'preflight',lambda *args,**kw:c);monkeypatch.setattr(e.subprocess,'check_output',lambda *args,**kw:'ARTIFICIAL_SOURCE\n')
    def launch(args,**kwargs):
        role=args[args.index('--worker')+1];out=tmp_path/e.FRESH/role;out.mkdir();limits=e.PREPARE if role=='prepare' else e.FIT
        name='initial.npz' if role=='prepare' else 'final.npz';data={k:v.copy() for k,v in a.items()}
        if role=='fit' and fault=='budget':data['J']*=100
        np.savez(out/name,**data)
        e.io.save(out/'STOP.json',dict(failed=False,scientific_units=sum(limits.values()),call_counts=limits,retry=0,resume=0,mandatory_stop=True))
        if role=='fit':
            e.io.save(out/'optimizer.json',dict(success=fault!='optimizer'));e.io.save(out/'initial_diagnostic.json',dict());e.io.save(out/'final_diagnostic.json',dict())
    monkeypatch.setattr(e.subprocess,'run',launch)
    if fault=='representation':
        def fail(*args,**kwargs):raise ValueError('ARTIFICIAL_REPRESENTATION_FAIL')
        monkeypatch.setattr(e.old.molecule,'representation_gate',fail)
    with threadpool_limits(limits=1):
        if fault=='representation':
            with pytest.raises(ValueError,match='REPRESENTATION'):e.run(tmp_path,tmp_path/'c.json','PY')
        else:
            r=e.run(tmp_path,tmp_path/'c.json','PY');assert r['PF_RTE_shots_native']==0
            assert r['sector']['initial']['status']=='SECTOR_EVALUATED'
            assert r['status']==('DF_ELIGIBLE_NOT_PERFORMANCE_GO' if fault is None else 'DF_NOT_MET_OR_OPTIMIZER_UNSUCCESSFUL')
    stop=e.io.read(tmp_path/e.FRESH/'STOP.json');assert stop['mandatory_stop'] and stop['retry']==stop['resume']==0
    if fault!='representation':assert stop['scientific_units']==e.MAX_CALLS==21 and stop['call_counts']==e.LIMITS
    else:assert stop['failed'] and (tmp_path/e.FRESH/'failure.json').exists()
    with pytest.raises(ValueError,match='FRESH'):e.run(tmp_path,tmp_path/'c.json','PY')


def test_passive_preflight_no_array_decode_and_hash_mismatch(tmp_path):
    (tmp_path/'opaque').write_bytes(b'not a numeric array');e.io.save(tmp_path/'source.json',dict(files={}))
    c=dict(scope=e.SCOPE,implementation=m.VERSION,conditions=e.CONDITIONS,call_limits=e.LIMITS,maximum_logical_units=e.MAX_CALLS,fresh_output=e.FRESH,source_manifest='source.json',source_manifest_sha256=e.io.sha(tmp_path/'source.json'),saved_inputs={'opaque':e.io.sha(tmp_path/'opaque')},evaluation_packages={})
    e.io.save(tmp_path/'c.json',c);assert e.preflight(tmp_path,tmp_path/'c.json')==c
    (tmp_path/'opaque').write_bytes(b'changed')
    with pytest.raises(ValueError,match='INPUT_SOURCE'):e.preflight(tmp_path,tmp_path/'c.json')


def test_import_is_passive_in_fresh_process():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_df_structural_execution as e;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'));assert e.MAX_CALLS==21"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
