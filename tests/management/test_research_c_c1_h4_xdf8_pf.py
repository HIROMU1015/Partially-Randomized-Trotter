"""Artificial packets/models only; never decode real H4 or its saved circuits."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import c1_h4_xdf8_pf as m


def fixture(tmp_path, monkeypatch):
    root = tmp_path/'ARTIFICIAL'; root.mkdir()
    h = np.diag([-.3, -.1, .1, .2]); h[0, 1] = h[1, 0] = .037
    rng = np.random.default_rng(118)
    gs = rng.normal(size=(10, 4, 4))*.03; gs = gs+gs.transpose(0, 2, 1)
    G = np.array([np.kron(x, np.eye(2)) for x in gs], dtype=complex)
    lam = np.array([(-1.)**i*(i+1)*.0002 for i in range(10)])
    path = root/'artificial.npz'
    np.savez(path, one_mo=h, eri_chemist=np.zeros((4,)*4), nuclear_repulsion=np.array(.25),
        mo_coeff=np.eye(4), df_one_body=np.kron(h, np.eye(2)).astype(complex), df_lambdas=lam,
        df_g_matrices=G, forbidden_old_ground=np.array('DO_NOT_LOAD'))
    digest = m.io.sha(path); monkeypatch.setattr(m.io, 'P_H4_SHA256', digest)
    m.io.save(root/'row.json', dict(rank=8, status='PASS', DF_budget_Ha=.00019875,
        allowance_version=m.stage1.ALLOWANCE_VERSION, sector_error_Ha=1e-5,
        allowance_Ha=1e-9, norm_plus_allowance_Ha=1.0001e-5))
    m.io.save(root/'old_STOP.json', dict(failed=False, mandatory_stop=True, scientific_units=356))
    m.io.save(root/'old_SOURCE.json', dict(execution_commit='ARTIFICIAL_DF_SOURCE'))
    (root/'code.txt').write_text('ARTIFICIAL CODE\n')
    m.io.save(root/'SOURCE.json', dict(files={'code.txt':m.io.sha(root/'code.txt')}))
    c = dict(scope=m.SCOPE, status='UNAPPROVED_SCIENCE_CONTRACT', conditions=copy.deepcopy(m.CONDITIONS),
        allowances=copy.deepcopy(m.ALLOWANCES), call_limits=copy.deepcopy(m.LIMITS), maximum_logical_units=m.MAX_CALLS,
        source_manifest='SOURCE.json', source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        original_input=dict(sha256=digest), records=dict(DF_row='row.json', DF_STOP='old_STOP.json', DF_SOURCE='old_SOURCE.json'),
        saved_inputs={p:m.io.sha(root/p) for p in ['row.json','old_STOP.json','old_SOURCE.json']},
        lineage=dict(DF_evaluation_SOURCE='ARTIFICIAL_DF_SOURCE'), evaluation_packages={},
        fresh_output=m.FAMILY+'/<execution-date>/run01', **{k:0 for k in m.FORBIDDEN})
    m.io.save(root/'contract.json', c)
    return root, c, path


def execute(root, path, **kw):
    original = subprocess.check_output
    def git_only(command, *args, **kwargs):
        return 'ARTIFICIAL_SOURCE\n' if command[:2] == ['git','rev-parse'] else original(command,*args,**kwargs)
    output = root/m.FAMILY/'2026-10-26/run01'
    with patch.object(m.subprocess, 'check_output', side_effect=git_only):
        return m.run(root, root/'contract.json', path, output, **kw), output


def test_passive_preflight_and_clean_import_no_numeric_work(tmp_path, monkeypatch):
    root, c, path = fixture(tmp_path, monkeypatch)
    with patch.object(np, 'load', side_effect=AssertionError('NO_DECODE')):
        assert m.preflight(root, root/'contract.json', path) == c
    assert not (root/m.FAMILY).exists()
    subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,sys.argv[1]);from trottertracks import c1_h4_xdf8_pf;assert 'numpy' not in sys.modules",str(ROOT/'src')], check=True)


def test_artificial_e2e_real_routes_26_cells_93_units_and_ground(tmp_path, monkeypatch):
    root, c, path = fixture(tmp_path, monkeypatch)
    forbidden = AssertionError('No DF norm/physical generation/fitter/circuit')
    with patch.object(m.stage1, 'physical_sector', side_effect=forbidden), \
         patch.object(m.prefix, 'error_norm', side_effect=forbidden), \
         patch.object(m.stage1, 'fit_once', side_effect=forbidden):
        result, output = execute(root, path, workers=2)
    assert result['status']=='XDF8_STAGE1_COMPLETE' and result['representation']=='PASS'
    assert result['ground']['status']=='PASS' and result['saved_DF']['recomputed'] is False
    assert {r['summary']['status'] for r in result['candidates'].values()}=={'POTENTIALLY_FEASIBLE'}
    assert len(list((output/'cells').glob('*.json')))==26
    stop = m.io.read(output/'STOP.json')
    assert stop['scientific_units']==m.MAX_CALLS==93 and stop['call_counts']==m.LIMITS
    assert stop['failed'] is False and stop['retry']==stop['resume']==0
    journal=[json.loads(r) for r in (output/'science_calls.jsonl').read_text().splitlines()]
    assert sum(r['event']=='attempt' for r in journal)==sum(r['event']=='COMPLETE' for r in journal)==93
    with np.load(output/'generators.npz') as a: generators=a['generators']
    with np.load(output/'ground_XDF8.npz') as a: psi=a['sector_state']; energy=float(a['energy'])
    assert np.linalg.norm(sum(generators)@psi-energy*psi)<1e-12
    with np.load(output/'spectrum_XDF8.npz') as a:
        assert a['energies'].shape==(36,) and a['eigenvectors'].shape==(36,36)
    # Independently rebuild chronological S2 and exact evolution for short and long round.
    for s in (1,4):
        h=.1/s; steps=[(i,h/2) for i in range(8)]+[(8,h)]+[(i,h/2) for i in reversed(range(8))]
        W=np.eye(36,dtype=complex)
        for i,t in steps: W=expm(-1j*t*generators[i])@W
        with np.load(output/'kernels'/f'XDF8_s{s}.npz') as a: np.testing.assert_allclose(a['pf'],W,atol=1e-14,rtol=0)
        for index in (0,12):
            error=float(np.linalg.norm(np.linalg.matrix_power(W,s*(1<<index))-expm(-1j*.1*(1<<index)*sum(generators)),2))
            row=result['candidates'][str(s)]['rounds'][index]
            assert abs(row['pf_operator_error']-error)<2e-11
            assert row['occurrences']==s*(1<<index) and row['time']==.1*(1<<index)
            assert row['kernel_time']==.1/s and row['sufficient_condition']=='PASS'


def test_raw_general_K_and_unchanged_prefix8_base_order(tmp_path, monkeypatch):
    root,c,path=fixture(tmp_path,monkeypatch);b=m.load_saved(root,c,path);ctx=m.contexts();m.identity(b,ctx)
    model=m.k_model(b);raw=[m.raw_factor(b,ctx,i) for i in range(9)]
    expected=[.25+sum(b['packet']['df_one_body'][p,p] for p in range(8) if state>>p&1)
        for state in ctx.df.indices]
    np.testing.assert_allclose(np.diag(raw[0]),expected,atol=1e-15,rtol=0)
    assert len(model.frames)==8 and [f.original_index for f in model.frames]==list(range(8))
    for i in range(9):
        gate,K=m.representation(ctx,model,raw[i],i);assert gate['max_entry']<1e-12
    # The unused rank9/10 factors cannot affect this target.
    p=b['packet'];p['df_lambdas']=p['df_lambdas'].copy();p['df_lambdas'][8:]=1e8
    for i in range(9):np.testing.assert_array_equal(m.raw_factor(b,ctx,i),raw[i])
    assert m.k_model(b)==model
    broken=copy.copy(model)
    object.__setattr__(broken,'frames',model.frames[::-1])
    with pytest.raises(ValueError,match='FRAME_ORDER'):m.representation(ctx,broken,raw[1],1)


def test_nonzero_original_g_kappa_checked_once_no_rank_recorrection(tmp_path,monkeypatch):
    root,c,path=fixture(tmp_path,monkeypatch);b=m.load_saved(root,c,path);p=b['packet'];ctx=m.contexts()
    p['eri_chemist']=p['eri_chemist'].copy();p['eri_chemist'][0,1,1,0]=.25
    kappa=p['one_mo']-np.diag([.125,0,0,0]);p['df_one_body']=np.kron(kappa,np.eye(2)).astype(complex)
    with patch.object(m.stage1,'corrected_kappa',wraps=m.stage1.corrected_kappa) as correction:
        m.identity(b,ctx);model=m.k_model(b)
        for i in range(9):m.representation(ctx,model,m.raw_factor(b,ctx,i),i)
        assert correction.call_count==1


def fake_pf(error):
    def run(generators,s,call,label,*,eigenpair):
        call('sector_kernel_reference',label,lambda:(dict(pf=np.eye(36)),None,None))
        rows=[]
        for index in range(13):
            rows.append(call('PF_round_assessment',f'{label}:m{index}',lambda index=index:dict(m=index,
                pf_operator_error=error,attenuation=1.,rte_corrected_operator_error=0.)))
        return rows
    return run


@pytest.mark.parametrize('value,status',[(.4,'INFEASIBLE_REGISTERED_PHASE_ROOM'),(None,'UNRESOLVED')])
def test_pf_insufficient_or_unknown_is_not_correctness_fail(tmp_path,monkeypatch,value,status):
    root,c,path=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m.molecule,'PF_rounds',fake_pf(value))
    result,output=execute(root,path,workers=2)
    assert {r['summary']['status'] for r in result['candidates'].values()}=={status}
    assert not m.io.read(output/'STOP.json')['failed']
    assert all(r['summary']['CA_achieved'] is None and not r['summary']['native_candidate_eligible'] for r in result['candidates'].values())


@pytest.mark.parametrize('value',[float('nan'),-.1])
def test_invalid_pf_stops_dependent_calls_and_preserves_partial(tmp_path,monkeypatch,value):
    root,c,path=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m.molecule,'PF_rounds',fake_pf(value))
    with pytest.raises(ValueError,match='CORRECTNESS_STOP'):execute(root,path,workers=1)
    output=root/m.FAMILY/'2026-10-26/run01';stop=m.io.read(output/'STOP.json')
    assert stop['failed'] and stop['scientific_units']<93
    assert (output/'failure.json').exists() and (output/'partial_candidates.json').exists()


def test_ground_alias_uses_H4_window_and_unresolved_stops_pf(tmp_path,monkeypatch):
    indices=m.stage1.sector_indices(4,2,2)
    H=np.eye(36)*-11.;pair=np.linalg.eigh(H)
    assert m.ground_gate(H,pair,indices)['status']=='UNRESOLVED_ALIAS_WINDOW'
    root,c,path=fixture(tmp_path,monkeypatch)
    original=m.ground_gate
    def unresolved(*args):return dict(original(*args),status='UNRESOLVED_ALIAS_WINDOW')
    monkeypatch.setattr(m,'ground_gate',unresolved)
    with patch.object(m.molecule,'PF_rounds',side_effect=AssertionError('NO_PF_AFTER_UNRESOLVED')):
        result,output=execute(root,path,workers=1)
    assert result['status']=='UNRESOLVED_GROUND_ALIAS' and result['candidates']=={}
    assert m.io.read(output/'STOP.json')['call_counts'].get('PF_round_assessment',0)==0


@pytest.mark.parametrize('defect',['spin','basis','permutation','kappa','nonfinite','hermiticity','K','ground'])
def test_correctness_stop_preserves_evidence(tmp_path,monkeypatch,defect):
    root,c,path=fixture(tmp_path,monkeypatch);load=m.load_saved;contexts=m.contexts
    def broken(*args):
        b=load(*args);p=b['packet']
        if defect in ('spin','kappa','nonfinite','hermiticity'):
            p['df_one_body']=p['df_one_body'].copy()
            if defect=='spin':p['df_one_body'][0,1]=1e-20
            if defect=='kappa':p['df_one_body']+=.02*np.eye(8)
            if defect=='nonfinite':p['df_one_body'][0,0]=np.nan
            if defect=='hermiticity':p['df_one_body'][0,2]=p['df_one_body'][1,3]=.1j
        return b
    def wrong_context():
        ctx=contexts()
        if defect=='basis':ctx.df.indices=ctx.df.indices[::-1]
        if defect=='permutation':ctx.df.perm=ctx.df.perm[::-1]
        return ctx
    monkeypatch.setattr(m,'load_saved',broken);monkeypatch.setattr(m,'contexts',wrong_context)
    if defect=='K':
        original=m.general.general_k_coefficients
        def wrong_K(K):
            rows=list(original(K));support,value=rows[0];rows[0]=(support,value+.01);return rows
        monkeypatch.setattr(m.general,'general_k_coefficients',wrong_K)
    if defect=='ground':
        def wrong_ground(H,output):
            vals,vecs=np.linalg.eigh(H);vecs[:,0]*=2;return vals,vecs
        monkeypatch.setattr(m,'ground_solve',wrong_ground)
    with pytest.raises(ValueError):execute(root,path,workers=1)
    output=root/m.FAMILY/'2026-10-26/run01';stop=m.io.read(output/'STOP.json')
    assert stop['failed'] and stop['scientific_units']<93 and stop['retry']==stop['resume']==0
    assert (output/'SOURCE.json').exists() and (output/'failure.json').exists()
    assert not (output/'cells'/'XDF8_s1_m0.json').exists()


@pytest.mark.parametrize('name',['code.txt','row.json','artificial.npz'])
def test_source_input_identity_fail_before_decode(tmp_path,monkeypatch,name):
    root,c,path=fixture(tmp_path,monkeypatch);(root/name).write_bytes(b'ARTIFICIAL_DRIFT')
    with patch.object(np,'load',side_effect=AssertionError('NO_DECODE')):
        with pytest.raises(ValueError,match='SOURCE|INPUT'):m.preflight(root,root/'contract.json',path)


def test_saved_df_mismatch_no_reassessment(tmp_path,monkeypatch):
    root,c,path=fixture(tmp_path,monkeypatch);b=m.load_saved(root,c,path)
    b['records']['DF_row']['rank']=10
    with pytest.raises(ValueError,match='DF_IDENTITY'):m.saved_df_gate(b,c)


def test_fresh_output_and_no_resume(tmp_path,monkeypatch):
    root,c,path=fixture(tmp_path,monkeypatch);result,output=execute(root,path,workers=2)
    with pytest.raises(ValueError,match='FRESH_OUTPUT'):execute(root,path,workers=1)
    with pytest.raises(ValueError):m.output_path(root,root/m.FAMILY/'2026-10-26/run02')
