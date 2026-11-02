"""Only artificial tensors/matrices/circuits and mocked chemistry/optimizer."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import json,subprocess,sys
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_molecule_extension as m


def artificial(n,rotated=False):
    U=np.eye(n)
    if rotated:U[:2,:2]=[[.8,-.6],[.6,.8]]
    eta=np.arange(1,n+1)/20
    A=U@np.diag(eta)@U.T
    g=np.einsum('pq,rs->pqrs',A,A)
    J=(eta[:,None]*eta[None,:])[None,:,:]
    return dict(h=np.diag(np.arange(n)/10-.3).astype(float),g=g,constant=np.asarray(.125),mo_coeff=np.eye(n)),J,U[None,:,:]


def factory(n,**kw):
    def f(p,call):
        assert p['atoms']==n
        call('scf','ARTIFICIAL no molecule',lambda:None)
        return call('spatial_integrals','ARTIFICIAL no chemistry',lambda:artificial(n,**kw)[0])
    return f


def provider(n):
    _,J,U=artificial(n)
    return lambda g,r,opt:(J.copy(),U.copy())


def test_scope_dimensions_memory_rank_ladder_and_no_h4_rank_inheritance():
    a,b=m.specification(2),m.specification(6)
    assert (a['sector_dimension'],a['full_dimension'],a['controlled_dimension'])==(4,16,32)
    assert (b['sector_dimension'],b['full_dimension'],b['controlled_dimension'])==(400,4096,8192)
    assert b['sector_matrix_bytes']==2_560_000 and b['controlled_matrix_bytes']==1_073_741_824
    assert a['rank_ladder']==[1,2,3] and b['rank_ladder']==list(range(1,22))
    assert sum(m.call_limits(2).values())==196 and sum(m.call_limits(6).values())==129
    assert 'compile' not in m.call_limits(6) and not b['native_allowed']
    with pytest.raises(ValueError):m.specification(4)


def test_rank_choice_needs_physical_norm_and_guard_and_no_monotonic_assumption():
    def row(r,e,guard=0.):return dict(rank=r,status='CONVERGED',sector_error=e,numerical_guard=guard)
    assert m.rank_choice([row(1,.2),row(2,.3),row(3,.01)],.02)==3
    assert m.rank_choice([row(1,.01,.02)],.02) is None
    with pytest.raises(ValueError):m.rank_choice([dict(status='FAILED')],.02)
    with pytest.raises(ValueError):m.rank_choice([row(1,float('nan'))],.02)


def test_one_existing_fitter_call_with_requested_rank_and_failure_stop():
    p,J,U=artificial(2);res=SimpleNamespace(success=True,status=0,message='ARTIFICIAL',fun=0.,nit=1,nfev=1)
    mock=Mock(return_value=(J,U,res));seen=[]
    a,b=m.factor(p['g'],2,optimize=True,fitter=mock,metadata_callback=seen.append)
    assert mock.call_count==1 and mock.call_args.kwargs==dict(tol=1e-8,max_vecs=2,optimize=True,
        cholesky=True,method='L-BFGS-B',return_optimize_result=True)
    assert seen[0]['success'] and np.array_equal(a,J)
    res.success=False
    with pytest.raises(ValueError,match='OPTIMIZER_STOP'):m.factor(p['g'],2,optimize=True,fitter=mock)
    mock=Mock(return_value=(J,U));m.factor(p['g'],3,optimize=False,fitter=mock)
    assert mock.call_count==1 and mock.call_args.kwargs['optimize'] is False


@pytest.mark.parametrize('rotated',[False,True])
def test_sparse_sector_CAR_physical_raw_general_K_and_full_reference_agree(rotated):
    p,J,U=artificial(2,rotated=rotated);ctx=m.SectorCAR(2)
    physical=ctx.physical(p['h'],p['g'],p['constant'])
    G=ctx.generators(p['h'],p['g'],p['constant'],J,U)
    assert np.max(abs(sum(G)-physical))<1e-13
    report=m.representation_gate(ctx,p,J,U,G);assert report['max_entry']<1e-13
    car=m.stage1.car_bilinears(4)
    full=m.stage1.raw_base(m.stage1.corrected_kappa(p['h'],p['g']),p['constant'],car)+m.stage1.raw_frame(J[0],U[0],car)[0]
    reverse=m.stage1.bit_reverse_indices(4);take=reverse[ctx.sector.basis_indices]
    assert np.max(abs(full[np.ix_(take,take)]-physical))<1e-13


def test_H6_sector_build_never_calls_full_CAR_or_constructs_full_dense():
    p,J,U=artificial(6);forbidden=Mock(side_effect=AssertionError('No full Fock CAR'))
    with patch.object(m.stage1,'car_bilinears',forbidden),patch.object(m,'native_H2',forbidden):
        ctx=m.SectorCAR(6);G=ctx.generators(p['h'],p['g'],p['constant'],J,U)
        assert all(x.shape==(400,400) for x in G)
        assert all(x.shape==(400,400) for x in ctx.bilinears.values())
        report=m.representation_gate(ctx,p,J,U,G)
    assert report['max_entry']<1e-12;forbidden.assert_not_called()


def test_PF_rounds_use_squaring_and_original_schedule_not_s_times_long_loop():
    G=[np.array([[.2,.03],[.03,-.1]]),np.diag([.01,-.02])]
    records=[]
    def call(k,l,f):records.append((k,l));return f()
    rows=m.PF_rounds(G,4,call,'ARTIFICIAL')
    assert len(rows)==13 and len(records)==14
    assert [x['m'] for x in rows]==list(range(13))
    assert all(x['pf_operator_error']>=0 and x['attenuation']==1. for x in rows)
    from scipy.linalg import expm
    pf=m.reference.reference_kernels(G,.025,ld=1)['pf']
    expected=np.linalg.norm(np.linalg.matrix_power(pf,4)-expm(-.1j*sum(G)),2)
    assert rows[0]['pf_operator_error']==pytest.approx(expected,abs=1e-14)


def test_artificial_H2_end_to_end_stage1_to_CA_cost_and_fresh_STOP(tmp_path):
    out=tmp_path/'h2'
    forbidden=Mock(side_effect=AssertionError('No SCF or real fit'))
    with patch.object(m,'factor',forbidden):
        m.run(2,out,'UNUSED',input_factory=factory(2),factor_provider=provider(2))
    result=json.loads((out/'result.json').read_text());stop=json.loads((out/'STOP.json').read_text())
    assert result['status']=='H2_NUMERICAL_PILOT_COMPLETE' and not stop['failed']
    assert all(c['energy']['same_accuracy'] for c in result['native']['candidates'])
    assert len(result['native']['candidates'])==4 and len(result['native']['comparisons'])==2
    assert result['CA_achieved'] is None and result['native']['guaranteed_total_resource_superiority'] is False
    assert stop['call_counts']['CDF_fit']==1 and stop['call_counts']['compile']==7
    assert stop['scientific_units']<=196 and stop['retry']==stop['resume']==0
    assert len(list((out/'circuits').glob('*.qpy')))==14
    old=(out/'STOP.json').read_bytes()
    with pytest.raises(FileExistsError):m.run(2,out,'UNUSED',input_factory=factory(2),factor_provider=provider(2))
    assert (out/'STOP.json').read_bytes()==old


def test_H6_run_stops_after_DF_PF_and_never_dispatches_native_or_shots(tmp_path):
    def pf(G,s,call,label,**kw):
        call('sector_kernel_reference',label,lambda:None)
        return [call('PF_round_assessment',f'{label}:m{i}',lambda i=i:dict(m=i,pf_operator_error=0.,attenuation=1.)) for i in range(13)]
    forbidden=Mock(side_effect=AssertionError('H6 native excluded'))
    from threadpoolctl import threadpool_limits
    with patch.object(m,'PF_rounds',side_effect=pf),patch.object(m,'native_H2',forbidden),threadpool_limits(1):
        m.run(6,tmp_path/'h6','UNUSED',input_factory=factory(6),factor_provider=provider(6))
    result=json.loads((tmp_path/'h6/result.json').read_text())
    stop=json.loads((tmp_path/'h6/STOP.json').read_text())
    assert result['status']=='H6_DF_PF_ONLY_COMPLETE' and not stop['failed']
    assert result['native']=='NOT_EXECUTED_OUT_OF_SCOPE' and result['shots'] is result['total_cost'] is None
    assert 'compile' not in stop['call_counts'] and stop['scientific_units']<=129
    assert not (tmp_path/'h6/circuits').exists();forbidden.assert_not_called()


def test_failure_keeps_partial_evidence_without_retry(tmp_path):
    bad=Mock(side_effect=ValueError('ARTIFICIAL_FIT_FAILURE'));out=tmp_path/'failed'
    with pytest.raises(ValueError,match='ARTIFICIAL_FIT_FAILURE'):
        m.run(2,out,'UNUSED',input_factory=factory(2),factor_provider=bad)
    stop=json.loads((out/'STOP.json').read_text());assert stop['failed'] and stop['retry']==stop['resume']==0
    assert (out/'input.npz').exists() and (out/'failure.json').exists()
    assert not (out/'PF.json').exists()


def test_rank_ladder_exhaustion_is_not_silently_extended(tmp_path):
    original=m.rank_assessment
    def reject(*args,**kw):
        row,G=original(*args,**kw);row['sector_error']=1.;return row,G
    forbidden=Mock(side_effect=AssertionError('No PF/native after DF budget failure'))
    with patch.object(m,'rank_assessment',side_effect=reject),patch.object(m,'PF_rounds',forbidden),patch.object(m,'native_H2',forbidden):
        m.run(2,tmp_path/'out','UNUSED',input_factory=factory(2),factor_provider=provider(2))
    r=json.loads((tmp_path/'out/result.json').read_text())
    assert r['status']=='DF_BUDGET_NOT_MET' and r['arm']=='XDF'
    assert len(r['rank_records']['XDF'])==1 and not (tmp_path/'out/PF.json').exists()
    forbidden.assert_not_called()


def test_fit_subprocess_keeps_venv_symlink_and_exact_one_worker_call(tmp_path):
    n=2;packet,J,U=artificial(n)
    link=tmp_path/'env/bin/python';link.parent.mkdir(parents=True);link.symlink_to(sys.executable)
    seen=[]
    def child(command,check):
        seen.append(command);assert check
        dest=Path(command[command.index('--output')+1])
        with dest.open('xb') as f:np.savez(f,J=J,U=U)
    forbidden=Mock(side_effect=AssertionError('Stop after fit boundaries'))
    # Rank assessment failure prevents all subsequent physics. The worker call
    # itself is mocked, so this invokes no optimizer or molecule.
    with patch.object(m.subprocess,'run',side_effect=child),patch.object(m,'rank_assessment',forbidden):
        with pytest.raises(AssertionError,match='fit boundaries'):
            m.run(n,tmp_path/'out',link,input_factory=factory(n))
    assert len(seen)==1 and seen[0][0]==str(link.absolute())
    assert '--fit-worker' in seen[0] and '--optimize' not in seen[0]


def test_mocked_PySCF_generation_uses_registered_geometry_and_existing_ao2mo():
    n=2;p,J,U=artificial(n)
    mf=SimpleNamespace(converged=True,conv_tol=None,mo_coeff=np.eye(n),
        kernel=Mock(),get_hcore=lambda:p['h'],get_ovlp=lambda:np.eye(n))
    molecule=SimpleNamespace(energy_nuc=lambda:.125)
    mol_factory=Mock(return_value=molecule)
    pyscf=SimpleNamespace(gto=SimpleNamespace(M=mol_factory),scf=SimpleNamespace(RHF=lambda mol:mf),
        ao2mo=SimpleNamespace(kernel=Mock(return_value='ARTIFICIAL_COMPRESSED'),restore=Mock(return_value=p['g'])))
    calls=[]
    with patch.dict(sys.modules,{'pyscf':pyscf}):
        out=m.integrals(2,lambda k,l,f:(calls.append(k),f())[1])
    assert calls==['scf','spatial_integrals'] and mf.conv_tol==1e-12
    kw=mol_factory.call_args.kwargs
    assert kw['atom']==[('H',(0.,0.,0.)),('H',(0.,0.,1.))] and kw['unit']=='Angstrom'
    assert kw['basis']=='sto-3g' and np.array_equal(out['g'],p['g'])


def test_import_and_default_cli_are_passive():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c1_molecule_extension;assert not any(x in sys.modules for x in ['numpy','qiskit','scipy','pyscf','ffsim'])"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_research_c_c1_h2_h6.py'),
        '--molecule','6','--contract','NEVER_READ','--execute'],capture_output=True,text=True)
    assert p.returncode==2 and 'fresh' in p.stderr
