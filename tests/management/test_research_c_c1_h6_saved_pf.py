"""Artificial end-to-end sector PF, saved-only dispatch, partial STOP and scope."""
from pathlib import Path
import json,subprocess,sys
from unittest.mock import Mock,patch
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_saved_pf as m


def artificial(root):
    U=np.array([[[.8,-.6],[.6,.8]]]);J=np.array([[[.02,.003],[.003,.01]]])
    g=np.einsum('kpi,kqi,kij,krj,ksj->pqrs',U,U,J,U,U,optimize=True)
    np.savez(root/'input.npz',h=np.diag([-.2,.1]),g=g,constant=np.asarray(.125))
    np.savez(root/'CDF.npz',J=J,U=U);np.savez(root/'XDF.npz',J=J.copy(),U=U.copy())
    row=dict(actual_rank=1,guard_policy=m.molecule.GUARD_POLICY,sector_error=0.,numerical_guard=1e-12)
    m.io.save(root/'CDF.json',row);m.io.save(root/'XDF.json',row);m.io.save(root/'optimizer.json',dict(success=True))
    return dict(original_input='input.npz',factor_arrays={a:a+'.npz' for a in m.ARMS},
        saved_sector_rows={a:a+'.json' for a in m.ARMS},saved_optimizer='optimizer.json',
        conditions=dict(m.CONDITIONS,atoms=2,rank=1,sector_dimension=4))


@pytest.mark.parametrize('fault',[None,'DF_budget','representation','ground_alias','PF_nonfinite'])
def test_artificial_saved_pipeline_no_fit_all_cells_or_partial_STOP(tmp_path,fault):
    c=artificial(tmp_path);out=tmp_path/'out';forbidden=Mock(side_effect=AssertionError('NO_CHEMISTRY_FIT_CIRCUIT_OR_NEW_NORM'))
    original=m.molecule.PF_rounds
    def pf(G,s,call,label,**kwargs):
        if fault=='PF_nonfinite':
            return [call('PF_round_assessment',label+':m0',lambda:dict(m=0,pf_operator_error=float('nan'),attenuation=1.,rte_corrected_operator_error=0.))]
        return original(G,s,call,label,**kwargs)
    if fault=='DF_budget':
        f=tmp_path/'CDF.json';v=m.io.read(f);v['sector_error']=.1;f.write_text(json.dumps(v))
    patches=[patch.object(m,'preflight',return_value=c),patch.object(m.molecule,'integrals',forbidden),
        patch.object(m.molecule,'factor',forbidden),patch.object(m.molecule,'rank_assessment',forbidden),
        patch.object(m.molecule,'native_H2',forbidden),patch.object(m.molecule.stage1,'car_bilinears',forbidden),
        patch.object(m.molecule,'PF_rounds',side_effect=pf)]
    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in patches:stack.enter_context(p)
        if fault=='representation':stack.enter_context(patch.object(m.molecule,'representation_gate',side_effect=ValueError('ARTIFICIAL_REPRESENTATION_FAIL')))
        if fault=='ground_alias':stack.enter_context(patch.object(m,'ground_alias',side_effect=ValueError('ARTIFICIAL_GROUND_FAIL')))
        if fault:
            with pytest.raises(ValueError):m.run(tmp_path,tmp_path/'unused',out)
        else:m.run(tmp_path,tmp_path/'unused',out)
    forbidden.assert_not_called();stop=m.io.read(out/'STOP.json')
    assert stop['failed']==bool(fault) and stop['scientific_units']<=72 and stop['retry']==stop['resume']==0
    if fault:
        assert (out/'failure.json').exists() and not (out/'result.json').exists()
        if fault in ('DF_budget','representation','ground_alias'):
            assert not stop['call_counts'].get('sector_kernel_reference') and not list((out/'cells').glob('*'))
        if fault=='PF_nonfinite':assert not stop['call_counts'].get('candidate_summary')
    else:
        r=m.io.read(out/'result.json');assert stop['scientific_units']==72
        assert len(list((out/'cells').glob('*.json')))==52 and len(list((out/'kernels').glob('*.npz')))==4
        assert list(r['candidates'])==['CDF:s1','CDF:s4','XDF:s1','XDF:s4']
        assert r['new_fit']==r['optimizer']==r['native_build_compile']==0
        assert r['CA_achieved'] is r['shots'] is r['total_cost'] is None
        assert all(v['summary']['status']=='POTENTIALLY_FEASIBLE' for v in r['candidates'].values())
        assert all(v['summary']['synthesis_error'] is None for v in r['candidates'].values())
        assert m.io.read(out/'CDF_representation.json')['max_entry']<1e-12
        # Independent first-round evolution, same elapsed time and original order.
        from scipy.linalg import expm
        for label,data in r['candidates'].items():
            with np.load(out/'kernels'/(label.replace(':','_')+'.npz')) as a:kernel=a['pf']
            packet,factors,_,_=m.load_saved(tmp_path,c);ctx=m.molecule.SectorCAR(2)
            G=ctx.generators(packet['h'],packet['g'],packet['constant'],factors[data['arm']]['J'],factors[data['arm']]['U'])
            expected=np.linalg.norm(np.linalg.matrix_power(kernel,data['s'])-expm(-.1j*sum(G)),2)
            assert data['rounds'][0]['pf_operator_error']==pytest.approx(expected,abs=1e-13)
    original_stop=(out/'STOP.json').read_bytes()
    with patch.object(m,'preflight',return_value=c),pytest.raises(FileExistsError):m.run(tmp_path,tmp_path/'unused',out)
    assert (out/'STOP.json').read_bytes()==original_stop


def test_sufficient_bound_rejection_is_not_scientific_FAILURE_or_CA_claim():
    rows=[dict(m=i,pf_operator_error=.3 if i==12 else .01,
        known_part_status='INFEASIBLE_REGISTERED_PHASE_ROOM' if i==12 else 'POTENTIALLY_FEASIBLE') for i in range(13)]
    out=m.summarize(rows);assert out['failed_sufficient_rounds']==[12]
    assert out['status']=='INFEASIBLE_REGISTERED_PHASE_ROOM' and out['CA_achieved'] is None
    with pytest.raises(ValueError,match='REGISTERED_ROUNDS'):m.summarize(rows[:-1])


@pytest.mark.parametrize('fault',['nonfinite','nonsymmetric','nonorthogonal','rank','failed_optimizer','guard'])
def test_saved_factor_gate_rejects_invalid_evidence(tmp_path,fault):
    c=artificial(tmp_path);_,data,rows,opt=m.load_saved(tmp_path,c);d=data['CDF'];r=rows['CDF']
    if fault=='nonfinite':d['U'][0,0,0]=np.nan
    if fault=='nonsymmetric':d['J'][0,0,1]=.2
    if fault=='nonorthogonal':d['U'][0,0,0]=.1
    if fault=='rank':r['actual_rank']=2
    if fault=='failed_optimizer':opt['success']=False
    if fault=='guard':r['numerical_guard']=-1e-12
    with pytest.raises(ValueError):m.saved_DF_gate(d,r,opt,rank=1,n=2,budget=.00159/8)


def test_actual_ground_alias_rule_and_finite_evidence(tmp_path):
    pair,row=m.ground_alias(np.diag([-.2,.3]),tmp_path,'artificial')
    assert row['status']=='PASS' and row['residual']==0 and pair[0][0]==-.2
    with pytest.raises(ValueError,match='GROUND_OR_ALIAS'):m.ground_alias(np.diag([-40.,.3]),tmp_path,'bad_alias')
    with pytest.raises(ValueError,match='NONHERMITIAN'):m.ground_alias(np.array([[0.,1.],[0.,0.]]),tmp_path,'bad_H')


def contract(root):
    (root/'SOURCE').write_text('artificial');m.io.save(root/'SOURCE.json',dict(files={'SOURCE':m.io.sha(root/'SOURCE')}))
    (root/'bytes').write_text('not numerical')
    return dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',conditions=m.CONDITIONS,
        allowances=m.ALLOWANCES,call_limits=m.CALL_LIMITS,maximum_logical_units=72,
        candidate_order=[f'{a}:s{s}' for a,s in m.CANDIDATES],new_fit=0,optimizer=0,retry=0,resume=0,
        guard_policy=m.molecule.GUARD_POLICY,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        saved_inputs={'bytes':m.io.sha(root/'bytes')},original_input='bytes',saved_optimizer='bytes',
        factor_arrays={a:'bytes' for a in m.ARMS},saved_sector_rows={a:'bytes' for a in m.ARMS},evaluation_packages={})


@pytest.mark.parametrize('change',[None,'conditions','grid','calls','fit','input','source'])
def test_passive_preflight_fixed_SCOPE_no_numerical_decode(tmp_path,change):
    c=contract(tmp_path)
    if change=='conditions':c['conditions']=dict(c['conditions'],rank=18)
    if change=='grid':c['candidate_order']=c['candidate_order'][::-1]
    if change=='calls':c['maximum_logical_units']=73
    if change=='fit':c['new_fit']=1
    if change=='input':c['original_input']='unlisted'
    if change=='source':(tmp_path/'SOURCE').write_text('modified')
    m.io.save(tmp_path/'contract.json',c)
    with patch.object(np,'load',side_effect=AssertionError('NO_MOLECULAR_DECODE')):
        if change:
            with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'contract.json')
        else:assert m.preflight(tmp_path,tmp_path/'contract.json')==c


def test_calls_and_import_and_execution_default_are_passive():
    assert sum(m.CALL_LIMITS.values())==72 and len(m.CANDIDATES)==4
    assert not set(m.CALL_LIMITS)&{'scf','fit','optimizer','rank_assessment','compile','shots','cost'}
    code="import sys;sys.path.insert(0,'src');from trottertracks import c1_h6_saved_pf;assert not any(x in sys.modules for x in ['numpy','qiskit','scipy','pyscf','ffsim'])"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_research_c_c1_h6_saved_pf.py'),
        '--contract','NEVER_READ','--execute'],capture_output=True,text=True)
    assert p.returncode==2 and 'Fresh' in p.stderr
