"""Artificial-only selected-JU control tests. Never decode saved molecule data."""
from copy import deepcopy
from pathlib import Path
import json,subprocess,sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from trottertracks import c_same_ju_strong_d as m


def test_passive_import_is_non_numeric():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_same_ju_strong_d as m;assert m.MAX_CALLS==156;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parents[2],check=True)


def saved_fixture(root):
    from test_research_c_pr_aware_df_m2 import synthetic_fixture
    from trottertracks import c1_molecule_extension as mol,c_pr_aware_df_stage_b as b
    from scipy.linalg import eigh
    # The old M2 invented fixture creates no repository scientific input.
    old,old_saved=synthetic_fixture(root)
    data=dict(np.load(root/old['exports']['proposed']));packet=dict(np.load(root/old['input']))
    np.savez(root/'selected.npz',**data);export=m.io.sha(root/'selected.npz')
    ctx=mol.SectorCAR(4);G=ctx.generators(**packet,**data);H=sum(G)
    np.savez(root/'physical.npz',**data,generators=np.asarray(G),H=H)
    np.savez(root/'basis.npz',sector_basis_indices=ctx.sector.basis_indices)
    c=dict(input=old['input'],export='selected.npz',reference='physical.npz',basis='basis.npz',
        ground=old['grounds']['proposed'],shells=old['shells'])
    physical=dict(actual_rank=8,candidate_sha256=export,kappa='original_g_corrected',scalar='nuclear_once',
        sector_error=1e-9,numerical_guard=1e-9,budget_Ha=.00019875,budget_PASS=True)
    scope='serial_native_kernel_composition_without_state_preparation'
    costs=dict(totals={k:1e12 for k in m.accuracy.METRICS},cost_scope=scope)
    pr={lane:dict(rounds=[dict(m=i,target_id='integrated:'+export) for i in range(13)],energy=dict(same_accuracy=True),conditional_totals=costs,new_over_old_PR={'rz_count':.9}) for lane in ('ideal_exact_native','assumed_raw_signal_budget')}
    pr['evidence']=dict(rounds=[dict(m=i,shots=None) for i in range(13)],energy=dict(same_accuracy=False),conditional_totals=dict(totals=None))
    new_source=dict(files={});m.io.save(root/'new_SOURCE.json',new_source)
    origin=dict(scope='C_PR_INTEGRATED_H4_REDUCED_PERFORMANCE_V1',target=m.m2.TARGET,compiler=m.COMPILER,
        input_paths=dict(input=c['input']),saved_inputs={c['input']:m.io.sha(root/c['input'])},source_manifest_sha256=m.io.sha(root/'new_SOURCE.json'))
    old_contract=dict(input=c['input'],exports=dict(proposed='OLD'),saved_inputs={'OLD':'OLD_HASH',c['input']:m.io.sha(root/c['input'])},target=m.m2.TARGET,compiler=m.COMPILER)
    records=dict(new_result=dict(selected=dict(id='update005',qualification=dict(status='NUMERICAL_PILOT_ADMITTED'),physical=physical),performance=dict(status='PERFORMANCE_NATIVE_COMPLETE',lanes=pr)),
        new_STOP=dict(failed=False,scientific_units=543),new_SOURCE=new_source,new_contract=origin,physical=physical,
        ground_report=old_saved['proposed_ground'],physical_ground_report=dict(status='PASS',alias_margin=1.6),shell_reports=old_saved['shell_reports'],
        old_D=dict(status='M2_SAME_JU_DETERMINISTIC_COMPLETE',candidates=dict(proposed=dict(cost=costs,export_sha256='OLD_HASH'))),
        old_D_STOP=dict(failed=False,scientific_units=253),old_D_SOURCE=dict(execution_commit='a6ffc23d94f6488a4769596f089a91c895d49214'),
        old_PR=dict(status='NATIVE128_FINITE_COHORT_COMPLETE',lanes={x:dict(conditional_totals=costs['totals']) for x in ('ideal_exact_native','assumed_raw_signal_budget')}),
        old_PR_STOP=dict(failed=False,scientific_units=332),old_PR_SOURCE=dict(execution_commit='dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad'),
        old_D_contract=old_contract,old_PR_contract=old_contract)
    c['records']={k:'new_SOURCE.json' if k=='new_SOURCE' else k+'.json' for k in records}
    for k,path in c['records'].items():
        if k!='new_SOURCE':m.io.save(root/path,records[k])
    names={c[k] for k in ('input','export','reference','basis','ground')};names.update(c['records'].values());names.update(p for x in c['shells'].values() for p in x.values())
    c['saved_inputs']={n:m.io.sha(root/n) for n in names}
    m.io.save(root/'test_SOURCE.json',dict(files={}))
    c['source_manifest']='test_SOURCE.json'
    return c,records


def passive_fixture(root):
    from trottertracks.c1_h4_native_fair import GAUSSIAN,PHASE
    (root/'opaque.npz').write_bytes(b'NOT_NUMERIC_DATA')
    m.io.save(root/'SOURCE.json',dict(files={}))
    c=dict(base_commit=m.BASE,scope=m.SCOPE,implementation=m.VERSION,conditions=m.CONDITIONS,
        compiler=m.COMPILER,target=m.m2.TARGET,allowances=m.m2.ALLOWANCES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        fresh_output=m.FRESH,retry=0,resume=0,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        records={k:'opaque.npz' for k in m.RECORDS},input='opaque.npz',export='opaque.npz',reference='opaque.npz',basis='opaque.npz',ground='opaque.npz',
        shells={k:{x:'opaque.npz' for x in ('raw','compiled')} for k in ('head','X','Y')},
        saved_inputs={'opaque.npz':m.io.sha(root/'opaque.npz')},packages={},gaussian=GAUSSIAN,compiler_phase=PHASE)
    m.io.save(root/'contract.json',c);return c


def test_passive_hashes_never_decode_and_input_mismatch_STOP(tmp_path,monkeypatch):
    c=passive_fixture(tmp_path);monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('numeric decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'opaque.npz').write_bytes(b'MISMATCH')
    with pytest.raises(ValueError,match='SOURCE_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('fault',['DF','JU','compiler','oldJU','PR_target','missing'])
def test_evidence_identity_missing_and_unqualified_STOP(tmp_path,fault):
    c,r=saved_fixture(tmp_path)
    if fault=='DF':r['physical']['budget_PASS']=False;r['new_result']['selected']['physical']=r['physical']
    if fault=='JU':r['physical']['candidate_sha256']='OTHER';r['new_result']['selected']['physical']=r['physical']
    if fault=='compiler':r['new_contract']['compiler']={}
    if fault=='oldJU':r['old_PR_contract']=deepcopy(r['old_PR_contract']);r['old_PR_contract']['exports']={'proposed':'OTHER'}
    if fault=='PR_target':r['new_result']['performance']['lanes']['ideal_exact_native']['rounds'][0]['target_id']='WRONG'
    if fault=='missing':(tmp_path/c['records']['ground_report']).unlink()
    for k,path in c['records'].items():
        if k=='new_SOURCE' or (fault=='missing' and k=='ground_report'):continue
        (tmp_path/path).write_text(json.dumps(r[k]))
    expected={'DF':'SAVED_DF','JU':'SAVED_DF','compiler':'ORIGINAL_SOURCE','oldJU':'OLD_SAME_JU','PR_target':'SAVED_PR_JU','missing':'ground_report'}
    with pytest.raises((ValueError,FileNotFoundError),match=expected[fault]):m.evidence(tmp_path,c)


def test_artificial_full512_single_kernel_end_to_end(tmp_path):
    from threadpoolctl import threadpool_limits
    c,r=saved_fixture(tmp_path);out=tmp_path/'out';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    assert m.evidence(tmp_path,c)['physical']['budget_PASS']
    with threadpool_limits(limits=1):result=m.execute(tmp_path,c,out,call)
    assert result['status']=='SAME_JU_STRONG_D_COMPLETE' and result['numerical']['status']=='NUMERICAL_PILOT_CA_PASS'
    assert result['ideal_exact_native']['energy']['same_accuracy']
    assert len(result['comparisons'])==6 and len(list((out/'circuits').glob('*.qpy')))==4
    assert call.counts['compile']==call.counts['kernel_build']==1
    assert call.counts['PF_round']==call.counts['round_accuracy']==call.counts['ideal_round_accuracy']==13
    assert sum(call.counts.values())<=156 and not call.stop.is_set()
    assert result['selected_export']==c['saved_inputs'][c['export']]
    assert result['actual_CA_achieved'] is None and result['PR_all_law_synthesis'] is None
    assert m.io.read(out/'XY.json')['status']=='PASS'
    for x in result['comparisons'].values():assert x['guaranteed_superiority'] is None


def test_ineligible_PF_skips_native_and_preserves_null(tmp_path,monkeypatch):
    from trottertracks import c1_molecule_extension as molecule
    c,r=saved_fixture(tmp_path);out=tmp_path/'out';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    def rounds(*a,**kw):return [call('PF_round',str(i),lambda i=i:dict(m=i,pf_operator_error=1.5)) for i in range(13)]
    monkeypatch.setattr(molecule,'PF_rounds',rounds)
    result=m.execute(tmp_path,c,out,call)
    assert result['status']=='INFEASIBLE_D_PF_NO_NATIVE' and result['metrics'] is None
    assert call.counts['kernel_build']==call.counts['compile']==0
    assert result['ideal_exact_native']['cost']['totals'] is None
    assert result['comparisons']['new_PR_ideal/new_D_ideal']['ratio'] is None


def test_correctness_partial_STOP_and_no_resume(tmp_path,monkeypatch):
    passive_fixture(tmp_path)
    def fail(root,c,out,call):call('evidence_load','synthetic correctness FAIL',lambda:(_ for _ in ()).throw(ValueError('CORRECTNESS_FAIL')))
    monkeypatch.setattr(m,'execute',fail)
    with pytest.raises(ValueError,match='CORRECTNESS_FAIL'):m.run(tmp_path,tmp_path/'contract.json')
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['failed'] and stop['scientific_units']==1 and stop['retry']==stop['resume']==0
    assert (out/'failure.json').exists()
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'contract.json')


def test_legacy_passive_preflight_on_artificial_source(tmp_path,monkeypatch):
    # Frozen 2026-10-26 molecular contracts are not re-executed against a later
    # checkout. Exercise that same byte-only preflight with its invented source.
    import test_research_c_c1_h4_ca_resources as old
    factory=old.artificial.__wrapped__(tmp_path,monkeypatch);path,_=factory()
    monkeypatch.setattr(np,'load',lambda *a,**kw:pytest.fail('numeric decode'))
    old.m.preflight(tmp_path,path)
    assert old.m.TARGET==old.m.reused.TARGET
    assert old.m.prior.CONDITIONS['candidate_failure']==old.m.TARGET['alpha_total']
    assert old.m.prior.CONDITIONS['axis_failure']==old.m.TARGET['alpha_axis']
    assert old.m.MAX_CALLS==119


@pytest.mark.parametrize('fault',['phase','frame','compile'])
def test_native_correctness_FAIL_stops_dependencies(tmp_path,monkeypatch,fault):
    from dataclasses import replace
    from trottertracks import c_pr_aware_df_stage_b as b,c1_h4_native_fair as fair
    c,_=saved_fixture(tmp_path);monkeypatch.setattr(m,'preflight',lambda *args:c)
    if fault=='compile':
        monkeypatch.setattr(fair,'compile_checked',lambda *args:(_ for _ in ()).throw(ValueError('INJECTED_COMPILE_FAIL')))
    else:
        original=b.prepare_arm
        def prepare(*args,**kwargs):
            arm=original(*args,**kwargs)
            if fault=='phase':model=replace(arm.model,constant=arm.model.constant+.2)
            else:model=replace(arm.model,blocks=arm.model.blocks[:1]+tuple(reversed(arm.model.blocks[1:])))
            return replace(arm,model=model)
        monkeypatch.setattr(b,'prepare_arm',prepare)
    with pytest.raises(ValueError):m.run(tmp_path,tmp_path/'unused.json')
    out=tmp_path/m.FRESH;stop=m.io.read(out/'STOP.json')
    assert stop['failed'] and stop['call_counts'].get('XY_gate',0)==0
    assert stop['call_counts'].get('metrics',0)==0 and not (out/'result.json').exists()


def test_nontrivial_rotations_general_signed_J_full512(tmp_path):
    from scipy.linalg import expm,eigh
    from threadpoolctl import threadpool_limits
    from trottertracks import c1_molecule_extension as molecule
    c,r=saved_fixture(tmp_path);rng=np.random.default_rng(902)
    J=rng.normal(size=(8,4,4))*1e-4;J=(J+J.transpose(0,2,1))/2
    X=rng.normal(size=(8,4,4))*.2;U=np.stack([expm(a-a.T) for a in X])
    np.savez(tmp_path/c['export'],J=J,U=U);h=m.io.sha(tmp_path/c['export']);c['saved_inputs'][c['export']]=h
    packet=dict(np.load(tmp_path/c['input']));ctx=molecule.SectorCAR(4);G=ctx.generators(**packet,J=J,U=U);H=sum(G);pair=eigh(H)
    np.savez(tmp_path/c['reference'],J=J,U=U,generators=np.asarray(G),H=H)
    np.savez(tmp_path/c['ground'],energy=pair[0][0],sector_state=pair[1][:,0])
    r['physical']['candidate_sha256']=h;r['new_result']['selected']['physical']=r['physical'];r['ground_report']['energy_Ha']=float(pair[0][0])
    for x in r['new_result']['performance']['lanes'].values():
        for row in x['rounds']:
            if 'target_id' in row:row['target_id']='integrated:'+h
    for k,path in c['records'].items():
        if k!='new_SOURCE':(tmp_path/path).write_text(json.dumps(r[k]))
    out=tmp_path/'nontrivial';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):result=m.execute(tmp_path,c,out,call)
    assert result['numerical']['status']=='NUMERICAL_PILOT_CA_PASS'
    assert call.counts['relative_gaussian']==16 and sum(call.counts.values())==156
    assert result['metrics']['rz_count']>0
    assert m.io.read(out/'segment.json')['compiled_reference']['max_entry']<1e-10
