"""M2 artificial arrays and mocked scalar evidence; no saved molecule decode."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_m2 as m


def test_passive_import_and_calls():
    subprocess.run([sys.executable,'-B','-c',"import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_m2 as m;assert m.MAX_CALLS==253;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','ffsim'))"],cwd=ROOT,check=True)


def arrays():
    p=dict(h=np.diag([-.4,-.2,.3,.5]),g=np.zeros((4,)*4),constant=np.array(.31))
    J=np.stack([np.eye(4)*(.001+.0001*i) for i in range(8)])
    U=np.stack([np.eye(4) for i in range(8)])
    return p,dict(J=J,U=U)


@pytest.mark.parametrize('fault',['shape','dtype','nonfinite','symmetry','rotation'])
def test_invalid_arrays_fail(fault):
    p,d=arrays()
    if fault=='shape':d['J']=d['J'][:7]
    if fault=='dtype':d['U']=d['U'].astype(np.float32)
    if fault=='nonfinite':p['g'][0,0,0,0]=np.nan
    if fault=='symmetry':d['J'][0,0,1]=.1
    if fault=='rotation':d['U'][0,0,0]=2.
    with pytest.raises(ValueError):m.validate_arrays(p,d,'baseline')


def scalar_cost(value):
    return dict(totals={k:float(value) for k in m.accuracy.METRICS},cost_scope='serial_native_kernel_composition_without_state_preparation')


def test_point_comparison_limits_and_ineligible():
    x=m.numerical_compare(scalar_cost(12),scalar_cost(10),'x',qualification='synthetic')
    assert x['lower_RZ']=='right' and x['ratio']['rz_count']==1.2
    assert x['guaranteed_superiority'] is None
    assert m.numerical_compare(dict(totals=None),scalar_cost(10),'x',qualification='synthetic')['ratio'] is None
    a=scalar_cost(1);a['cost_scope']='wrong'
    with pytest.raises(ValueError,match='COST_SCOPE'):m.numerical_compare(a,scalar_cost(2),'x',qualification='x')


def synthetic_fixture(root):
    from qiskit import qpy
    from scipy.linalg import eigh
    from trottertracks import c_pr_aware_df_stage_b as b
    from trottertracks import c_pr_aware_df_stage_b2 as b2
    from trottertracks import c1_molecule_extension as mol
    from trottertracks.pr_dictionary_validation.f2_execution import circuit_metrics
    root.mkdir(exist_ok=True)
    p,d=arrays();c=dict(input='input.npz',exports={a:a+'.npz' for a in m.ARMS},
        original_input_sha256='SYNTHETIC_INPUT',references={},grounds={},shells={},saved_inputs={})
    np.savez(root/'input.npz',**p)
    saved=dict(shell_reports={},B1=dict(arms={}),C1=dict(assumed_scenarios={lane:{} for lane in ('ideal_exact_native','assumed_raw_signal_budget')}))
    for arm in m.ARMS:
        dd=deepcopy(d)
        if arm=='proposed':dd['J']*=1.1
        np.savez(root/c['exports'][arm],**dd);c['saved_inputs'][c['exports'][arm]]=m.io.sha(root/c['exports'][arm])
        ctx=mol.SectorCAR(4);G=ctx.generators(**p,**dd);H=sum(G);vals,vecs=eigh(H)
        indices=b.stage1.bit_reverse_indices(8)[ctx.sector.basis_indices]
        c['references'][arm]=arm+'_reference.npz';np.savez(root/c['references'][arm],generators=np.array(G),H=H,full_input_indices=indices)
        c['grounds'][arm]=arm+'_ground.npz';np.savez(root/c['grounds'][arm],energy=vals[0],sector_state=vecs[:,0])
        saved[arm+'_ground']=dict(status='PASS',energy_Ha=float(vals[0]),alias_margin=1.6)
        saved['B1']['arms'][arm]=dict(saved_DF_bound_Ha=1e-8)
        for lane in saved['C1']['assumed_scenarios']:
            saved['C1']['assumed_scenarios'][lane][arm]=dict(resources=scalar_cost(1e12))
    for label in ('head','X','Y'):
        q=b2.shell_build(8,label);c['shells'][label]={}
        for kind in ('raw','compiled'):
            name=label+'_'+kind+'.qpy';c['shells'][label][kind]=name
            with (root/name).open('xb') as f:qpy.dump(q,f)
        saved['shell_reports'][label]=dict(metrics=circuit_metrics(q))
    saved['external']=dict(candidates={'B3_XDF_spin_relative':dict(cost=scalar_cost(1e11))})
    return c,saved


def test_artificial_full512_end_to_end_native_PF_accuracy_cost(tmp_path,monkeypatch):
    from threadpoolctl import threadpool_limits
    from trottertracks import c1_h4_native_fair as fair
    root=tmp_path/'invented';c,saved=synthetic_fixture(root);out=root/'out';out.mkdir()
    monkeypatch.setattr(m,'evidence',lambda *args:saved)
    # Compile a genuine artificial circuit with the existing P0, no mock gates.
    call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    with threadpool_limits(limits=1):r=m.execute(root,c,out,call)
    assert r['status']=='M2_SAME_JU_DETERMINISTIC_COMPLETE'
    assert len(r['candidates'])==2 and len(r['comparisons'])==7
    assert len(list((out/'circuits').glob('*.qpy')))==8
    assert sum(call.counts.values())<=253
    assert call.counts['PF_round']==call.counts['round_accuracy']==26
    assert call.counts['compile']==2 and call.counts['operator_decode']==4
    for x in r['candidates'].values():
        assert x['status']=='NUMERICAL_PILOT_CA_PASS'
        assert x['segment']['input_dimension']==x['segment']['output_dimension']==512
        assert x['segment']['global_phase_fit'] is False
        assert x['segment']['compiled_reference']['max_entry']<1e-10
        assert len(x['rounds'])==13 and x['cost']['totals']['rz_count']>0
        assert x['CA_achieved'] is None and x['XY']['status']=='PASS'
        assert x['per_shot_costs'][0]['kernel_occurrences']==1
        assert x['per_shot_costs'][-1]['kernel_occurrences']==4096


@pytest.mark.parametrize('state',['ineligible','unresolved'])
def test_accuracy_nonpass_preserves_null_and_does_not_price(tmp_path,monkeypatch,state):
    from trottertracks import c1_h4_full_synthesis as s
    from trottertracks import c1_h6_ca_resources as reused
    packet=dict(id='D_baseline',arm='baseline',s=1,aliases=['D_baseline'],df_error_with_guard=1e-8,
        state_model='own_DF_ground_state_oracle',alias_verified=True)
    status='UNRESOLVED_SAVED_SIGNAL_BOUND' if state=='unresolved' else 'INFEASIBLE_PHASE'
    monkeypatch.setattr(s,'propagate',lambda *args,**kw:dict(m=kw['m']))
    monkeypatch.setattr(reused,'assess_saved_round',lambda t,p,r:dict(m=r['m'],status=status,shots=None))
    call=m.integrated.Calls(tmp_path,m.LIMITS,m.MAX_CALLS)
    result=m.evaluate_accuracy(packet,[dict(m=i,pf_operator_error=1.) for i in range(13)],{},None,tmp_path,call)
    assert result['cost']['totals'] is None and call.counts['round_serial_cost']==0
    assert result['status']==('UNRESOLVED' if state=='unresolved' else 'INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION')


def passive_fixture(root):
    (root/'opaque.npz').write_bytes(b'synthetic closed bytes')
    m.io.save(root/'source.json',dict(files={}))
    c=dict(scope=m.SCOPE,implementation=m.VERSION,base_commit=m.BASE,conditions=m.CONDITIONS,
        target=m.TARGET,allowances=m.ALLOWANCES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        fresh_output=m.FRESH,retry=0,resume=0,source_manifest='source.json',
        source_manifest_sha256=m.io.sha(root/'source.json'),saved_inputs={'opaque.npz':m.io.sha(root/'opaque.npz')},packages={})
    m.io.save(root/'contract.json',c);return c


def test_passive_source_input_mismatch_and_no_numeric_decode(tmp_path):
    c=passive_fixture(tmp_path)
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'opaque.npz').write_bytes(b'wrong')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')


def test_failure_partial_STOP_no_resume(tmp_path,monkeypatch):
    passive_fixture(tmp_path)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *args,**kwargs:'SYNTHETIC_COMMIT\n')
    def fail(root,c,out,call):
        call('evidence_load','artificial failure',lambda:(_ for _ in ()).throw(ValueError('ARTIFICIAL_CORRECTNESS_FAIL')))
    monkeypatch.setattr(m,'execute',fail)
    with pytest.raises(ValueError,match='ARTIFICIAL'):m.run(tmp_path,tmp_path/'contract.json')
    out=tmp_path/m.FRESH;s=m.io.read(out/'STOP.json')
    assert s['failed'] and s['scientific_units']==1 and s['retry']==s['resume']==0
    assert (out/'failure.json').exists() and not(out/'result.json').exists()
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'contract.json')


def test_existing_fixed_N_regression_with_synthetic_commit(tmp_path,monkeypatch):
    # The historical test assumes its invented temp directory is a Git repo.
    # Keep old SOURCE unchanged and supply only that missing synthetic identity.
    import test_research_c_c1_h4_ca_resources as old
    import test_research_c_c1_h4_strong_ca_resources as strong
    legacy=old.artificial.__wrapped__(tmp_path,monkeypatch)
    artificial=strong.artificial.__wrapped__(tmp_path,monkeypatch,legacy)
    monkeypatch.setattr(strong.m.prior,'source_commit',lambda root:'SYNTHETIC_COMMIT')
    strong.test_fixed_N_failure_is_unresolved_without_cost(tmp_path,artificial)


@pytest.mark.parametrize('fault',[None,'source','arm','reference','compiler','physical','DF','PR_target'])
def test_saved_evidence_binding_gate(tmp_path,fault):
    c=dict(input='input.npz',exports={a:a+'.npz' for a in m.ARMS},references={a:a+'_reference.npz' for a in m.ARMS},
        grounds={a:a+'_ground.npz' for a in m.ARMS},original_input_sha256='ORIGINAL',
        compiler={'profile':'invented'},gaussian='artificial_G',compiler_phase='artificial_P',saved_inputs={a+'.npz':a+'_hash' for a in m.ARMS})
    b=dict(A=dict(status='STAGE_A_PROXY_AND_DF_PASS'),B1=dict(status='B1_MECHANISM_EVALUATION_COMPLETE',arms={a:dict(saved_DF_bound_Ha=1e-8) for a in m.ARMS}),
        B2=dict(status='B2_FINITE_BIT_NATIVE_CALIBRATION_COMPLETE'),M1=dict(status='M1_DESCRIPTIVE_SAVED_ANALYSIS_COMPLETE'),
        C1=dict(status='C1_SAVED_SCALAR_SCENARIOS_COMPLETE',assumed_scenarios={lane:{a:dict(target_id=a+':'+a+'_hash',qualification=dict(kind='ASSUMED_SCENARIO')) for a in m.ARMS} for lane in ('ideal_exact_native','assumed_raw_signal_budget')}),
        B2_contract=dict(final_exports=c['exports'],stage_A_input=c['input'],B1_references=c['references'],B1_grounds=c['grounds'],original_input=dict(sha256='ORIGINAL'),compiler=c['compiler'],gaussian=c['gaussian'],compiler_phase=c['compiler_phase'],saved_inputs=c['saved_inputs']),
        physical_ground=dict(status='PASS',alias_margin=1.6),external=dict(status='H4_STRONG_BASELINE_S1_CA_RESOURCES_COMPLETE'),
        external_STOP=dict(failed=False,scientific_units=86),external_SOURCE=dict(execution_commit='1aeedd96178a3f50a15bc2701c3c332f4f391b64'))
    for a in m.ARMS:b[a+'_ground']=dict(status='PASS')
    for k,units,commit in [('B1',141,'7da0fdb3ff7c2e5e22c7d8658f6980077993c5d9'),('B2',652,'a1cf0046967c206f414dbad568a75884ee076ebf'),('C1',164,'b1c735da394d4b0a444cbc618ead9781ef4d9134')]:
        b[k+'_STOP']=dict(failed=False,mandatory_stop=True,scientific_units=units)
        b[k+'_SOURCE']=dict(execution_commit=commit)
    if fault=='source':b['B1_SOURCE']['execution_commit']='wrong'
    if fault=='arm':b['B2_contract']['final_exports']=dict(baseline='proposed.npz',proposed='baseline.npz')
    if fault=='reference':b['B2_contract']['B1_references']={}
    if fault=='compiler':b['B2_contract']['compiler']={}
    if fault=='physical':b['physical_ground']['status']='FAIL'
    if fault=='DF':b['B1']['arms']['baseline']['saved_DF_bound_Ha']=.01
    if fault=='PR_target':b['C1']['assumed_scenarios']['ideal_exact_native']['baseline']['target_id']='wrong'
    c['records']={k:k+'.json' for k in b}
    for k,p in c['records'].items():m.io.save(tmp_path/p,b[k])
    if fault:
        with pytest.raises(ValueError):m.evidence(tmp_path,c)
    else:assert m.evidence(tmp_path,c)['physical_ground']['status']=='PASS'

