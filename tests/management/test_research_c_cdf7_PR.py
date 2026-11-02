"""Invented rank7 J/U, true existing CAR/PF/RTE/RPE;never saved H4 arrays."""
from pathlib import Path
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace
import json,sys,subprocess
import numpy as np
import pytest
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_cdf7_pr as m


def fixture(root):
    rng=np.random.default_rng(301071)
    x=rng.normal(size=(7,4,4))*1e-5;J=x+x.transpose(0,2,1)
    U=[]
    for _ in J:
        a=rng.normal(size=(4,4));U.append(expm(.05*(a-a.T)))
    U=np.array(U);U[0,:,0]*=-1
    packet=dict(h=np.diag([.1,.11,.12,.13]),g=np.zeros((4,)*4),constant=np.array(.31))
    np.savez(root/'input.npz',**packet);np.savez(root/'JU.npz',J=J,U=U)
    mass=float(np.sum(abs(m.fit.objective.coefficients(J[2:],np))))
    sector=dict(actual_rank=7,sector_error=1e-8,numerical_guard=1e-9,guard_policy=m.fit.molecule.GUARD_POLICY)
    stop=dict(failed=False,scientific_units=25,retry=0,resume=0,mandatory_stop=True)
    plans=[dict(m=i,shots=1024) for i in range(13)]
    costs={lane:dict(conditional_totals={k:1e12 for k in m.old.old.METRICS})
        for lane in ('ideal_exact_native','assumed_raw_signal_budget')}
    policy=dict(m.POLICY,rank=8,frame_order=list(range(8)))
    records=dict(fit_result=dict(status='CDF7_DF_ELIGIBLE',optimizer_success=True,DF_budget_status='PASS',
            sector=dict(final=sector),tail_auxiliary=dict(lambda_R=mass)),
        fit_STOP=stop,fit_SOURCE=dict(execution_commit=m.FIT_SOURCE),
        fit_contract=dict(conditions=m.fit.CONDITIONS,original_input_sha256='ARTIFICIAL_INPUT'),
        fit_sector=sector,fit_optimizer=dict(success=True),
        physical_ground=dict(status='PASS',alias_margin=1.),
        physical_contract=dict(original_input=dict(sha256='ARTIFICIAL_INPUT')),
        shell_reports={label:dict(metrics={k:v for k in m.old.old.METRICS}) for label,v in [('head',2),('X',3),('Y',4)]},
        M4=dict(status='M4_R128_SHOTS_BREAKEVEN_COMPLETE',B={lane:dict(rounds=plans) for lane in costs}),
        M4_native=dict(status='NATIVE128_FINITE_COHORT_COMPLETE',lanes=costs),
        M4_STOP=dict(stop,scientific_units=332),M4_SOURCE=dict(execution_commit='dcf7ab6da2ea540ae54357b82ac3ea7a02df99ad'),
        M4_contract=dict(original_input_sha256='ARTIFICIAL_INPUT',policy=policy,target=m.TARGET),
        CDF8_native=dict(status='GAMMA0_FIXED32_CONDITIONAL_PILOT_COMPLETE',lanes=costs),
        CDF8_STOP=dict(stop,scientific_units=493),CDF8_SOURCE=dict(execution_commit='30436b142d7fa79c841b5977d1b4c0fbfbd1bb4c'),
        CDF8_contract=dict(original_input_sha256='ARTIFICIAL_INPUT',policy=policy,target=m.TARGET))
    paths={}
    for name,value in records.items():
        paths[name]=name+'.json';m.io.save(root/paths[name],value)
    c=dict(input='input.npz',export='JU.npz',export_sha256='ARTIFICIAL_JU',original_input_sha256='ARTIFICIAL_INPUT',
        records=paths)
    return c,records,packet,dict(J=J,U=U)


def forbid_native(monkeypatch):
    from trottertracks import c_pr_aware_df_stage_b as b
    def fail(*a,**kw):raise AssertionError('NO NATIVE OR SAMPLING IN119')
    monkeypatch.setattr(b,'build_kernel',fail)
    monkeypatch.setattr(b.connection,'sample_occurrence',fail)


def test_true_artificial119_and_component_consumer(tmp_path,monkeypatch):
    c,records,packet,d=fixture(tmp_path);forbid_native(monkeypatch)
    out=tmp_path/'run';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    result=m.execute(tmp_path,c,out,call)
    assert dict(call.counts)==m.LIMITS and sum(call.counts.values())==119
    assert result['status']=='CDF7_PR_REFERENCE_RPE_PRICE_COMPLETE'
    assert len(result['reference']['rounds'])==13
    assert result['reference']['ground']['status']=='PASS'
    assert len(m.io.read(out/'reference/factors.json'))==8
    assert result['accuracy']['evidence']['shots']['total_shots'] is None
    assert result['price']['affine']['evidence']['W'] is None
    for lane in ('ideal_exact_native','assumed_raw_signal_budget'):
        assert result['accuracy'][lane]['energy']['same_accuracy']
        assert set(result['price']['comparisons'][lane+':CDF8_gamma0']['thresholds'])==set(m.old.old.METRICS)
    assert result['native_recommendation']['status']=='ELIGIBLE_FOR_SEPARATE_NATIVE_PROPOSAL'
    assert result['actual_CA_achieved'] is result['native_price'] is result['total_resources'] is None
    assert not (out/'native').exists() and not list(out.rglob('*.qpy'))
    # Exercise the real downstream component-record consumer at sector36 with
    # independent raw-CAR atoms. This is not a full-Fock/native science claim.
    from trottertracks import c_pr_aware_df_stage_b as b,c_pr_aware_df_stage_b2 as b2
    arm=b.prepare_arm(**packet,**d,role='baseline',source_identity='ARTIFICIAL_JU',input_identity='ARTIFICIAL_INPUT',ld=2)
    ctx=m.fit.molecule.SectorCAR(4);G=ctx.generators(**packet,**d)
    tail,_,_=b.configure(arm,Fraction(1,10),128,0)
    external=dict(left=np.eye(36),atoms=b.atom_operators(ctx,arm),generators=G)
    info=m.io.read(out/'reference/tail.json')
    report=b2.law_gate(arm,tail,external,info)
    assert report['max_entry']<1e-10
    bad=deepcopy(info);bad['components'][0]['probability']+=.01
    with pytest.raises(ValueError,match='LAW_OR_IDENTITY'):b2.law_gate(arm,tail,external,bad)
    assert len(info['components'])==len(arm.plan.atoms)


@pytest.mark.parametrize('fault',['infeasible','correctness','unqualified'])
def test_limits_STOP_and_unknowns(tmp_path,monkeypatch,fault):
    c,_,_,_=fixture(tmp_path);forbid_native(monkeypatch)
    out=tmp_path/'run';out.mkdir();call=m.integrated.Calls(out,m.LIMITS,m.MAX_CALLS)
    from trottertracks import c1_stage3_feasibility as feas,c_pr_aware_df_stage_c as rpe
    if fault=='infeasible':
        monkeypatch.setattr(feas,'classify_round',lambda *a:dict(status='INFEASIBLE',reason='ARTIFICIAL_PHASE_ROOM'))
    elif fault=='correctness':
        def fail(*a):raise ValueError('ARTIFICIAL_FACTOR_MISMATCH')
        monkeypatch.setattr(m.fit.molecule.SectorCAR,'candidate',fail)
    else:
        qualify=rpe.qualification
        monkeypatch.setattr(rpe,'qualification',lambda packet,lane:qualify(packet,'evidence'))
    if fault=='correctness':
        with pytest.raises(ValueError,match='ARTIFICIAL_FACTOR'):m.execute(tmp_path,c,out,call)
        assert call.stop.is_set() and call.counts.get('own_ground',0)==0
        assert call.counts.get('round_reference',0)==0
    else:
        result=m.execute(tmp_path,c,out,call)
        assert not call.stop.is_set()
        assert sum(call.counts.values())==(63 if fault=='infeasible' else 119)
        if fault=='infeasible':
            assert result['accuracy']=={} and result['price'] is None
            assert call.counts.get('signal_qualification',0)==0
        else:
            assert result['price']['affine']['ideal_exact_native']['W'] is None
            assert result['native_recommendation']['status']=='NOT_RECOMMENDED_UNDER_REGISTERED_CONDITIONS'
            assert all(x['thresholds'] is None for x in result['price']['comparisons'].values())


@pytest.mark.parametrize('fault',['fit_source','budget','rank','comparator_policy','input'])
def test_evidence_rejects_unbound_scientific_context(tmp_path,fault):
    c,records,_,_=fixture(tmp_path)
    if fault=='fit_source':name='fit_SOURCE';records[name]['execution_commit']='wrong'
    elif fault=='budget':
        name='fit_sector';records[name]=dict(records[name],sector_error=.01)
    elif fault=='rank':name='fit_sector';records[name]=dict(records[name],actual_rank=8)
    elif fault=='comparator_policy':name='CDF8_contract';records[name]['policy']=dict(records[name]['policy'],r=64)
    else:name='physical_contract';records[name]['original_input']=dict(sha256='wrong')
    (tmp_path/c['records'][name]).write_text(json.dumps(records[name]))
    with pytest.raises(ValueError):m.evidence(tmp_path,c)


def test_passive_import_and_fixed_counts():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_cdf7_pr as m;assert m.MAX_CALLS==119;assert sum(m.REFERENCE.values())==62;assert not any(x in sys.modules for x in ['numpy','scipy','qiskit','ffsim','jax'])"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_byte_preflight_source_input_and_fresh(tmp_path,monkeypatch):
    c,_,_,_=fixture(tmp_path)
    # Opaque payloads confirm no numerical decode is needed for preflight.
    for name in ('input.npz','JU.npz'):(tmp_path/name).write_bytes(b'OPAQUE')
    (tmp_path/'source.py').write_text('# synthetic')
    m.io.save(tmp_path/'manifest.json',dict(files={'source.py':m.io.sha(tmp_path/'source.py')}))
    monkeypatch.setattr(m,'EXPORT','JU.npz');monkeypatch.setattr(m,'EXPORT_SHA',m.io.sha(tmp_path/'JU.npz'))
    monkeypatch.setattr(m,'FRESH','future_run');monkeypatch.setattr(m.fit.io,'P_H4_SHA256','ARTIFICIAL_INPUT')
    from trottertracks.c_pr_aware_df_stage_c import SCENARIOS
    c.update(scope=m.SCOPE,implementation=m.VERSION,policy=m.POLICY,target=m.TARGET,
        allowances=m.ALLOWANCES,stage_call_limits=m.STAGES,call_limits=m.LIMITS,maximum_logical_units=119,
        export_sha256=m.EXPORT_SHA,fit_SOURCE=m.FIT_SOURCE,fresh_output=m.FRESH,
        comparators=list(m.COMPARATORS),scenarios=list(SCENARIOS),new_fit=0,optimizer=0,retry=0,resume=0,
        DF_error_recalculation=False,native_build_compile=0,sampling=0,actual_CDF7_native_price=None,cost_support=None,
        source_manifest='manifest.json',source_manifest_sha256=m.io.sha(tmp_path/'manifest.json'),packages={})
    c['saved_inputs']={p:m.io.sha(tmp_path/p) for p in [c['input'],c['export'],*c['records'].values()]}
    m.io.save(tmp_path/'contract.json',c)
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c
    (tmp_path/'source.py').write_text('# changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')
    (tmp_path/'source.py').write_text('# synthetic');(tmp_path/'JU.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,tmp_path/'contract.json')
    (tmp_path/'JU.npz').write_bytes(b'OPAQUE');(tmp_path/'future_run').mkdir()
    with pytest.raises(ValueError,match='FRESH'):m.preflight(tmp_path,tmp_path/'contract.json')
