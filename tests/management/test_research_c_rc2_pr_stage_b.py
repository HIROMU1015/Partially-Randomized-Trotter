"""Invented matrices and JSON only; no saved molecule payload is decoded."""
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_rc2_pr_stage_b as m


def invented(root):
    rng = np.random.default_rng(20261101)
    a = rng.normal(size=(8,4,4))*.004
    J = a+a.transpose(0,2,1)
    U = np.array([np.linalg.qr(rng.normal(size=(4,4)))[0] for _ in J])
    packet = dict(h=np.diag([-.2,-.1,.1,.2]),g=np.zeros((4,)*4),constant=np.array(.31))
    np.savez(root/'input.npz',**packet)
    result = dict(status='ONE_OR_BOTH_UNQUALIFIED',qualified=dict(rc2=False,pr=False),
        DF_budget=dict(rc2=True,pr=True),sectors={},reports={})
    exports = {}
    identities = {}
    for role,scale in [('rc2',1.),('pr',.8)]:
        (root/role).mkdir()
        exports[role] = role+'/final.npz'
        np.savez(root/exports[role],J=scale*J,U=U,x=np.zeros(128),jac=np.zeros(128))
        identities[role] = dict(role=role,export=exports[role],sha256=m.io.sha(root/exports[role]),
            fit_source=m.FIT_SOURCE,physical_input=m.io.sha(root/'input.npz'))
        mass = float(np.sum(abs(m.b.objective.coefficients(scale*J[2:],np))))
        result['sectors'][role] = dict(actual_rank=8,sector_error=1e-8,numerical_guard=1e-9,
            guard_policy=m.old.molecule.GUARD_POLICY)
        result['reports'][role] = dict(tensor=dict(actual_rank=8),optimizer=dict(success=False,status=1),
            diagnostic=dict(assessment=dict(convergence_criteria_met=False),
                components=dict(exact_tail_mass_Ha=mass)))
    records = dict(result='result.json',recovery='recovery.json',fit_STOP='old_STOP.json',
        fit_contract='old_contract.json',fit_postflight='old_postflight.json',repair_STOP='repair_STOP.json')
    m.save(root/records['result'],result)
    m.save(root/records['recovery'],dict(status='JSON_EXPORT_RECOVERED',new_fits=0,NPZ_decodes=0))
    m.save(root/records['fit_STOP'],dict(failed=True,scientific_units=25,retry=0,resume=0,mandatory_stop=True))
    m.save(root/records['fit_contract'],dict(conditions=dict(rank=8,ld=2),
        input_snapshot='input.npz',fresh_output='.'))
    m.save(root/records['fit_postflight'],dict(scientific_SOURCE=m.FIT_SOURCE,COMPLETE=25,
        call_FAIL=0,new_fits=2,raw_sha256={p:m.io.sha(root/p) for p in exports.values()}))
    m.save(root/records['repair_STOP'],dict(failed=False,scientific_units=1))
    m.save(root/'SOURCE.json',dict(files={}))
    names = ['input.npz',*exports.values(),*records.values()]
    return dict(scope=m.SCOPE,implementation=m.VERSION,base_commit=m.BASE,fit_source=m.FIT_SOURCE,
        policy=m.POLICY,allowances=m.ALLOWANCES,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        fresh_output=m.FRESH,roles=list(m.ROLES),api_roles=m.API_ROLES,retry=0,resume=0,
        input='input.npz',exports=exports,identities=identities,records=records,
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        saved_inputs={p:m.io.sha(root/p) for p in names},packages={})


def test_passive_import_no_numerical_runtime():
    code = "import sys;sys.path.insert(0,'src');from trottertracks import c_rc2_pr_stage_b as m;assert m.MAX_CALLS==143;assert not any(x in sys.modules for x in ('numpy','scipy','qiskit','ffsim','jax'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    assert m.POLICY['s']==1 and m.POLICY['r']==128 and m.POLICY['K']==0
    assert not set(m.LIMITS)&{'fit','optimizer','compile','sampling','shots','native_build'}


def test_passive_preflight_only_headers_and_hashes(tmp_path,monkeypatch):
    c = invented(tmp_path)
    m.save(tmp_path/'contract.json',c)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('passive decode'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c


@pytest.mark.parametrize('fault',['swap','coordinated_swap','duplicate','role','rank','hash','source','policy','old_STOP','budget'])
def test_passive_mismatches_stop(tmp_path,fault):
    c = invented(tmp_path)
    if fault=='swap':c['exports']['rc2'],c['exports']['pr']=c['exports']['pr'],c['exports']['rc2']
    if fault=='coordinated_swap':
        c['exports']['rc2'],c['exports']['pr']=c['exports']['pr'],c['exports']['rc2']
        for role in m.ROLES:
            c['identities'][role].update(export=c['exports'][role],sha256=c['saved_inputs'][c['exports'][role]])
    if fault=='duplicate':c['exports']['pr']=c['exports']['rc2']
    if fault=='role':c['identities']['rc2']['role']='baseline'
    if fault=='rank':
        name=c['exports']['rc2']
        np.savez(tmp_path/name,J=np.zeros((7,4,4)),U=np.zeros((7,4,4)),x=np.zeros(128),jac=np.zeros(128))
        c['saved_inputs'][name]=m.io.sha(tmp_path/name)
        c['identities']['rc2']['sha256']=c['saved_inputs'][name]
    if fault=='hash':c['saved_inputs'][c['exports']['pr']]='0'*64
    if fault=='source':c['source_manifest_sha256']='0'*64
    if fault=='policy':c['policy']=dict(c['policy'],r=64)
    if fault=='old_STOP':
        p=tmp_path/c['records']['fit_STOP'];d=m.io.read(p);d['failed']=False;p.write_text(json.dumps(d))
        c['saved_inputs'][c['records']['fit_STOP']]=m.io.sha(p)
    if fault=='budget':
        p=tmp_path/c['records']['result'];d=m.io.read(p);d['DF_budget']['rc2']=False;p.write_text(json.dumps(d))
        c['saved_inputs'][c['records']['result']]=m.io.sha(p)
    m.save(tmp_path/'contract.json',c)
    with pytest.raises(ValueError):m.preflight(tmp_path,tmp_path/'contract.json')


def test_actual_adapter_role_kappa_scalar_frame_law_reference(tmp_path):
    c = invented(tmp_path)
    p = m.old.load(tmp_path/'input.npz',('h','g','constant'))
    # Nonzero invented g makes the correction and its sign observable.
    p['g'][0,1,1,0]=.024
    data = m.old.load(tmp_path/c['exports']['rc2'],('J','U'))
    arm = m.prepare_candidate(p,data,identity=c['identities']['rc2'])
    assert arm.role=='rc2' and 'baseline' not in arm.model.label
    assert arm.model.source_identity==c['saved_inputs'][c['exports']['rc2']]
    assert arm.model.input_identity==c['saved_inputs']['input.npz']
    assert [x.block_id for x in arm.model.blocks]==['one_body']+[f'F{i}' for i in range(8)]
    np.testing.assert_allclose(arm.general.spatial_one_body_tensor,m.b.stage1.corrected_kappa(p['h'],p['g']))
    ctx = m.old.molecule.SectorCAR(4)
    G = ctx.generators(**p,**data)
    for i,g in enumerate(G):
        np.testing.assert_allclose(m.b.candidate_factor(ctx,arm,i),g,atol=1e-12)
    tail,config,dist=m.b.configure(arm,Fraction(1,10),128,0)
    assert m.b.tail_gate(ctx,arm,G,tail)['status']=='PASS'
    assert m.distribution_gate(tail,config,dist)['status']=='PASS'
    assert arm.plan.extracted_identity!=0
    assert any(a.coefficient<0 for a in arm.plan.atoms)
    kernel=m.b.kernels(arm,G,Fraction(1,10),config)
    exact=expm(-1j*.1*sum(G));psi=np.eye(ctx.d)[:,0]
    row=m.round_reference(kernel,sum(G),psi,0,exact)
    direct=m.b.reference.reference_round(kernel,(sum(G),Fraction(1,10)),repetitions=1,state=psi,exact_operator=exact)
    for key in ('pf_operator_error','rte_corrected_operator_error','attenuation'):
        assert row[key]==direct[key]
    np.testing.assert_allclose(row['raw_signal_parts'],[direct['raw_signal'].real,direct['raw_signal'].imag])
    with pytest.raises(ValueError):m.distribution_gate(tail,replace(config,rte_steps=64),dist)


@pytest.mark.parametrize('fault',[None,'representation','mass','alias','law','nan'])
def test_artificial_pipeline_143_units_preservation_and_STOP(tmp_path,monkeypatch,fault):
    c = invented(tmp_path)
    m.save(tmp_path/'contract.json',c)
    protected={p:m.io.sha(tmp_path/p) for p in c['saved_inputs']}
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'SYNTHETIC_SOURCE\n')
    # Guard forbidden dependencies; actual invented CAR/eigensolve/PF/RTE remain live.
    monkeypatch.setattr(m.b,'build_kernel',lambda *a,**k:pytest.fail('native build forbidden'))
    if fault=='representation':monkeypatch.setattr(m.b,'candidate_factor',lambda ctx,a,i:np.zeros((ctx.d,ctx.d)))
    if fault=='mass':monkeypatch.setattr(m,'mass_gate',lambda *a:(_ for _ in ()).throw(ValueError('MASS')))
    if fault=='alias':monkeypatch.setattr(m.old,'ground_gate',lambda *a:(_ for _ in ()).throw(ValueError('ALIAS')))
    if fault=='law':monkeypatch.setattr(m,'distribution_gate',lambda *a:(_ for _ in ()).throw(ValueError('LAW')))
    if fault=='nan':monkeypatch.setattr(m,'round_reference',lambda *a:dict(attenuation=float('nan'),pf_operator_error=0.,rte_corrected_operator_error=0.))
    out=tmp_path/m.FRESH
    if fault:
        with pytest.raises((ValueError,m.old.integrated.Stopped)):
            m.run(tmp_path,tmp_path/'contract.json',out)
        stop=m.io.read(out/'STOP.json')
        assert stop['failed'] and stop['scientific_units']<143
        assert (out/'failure.json').exists() and not (out/'result.json').exists()
        assert m.io.read(out/'failure.json')['first_failure'] is not None
        if fault!='nan':assert not list(out.glob('*_m[0-9][0-9].json'))
    else:
        result=m.run(tmp_path,tmp_path/'contract.json',out)
        stop=m.io.read(out/'STOP.json')
        assert not stop['failed'] and stop['scientific_units']==143
        assert stop['call_counts']==m.LIMITS
        for role,a in result['arms'].items():
            assert a['identity']==c['identities'][role]
            assert len(a['rounds'])==13 and [r['m'] for r in a['rounds']]==list(range(13))
            assert a['classification']['status']=='POTENTIALLY_FEASIBLE'
            assert a['DF_budget']=='PASS' and a['optimizer_qualification']=='UNQUALIFIED'
            assert a['synthesis_bound'] is a['implemented_law_bound'] is a['CA_achieved'] is None
            assert a['shots'] is a['total_cost'] is None
            assert all(x['physical_time_Ha_inverse']==.1*(1<<x['m']) for x in a['rounds'])
        journal=[json.loads(x) for x in (out/'science_calls.jsonl').read_text().splitlines()]
        assert len([x for x in journal if x.get('event')=='COMPLETE'])==143
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert protected=={p:m.io.sha(tmp_path/p) for p in protected}
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'contract.json',out)


@pytest.mark.parametrize('status',['INFEASIBLE','UNRESOLVED'])
def test_ineligible_or_missing_rounds_continue_other_registered_cells(tmp_path,monkeypatch,status):
    c=invented(tmp_path);m.save(tmp_path/'contract.json',c)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'SYNTHETIC_SOURCE\n')
    original=m.round_reference
    def row(kernel,H,psi,index,exact):
        r=original(kernel,H,psi,index,exact)
        if index==7:r['pf_operator_error']=1. if status=='INFEASIBLE' else None
        return r
    monkeypatch.setattr(m,'round_reference',row)
    result=m.run(tmp_path,tmp_path/'contract.json',tmp_path/m.FRESH)
    assert all(a['classification']['status']==status for a in result['arms'].values())
    assert all(len(a['rounds'])==13 for a in result['arms'].values())
    assert m.io.read(tmp_path/m.FRESH/'STOP.json')['scientific_units']==143


def test_underflow_retains_PF_measurement_unknown_RTE(monkeypatch):
    monkeypatch.setattr(m.b.reference,'reference_round',lambda *a,**k:dict(
        attenuation=0.,pf_operator_error=None,rte_corrected_operator_error=None))
    row=m.round_reference(dict(pf=np.eye(2)),np.zeros((2,2)),np.array([1.,0.]),0,np.diag([1.,-1.]))
    assert row['pf_operator_error']==2. and row['rte_corrected_operator_error'] is None
    assert row['raw_signal_parts'] is None


@pytest.mark.parametrize('key',['J','U'])
def test_rank_shape_rejected_before_scientific_preparation(tmp_path,key):
    c=invented(tmp_path);data=m.old.load(tmp_path/c['exports']['rc2'],('J','U'))
    data[key]=data[key][:7]
    with pytest.raises(ValueError,match='SHAPE'):
        m.prepare_candidate({},data,identity=c['identities']['rc2'])
