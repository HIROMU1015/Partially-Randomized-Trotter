"""Invented scalar evidence only. No stored H4 shots/W/energy evaluation."""
from copy import deepcopy
from fractions import Fraction
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_rc2_pr_conditional_shots as m


def packet(role='rc2',A=.8,pf=.002,rte=.003,df=1e-7):
    return dict(arm=role,target_id=role+':invented',identity=dict(role=role),df_bound_Ha=df,
        optimizer_qualification='UNQUALIFIED',error_evidence=None,
        rounds=[dict(m=i,reference=dict(attenuation=A,pf_operator_error=pf,rte_corrected_operator_error=rte)) for i in range(13)])


def invented(root,A=.8,pf=.002,rte=.003):
    identities={}
    for role in m.ROLES:
        p=root/(role+'.npz');p.write_bytes(('invented opaque provenance '+role).encode())
        identities[role]=dict(role=role,export=p.name,sha256=m.io.sha(p),fit_source=m.stage_b.FIT_SOURCE,physical_input='invented')
    result=dict(status='SAVED_ENDPOINT_STAGE_B_COMPLETE',arms={},matched_rounds=[dict(m=i) for i in range(13)])
    for role in m.ROLES:
        rows=[]
        for row in packet(role,A=A,pf=pf,rte=rte)['rounds']:
            row.update(arm=role,coherent_repetitions=1<<row['m'],physical_time_Ha_inverse=.1*(1<<row['m']),
                feasibility=dict(status='POTENTIALLY_FEASIBLE'),synthesis_error=None,implemented_law_error=None)
            rows.append(row);result['matched_rounds'][row['m']][role]=row
        result['arms'][role]=dict(arm=role,identity=identities[role],DF_budget='PASS',optimizer_qualification='UNQUALIFIED',
            optimizer=dict(success=False),classification=dict(status='POTENTIALLY_FEASIBLE'),rounds=rows,saved_DF_bound_Ha=1e-7)
    records=dict(result='B_result.json',STOP='B_STOP.json',SOURCE='B_SOURCE.json',contract='B_contract.json',
        manifest='B_manifest.json',rc2_ground='rc2_ground.json',pr_ground='pr_ground.json')
    values=dict(result=result,STOP=dict(failed=False,scientific_units=143,call_counts=m.stage_b.LIMITS,retry=0,resume=0,mandatory_stop=True),
        SOURCE=dict(execution_commit=m.STAGE_B_SOURCE,files={}),contract=dict(policy=m.stage_b.POLICY,fit_source=m.stage_b.FIT_SOURCE,identities=identities),
        manifest=dict(files={}),rc2_ground=dict(status='PASS',alias_margin=1.),pr_ground=dict(status='PASS',alias_margin=1.))
    for key,name in records.items():m.save(root/name,values[key])
    m.save(root/'SOURCE.json',dict(files={}))
    c=dict(scope=m.SCOPE,implementation=m.VERSION,base_commit=m.BASE,stage_B_SOURCE=m.STAGE_B_SOURCE,
        target=m.TARGET,scenarios=m.SCENARIOS,roles=list(m.ROLES),lanes=list(m.LANES),call_limits=m.LIMITS,
        maximum_logical_units=m.MAX_CALLS,fresh_output=m.FRESH,retry=0,resume=0,identities=identities,
        records=records,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        saved_inputs={p:m.io.sha(root/p) for p in [*records.values(),'rc2.npz','pr.npz']},packages={})
    m.save(root/'contract.json',c)
    return c


def test_import_passive_and_fixed_scope():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_rc2_pr_conditional_shots as m;assert m.MAX_CALLS==105;assert not any(n in sys.modules for n in ('numpy','scipy','qiskit','ffsim','jax'))"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)
    assert m.LIMITS['round_accuracy']==78 and m.LANES==('evidence','ideal_exact_native','assumed_raw_signal_budget')
    assert not set(m.LIMITS)&{'fit','ground','PF','RTE','compile','sampling','cost'}
    assert all(not any('proxy' in a for a in s['assumptions']) for s in m.SCENARIOS)


def test_preflight_does_not_evaluate_saved_shots_or_energy(tmp_path,monkeypatch):
    c=invented(tmp_path)
    for module,method in [(m.accuracy,'assess_round'),(m.accuracy,'assess_energy'),(m.screen,'_count')]:
        monkeypatch.setattr(module,method,lambda *a,**k:pytest.fail('passive numerical evaluation'))
    assert m.preflight(tmp_path,tmp_path/'contract.json')==c


@pytest.mark.parametrize('fault',['hash','role','source','coverage','duplicate','rank','failure','alias','target','nan'])
def test_metadata_mismatches_STOP(tmp_path,fault):
    c=invented(tmp_path)
    def change(key,func):
        p=tmp_path/c['records'][key];d=m.io.read(p);func(d);p.write_text(json.dumps(d));c['saved_inputs'][str(p.relative_to(tmp_path))]=m.io.sha(p)
    if fault=='hash':c['saved_inputs']['B_result.json']='0'*64
    if fault=='role':c['identities']=deepcopy(c['identities']);c['identities']['rc2']['role']='baseline'
    if fault=='source':change('SOURCE',lambda d:d.update(execution_commit='old-source'))
    if fault=='coverage':change('result',lambda d:d['arms']['pr']['rounds'].pop())
    if fault=='duplicate':change('result',lambda d:d['matched_rounds'].__setitem__(1,d['matched_rounds'][0]))
    if fault=='rank':change('contract',lambda d:d.update(policy=dict(d['policy'],rank=7)))
    if fault=='failure':change('STOP',lambda d:d.update(failed=True))
    if fault=='alias':change('pr_ground',lambda d:d.update(alias_margin=-1.))
    if fault=='target':c['target']=dict(c['target'],alpha_total=.05)
    if fault=='nan':change('result',lambda d:d['arms']['rc2']['rounds'][0]['reference'].update(attenuation=float('nan')))
    (tmp_path/'contract.json').write_text(json.dumps(c))
    with pytest.raises((ValueError,KeyError)):m.preflight(tmp_path,tmp_path/'contract.json')


@pytest.mark.parametrize('A',[1.,.7,1e-5])
@pytest.mark.parametrize('lane',m.LANES)
def test_existing_assess_round_and_qualification_agree(A,lane):
    p=packet(A=A);t=m.accuracy.AccuracyTarget(**m.TARGET);q=m.qualification(p,lane)
    old=m.c1.assess_round(t,p,q,p['rounds'][0]);new=m.assess_round(t,p,q,p['rounds'][0])
    for k,v in old.items():assert new[k]==v
    assert new['shots_decimal']==(None if old['shots'] is None else str(old['shots']))
    assert new['axes']['X']==new['axes']['Y']
    assert not q['all_law_bound_qualified'] and not q['native_cost_proxy_assumed']
    if lane=='evidence':
        assert new['status']=='UNRESOLVED_SIGNAL_ERROR' and new['shots'] is None
        assert new['raw_signal_error'] is new['radius_lower'] is new['epsilon_coordinate'] is None


@pytest.mark.parametrize('A',[1e-40,1e-200,float.fromhex('0x0.0000000000001p-1022')])
def test_big_integer_ceil_is_exact_for_declared_binary64_constants(A):
    p=packet(A=A);t=m.accuracy.AccuracyTarget(**m.TARGET);q=m.qualification(p,'ideal_exact_native')
    r=m.assess_round(t,p,q,p['rounds'][0]);n=r['shots']
    assert type(n) is int and n>2**63 and str(n)==r['shots_decimal']
    epsilon=(Fraction(r['epsilon_coordinate_fraction_binary64_constants']) if r['range_extension']
        else Fraction.from_float(r['epsilon_coordinate']))
    numerator=2*Fraction.from_float(math.log(2/t.alpha_axis))
    assert (n-1)*epsilon*epsilon < numerator <= n*epsilon*epsilon
    assert json.loads(json.dumps(r))['shots']==n
    if A==float.fromhex('0x0.0000000000001p-1022'):
        assert r['epsilon_coordinate'] is None and epsilon>0


def test_beyond_binary64_integer_resolution_refines_ceil_only():
    p=packet(A=1e-7);t=m.accuracy.AccuracyTarget(**m.TARGET);q=m.qualification(p,'assumed_raw_signal_budget')
    old=m.c1.assess_round(t,p,q,p['rounds'][0]);new=m.assess_round(t,p,q,p['rounds'][0])
    assert old['shots']>2**53 and new['arithmetic']=='rational_ceil_binary64_constants'
    for key in ('raw_signal_error','radius_lower','beta_deterministic','epsilon_coordinate'):assert new[key]==old[key]
    eps=Fraction.from_float(old['epsilon_coordinate']);bound=2*Fraction.from_float(math.log(2/t.alpha_axis))
    assert (new['shots']-1)*eps*eps < bound <= new['shots']*eps*eps


@pytest.mark.parametrize('A',[float('nan'),float('inf'),-1.,1.1])
def test_invalid_A_is_technical_failure(A):
    p=packet(A=A)
    with pytest.raises(ValueError):m.assess_round(m.accuracy.AccuracyTarget(**m.TARGET),p,m.qualification(p,'evidence'),p['rounds'][0])


def test_zero_radius_and_phase_failure_are_distinct_from_overflow():
    t=m.accuracy.AccuracyTarget(**m.TARGET)
    for p,want in [(packet(A=0.),'INFEASIBLE_RADIUS'),(packet(pf=.5),'INFEASIBLE_PHASE')]:
        r=m.assess_round(t,p,m.qualification(p,'ideal_exact_native'),p['rounds'][0])
        assert r['status']==want and r['shots'] is None
        assert 'raw_signal_error' in r and 'radius_lower' in r


def test_subnormal_phase_failure_preserves_positive_exact_error_diagnostics():
    p=packet(A=float.fromhex('0x0.0000000000001p-1022'),pf=.3)
    r=m.assess_round(m.accuracy.AccuracyTarget(**m.TARGET),p,m.qualification(p,'ideal_exact_native'),p['rounds'][0])
    assert r['status']=='INFEASIBLE_PHASE' and r['shots'] is None
    assert r['raw_signal_error'] is None
    assert Fraction(r['raw_signal_error_fraction_binary64_constants'])>0


def test_summary_W_XY_bigints_and_energy_reuse():
    t=m.accuracy.AccuracyTarget(**m.TARGET);p=packet(A=1e-40);q=m.qualification(p,'ideal_exact_native')
    plans=[m.assess_round(t,p,q,row) for row in p['rounds']]
    e=m.c1.energy(t,p,plans)
    assert e==m.accuracy.assess_energy(t,p['df_bound_Ha'],plans,state_model='own_DF_ground_state_oracle',alias_branch_verified=True)
    result=m.shots_weight_summary(p,q,plans,e)
    assert result['status']=='CONDITIONAL_SHOTS_AVAILABLE'
    assert int(result['N_total_decimal'])==sum(2*r['shots'] for r in plans)
    assert int(result['W_decimal'])==sum(2*(1<<r['m'])*r['shots'] for r in plans)
    assert int(result['W_decimal'])!=128*sum(2*(1<<r['m'])*r['shots'] for r in plans)
    assert int(result['axis_total_decimal']['X'])+int(result['axis_total_decimal']['Y'])==int(result['N_total_decimal'])
    assert sum(int(result['groups'][g]['N_decimal']) for g in ('m0_to9','m10_to12'))==int(result['N_total_decimal'])
    assert result['mu'] is result['S'] is result['total_resources'] is None
    broken=deepcopy(plans);broken[0]['axes']['X']['shots_decimal']='1'
    with pytest.raises(ValueError,match='XY'):m.shots_weight_summary(p,q,broken,e)


def test_comparison_signed_decimal_differences_have_no_float_roundtrip():
    n=(1<<220)+7
    arms={r:dict(shots_weight=dict(status='CONDITIONAL_SHOTS_AVAILABLE',N_total_decimal=str(v),W_decimal=str(3*v)))
        for r,v in [('rc2',n),('pr',n-9)]}
    result=m.compare('ideal_exact_native',arms)
    assert result['PR_minus_RC2_decimal']==dict(N_total_decimal='-9',W_decimal='-27')
    assert result['native_price'] is result['total_resources'] is result['guaranteed_advantage'] is None


def test_DF_budget_counted_once_only_in_energy():
    t=m.accuracy.AccuracyTarget(**m.TARGET);p=packet();q=m.qualification(p,'ideal_exact_native')
    plans=[m.assess_round(t,p,q,row) for row in p['rounds']]
    a=m.c1.energy(t,p,plans);b=m.c1.energy(t,dict(p,df_bound_Ha=4e-5),plans)
    assert b['energy_error_bound_ha']-a['energy_error_bound_ha']==pytest.approx(4e-5-p['df_bound_Ha'])
    assert plans[0]['raw_signal_error']==pytest.approx(.8*(.002+.003))


@pytest.mark.parametrize('fault',[None,'technical','sensitivity_phase','huge'])
def test_artificial_E2E_105_calls_fresh_preserve_156axes(tmp_path,monkeypatch,fault):
    c=invented(tmp_path,A=1e-40 if fault=='huge' else .8,
        pf=.257 if fault=='sensitivity_phase' else .002,rte=.001 if fault=='sensitivity_phase' else .003)
    old_hashes={n:m.io.sha(tmp_path/n) for n in c['saved_inputs']}
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'SYNTHETIC_SOURCE\n')
    for module,name in [(m.stage_b,'run'),(m.c1,'run'),(m.stage_b.b,'build_kernel')]:
        monkeypatch.setattr(module,name,lambda *a,**k:pytest.fail('forbidden physics/old runner/native'))
    if fault=='technical':monkeypatch.setattr(m,'assess_round',lambda *a:(_ for _ in ()).throw(ValueError('INVENTED_NUMERICAL_MISMATCH')))
    out=tmp_path/m.FRESH
    if fault=='technical':
        with pytest.raises(ValueError):m.run(tmp_path,tmp_path/'contract.json',out)
        stop=m.io.read(out/'STOP.json');assert stop['failed'] and stop['scientific_units']<105
        assert m.io.read(out/'failure.json')['status']=='TECHNICAL_FAILURE'
    else:
        result=m.run(tmp_path,tmp_path/'contract.json',out);stop=m.io.read(out/'STOP.json')
        assert stop['scientific_units']==105 and stop['call_counts']==m.LIMITS and not stop['failed']
        assert result['comparisons']['evidence']['status']=='EVIDENCE_UNQUALIFIED'
        assert result['comparisons']['ideal_exact_native']['status']=='CONDITIONAL_SHOTS_AVAILABLE'
        assert result['comparisons']['assumed_raw_signal_budget']['status']==('ONE_OR_BOTH_INFEASIBLE' if fault=='sensitivity_phase' else 'CONDITIONAL_SHOTS_AVAILABLE')
        rows=list(csv.DictReader((out/'axes156.csv').open()));assert len(rows)==156
        for row in rows:
            r=result['arms'][row['arm']][row['lane']]['rounds'][int(row['m'])]
            assert row['shots_decimal']==('' if r['shots_decimal'] is None else r['shots_decimal'])
            assert row['status']==r['status']
        assert result['actual_CA_achieved'] is result['native_price'] is result['total_resources'] is None
    assert stop['retry']==stop['resume']==0 and stop['mandatory_stop']
    assert old_hashes=={n:m.io.sha(tmp_path/n) for n in old_hashes}
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,tmp_path/'contract.json',out)


def test_atomic_JSON_bigints_numpy_scalars_and_no_precision_loss(tmp_path):
    import numpy as np
    n=(1<<250)+17
    m.save(tmp_path/'big.json',dict(shots=n,shots_decimal=str(n),small=np.int64(7),guard=np.float64(.125),flag=np.bool_(True)))
    d=m.io.read(tmp_path/'big.json')
    assert d['shots']==int(d['shots_decimal'])==n and type(d['small']) is int and d['flag'] is True
