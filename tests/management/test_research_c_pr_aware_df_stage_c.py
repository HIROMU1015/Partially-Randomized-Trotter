"""Invented scalar data only. Never computes shots/costs from saved H4."""
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c_pr_aware_df_stage_c as m


def packet(arm='baseline',A=.8,unit_cost=100,df=1e-5):
    metrics=[{k:unit_cost+i for k in m.METRICS} for i in range(32)]
    mean={k:unit_cost+15.5 for k in m.METRICS}
    shells={name:dict(metrics={k:cost for k in m.METRICS}) for name,cost in [('head',2),('X',3),('Y',5)]}
    return dict(arm=arm,target_id=arm+':artificial',rounds=[dict(m=i,reference=dict(attenuation=A,
        pf_operator_error=.002,rte_corrected_operator_error=.003,raw_signal_parts=[.7,.1],
        evidence_kind='binary64_sector_operator_numerical_pilot'),synthesis_error=None,implemented_law_error=None) for i in range(13)],
        df_bound_Ha=df,kernel_samples=metrics,shells=shells,error_evidence=None,
        saved_calibration=dict(samples=32,sample_mean=mean,sample_mean_standard_error={k:1.5 for k in m.METRICS}))


def test_import_passive_and_fixed_calls():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c_pr_aware_df_stage_c as m;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit','ffsim'));assert m.MAX_CALLS==164"
    subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,check=True)


def test_13_round_XY_serial_fences_shell_once_and_no_32_multiplier():
    p=packet();k=m.cost_calibration(p)
    costs=[m.serial_cost(p,k,i,axis) for i in range(13) for axis in ('X','Y')]
    assert len(costs)==26
    for row in costs:
        n=4*(1<<row['m']);shell=5 if row['axis']=='X' else 7
        assert row['kernel_occurrences']==n
        assert row['mean']['rz_count']==n*115.5+shell
        assert row['mean']['total_depth']==n*115.5+shell
        assert row['depth_semantics']=='scheduled_serial_sum' and row['classical_cost_samples']==32
        assert row['lower'] is row['upper'] is None


def test_unknown_errors_do_not_produce_evidence_shots_or_CA():
    p=packet();q=m.qualification(p,'evidence');target=m.accuracy.AccuracyTarget(**m.TARGET)
    rounds=[m.assess_round(target,p,q,r) for r in p['rounds']]
    assert all(r['shots'] is None and r['status']=='UNRESOLVED_SIGNAL_ERROR' for r in rounds)
    energy=m.energy(target,p,rounds);assert not energy['same_accuracy']
    assert m.accuracy.aggregate_totals(energy,rounds,[],cost_scope=m.COST_SCOPE)['totals'] is None


@pytest.mark.parametrize('lane',m.LANES[1:])
def test_explicit_scenario_shots_with_no_evidence_promotion(lane):
    p=packet();q=m.qualification(p,lane);t=m.accuracy.AccuracyTarget(**m.TARGET)
    rows=[m.assess_round(t,p,q,r) for r in p['rounds']]
    assert all(r['status']=='CONDITIONAL_ROUND_PASS' and r['assumption_applied'] for r in rows)
    assert q['kind']=='ASSUMED_SCENARIO' and not q['all_law_bound_qualified']
    assert q['evidence_promoted'] is False
    assert m.energy(t,p,rows)['same_accuracy']


def test_DF_Ha_is_only_counted_in_energy_and_not_signal_room():
    a=packet(df=1e-8);b=packet(df=.0001);t=m.accuracy.AccuracyTarget(**m.TARGET)
    plan=[]
    for p in (a,b):
        q=m.qualification(p,'ideal_exact_native');plan.append([m.assess_round(t,p,q,r) for r in p['rounds']])
    assert [r['shots'] for r in plan[0]]==[r['shots'] for r in plan[1]]
    assert plan[0][0]['raw_signal_error']==pytest.approx(.8*(.002+.003))
    e=[m.energy(t,p,r)['energy_error_bound_ha'] for p,r in zip((a,b),plan)]
    assert e[1]-e[0]==pytest.approx(.0001-1e-8)


def test_lower_attenuation_increases_shots_and_can_outweigh_cheaper_kernel():
    t=m.accuracy.AccuracyTarget(**m.TARGET);totals=[]
    for p in (packet(A=.9,unit_cost=100),packet('proposed',A=.2,unit_cost=50)):
        q=m.qualification(p,'ideal_exact_native');r=[m.assess_round(t,p,q,x) for x in p['rounds']]
        k=m.cost_calibration(p);c=[m.serial_cost(p,k,i,a) for i in range(13) for a in ('X','Y')]
        e=m.energy(t,p,r);totals.append(m.accuracy.aggregate_totals(e,r,c,cost_scope=m.COST_SCOPE))
    assert totals[1]['shots']>totals[0]['shots']
    assert totals[1]['totals']['rz_count']>totals[0]['totals']['rz_count']
    assert totals[0]['classical_cost_samples_are_shot_multipliers'] is False


def test_shared_calibration_uncertainty_reuses_one_mean_not_13_iid_means():
    p=packet();plans=[dict(m=i,shots=i+1) for i in range(13)]
    out=m.calibration_uncertainty(p,plans,dict(totals={}))
    W=sum(2*(i+1)*4*(1<<i) for i in range(13))
    assert out['weighted_SE']['rz_count']==W*1.5
    assert out['independent_round_calibrations']==1 and out['guaranteed_CI'] is None


def test_observed_extrema_are_not_support_and_no_false_intervals():
    p=packet();k=m.cost_calibration(p)
    assert k['lower'] is k['upper'] is k['cost_support'] is None
    fake=dict(kind='sample_min_max',law='actual_implemented_law',provenance='observed32',
        bounds={k:[0,200] for k in m.METRICS})
    with pytest.raises(ValueError,match='OBSERVED_EXTREMA'):m.cost_calibration(p,support=fake,alpha_per_metric=.01/12)
    # Purely artificial analytic supports demonstrate the extension API.
    fake['kind']='analytic_all_law';fake['provenance']='invented exact cost in[0,200]'
    k=m.cost_calibration(p,support=fake,alpha_per_metric=.01/12)
    assert k['simultaneous_interval_qualified']


@pytest.mark.parametrize('bad_kind',['sample_max_residual','finite_probe','rigorous_bound_missing_scope'])
def test_sample_errors_cannot_become_all_law_qualification(bad_kind):
    p=packet();p['error_evidence']=dict(kind=bad_kind)
    with pytest.raises(ValueError,match='UNQUALIFIED_ERROR'):m.qualification(p,'evidence')


def test_artificial_all_law_bound_can_qualify_conditional_shots_but_not_B1_rigour():
    p=packet();p['error_evidence']=dict(kind='rigorous_all_law_bound',norm='raw_complex_signal',
        law_model='actual_implemented_law',coverage='all_trajectories_all_occurrences_and_shells',
        provenance='invented analytical model',round_errors=[dict(m=i,synthesis=1e-5,law=1e-5) for i in range(13)])
    q=m.qualification(p,'evidence');t=m.accuracy.AccuracyTarget(**m.TARGET)
    rows=[m.assess_round(t,p,q,r) for r in p['rounds']]
    e=m.energy(t,p,rows)
    assert q['all_law_bound_qualified'] and all(r['shots']>0 for r in rows)
    assert e['numerical_pilot'] # saved PF/DF were numerical; no upgrade.


@pytest.mark.parametrize('A',[0.,1e-200])
def test_extreme_attenuation_is_explicit_not_clipped(A):
    p=packet(A=A);q=m.qualification(p,'ideal_exact_native')
    row=m.assess_round(m.accuracy.AccuracyTarget(**m.TARGET),p,q,p['rounds'][0])
    assert row['shots'] is None
    assert row['status'] in ('INFEASIBLE_RADIUS','UNRESOLVED_NUMERICAL_SHOT_RANGE')


def fixture(root):
    packets={a:packet(a,A=.8 if a=='baseline' else .9,unit_cost=100 if a=='baseline' else 90) for a in m.ARMS}
    records={};paths={};samples={};exports={};saved={}
    for a in m.ARMS:
        name=a+'.npz';(root/name).write_bytes(b'opaque synthetic export '+a.encode());exports[a]=name
    c=dict(final_exports=exports,export_hashes={p:m.io.sha(root/p) for p in exports.values()})
    Ar=dict(status='STAGE_A_PROXY_AND_DF_PASS')
    B1=dict(status='B1_MECHANISM_EVALUATION_COMPLETE',arms={})
    for a in m.ARMS:
        B1['arms'][a]=dict(arm=a,classification=dict(status='POTENTIALLY_FEASIBLE'),
            rounds=packets[a]['rounds'],saved_DF_bound_Ha=packets[a]['df_bound_Ha'])
    B2=dict(status='B2_FINITE_BIT_NATIVE_CALIBRATION_COMPLETE',arms={a:packets[a]['saved_calibration'] for a in m.ARMS},
        shells=packets['baseline']['shells'],cost_support=None,whole_law_synthesis_bound=None,implemented_law_error=None)
    for stage,result,count,source in [('A',Ar,30,'e1085c95f5c01b68566e17d543ace5832cb94061'),('B1',B1,141,m.B1_SOURCE),('B2',B2,652,m.B2_SOURCE)]:
        records[stage+'_result']=result
        records[stage+'_STOP']=dict(failed=False,mandatory_stop=True,scientific_units=count)
        records[stage+'_SOURCE']=dict(execution_commit=source,files={})
    records['B1_STOP']['call_counts']=m.b1.LIMITS
    records['B2_STOP'].update(call_counts=m.b2.LIMITS,technical_relaunch=1,resume=0,retry_within_run=0,cumulative_scientific_units=1081)
    records['B2_SOURCE']['sampling_SOURCE_sha256']=m.SAMPLING_SOURCE
    records['B1_manifest']=dict(files={});records['B2_manifest']=dict(files={})
    records['B1_contract']=dict(policy=m.b1.POLICY,final_exports=exports,saved_inputs=c['export_hashes'])
    records['B2_original_contract']=dict(policy=m.b2.POLICY,compiler=m.b2.profile_dict(),final_exports=exports,saved_inputs=c['export_hashes'])
    records['B2_shells']=B2['shells']
    for a in m.ARMS:records[a+'_ground']=dict(status='PASS')
    gate=dict(rtol=0,tolerance=1e-10,max_entry=1e-14)
    for a in m.ARMS:
        samples[a]=[]
        for i in range(32):
            row=dict(trajectory=dict(arm=a,sample=i,SOURCE_sha256=m.SAMPLING_SOURCE,
                final_JU_sha256=c['export_hashes'][exports[a]],microsteps=16,quantum_shots=0,
                reused_as_coherent_repetitions=False,seed=i+(0 if a=='baseline' else 100)),
                raw=dict(reference=gate,unitarity=gate),compiled=dict(reference=gate,raw=gate,unitarity=gate),
                adjoint={kind:dict(reference=gate,returned=gate) for kind in ('raw','compiled')},
                XY=dict(status='PASS'),metrics=dict(packets[a]['kernel_samples'][i],measurement_count=0))
            ps={}
            for k,v in row.items():
                name=f'{a}_{i}_{k}.json';m.io.save(root/name,v);ps[k]=name
            samples[a].append(ps)
    for k,v in records.items():
        name=k+'.json';m.io.save(root/name,v);paths[k]=name
    m.io.save(root/'source.json',dict(files={}))
    required={*paths.values(),*exports.values()}
    for rows in samples.values():
        for row in rows:required.update(row.values())
    c.update(records=paths,samples=samples,source_manifest='source.json',source_manifest_sha256=m.io.sha(root/'source.json'),
        saved_inputs={p:m.io.sha(root/p) for p in required},scope=m.SCOPE,status='UNAPPROVED_PROPOSAL',implementation=m.VERSION,
        target=m.TARGET,scenarios=list(m.SCENARIOS),lanes=list(m.LANES),call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        fresh_output=m.FRESH,cost_scope=m.COST_SCOPE,retry=0,resume=0,new_fit=0,new_sampling=0,build_compile=0,evaluation_packages={})
    return c


def test_passive_preflight_blocks_real_numeric_helpers(tmp_path,monkeypatch):
    c=fixture(tmp_path);path=tmp_path/'contract.json';m.io.save(path,c)
    def forbidden(*a,**k):raise AssertionError('NO_SCIENTIFIC_REANALYSIS')
    monkeypatch.setattr(m.accuracy,'assess_round',forbidden)
    monkeypatch.setattr(m.accuracy,'bounded_cost_intervals',forbidden)
    monkeypatch.setattr(m.connection,'compose_serial_cost',forbidden)
    assert m.preflight(tmp_path,path)==c
    p=tmp_path/c['records']['B2_STOP'];p.write_text('{}')
    with pytest.raises(ValueError,match='SOURCE_OR_INPUT'):m.preflight(tmp_path,path)


@pytest.mark.parametrize('fault',['source','compiled','duplicate_run01','arm'])
def test_static_provenance_and_verdict_faults_STOP(fault,tmp_path):
    c=fixture(tmp_path);bundle=m.load_bundle(tmp_path,c)
    if fault=='source':bundle['records']['B2_SOURCE']['execution_commit']='wrong'
    if fault=='compiled':bundle['samples']['baseline'][0]['compiled']['raw']['max_entry']=.1
    if fault=='duplicate_run01':bundle['samples']['baseline'].append(bundle['samples']['baseline'][0])
    if fault=='arm':bundle['samples']['baseline'][0]['trajectory']['arm']='proposed'
    with pytest.raises((ValueError,IndexError)):m.identity_gate(bundle,c)


@pytest.mark.parametrize('fault',[None,'identity','numeric'])
def test_synthetic_end_to_end_164_calls_and_no_real_science(tmp_path,monkeypatch,fault):
    c=fixture(tmp_path);path=tmp_path/'contract.json';m.io.save(path,c)
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'SYNTHETIC_SOURCE\n')
    if fault=='identity':
        monkeypatch.setattr(m,'preflight',lambda *a:c)
        p=tmp_path/c['records']['B2_SOURCE'];x=m.io.read(p);x['execution_commit']='wrong';p.write_text(json.dumps(x))
    if fault=='numeric':
        monkeypatch.setattr(m,'cost_calibration',lambda *a,**k:(_ for _ in ()).throw(ValueError('INVENTED_COST_FAULT')))
    before={p:(tmp_path/p).read_bytes() for p in c['saved_inputs']}
    out=tmp_path/m.FRESH
    if fault:
        with pytest.raises(ValueError):m.run(tmp_path,path,out)
        assert m.io.read(out/'STOP.json')['failed'] and not (out/'result.json').exists()
    else:
        m.run(tmp_path,path,out);stop=m.io.read(out/'STOP.json')
        assert stop['scientific_units']==164 and stop['call_counts']==m.LIMITS and not stop['failed']
        result=m.io.read(out/'result.json')
        assert result['CA_achieved'] is result['total_resources_guaranteed'] is None
        for a,v in result['evidence_lane'].items():
            assert v['resources']['totals'] is None and v['evidence_qualified_shots'] is None
            assert len(m.io.read(out/(a+'_serial_costs26.json')))==26
        for lane,arms in result['assumed_scenarios'].items():
            for v in arms.values():
                assert v['resources']['totals'] and v['evidence_qualified_shots'] is None
                assert v['calibration_uncertainty']['guaranteed_CI'] is None
            assert result['comparisons'][lane]['guaranteed_advantage'] is None
        assert sum(len(v['rounds']) for v in result['evidence_lane'].values())==26
    for p,data in before.items():assert (tmp_path/p).read_bytes()==data
    with pytest.raises(ValueError,match='FRESH'):m.run(tmp_path,path,out)
