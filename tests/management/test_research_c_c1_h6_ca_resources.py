"""Artificial scalar JSON only; no real H6 joins or circuit actions."""
from pathlib import Path
import copy,json,math,sys,subprocess
from unittest.mock import patch,Mock
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h6_ca_resources as m


def metrics(value,measurement=0):
    return {k:value for k in m.accuracy.METRICS}|dict(measurement_count=measurement)


def make_fixture(root,fault=None):
    pf=dict(status='H6_SAVED_RANK14_SECTOR_PF_COMPLETE',saved_DF={},ground={a:dict(status='PASS') for a in ['physical','CDF','XDF']},candidates={})
    bound=dict(status='H6_NUMERICAL_RESTRICTED_BOUND_COMPLETE',evaluated_groups=4,routes={})
    native=dict(status='H6_NATIVE_FINITE_PROBE_COST_COMPLETE',candidates={})
    identity=dict(groups=[],aliases={},identities={})
    for arm in ['CDF','XDF']:
        df=.00001;guard=.000001
        pf['saved_DF'][arm]=dict(status='PASS',actual_rank=14,budget_Ha=.00159/8,
            saved_sector_error_Ha=df,saved_guard_Ha=guard,norm_plus_guard_Ha=df+guard)
        for s in (1,4):
            key=f'{arm}:s{s}';labels=[f'{arm}_s{s}_{r}' for r in ['literal','shared']];identity['groups'].append(labels)
            rows=[];old=[]
            for i in range(13):
                error=.002*(1<<i)/4096;syn=1e-8*(1<<i);combined=math.fsum([error,syn])
                if fault=='infeasible' and arm=='CDF' and s==1 and i==12:
                    error=.3;combined=math.fsum([error,syn])
                row=dict(m=i,s=s,occurrences=s*(1<<i),evidence_level=m.synthesis.bounds.LEVEL,
                    saved_PF_operator_error=error,complex_signal_bound_pilot=syn,
                    combined_complex_error_pilot=combined,phase_bias_bound_pilot=math.asin(combined),registered_phase_room=math.pi/12)
                if fault=='signal' and arm=='CDF' and s==1 and i==3:row['combined_complex_error_pilot']+=.01
                if fault=='unresolved' and arm=='CDF' and s==1 and i==0:
                    row.update(complex_signal_bound_pilot=None,combined_complex_error_pilot=None,phase_bias_bound_pilot=None)
                rows.append(row);old.append(dict(m=i,attenuation=1.,rte_corrected_operator_error=0.,pf_operator_error=error))
            pf['candidates'][key]=dict(arm=arm,s=s,rounds=old)
            for label in labels:
                identity['aliases'][label]=labels[0];identity['identities'][label]=dict(signature=key)
                bound['routes'][label]=dict(kernel=dict(covered_columns=800,input_dimension=800,output_dimension=8192,evidence_level=m.synthesis.bounds.LEVEL),rounds=copy.deepcopy(rows))
                native['candidates'][label]=dict(metrics=metrics(10 if arm=='CDF' else 20))
    shells=dict(metrics={k:metrics(1 if k=='head' else 3,0 if k=='head' else 1) for k in ['head','X','Y']})
    if fault=='identity':identity['identities']['CDF_s1_shared']['signature']='changed'
    if fault=='nonfinite':native['candidates']['CDF_s1_literal']['metrics']['rz_count']=float('nan')
    if fault=='alias':pf['ground']['CDF']['status']='FAIL'
    data=dict(PF=pf,bound=bound,native=native,identity=identity,shells=shells)
    for k in ['PF','bound','native']:data[k+'_STOP']=dict(failed=False,mandatory_stop=True,scientific_units=3335 if k=='bound' else 72)
    records={}
    for k,v in data.items():
        f=k+'.json'
        if fault=='nonfinite' and k=='native':(root/f).write_text(json.dumps(v))
        else:m.io.save(root/f,v)
        records[k]=f
    (root/'source.py').write_text('# artificial source only\n');m.io.save(root/'SOURCE.json',dict(files={'source.py':m.io.sha(root/'source.py')}))
    c=dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',conditions=m.synthesis.native.saved.CONDITIONS,
        accuracy_target=m.TARGET,cost_scope=m.COST_SCOPE,call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,
        candidate_order=[f'{a}:s{s}' for a,s in m.CANDIDATES],**{k:0 for k in m.FORBIDDEN},
        records=records,source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),
        saved_inputs={f:m.io.sha(root/f) for f in records.values()},evaluation_packages={})
    path=root/'contract.json';m.io.save(path,c);return path,data


@pytest.mark.parametrize('fault',[None,'infeasible','unresolved','identity','nonfinite','alias','signal'])
def test_artificial_end_to_end_and_partial_STOP(tmp_path,fault):
    path,data=make_fixture(tmp_path,fault);out=tmp_path/'out'
    forbidden=Mock(side_effect=AssertionError('NO_H6_OPERATOR_FIT_QPY_BUILD_COMPILE_OR_REGENERATION'))
    with patch.object(m.synthesis,'run',forbidden),patch.object(m.synthesis.native,'run',forbidden):
        if fault in ('identity','nonfinite','alias','signal'):
            with pytest.raises(ValueError):m.run(tmp_path,path,out)
        else:m.run(tmp_path,path,out)
    forbidden.assert_not_called();stop=m.io.read(out/'STOP.json');assert stop['retry']==stop['resume']==0
    assert stop['scientific_units']<=m.MAX_CALLS==119
    if fault in ('identity','nonfinite','alias','signal'):
        assert stop['failed'] and (out/'failure.json').exists() and not (out/'result.json').exists()
        e=[json.loads(s) for s in (out/'science_calls.jsonl').read_text().splitlines()]
        fail=next(i for i,x in enumerate(e) if x['event']=='FAIL');assert not any(x['event']=='attempt' for x in e[fail+1:])
        if fault=='signal':
            partial=[json.loads(s) for s in (out/'candidates/CDF:s1.progress.jsonl').read_text().splitlines()]
            assert [x['m'] for x in partial if x['phase']=='round_accuracy']==[0,1,2]
    else:
        r=m.io.read(out/'result.json');assert not stop['failed'] and len(r['candidates'])==4
        candidate=r['candidates']['CDF:s1']
        if fault:
            assert candidate['status']==('UNRESOLVED' if fault=='unresolved' else 'INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION')
            assert candidate['cost']['totals'] is None and candidate['per_shot_costs']==[]
        else:
            assert stop['scientific_units']==119 and stop['call_counts']['round_serial_cost']==52
            assert all(v['status']=='NUMERICAL_PILOT_CA_PASS' for v in r['candidates'].values())
            assert all(v['cost']['shots']==v['cost']['measurement_count']==v['cost']['executions'] for v in r['candidates'].values())
            assert r['comparisons']['1']['comparison']['verdict']=='CDF_LOWER_CONDITIONAL_INTERROGATION_COST'
            for v in r['candidates'].values():
                assert len(v['rounds'])==13 and len(v['per_shot_costs'])==26
                assert v['CA_achieved'] is v['rigorous_enclosure'] is v['guaranteed_actual_shots'] is None
                assert v['guaranteed_total_resource_superiority'] is False
                expected_df=data['PF']['saved_DF'][v['arm']]['norm_plus_guard_Ha']
                assert v['numerical_energy_assessment']['energy_error_bound_ha']==pytest.approx(expected_df+m.TARGET['beta_total']/(m.TARGET['base_time']*4096))
                assert v['numerical_energy_assessment']['statistical_failure_upper']<=.01
                assert 'one predeclared candidate' in v['failure_scope']
                for p in v['rounds']:assert p['raw_signal_error']==p['source_combined_complex_error'] and p['statistical_margin']>=0
    saved=(out/'STOP.json').read_bytes()
    with pytest.raises(FileExistsError):m.run(tmp_path,path,out)
    assert (out/'STOP.json').read_bytes()==saved


def packet_and_row(tmp_path):
    _,b=make_fixture(tmp_path);p=m.attach_evidence(b,'CDF',1);return p,copy.deepcopy(p['rounds'][0])


def test_reuse_actual_solver_energy_and_serial_cost(tmp_path):
    p,row=packet_and_row(tmp_path);target=m.accuracy.AccuracyTarget(**m.TARGET)
    with patch.object(m.accuracy,'assess_round',wraps=m.accuracy.assess_round) as solver:
        r=m.assess_saved_round(target,p,row)
    assert solver.call_count==1 and r['raw_signal_error']==row['combined_complex_error_pilot']
    args=solver.call_args.args[1];assert args.pf_operator_error==row['saved_PF_operator_error'] and args.synthesis_signal_error==row['complex_signal_bound_pilot']
    assert args.rte_corrected_operator_error==args.finite_law_signal_error==0 and args.attenuation==1
    assert r['shots']==m.accuracy._legacy('rpe_resource_accounting')._hoeffding_shots(r['epsilon_coordinate'],target.alpha_axis)
    p['s']=4;cost=m.serial_round_cost(p,12)
    assert all(x['kernel_occurrences']==16384 and x['mean']['total_depth']==16384*10+1+3 for x in cost)
    assert all(x['depth_semantics']=='scheduled_serial_sum' for x in cost)


@pytest.mark.parametrize('kind',['overflow','underflow','huge','noninteger','zero','too_few'])
def test_shots_overflow_and_integer_margin_handling(tmp_path,kind):
    p,row=packet_and_row(tmp_path);t=m.accuracy.AccuracyTarget(**m.TARGET)
    legacy=m.accuracy._legacy('rpe_resource_accounting')
    if kind=='overflow':mock=Mock(side_effect=OverflowError('artificial overflow'))
    elif kind=='underflow':mock=Mock(side_effect=ZeroDivisionError('artificial coordinate-square underflow'))
    else:mock=Mock(return_value={'huge':1<<60,'noninteger':1.5,'zero':0,'too_few':1}[kind])
    with patch.object(legacy,'_hoeffding_shots',mock):
        if kind in ('noninteger','zero'):
            with pytest.raises(ValueError,match='INTEGER'):m.assess_saved_round(t,p,row)
        else:
            r=m.assess_saved_round(t,p,row);assert r['status'].startswith('UNRESOLVED') and r['shots'] is None
            if kind=='huge':assert r['unqualified_shots_decimal']==str(1<<60)


@pytest.mark.parametrize('kind',['double_count','phase','unknown','level','nonfinite','radius'])
def test_saved_bound_integrity_and_qualification(tmp_path,kind):
    p,r=packet_and_row(tmp_path)
    if kind=='double_count':r['combined_complex_error_pilot']+=r['saved_PF_operator_error']
    if kind=='phase':r['phase_bias_bound_pilot']+=.01
    if kind=='unknown':r['complex_signal_bound_pilot']=None
    if kind=='level':r['evidence_level']='A_FINITE_PROBES'
    if kind=='nonfinite':r['combined_complex_error_pilot']=float('nan')
    if kind=='radius':r.update(saved_PF_operator_error=1.,complex_signal_bound_pilot=.1,combined_complex_error_pilot=1.1,phase_bias_bound_pilot=None)
    if kind in ('unknown','level'):
        assert m.assess_saved_round(m.accuracy.AccuracyTarget(**m.TARGET),p,r)['status'].startswith('UNRESOLVED')
    elif kind=='radius':assert m.assess_saved_round(m.accuracy.AccuracyTarget(**m.TARGET),p,r)['status']=='INFEASIBLE_RADIUS'
    else:
        with pytest.raises(ValueError):m.assess_saved_round(m.accuracy.AccuracyTarget(**m.TARGET),p,r)


def test_cost_capacity_never_silently_clipped(tmp_path):
    p,_=packet_and_row(tmp_path)
    fake=dict(status='CONDITIONAL_TOTAL_AVAILABLE',totals={k:float(1<<60) for k in m.accuracy.METRICS})
    with patch.object(m.accuracy,'aggregate_totals',return_value=fake):r=m.aggregate_candidate(p,[],dict(same_accuracy=True),[])
    assert r['status']=='UNRESOLVED' and r['cost']['totals'] is None


@pytest.mark.parametrize('status',['INCOMPLETE_COST_COVERAGE','UNQUALIFIED_LONG_ROUND_COST'])
def test_missing_cost_qualification_is_unresolved_even_if_energy_passes(tmp_path,status):
    p,_=packet_and_row(tmp_path)
    with patch.object(m.accuracy,'aggregate_totals',return_value=dict(status=status,totals=None)):
        r=m.aggregate_candidate(p,[],dict(same_accuracy=True),[])
    assert r['status']=='UNRESOLVED' and r['cost']['totals'] is None


@pytest.mark.parametrize('fault',[None,'source','input','budget','calls'])
def test_passive_preflight_and_fixed_policy(tmp_path,fault):
    path,_=make_fixture(tmp_path);c=m.io.read(path)
    if fault=='source':(tmp_path/'source.py').write_text('changed')
    if fault=='input':(tmp_path/'PF.json').write_text('{}')
    if fault=='budget':c['accuracy_target']=dict(c['accuracy_target'],alpha_total=.04)
    if fault=='calls':c['maximum_logical_units']+=1
    path.write_text(json.dumps(c))
    with patch.object(m,'load_bundle',side_effect=AssertionError('NO_NUMERIC_JOIN')):
        if fault:
            with pytest.raises(ValueError):m.preflight(tmp_path,path)
        else:assert m.preflight(tmp_path,path)==c


def test_passive_import_does_not_import_numeric_or_circuit_packages():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c1_h6_ca_resources;assert not any(k in sys.modules for k in ['numpy','qiskit','scipy','ffsim'])"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)


def test_fixed_failure_policy_matches_previous_registered_contracts():
    old=m.io.read(ROOT/'docs/research/research_c_c1_stage3_cost_compat_contract_20261025.json')
    assert old['proposal']['accuracy_target']==m.TARGET
    context=m.io.read(ROOT/'docs/research/research_c_c1_h6_saved_pf_contract_unapproved_20261025.json')['phase_budget_context']
    assert context['beta_total']=='pi/6' and context['beta_stat']=='pi/12'
