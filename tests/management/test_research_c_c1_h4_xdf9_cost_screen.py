"""Invented arrays/scalars/circuits only. Never decode saved H4 NPZ or QPY."""
import copy
from dataclasses import asdict
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import qpy
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_h4_xdf9_cost_screen as m

spec = importlib.util.spec_from_file_location('native_fair_artificial_tests',
    ROOT/'tests/management/test_research_c_c1_h4_native_fair.py')
prior_tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior_tests)


def metrics(n, measurement=0):
    return {k:n for k in m.accuracy.METRICS} | dict(measurement_count=measurement)


def setup(tmp_path, monkeypatch):
    root = tmp_path/'ARTIFICIAL'; root.mkdir(); inp = root/'fake_packet.npz'
    inp.write_bytes(b'ARTIFICIAL NON-NUMERIC INPUT')
    digest = m.io.sha(inp); monkeypatch.setattr(m.io,'P_H4_SHA256',digest)
    p,_ = prior_tests.artificial_data(); packet = p['packet']
    versions = dict(compiler=m.fair.profile_dict(),gaussian=m.fair.GAUSSIAN,compiler_phase=m.fair.PHASE)
    b = dict(PF=dict(status='XDF9_STAGE1_COMPLETE',representation='PASS',ground=dict(status='PASS'),
        saved_DF=dict(rank=9,status='PASS',saved_norm_plus_allowance_Ha=1e-5),candidates={}),
        PF_contract=dict(scope=m.pf.SCOPE,conditions=copy.deepcopy(m.pf.CONDITIONS),original_input=dict(sha256=digest)),
        native=dict(status='H4_FAIR_NATIVE_FULL512_PILOT_COMPLETE',shell_metrics={
            k:metrics(1 if k=='head' else 2,int(k!='head')) for k in ('head','X','Y')}),
        native_contract=dict(conditions=copy.deepcopy(m.fair.CONDITIONS),original_input=dict(sha256=digest),**versions),
        fairness=dict(status='PASS',route=m.CONDITIONS['route'],serial_policy=m.CONDITIONS['serial_policy'],
            additional_relative_Gaussian_fusion=False,**versions),
        resources=dict(status='H4_SAVED_CA_RPE_SERIAL_RESOURCES_COMPLETE',candidates={}),
        resources_contract=dict(conditions=copy.deepcopy(m.fair.CONDITIONS),accuracy_target=m.TARGET,**versions))
    for name,units in [('PF',95),('native',305),('resources',119)]:
        b[name+'_SOURCE'] = dict(execution_commit=m.LINEAGE[name+'_SOURCE'])
        b[name+'_STOP'] = dict(failed=False,mandatory_stop=True,scientific_units=units,retry=0,resume=0)
    csvrows = []
    for s in (1,4):
        rows = [dict(target='XDF9',s=s,m=i,time=.1*(1<<i),kernel_time=.1/s,occurrences=s*(1<<i),
            pf_operator_error=.02*(1<<i)/4096,attenuation=1.,rte_corrected_operator_error=0.,sufficient_condition='PASS') for i in range(13)]
        b['PF']['candidates'][str(s)] = dict(summary=dict(status='POTENTIALLY_FEASIBLE'),rounds=rows)
        for arm in ('CDF','XDF'):
            key=f'{arm}:s{s}'; total=m.BASELINES[str(s)][arm]
            b['resources']['candidates'][key] = dict(status='NUMERICAL_PILOT_CA_PASS',cost=dict(totals=dict(rz_count=total)))
            csvrows.append(dict(candidate=key,status='NUMERICAL_PILOT_CA_PASS',rz_count=str(total)))
    records = {k:k+'.json' for k in b}
    for k,v in b.items():m.io.save(root/records[k],v)
    import csv
    with (root/'baseline.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['candidate','status','rz_count']);writer.writeheader();writer.writerows(csvrows)
    (root/'source.txt').write_text('ARTIFICIAL SOURCE')
    m.io.save(root/'SOURCE.json',dict(files={'source.txt':m.io.sha(root/'source.txt')}))
    c=dict(scope=m.SCOPE,status='UNAPPROVED_SCIENCE_CONTRACT',conditions=copy.deepcopy(m.CONDITIONS),
        accuracy_target=m.TARGET,compiler=m.fair.profile_dict(),gaussian=m.fair.GAUSSIAN,compiler_phase=m.fair.PHASE,
        monotonicity=m.MONOTONICITY,lineage=m.LINEAGE,baseline_total_RZ=m.BASELINES,
        call_limits=m.LIMITS,maximum_logical_units=m.MAX_CALLS,fresh_output=m.FAMILY+'/<execution-date>/run01',
        source_manifest='SOURCE.json',source_manifest_sha256=m.io.sha(root/'SOURCE.json'),original_input=dict(sha256=digest),
        records=records,baseline_csv='baseline.csv',saved_inputs={p:m.io.sha(root/p) for p in [*records.values(),'baseline.csv']},
        packages={},**{k:0 for k in m.FORBIDDEN})
    m.io.save(root/'contract.json',c)
    b['baseline_csv']=csvrows
    return root,c,inp,dict(packet=packet,evidence=b)


def execute(tmp_path,monkeypatch,defect=None):
    root,c,inp,b=setup(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'load_bundle',lambda *args:b)
    # This small invented molecule has 2 spatial orbitals; the production
    # whitelist/spin8 packet gate is tested independently with invented H4 shape.
    monkeypatch.setattr(m,'packet_gate',lambda *args:dict(status='ARTIFICIAL_SCHEMA_ONLY'))
    original=subprocess.check_output
    monkeypatch.setattr(m.subprocess,'check_output',lambda command,*a,**kw:
        'ARTIFICIAL_SOURCE\n' if command[:2]==['git','rev-parse'] else original(command,*a,**kw))
    if defect in ('phase','order','time'):
        ordered=m.native.NativeBuilder.ordered
        def wrong(self,*a,**kw):
            if defect=='time':a=(a[0],a[1]*2)
            q=ordered(self,*a,**kw)
            if defect=='phase':q.p(.02,q.num_qubits-1)
            if defect=='order':q.data=list(reversed(q.data))
            return q
        monkeypatch.setattr(m.native.NativeBuilder,'ordered',wrong)
    if defect=='Gaussian':
        request=m.fair.basis_request
        def wrong(*a):
            q=request(*a);a[0].basis_reports['XDF:'+a[2].block_id]['single_particle_max_entry']=.01;return q
        monkeypatch.setattr(m.fair,'basis_request',wrong)
    if defect=='compile':
        compile=m.fair.compile_checked
        def wrong(*a):
            q=compile(*a);q.reset(0);return q
        monkeypatch.setattr(m.fair,'compile_checked',wrong)
    if defect in ('unknown','infeasible'):
        assess=m.optimistic_round
        def unresolved(t,row,target):
            return assess(t,dict(row,pf_operator_error=None if defect=='unknown' else .4),target)
        monkeypatch.setattr(m,'optimistic_round',unresolved)
    if defect=='expensive':
        from trottertracks.pr_dictionary_validation import f2_execution
        monkeypatch.setattr(f2_execution,'circuit_metrics',lambda *a:metrics(10**8))
    out=root/m.FAMILY/'2026-10-26/run01'
    return root,c,inp,b,out


@pytest.mark.parametrize('case',[None,'expensive','unknown','infeasible'])
def test_artificial_e2e_two_kernels_83_calls_no_full_operator_or_energy(tmp_path,monkeypatch,case):
    root,c,inp,b,out=execute(tmp_path,monkeypatch,case)
    forbidden=AssertionError('NO MOLECULAR PF/DF/FIT/ENERGY/FULL OPERATOR')
    with patch.object(m.pf,'run',side_effect=forbidden),patch.object(m.fair,'decode',side_effect=forbidden), \
         patch.object(m.fair.stage,'physical_sector',side_effect=forbidden),patch.object(m.accuracy,'assess_energy',side_effect=forbidden):
        result=m.run(root,root/'contract.json',inp,out)
    stop=m.io.read(out/'STOP.json')
    assert not stop['failed'] and stop['scientific_units']==m.MAX_CALLS==83 and stop['call_counts']==m.LIMITS
    assert stop['retry']==stop['resume']==0
    assert len(list((out/'circuits').glob('*.qpy')))==4
    assert len(list((out/'candidates').glob('*_m*.json')))==26
    assert result['CA_achieved'] is None and not result['full_operator_correctness']
    for s,value in result['candidates'].items():
        assert len(value['rounds'])==13 and value['actual_synthesis_error'] is None
        if case=='unknown':assert value['status']=='UNRESOLVED'
        elif case=='infeasible':assert value['status']==value['cost']['status']=='INFEASIBLE_OPTIMISTIC_PRESCRIPTION'
        else:
            assert (value['comparisons']['XDF']['strictly_beating_excluded']) == (case=='expensive')
            assert value['cost']['aggregation']['synthesis_signal_error'] is None
            assert value['cost']['analytic_prescription_lower_bound']
            assert value['cost']['rigorous_numerical_lower_bound'] is None
            assert not value['cost']['aggregation']['guaranteed_total_resource_superiority']
    if case is None:
        # Independent CAR reference, all32 input/output components, artificial only.
        p=b['packet'];car=m.fair.stage.car_bilinears(4)
        generators=[float(p['nuclear_repulsion'])*np.eye(16)+m.native.xdf_car_generator(p['df_one_body'],None,car)]
        generators += [m.native.xdf_car_generator(p['df_g_matrices'][i],p['df_lambdas'][i],car) for i in range(9)]
        for s in (1,4):
            target=m.fair.reference._ordered_exponentials(generators,m.native.ordered_schedule(10,Fraction(1,10*s)))
            controlled=m.fair.controlled(target)
            for kind in ('raw','compiled'):
                with (out/'circuits'/f'XDF9_s{s}_{kind}.qpy').open('rb') as f:qc=qpy.load(f)[0]
                np.testing.assert_allclose(Operator(qc).data,controlled,atol=1e-10,rtol=0)
    with pytest.raises(ValueError,match='FRESH'):m.run(root,root/'contract.json',inp,out)


@pytest.mark.parametrize('defect',['phase','order','time','Gaussian','compile'])
def test_structural_correctness_fail_partial_STOP(tmp_path,monkeypatch,defect):
    root,c,inp,b,out=execute(tmp_path,monkeypatch,defect)
    with pytest.raises(ValueError):m.run(root,root/'contract.json',inp,out)
    stop=m.io.read(out/'STOP.json');failure=m.io.read(out/'failure.json')
    assert stop['failed'] and stop['scientific_units']<83 and failure['traceback']
    assert (out/'SOURCE.json').exists() and (out/'partial_candidates.json').exists()
    if defect!='compile':assert stop['call_counts'].get('compile',0)==0
    else:assert failure['workers'] and failure['workers'][0]['traceback']


def test_passive_preflight_no_numeric_import_decode_or_build(tmp_path,monkeypatch):
    root,c,inp,b=setup(tmp_path,monkeypatch)
    with patch.object(np,'load',side_effect=AssertionError('NO NPZ')):
        assert m.preflight(root,root/'contract.json',inp)==c
    subprocess.run([sys.executable,'-c',
        "import sys;sys.path.insert(0,sys.argv[1]);from trottertracks import c1_h4_xdf9_cost_screen;assert not any(x in sys.modules for x in ['numpy','qiskit','scipy','ffsim'])",str(ROOT/'src')],check=True)


@pytest.mark.parametrize('name',['source.txt','fake_packet.npz','PF.json'])
def test_byte_mismatch_stops_before_any_decode(tmp_path,monkeypatch,name):
    root,c,inp,b=setup(tmp_path,monkeypatch);(root/name).write_bytes(b'ARTIFICIAL_DRIFT')
    with patch.object(np,'load',side_effect=AssertionError('NO DECODE')):
        with pytest.raises(ValueError):m.preflight(root,root/'contract.json',inp)


@pytest.mark.parametrize('defect',['phase_version','rank','frame','baseline','shell','source','PF'])
def test_saved_metadata_identity_rejected(tmp_path,monkeypatch,defect):
    root,c,inp,b=setup(tmp_path,monkeypatch);e=b['evidence']
    if defect=='phase_version':e['fairness']['compiler_phase']='changed'
    if defect=='rank':e['PF']['saved_DF']['rank']=8
    if defect=='frame':e['PF_contract']['conditions']['rank']=10
    if defect=='baseline':e['resources']['candidates']['XDF:s1']['cost']['totals']['rz_count']+=1
    if defect=='shell':e['native']['shell_metrics']['head']['measurement_count']=1
    if defect=='source':e['PF_SOURCE']['execution_commit']='OTHER'
    if defect=='PF':e['PF']['candidates']['1']['rounds'][0]['target']='XDF8'
    with pytest.raises(ValueError):m.saved_identity(b,c)


def test_fixed_formula_monotonicity_including_ceil_and_phase_rejection():
    t=m.accuracy.AccuracyTarget(**m.TARGET)
    for pferror in (0.,1e-12,.001,.07,.2):
        row=dict(m=0,pf_operator_error=pferror,attenuation=1.,rte_corrected_operator_error=0.)
        optimistic=m.optimistic_round(t,row,'ARTIFICIAL')
        n0=int(optimistic['axes']['X']['optimistic_shots_decimal'])
        observed=[]
        for extra in (0.,1e-18,1e-12,1e-8,.001,.01,.05,.2):
            evidence=m.accuracy.RoundEvidence(m=0,attenuation=1.,pf_operator_error=pferror,
                rte_corrected_operator_error=0.,synthesis_signal_error=extra,finite_law_signal_error=0.,
                target_id='ARTIFICIAL',state_model='own_DF_ground_state_oracle',alias_branch_verified=True,
                iid_per_shot=True,law_model='deterministic')
            value=m.accuracy.assess_round(t,evidence)
            if value['status']=='CONDITIONAL_ROUND_PASS':
                assert value['shots']>=n0;observed.append(value['shots'])
        assert observed==sorted(observed)
        assert optimistic['actual_synthesis_signal_error'] is None
    # Probe both neighbours of a floating ceil boundary, not only coarse points.
    solver=m.accuracy._legacy('rpe_resource_accounting')._hoeffding_shots
    x=math.sqrt(2*math.log(2/t.alpha_axis)/512)
    coords=[math.nextafter(x,0.),x,math.nextafter(x,math.inf)]
    ns=[solver(c,t.alpha_axis) for c in coords]
    assert ns==sorted(ns,reverse=True) and ns[0]>=512>=ns[-1]


def test_integer_serial_aggregation_and_comparison_equality_no_CA_bypass():
    s=4;key=m.integrated.Candidate('XDF9','D',s).id
    shells={k:{z:n for z in m.accuracy.METRICS} for k,n in [('head',2),('X',3),('Y',5)]}
    kernel={k:10 for k in m.accuracy.METRICS}
    rounds=[dict(m=i,status='OPTIMISTIC_FORMULA_ESTIMATE',axes={a:dict(optimistic_shots_decimal=str(i+1)) for a in ('X','Y')}) for i in range(13)]
    costs=[row for i in range(13) for row in m.serial_cost_pair(kernel,shells,s,i,key)]
    with patch.object(m.accuracy,'aggregate_totals',side_effect=AssertionError('DO NOT BYPASS CA GATE')):
        result=m.aggregate_screen(s,rounds,costs)
    expected=sum((i+1)*(2*(4*(1<<i)*10)+2*2+3+5) for i in range(13))
    assert int(result['optimistic_total_RZ_decimal'])==expected
    assert m.compare_screen(result,expected,'XDF8')['strictly_beating_excluded'] is True
    assert m.compare_screen(result,expected+1,'XDF8')['status']=='COMPETITIVENESS_NOT_EXCLUDED'
    assert m.compare_screen(dict(optimistic_total_RZ_decimal=None),expected,'CDF8')['status']=='UNRESOLVED'
    assert result['aggregation']['CA_achieved'] is None
    huge={k:10**16+1 for k in m.accuracy.METRICS}
    bigcosts=[row for i in range(13) for row in m.serial_cost_pair(huge,shells,s,i,key)]
    big=m.aggregate_screen(s,rounds,bigcosts)
    integer=sum((i+1)*(2*(s*(1<<i)*(10**16+1))+12) for i in range(13))
    assert int(big['optimistic_total_RZ_decimal'])==integer>2**53
    assert m.compare_screen(big,integer+1,'ARTIFICIAL')['strictly_beating_excluded'] is False


def test_whitelist_loader_artificial_records_only(tmp_path,monkeypatch):
    root,c,inp,b=setup(tmp_path,monkeypatch)
    with patch.object(m.io,'load_input',return_value=b['packet']) as load:
        got=m.load_bundle(root,c,inp)
    load.assert_called_once_with(inp,evaluation=True)
    assert got['evidence']['PF']['status']=='XDF9_STAGE1_COMPLETE'
    assert not any(x.flags.writeable for x in got['packet'].values())


def test_real_packet_gate_with_invented_H4_shape(tmp_path,monkeypatch):
    # Reuse the existing H4-shaped invented fixture; no saved NPZ is accessed.
    spec=importlib.util.spec_from_file_location('xdf9_artificial_tests',ROOT/'tests/management/test_research_c_c1_h4_xdf9_pf.py')
    test=importlib.util.module_from_spec(spec);spec.loader.exec_module(test)
    root,c,inp=test.fixture(tmp_path,monkeypatch);packet=m.io.load_input(inp,evaluation=True)
    assert m.packet_gate(packet)['rank']==9
    packet['df_g_matrices']=packet['df_g_matrices'].copy();packet['df_g_matrices'][8,0,1]=1e-20
    with pytest.raises(ValueError,match='SPIN_LIFT'):m.packet_gate(packet)


def test_overflow_unknown_and_nonfinite_are_not_clipped_or_zero_filled():
    t=m.accuracy.AccuracyTarget(**m.TARGET)
    row=dict(m=0,pf_operator_error=.01,attenuation=1.,rte_corrected_operator_error=0.)
    with patch.object(m.accuracy,'assess_round',side_effect=OverflowError):
        value=m.optimistic_round(t,row,'ARTIFICIAL')
    assert value['status']=='UNRESOLVED_SHOT_RANGE' and value['axes'] is None
    assert m.optimistic_round(t,dict(row,pf_operator_error=None),'ARTIFICIAL')['status']=='UNRESOLVED_PF'
    with pytest.raises(ValueError):m.optimistic_round(t,dict(row,pf_operator_error=float('nan')),'ARTIFICIAL')
    with pytest.raises(ValueError):m.optimistic_round(t,dict(row,attenuation=.9),'ARTIFICIAL')
    with patch.object(m.cost,'aggregate_optimistic',side_effect=OverflowError):
        got=m.aggregate_screen(1,[dict(status='OPTIMISTIC_FORMULA_ESTIMATE')]*13,[])
    assert got['status']=='UNRESOLVED_COST_RANGE' and got['optimistic_total_RZ_decimal'] is None


def test_structural_gate_does_not_claim_compile_operator_phase_equivalence():
    # A finite, unitary compiled global-phase change passes the deliberately
    # limited structural check; it is caught by a full artificial reference.
    from qiskit import QuantumCircuit
    q=QuantumCircuit(2);q.rz(.2,0);q.cx(0,1)
    changed=q.copy();changed.global_phase+=.03
    assert m.circuit_structure(changed,1,compiled=True)==m.circuit_structure(q,1,compiled=True)
    assert np.max(abs(Operator(changed).data-Operator(q).data))>.02


def test_gate_name_alone_does_not_qualify_unitary_structure():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate
    q=QuantumCircuit(2);q.append(Gate('rz',1,[.2]),[0])
    with pytest.raises(ValueError,match='GATE_CLASS'):m.circuit_structure(q,1,compiled=True)
