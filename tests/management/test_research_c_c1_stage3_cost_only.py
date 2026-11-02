"""Artificial/mock cost-only validation. No saved H4 values decoded."""
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock,patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_stage3_cost_only as c
from trottertracks import c1_stage3_reference as reference
from trottertracks import c1_stage3_shots_screen as shots


def proposal():
    p=json.loads((ROOT/'docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json').read_text())['proposal']
    return {k:p[k] for k in c.PROPOSAL_KEYS}


def saved(candidate):
    return dict(id=candidate.id,candidate=c.asdict(candidate),status='OPTIMISTIC_FORMULA_SCREENED',
        prior_feasibility='POTENTIALLY_FEASIBLE',CA_achieved=None,
        actual_synthesis_signal_error=None,actual_law_signal_error=None,
        rounds=[dict(m=m,status='OPTIMISTIC_FORMULA_ESTIMATE',CA_achieved=None,
            guaranteed_actual_shots=None,actual_synthesis_signal_error=None,actual_law_signal_error=None,
            axes={a:dict(optimistic_shots_decimal=str(m+3)) for a in ('X','Y')}) for m in range(13)])


def calibration(candidate,qualified=False):
    # Unequal sample costs: the expectation must retain both, not repeat one.
    samples=[{k:v for k in c.accuracy.METRICS} for v in (2.,10.)]
    bound=c.qualification.support_cost_envelope(samples,complete_support=qualified)
    kernel=dict(target_id=candidate.id,policy='native_kernel_repeat_with_fences',
        mean={k:6. for k in c.accuracy.METRICS},lower=bound['lower'],upper=bound['upper'],
        simultaneous_interval_qualified=qualified,classical_cost_samples=2,support_qualification=bound)
    shells={a:dict(axis=a,policy='separate_native_readout',mean={k:1. for k in c.accuracy.METRICS},
        lower={k:1. for k in c.accuracy.METRICS},upper={k:1. for k in c.accuracy.METRICS},
        simultaneous_interval_qualified=True) for a in ('X','Y')}
    return dict(kernel=kernel,shells=shells,whole_support_covered=qualified)


class ArtificialBackend:
    def __init__(self,qualified=False,fail=False):
        self.shots={x.id:saved(x) for x in c.candidates()};self.qualified=qualified;self.fail=fail
    def prepare(self,call):
        self.call=call
        for kind in ('input_load','fit_export_load','model_prepare','saved_shots_load','saved_stage2_load','split_native_model','gaussian_build'):
            for j in range(c.call_limits()[kind]):call(kind,'ARTIFICIAL:'+str(j),lambda:None)
    def cost_kernel(self,candidate):
        if candidate.family=='PR':
            for kind in ('rte_configuration','cost_support_probe'):self.call(kind,candidate.id,lambda:None)
        return None
    def calibrate(self,candidate,kernel,call):
        if self.fail and candidate==c.candidates()[1]:raise ValueError('ARTIFICIAL_COMPILE_FAILURE')
        segments=32 if candidate.family=='PR' else 0 if candidate.s==1 else 1
        # Account shared shells in the first candidate for the exact max audit.
        if candidate==c.candidates()[0]:segments+=3
        for j in range(segments):
            for kind in ('circuit_build','compile','compile_equivalence_check','metrics'):
                call(kind,candidate.id+':'+str(j),lambda:None)
        if candidate.family=='PR':
            for j in range(32):call('cost_kernel_sample',candidate.id+':'+str(j),lambda:None)
            call('cost_interval_assessment',candidate.id,lambda:None)
        return calibration(candidate,self.qualified)


def test_exact_eight_subset_and_pair_scope():
    assert len(c.candidates())==8 and len(c.comparison_pairs())==12
    assert [(x.family,x.s,x.r,x.K) for x in c.candidates()[:4]]==[
        ('D',1,0,0),('D',4,0,0),('PR',4,16,0),('PR',4,16,2)]
    assert sum(c.call_limits().values())==936


@pytest.mark.parametrize('qualified',[False,True])
def test_e2e_saved_shots_reuse_936_calls_all_scope_limits_and_no_accuracy_claim(tmp_path,qualified):
    forbidden=Mock(side_effect=AssertionError('Not cost-only science'))
    with patch.object(shots,'assess_optimistic_round',forbidden),patch.object(c.accuracy,'assess_round',forbidden),\
        patch.object(c.accuracy,'assess_energy',forbidden),patch.object(c.accuracy,'aggregate_totals',forbidden),\
        patch.object(reference,'reference_round',forbidden):
        r=c.run(ArtificialBackend(qualified),proposal(),tmp_path/'out',workers=4)
    forbidden.assert_not_called()
    assert r['scientific_units']==936 and r['call_counts']==c.call_limits()
    assert len(r['candidates'])==8 and len(r['comparisons'])==12 and len(r['out_of_scope_candidates'])==12
    assert r['CA_achieved'] is r['synthesis_signal_error'] is r['actual_law_signal_error'] is None
    assert not r['guaranteed_total_resource_superiority'] and r['shots_formula_recomputed']==0
    for candidate,total in zip(c.candidates(),r['candidates']):
        expected=sum(2*(m+3)*(candidate.s*(1<<m)*6+1) for m in range(13))
        assert total['point_totals']['rz_count']==expected
        assert total['point_totals_rational']['rz_count']==str(expected)
        assert total['whole_law_cost_support_qualified']==qualified
        if not qualified:assert total['support_ranges'] is None
        assert total['independent_per_quantum_occurrence'] and total['calibration_is_not_repeated_quantum_trajectory']
        raw=json.loads((tmp_path/'out/candidates'/(candidate.id+'.json')).read_text())
        assert len(raw['rounds'])==26
        assert raw['rounds'][-1]['kernel_occurrences']==candidate.s*4096
    if not qualified:assert all(x['cost_relation']['verdict']=='POINT_ESTIMATES_ONLY' for x in r['comparisons'])
    assert not json.loads((tmp_path/'out/STOP.json').read_text())['failed']


def test_nonoverlapping_complete_support_only_supports_conditional_cost_relation():
    x=dict(id='L',point_totals=dict(rz_count=3.),support_ranges=dict(rz_count=[2.,4.]),cost_scope='same')
    y=dict(id='R',point_totals=dict(rz_count=9.),support_ranges=dict(rz_count=[8.,10.]),cost_scope='same')
    result=c.compare_costs(x,y)
    assert result['cost_relation']['verdict']=='LEFT_LOWER_CONDITIONAL_COST_AT_OPTIMISTIC_SHOTS'
    assert not result['guaranteed_total_resource_superiority'] and result['CA_achieved'] is None
    x['support_ranges']=None
    assert c.compare_costs(x,y)['cost_relation']['verdict']=='POINT_ESTIMATES_ONLY'


def test_error_preserves_partial_metrics_and_stops_no_retry(tmp_path):
    with pytest.raises(ValueError,match='ARTIFICIAL_COMPILE_FAILURE'):
        c.run(ArtificialBackend(fail=True),proposal(),tmp_path/'out',workers=1)
    stop=json.loads((tmp_path/'out/STOP.json').read_text())
    assert stop['failed'] and stop['retry']==stop['resume']==0
    assert (tmp_path/'out/candidates'/(c.candidates()[0].id+'.json')).exists()
    with pytest.raises(FileExistsError):c.run(ArtificialBackend(),proposal(),tmp_path/'out')


def test_aggregate_rejects_wrong_scope_missing_axes_or_invalid_cost():
    candidate=c.candidates()[0];cal=calibration(candidate)
    rows=[c.connection.compose_serial_cost(cal['kernel'],cal['shells'][axis],m=m,substeps=1,target_id=candidate.id)
          for m in range(13) for axis in ('X','Y')]
    with pytest.raises(ValueError):c.aggregate_optimistic(candidate,saved(candidate),rows[:-1])
    rows[0]['mean']['rz_count']=float('nan')
    with pytest.raises(ValueError,match='INVALID_COST'):c.aggregate_optimistic(candidate,saved(candidate),rows)
    rows[0]['mean']['rz_count']=7.;rows[0]['kernel_occurrences']=2
    with pytest.raises(ValueError,match='ROUND_COST_SCOPE'):c.aggregate_optimistic(candidate,saved(candidate),rows)


def test_saved_arbitrary_integer_shots_weight_cost_without_float_roundtrip():
    candidate=c.candidates()[0];cal=calibration(candidate);evidence=saved(candidate)
    evidence['rounds'][0]['axes']['X']['optimistic_shots_decimal']=str(10**40+17)
    rows=[c.connection.compose_serial_cost(cal['kernel'],cal['shells'][axis],m=m,substeps=1,target_id=candidate.id)
          for m in range(13) for axis in ('X','Y')]
    total=c.aggregate_optimistic(candidate,evidence,rows)
    exact=sum(int(evidence['rounds'][r['m']]['axes'][r['axis']]['optimistic_shots_decimal'])*Fraction(r['mean']['rz_count']) for r in rows)
    assert Fraction(total['point_totals_rational']['rz_count'])==exact
    assert not total['whole_law_cost_support_qualified']


def test_independent_sequence_uses_a_new_seed_and_factory_for_every_occurrence():
    from qiskit import QuantumCircuit
    from trottertracks import c1_stage3_native_bridge as bridge
    candidate=c.candidates()[2];kernel=dict(tail='tail',distribution='law',plan='plan',h=Fraction(1,40),config='config')
    seeds=[];factory_events=[]
    def sample(tail,law,count,seed):seeds.append(seed);return (seed,)
    def build(plan,h,events,**kw):factory_events.append(events);q=QuantumCircuit(2);q.rz((events[0]%17)/10,0);return q
    with patch.object(c.connection,'sample_occurrence',side_effect=sample),patch.object(bridge,'partial_kernel',side_effect=build):
        qc=c.independent_sequence(candidate,kernel,master=101,m=1,axis='X',shot=3,modes=1)
    assert len(seeds)==len(set(seeds))==len(factory_events)==8
    assert qc.count_ops()['rz']==8 and qc.count_ops()['barrier']==9
    assert seeds[0]!=c.connection.trajectory_seed(101,candidate.arm,candidate.id,1,'Y',3,0)


def test_real_compiler_check_detects_relative_phase_and_never_calls_molecular_target(tmp_path):
    from qiskit import QuantumCircuit
    from trottertracks.pr_dictionary_validation import gates
    p=proposal();backend=c.CostBackend(dict(root=tmp_path,contract=dict(proposal=p)),tmp_path)
    backend.p=p;backend.tol=p['correctness_tolerance'];backend.call=lambda kind,label,fn:fn()
    qc=QuantumCircuit(2);qc.cp(-.17,1,0)
    forbidden=Mock(side_effect=AssertionError('No CAR/PF/RTE reference'))
    r=backend._validate_compile('artificial',qc,forbidden,controlled=True,arm='CDF')
    assert r['metrics']['rz_count']>0 and r['diagnostics']['compiled']['max_entry']<1e-10
    real_compile=gates.compiled_wrapper
    def bad(raw,profile):q=real_compile(raw,profile);q.rz(.2,1);return q
    with patch.object(gates,'compiled_wrapper',side_effect=bad):
        with pytest.raises(ValueError):backend._validate_compile('bad',qc,forbidden,controlled=True,arm='CDF')
    forbidden.assert_not_called()


def write_artificial_shots(folder):
    (folder/'candidates').mkdir(parents=True)
    records=[saved(x) for x in c.candidates()]
    selected={x['id']:x for x in records}
    full=[{k:v for k,v in selected[x.id].items() if k!='rounds'} if x.id in selected else dict(id=x.id,status='NOT_SELECTED') for x in c.integrated.candidates()]
    for x in records:(folder/'candidates'/(x['id']+'.json')).write_text(json.dumps(x))
    (folder/'result.json').write_text(json.dumps(dict(status='COMPLETE_OPTIMISTIC_FORMULA_SCREEN',scientific_units=282,candidates=full)))
    (folder/'STOP.json').write_text(json.dumps(dict(failed=False,mandatory_stop=True)))


def test_loader_reads_artificial_shots_only_preserves_twenty_and_rejects_false_qualification(tmp_path):
    folder=tmp_path/'saved';write_artificial_shots(folder)
    v=dict(root=tmp_path,contract=dict(saved_shots_run='saved'))
    assert len(c.load_shots(v))==8
    path=folder/'candidates'/(c.candidates()[0].id+'.json');r=json.loads(path.read_text())
    r['rounds'][0]['actual_synthesis_signal_error']=0.;path.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='SAVED_SHOTS_QUALIFICATION'):c.load_shots(v)


def test_real_value_prepare_and_old_calibrator_on_artificial_data_no_CAR_PF_shot_recompute(tmp_path):
    from trottertracks import c1_stage1 as stage1
    from trottertracks import c1_stage3_molecular as molecular
    p=proposal();folder=tmp_path/'shots';write_artificial_shots(folder)
    inp=tmp_path/'input.npz';np.savez(inp,df_one_body=np.diag([.2,.2]).astype(complex),
        df_lambdas=np.full(10,.01),df_g_matrices=np.tile(np.eye(2),(10,1,1)).astype(complex),nuclear_repulsion=np.array(.375))
    (tmp_path/'fit').mkdir();(tmp_path/'s2.json').write_text(json.dumps(dict(status='STAGE2_LIMITED_CIRCUIT_VALIDATION_PASS',
        metrics={arm+'::shared':{k:3 for k in c.accuracy.METRICS} for arm in ('CDF','XDF')})))
    contract=dict(proposal=p,saved_shots_run='shots',stage2_result='s2.json',stage2_source='ARTIFICIAL',
        saved_native_kernels={arm:dict(raw='unused',compiled='unused') for arm in ('CDF','XDF')},
        physical_conditions=dict(system_qubits=2,CDF_rank=8,XDF_rank=10),saved_fit_export='fit/fit_export.npz',
        saved_fit_export_sha256='ARTIFICIAL',input_sha256='ARTIFICIAL')
    arrays=dict(kappa=np.array([[.2]]),diag_coulomb_mats=np.full((8,1,1),.01),orbital_rotations=np.ones((8,1,1)),constant=np.array(.375))
    metadata=dict(actual_rank=8,frame_ids=[f'F{i}' for i in range(8)],group_ids=[f'G{i}' for i in range(8)])
    backend=c.CostBackend(dict(root=tmp_path,input=inp,contract=contract),tmp_path/'out')
    forbidden=Mock(side_effect=AssertionError('Forbidden scientific re-evaluation'))
    real_load=np.load;seen=[];real_sample=c.connection.sample_occurrence;seeds=[]
    def load(path,*args,**kwargs):assert Path(path).is_relative_to(tmp_path);seen.append(path);return real_load(path,*args,**kwargs)
    def sample(tail,law,n,seed):seeds.append(seed);return real_sample(tail,law,n,seed)
    with patch.object(c.io,'load_export',return_value=(arrays,metadata)),patch.object(np,'load',side_effect=load),\
        patch.object(c.connection,'sample_occurrence',side_effect=sample),patch.object(stage1,'car_bilinears',forbidden),\
        patch.object(molecular.MolecularBackend,'prepare',forbidden),patch.object(backend,'_trajectory_reference',forbidden),\
        patch.object(reference,'reference_kernels',forbidden),patch.object(shots,'assess_optimistic_round',forbidden),\
        patch.object(c.accuracy,'assess_round',forbidden),patch.object(c.accuracy,'aggregate_totals',forbidden):
        result=c.run(backend,p,tmp_path/'out',workers=4)
    forbidden.assert_not_called()
    assert len(seen)==1 and len(seeds)==len(set(seeds))==128
    assert len(result['candidates'])==8 and result['scientific_units']<=936
    assert not backend.states and not backend.generators and not backend.atom_operators and not backend.exact
    assert result['call_counts']['rte_configuration']==4 and result['call_counts']['serial_cost_expansion']==208
    assert all(x['CA_achieved'] is None for x in result['candidates'])


def test_passive_import_cli_and_preflight_with_undecodable_inputs(tmp_path):
    subprocess.run([sys.executable,'-c',"import sys;sys.path.insert(0,'src');from trottertracks import c1_stage3_cost_only;assert not any(k in sys.modules for k in ['numpy','qiskit','scipy','ffsim'])"],cwd=ROOT,check=True)
    p=subprocess.run([sys.executable,str(ROOT/'scripts/run_research_c_c1_stage3_cost_only.py'),'--mode','full','--contract','/NEVER','--input','/NEVER'],capture_output=True,text=True)
    assert p.returncode==2
    parent=json.loads((ROOT/'docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json').read_text())
    (tmp_path/'parent').write_text(json.dumps(parent));(tmp_path/'source').write_text(json.dumps(dict(files={})))
    (tmp_path/'input').write_bytes(b'not NPZ');(tmp_path/'saved').write_bytes(b'not JSON')
    contract=dict(scope=c.SCOPE,candidate_ids=[x.id for x in c.candidates()],call_limits=c.call_limits(),
        maximum_logical_scientific_units=936,new_fit=0,retry=0,resume=0,parent_contract='parent',
        parent_contract_sha256=c.io.sha(tmp_path/'parent'),proposal=proposal(),physical_conditions=parent['physical_conditions'],
        input_sha256=c.io.sha(tmp_path/'input'),source_manifest='source',source_manifest_sha256=c.io.sha(tmp_path/'source'),
        saved_inputs={'saved':c.io.sha(tmp_path/'saved')},runtime_required_packages={})
    path=tmp_path/'contract';path.write_text(json.dumps(contract))
    with patch.object(np,'load',side_effect=AssertionError('No numeric decode')):c.preflight(tmp_path,path,tmp_path/'input')
    contract['proposal']['compiler']['optimization_level']=3;path.write_text(json.dumps(contract))
    with pytest.raises(ValueError,match='CHANGED_COST_SCIENTIFIC_CONDITIONS'):c.preflight(tmp_path,path,tmp_path/'input')


def test_execute_dispatch_provenance_caches_and_fresh_output_with_artificial_backend(tmp_path):
    v=dict(root=tmp_path,contract=dict(proposal=proposal()),source=dict(files={}),runtime=dict(artificial=True))
    output=tmp_path/c.FAMILY/'ARTIFICIAL'
    with patch.object(c,'preflight',return_value=v),patch.object(c,'CostBackend',return_value=ArtificialBackend()),\
            patch.dict(os.environ,os.environ.copy()),patch.object(sys,'pycache_prefix',sys.pycache_prefix):
        result=c.execute(tmp_path,tmp_path/'contract',tmp_path/'input',output)
        assert result['scientific_units']==936 and (output/'contract.json').exists()
        assert Path(os.environ['NUMBA_CACHE_DIR']).is_relative_to(tmp_path/'.cache')
        with pytest.raises(ValueError,match='FRESH_COST_OUTPUT'):c.execute(tmp_path,tmp_path/'contract',tmp_path/'input',output)
