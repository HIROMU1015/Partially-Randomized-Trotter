"""Artificial end-to-end tests only; no real molecular data are decoded."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_stage3_integrated as run
from trottertracks import c1_stage3_reference as reference
from trottertracks import c1_stage3_qualification as qualification
from trottertracks import c1_stage3_connection as connection
from trottertracks import c1_stage2_native as native
from trottertracks.c1_stage3_accuracy import METRICS,evaluate_evidence_packet


def proposal():
    # Conditions only; never read scientific result arrays in these tests.
    return json.loads((ROOT/'docs/research/research_c_c1_stage3_contract_unapproved_20261025.json').read_text())['proposal']


def limits():
    c=json.loads((ROOT/'docs/research/research_c_c1_stage3_contract_unapproved_20261025.json').read_text())
    return {k:v['maximum'] for k,v in c['scientific_calls'].items()}


class ArtificialBackend:
    """Two commuting artificial levels and exact mock circuit/cost semantics."""
    def __init__(self,mode='pass'):
        self.mode=mode;self.calibrated=[];self.df_errors={'CDF':0.,'XDF':0.}

    def prepare(self,call):
        self.call=call
        call('input_load','ARTIFICIAL',lambda:None)
        call('fit_export_load','ARTIFICIAL saved export; no fit',lambda:None)

    def prepared_summary(self):return dict(artificial=True,new_fit=0,optimizer=0)

    def kernel(self,c):
        h=Fraction(1,10*c.s)
        H=np.diag([0.,.25])
        return reference.reference_kernels((H,np.zeros_like(H)),h,ld=1)

    def reference(self,c,k,m):
        if self.mode=='fail' and c==run.candidates()[0] and m==3:raise ValueError('ARTIFICIAL_SCIENTIFIC_FAIL')
        r=reference.reference_round(k,(np.diag([0.,.25]),Fraction(1,10)*(1<<m)),
                                   repetitions=(1<<m)*c.s,state=np.array([1.,0.]))
        r.pop('raw_signal');r['raw_signal_parts']=[1.,0.]
        if self.mode=='infeasible' or (self.mode=='mixed' and c.family=='PR' and c.K==2):
            r['pf_operator_error']=1.
        return r

    def calibrate(self,c,k,call):
        self.calibrated.append(c.id)
        mean={key:2. if c.arm=='CDF' else 10. for key in METRICS}
        qualified=self.mode!='unknown_support'
        kernel=dict(target_id=c.id,policy='native_kernel_repeat_with_fences',mean=mean,
            lower=mean if qualified else None,upper=mean if qualified else None,
            simultaneous_interval_qualified=qualified,classical_cost_samples=32)
        shell={key:1. for key in METRICS}
        shells={axis:dict(axis=axis,policy='separate_native_readout',mean=shell,lower=shell,upper=shell,
            simultaneous_interval_qualified=True) for axis in ('X','Y')}
        return dict(kernel=kernel,shells=shells,whole_support_covered=qualified,artificial=True)

    def qualify(self,c,calibration):
        unknown=self.mode=='unresolved' or (self.mode=='mixed' and c.family=='PR')
        return qualification.synthesis_envelope(whole_support_covered=True,
            reference_enclosures=None if unknown else [0.],shell_enclosure=0.)

    def virtual(self,chosen):
        return reference.sample_rpe_phases([1.]*13,[p['shots'] for p in chosen['evaluation']['rounds']],
                                           trials=8,seed=19)


def test_full_registered_artificial_pipeline_reaches_same_accuracy_comparison(tmp_path):
    output=tmp_path/'fresh';backend=ArtificialBackend()
    result=run.run(backend,proposal(),output,limits(),workers=2)
    assert result['candidate_status_counts']=={'PASS':20}
    assert result['comparisons']['D']['verdict']=='CDF_LOWER_CONDITIONAL_INTERROGATION_COST'
    assert result['comparisons']['PR']['verdict']=='CDF_LOWER_CONDITIONAL_INTERROGATION_COST'
    assert result['call_counts']['round_reference']==result['call_counts']['final_round_assessment']==260
    assert result['scientific_units']<=5637 and len(result['virtual'])==4
    assert result['new_fit']==result['retry']==result['resume']==0 and result['mandatory_stop']
    packets=list((output/'candidates').glob('*.json'));assert len(packets)==20
    record=json.loads(packets[0].read_text())
    # The integration reuses the exact old evidence-entry computation once.
    assert record['evaluation']=={k:v for k,v in evaluate_evidence_packet(record['evidence']).items()
        if k!='scientific_target'}
    assert json.loads((output/'STOP.json').read_text())['failed'] is False


@pytest.mark.parametrize('mode,counts',[('infeasible',{'INFEASIBLE':20}),
    ('unresolved',{'UNRESOLVED':20}),('mixed',{'PASS':4,'INFEASIBLE':8,'UNRESOLVED':8})])
def test_ineligible_and_unresolved_do_not_enter_qualified_comparison(tmp_path,mode,counts):
    backend=ArtificialBackend(mode);out=tmp_path/mode
    result=run.run(backend,proposal(),out,limits(),workers=2)
    assert result['candidate_status_counts']==counts
    for file in (out/'candidates').glob('*.json'):
        row=json.loads(file.read_text())
        if row['status']!='PASS':assert row['evaluation']['resources']['totals'] is None
        if row['status']=='INFEASIBLE':assert row['id'] not in backend.calibrated
        if row['status']=='UNRESOLVED':
            assert row['evidence']['rounds'][0]['synthesis_signal_error'] is None
            assert row['calibration']['kernel']['mean'] is not None
    assert result['comparisons']['PR']['verdict']=='INDETERMINATE_MISSING_QUALIFICATION'


def test_unknown_cost_support_yields_only_points_never_superiority(tmp_path):
    result=run.run(ArtificialBackend('unknown_support'),proposal(),tmp_path/'points',limits(),workers=2)
    assert result['candidate_status_counts']=={'PASS':20}
    assert all(x['verdict']=='POINT_ESTIMATES_ONLY' for x in result['comparisons'].values())
    assert not result['unconditional_total_resource_claim']


def test_first_scientific_failure_preserves_partial_evidence_and_stops_dependents(tmp_path):
    out=tmp_path/'fail';backend=ArtificialBackend('fail')
    with pytest.raises(ValueError,match='ARTIFICIAL_SCIENTIFIC_FAIL'):
        run.run(backend,proposal(),out,limits(),workers=1)
    failure=json.loads((out/'failure.json').read_text());stop=json.loads((out/'STOP.json').read_text())
    assert failure['first_failure']['kind']=='round_reference' and stop['failed']
    first=json.loads((out/'candidates'/(run.candidates()[0].id+'.json')).read_text())
    assert first['status']=='FAIL' and not backend.calibrated
    partial=[json.loads(x) for x in (out/'candidates'/(run.candidates()[0].id+'.progress.jsonl')).read_text().splitlines()]
    assert [x['m'] for x in partial]==[0,1,2]
    assert 'round_reference' in failure['call_counts'] and not (out/'result.json').exists()


def test_fresh_output_and_call_budget_do_not_allow_retry_or_overwrite(tmp_path):
    out=tmp_path/'one'
    with pytest.raises(ValueError,match='TOTAL_CALL_LIMIT'):
        run.run(ArtificialBackend(),proposal(),out,limits(),maximum=1,workers=1)
    before=(out/'STOP.json').read_bytes()
    with pytest.raises(FileExistsError):run.run(ArtificialBackend(),proposal(),out,limits(),workers=1)
    assert before==(out/'STOP.json').read_bytes()


def test_import_and_cli_argument_validation_are_passive():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c1_stage3_integrated,c1_stage3_molecular,c1_stage3_qualification;assert not any(x in sys.modules for x in ['numpy','qiskit','ffsim','scipy'])"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_research_c_c1_stage3_integrated.py'),
        '--contract','NEVER_READ','--input','NEVER_READ','--execute'],capture_output=True,text=True)
    assert result.returncode==2 and 'fresh' in result.stderr


def test_preflight_only_reads_bytes_and_detects_source_input_changes(tmp_path):
    c=json.loads((ROOT/'docs/research/research_c_c1_stage3_contract_unapproved_20261025.json').read_text())
    toy=tmp_path/'not_really_an_npz.npz';toy.write_bytes(b'ARTIFICIAL_BYTE_PREFLIGHT')
    source=tmp_path/'SOURCE.json';source.write_text(json.dumps({'files':{'not_really_an_npz.npz':hashlib.sha256(toy.read_bytes()).hexdigest()}}))
    c.update(scope='C1_STAGE3_INTEGRATED_CA_V1',candidate_ids=[x.id for x in run.candidates()],
             source_manifest='SOURCE.json',source_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             input_sha256=hashlib.sha256(toy.read_bytes()).hexdigest(),saved_inputs={},runtime_required_packages={})
    contract=tmp_path/'contract.json';contract.write_text(json.dumps(c))
    with patch.object(np,'load',side_effect=AssertionError('No NPZ decode in preflight')):
        assert run.preflight(tmp_path,contract,toy)['contract']['scope']=='C1_STAGE3_INTEGRATED_CA_V1'
        toy.write_bytes(b'CHANGED')
        with pytest.raises(ValueError,match='SOURCE'):run.preflight(tmp_path,contract,toy)


def test_all_law_support_envelope_is_exhaustive_not_a_sample_maximum():
    assert qualification.support_size(500,(0,2),16,budget=32)==33
    assert qualification.support_size(2,(0,),4,budget=32)==16
    records=[{k:3 for k in METRICS},{k:7 for k in METRICS}]
    assert not qualification.support_cost_envelope(records,complete_support=False)['simultaneous_interval_qualified']
    bound=qualification.support_cost_envelope(records,complete_support=True)
    assert bound['lower']['rz_count']==3 and bound['upper']['rz_count']==7
    assert qualification.synthesis_envelope(whole_support_covered=True,
        numerical_diagnostic_max=1e-15)['synthesis_error_per_kernel'] is None
    exact=qualification.synthesis_envelope(whole_support_covered=True,reference_enclosures=[0.,.01],shell_enclosure=.001)
    assert exact['synthesis_error_per_kernel']==.01


def test_underflowed_ideal_weight_does_not_disappear_from_the_support():
    tail=SimpleNamespace(components=(object(),))
    distribution=SimpleNamespace(dimensionless_step_time=1e-200,orders=(0,2),order_probabilities=(1.,0.))
    assert qualification.exhaustive_trajectories(tail,distribution,4,budget=32) is None


def test_negative_atom_bridge_matches_independent_artificial_controlled_operator():
    from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives as D
    from qiskit.quantum_info import Operator
    from trottertracks.c1_stage3_native_bridge import partial_kernel
    U=((1.+0j,0j),(0j,1.+0j))
    model=native.NativeModel('ARTIFICIAL',(native.NativeBlock('one',U,D(0.,((0,.5),),())),
        native.NativeBlock('F0',U,D(0.,((0,-.2),),()))),.375,'ARTIFICIAL','ARTIFICIAL')
    plan=connection.split_native_model(model,0)
    tail,config,distribution=connection.configure_rte(plan,'1/10',4,0)
    events=connection.sample_occurrence(tail,distribution,4,17)
    with pytest.raises(ValueError,match='Event angle'):connection.partial_kernel(plan,'1/10',events,rte_config=config)
    actual=Operator(partial_kernel(plan,'1/10',events,rte_config=config)).data
    z=np.array([1.,-1.,1.,-1.]);phase=np.exp(-1j*(.25*z+.375)/10+1j*4*math.atan(.1/40)*z)
    expected=np.diag(np.concatenate((np.ones(4),phase)))
    assert np.max(abs(actual-expected))<1e-14


def test_molecular_loader_and_ground_path_on_saved_artificial_arrays(tmp_path):
    """Exercise the real saved-data backend on 4D artificial matrices only."""
    from qiskit import QuantumCircuit,qpy
    from trottertracks.c1_stage3_molecular import MolecularBackend
    p=proposal();c={'proposal':p,'physical_conditions':dict(system_qubits=2,CDF_rank=8,XDF_rank=10,
        sector_dimension=1,Nalpha=1,Nbeta=0),'saved_fit_export':'fit/fit_export.npz',
        'saved_fit_export_sha256':'ARTIFICIAL','input_sha256':'ARTIFICIAL',
        'stage1_operator_directory':'operators','stage1_result':'stage1.json','stage2_result':'stage2.json',
        'stage2_source':'ARTIFICIAL','saved_native_kernels':{arm:{'raw':arm+'_raw.qpy','compiled':arm+'_compiled.qpy'}
            for arm in ('CDF','XDF')}}
    (tmp_path/'fit').mkdir();(tmp_path/'operators').mkdir()
    input_path=tmp_path/'input.npz'
    np.savez(input_path,df_one_body=np.diag([.2,.2]).astype(complex),df_lambdas=np.zeros(10),
             df_g_matrices=np.zeros((10,2,2),complex),nuclear_repulsion=np.array(.375))
    base=np.diag([.375,.575,.575,.775]).astype(complex)
    for name in ['physical_sector','CDF_sector','XDF_sector','raw_base','raw_total']+[f'raw_frame_{i}' for i in range(8)]:
        value=np.array([[.575]],complex) if name.endswith('_sector') else np.zeros((4,4),complex) if 'frame' in name else base
        np.savez(tmp_path/'operators'/(name+'.npz'),matrix=value)
    (tmp_path/'stage1.json').write_text(json.dumps(dict(records=[dict(id='sector_DF_'+arm,error=0.,allowance=1e-12) for arm in ('CDF','XDF')])))
    (tmp_path/'stage2.json').write_text(json.dumps(dict(metrics={})))
    for files in c['saved_native_kernels'].values():
        for path in files.values():(tmp_path/path).write_bytes(b'ARTIFICIAL_QPY_STUB')
    arrays=dict(kappa=np.array([[.2]]),diag_coulomb_mats=np.zeros((8,1,1)),
                orbital_rotations=np.ones((8,1,1)),constant=np.array(.375))
    metadata=dict(actual_rank=8,frame_ids=[f'F{i}' for i in range(8)],group_ids=[f'G{i}' for i in range(8)])
    output=tmp_path/'output';output.mkdir();calls=run.Calls(output,limits(),5637)
    backend=MolecularBackend(dict(root=tmp_path,input=input_path,contract=c),output)
    (output/'candidates').mkdir()
    actual_load=np.load;loaded=[]
    def load(path,*args,**kwargs):
        assert Path(path).is_relative_to(tmp_path);loaded.append(Path(path))
        return actual_load(path,*args,**kwargs)
    with patch('trottertracks.c1_stage1_execution.load_export',return_value=(arrays,metadata)),\
            patch.object(qpy,'load',side_effect=lambda f:[QuantumCircuit(3)]),\
            patch.object(np,'load',side_effect=load),\
            patch('trottertracks.c1_stage1.fit_once',side_effect=AssertionError('No fit')):
        backend.prepare(calls)
    assert len(loaded)==14 and backend.energies['physical']==pytest.approx(.575)
    assert calls.counts['saved_operator_load']==13 and calls.counts['xdf_full_generator']==11
    assert calls.counts['gaussian_build']==0 and calls.counts['exact_model_evolution']==26
    assert np.max(abs(backend.states['CDF']-np.array([0,1,0,0])))==0
    assert backend.prepared_summary()['sampling_law']['actual_finite_bit_implementation_error'] is None


@pytest.mark.parametrize('route',['PR_sample','PR_exhaustive','D_new','D_reused','FAIL_compiled'])
def test_real_native_cost_and_compiler_path_on_small_artificial_generalK(tmp_path,route):
    from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives as D
    from trottertracks.c1_stage3_molecular import MolecularBackend
    from trottertracks.c1_stage1 import car_bilinears
    p=proposal();p['PR_LD']=0;p['cost_samples_per_PR_candidate']=4
    out=tmp_path/'native';out.mkdir();(out/'candidates').mkdir();calls=run.Calls(out,limits(),5637)
    backend=MolecularBackend(dict(root=tmp_path,contract={'proposal':p}),out)
    backend.call=calls;backend.modes=2;backend.dimension=4;backend.tol=p['correctness_tolerance']
    U=((1.+0j,0j),(0j,1.+0j));V=((.8+0j,-.6+0j),(.6+0j,.8+0j))
    diagonal=D(-.3,((0,-.4),),()) if route=='PR_exhaustive' else D(-.3,((0,-.4),(1,.2)),((0,1,.6),))
    model=native.NativeModel('ARTIFICIAL',(native.NativeBlock('one',U,D(-.12,((0,.2),(1,-.1)),())),
        native.NativeBlock('F0',V,diagonal)),.375,'TOY','TOY')
    backend.models={'CDF':model};backend.car=car_bilinears(2)
    plan=connection.split_native_model(model,0);backend.plans={'CDF':plan}
    backend.atom_operators={'CDF':backend._original_atoms(plan)}
    z0=np.diag([1.,-1.,1.,-1.]);z1=np.diag([1.,1.,-1.,-1.]);eye=np.eye(4)
    base=.495*eye+.1*z0-.05*z1
    frame=.3*eye+sum(float(abs(a.coefficient))*backend.atom_operators['CDF'][a.atom_id] for a in plan.atoms)
    backend.generators={'CDF':[base,frame]};backend.full={'CDF':base+frame}
    _,eigenvectors=np.linalg.eigh(base+frame);backend.states={'CDF':eigenvectors[:,0]}
    backend.builder=native.NativeBuilder()
    for b in model.blocks:backend.builder._basis(model,b)
    if route.startswith('D'):
        candidate=run.Candidate('CDF','D',1 if route=='D_reused' else 4)
    else:candidate=run.Candidate('CDF','PR',1,4,0 if route=='PR_exhaustive' else 2)
    if route=='D_reused':
        backend.saved_metrics={'CDF::shared':{k:7 for k in METRICS}}
        backend.c.update(stage2_source='ARTIFICIAL',saved_native_kernels={'CDF':dict(raw='TOY',compiled='TOY')})
    kernel=backend.kernel(candidate)
    if route=='FAIL_compiled':
        from trottertracks.pr_dictionary_validation.gates import compiled_wrapper as compiler
        def wrong_phase(qc,profile):
            compiled=compiler(qc,profile);compiled.rz(.1,compiled.num_qubits-1)
            return compiled
        with patch('trottertracks.pr_dictionary_validation.gates.compiled_wrapper',side_effect=wrong_phase):
            with pytest.raises(ValueError,match='compiled:shell_head'):backend.calibrate(candidate,kernel,calls)
        assert calls.stop.is_set() and calls.first_failure['kind']=='compiled_reference_check'
        assert calls.counts['compile']==1 and calls.counts['metrics']==0
        assert len(list((out/'circuits').glob('*.qpy')))==2
        return
    if route=='PR_sample':
        # Exercise rare K2 product/sign phases deliberately, rather than hope
        # that four short-time random calibration draws include order2.
        from trotterlib.rte import _make_event
        components=kernel['tail'].components;distribution=kernel['distribution']
        rare=[_make_event((i%3,(i+1)%3,(i+2)%3),components,distribution,1) for i in range(4)]
        simple=_make_event((0,),components,distribution,0)
        draws=[(rare[i],simple,simple,rare[(i+1)%4]) for i in range(4)]
        draws[0]=(rare[0],simple,simple,simple)  # odd order2 count: retain controlled -1 phase
        # The last rare word repeats by component indices; vary one product
        # position to retain four distinct whole trajectories.
        draws[-1]=(simple,rare[0],simple,rare[1])
        with patch.object(connection,'sample_occurrence',side_effect=draws):
            record=backend.calibrate(candidate,kernel,calls)
    else:record=backend.calibrate(candidate,kernel,calls)
    expected_compile={'PR_sample':7,'PR_exhaustive':4,'D_new':4,'D_reused':3}[route]
    expected_xy={'PR_sample':4,'PR_exhaustive':1,'D_new':1,'D_reused':0}[route]
    assert calls.counts['compile']==expected_compile and calls.counts['xy_pair_check']==expected_xy
    assert record['kernel']['mean']['rz_count']>0
    assert record['kernel']['support_qualification']['status']==('UNRESOLVED' if route=='PR_sample' else 'ALL_FINITE_SUPPORT_COVERED')
    assert backend.qualify(candidate,record)['synthesis_error_per_kernel'] is None
    assert len(list((out/'circuits').glob('*.qpy')))==2*expected_compile


def test_execute_connects_preflight_backend_and_provenance_with_artificial_data(tmp_path):
    import os
    from trottertracks import c1_stage3_molecular
    c=dict(proposal=proposal(),scientific_calls={k:dict(maximum=v) for k,v in limits().items()})
    v=dict(root=tmp_path,contract_path=tmp_path/'toy_contract.json',input=tmp_path/'toy_input.npz',
           contract=c,source={'files':{}},runtime={'artificial':True})
    output=tmp_path/'artifacts/research_c_c1_stage3_science/artificial_only'
    with patch.object(run,'preflight',return_value=v) as preflight,\
         patch.object(c1_stage3_molecular,'MolecularBackend',return_value=ArtificialBackend('unresolved')),\
         patch.dict(os.environ,os.environ.copy()),patch.object(sys,'pycache_prefix',sys.pycache_prefix):
        result=run.execute(v,output)
    assert preflight.call_count==1 and result['candidate_status_counts']=={'UNRESOLVED':20}
    assert json.loads((output/'contract.json').read_text())==c and (output/'SOURCE.json').exists()
    assert not list(output.glob('*grant*')) and not list(output.glob('*approval*'))
