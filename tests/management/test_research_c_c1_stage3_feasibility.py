"""Artificial feasibility-only tests; no saved molecular arrays are decoded."""
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_stage3_feasibility as f
from trottertracks import c1_stage3_reference as ref
from trottertracks import c1_stage3_integrated as old
from trottertracks import c1_stage3_accuracy as accuracy


def proposal():
    return f.inherited_parameters(json.loads((ROOT/'docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json').read_text()))


class ArtificialBackend:
    def __init__(self,mode='potential'):
        self.mode = mode; self.df_errors = dict(CDF=0.,XDF=0.)
        self.calibrate = self.qualify = self.virtual = Mock(side_effect=AssertionError('Excluded full-mode operation'))

    def prepare(self,call):
        self.call = call
        excluded = {'rte_configuration','kernel_reference','round_reference',
                    'feasibility_assessment','candidate_feasibility_summary'}
        # Fake the exact registered preparation shape, with zero real-data work.
        for kind,count in f.call_limits().items():
            if kind not in excluded:
                for i in range(count):call(kind,f'ARTIFICIAL:{i}',lambda:None)

    def prepared_summary(self):return dict(artificial=True,new_fit=0)

    def kernel(self,c):
        if c.family=='PR': self.call('rte_configuration',c.id,lambda:None)
        return ref.reference_kernels((np.diag([0.,.25]),np.zeros((2,2))),Fraction(1,10*c.s),ld=1)

    def reference(self,c,k,m):
        if self.mode=='fail' and c==old.candidates()[0] and m==3:raise ValueError('ARTIFICIAL_REFERENCE_FAIL')
        row = ref.reference_round(k,(np.diag([0.,.25]),Fraction(1,10)*(1<<m)),repetitions=c.s*(1<<m))
        if self.mode=='infeasible' or (self.mode=='mixed' and c.family=='PR' and c.K==2):
            row['pf_operator_error'] = 1.
        if self.mode=='unresolved' or (self.mode=='mixed' and c.family=='PR' and c.K==0):
            row['rte_corrected_operator_error'] = None
        return row


@pytest.mark.parametrize('mode,counts',[
    ('potential',{'POTENTIALLY_FEASIBLE':20}),('infeasible',{'INFEASIBLE':20}),
    ('unresolved',{'UNRESOLVED':20}),('mixed',{'POTENTIALLY_FEASIBLE':4,'INFEASIBLE':8,'UNRESOLVED':8})])
def test_registered_e2e_classification_and_exact_limited_calls(tmp_path,mode,counts):
    backend = ArtificialBackend(mode);output = tmp_path/mode
    forbidden = Mock(side_effect=AssertionError('Excluded science stage'))
    with patch.object(accuracy,'assess_round',forbidden),patch.object(accuracy,'aggregate_totals',forbidden),\
            patch.object(ref,'sample_rpe_phases',forbidden):
        result = f.run(backend,proposal(),output,workers=2)
    assert result['candidate_status_counts']==counts
    assert result['candidate_ids']==[c.id for c in old.candidates()]
    assert result['scientific_units']==639 and result['call_counts']==f.call_limits()
    assert result['shots'] is result['total_resources'] is result['final_resource_ranking'] is None
    assert result['CA_achieved'] is None and result['native_build']==result['compile']==0
    assert result['new_fit']==result['optimizer']==result['retry']==result['resume']==0
    assert result['mandatory_stop'] and not json.loads((output/'STOP.json').read_text())['failed']
    forbidden.assert_not_called();backend.calibrate.assert_not_called()
    for path in (output/'candidates').glob('*.json'):
        row=json.loads(path.read_text())
        assert len(row['rounds'])==len(row['feasibility'])==13
        assert row['CA_achieved'] is None and row['synthesis_signal_error'] is None
    assert not (output/'circuits').exists()


def test_sufficient_bound_and_binary64_zero_radius_policy_are_unchanged():
    t=accuracy.AccuracyTarget(**proposal()['accuracy_target'])
    for pf,rte,a in ((.01,.02,1.),(.2,.1,.3),(0.,0.,1.),(None,None,0.)):
        row=dict(pf_operator_error=pf,rte_corrected_operator_error=rte,attenuation=a)
        assert f.classify_round(row,t)==old._feasibility(row,t)
    unknown=f.classify_round(dict(pf_operator_error=None,rte_corrected_operator_error=.01,attenuation=1.),t)
    assert unknown['status']=='UNRESOLVED'
    assert f.classify_round(dict(pf_operator_error=1.,rte_corrected_operator_error=None,attenuation=1.),t)['status']=='INFEASIBLE'
    assert f.summarize([unknown],None,t)['status']=='UNRESOLVED'
    assert f.summarize([unknown],t.df_cap*2,t)['status']=='INFEASIBLE'


@pytest.mark.parametrize('updates',[dict(attenuation=float('nan')),dict(attenuation=1.1),
    dict(pf_operator_error=-.1),dict(rte_corrected_operator_error=float('inf')),
    dict(attenuation=0.,pf_operator_error=float('nan'))])
def test_invalid_numeric_results_are_failures_not_silently_unresolved(updates):
    row=dict(pf_operator_error=0.,rte_corrected_operator_error=0.,attenuation=1.)|updates
    with pytest.raises(ValueError): f.classify_round(row,accuracy.AccuracyTarget(**proposal()['accuracy_target']))


def test_first_failure_keeps_numeric_partial_and_fresh_output(tmp_path):
    output=tmp_path/'failed'
    with pytest.raises(ValueError,match='ARTIFICIAL_REFERENCE_FAIL'):
        f.run(ArtificialBackend('fail'),proposal(),output,workers=1)
    stop=(output/'STOP.json').read_bytes()
    assert json.loads(stop)['failed'] and json.loads(stop)['scientific_units']<=639
    partial=[json.loads(x) for x in (output/'candidates'/(old.candidates()[0].id+'.progress.jsonl')).read_text().splitlines()]
    assert [row['m'] for row in partial]==[0,1,2]
    assert json.loads((output/'failure.json').read_text())['first_failure']['kind']=='round_reference'
    with pytest.raises(FileExistsError):f.run(ArtificialBackend(),proposal(),output,workers=1)
    assert (output/'STOP.json').read_bytes()==stop


def test_cli_and_import_are_passive_and_do_not_expose_full_mode():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c1_stage3_feasibility;assert not any(x in sys.modules for x in ['numpy','qiskit','ffsim','scipy'])"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    cli=str(ROOT/'scripts/run_research_c_c1_stage3_feasibility.py')
    for args in (['--execute'],['--mode','full']):
        result=subprocess.run([sys.executable,cli,'--contract','NEVER_READ','--input','NEVER_READ',*args],capture_output=True,text=True)
        assert result.returncode==2


def test_preflight_checks_fixed_parent_conditions_without_decode(tmp_path):
    parent=json.loads((ROOT/'docs/research/research_c_c1_stage3_integrated_contract_unapproved_20261025.json').read_text())
    (tmp_path/'parent.json').write_text(json.dumps(parent))
    toy=tmp_path/'input.npz';toy.write_bytes(b'ARTIFICIAL_BYTE_PREFLIGHT')
    (tmp_path/'SOURCE.json').write_text(json.dumps(dict(files={})))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    c=parent|dict(scope=f.SCOPE,mode=f.MODE,call_limits=f.call_limits(),maximum_logical_scientific_units=639,
        proposal=f.inherited_parameters(parent),
        parent_contract='parent.json',parent_contract_sha256=sha(tmp_path/'parent.json'),
        source_manifest='SOURCE.json',source_manifest_sha256=sha(tmp_path/'SOURCE.json'),
        input_sha256=sha(toy),saved_inputs={},runtime_required_packages={})
    path=tmp_path/'contract.json';path.write_text(json.dumps(c))
    with patch.object(np,'load',side_effect=AssertionError('No decode in preflight')):
        assert f.preflight(tmp_path,path,toy)['contract']['mode']==f.MODE
        c['proposal']['accuracy_target']['beta_total']+=.01
        path.write_text(json.dumps(c))
        with pytest.raises(ValueError,match='CHANGED_SCIENTIFIC_CONDITIONS'):f.preflight(tmp_path,path,toy)


def test_real_numeric_backend_e2e_uses_artificial_saved_arrays_and_never_builds_circuits(tmp_path):
    # Warm library definitions before constructor sentinels; no physical input.
    from trotterlib.df_trotter import ops
    from qiskit import qpy
    import qiskit
    from trottertracks import c1_stage3_molecular as molecular
    p=proposal()
    c=dict(proposal=p,physical_conditions=dict(system_qubits=2,CDF_rank=8,XDF_rank=10,
        sector_dimension=1,Nalpha=1,Nbeta=0),saved_fit_export='fit/fit_export.npz',
        saved_fit_export_sha256='ARTIFICIAL',input_sha256='ARTIFICIAL',
        stage1_operator_directory='operators',stage1_result='stage1.json')
    (tmp_path/'fit').mkdir();(tmp_path/'operators').mkdir()
    inp=tmp_path/'artificial_input.npz'
    np.savez(inp,df_one_body=np.diag([.2,.2]).astype(complex),df_lambdas=np.zeros(10),
        df_g_matrices=np.zeros((10,2,2),complex),nuclear_repulsion=np.array(.375))
    base=np.diag([.375,.575,.575,.775]).astype(complex)
    for name in ['physical_sector','CDF_sector','XDF_sector','raw_base','raw_total']+[f'raw_frame_{i}' for i in range(8)]:
        value=np.array([[.575]],complex) if name.endswith('_sector') else np.zeros((4,4),complex) if 'frame' in name else base
        np.savez(tmp_path/'operators'/(name+'.npz'),matrix=value)
    (tmp_path/'stage1.json').write_text(json.dumps(dict(records=[dict(id='sector_DF_'+arm,error=0.,allowance=1e-12) for arm in ('CDF','XDF')])))
    arrays=dict(kappa=np.array([[.2]]),diag_coulomb_mats=np.zeros((8,1,1)),
        orbital_rotations=np.ones((8,1,1)),constant=np.array(.375))
    metadata=dict(actual_rank=8,frame_ids=[f'F{i}' for i in range(8)],group_ids=[f'G{i}' for i in range(8)])
    output=tmp_path/'output';backend=f.FeasibilityBackend(dict(root=tmp_path,input=inp,contract=c),output)
    real_load=np.load;loaded=[]
    def load(path,*args,**kwargs):
        assert Path(path).is_relative_to(tmp_path);loaded.append(path)
        return real_load(path,*args,**kwargs)
    forbidden=Mock(side_effect=AssertionError('Outside feasibility-only scope'))
    with patch.object(f.io,'load_export',return_value=(arrays,metadata)),patch.object(np,'load',side_effect=load),\
         patch.object(qiskit,'QuantumCircuit',forbidden),patch.object(qiskit,'transpile',forbidden),\
         patch.object(qpy,'load',forbidden),patch.object(f.native,'NativeBuilder',forbidden),\
         patch.object(molecular.MolecularBackend,'prepare',forbidden),\
         patch.object(molecular.MolecularBackend,'_original_atoms',forbidden),\
         patch.object(f.stage1,'fit_once',forbidden),patch.object(accuracy,'assess_round',forbidden),\
         patch.object(accuracy,'aggregate_totals',forbidden),patch.object(f.connection,'sample_occurrence',forbidden):
        result=f.run(backend,p,output,workers=2)
    forbidden.assert_not_called()
    assert result['scientific_units']==639 and result['candidate_status_counts']=={'POTENTIALLY_FEASIBLE':20}
    assert len(loaded)==14 and len(list(output.glob('ground_*.npz')))==3
    assert not (output/'circuits').exists() and not backend.atom_operators and not backend.reused


def test_finite_RTE_kernel_keeps_old_error_definitions_without_cost_catalog(tmp_path):
    from scipy.linalg import expm
    from trotterlib.df_trotter.ops import DiagonalEvolutionPrimitives as D
    from trottertracks import c1_stage3_qualification as qualification
    p=proposal();p['PR_LD']=0
    output=tmp_path/'kernel';output.mkdir()
    backend=f.FeasibilityBackend(dict(root=tmp_path,contract={'proposal':p}),output)
    U=((1.+0j,0j),(0j,1.+0j));V=((math.sqrt(.5)+0j,math.sqrt(.5)+0j),(math.sqrt(.5)+0j,-math.sqrt(.5)+0j))
    model=f.native.NativeModel('ARTIFICIAL',(f.native.NativeBlock('one',V,D(0.,((0,-.1),(1,.1)),())),
        f.native.NativeBlock('F0',U,D(-.15,((0,-.3),),()))),.375,'TOY','TOY')
    base=.375*np.eye(4);base[1,2]=base[2,1]=.1
    frame=np.diag([0.,.3,0.,.3])
    backend.plans={'CDF':f.connection.split_native_model(model,0)}
    backend.generators={'CDF':[base,frame]};backend.full={'CDF':base+frame}
    backend.exact={('CDF',0):expm(-.1j*(base+frame))}
    backend.call=old.Calls(output,f.call_limits(),639)
    forbidden=Mock(side_effect=AssertionError('No cost catalog or signal evaluation'))
    with patch.object(qualification,'exhaustive_trajectories',forbidden),patch.object(f.connection,'sample_occurrence',forbidden):
        c=old.Candidate('CDF','PR',1,4,2);kernel=backend.kernel(c);row=backend.reference(c,kernel,0)
    expected=expm(-.05j*base)@expm(-.1j*frame)@expm(-.05j*base)
    assert np.max(abs(kernel['pf']-expected))<1e-14
    assert row['pf_operator_error']==pytest.approx(float(np.linalg.norm(expected-backend.exact['CDF',0],2)),abs=1e-14)
    assert row['pf_operator_error']>0 and row['rte_corrected_operator_error']>=0 and 0<row['attenuation']<1
    assert kernel['config'].rte_steps==4 and kernel['config'].finite_taylor_order==2
    assert 'raw_signal' not in row
    forbidden.assert_not_called()


def test_execute_dispatches_only_the_limited_backend_and_saves_provenance(tmp_path):
    v=dict(root=tmp_path,contract_path=tmp_path/'contract.json',input=tmp_path/'input.npz',
        contract=dict(proposal=proposal()),source=dict(files={}),runtime=dict(artificial=True))
    output=tmp_path/f.FAMILY/'artificial'
    with patch.object(f,'preflight',return_value=v),patch.object(f,'FeasibilityBackend',return_value=ArtificialBackend()),\
         patch.dict(os.environ,os.environ.copy()),patch.object(sys,'pycache_prefix',sys.pycache_prefix):
        result=f.execute(v,output)
    assert result['scientific_units']==639 and (output/'contract.json').exists()
    assert not list(output.glob('*grant*')) and not list(output.glob('*approval*'))
