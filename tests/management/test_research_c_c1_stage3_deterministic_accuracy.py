"""Artificial matrices/circuits/scalars only; never decode saved H4 values."""
from dataclasses import asdict
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import c1_stage3_deterministic_accuracy as d
from trottertracks import c1_stage3_connection as connection
from trottertracks import c1_stage3_accuracy as accuracy


def proposal():
    return json.loads((ROOT/'docs/research/research_c_c1_stage3_cost_compat_contract_20261025.json').read_text())['proposal']


def bundle():
    t=accuracy.AccuracyTarget(**proposal()['accuracy_target'])
    n=accuracy.assess_round(t,accuracy.RoundEvidence(m=0,attenuation=1.,pf_operator_error=0.,
        rte_corrected_operator_error=0.,synthesis_signal_error=0.,finite_law_signal_error=0.,
        target_id='ARTIFICIAL',state_model='own_DF_ground_state_oracle',alias_branch_verified=True,
        iid_per_shot=True,law_model='deterministic'))['shots']
    records={}
    for c in d.candidates():
        rows=[dict(m=m,attenuation=1.,pf_operator_error=0.,rte_corrected_operator_error=0.) for m in range(13)]
        saved=dict(id=c.id,candidate=asdict(c),status='OPTIMISTIC_FORMULA_SCREENED',
            rounds=[dict(m=m,saved_reference=rows[m],actual_synthesis_signal_error=None,
                actual_law_signal_error=None,CA_achieved=None,
                axes={a:dict(optimistic_shots_decimal=str(n)) for a in ('X','Y')}) for m in range(13)])
        kernel=dict(target_id=c.id,policy='native_kernel_repeat_with_fences',mean={k:2. for k in accuracy.METRICS},lower={k:2. for k in accuracy.METRICS},
            upper={k:2. for k in accuracy.METRICS},simultaneous_interval_qualified=True,classical_cost_samples=1)
        shell=dict(policy='separate_native_readout',mean={k:1. for k in accuracy.METRICS},lower={k:1. for k in accuracy.METRICS},
            upper={k:1. for k in accuracy.METRICS},simultaneous_interval_qualified=True)
        costs=[connection.compose_serial_cost(kernel,dict(shell,axis=a),m=m,substeps=c.s,target_id=c.id)
               for m in range(13) for a in ('X','Y')]
        total=d.cost.aggregate_optimistic(c,saved,costs)
        records[c.id]=dict(feasibility=dict(id=c.id,candidate=asdict(c),status='POTENTIALLY_FEASIBLE',
            rounds=rows,df_error_bound_ha=0.),shots=saved,
            cost=dict(**total,rounds=costs,calibration=dict(whole_support_covered=True,
                diagnostics=[dict(diagnostics=dict(compiled=stored_report()))])))
    return dict(records=records,alias_verified=True,prepared=dict(state_preparation='ORACLE_EXCLUDED',
        df_error_bounds_ha=dict(CDF=0.,XDF=0.)),
        stage2_gates={'compiled:'+arm+'::shared':dict(id='compiled:'+arm+'::shared',**stored_report())
                      for arm in ('CDF','XDF')})


def stored_report():
    return dict(max_entry=0.,frobenius=0.,unitarity=dict(max_entry=0.),rtol=0,tolerance=1e-10)


class ArtificialBackend:
    def __init__(self,mode='pass'): self.bundle=bundle();self.mode=mode
    def prepare(self,call):
        self.call=call
        for kind in ('saved_bundle_load','input_load','saved_operator_load','car_context','xdf_full_generator'):
            for i in range(d.call_limits()[kind]):call(kind,f'ARTIFICIAL:{i}',lambda:None)
        self.shell_reports={}
        for name in ('head','X','Y'):
            for kind in ('qpy_load','operator_decode'):
                for side in ('raw','compiled'):call(kind,name+side,lambda:None)
            self.shell_reports[name]=call('shell_assessment',name,lambda:dict(unitary_distance_pilot=0.,rigorous_enclosure=None))
    def kernel(self,candidate):
        if candidate.s==4:
            self.call('local_kernel_reference',candidate.id,lambda:None)
            for kind in ('qpy_load','operator_decode'):self.call(kind,candidate.id+'raw',lambda:None)
        if self.mode=='fail':raise ValueError('ARTIFICIAL_PHASE_FAIL')
        return dict(unitary_distance_pilot={'pass':0.,'infeasible':.1,'unresolved':None}[self.mode],rigorous_enclosure=None)


@pytest.mark.parametrize('mode,status',[
    ('pass','NUMERICAL_PILOT_CA_PASS'),('infeasible','INFEASIBLE_REGISTERED_SUFFICIENT_CONDITION'),
    ('unresolved','UNRESOLVED')])
def test_artificial_e2e_exact_110_calls_cost_gating_and_no_rigorous_claim(tmp_path,mode,status):
    forbidden=Mock(side_effect=AssertionError('No shot solving or PF round recomputation'))
    backend=ArtificialBackend(mode)
    with patch.object(accuracy,'assess_round',forbidden),patch.object(d.reference,'reference_round',forbidden):
        r=d.run(backend,proposal(),tmp_path/'out',workers=4)
    forbidden.assert_not_called()
    assert r['call_counts']==d.call_limits() and r['scientific_units']==110
    assert r['candidate_status_counts']=={status:4}
    assert len(r['comparisons'])==2 and r['CA_achieved'] is None
    assert r['new_fit']==r['optimizer']==r['retry']==r['resume']==r['circuit_build']==r['compile']==r['shot_count_solved']==0
    for c in r['candidates']:
        assert len(c['rounds'])==13 and c['rigorous_qualification'].startswith('UNRESOLVED')
        assert c['CA_achieved'] is c['guaranteed_actual_shots'] is None
        assert c['primary']==(c['candidate']['s']==1)
        if mode=='pass':assert c['cost']['totals']['rz_count']>0
        else:assert c['cost']['totals'] is None
    stop=json.loads((tmp_path/'out/STOP.json').read_text())
    assert stop['mandatory_stop'] and not stop['failed']


def test_full_relative_phase_norm_and_shell_factor_are_not_max_entry_tolerances():
    tol=proposal()['correctness_tolerance'];eye=np.eye(4,dtype=complex)
    wrong=eye.copy();wrong[2:,2:]*=np.exp(1j*1e-4)
    with pytest.raises(ValueError,match='raw:'):d.numerical_segment(wrong,wrong,eye,tol,'ARTIFICIAL')
    phase=eye*np.exp(1j*1e-12)
    k=d.numerical_segment(eye,phase,eye,tol,'ARTIFICIAL')
    assert k['unitary_distance_pilot']==pytest.approx(np.linalg.norm(phase-eye,'fro'))
    assert k['rigorous_enclosure'] is None
    shells={x:dict(unitary_distance_pilot=1e-13) for x in ('head','X','Y')}
    for n in (1,4,4096,16384):
        r=d.round_synthesis(k,shells,n)
        assert r['complex_signal_error_pilot']==pytest.approx(2*math.sqrt(2)*(n*k['unitary_distance_pilot']+2e-13))
        assert r['rigorous_signal_error_bound'] is None
    # Direct artificial product bound, including noncommuting kernels.
    a=np.diag([1.,np.exp(.03j)]);b=np.array([[math.cos(.02),-math.sin(.02)],[math.sin(.02),math.cos(.02)]])@a
    for n in (1,4,16):
        assert np.linalg.norm(np.linalg.matrix_power(a,n)-np.linalg.matrix_power(b,n),2)<=n*np.linalg.norm(a-b,2)+1e-14


def test_fixed_saved_N_margin_infeasible_unknown_and_underflow_are_explicit():
    b=bundle();c=d.candidates()[0];rec=b['records'][c.id];t=accuracy.AccuracyTarget(**proposal()['accuracy_target'])
    syn=dict(occurrences=1,complex_signal_error_pilot=0.)
    good=d.assess_fixed_round(t,c,rec['feasibility']['rounds'][0],rec['shots']['rounds'][0],syn,alias_verified=True)
    assert good['status']=='CONDITIONAL_ROUND_PASS' and good['fixed_statistical_margin']>=0
    assert d.assess_fixed_round(t,c,rec['feasibility']['rounds'][0],rec['shots']['rounds'][0],
        dict(occurrences=1,complex_signal_error_pilot=.001),alias_verified=True)['status']=='INFEASIBLE_FIXED_SHOTS'
    assert d.assess_fixed_round(t,c,rec['feasibility']['rounds'][0],rec['shots']['rounds'][0],
        dict(occurrences=1,complex_signal_error_pilot=None),alias_verified=True)['status']=='UNRESOLVED_SIGNAL_ERROR'
    for axis in ('X','Y'):rec['shots']['rounds'][0]['axes'][axis]['optimistic_shots_decimal']=str(10**400)
    assert d.assess_fixed_round(t,c,rec['feasibility']['rounds'][0],rec['shots']['rounds'][0],syn,
        alias_verified=True)['status']=='UNRESOLVED_SHOT_RESOLUTION'
    with pytest.raises(ValueError,match='ALIAS'):d.assess_fixed_round(t,c,{}, {},{},alias_verified=False)


def test_failure_preserves_partial_and_no_retry_or_overwrite(tmp_path):
    out=tmp_path/'out'
    with pytest.raises(ValueError,match='ARTIFICIAL_PHASE_FAIL'):d.run(ArtificialBackend('fail'),proposal(),out,workers=1)
    old=(out/'STOP.json').read_bytes();assert json.loads(old)['failed']
    assert (out/'failure.json').exists()
    with pytest.raises(FileExistsError):d.run(ArtificialBackend(),proposal(),out,workers=1)
    assert (out/'STOP.json').read_bytes()==old


def test_fresh_execution_entry_saves_source_and_uses_task_local_caches(tmp_path):
    v=dict(root=tmp_path,contract=dict(proposal=proposal()),source=dict(files={}),runtime=dict(artificial=True))
    backend=ArtificialBackend();out=tmp_path/d.FAMILY/'ARTIFICIAL'
    with patch.object(d,'preflight',return_value=v),patch.object(d,'SavedBackend',return_value=backend),\
            patch.dict(os.environ,os.environ.copy()),patch.object(sys,'pycache_prefix',sys.pycache_prefix):
        r=d.execute(tmp_path,'ARTIFICIAL_CONTRACT','ARTIFICIAL_INPUT',out)
        assert r['scientific_units']==110
        assert Path(sys.pycache_prefix).is_relative_to(tmp_path/'.cache')
        assert Path(os.environ['XDG_CACHE_HOME']).is_relative_to(tmp_path/'.cache')
        assert json.loads((out/'SOURCE.json').read_text())==v['source']
        with pytest.raises(ValueError,match='FRESH_DETERMINISTIC_OUTPUT'):
            d.execute(tmp_path,'ARTIFICIAL_CONTRACT','ARTIFICIAL_INPUT',out)


def test_import_and_cli_do_not_decode_science():
    code="import sys;sys.path.insert(0,'src');from trottertracks import c1_stage3_deterministic_accuracy;assert not any(x in sys.modules for x in ['numpy','scipy','qiskit','ffsim'])"
    subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)
    cli=ROOT/'scripts/run_research_c_c1_stage3_deterministic_accuracy.py'
    p=subprocess.run([sys.executable,str(cli),'--contract','/NEVER','--input','/NEVER','--execute'],capture_output=True,text=True)
    assert p.returncode==2 and 'fresh' in p.stderr


def test_artificial_byte_preflight_never_parses_saved_evidence_and_fixes_CA(tmp_path):
    p=proposal();parent=dict(proposal=p,physical_conditions=dict(artificial=True))
    (tmp_path/'parent.json').write_text(json.dumps(parent))
    (tmp_path/'input.npz').write_bytes(b'ARTIFICIAL_NOT_AN_NPZ')
    (tmp_path/'SOURCE.json').write_text(json.dumps(dict(files={})))
    c=dict(scope=d.SCOPE,candidate_ids=[x.id for x in d.candidates()],numerical_policy=d.NUMERIC,
        call_limits=d.call_limits(),maximum_logical_scientific_units=110,
        new_fit=0,optimizer=0,retry=0,resume=0,proposal=p,physical_conditions=parent['physical_conditions'],
        parent_contract='parent.json',parent_contract_sha256=d.io.sha(tmp_path/'parent.json'),
        input_sha256=d.io.sha(tmp_path/'input.npz'),source_manifest='SOURCE.json',
        source_manifest_sha256=d.io.sha(tmp_path/'SOURCE.json'),saved_inputs={},runtime_required_packages={},
        **{k+'_run':k for k in ('feasibility','shots','cost','stage2')},stage1_operator_directory='matrices',
        raw_s4_circuits={x.id:dict(raw=x.id+'.qpy') for x in d.candidates() if x.s==4},
        shell_circuits={x:dict(raw=x+'raw.qpy',compiled=x+'compiled.qpy') for x in ('head','X','Y')})
    for name in d.required_saved_paths(c):
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'NOT_JSON_OR_SCIENTIFIC_DATA')
        c['saved_inputs'][name]=d.io.sha(path)
    path=tmp_path/'contract.json';path.write_text(json.dumps(c))
    original=d.io.read
    allowed={path,tmp_path/'parent.json',tmp_path/'SOURCE.json'}
    def checked_read(name):
        assert Path(name) in allowed, 'Preflight tried to parse scientific saved values'
        return original(name)
    with patch.object(d.io,'read',side_effect=checked_read),patch.object(np,'load',side_effect=AssertionError('No decode')):
        d.preflight(tmp_path,path,tmp_path/'input.npz')
        c['proposal']['accuracy_target']['energy_ha']=.1;path.write_text(json.dumps(c))
        with pytest.raises(ValueError,match='CHANGED_DETERMINISTIC_CONDITIONS'):d.preflight(tmp_path,path,tmp_path/'input.npz')


def test_saved_cost_is_only_attached_after_numerical_CA_and_must_reconcile():
    backend=ArtificialBackend();r=backend.bundle['records'][d.candidates()[0].id]
    assert d.attach_cost(r,[],dict(same_accuracy=False))['totals'] is None
    t=accuracy.AccuracyTarget(**proposal()['accuracy_target']);c=d.candidates()[0]
    plans=[d.assess_fixed_round(t,c,row,saved,dict(occurrences=1<<row['m'],complex_signal_error_pilot=0.),alias_verified=True)
        for row,saved in zip(r['feasibility']['rounds'],r['shots']['rounds'])]
    energy=accuracy.assess_energy(t,0.,plans,state_model='own_DF_ground_state_oracle',alias_branch_verified=True)
    r['cost']['point_totals']['rz_count']+=1
    with pytest.raises(ValueError,match='SAVED_COST_WEIGHT_IDENTITY'):d.attach_cost(r,plans,energy)


def write_saved_bundle(root,b):
    for name,status,n in [('feasibility','COMPLETE_FEASIBILITY_ONLY',639),('shots','COMPLETE_OPTIMISTIC_FORMULA_SCREEN',282),('cost','COMPLETE_COST_ONLY',936)]:
        folder=root/name;folder.mkdir();(folder/'candidates').mkdir()
        (folder/'result.json').write_text(json.dumps(dict(status=status,scientific_units=n,call_counts=dict(state_alias_sector_checks=2))))
        (folder/'STOP.json').write_text(json.dumps(dict(failed=False,mandatory_stop=True)))
        for c in d.candidates():(folder/'candidates'/(c.id+'.json')).write_text(json.dumps(b['records'][c.id][name]))
    (root/'feasibility/prepared.json').write_text(json.dumps(b['prepared']))
    (root/'stage2').mkdir()
    (root/'stage2/result.json').write_text(json.dumps(dict(status='STAGE2_LIMITED_CIRCUIT_VALIDATION_PASS',gates=list(b['stage2_gates'].values()))))
    (root/'stage2/STOP.json').write_text(json.dumps(dict(mandatory_stop=True)))


def test_loader_joins_only_four_registered_saved_PF_shots_cost_records(tmp_path):
    b=bundle();write_saved_bundle(tmp_path,b)
    v=dict(root=tmp_path,contract={k+'_run':k for k in ('feasibility','shots','cost','stage2')})
    assert d.load_saved(v)['records']==b['records']
    p=tmp_path/'shots/candidates'/(d.candidates()[0].id+'.json')
    r=json.loads(p.read_text());r['rounds'][0]['saved_reference']['pf_operator_error']=.1;p.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='PF_IDENTITY'):d.load_saved(v)


def test_real_saved_backend_on_artificial_CAR_npz_QPY_only(tmp_path):
    from qiskit import QuantumCircuit,qpy
    b=bundle();v=dict(root=tmp_path,input=tmp_path/'input.npz',contract=dict(proposal=proposal(),
        physical_conditions=dict(system_qubits=2,XDF_rank=10),stage1_operator_directory='matrices',raw_s4_circuits={},shell_circuits={}))
    np.savez(v['input'],df_one_body=np.diag([.1,.1]).astype(complex),df_lambdas=np.zeros(10),
        df_g_matrices=np.zeros((10,2,2),complex),nuclear_repulsion=np.array(.375))
    folder=tmp_path/'matrices';folder.mkdir()
    base=np.diag([.375,.475,.475,.575]).astype(complex)
    for name in ['raw_base']+[f'raw_frame_{i}' for i in range(8)]:
        np.savez(folder/(name+'.npz'),matrix=base if name=='raw_base' else np.zeros((4,4),complex))
    def save(label,q):
        path=label+'.qpy'
        with (tmp_path/path).open('wb') as f:qpy.dump(q,f)
        return path
    for c in d.candidates():
        if c.s==1:continue
        q=QuantumCircuit(3);h=.1/c.s;q.p(-.375*h,2);q.cp(-.1*h,2,0);q.cp(-.1*h,2,1)
        v['contract']['raw_s4_circuits'][c.id]=dict(raw=save(c.id+'raw',q))
    for name in ('head','X','Y'):
        head,tail=connection.readout_segments(2,'Y' if name=='Y' else 'X')
        q=head if name=='head' else tail
        v['contract']['shell_circuits'][name]={side:save('shell'+name+side,q) for side in ('raw','compiled')}
    forbidden=Mock(side_effect=AssertionError('Excluded recomputation'))
    with patch.object(d,'load_saved',return_value=b),patch.object(d.reference,'reference_round',forbidden),\
            patch.object(d.MolecularBackend,'_ground',forbidden),patch.object(native_builder(),'ordered',forbidden):
        r=d.run(d.SavedBackend(v,tmp_path/'out'),proposal(),tmp_path/'out',workers=4)
    assert r['scientific_units']==110 and r['candidate_status_counts']=={'NUMERICAL_PILOT_CA_PASS':4}
    forbidden.assert_not_called()


def native_builder():
    return d.native.NativeBuilder
