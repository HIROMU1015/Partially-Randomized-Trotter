"""Off-domain/math/mock controls only; never open registered E matrices."""
import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from fractions import Fraction as F
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import acceptance as ac, launch, saved_data as sd
from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import accounting,worker
from trottertracks.algorithm_codesign.df_correction_comparison import intervals,inventory_integrity as inv
from trottertracks.algorithm_codesign.df_correction_et_storage_v5 import storage as st
from trottertracks.algorithm_codesign.df_correction_comparison.native_evaluation import operator_numeric_guard


@pytest.fixture(autouse=True)
def forbidden_registered_routes(monkeypatch):
    from trottertracks.algorithm_codesign.df_correction_comparison import comparison,angles
    def reject(*a,**k):raise AssertionError('No registered recipes or synthesis in preparation')
    monkeypatch.setattr(comparison,'cd_recipe',reject)
    monkeypatch.setattr(comparison,'e_recipe',reject)
    monkeypatch.setattr(angles,'synthesize',reject)
    intervals.configure()


def compiler():
    from trotterlib.rte import CompilerSettings
    return CompilerSettings(basis_gates=('rz','sx','x','cx'),backend_name=None,
        coupling_map=None,optimization_level=0,layout_method=None,routing_method=None,
        transpiler_seed=20261011,qiskit_version='1.3.0')


def save_list():
    records=[]
    def save(row):records.append(json.loads(json.dumps(row)))
    return records,save


def test_historical_name_difference_does_not_reject_and_raw_names_stay():
    from qiskit import QuantumCircuit
    a=QuantumCircuit(1,name='off-domain-A');a.rz(1/7,0)
    b=QuantumCircuit(1,name='off-domain-B');b.rz(1/7,0)
    # Names of nested definitions differ while the unitary is the same.
    c=QuantumCircuit(1);c.append(a.to_gate(),[0])
    d=QuantumCircuit(1);d.append(b.to_gate(),[0])
    results=[]
    for qc in (c,d):
        records,save=save_list()
        cost=ac.layered_transpile_and_measure_cost(qc,compiler(),expected='historical-unmatched',arrays={},context={'off_domain':True},save=save)
        assert cost.actual_circuit_fingerprint and not records[-1]['matches_expected']
        assert records[-1]['historical_mismatch_rejects'] is False
        assert records[-1]['fingerprint_attempts']==records[-1]['transpile_attempts']==1
        results.append(records[-1])
    assert results[0]['actual_fingerprint']!=results[1]['actual_fingerprint']
    assert results[0]['recursive_payload']['instructions'][0]['operation']['name']=='off-domain-A'
    assert results[1]['recursive_payload']['instructions'][0]['operation']['name']=='off-domain-B'


@pytest.mark.parametrize('exc',[TypeError('payload'),ValueError('payload'),RuntimeError('payload')])
def test_payload_exception_is_saved_and_stops_before_native_transpile(monkeypatch,exc):
    from trotterlib import rte_compiled_cost as shared
    from qiskit import QuantumCircuit
    monkeypatch.setattr(shared,'_circuit_payload',lambda c:(_ for _ in ()).throw(exc))
    monkeypatch.setattr(shared,'transpile',lambda *a,**k:pytest.fail('Must not compile after payload error'))
    records,save=save_list()
    with pytest.raises(type(exc)):
        ac.layered_transpile_and_measure_cost(QuantumCircuit(1),compiler(),expected='old',arrays={},context={},save=save)
    assert records[-1]['actual_fingerprint'] is None
    assert records[-1]['fingerprint_exception']['message']=='payload'
    assert records[-1]['phase']=='FINGERPRINT_EXCEPTION'
    assert records[-1]['transpile_attempts']==0


def test_storage_failure_stops_before_transpile(monkeypatch):
    from trotterlib import rte_compiled_cost as shared
    from qiskit import QuantumCircuit
    monkeypatch.setattr(shared,'transpile',lambda *a,**k:pytest.fail('No compile'))
    def fail(x):raise OSError('disk')
    with pytest.raises(OSError):ac.layered_transpile_and_measure_cost(QuantumCircuit(1),compiler(),expected=None,arrays={},context={},save=fail)


@pytest.mark.parametrize('mutation',['sign','phase','wire','dtype'])
def test_whole_constructor_rejects_input_parameter_phase_or_wire_difference(mutation):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    target=np.diag([1,complex(math.cos(.071),math.sin(.071))]).astype(complex)
    actual=target.copy();qc=QuantumCircuit(1)
    if mutation=='sign':actual=actual.conj()
    if mutation=='dtype':actual=actual.astype(np.complex64)
    qc.append(UnitaryGate(actual),[0])
    if mutation=='phase':qc.global_phase=.1
    if mutation=='wire':
        qc=QuantumCircuit(2);target=np.kron(target,np.eye(2));qc.append(UnitaryGate(target),[1,0])
    with pytest.raises(PermissionError):ac.validate_constructor(qc,{'input_target':target})


@pytest.mark.parametrize('controlled',[False,True])
@pytest.mark.parametrize('mutation',['sign','phase','wire'])
def test_strict_operator_guard_detects_sign_phase_and_wire_errors(controlled,mutation):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from trottertracks.algorithm_codesign.df_correction_comparison.native import pauli_ir,circuit_from_ir
    qc=circuit_from_ir(pauli_ir('XI',.071,controlled));target=Operator(qc).data.copy()
    bad=circuit_from_ir(pauli_ir('XI',-.071 if mutation=='sign' else .071,controlled))
    if mutation=='phase':bad.global_phase+=.1
    if mutation=='wire':bad=circuit_from_ir(pauli_ir('IX',.071,controlled))
    assert operator_numeric_guard(qc,target,F(0))[0]<F(1,5000000)
    assert operator_numeric_guard(bad,target,F(0))[0]>F(1,5000000)


def test_full_new_evaluator_uses_independent_reference_on_off_domain_rank1_frames():
    from trottertracks.algorithm_codesign.df_correction_et_storage_v5.native_evaluation import native_fixture
    frames=[]
    for a,b,p in [(3/5,4/5,.13),(5/13,12/13,-.27),(8/17,15/17,.41)]:
        frames.append(np.array([[a],[b*complex(math.cos(p),math.sin(p))]],complex))
    records={};prefixes=[]
    def save(label,row):records[label]=json.loads(json.dumps(row))
    result=native_fixture(2,frames,[{'active_modes':2,'controlled':c,'circuit_fingerprint':'historical-only'} for c in (False,True)],lambda p:prefixes.append(p['compile_calls']),on_diagnostic=save)
    assert result['active_modes']==2 and result['compile_calls']<=6
    assert prefixes==list(range(1,result['compile_calls']+1))
    assert len(records)==result['compile_calls']
    for row in result['native']:
        assert F(row['numeric_operator_error_upper'])<=F(1,5000000)
        ac.validate_native_inventory(row['inventory'])
        if row['arm']=='A':assert row['historical_matches_expected'] is False
    assert result['CD_cost_reuse'] is False and result['whole_PR_cost'] is None


@pytest.mark.parametrize('mutation',['hash','counts','angle','phase','wire'])
def test_saved_native_ir_and_inventory_tampering_is_rejected(mutation):
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(2);qc.rz(.123,0);qc.cx(0,1)
    row=ac.native_inventory(qc)
    if mutation=='hash':row['ordered_native_ir_sha256']='x'
    elif mutation=='counts':row['CX']=2
    elif mutation=='angle':row['occurrences'][0]['raw_binary64_radians']='1/7'
    elif mutation=='phase':row['global_phase_raw']='1/7'
    else:
        row['ordered_native_ir']['operations'][1]['qubits']=[0,0]
        row['ordered_native_ir_sha256']=ac.payload_digest(row['ordered_native_ir'])
    with pytest.raises(PermissionError):ac.validate_native_inventory(row)


def test_native_result_and_t_inventory_binding_detects_cost_substitution(tmp_path):
    from qiskit import QuantumCircuit
    q=QuantumCircuit(1);q.rz(.123,0)
    # Fabricated r3/r6 metadata containers; these are not registered matrices.
    native={r:{'native':[{'arm':'A','controlled':False,'inventory':ac.native_inventory(q)}]} for r in (3,6)}
    for r in (3,6):st.dump_json(tmp_path/('E-r'+str(r)+'.json'),native[r])
    inventory={'unique_keys':[],'native_result_bindings':ac.bind_native_results(tmp_path,native)}
    binding=inv.freeze_inventory(tmp_path,inventory,{'mock':True},'mock-marker')
    ac.verify_native_bindings(tmp_path,inv.verify_inventory(tmp_path,binding))
    (tmp_path/'E-r3.json').write_text('{}')
    with pytest.raises(PermissionError):ac.verify_native_bindings(tmp_path,inventory)


@pytest.mark.parametrize('mutation',['sequence','T','phase','error','key'])
def test_saved_T_identity_count_phase_and_error_mismatch_rejected(mutation):
    from trottertracks.algorithm_codesign.df_correction_comparison.angles import Angle
    row={'angle':Angle('raw-rad',F(1,7)).as_dict(),'sequence':'TtW','sequence_sha256':hashlib.sha256(b'TtW').hexdigest(),'T_count':2,'Tdagger_count':1,'W_count':1,'error_pass':True,'strict_operator_error_upper':'0'}
    contract={'T':{'gate_precision':'1/40000000000'},'caps':{'synthesis':{'sequence_characters':20000}}}
    accounting.validate_saved_sequence(row,contract)
    if mutation=='sequence':row['sequence']='T'
    elif mutation=='T':row['T_count']=3
    elif mutation=='phase':row['W_count']=0
    elif mutation=='error':row['strict_operator_error_upper']='1/10'
    else:row['angle']['key']='other'
    with pytest.raises((PermissionError,ValueError)):accounting.validate_saved_sequence(row,contract)


def fake_source(tmp_path,monkeypatch):
    art=tmp_path/launch.ART;art.mkdir(parents=True)
    code=tmp_path/'protected.py';code.write_text('x=1\n')
    (art/'contract_v5.json').write_text('{}')
    manifest=tmp_path/launch.MANIFEST
    manifest.write_text(json.dumps({'kind':'DF_CORRECTION_ET_SOURCE_MANIFEST_V5','files':[{'path':'protected.py','bytes':4,'sha256':launch.sha(code)}]}))
    auth={'status':'APPROVED_FOR_ONE_DF_CORRECTION_E_T_V5_RUN','science_execution_authorized':True,'runs':1,'retries':0,'mandatory_STOP':True,'source_commit':'a'*40,'contract_sha256':launch.sha(art/'contract_v5.json'),'source_manifest_sha256':launch.sha(manifest),'explicit_user_instruction':'MOCK ONLY'}
    (art/'authorization.json').write_text(json.dumps(auth))
    def git(root,*args):
        if args==('rev-parse','HEAD'):return 'b'*40
        if args[:2]==('rev-list','--parents'):return 'b'*40+' '+'a'*40
        if args in [('diff','--name-only','HEAD'),('diff','--cached','--name-only','HEAD'),('status','--porcelain=v1')]:return ''
        if args[:2]==('diff','--name-only'):return str(launch.ART/'authorization.json')
        if args==('branch','--show-current'):return 'mock-execution'
        if args==('remote','get-url','origin'):return 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
        if args[:2]==('ls-remote','origin'):return 'b'*40+' refs/heads/mock-execution'
        raise AssertionError(args)
    monkeypatch.setattr(launch,'git',git)
    return auth,git


@pytest.mark.parametrize('mutation',['parent','diff','remote','contract','manifest','source','instruction','namespace'])
def test_new_launch_rejects_wrong_auth_source_namespace_or_remote(tmp_path,monkeypatch,mutation):
    auth,git=fake_source(tmp_path,monkeypatch)
    assert launch.verify_launch(tmp_path)['authorization']=='b'*40
    if mutation=='parent':monkeypatch.setattr(launch,'git',lambda root,*args:'b'*40+' '+'c'*40 if args[:2]==('rev-list','--parents') else git(root,*args))
    elif mutation=='diff':monkeypatch.setattr(launch,'git',lambda root,*args:'protected.py' if args[:2]==('diff','--name-only') else git(root,*args))
    elif mutation=='remote':monkeypatch.setattr(launch,'git',lambda root,*args:'c'*40+' refs/heads/mock-execution' if args[:2]==('ls-remote','origin') else git(root,*args))
    elif mutation=='contract':(tmp_path/launch.ART/'contract_v5.json').write_text('{"changed":true}')
    elif mutation=='manifest':(tmp_path/launch.MANIFEST).write_text('{}')
    elif mutation=='source':(tmp_path/'protected.py').write_text('x=2\n')
    elif mutation=='instruction':
        auth['explicit_user_instruction']='';(tmp_path/launch.ART/'authorization.json').write_text(json.dumps(auth))
    else:(tmp_path/launch.RESULT).mkdir(parents=True)
    with pytest.raises(PermissionError):launch.verify_launch(tmp_path)


def test_new_child_marker_provenance_and_stop_flags_are_strict(tmp_path,monkeypatch):
    fake_source(tmp_path,monkeypatch);info=launch.verify_local(tmp_path)
    out=tmp_path/launch.RESULT;out.mkdir(parents=True)
    marker=out/'one_shot_marker.json';marker.write_text(json.dumps(dict(info,runs=1,retries=0,mandatory_STOP=True)))
    launch.verify_child(tmp_path,{'marker_sha256':launch.sha(marker)})
    saved=json.loads(marker.read_text());saved['mandatory_STOP']=False;marker.write_text(json.dumps(saved))
    with pytest.raises(PermissionError):launch.verify_child(tmp_path,{'marker_sha256':launch.sha(marker)})


@pytest.mark.parametrize('stage',['CD','E_inputs','E-r6','controlled','LP'])
def test_worker_cannot_submit_old_or_forbidden_stage(tmp_path,stage):
    with pytest.raises(PermissionError):worker.claim_job(tmp_path,{'stage':stage,'output':'not-used','marker_sha256':'mock'})


def test_pending_gate_and_new_namespace_do_not_consume_any_real_marker():
    assert not (ROOT/launch.RESULT).exists()
    with pytest.raises(PermissionError):launch.verify_local(ROOT)
    assert not (ROOT/launch.RESULT).exists()


@pytest.mark.parametrize('bad',['shape','rank','nonfinite','bytes'])
def test_saved_frame_validation_is_byte_exact_without_numerical_reconstruction(bad):
    frame=[[['0x1p+0','0x0p+0'],['0x0p+0','0x0p+0']] for i in range(2)]
    expected=sd.frame_identity(frame,0,2);broken=copy.deepcopy(frame)
    if bad=='shape':broken.pop()
    elif bad=='rank':broken[0].pop()
    elif bad=='nonfinite':broken[0][0][0]='inf'
    else:broken[0][0][0]='0x1.0000000000001p+0'
    if bad=='bytes':assert sd.frame_identity(broken,0,2)!=expected
    else:
        with pytest.raises(PermissionError):sd.frame_identity(broken,0,2)


def test_contract_preserves_unchanged_scientific_budgets_and_review_delegation():
    old=json.loads((ROOT/'artifacts/track_b_df_correction_comparison_source_v2/2026-10-11/contract_v2.json').read_text())
    new=json.loads((ROOT/launch.ART/'contract_v5.json').read_text())
    for key in ['arms','CD','T','caps','runtimes','numerical_certification']:
        assert new[key]==old[key]
    for k,v in old['E'].items():
        if k!='A_native':assert new['E'][k]==v
    assert new['resumption']['run_stages']==['E_NATIVE','T_KEYS']
    assert new['source_review_policy']['routine_final_source_validation'].startswith('DELEGATED_TO_CODEX')


def test_source_has_no_recipe_or_CD_evaluation_entry_and_no_name_stripping():
    paths=list((ROOT/'src/trottertracks/algorithm_codesign/df_correction_et_storage_v5').glob('*.py'))+[ROOT/'scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v5.py']
    for p in paths:
        tree=ast.parse(p.read_text())
        names={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
        assert not names & {'cd_recipe','e_recipe','cd_task','random_unitary','default_rng','setprofile'}


@pytest.mark.parametrize('failure',[None,'native_child','inventory','terminal_storage'])
def test_mock_coordinator_complete_or_partial_terminal_preserves_stop_and_no_CD(tmp_path,monkeypatch,failure):
    # No worker, matrix, synthesis or registered fixture is executed here.
    spec=importlib.util.spec_from_file_location('et_mock_runner',ROOT/'scripts/tracks/algorithm_codesign/run_df_correction_et_resumption_v5.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(sys,'argv',['mock-runner'])
    # main sets PYTHONPATH for children; contain that process-global change in
    # this mock test so subsequent real storage controls retain their source.
    monkeypatch.setenv('PYTHONPATH',str(ROOT/'src'))
    provenance={'source':'a'*40,'authorization':'b'*40,'authorization_sha256':'mock','contract_sha256':'mock','source_manifest_sha256':'mock','branch':'mock','remote_sha':'b'*40}
    monkeypatch.setattr(runner,'verify_launch',lambda root:provenance)
    contract=json.loads((ROOT/launch.ART/'contract_v5.json').read_text())
    art=tmp_path/launch.ART;art.mkdir(parents=True);(art/'contract_v5.json').write_text(json.dumps(contract))
    monkeypatch.setattr(worker,'runtime_check',lambda *a:None)
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(sd,'saved_cd_receipt',lambda *a:{'CD_recomputed':False,'old_run_reclassified':False})
    monkeypatch.setattr(sd,'load_saved_inputs',lambda *a:{'3':'MOCK_NO_FRAME','6':'MOCK_NO_FRAME'})
    from qiskit import QuantumCircuit
    from trottertracks.algorithm_codesign.df_correction_comparison.angles import Angle
    q=QuantumCircuit(1);q.rz(.123,0)
    seen=[]
    class Supervisor:
        def __init__(self,total,out,**kwargs):self.out=out
        def cpu(self):return 0
        def check_total(self):pass
        def receipt(self):return {'mock_only':True,'retries':0}
        def phase(self,jobs,limits,workers,**kwargs):
            if failure=='inventory' and any(j['label'].startswith('T-') for j in jobs):
                (self.out/'synthesis_inventory.json').chmod(0o644)
                (self.out/'synthesis_inventory.json').write_text('{}')
                raise RuntimeError('mock inventory corruption')
            for j in jobs:
                job=json.loads(Path(j['command'][-1]).read_text());seen.append(job['stage'])
                if job['stage']=='native':
                    if failure=='native_child':raise RuntimeError('mock child failed; prefix only')
                    native=[]
                    for controlled in (False,True):
                        inventory=ac.native_inventory(q);inventory['numeric_operator_error_upper']='0'
                        native.append({'arm':'A','controlled':controlled,'inventory':inventory})
                    row={'compile_calls':2,'native':native,'T_semantic_angles':[Angle('atan',F(1,7),F(s)).as_dict() for s in (-1,1,-2,2)],'ell_computational':'128/7','B_local_numeric_uniform_upper':'0',
                         'all_B_event_costs':[{'controlled':c,'sign':1,'executed_conditional_probability':'1','total_RZ':2 if c else 1} for c in (False,True)]}
                else:
                    seq='TtW';row={'angle':job['angle'],'sequence':seq,'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest(),'T_count':2,'Tdagger_count':1,'W_count':1,'strict_operator_error_upper':'0','error_pass':True,
                                'inventory_sha256':job['inventory_sha256'],'inventory_receipt_sha256':job['inventory_receipt_sha256']}
                st.dump_json(j['output'],row)
    monkeypatch.setattr(runner,'Supervisor',Supervisor)
    if failure=='terminal_storage':
        original_dump=runner.dump_json
        def failed_dump(path,*args,**kwargs):
            if Path(path).name=='terminal_status.json':raise OSError('mock terminal storage')
            return original_dump(path,*args,**kwargs)
        monkeypatch.setattr(runner,'dump_json',failed_dump)
    code=runner.main();out=tmp_path/launch.RESULT
    terminal=st.load_json(out/('terminal_storage_failure.json' if failure=='terminal_storage' else 'terminal_status.json'))
    assert terminal['retries']==0 and terminal['mandatory_STOP'] is True
    if failure!='terminal_storage':
        assert terminal['scientific_winner'] is None and terminal['CD_recalculation_calls']==0
    else:assert terminal['partial_scientific_outcome_allowed'] is False
    assert all(s in ('native','synthesis') for s in seen)
    assert (out/'one_shot_marker.json').exists()
    if failure:
        assert code==2 and terminal['classification']=='TECHNICAL_INCONCLUSIVE'
        if failure=='native_child':assert seen==['native']
    else:
        assert code==0 and terminal['classification']=='COMPLETE_E_T_COMPARISON_NO_AUTOMATIC_GO'
        assert terminal['completed_stages']==['CD_SAVED_REFERENCE_ONLY','E_NATIVE','T_KEYS']
        assert len(st.load_json(out/'T_accounting.json')['rows'])==4


def test_whole_wrong_raw_target_cannot_pass_by_self_comparison(monkeypatch):
    # Corrupt raw expm on analytic 2-mode/rank1; spectral target stays independent.
    import scipy.linalg
    from trottertracks.algorithm_codesign.df_correction_et_storage_v5.native_evaluation import native_fixture
    frames=[]
    for a,b,p in [(3/5,4/5,.13),(5/13,12/13,-.27),(8/17,15/17,.41)]:
        frames.append(np.array([[a],[b*complex(math.cos(p),math.sin(p))]],complex))
    monkeypatch.setattr(scipy.linalg,'expm',lambda a:np.eye(len(a),dtype=complex))
    with pytest.raises(ArithmeticError,match='Whole native numerical allowance'):
        native_fixture(2,frames,[{'active_modes':2,'controlled':c,'circuit_fingerprint':'historical-only'} for c in (False,True)],lambda p:None,on_diagnostic=lambda *a:None)
