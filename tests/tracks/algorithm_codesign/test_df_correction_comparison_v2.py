"""Off-domain semantic and containment controls. No registered recipes or synthesis."""
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import mpmath as mp
import pytest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.algorithm_codesign.df_correction_comparison import intervals as ci
from trottertracks.algorithm_codesign.df_correction_comparison import digital as d
from trottertracks.algorithm_codesign.df_correction_comparison import certification as cert
from trottertracks.algorithm_codesign.df_correction_comparison.angles import Angle,strict_guard,clifford_sequence
from trottertracks.algorithm_codesign.df_correction_comparison.native import pauli_ir,circuit_from_ir,ir_counts,t_completion,selected_positions,local_t_separation
from trottertracks.algorithm_codesign.df_correction_comparison.guard import Supervisor,Limits,CapFailure,exclusive_json
from trottertracks.algorithm_codesign.df_correction_comparison import launch


@pytest.fixture(autouse=True)
def configure():ci.configure()


def test_finite_law_changes_cost_and_moment_instead_of_using_ideal_law():
    law=d.largest_remainder({'a':F(1,3),'b':F(2,3)},2)
    rows=[{'p':law['a'],'weight':F(1),'cost':0},{'p':law['b'],'weight':F(2),'cost':4}]
    actual=d.account(rows)
    assert actual['expected_cost']==3 != F(8,3)
    assert actual['second_moment']==F(13,4) != 3
    assert actual['range']==2


def test_weight_gate_error_correlation_is_retained():
    rows=[{'p':F(1,2),'weight':1,'error':0},{'p':F(1,2),'weight':3,'error':2}]
    assert d.account(rows)['weighted_native_bias']==6
    assert 6 != 2*F(2)*F(1)


def test_exact_tie_and_small_mass_rounding_are_not_threshold_drop():
    assert d.largest_remainder({'b':F(1,2),'a':F(1,2)},0)=={'b':F(0),'a':F(1)}
    law=d.largest_remainder({'a':F(1,10**8),'b':1-F(1,10**8)},3)
    assert law['a']==0 and 'a' in law


def test_interval_midpoint_law_distance_covers_ideal_and_rounding():
    law,rad=d.rounded_law({'a':mp.iv.mpf(['0.32','0.34']),'b':mp.iv.mpf(['0.66','0.68'])},3)
    assert sum(law.values())==1
    assert sum(abs(law[k]-q) for k,q in {'a':F(1,3),'b':F(2,3)}.items())<=rad


def test_complete_coin_preserves_exact_acceptance_and_zero_identity():
    rows,radius=d.corrector_events(F(3,16),{'x':F(1,3),'y':F(2,3)},
                                  {'x':{'Z':0},'y':{'Z':F(1,5),'X':F(-2,5)}},bits=4)
    assert sum(r['p'] for r in rows)==1
    assert rows[0]['p']==F(13,16)
    assert any(r['label']=='x:zero' and r['pauli'] is None for r in rows)
    assert radius>0


def test_rounded_implemented_weight_range_is_not_ideal_range():
    point,radius=d.weight_point(F(1,9),F(7,11))
    ideal=mp.iv.sqrt(1+(ci.iv(F(7,99)))**2)
    a,b=ci.bounds(ideal)
    assert a-radius<=point<=b+radius
    rows=[{'p':F(1,8),'weight':point},{'p':F(7,8),'weight':1}]
    actual=d.account(rows)
    assert actual['second_moment']==F(7,8)+point*point/8


@pytest.mark.parametrize('kind,value',[('raw-rad',F(1)),('pi',F(1,3)),('atan',F(2,7))])
def test_closed_angle_types_have_correct_meaning(kind,value):
    angle=Angle(kind,value)
    wanted=float(value) if kind=='raw-rad' else (math.pi*float(value) if kind=='pi' else math.atan(float(value)))
    a,b=ci.bounds(angle.at(mp.iv))
    assert abs(float((a+b)/2)-wanted)<1e-15
    assert Angle.load(angle.as_dict())==angle


def test_raw_one_radian_is_not_atan_one_and_unknown_kind_rejected():
    assert Angle('raw-rad',1).at(mp.mp)!=Angle('atan',1).at(mp.mp)
    with pytest.raises(ValueError):Angle('radians?',1)


@pytest.mark.parametrize('k',[-5,-2,0,1,4,8])
def test_strict_clifford_phase_and_four_pi_period(k):
    angle=Angle('pi',F(k,2))
    assert strict_guard(clifford_sequence(angle),angle)<F(1,10**90)


def test_projective_T_and_two_pi_phase_are_rejected_strictly():
    assert clifford_sequence(Angle('pi',F(1,4))) is None
    assert strict_guard('T',Angle('pi',F(1,4)))>F(1,4)
    assert strict_guard('',Angle('pi',2))>2
    assert strict_guard('',Angle('pi',4))<F(1,10**90)


def test_B_pairing_mean_weight_and_sign_independent_of_whole_mean():
    from scipy.linalg import expm
    x,z=cert.pauli_matrix('X'),cert.pauli_matrix('Z')
    beta={'X':F(-1,5),'Z':F(2,7)};ell=sum(abs(v) for v in beta.values());u=F(1,37)
    alpha=math.atan(float(u*ell));w=math.sqrt(1+float(u*ell)**2)
    whole=sum(float(v)*p for v,p in zip(beta.values(),[x,z]))
    mean=sum(float(abs(v)/ell)*w*expm(1j*(1 if v>0 else -1)*alpha*p) for v,p in zip(beta.values(),[x,z]))
    assert np.linalg.norm(mean-(np.eye(2)+1j*float(u)*whole))<5e-16
    assert np.linalg.norm(mean-expm(1j*float(u)*whole))>1e-6
    assert not np.allclose(mean,mean.conj().T)


def test_ball_frame_and_primitive_cover_ideal_orthonormalization():
    raw=np.array([[1+1e-12,1e-12j],[1e-12,1-1e-12]],complex)
    errors,proof=cert.frame_guard(raw)
    assert all(e>0 for e in errors) and proof['positive_norms_certified']
    q,ledger=cert.primitive(raw,1)
    assert np.linalg.norm(q.value-q.value.conj().T)<float(q.error)*2
    assert np.linalg.norm(q.value-np.diag([1,-1,-1,1]))<float(q.error)+1e-12


def test_rank_uncertainty_stops_instead_of_dropping_columns():
    with pytest.raises(ArithmeticError):cert.frame_guard(np.array([[1,1],[0,0]],complex))


def test_exact_pauli_proxy_reconstructs_noncommuting_small_matrix():
    x,z=cert.pauli_matrix('X'),cert.pauli_matrix('Z')
    value=.25*x-.125*z
    table,rebuild,radius,rows=cert.pauli_table(cert.Ball(value))
    assert table['X']==F(1,4) and table['Z']==F(-1,8)
    assert np.linalg.norm(rebuild-value)<=float(radius)
    assert all(r['zero_status']!='PROVED_IDEAL_TRACE_OR_PARITY_ZERO' for r in rows)


def test_trace_parity_projection_is_proof_distinct_from_computational_zero():
    a=cert.Ball(np.diag([0,.125,-.125,0]).astype(complex),F(1,10**12))
    table,_,radius,rows=cert.pauli_table(a,number_conserving=True,traceless=True)
    assert table['II']==0 and table['IX']==0
    assert any(r['zero_status']=='EXACT_COMPUTATIONAL_ZERO_IDEAL_UNRESOLVED_CHARGED' for r in rows)
    assert radius>=a.error


@pytest.mark.parametrize('label',['X','Y','Z','XY','YZ','XX','ZZ','IX'])
@pytest.mark.parametrize('controlled',[False,True])
def test_native_IR_counts_and_controlled_relative_phase(label,controlled):
    from qiskit.quantum_info import Operator
    from scipy.linalg import expm,block_diag
    alpha=-.047  # Off-domain, not any E/CD registered angle.
    ir=pauli_ir(label,alpha,controlled);qc=circuit_from_ir(ir)
    target=expm(1j*alpha*cert.pauli_matrix(label))
    if controlled:target=block_diag(np.eye(len(target)),target)
    assert np.linalg.norm(Operator(qc).data-target)<2e-14
    count=ir_counts(ir);w=sum(c!='I' for c in label)
    assert count['CX']==(2*w if controlled else 2*(w-1))
    assert count['arbitrary_angle_RZ']==(2 if controlled else 1)
    if 'X' in label or 'Y' in label:assert count['total_RZ']>count['arbitrary_angle_RZ']


def test_one_off_domain_transpile_validates_fixed_native_lowering_counts():
    from qiskit import transpile
    ir=pauli_ir('XY',.047,True);qc=circuit_from_ir(ir)
    compiled=transpile(qc,basis_gates=['rz','sx','x','cx'],optimization_level=0,seed_transpiler=20261011)
    assert dict(compiled.count_ops())==ir_counts(ir)['gate_counts']


def test_fixed_position_selection_never_refills_dedup_slots():
    assert selected_positions(3,8)==[0,1,2]
    assert selected_positions(101,8)==[j*100//7 for j in range(8)]


def test_T_partial_lower_counts_multiplicity_and_does_not_claim_completion():
    angle=Angle('raw-rad',F(1,7));other=Angle('raw-rad',F(2,7))
    seq='TtT';row={'angle':angle.as_dict(),'sequence':seq,'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest(),
        'T_count':3,'error_pass':True,'strict_operator_error_upper':'1/100000000000'}
    inv={'occurrences':[{'angle':angle.as_dict()}]*5+[{'angle':other.as_dict()}],
         'phase_supported':True,'total_RZ':6,'numeric_operator_error_upper':'0'}
    out=t_completion(inv,{angle.key:row})
    assert out['fixed_plan_completion_T_lower']==15
    assert out['actual_total_T'] is None and out['A_TOTAL_T_UNKNOWN']
    assert local_t_separation(out,{'actual_total_T':14})['FIXED_PLAN_LOCAL_T_SEPARATION']
    assert not local_t_separation(out,{'actual_total_T':None})['FIXED_PLAN_LOCAL_T_SEPARATION']


def test_T_no_numeric_certificate_cannot_claim_full_accuracy():
    inv={'occurrences':[],'phase_supported':True,'total_RZ':0}
    assert not t_completion(inv,{})['full_circuit_synthesis_certified']


def test_axis_union_bound_and_sufficient_confidence_arithmetic():
    assert 12*F(1,240)==F(1,20)
    assert 2*F(1,20000)**2<F(1,10000)**2
    shots=d.sufficient_shots(F(3,2),F(2),F(1,100),F(1,25))
    assert shots>10000


def test_interval_upper_serialization_is_directed():
    x=mp.iv.exp(ci.iv(F(1,19)))
    rec=ci.record(x);restored=ci.from_record(rec)
    assert ci.bounds(restored)[0]<=ci.bounds(x)[0] and ci.bounds(restored)[1]>=ci.bounds(x)[1]
    assert ci.outward_upper(F(1,3),16)>=F(1,3)


def test_preparation_gate_and_registered_recipe_fail_before_input_access():
    from trottertracks.algorithm_codesign.df_correction_comparison.comparison import cd_recipe,e_recipe
    with pytest.raises(PermissionError):launch.verify_launch(ROOT)
    with pytest.raises(PermissionError):cd_recipe({})
    with pytest.raises(PermissionError):e_recipe({})
    assert not (ROOT/launch.RESULT/'one_shot_marker.json').exists()


def _job(tmp_path,label,code):
    output=tmp_path/(label+'.json')
    return {'label':label,'output':str(output),'command':[sys.executable,'-B','-c',code.replace('OUTPUT',repr(str(output)))]}


def test_external_wall_guard_kills_and_leaves_queue_unlaunched(tmp_path):
    sup=Supervisor(Limits(10,10,10000,10000),tmp_path)
    jobs=[_job(tmp_path,'slow','import time; time.sleep(5)'),_job(tmp_path,'unlaunched','open(OUTPUT,"w").write("{}")')]
    with pytest.raises(CapFailure):sup.phase(jobs,Limits(.15,10,10000,10000),1)
    assert not (tmp_path/'unlaunched.json').exists()
    assert sup.receipt()['child_receipts'][0]['guard_terminated']


def test_external_CPU_guard_is_enforced_during_child_work(tmp_path):
    sup=Supervisor(Limits(10,10,10000,10000),tmp_path)
    job=_job(tmp_path,'burn','while True: pass')
    with pytest.raises(CapFailure):sup.phase([job],Limits(5,10,10000,10000,5,.12),1,per_key=True)
    assert sup.receipt()['CPU_seconds_parent_and_reaped_children']>=.1


def test_external_RSS_guard_detects_allocations_and_records_peak(tmp_path):
    sup=Supervisor(Limits(10,10,10000,10000),tmp_path)
    job=_job(tmp_path,'memory','import time; x=bytearray(40*1024*1024); time.sleep(5)')
    with pytest.raises(CapFailure):sup.phase([job],Limits(5,10,10000,20),1)
    assert sup.peak>20 and sup.receipt()['instantaneous_memory_ceiling_claim'] is False


def test_reaped_child_CPU_and_output_receipt_survive_normal_completion(tmp_path):
    sup=Supervisor(Limits(10,10,10000,10000),tmp_path)
    jobs=[_job(tmp_path,'done'+str(i),'import time; end=time.process_time()+.08\nwhile time.process_time()<end: pass\nopen(OUTPUT,"w").write("{}")') for i in range(2)]
    assert len(sup.phase(jobs,Limits(5,5,10000,10000),2))==2
    assert sup.receipt()['CPU_seconds_parent_and_reaped_children']>=.16
    assert len(sup.receipt()['completed_labels'])==2


def test_marker_exclusive_create_cannot_be_reused(tmp_path):
    marker=tmp_path/'marker.json';exclusive_json(marker,{'run':1})
    identity=marker.read_bytes()
    with pytest.raises(FileExistsError):exclusive_json(marker,{'run':2})
    assert marker.read_bytes()==identity


def test_completed_per_key_CPU_receipt_can_still_trigger_stop(tmp_path):
    sup=Supervisor(Limits(10,10,10000,10000),tmp_path)
    job=_job(tmp_path,'finished','open(OUTPUT,"w").write(\'{"_process_receipt":{"CPU_seconds":1}}\')')
    with pytest.raises(CapFailure):sup.phase([job],Limits(5,10,10000,10000,5,.2),1,per_key=True)


def test_exhausted_budget_does_not_spawn_a_child(tmp_path):
    sup=Supervisor(Limits(10,10,10000,10000),tmp_path)
    job=_job(tmp_path,'absent','open(OUTPUT,"w").write("{}")')
    with pytest.raises(CapFailure):sup.phase([job],Limits(0,10,10000,10000),1)
    assert not (tmp_path/'absent.log').exists()


def test_proved_zero_whole_tuple_is_identity_without_zeroing_PR_base():
    rows,_=d.corrector_events(F(3,8),{'z':F(1)},proved_zero_tuples={'z'})
    assert all(r['pauli'] is None for r in rows)
    assert sum(r['p'] for r in rows)==1


def test_off_domain_paired_P5_execution_law_preserves_normalization_accounting():
    from trottertracks.algorithm_codesign.df_correction_comparison.comparison import rte_parameters
    delta=F(1,11)  # NOT any registered CD step.
    law,l1,norm,error,ideal=rte_parameters(delta,bits=8)
    assert sum(law.values())==1 and norm>1 and l1>0
    assert ci.bounds(ideal)[0]-error<=norm<=ci.bounds(ideal)[1]+error
    assert set(law)=={'0','2','4'}


def test_scalar_A_and_noncontraction_B_bounds_on_off_domain_parameters():
    x=ci.scalar_error_bounds(F(1,11),F(1,100),F(1,9),11)
    assert ci.lower(x['A'])>0 and ci.lower(x['B'])>0
    assert ci.upper(x['A'])>ci.upper(x['B'])


def test_native_inventory_keeps_raw_radians_and_certifies_clifford_distance():
    from qiskit import QuantumCircuit
    from trottertracks.algorithm_codesign.df_correction_comparison.native import native_inventory
    qc=QuantumCircuit(1);qc.rz(1.0,0);qc.rz(math.pi/2,0);qc.global_phase=math.pi/4
    inv=native_inventory(qc)
    assert inv['occurrences'][0]['angle']['kind']=='raw-rad'
    assert F(inv['occurrences'][0]['angle']['value'])==1
    assert inv['occurrences'][1]['angle']['kind']=='pi'
    assert F(inv['angle_lowering_error_upper'])>0
    assert inv['phase_supported'] and inv['global_phase_W_exponent']==1


def test_source_worker_job_is_consumed_once_and_respects_terminal_STOP(tmp_path):
    from trottertracks.algorithm_codesign.df_correction_comparison.worker import claim_job
    out=tmp_path/launch.RESULT;out.mkdir(parents=True)
    job={'stage':'CD','output':str(out/'CD.json'),'marker_sha256':'fixture-only'}
    claim_job(tmp_path,job)
    with pytest.raises(FileExistsError):claim_job(tmp_path,job)
    (out/'terminal_status.json').write_text('{}')
    with pytest.raises(PermissionError):claim_job(tmp_path,job)


def _fake_authorized_source(tmp_path,monkeypatch):
    art=tmp_path/launch.ART;art.mkdir(parents=True)
    source='a'*40;head='b'*40
    code=tmp_path/'protected.py';code.write_text('x=1\n')
    (art/'contract_v2.json').write_text('{}')
    contract_hash=launch.sha(art/'contract_v2.json')
    (art/'authorization.json').write_text(json.dumps({'status':'APPROVED_FOR_ONE_DF_CORRECTION_COMPARISON_V2_RUN',
        'science_execution_authorized':True,'runs':1,'retries':0,'mandatory_STOP':True,
        'source_commit':source,'contract_sha256':contract_hash}))
    (art/'source_manifest_v2.json').write_text(json.dumps({'files':[{'path':'protected.py','bytes':4,'sha256':launch.sha(code)}]}))
    def git(root,*args):
        if args==('rev-parse','HEAD'):return head
        if args[:2]==('rev-list','--parents'):return head+' '+source
        if args==('diff','--name-only','HEAD'):return ''
        if args[:2]==('diff','--name-only'):return str(launch.ART/'authorization.json')
        if args==('diff','--cached','--name-only','HEAD'):return ''
        if args==('status','--porcelain=v1'):return ''
        if args==('branch','--show-current'):return 'test-execution'
        if args==('remote','get-url','origin'):return 'git@github.com:HIROMU1015/Partially-Randomized-Trotter.git'
        if args[:2]==('ls-remote','origin'):return head+' refs/heads/test-execution'
        raise AssertionError(args)
    monkeypatch.setattr(launch,'git',git)
    return git


def test_mock_direct_child_launch_then_protected_source_mutation_rejected(tmp_path,monkeypatch):
    _fake_authorized_source(tmp_path,monkeypatch)
    info=launch.verify_launch(tmp_path)
    assert info['source']=='a'*40 and info['authorization']=='b'*40
    (tmp_path/'protected.py').write_text('x=2\n')
    with pytest.raises(PermissionError):launch.verify_launch(tmp_path)


def test_mock_authorization_only_diff_and_parent_must_be_exact(tmp_path,monkeypatch):
    real=_fake_authorized_source(tmp_path,monkeypatch)
    def wrongdiff(root,*args):
        return 'protected.py' if args[:2]==('diff','--name-only') else real(root,*args)
    monkeypatch.setattr(launch,'git',wrongdiff)
    with pytest.raises(PermissionError):launch.verify_launch(tmp_path)
    def wrongparent(root,*args):
        return 'b'*40+' '+'c'*40 if args[:2]==('rev-list','--parents') else real(root,*args)
    monkeypatch.setattr(launch,'git',wrongparent)
    with pytest.raises(PermissionError):launch.verify_launch(tmp_path)


def test_mock_consumed_marker_blocks_launch_without_modification(tmp_path,monkeypatch):
    _fake_authorized_source(tmp_path,monkeypatch)
    out=tmp_path/launch.RESULT;out.mkdir(parents=True)
    marker=out/'one_shot_marker.json';marker.write_text('consumed')
    with pytest.raises(PermissionError):launch.verify_launch(tmp_path)
    assert marker.read_text()=='consumed'


def test_off_domain_float_assembly_is_bit_identical_to_frozen_recipe_helper():
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_df_audit_helper',ROOT/'scripts/tracks/algorithm_codesign/audit_df_sparse_correction.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    frame=np.array([[1/math.sqrt(2),1j/math.sqrt(2)],
                    [1j/math.sqrt(2),1/math.sqrt(2)]],complex)
    new,_=cert.primitive(frame,-1)
    raw=old.primitive(frame,-1)
    assert new.value.tobytes()==raw.tobytes()
    x=cert.Ball(np.array([[0,1j],[-1j,0]],complex))
    z=cert.Ball(np.diag([1,-1]).astype(complex))
    assert cert.comm(x,z).value.tobytes()==old.comm(x.value,z.value).tobytes()


def test_total_guard_covers_final_metadata_boundary(tmp_path):
    sup=Supervisor(Limits(0,10,10000,10000),tmp_path)
    with pytest.raises(CapFailure):sup.check_total()
