"""Preparation only: saved metadata joins and tiny artificial diagnostics.

No real-H4 QPY loader, state propagator, active contract or grant issuer.
The numerical examples below reject system sizes above three before evolution.
"""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import struct

from . import h4_certificate_control as C
from .h4_conditional_cng import extract
from .h4_enclosure_prototype import Interval

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'artifacts/non_df_h4_central_rho_preparation/2026-10-24'
AXES=('cosine','sine')
D2='74515d8eb4f073c40fc783681a3775dbca89e9b1'
AG='ee7d281b5daead1d0d5e81e00c9a0ed4727792b0'
AA='a37232f85812688c3b9ab957c5073c5ce143b17e'
OLD='artifacts/non_df_h4_central_conditional_cng/2026-10-24/recovery02'
MODEL='artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_model.json'
CRESULT='artifacts/non_df_h4_c_reference_evidence/2026-10-23/run01/pauli_jw_result.json'
TARGETS=(('PF',0,'d761627d56a4454e644358fef8907b8a7db28c4feeb44130224c20e4e50aae4d'),
         ('PR',0,'93ef5e4e28b499472c8ba9329f6cb139f63c5de071279783b3cca3319793fc97'),
         ('PR',1,'596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa'))


def require_draft(contract):
    C.require(all(contract.get(k) is False for k in ('execution_authorized','dispatch_enabled','grant_valid')), 'PREPARATION_FLAGS_MUST_BE_FALSE')
    C.require(all(contract.get(k) is None for k in ('grant','scientific_runner_SOURCE','launch_command')), 'NO_SCIENCE_CAPABILITY')
    C.require(contract.get('mandatory_STOP') is True and contract.get('automatic_next_stage') is False, 'NO_AUTOMATIC_STAGE')


def execution_request(*_args, **_kwargs):
    raise PermissionError('PREPARATION_ONLY_NO_H4_EXECUTOR_OR_VALID_GRANT')


def check_saved_tags(inp,metrics,expected_source):
    C.require(inp['SOURCE']==metrics['SOURCE']==expected_source,'SAVED_SOURCE')
    C.require(inp['input_fingerprint']==metrics['input_fingerprint'],'INPUT_FINGERPRINT_JOIN')
    C.require(metrics['compiler']==dict(backend_name=None,basis_gates=['rz','sx','x','cx'],coupling_map=None,
        layout_method=None,optimization_level=1,qiskit_version='1.3.0',routing_method=None,transpiler_seed=17),'FIXED_COMPILER')


def metadata_targets():
    """Read/hash bytes only: never decode QPY, NPZ or evaluate a saved signal."""
    expected=C.read(ROOT/OLD/'preservation_before.json')
    expected.update({OLD+'/run01/'+k:v for k,v in C.read(ROOT/OLD/'output_inventory.json').items()})
    files={}
    def bound(p):
        C.require(p in expected,'MISSING_PRIOR_SHA:'+p)
        C.require(C.sha(ROOT/p)==expected[p],'PRIOR_SHA:'+p)
        files[p]=expected[p]
        return p
    def read(p):return C.read(ROOT/bound(p))
    rows=read('artifacts/non_df_h4_a_axes/2026-10-24/run01/paired_costs.json')['rows']
    model=read(MODEL); cres=read(CRESULT)
    ah=extract('src/trotterlib/h4_ball_evidence.py','array_hash',dict(struct=struct,hashlib=hashlib,json=json))
    C.require(ah(model['state'],vector=True)==model['state_hash'],'SAVED_STATE_BITS')
    C.require(C.digest({k:v for k,v in model.items() if k!='binding_hash'})==model['binding_hash'],'C_MODEL_BINDING')
    result=[]
    for method,index,tid in TARGETS:
        row=next(r for r in rows if (r['case'],r['index'])==(method,index))
        C.require(row['trajectory_id']==tid,'SELECTED_TRAJECTORY_ID')
        oldcase='pauli_jw_'+method
        old=read('artifacts/non_df_h4_pilot/2026-10-23/run01/run/'+oldcase+'/result.json')
        C.require(old['state']['hash']==model['state_hash'],'SHARED_STATE_HASH')
        joinpath='artifacts/non_df_h4_pilot/2026-10-23/run01/run/'+oldcase+'/trajectories/'+tid+'/price_join.json'
        join=read(joinpath)
        C.require(all(join[k]==row[k] for k in ('trajectory_id','event_record_hash','trajectory_record_hash')),'EVENT_JOIN')
        verify_join=extract('src/trotterlib/rte_event_audit.py','verify_join',dict(digest=C.digest))
        verify_join(join,expected_case=oldcase)
        bodyfolder=row['cosine_origin'] if index==0 else 'artifacts/non_df_h4_a_gate_order2/2026-10-24/run01'
        source_body=bound(bodyfolder+'/source_body.qpy')
        controlled=bound(bodyfolder+'/controlled_body.qpy')
        side=read(bodyfolder+'/validation.json') if index==0 else read(bodyfolder+'/body_source_check.json')
        for axis in AXES:
            folder=row[axis+'_origin']; inp=read(folder+'/input.json'); metrics=read(folder+'/metrics.json')
            src=D2 if axis=='cosine' and index==0 else AG if index==1 else AA
            C.require(row[axis+'_SOURCE']==src,'PAIRED_ROW_SOURCE')
            check_saved_tags(inp,metrics,src)
            compiler=metrics['compiler']
            input_qpy=bound(folder+'/input.qpy'); output_qpy=bound(folder+'/output.qpy')
            C.require(inp['input_qpy_sha256']==files[input_qpy],'INPUT_QPY_SHA')
            if 'output_qpy_sha256' in metrics:C.require(metrics['output_qpy_sha256']==files[output_qpy],'OUTPUT_QPY_SHA')
            phase=(inp['sidecar'].get('aggregate_source_global_phase') if index==1 or axis=='cosine'
                   else read(row['cosine_origin']+'/input.json')['sidecar']['aggregate_source_global_phase'])
            result.append(dict(key=f'{method}{index}_{axis}',method=method,index=index,trajectory_id=tid,axis=axis,
                event_record_hash=row['event_record_hash'],trajectory_record_hash=row['trajectory_record_hash'],price_join=joinpath,
                SOURCE=src,source_body=source_body,controlled_body=controlled,input_qpy=input_qpy,compiled_qpy=output_qpy,
                input_fingerprint=inp['input_fingerprint'],output_fingerprint=metrics['output_fingerprint'],
                input_sha256=files[input_qpy],output_sha256=files[output_qpy],compiler=compiler,
                saved_RZ=metrics['RZ'],saved_CX=metrics['CX'],saved_post_histogram=metrics['posttranspile_histogram'],
                source_body_instruction_count=side['source_instruction_count'],aggregate_source_phase=phase,
                state_hash=model['state_hash'],state_model=MODEL,state_pointer=['state'],state_norm_pointer=[CRESULT,'state_norm'],
                C_selected_ideal_saved=method=='PF',C_encoded_mean_pointer=[CRESULT,'implemented_law',method,'corrected_encoded_mean'],
                C_pointer_is_law_mean_not_PR_trajectory=method=='PR',reference_pointer=[CRESULT,'reference'],
                saved_signed_bias=old['signal']['signed_bias'],B_saved=old['normalization']['B'],saved_log_B_upper=old['normalization']['log_B_upper'],
                measurement='Z_ancilla = p0-p1; +Re for cosine, +Im for Sdg-before-final-H sine',
                raw_state_replaced=False,new_signal=None,real_rho_new=None))
    guard=read(OLD+'/run01/axis_results.json')
    for name in ('artifacts/non_df_h4_pilot/2026-10-23/run01/SOURCE.json',
                 'artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/receipt.json',
                 'artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/snapshot.npz'):
        bound(name)
    return result, files, dict(state_hash=model['state_hash'],C_state_norm=cres['state_norm'],
        C_reference=cres['reference'],guard_rows=guard,normalization_recomputed=False,
        real_QPY_decodes=0,real_NPZ_decodes=0,real_state_norm_calculations=0)


def toy_wrapper(controlled, axis):
    """Reuse the original pure wrapper only on tiny supplied artificial bodies."""
    C.require(2<=controlled.num_qubits<=4,'ARTIFICIAL_MAX_3_SYSTEM_QUBITS')
    from qiskit import QuantumCircuit
    fn=extract('src/trotterlib/fixed_signal_evaluation.py','hadamard_wrapper',dict(AXES=AXES,QuantumCircuit=QuantumCircuit))
    return fn(controlled,axis,already_controlled=True)


def toy_program_profile(circuit, *, compiled=False):
    C.require(2<=circuit.num_qubits<=4 and circuit.num_clbits==1,'ARTIFICIAL_PROGRAM_SHAPE')
    C.require(not circuit.parameters and circuit.layout is None,'NO_PARAMETERS_OR_LAYOUT')
    n=circuit.num_qubits-1
    last=circuit.data[-1]
    C.require(last.operation.name=='measure' and [circuit.find_bit(q).index for q in last.qubits]==[n]
              and [circuit.find_bit(c).index for c in last.clbits]==[0],'TERMINAL_ANCILLA_MEASUREMENT')
    for item in circuit.data[:-1]:
        C.require(not item.clbits and getattr(item.operation,'condition',None) is None and item.operation.name not in ('measure','reset'),'NO_MID_MEASURE_RESET_CONDITION')
        if compiled:C.require(item.operation.name in ('rz','sx','x','cx'),'FIXED_COMPILED_BASIS')
    return dict(n_system=n,ancilla=n,terminal_measurement=True,global_phase=float(circuit.global_phase),
                instructions=len(circuit.data),compiled_profile_checked=compiled)


def toy_signal(circuit, state, *, fixture_id):
    """Reuse old Statevector evolution, with explicit unclipped ray observable.

    Binary64 output is EMPIRICAL, with no fabricated interval or roundoff bound.
    No preparation gates or physical normalization change to the input bytes.
    Qiskit probabilities() clips to [0,1], so it is unsuitable on raw nonunit
    toy vectors. Reduce the evolved amplitudes before dividing by original nu.
    """
    C.require(fixture_id.startswith('TOY_'),'ARTIFICIAL_FIXTURE_REQUIRED')
    profile=toy_program_profile(circuit)
    C.require(len(state)==1<<profile['n_system'],'ARTIFICIAL_STATE_SHAPE')
    import numpy as np
    from qiskit.quantum_info import Statevector
    raw=np.asarray(state,dtype=complex)
    norm=float(np.vdot(raw,raw).real)
    C.require(math_isfinite_positive(norm),'STATE_NORM_POSITIVE_FINITE')
    initial=Statevector(np.concatenate((raw,np.zeros_like(raw))))
    final=initial.evolve(circuit.remove_final_measurements(inplace=False)).data
    probabilities=np.sum(np.abs(final.reshape(2,-1))**2,axis=1)
    return dict(signal=float((probabilities[0]-probabilities[1])/norm),raw_norm=norm,
                status='EMPIRICAL_ARTIFICIAL_ONLY',numerical_error_upper=None,rho_upper=None,formal_certificate_accepted=False)


def math_isfinite_positive(value):
    import math
    return math.isfinite(value) and value>0


def toy_compare(ideal,compiled,state,axis,*,B_saved,c_target_value=None,fixture_id):
    C.require(axis in AXES and math_isfinite_positive(float(B_saved)),'TOY_AXIS_SCALE')
    before=toy_signal(ideal,state,fixture_id=fixture_id)
    after=toy_signal(compiled,state,fixture_id=fixture_id)
    return dict(status='EMPIRICAL_ARTIFICIAL_ONLY',axis=axis,
        input_signal=before['signal'],compiled_signal=after['signal'],
        delta_input_output=float(B_saved)*abs(after['signal']-before['signal']),
        selected_C_target_gap=None if c_target_value is None else float(B_saved)*abs(after['signal']-c_target_value),
        selected_C_target_gap_is_empirical=True,rho_new=None,rho_upper=None,
        PR_law_guaranteed=False,formal_certificate_accepted=False,numerical_error_upper=None)


def drafts(targets):
    """Explicit proposals, never authority. Counts distinguish logical vs actions."""
    common=dict(execution_authorized=False,dispatch_enabled=False,grant_valid=False,grant=None,
        scientific_runner_SOURCE=None,launch_command=None,mandatory_STOP=True,automatic_next_stage=False,
        automatic_retry=False,formal_certificate_accepted=False,base_commit='388b1e525d3cfbeba8f231acd42c813f67a7f531',
        fixed='H4 linear1A STO-3G n8+ancilla8 Pauli184/prefix27 S2 T.8 q8 r4 K4 D2 Qiskit1.3 rz/sx/x/cx opt1 seed17 no routing/layout/backend',
        forbidden_new_calls=dict(compile=0,circuit_build=0,sampling=0,RNG_replay=0,SCF_FCI_DF=0,state_generation=0,C_L_rerun=0,N_G=0,certificate_acceptance=0),
        current_real_H4_calls=0,real_rho_new=None,population_cost_tail='UNKNOWN',
        error_ownership='u_enc unchanged; direct C-target vs compiled residual, or separate C-to-input eta plus input-to-output delta; no C-total subtraction or C/L addition',
        numeric_resolution=dict(weighted_measurement_abs_width_target='1/100000000',
            diagnostic_resolution_only_not_u_enc_or_guard_change=True,source_value='proposed, not measured',
            B_scale='B_saved once; B_up is shot range only',state='unchanged encoded vector; divide observables by same nu, do not replace state bytes'),
        future_runtime=dict(point='/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python',
            interval='/home/AbeHiromu/projects/partially-randomized-trotter-worktrees/non-df-h4-c-reference-fix-20261023/.c-reference-fix-venv/bin/python',
            point_environment_has_flint=False,bridge='Reuse frozen JSON state/model; QPY parse in Qiskit environment, preserve binary64 parameters/wires/phase in exact IR; Arb environment read-only',
            execution_seal='new exact runtime binding only after separate execution approval; no environment modifications now'),
        numerical_path=dict(binary64='EMPIRICAL: Qiskit Statevector only; no finite roundoff bound assumed',
            interval='C Arb helpers reusable; local-gate state action/selected-C connector requires technical binding and artificial tests before new scientific SOURCE/grant',
            precision_tiers_bits=[64,96,128],interval_tiers_default_enabled=False,
            escalation='only under separately chosen finite interval schedule, valid enclosure and WIDTH_INSUFFICIENT; no escalation for budget failure or semantic mismatch',
            ieee_binary64_mantissa_bits=53,saved_PF_target_precision_bits=64,
            no_numeric_guarantee_from_precision_label=True),
        preflight=['separate explicit scientific approval','new scientific SOURCE/runtime/input/schema seal and unused one-shot grant',
            'recompute fingerprints from stored QPY, reject layout/mid-measure/reset/condition or unsupported gate before propagation',
            'preserve all phases and C-target parameters; no ideal-body regeneration or recompile',
            'new real-H4 executor must not be the preparation-only toy entrypoint'],
        outcome_states=['EMPIRICAL_ONLY_NO_RHO_UPPER','WIDTH_INSUFFICIENT','LOCAL_BUDGET_MET','LOCAL_BUDGET_EXCEEDED','BUDGET_OVERLAP_STOP','SEMANTIC_MISMATCH_STOP'],
        recovery='diagnosed technical only; preserve successful records, failed native receipts and consumed calls; code changes require new SOURCE/seal/namespace/unused grant',
        completion='native exit/reap plus result SHA/claim/call ledger/STOP; unrelated worktree status or transient PSI is observation only',
        resources='512-amplitude state, 256-amplitude ideal; local gate updates, immutable shared inputs, independent cases parallel when useful; no CPU/RAM/worker ceiling or shared changes')
    r0=dict(common,id='R0',purpose='minimal PF/PR0 cosine pilot with distinct compiler-only gap and matched C-target gap',
        targets=[t['key'] for t in targets if t['axis']=='cosine' and t['index']==0],
        proposed_calls=dict(logical_diagnostics=2,ideal_input_state_actions=2,compiled_state_actions=2,
            new_selected_PR_C_target_actions=1,saved_PF_C_target_reads=1,max_new_state_actions_binary64=5,
            max_logical_tier_calls_if_interval_schedule_explicitly_chosen=6,max_state_actions_for_three_tiers=15),
        sequence='PF0 cosine and PR0 cosine independent; STOP after this R0 only',
        supported='fixed PF/selected PR signal-gap diagnostic; empirical vs enclosed explicitly tagged',
        unsupported='PR law mean, other trajectories/axes, formal certificate or N/G upgrade',
        saved_target_floor='If saved PF target/norm width prevents resolution, STOP; no C rerun under this draft')
    r1=dict(common,id='R1',purpose='order2 product/sign/phase coverage then actual sine wrappers',
        targets=['PR1_cosine','PF0_sine','PR0_sine','PR1_sine'],
        proposed_calls=dict(logical_diagnostics=4,ideal_input_state_actions=4,compiled_state_actions=4,
            new_selected_PR_C_target_actions=1,saved_PF_C_target_reads=1,max_new_state_actions_binary64=9,
            max_logical_tier_calls_if_interval_schedule_explicitly_chosen=12,max_state_actions_for_three_tiers=27),
        reuse='reuse R0 PR0 ideal complex signal for sine; compute PR1 ideal complex signal once for both axes; never reuse a cosine compiled signal as sine',
        sequence='Prefer PR1 cosine if selecting one extra case; sine subset must be explicitly included in a separately approved fixed grant, never expanded automatically',
        precondition='R0 findings reviewed and separate R1 approval; no automatic progression',
        supported='selected order0/order2 and both-axis scope only',unsupported='unobserved order4 or PR law guarantee')
    r2=dict(common,id='R2',purpose='design a uniform or valid law-weighted C-target measurement envelope over all allowed events, including order4',
        targets=[],proposed_calls=dict(logical_diagnostics=0,new_state_actions=0,new_sampling=0,new_compile=0),
        sequence='static design only unless a new finite numerical contract is separately approved',
        required='same effective event law/phase/state/B; uniform per-trajectory envelope or valid sum p_i*d_i; 256 sample mean is insufficient',
        unsupported='assign missing strata zero or infer a full-law rho from selected programs; cost-tail guarantee is a separate problem')
    for r in (r0,r1,r2):require_draft(r)
    return dict(kind='H4_CENTRAL_RHO_DIAGNOSTICS_DRAFT_V1',execution_authorized=False,dispatch_enabled=False,
                grant_valid=False,grant=None,scientific_runner_SOURCE=None,launch_command=None,mandatory_STOP=True,
                automatic_next_stage=False,candidates=[r0,r1,r2])


def mock_interval_diagnostic(target,compiled,*,B,guard,width_target,fixture_id):
    C.require(fixture_id.startswith('TOY_'),'ARTIFICIAL_FIXTURE_REQUIRED')
    C.require(F(B)>0 and F(guard)>0 and F(width_target)>0,'MOCK_SCALE_DOMAIN')
    d=(Interval(*compiled)-Interval(*target))*F(B)
    low=F(0) if d.lo<=0<=d.hi else min(abs(d.lo),abs(d.hi))
    high=max(abs(d.lo),abs(d.hi))
    width=high-low
    status=('WIDTH_INSUFFICIENT' if width>F(width_target) else
            'LOCAL_BUDGET_MET_MOCK' if high<=F(guard) else
            'LOCAL_BUDGET_EXCEEDED_MOCK' if low>F(guard) else 'BUDGET_OVERLAP_STOP')
    return dict(status=status,residual_interval=[str(low),str(high)],width=str(width),
                PR_law_guaranteed=False,formal_certificate_accepted=False)
