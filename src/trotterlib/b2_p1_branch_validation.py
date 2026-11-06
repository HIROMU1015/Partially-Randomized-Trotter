"""Saved raw means -> unchanged RPE decoder. No signal generation or shots.

The original-H energy reference is loaded only after all decoder steps finish.
Private files/operators/states are neither read nor reconstructed here.
"""
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from .rpe_four_round_phase_validation import (
    reconstruct_rpe_phase, _vectorized_reconstruction, _oracle_branch,
    _wrap_phase, _phase_distance,
)
from .rpe_resource_accounting import RPE_STRICT_BRANCH_CERTIFICATION_LIMIT

ROOT=Path(__file__).resolve().parents[2]
ALL_COMMIT='d7a55ac181de71cd2b2c49644810373cafd0a450'
ALL_PATH='artifacts/track_a_b2_p1_remaining_rounds/2026-11-07/run01/all_18_rounds.json'
NEW_PATH='artifacts/track_a_b2_p1_remaining_rounds/2026-11-07/run01/result.json'
END_COMMIT='ea2c0915ba2f7e11a6dac58dc01cf1af3c0d0ff5'
END_PATH='artifacts/track_a_b2_p1_two_round/2026-11-07/run01/result.json'
DECODER_COMMIT='66a41ad9219f12377e7d325e272c1a90d66ab7e5'
DECODER_PATH='src/trotterlib/rpe_four_round_phase_validation.py'
ACCOUNTING_PATH='src/trotterlib/rpe_resource_accounting.py'
STATE_COMMIT='89165cf70958e7e4676d0580e3acb7faf588772c'
STATE_RESULT='artifacts/b2_hphys_ground_state/2026-11-06/run01/result.json'
DELTA=.02
BETA=.4
EPSILON=.00015936001019904
PRIOR=(-128.,128.)
U=1e-8
# Existing _vectorized_reconstruction and saved V2 branch diagnostic definition.
# This is not a new P1 science/phase allowance.
BRANCH_AMBIGUITY_ATOL=1e-12


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def require(condition,name):
    if not condition:
        raise ValueError(name)


def fixed_json(commit,path):
    raw=git('show',commit+':'+path)
    return json.loads(raw),dict(commit=commit,path=path,
        blob=git('rev-parse',commit+':'+path).decode().strip(),
        sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def finite(value):
    require(not isinstance(value,bool) and math.isfinite(float(value)),'FINITE_NUMBER_REQUIRED')
    return float(value)


def complex_signal(record):
    z=complex(finite(record['real']),finite(record['imag']))
    require(abs(z)>0,'NONZERO_SIGNAL_REQUIRED')
    return z


def validate_saved_bundle(all_rounds,new,end):
    """Identity/evidence only; no PF/RTE/radius/model error reevaluation."""
    require(all_rounds['status']=='ALL_18_P1_CONDITIONAL_SIGNAL_PASS','ALL18_STATUS')
    require(new['execution_SOURCE']=='071cb00955379b0bf4da149bd9793d4125968cad' and
            end['execution_SOURCE']=='5ef55f923e73e61bada383d0cfe98283e7984e9d','SIGNAL_SOURCE')
    conditions=dict(new['fixed_conditions']);conditions.pop('requested_rounds')
    other=dict(end['fixed_conditions']);other.pop('requested_rounds')
    require(conditions==other,'ENDPOINT_CURRENT_SCIENTIFIC_CONDITIONS')
    expected=dict(target='H_phys',DF_rank=12,L_D=3,r=4,K=2,delta=DELTA,
        energy_precision_Ha=EPSILON,success_probability_target=.95,beta_RPE=BETA,
        energy_prior_Ha=list(PRIOR),energy_shift_Ha=0,phase_convention='exp(-itH); phi=-delta E',
        caps_rad=dict(model=.02,PF=.06,RTE=.04,statistical=.24,implementation=.02,reserve=.02),
        allocation=dict(beta_pf_budget=.06,beta_rte_budget=.04,beta_stat_budget=.24,
                        alpha_cosine=.05/36,alpha_sine=.05/36),
        numerical_allowance=U,independent_match_tolerance=U/10)
    require(conditions==expected,'FIXED_P1_CONDITIONS')
    expected_identity=dict(
        original_hamiltonian='H_phys:S0_array_hash:bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2',
        original_state='S0_array_hash:1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db',
        df_hamiltonian='de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424',
        state_role='original_H_ground_state',
        sector='Nalpha=2,Nbeta=2;OF_basis_S0_hash:665f72f65a4300043a32f3aabc76b7deafbd2563ad3947228dd937bc6d521dd7',
        spin_orbital_order='2p=alpha,2p+1=beta;Qiskit full index=sum(n_q*2**q)',
        state_preparation_included=False,DF_state_used=False)
    files={'state.npz':'deb561367e216fd4c62d013c10a4adbbef5da8c3e8139c39170d71f80530a80c',
        'H4_sto-3g_singlet_df_d100.hdf5':'a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958'}
    for evidence in (new,end):
        require(evidence['input_identity']['status']=='PASS','PRIOR_INPUT_STATUS')
        require(all(evidence['input_identity'][k]==v for k,v in expected_identity.items()),'PHYSICAL_CONTEXT')
        require(all(evidence['input_identity']['file_SHA256'][k]==v for k,v in files.items()),'INPUT_FILE_IDENTITY')
        for path,expected_hash in evidence['SOURCE_SHA256'].items():
            require(hashlib.sha256(git('show',evidence['execution_SOURCE']+':'+path)).hexdigest()==expected_hash,
                    'SAVED_SIGNAL_SOURCE_HASH')
    require([r['m'] for r in new['rounds']]==list(range(1,17)),'NEW_ROUND_ORDER')
    require([r['m'] for r in end['rounds']]==[0,17],'ENDPOINT_ROUND_ORDER')
    rows=all_rounds['rounds']
    require([r['m'] for r in rows]==list(range(18)),'ALL18_ROUND_ORDER')
    originals={r['m']:r for evidence in (new,end) for r in evidence['rounds']}
    selected=[]
    for m,row in enumerate(rows):
        copy=dict(row);kind=copy.pop('evidence');commit=copy.pop('result_commit')
        require(kind==('reused' if m in (0,17) else 'new'),'ROUND_PROVENANCE_KIND')
        require(commit==(END_COMMIT if m in (0,17) else None),'ROUND_RESULT_REFERENCE')
        require(copy==originals[m],'ROUND_BYTES_VALUES_PROVENANCE')
        if m not in (0,17):
            require(row['execution_SOURCE']==new['execution_SOURCE'],'ROUND_EXECUTION_SOURCE')
        require(row['status']=='CONDITIONAL_PASS' and row['qualification']['feasible'],'ROUND_QUALIFICATION')
        require(type(row['Q']) is int and row['Q']==1<<m and row['T_Ha_inverse']==(1<<m)*DELTA,'ROUND_Q_TIME')
        raw=complex_signal(row['signals']['raw']);corrected=complex_signal(row['signals']['finite'])
        B=finite(row['normalization']);A=finite(row['attenuation'])
        require(B>=1 and 0<A<=1 and abs(B*A-1)<=U,'POSITIVE_NORMALIZATION')
        require(abs(raw-A*corrected)<=U,'RAW_CORRECTED_RELATION')
        phase=float(np.angle(raw));corrected_phase=float(np.angle(corrected))
        require(_phase_distance(phase,row['signals']['raw']['phase_rad'])<=U,'SAVED_RAW_PHASE_CONVENTION')
        require(_phase_distance(phase,corrected_phase)<=U,'NORMALIZATION_PHASE_SIGN')
        selected.append(dict(m=m,Q=row['Q'],T_Ha_inverse=row['T_Ha_inverse'],raw=raw,
            raw_phase=phase,corrected_phase=corrected_phase,normalization=B,attenuation=A,
            signal_result_commit=END_COMMIT if m in (0,17) else ALL_COMMIT,
            signal_result_path=END_PATH if m in (0,17) else NEW_PATH,
            signal_execution_SOURCE=end['execution_SOURCE'] if m in (0,17) else new['execution_SOURCE'],
            normalization_phase_difference=float(_phase_distance(phase,corrected_phase))))
    return selected,expected_identity,conditions


def initial_alias(raw_phase,*,energy_interval=PRIOR,delta=DELTA,beta=BETA):
    """P1 initial interval guard, independently of any ground energy oracle."""
    low,high=map(finite,energy_interval)
    require(low<high and delta>0 and beta>0,'ALIAS_PARAMETERS')
    phase_low,phase_high=-delta*high,-delta*low
    margin=min(phase_low+math.pi,math.pi-phase_high)-beta
    valid=margin>0 and phase_low-beta<=raw_phase<=phase_high+beta
    return dict(status='PASS' if valid else 'STOP_INITIAL_ALIAS',
        energy_prior_Ha=[low,high],energy_shift_Ha=0,delta_Ha_inverse=delta,
        principal_phase_interval='[-pi,pi)',principal_energy_interval='(-pi/delta,pi/delta]',
        prior_base_phase_interval=[phase_low,phase_high],
        prior_with_error_phase_interval=[phase_low-beta,phase_high+beta],
        interval_endpoint_phase_margin_after_beta_rad=margin,initial_raw_phase=raw_phase,
        raw_phase_consistent_with_prior_and_beta=phase_low-beta<=raw_phase<=phase_high+beta,
        initial_branch=0,reference_energy_used=False,
        qualification='saved coefficient-sum prior adopted by P1; not a rigorous interval certificate')


def decode_raw_means(raw_signals,q_values,*,energy_interval=PRIOR,delta=DELTA,beta=BETA):
    """No known energy, ideal-H phase or corrected signal is accepted here."""
    require(len(raw_signals)==len(q_values) and len(q_values)>0,'ROUND_GRID_LENGTH')
    require(all(math.isfinite(z.real) and math.isfinite(z.imag) and abs(z)>0 for z in raw_signals),
            'FINITE_NONZERO_RAW_MEANS')
    phases=[float(np.angle(z)) for z in raw_signals]
    alias=initial_alias(phases[0],energy_interval=energy_interval,delta=delta,beta=beta)
    require(alias['status']=='PASS','INITIAL_ALIAS_INELIGIBLE')
    require(beta<RPE_STRICT_BRANCH_CERTIFICATION_LIMIT,'STRICT_BRANCH_PRECONDITION')
    result=reconstruct_rpe_phase(phases,q_values)
    # Existing second entry point, same unchanged candidate/distance rule.
    second,branches,ambiguous=_vectorized_reconstruction([np.asarray([v]) for v in phases],q_values)
    rows=[]
    for m,(phase,row) in enumerate(zip(phases,result['rounds'],strict=True)):
        q=row['q_m'];base=row['base_phase_estimate']
        gap=None if m==0 else row['next_smallest_branch_distance']-row['distance_from_previous']
        tie=False if m==0 else bool(np.isclose(row['distance_from_previous'],
            row['next_smallest_branch_distance'],rtol=0.,atol=BRANCH_AMBIGUITY_ATOL))
        branch_match=row['selected_branch']==int(branches[m][0])
        rows.append(dict(row,m=m,T_Ha_inverse=q*delta,input_raw_phase=phase,
            wrapped_round_phase=row['measured_round_phase'],
            unwrapped_round_phase=q*base,reconstructed_base_phase=base,energy_estimate_Ha=-base/delta,
            best_runner_up_distance_gap=gap,ambiguity_within_principal_window=tie,
            scalar_vectorized_branch_match=branch_match,
            beta_RPE=beta,base_phase_tolerance_rad=beta/q,nominal_energy_resolution_Ha=beta/(q*delta)))
    unique=not bool(ambiguous[0]) and not any(r['ambiguity_within_principal_window'] for r in rows)
    match=all(r['scalar_vectorized_branch_match'] for r in rows) and float(second[0])==result['final_phase_estimate']
    return dict(rounds=rows,initial_alias=alias,unique_saved_mean_branches=unique,
        scalar_vectorized_match=match,all_decoding_complete=True,
        status='RAW_MEAN_DECODE_PASS' if unique and match else 'STOP_BRANCH_AMBIGUITY_OR_MISMATCH',
        final_base_phase=result['final_phase_estimate'],final_energy_estimate_Ha=-result['final_phase_estimate']/delta,
        decoder_input='Arg(saved raw means)',reference_energy_used_in_decoder=False,
        strict_branch_limit_rad=RPE_STRICT_BRANCH_CERTIFICATION_LIMIT,
        strict_branch_margin_rad=RPE_STRICT_BRANCH_CERTIFICATION_LIMIT-beta)


def posthoc_correctness(decoded,raw_phases,energy_reference,*,delta=DELTA,beta=BETA,epsilon=EPSILON):
    """Called only after complete decoding. No branch is selected or repaired."""
    require(decoded['all_decoding_complete'] and decoded['status']=='RAW_MEAN_DECODE_PASS','DECODE_BEFORE_REFERENCE')
    reference=finite(energy_reference)
    true_base=-delta*reference
    require(PRIOR[0]<=reference<=PRIOR[1],'POSTHOC_REFERENCE_OUTSIDE_ADOPTED_PRIOR')
    require(-math.pi<true_base<math.pi,'REFERENCE_OUTSIDE_ADOPTED_ALIAS')
    all_oracle=True;all_phase=True;all_energy=True
    for row,phase in zip(decoded['rounds'],raw_phases,strict=True):
        q=row['q_m']
        error=float(_phase_distance(phase,q*true_base))
        oracle=int(_oracle_branch(np.asarray(phase),q,_wrap_phase(true_base)))
        energy_error=abs(row['energy_estimate_Ha']-reference)
        row.update(posthoc_common_target_phase_error_rad=error,
            posthoc_common_target_phase_within_beta=error<=beta,
            posthoc_oracle_branch=oracle,posthoc_oracle_agreement=row['selected_branch']==oracle,
            energy_absolute_error_Ha=energy_error,
            energy_within_round_nominal_resolution=energy_error<=row['nominal_energy_resolution_Ha'])
        all_oracle &= row['posthoc_oracle_agreement']
        all_phase &= error<=beta
        all_energy &= row['energy_within_round_nominal_resolution']
    final_error=abs(decoded['final_energy_estimate_Ha']-reference)
    nominal=beta/(decoded['rounds'][-1]['q_m']*delta)
    sufficient=(decoded['strict_branch_margin_rad']>0 and decoded['initial_alias']['status']=='PASS' and all_phase)
    return dict(reference_energy_Ha=reference,reference_role='post-decoder correctness oracle only',
        posthoc_reference_inside_adopted_prior=True,
        all_round_oracle_agreement=bool(all_oracle),all_saved_mean_phases_within_common_beta=bool(all_phase),
        all_round_energy_errors_within_nominal_resolution=bool(all_energy),
        final_energy_absolute_error_Ha=final_error,energy_precision_target_Ha=epsilon,
        final_within_energy_target=final_error<=epsilon,nominal_final_resolution_Ha=nominal,
        nominal_horizon_meets_target=nominal<=epsilon,
        conditional_branch_sufficient_condition='SATISFIED' if sufficient else 'UNASSESSED',
        robustness_scope='existing strict beta<pi/3 condition if every supplied phase is within beta of the same original-H phase; no perturbation sampling',
        arbitrary_perturbations_centered_on_raw_means='UNASSESSED',
        quantum_measurement_95_percent_guarantee='NOT_CERTIFIED',shots_prices_total_cost='NOT_EVALUATED',
        status='P1_CONDITIONAL_BRANCH_ENERGY_PASS' if all_oracle and all_phase and all_energy and
            final_error<=epsilon and nominal<=epsilon else 'STOP_POSTHOC_CORRECTNESS')


def run(output):
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=git('rev-parse','HEAD').decode().strip()
    paths=['src/trotterlib/b2_p1_branch_validation.py','scripts/run_b2_p1_branch_validation.py',DECODER_PATH,ACCOUNTING_PATH]
    for path in paths:
        require((ROOT/path).read_bytes()==git('show',source+':'+path),'COMMITTED_EXECUTION_SOURCE')
    for path in (DECODER_PATH,ACCOUNTING_PATH):
        require((ROOT/path).read_bytes()==git('show',DECODER_COMMIT+':'+path),'UNCHANGED_DECODER_SOURCE')
    result=dict(schema_version='b2_p1_saved_raw_branch_energy_v1',execution_SOURCE=source,
        SOURCE_SHA256={path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
        started_UTC=datetime.now(timezone.utc).isoformat(),mandatory_stop=True,
        guarantee_status='not_certified',forbidden_work=dict(state_operator_generation=0,signal_simulation=0,
            private_input_reads=0,sampling=0,circuit_build_compile=0,shot_accounting=0,
            price_MC=0,total_cost=0,baseline_evaluation=0,scientific_review=0))
    error_to_raise=None
    try:
        combined,cm=fixed_json(ALL_COMMIT,ALL_PATH)
        new,nm=fixed_json(ALL_COMMIT,NEW_PATH)
        endpoints,em=fixed_json(END_COMMIT,END_PATH)
        result['saved_inputs']=[cm,nm,em]
        selected,identity,conditions=validate_saved_bundle(combined,new,endpoints)
        result.update(scientific_context=identity,fixed_conditions=conditions,
            input_validation='PASS',branch_ambiguity_atol=BRANCH_AMBIGUITY_ATOL,
            ambiguity_definition_SOURCE=DECODER_COMMIT+':'+DECODER_PATH+'::_vectorized_reconstruction')
        raw=[row['raw'] for row in selected];q=[row['Q'] for row in selected]
        result['initial_alias']=initial_alias(float(np.angle(raw[0])))
        require(result['initial_alias']['status']=='PASS','INITIAL_ALIAS_INELIGIBLE')
        decoded=decode_raw_means(raw,q)
        result['reconstruction']=decoded
        require(decoded['status']=='RAW_MEAN_DECODE_PASS','RAW_BRANCH_RECONSTRUCTION')
        # Boundary: no original ground-energy record is opened before this point.
        reference,rm=fixed_json(STATE_COMMIT,STATE_RESULT)
        result['posthoc_reference_input']=rm
        result['reference_loaded_after_all_rounds_decoded']=True
        correctness=posthoc_correctness(decoded,[r['raw_phase'] for r in selected],reference['validation']['energy_Ha'])
        result['correctness']=correctness;result['status']=correctness['status']
        for row,evidence in zip(decoded['rounds'],selected,strict=True):
            row['saved_signal_evidence']={k:v for k,v in evidence.items() if k!='raw'}
        fields=['m','q_m','T_Ha_inverse','input_raw_phase','wrapped_round_phase','unwrapped_round_phase',
            'reconstructed_base_phase','selected_branch','distance_from_previous',
            'next_smallest_branch_distance','best_runner_up_distance_gap',
            'ambiguity_within_principal_window','energy_estimate_Ha','energy_absolute_error_Ha',
            'posthoc_common_target_phase_error_rad','base_phase_tolerance_rad','nominal_energy_resolution_Ha']
        with (output/'rounds.csv').open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
            writer.writerows(decoded['rounds'])
        print(result['status'],flush=True)
    except Exception as error:
        result['status']='STOP_'+type(error).__name__
        result['stop_reason']=str(error) if isinstance(error,ValueError) else type(error).__name__
        error_to_raise=error
    result['elapsed_seconds']=time.perf_counter()-started
    result['environment']=dict(host=platform.node(),python=platform.python_version(),
        numpy=importlib.metadata.version('numpy'),cpu_affinity=len(os.sched_getaffinity(0)),
        BLAS_env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},gpu_used=False)
    (output/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    if error_to_raise is not None:
        raise error_to_raise
    return result
