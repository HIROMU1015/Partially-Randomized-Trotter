"""Future p7 compile adapter: permit and saved receipts precede science imports."""
from fractions import Fraction
import importlib
from pathlib import Path
import time


def check_saved_inputs(config,permit,contract):
    frozen=permit.json_input('frozen_model.json')
    if contract.digest(frozen)!=config['frozen_model_digest']:raise ValueError('FROZEN_MODEL_CHANGED')
    predictions=contract.predictions(frozen)
    previous=permit.json_input('phase_b_validation.json')
    for name in ('phase_a_terminal.json','phase_b_terminal.json','reaccount_terminal.json'):
        terminal=permit.json_input(name)
        if terminal.get('status')!='COMPLETE_MANDATORY_STOP' or terminal.get('exit_code')!=0:
            raise ValueError('COMPLETE_PREVIOUS_PHASE_REQUIRED:'+name)
    for name in ('phase_a_cleanup.json','phase_b_cleanup.json'):
        cleanup=permit.json_input(name)
        if (cleanup.get('status')!='COMPLETE_MANDATORY_STOP' or cleanup.get('native_exit_code')!=0
                or cleanup.get('all_reaped') is not True or cleanup.get('first_stop') is not None
                or cleanup.get('cleanup_errors')!=[]):raise ValueError('COMPLETE_CLEANUP_REQUIRED:'+name)
    # Reuse the original Phase B validator without changing its scientific code.
    permit.base_contract.validate_predictions(previous['rows'],expected_axes=frozen['validation_axes'],
                                               frozen_model_sha256=contract.MODEL_SHA)
    signal=permit.json_input('p7_signal.json')
    if signal['cell']!=contract.cell() or contract.digest(signal)!=config['p7_signal_digest']:
        raise ValueError('SAVED_P7_SIGNAL_CHANGED')
    receipt=permit.json_input('p7_saved_binding.json')
    if (receipt['p7_signal_digest']!=contract.digest(signal) or receipt['predictions']!=predictions
            or receipt['runtime_closure_digest']!=contract.digest(config['runtime_file_hashes'])
            or config['compiler_options']!=contract.OPTIONS or config['binding']['compiler']!=contract.COMPILER):
        raise ValueError('P7_PROFILE_COMPILER_BINDING')
    scalar=signal['scalar_receipt']
    if scalar['full_Fock_identity'] is not True or signal['allowance']['kind']!='EMPIRICAL':
        raise ValueError('SAVED_SCALAR_OR_ALLOWANCE_POLICY')
    return signal,predictions


def evaluate(config,permit,contract,writer):
    permit.require('p7_compile_validation')
    saved,predictions=check_saved_inputs(config,permit,contract)
    m=importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    native=importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    codec=importlib.import_module('trottertracks.resource_applicability.h6_roundoff_preparation_v1')
    qiskit=importlib.import_module('qiskit')
    permit.take('snapshot_load')
    ham,sector,full,state,_,_=m.load_h6_snapshot(Path(config['input_mapping']['h6_input_snapshot.npz']),
        permit.json_input('snapshot_receipt.json'),permit.json_input('df_receipt.json'))
    if (ham.n_qubits,len(ham.lambdas),sector.dimension)!=(12,19,400):raise ValueError('FIXED_H6_LAYOUT')
    _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
    permit.take('frozen_preparation_decode')
    preparations,archive=codec.read_preparations(config['archive']['path'],config['archive']['sha256'])
    if archive['binding']!=config['archive']['binding']:raise ValueError('ARCHIVE_CHANGED')
    whole=preparations['faithful:p19']
    if whole.hamiltonian_hash!=config['binding']['hamiltonian'] or whole.deterministic_fragment_indices!=tuple(range(19)):
        raise ValueError('FULL_PREPARATION_IDENTITY')
    scalar=whole.constant_coefficient+saved['scalar_receipt']['C']
    permit.take('native_evolution_max')
    evolution=native.build_deterministic_native(whole.deterministic_blocks[:8],num_system_qubits=12,
        T=.8,q=1,formula='2nd',scalar=scalar,control_policy='symmetric_directional',
        max_instructions=contract.plan()['native_instruction_cap'])
    np=m.np;low=np.concatenate((qstate,np.zeros_like(qstate)));high=np.concatenate((np.zeros_like(qstate),qstate))
    permit.take('control_probes_max');a=m.simulate_statevector(evolution.circuit,low)
    permit.take('control_probes_max');b=m.simulate_statevector(evolution.circuit,high)
    expected=complex(saved['signals']['corrected']['real'],saved['signals']['corrected']['imag'])
    overlap=complex(np.vdot(qstate,b[len(qstate):]))
    errors=[float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:len(qstate)])),abs(overlap-expected)]
    m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=max(errors))
    cp=config['compiler_options']
    compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,
                                cp['seed_transpiler'],qiskit.__version__)
    rows=[]
    for prediction in predictions:
        axis=prediction['axis']
        permit.take('measured_wrappers_max')
        wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,
            max_instructions=contract.plan()['native_instruction_cap'])
        permit.take('control_probes_max');probe=wrapper.remove_final_measurements(inplace=False)
        errors.append(abs(m.hadamard_expectation(m.simulate_statevector(probe,low),12)-
                          (overlap.real if axis=='cosine' else overlap.imag)))
        m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=max(errors))
        fingerprint=m.canonical_qiskit_circuit_fingerprint(wrapper)
        writer.write(axis+'_control.json',dict(cell_id=contract.CELL_ID,axis=axis,
            fingerprint=fingerprint,maximum_discrepancy=max(errors),certified=False,
            saved_signal_unchanged=True,full_H_reference_recomputed=False))
        permit.take('native_compile_max')
        writer.write(axis+'_compile_started.json',dict(cell_id=contract.CELL_ID,axis=axis,
            exact_prediction=prediction['exact_prediction'],fingerprint=fingerprint,
            input_instructions=len(wrapper.data),monotonic_seconds=time.monotonic()))
        start=time.monotonic()
        result=m.transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=fingerprint)
        identical=(result.compiler_settings_hash==contract.COMPILER and result.actual_circuit_fingerprint==fingerprint
            and m.canonical_qiskit_circuit_fingerprint(wrapper)==fingerprint
            and len(result.transpiled_circuit.data)<=contract.plan()['transpiled_instruction_cap'])
        error=(abs(Fraction(prediction['exact_prediction'])-result.rz_count)/result.rz_count
               if type(result.rz_count) is int and result.rz_count>0 else None)
        row=dict(cell_id=contract.CELL_ID,axis=axis,exact_prediction=prediction['exact_prediction'],
            predicted=prediction['predicted'],measured=result.rz_count,
            exact_relative_error=str(error) if error is not None else None,model_sha256=contract.MODEL_SHA,
            status='MEASURED_COMPLETE' if identical else 'STOP_IDENTITY',identity_verified=identical,
            compiler_hash=result.compiler_settings_hash,fingerprint=fingerprint,CX=result.cx_count,
            compile_wall_seconds=time.monotonic()-start,maximum_correctness_discrepancy=max(errors))
        writer.write(axis+'_measurement.json',row);rows.append(row)
        contract.validate_axis(row)  # Immediate STOP before another wrapper or compile.
        del result,probe,wrapper
    validation=contract.validate_pair(rows)
    writer.write('validation_complete.json',dict(validation=validation,rows=rows,
        cost_reaccounting_performed=False,signal_recomputed=False,model_refit=False,next_stage_authorized=False))
