"""Future maximum-four-axis validation; never invoked during preparation.

Frozen models/selection are inputs. Disagreement stops; no refitting or automatic
additional compilation is available. Only signal-eligible S4 cases are built.
"""
import importlib
import math
from pathlib import Path


def evaluate(config, permit, contract, writer):
    permit.require('compile_validation')
    frozen = permit.json_input('frozen_model.json')
    if contract.digest(frozen) != config['frozen_model_digest']:
        raise ValueError('FROZEN_MODEL_DIGEST_CHANGED')
    signal_result = permit.json_input('V1_signal_complete.json')
    signal_terminal = permit.json_input('V1_signal_terminal.json')
    if (signal_result.get('status') != 'COMPLETE_MANDATORY_STOP' or
            signal_terminal.get('status') != 'COMPLETE_MANDATORY_STOP' or
            signal_terminal.get('exit_code') != 0 or signal_terminal.get('scope') != 'signal'):
        raise ValueError('COMPLETE_SIGNAL_STAGE_REQUIRED')
    eligible = {r['cell']['id'] for r in signal_result['signals'] if any(
        v['1.0']['cost_acquisition_eligible'] for v in r['precision_rows'].values())}
    axes = contract.select_validation_axes(eligible)
    if frozen['validation_axes'] != axes or len(axes) > 4:
        raise ValueError('SIGNAL_ONLY_VALIDATION_SELECTION')
    if frozen['phase_and_profile_transfer_verified'] is not True:
        raise ValueError('DECLARED_MODEL_STRATUM_TRANSFER_RECEIPTS_REQUIRED')
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
    preps,archive=codec.read_preparations(config['archive']['path'],config['archive']['sha256'])
    if archive['binding']!=config['archive']['binding']:raise ValueError('ARCHIVE_CHANGED')
    whole=preps['faithful:p19']
    cp=config['compiler_options']
    compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],
                              None,None,cp['seed_transpiler'],qiskit.__version__)
    np=m.np; low=np.concatenate((qstate,np.zeros_like(qstate))); high=np.concatenate((np.zeros_like(qstate),qstate))
    rows=[]
    for cid in dict.fromkeys(x['cell']['id'] for x in axes):
        c=next(x['cell'] for x in axes if x['cell']['id']==cid)
        r=next(x for x in signal_result['signals'] if x['cell']['id']==cid)
        scalar=whole.constant_coefficient+(r['scalar_receipt']['C'] if c['method']=='B0I' else 0.)
        permit.take('native_evolution_max')
        evolution=native.build_deterministic_native(whole.deterministic_blocks[:c['prefix']+1],
            num_system_qubits=12,T=.8,q=c['q'],formula=c['formula'],scalar=scalar,
            control_policy='symmetric_directional',max_instructions=1000000)
        permit.take('control_probes_max'); a=m.simulate_statevector(evolution.circuit,low)
        permit.take('control_probes_max'); b=m.simulate_statevector(evolution.circuit,high)
        expected=complex(r['signals']['corrected']['real'],r['signals']['corrected']['imag'])
        z=complex(np.vdot(qstate,b[len(qstate):]))
        errors=[float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:len(qstate)])),abs(z-expected)]
        for axis in contract.AXES:
            permit.take('measured_wrappers_max')
            wrapper=native.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,max_instructions=1000000)
            permit.take('control_probes_max')
            probe=wrapper.remove_final_measurements(inplace=False)
            errors.append(abs(m.hadamard_expectation(m.simulate_statevector(probe,low),12)-
                              (z.real if axis=='cosine' else z.imag)))
            m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=max(errors))
            before=m.canonical_qiskit_circuit_fingerprint(wrapper)
            permit.take('native_compile_max')
            result=m.transpile_and_measure_cost(wrapper,compiler,actual_circuit_fingerprint=before)
            if (result.compiler_settings_hash!=config['binding']['compiler'] or
                    result.actual_circuit_fingerprint!=before or
                    m.canonical_qiskit_circuit_fingerprint(wrapper)!=before or
                    len(result.transpiled_circuit.data)>5000000):raise ValueError('COMPILE_IDENTITY_OR_SIZE')
            hypothesis=next(x for x in frozen['predictions'] if x['cell_id']==cid and x['axis']==axis)
            row=dict(cell_id=cid,axis=axis,predicted=hypothesis['predicted'],measured=result.rz_count,
                model_sha256=config['frozen_model_sha256'],status='MEASURED_COMPLETE',identity_verified=True,
                fingerprint=before,compiler_hash=result.compiler_settings_hash,
                CX=result.cx_count,maximum_correctness_discrepancy=max(errors))
            writer.write(cid+'_'+axis+'.json',row);rows.append(row)
            if result.rz_count<=0 or abs(row['predicted']-result.rz_count)/result.rz_count>=.02:
                raise ValueError('COMPILE_MODEL_DISAGREEMENT_GE_2_PERCENT')
            del result,probe,wrapper
    validation=contract.validate_predictions(rows,expected_axes=axes,frozen_model_sha256=config['frozen_model_sha256'])
    writer.write('validation_complete.json',dict(validation=validation,rows=rows,
        selected_axes=axes,new_science_selection=False,refit=False,automatic_fallback=False))
