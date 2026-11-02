"""Reuse original cosine inputs; build only sine's terminal Sdg and compile it."""
import math
from pathlib import Path
import time
import traceback
import h6_m5_ct01_sine_contract as c


def sine_wrapper(cosine):
    from qiskit.circuit import CircuitInstruction,Measure
    from qiskit.circuit.library import HGate,SdgGate
    c.require(cosine.num_qubits>=2 and cosine.num_clbits==1 and not cosine.parameters and cosine.layout is None,'BOUND_SAVED_COSINE_WRAPPER')
    n=cosine.num_qubits-1;items=cosine.data
    c.require(len(items)>=3 and items[0].operation.base_class is HGate and cosine.find_bit(items[0].qubits[0]).index==n,'ORIGINAL_LEADING_H')
    h,m=items[-2:]
    c.require(h.operation.base_class is HGate and m.operation.base_class is Measure and
      [cosine.find_bit(x.qubits[0]).index for x in (h,m)]==[n,n] and cosine.find_bit(m.clbits[0]).index==0,'ORIGINAL_TERMINAL_H_MEASURE')
    c.require(sum(x.operation.name=='measure' for x in items)==1 and not any(x.operation.name=='sdg' for x in items),'COSINE_NOT_SINE_NO_MID_MEASURE')
    result=cosine.copy();result.data.insert(len(result.data)-2,CircuitInstruction(SdgGate(),(result.qubits[n],),()))
    c.require(len(result.data)==len(items)+1 and float(result.global_phase).hex()==float(cosine.global_phase).hex(),'ONLY_TERMINAL_SDG_INSERTED')
    return result


def expected_sine_from_cosine(before):
    """H Sdg H on the saved final ancilla; no full uncompiled replay."""
    import numpy as np
    c.require(before.dtype==np.dtype('complex128') and before.ndim==1 and len(before)>=4 and len(before)&(len(before)-1)==0 and np.isfinite(before).all(),'SAVED_BINARY64_POWER_TWO_VECTOR')
    half=len(before)//2;a=(before[:half]+before[half:])/math.sqrt(2);b=(before[:half]-before[half:])/math.sqrt(2)
    result=np.concatenate(((a-1j*b)/math.sqrt(2),(a+1j*b)/math.sqrt(2)))
    c.require(np.isfinite(result).all(),'FINITE_TERMINAL_TRANSFORM')
    return result


def check_error(name,value,threshold,out):
    row=dict(check=name,raw_error=float(value) if math.isfinite(value) else None,strictly_below=threshold,
      PASS=math.isfinite(value) and 0<=value<threshold,phase_alignment=False)
    c.save(Path(out)/(name+'.json'),row);c.require(row['PASS'],'NUMERIC_DISAGREEMENT:'+name);return row


def run_science(cfg,cancelled):
    c.authorize(cfg);c.consumed(cfg)
    import h46_m5_native as original
    original.install(cfg)  # Same original provider and preset opt1 in this process.
    import numpy as np,qiskit
    from qiskit import qpy
    from trotterlib.df_trotter.circuit import simulate_statevector
    from trotterlib.rte_compiled_cost import CompilerSettings,transpile_and_measure_cost
    from trottertracks.resource_applicability.ax2b_stream_fingerprint_v5 import canonical_qiskit_circuit_fingerprint as fingerprint
    from h6_m5_recovery_native import atomic_binary
    from h6_m5_phase_engine import validate_circuit,finite_vector
    c.require(qiskit.__version__=='1.3.0' and np.__version__=='1.26.4','ORIGINAL_CT01_RUNTIME')
    sealed={r['path'] for r in cfg['sealed_files']}
    c.require(all(str(Path(m.__file__).resolve()) in sealed for m in (np,qiskit)),'LOADED_ORIGINAL_RUNTIME_IDENTITY')
    out=Path(cfg['output']);inputs=cfg['inputs'];started=time.monotonic()
    def progress(phase):
        c.require(not cancelled.is_set(),'OWN_CT01_SIGNAL')
        with (out/'progress.jsonl').open('a') as f:
            import json
            f.write(json.dumps(dict(phase=phase,elapsed=time.monotonic()-started))+'\n')
    c.reserve(cfg,'RESTORE_SAVED_INPUTS');progress('restore_saved_cosine_inputs')
    with Path(inputs['wrapper_qpy']['path']).open('rb') as f:
        circuits=qpy.load(f);c.require(len(circuits)==1 and f.read()==b'','ONE_SAVED_COSINE_QPY')
    cosine=circuits[0];validate_circuit(cosine,'WRAPPER',13)
    c.require(fingerprint(cosine)==cfg['cosine_acceptance']['wrapper_fingerprint'],'EXACT_SAVED_COSINE_FINGERPRINT')
    sine=sine_wrapper(cosine)
    with np.load(inputs['initial_npz']['path'],allow_pickle=False) as z:
        c.require(z.files==['low'],'ORIGINAL_INITIAL_KEY');low=z['low'].copy()
    with np.load(inputs['saved_vectors_npz']['path'],allow_pickle=False) as z:
        c.require(set(z.files)=={'low','before','after','independent_expected_high'} and np.array_equal(z['low'],low),'SAVED_COSINE_VECTOR_JOIN')
        before=expected_sine_from_cosine(z['before'])
    for v in (low,before):finite_vector(v,8192);c.require(v.dtype==np.dtype('complex128'),'ORIGINAL_DTYPE')
    half=len(before)//2;observed=float(np.vdot(before[:half],before[:half]).real-np.vdot(before[half:],before[half:]).real)
    expected=c.read(inputs['saved_signal']['path'])['raw']['imag']
    checks=[check_error('sine_terminal_transform_vs_saved_signal',abs(observed-expected),1e-9,out)]
    c.save(out/'saved_cosine_acceptance.json',cfg['cosine_acceptance'])
    wrapper=atomic_binary(out/'sine_wrapper.qpy',lambda f:qpy.dump(sine,f))
    expected_artifact=atomic_binary(out/'expected_sine.npz',lambda f:np.savez(f,vector=before))
    c.save(out/'input_receipt.json',dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],new_cosine_compile=0,
      new_native_or_oracle=0,new_full_H_reference=0,new_uncompiled_full_replay=0,terminal_transform='H_SDG_H_ON_SAVED_COSINE',
      normalization=False,phase_alignment=False,wrapper=wrapper,expected_sine=expected_artifact))
    c.reserve(cfg,'COMPILE_SINE');progress('compile_sine_original_preset_opt1')
    compiler=CompilerSettings(('rz','sx','x','cx'),None,None,1,None,None,17,qiskit.__version__)
    fp=fingerprint(sine);cost=transpile_and_measure_cost(sine,compiler,actual_circuit_fingerprint=fp);progress('compile_finished')
    c.require(cost.compiler_settings_hash==c.COMPILER_HASH and cost.actual_circuit_fingerprint==fp==fingerprint(sine),'ORIGINAL_COMPILED_IDENTITY')
    validate_circuit(cost.transpiled_circuit,'COMPILED',13)
    compiled=atomic_binary(out/'compiled_sine.qpy',lambda f:qpy.dump(cost.transpiled_circuit,f))
    metrics={k:getattr(cost,k) for k in ('rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size')}
    c.save(out/'compiled_receipt.json',dict(status='UNVALIDATED_NOT_FOR_ACCOUNTING',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],compiler_settings_hash=c.COMPILER_HASH,wrapper_fingerprint=fp,artifact=compiled,metrics=metrics))
    c.reserve(cfg,'REPLAY_COMPILED_SINE');progress('replay_compiled_sine')
    after=simulate_statevector(cost.transpiled_circuit.remove_final_measurements(inplace=False),low);progress('replay_finished')
    vector=atomic_binary(out/'compiled_sine_vector.npz',lambda f:np.savez(f,vector=after))
    checks.append(check_error('compiled_vector_sine',float(np.linalg.norm(after-before)),c.TOLERANCE,out))
    c.require(all(type(v) is int and v>=0 for v in metrics.values()) and metrics['rz_count']>0,'INTEGER_SINE_METRICS')
    c.verify_seals(cfg);c.consumed(cfg)
    c.save(out/'cost_sine.json',dict(status='PILOT_AXIS_ACCEPTED_CT01',axis='sine',metrics=metrics,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],compiler_settings_hash=c.COMPILER_HASH,wrapper_fingerprint=fp))
    result=dict(status='CT01_SINE_PASS_MANDATORY_STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
      checks=checks,reused_cosine=cfg['cosine_acceptance'],new_compile=1,compiled_state_actions=1,new_uncompiled_full_replays=0,
      new_cosine_compile=0,new_sampling=0,new_signal_or_oracle=0,full_H_reference_generation=0,compiler_policy='ORIGINAL_PRESET_OPT1_UNMODIFIED',
      artifacts=dict(sine_wrapper=wrapper,expected_sine=expected_artifact,compiled_sine=compiled,compiled_vector=vector),
      pilot_axis_receipts=2,main_table_adoption=False,cost_accounting=False,ranking=False,old_STOP_preserved=True,old_58_axes_preserved=True,
      AD03='DEFERRED',retry=False,next_phase_started=False,mandatory_STOP=True)
    c.save(out/'report.json',result);return result


def worker(cfg,cancelled):
    try:run_science(cfg,cancelled)
    except BaseException as exc:c.save(Path(cfg['output'])/'STOP.json',dict(status='STOP',reason=str(exc),traceback=traceback.format_exc(),SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],retry=False,mandatory_STOP=True))
