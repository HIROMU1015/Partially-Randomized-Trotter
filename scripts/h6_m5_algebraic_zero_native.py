"""One exact-algebra/zero-rounding cosine compile/action; old STOP is retained."""
import json
import math
from pathlib import Path
import time
import traceback
import h6_m5_algebraic_zero_contract as c
from h6_m5_phase_repair_native import atomic_binary, gate_error


def run_science(cfg,cancelled):
    c.authorize(cfg);c.consumed(cfg)
    import numpy as np
    import qiskit,mpmath,symengine
    from qiskit import qpy
    import h6_m5_phase_engine as engine
    from m5_algebraic_zero_compiler import compile_with_algebraic_zero
    c.require(qiskit.__version__=='1.3.0' and np.__version__=='1.26.4' and mpmath.__version__=='1.3.0' and mpmath.libmp.BACKEND=='python','FROZEN_ALGEBRAIC_ZERO_RUNTIME')
    sealed={r['path'] for r in cfg['sealed_files']}
    c.require(all(str(Path(m.__file__).resolve()) in sealed for m in (np,qiskit,mpmath,symengine)),'IMPORTED_ALGEBRAIC_ZERO_RUNTIME_SEALED')
    out=Path(cfg['output']);inputs=cfg['inputs'];started=time.monotonic()
    def progress(phase):
        c.require(not cancelled.is_set(),'OWN_ALGEBRAIC_ZERO_SIGNAL')
        with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(dict(phase=phase,elapsed=time.monotonic()-started))+'\n')
    c.reserve(cfg,'LOAD_SAVED_PASS_INPUT');progress('load_saved_PASS_input')
    with Path(inputs['lowered_qpy']['path']).open('rb') as f:
        circuits=qpy.load(f);c.require(len(circuits)==1 and f.read()==b'','ONE_SAVED_LOWERED_QPY')
    candidate=circuits[0];schema=engine.validate_circuit(candidate,'COMPILED',13)
    c.require(sum(schema['gate_counts'].values())==4076368,'SAVED_CANDIDATE_OPERATION_COUNT')
    arrays={}
    for role,key in [('initial_npz','low'),('lowered_vector','vector'),('ordinary_vector','vector')]:
        with np.load(inputs[role]['path'],allow_pickle=False) as z:
            c.require(z.files==[key],'FIXED_ARRAY_KEYS:'+role);v=z[key].copy()
        c.require(v.dtype==np.dtype('complex128'),'ORIGINAL_BINARY64_ARRAY');engine.finite_vector(v,8192)
        arrays[role]=v
    low,before,reference=arrays['initial_npz'],arrays['lowered_vector'],arrays['ordinary_vector']
    c.require(not np.count_nonzero(low[4096:]),'SAVED_LOW_ANCILLA_ZERO')
    reused=dict(c.read(inputs['lowered_gate']['path']),reused=True,
        origin_gate_sha256=inputs['lowered_gate']['sha256'],origin_vector_sha256=inputs['lowered_vector']['sha256'],new_action=0)
    c.save(out/'lowered_vs_saved_wrapper_vector_REUSED.json',reused)
    c.save(out/'input_receipt.json',dict(status='SAVED_FIRST_PASS_ACCEPTED_WITH_OLD_STOP_RETAINED',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],schema=schema,
        inputs=[dict(role=k,sha256=v['sha256'],bytes=v['bytes']) for k,v in inputs.items()],new_lowering=0,new_lowered_replay=0,normalized=False,phase_corrected=False))
    c.reserve(cfg,'COMPILE_COSINE');progress('compile_cosine_algebraic_zero_control')
    with (out/'compiler_passes.jsonl').open('x') as stream:
        def observe(**kwargs):
            c.require(not cancelled.is_set(),'OWN_SIGNAL_IN_ALGEBRAIC_ZERO_COMPILER')
            dag=kwargs['dag'];phase=float(dag.global_phase);counts=dict(dag.count_ops())
            c.require(math.isfinite(phase),'FINITE_ALGEBRAIC_ZERO_CALLBACK_PHASE')
            stream.write(json.dumps(dict(pass_name=kwargs['pass_'].name(),count=kwargs['count'],pass_seconds=kwargs['time'],global_phase_binary64_hex=phase.hex(),gate_counts=counts,
                unexpected_names=sorted(set(counts)-{'rz','sx','x','cx','id','barrier','measure'})))+'\n');stream.flush()
        compiled=compile_with_algebraic_zero(candidate,phase_journal=out/'algebraic_passes.jsonl',callback=observe,
            basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17,num_processes=1)
    algebraic=[json.loads(line) for line in (out/'algebraic_passes.jsonl').read_text().splitlines()]
    c.require(algebraic and all(row['policy']==c.POLICY and row['spent_after']['numerator']=='0' and row['budget']['numerator']=='0' and
        row['global_phase_bits_unchanged'] and not row['Euler_resynthesis'] and not row['angle_wrap'] and not row['small_angle_threshold']
        for row in algebraic),'EXACT_ALGEBRAIC_ZERO_JOURNAL_BEFORE_ACTION')
    c.save(out/'algebraic_guard.json',dict(PASS=True,policy=c.POLICY,passes=len(algebraic),additional_angle_rounding=0,global_phase_bits_unchanged=True))
    progress('compile_finished')
    artifact=atomic_binary(out/'compiled.qpy',lambda f:qpy.dump(compiled,f))
    compiled_schema=engine.validate_circuit(compiled,'COMPILED',13)
    c.save(out/'compiled_receipt.json',dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],artifact=artifact,schema=compiled_schema,
        compiler=c.scope()['compiler'],compiler_policy=c.POLICY,status='UNVALIDATED_DIAGNOSTIC_NOT_FOR_ACCOUNTING'))
    del candidate,circuits
    c.reserve(cfg,'REPLAY_COMPILED');progress('replay_compiled')
    after=engine.ordinary(compiled,low);progress('replay_compiled_finished')
    vector=atomic_binary(out/'compiled_vector.npz',lambda f:np.savez(f,vector=after))
    checks=[reused,gate_error('compiled_vs_lowered',after,before,out)]
    checks.append(gate_error('compiled_vs_saved_wrapper_vector',after,reference,out))
    c.verify_seals(cfg);c.consumed(cfg)
    result=dict(status='ALGEBRAIC_ZERO_SINGLE_STATE_PASS_MANDATORY_STOP',SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],
        checks=checks,artifacts=dict(compiled_qpy=artifact,compiled_vector=vector),compiled_schema=compiled_schema,
        compiler_policy=c.POLICY,algebraic_passes_sha256=c.sha(out/'algebraic_passes.jsonl'),
        new_compile=1,statevector_replays=1,reused_initial_raw_gate=1,new_lowering=0,new_lowered_replay=0,
        accepted_cost_rows=0,sine=0,sampling=0,old_cost_compatibility=False,old_STOP_remains_STOP=True,old_58_axes_preserved=True,
        phase_alignment_for_acceptance=False,postcompile_phase_correction=False,AD03='DEFERRED',retry=False,next_phase_started=False,mandatory_STOP=True,wall_seconds=time.monotonic()-started)
    c.save(out/'report.json',result)
    return result


def worker(cfg,cancelled):
    try:run_science(cfg,cancelled)
    except BaseException as exc:
        c.save(Path(cfg['output'])/'STOP.json',dict(status='STOP',reason=str(exc),traceback=traceback.format_exc(),SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
            old_STOP_remains_STOP=True,retry=False,next_phase_started=False,mandatory_STOP=True))
