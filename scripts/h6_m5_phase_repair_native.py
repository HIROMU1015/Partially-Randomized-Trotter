"""Isolated authorized adapter; copied numeric core is tested against the prototype."""
from collections import Counter
from fractions import Fraction
import math
import os
from pathlib import Path
import time
import traceback
from m5_phase_repair_prototype import _types, _supported, _rounded_phase, exact, payload, require
import h6_m5_phase_repair_contract as c

def _lower_native_core(circuit):
    """Experimental explicit native-to-basis lowering, never installed.

    Avoid parameterized R/RY equivalence routes: RY(t) is S H RZ(t) H Sdg
    in matrix order. Expand XXPlusYY using its checked built-in definition,
    then use only RZ/SX/X/CX and one exact scalar sum. Fixed Clifford angles
    are still binary64 approximations; whole-operator tests and a future raw
    molecular check are indispensable. This is a new compiler-input policy.
    """
    require(not circuit.parameters, 'NO_SYMBOLIC_PARAMETERS')
    from qiskit.circuit.library import CXGate, HGate, RYGate, RZGate, SGate, SdgGate, SXGate, SXdgGate, XGate
    types = _types()
    types.update(ry=RYGate, s=SGate, sdg=SdgGate, sxdg=SXdgGate)
    result = circuit.copy_empty_like()
    phase = exact(circuit.global_phase)
    counts = Counter()

    def emit(op, qubits, clbits=()):
        nonlocal phase
        require(_supported(op, types), 'UNSUPPORTED_NATIVE_OPERATION:' + op.name)
        for value in op.params:
            exact(value)
        counts[op.name] += 1
        name = op.name
        if name == 'p':
            phase += exact(op.params[0]) / 2
            emit(RZGate(op.params[0]), qubits)
        elif name == 'h':
            phase += exact(math.pi) / 4
            emit(RZGate(math.pi / 2), qubits)
            emit(SXGate(), qubits)
            emit(RZGate(math.pi / 2), qubits)
        elif name in ('s', 'sdg'):
            sign = 1 if name == 's' else -1
            phase += sign * exact(math.pi) / 4
            emit(RZGate(sign * math.pi / 2), qubits)
        elif name == 'sxdg':
            emit(XGate(), qubits)
            emit(SXGate(), qubits)
        elif name == 'ry':
            emit(SdgGate(), qubits)
            emit(HGate(), qubits)
            emit(RZGate(op.params[0]), qubits)
            emit(HGate(), qubits)
            emit(SGate(), qubits)
        elif name == 'rzz':
            emit(CXGate(), qubits)
            emit(RZGate(op.params[0]), [qubits[1]])
            emit(CXGate(), qubits)
        elif name == 'xx_plus_yy':
            definition = op.definition
            phase += exact(definition.global_phase)
            for item in definition.data:
                local = [qubits[definition.find_bit(q).index] for q in item.qubits]
                require(not item.clbits, 'NO_CLASSICAL_XX_DEFINITION')
                emit(item.operation, local)
        else:
            result.append(op, qubits, clbits, copy=False)
    measured = False
    for item in circuit.data:
        if item.operation.name == 'measure':
            measured = True
        else:
            require(not measured, 'NO_MID_CIRCUIT_MEASUREMENT')
        emit(item.operation, list(item.qubits), list(item.clbits))
    assigned = _rounded_phase(phase)
    result.global_phase = assigned
    require(set(result.count_ops()) <= {'rz', 'sx', 'x', 'cx', 'id', 'barrier', 'measure'}, 'ONLY_REGISTERED_BASIS_REMAINS')
    return (result, dict(status='EXPLICIT_NATIVE_BASIS_ARTIFICIAL_PROTOTYPE_NOT_ADOPTED', exact_scalar_before_single_rounding=payload(phase), visited_operation_counts=dict(sorted(counts.items())), assigned_global_phase_binary64_hex=assigned.hex(), stored_global_phase_binary64_hex=float(result.global_phase).hex(), changed_compiler_input_policy=True, old_cost_compatibility=False, Clifford_pi_policy='BINARY64_MATH_PI_EXPLICIT_NOT_EXACT_CLIFFORD_CERTIFICATE', production_installed=False, postcompile_correction=False, acceptance_override=False))
def atomic_binary(path, writer):
    path=Path(path); tmp=path.with_name(path.name+'.partial')
    with tmp.open('xb') as stream:
        writer(stream); stream.flush(); os.fsync(stream.fileno())
    os.link(tmp,path); tmp.unlink()
    return dict(file=path.name,bytes=path.stat().st_size,sha256=c.sha(path))


def gate_error(name, first, second, out):
    import numpy as np
    value=float(np.linalg.norm(first-second))
    row=dict(check=name,raw_error=value if math.isfinite(value) else None,strictly_below=1e-9,
             PASS=math.isfinite(value) and value<1e-9,phase_alignment=False)
    c.save(Path(out)/(name+'.json'),row)
    c.require(row['PASS'],'NUMERIC_DISAGREEMENT:'+name)
    return row


def run_science(cfg, cancelled, *, checkpoint=None):
    """One ordered compile and two original-precision actions, no retries."""
    c.authorize(cfg); c.consumed(cfg)
    import numpy as np
    import qiskit, mpmath
    from qiskit import qpy, transpile
    import h6_m5_phase_engine as engine
    import inspect_h6_m5_phase_structure as structure
    require(qiskit.__version__=='1.3.0' and np.__version__=='1.26.4' and
            mpmath.__version__=='1.3.0' and mpmath.libmp.BACKEND=='python','FROZEN_REPAIR_RUNTIME')
    sealed={row['path'] for row in cfg['sealed_files']}
    require(all(str(Path(mod.__file__).resolve()) in sealed for mod in (np,qiskit,mpmath)), 'IMPORTED_RUNTIME_SEALED')
    out=Path(cfg['output']); started=time.monotonic(); artifacts={}; gates=[]

    def progress(phase):
        c.require(not cancelled.is_set(),'OWN_REPAIR_SIGNAL')
        if checkpoint: checkpoint(phase)
        with (out/'progress.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(dict(phase=phase,elapsed=time.monotonic()-started))+'\n')

    c.reserve(cfg,'LOAD_SAVED_INPUT'); progress('load_saved_input')
    inputs=cfg['inputs']
    with Path(inputs['wrapper_qpy']['path']).open('rb') as stream:
        circuits=qpy.load(stream); require(stream.read()==b'','NO_QPY_TRAILING_BYTES')
    require(len(circuits)==1,'ONE_SAVED_WRAPPER'); wrapper=circuits[0]
    engine.validate_circuit(wrapper,'WRAPPER',13)
    metadata=structure.summarize(wrapper,'WRAPPER',13)
    expected=c.read(inputs['static_metadata']['path'])['circuits']['WRAPPER']
    require(all(metadata[k]==expected[k] for k in ('ordered_metadata_sha256','gate_counts','global_phase_binary64_hex','operation_count')), 'EXACT_SAVED_WRAPPER_METADATA')
    c.save(out/'input_wrapper_metadata.json',metadata)
    with np.load(inputs['initial_npz']['path'],allow_pickle=False) as z:
        require(z.files==['low'],'INITIAL_KEYS'); low=z['low'].copy()
    with np.load(inputs['ordinary_vector']['path'],allow_pickle=False) as z:
        require(z.files==['vector'],'REFERENCE_KEYS'); reference=z['vector'].copy()
    with np.load(inputs['saved_vectors_npz']['path'],allow_pickle=False) as z:
        require(set(z.files)=={'low','before','after','independent_expected_high'},'ORIGINAL_VECTOR_KEYS')
        require(np.array_equal(z['low'],low) and np.array_equal(z['before'],reference),'EXACT_SAVED_ARRAY_JOIN')
    require(low.dtype==reference.dtype==np.dtype('complex128'),'ORIGINAL_VECTOR_PRECISION')
    for v in (low,reference): engine.finite_vector(v,8192)
    require(not np.count_nonzero(low[4096:]),'ORIGINAL_LOW_ANCILLA_ZERO')
    c.save(out/'saved_input_gate.json',dict(status='PASS',initial_sha256=inputs['initial_npz']['sha256'],
        reference_sha256=inputs['ordinary_vector']['sha256'],original_vector_join_exact=True,
        normalized=False,phase_corrected=False,new_reference=0))

    c.reserve(cfg,'LOWER_NATIVE'); progress('lower_native')
    candidate,receipt=_lower_native_core(wrapper)
    receipt.update(status='ISOLATED_NATIVE_BASIS_REPAIR_CANDIDATE',
        isolated_authorized_adapter=True,scientific_execution_scope=c.SCOPE,
        native_policy=c.POLICY,algorithm_prototype_SOURCE='9fe4981ac21da17888c7f5dcadc094257c282e4e')
    engine.validate_circuit(candidate,'COMPILED',13)
    c.require(len(candidate.data)<=cfg['resources']['candidate_instruction_bound'],'REGISTERED_EXPANSION_BOUND')
    artifacts['lowered_qpy']=atomic_binary(out/'lowered.qpy',lambda f:qpy.dump(candidate,f))
    c.save(out/'lowering_receipt.json',dict(receipt,SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        pipeline_identity=cfg['pipeline_identity'],candidate_operations=len(candidate.data),artifact=artifacts['lowered_qpy']))
    del circuits,wrapper

    c.reserve(cfg,'REPLAY_LOWERED'); progress('replay_lowered')
    before=engine.ordinary(candidate,low); progress('replay_lowered_finished')
    artifacts['lowered_vector']=atomic_binary(out/'lowered_vector.npz',lambda f:np.savez(f,vector=before))
    gates.append(gate_error('lowered_vs_saved_wrapper_vector',before,reference,out))

    c.reserve(cfg,'COMPILE_COSINE'); progress('compile_cosine')
    with (out/'compiler_passes.jsonl').open('x') as journal:
        def observe_pass(**kwargs):
            import json
            c.require(not cancelled.is_set(),'OWN_SIGNAL_IN_COMPILER')
            dag=kwargs['dag']; counts=dict(dag.count_ops()); phase=float(dag.global_phase)
            require(math.isfinite(phase),'NONFINITE_PASS_PHASE')
            journal.write(json.dumps(dict(pass_name=kwargs['pass_'].name(),count=kwargs['count'],
                pass_seconds=kwargs['time'],global_phase_binary64_hex=phase.hex(),
                gate_counts=counts,scalar_observation='BASIS_NAMES_ONLY_NOT_GATE_IDENTITY_OR_CAUSAL_PROOF',
                unexpected_names=sorted(set(counts)-{'rz','sx','x','cx','id','barrier','measure'})))+'\n')
            journal.flush()
        compiled=transpile(candidate,basis_gates=['rz','sx','x','cx'],optimization_level=1,
                           seed_transpiler=17,num_processes=1,callback=observe_pass)
    progress('compile_finished')
    artifacts['compiled_qpy']=atomic_binary(out/'compiled.qpy',lambda f:qpy.dump(compiled,f))
    compiled_schema=engine.validate_circuit(compiled,'COMPILED',13)
    c.save(out/'compiled_receipt.json',dict(SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],
        pipeline_identity=cfg['pipeline_identity'],artifact=artifacts['compiled_qpy'],schema=compiled_schema,
        status='UNVALIDATED_DIAGNOSTIC_NOT_FOR_ACCOUNTING',compiler=c.scope()['compiler']))
    del candidate

    c.reserve(cfg,'REPLAY_COMPILED'); progress('replay_compiled')
    after=engine.ordinary(compiled,low); progress('replay_compiled_finished')
    artifacts['compiled_vector']=atomic_binary(out/'compiled_vector.npz',lambda f:np.savez(f,vector=after))
    gates.append(gate_error('compiled_vs_lowered',after,before,out))
    gates.append(gate_error('compiled_vs_saved_wrapper_vector',after,reference,out))
    c.verify_seals(cfg); c.consumed(cfg)
    report=dict(status='REPAIR_COSINE_SINGLE_STATE_PASS_MANDATORY_STOP',SOURCE=cfg['SOURCE'],
        plan_sha256=cfg['plan_sha256'],pipeline_identity=cfg['pipeline_identity'],checks=gates,
        artifacts=artifacts,compiled_schema=compiled_schema,wall_seconds=time.monotonic()-started,
        new_compile=1,statevector_replays=2,accepted_cost_rows=0,old_cost_compatibility=False,
        sine=0,sampling=0,phase_alignment_for_acceptance=False,postcompile_phase_correction=False,
        old_STOP_remains_STOP=True,AD03='DEFERRED',next_phase_started=False,mandatory_STOP=True)
    c.save(out/'report.json',report)
    return report


def worker(cfg,cancelled):
    try: run_science(cfg,cancelled)
    except BaseException as exc:
        c.save(Path(cfg['output'])/'STOP.json',dict(status='STOP',reason=str(exc),traceback=traceback.format_exc(),
            SOURCE=cfg['SOURCE'],plan_sha256=cfg['plan_sha256'],retry=False,old_STOP_remains_STOP=True,
            next_phase_started=False,mandatory_STOP=True))

