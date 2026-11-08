"""Memoize compiler A's BasisTranslator output, with exact numeric/context keys.

Keep A's first inverse pass and unroll on the complete input, and its final
inverse pass on the complete native output. Only the middle primitive
translation is reused, using A itself on barrier-tagged full-context probes.
No angle invariance or isolated-basis cost approximation is assumed.
"""
from importlib.metadata import version
import math
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit.circuit.library import CXGate,XGate,HGate,ZGate,SGate,SdgGate,SXGate,SXdgGate,TGate,TdgGate,RZGate
from qiskit.converters import dag_to_circuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import UnrollCustomDefinitions,InverseCancellation
from .track_c_compiler_a import compile_reference,RejectOpaqueUnitary,BASIS
from .track_c_compiler_a_18round import CONTEXT_FAMILIES

_templates={}
_witnesses={}
_stats={'calls':0,'probe_calls':0,'templates':0,'fallbacks':0}


def key(op):
    params=tuple(float(x).hex() for x in op.params)
    if any(not math.isfinite(float(x)) for x in op.params):raise ValueError('nonfinite parameter')
    if hasattr(op,'ctrl_state') and op.ctrl_state!=(1<<op.num_ctrl_qubits)-1:raise ValueError('nonstandard control')
    return op.name,op.num_qubits,params


def populate(context,missing):
    """Same barrier/context construction as existing context_compile.

    Capture immediately AFTER BasisTranslator, before the final cancellation;
    cancellation must remain global, not be approximated per primitive.
    """
    probe=QuantumCircuit(3);tags=[]
    for k,op in missing.items():probe.append(op,range(op.num_qubits));probe.barrier();tags.append(k)
    for op in _witnesses[context].values():probe.append(op,range(op.num_qubits));probe.barrier();tags.append(None)
    probe.rz(.125,0);probe.barrier();tags.append(None)
    trace={}
    def callback(**kw):
        name=type(kw['pass_']).__name__
        if name=='UnrollCustomDefinitions':trace['context']=frozenset(set(kw['dag'].count_ops())-set(BASIS)-{'barrier','measure'})
        elif name=='BasisTranslator':trace['translated']=dag_to_circuit(kw['dag'])
    compile_reference(probe,callback=callback)
    assert trace['context']==context,(trace['context'],context)
    native=trace['translated'];slices=[];current=[]
    for inst in native.data:
        if inst.operation.name=='barrier':slices.append(current);current=[]
        else:current.append((inst.operation,tuple(native.find_bit(q).index for q in inst.qubits)))
    assert not current and len(slices)==len(tags)
    for k,ops in zip(tags,slices):
        if k is not None:
            assert all(op.name in BASIS and max(w)<k[1] for op,w in ops)
            _templates[(context,k)]=tuple(ops)
    _stats['probe_calls']+=1;_stats['templates']=len(_templates)


def compile_cached(circuit,*,callback=None):
    if callback is not None:return compile_reference(circuit,callback=callback)
    if version('qiskit')!='1.3.0':raise RuntimeError('Compiler A requires Qiskit1.3.0')
    fixed=[CXGate(),XGate(),HGate(),ZGate(),(SGate(),SdgGate()),(SXGate(),SXdgGate()),(TGate(),TdgGate())]
    first=PassManager([RejectOpaqueUnitary(),InverseCancellation(fixed),
        UnrollCustomDefinitions(SessionEquivalenceLibrary,list(BASIS)),RejectOpaqueUnitary()])
    q=first.run(circuit.copy(),num_processes=1)
    context=frozenset(set(q.count_ops())-set(BASIS)-{'barrier','measure'})
    if not context<=set(CONTEXT_FAMILIES):
        _stats['fallbacks']+=1;return compile_reference(circuit)
    _witnesses.setdefault(context,{})
    missing={}
    try:
        for inst in q.data:
            op=inst.operation
            if op.name in context:
                k=key(op);_witnesses[context].setdefault(op.name,op)
                if (context,k) not in _templates:missing[k]=op
    except (TypeError,ValueError):
        _stats['fallbacks']+=1;return compile_reference(circuit)
    if missing:populate(context,missing)
    native=QuantumCircuit(q.num_qubits,q.num_clbits)
    # Dropped primitive decomposition phases are one common global phase of
    # the already-controlled full joint circuit, allowed by reference A.
    native.global_phase=q.global_phase
    for inst in q.data:
        op=inst.operation;wires=tuple(q.find_bit(x).index for x in inst.qubits);bits=tuple(q.find_bit(x).index for x in inst.clbits)
        if op.name in context:
            for gate,local in _templates[(context,key(op))]:
                native._append(CircuitInstruction(gate,tuple(native.qubits[wires[i]] for i in local),()))
        else:native._append(CircuitInstruction(op,tuple(native.qubits[i] for i in wires),tuple(native.clbits[i] for i in bits)))
    result=PassManager([InverseCancellation([CXGate(),XGate()])]).run(native,num_processes=1)
    assert set(result.count_ops())<=set(BASIS)|{'barrier','measure'}
    _stats['calls']+=1
    return result


def statistics():return dict(_stats)


def validate_saved():
    """Rebuild existing trajectories, without new draws or a new price policy."""
    import gzip,json,time,hashlib
    from pathlib import Path
    from . import track_c_closed_p3_joint_opt as j
    from .track_c_closed_p3_joint_opt_checks import whole_fixture
    j.cr.compile_reference=compile_cached;j.compile_reference=compile_cached
    rows=[]
    for t,n,jj in [(.02,1,0),(.32,512,0),(.32,1024,0),(.16,512,0)]:
        for rare in (False,True):
            path=j.OUT/'training'/(j.cp.cellkey(t,n)+f'_{int(rare)}_{jj}.json.gz')
            with gzip.open(path,'rt') as f:r=json.load(f)
            start=time.monotonic();a,av,_=j.canonical_native(t,n,r['canonical_choices']);g=j.cp.generator(t,n)
            events=[g.event(v[0],tuple(v[1:-1]),v[-1]) for v in r['closed_choices']]
            b,bv,_=j.cr.closed_native(t,n,events)
            assert j.metrics(a)==r['canonical'] and j.metrics(b)==r['closed']
            assert bv==r['closed_variants']
            if 'legacy_variants' in r['canonical_variants']:assert av==r['canonical_variants']
            rows.append(dict(t=t,n=n,rare=rare,j=jj,all_available_variant_metrics_identical=True,
                             cached_rebuild_wall_seconds=time.monotonic()-start,original_row_wall_seconds=r.get('wall_seconds')))
            print('SAVED_VARIANTS_MATCH',t,n,rare,rows[-1]['cached_rebuild_wall_seconds'],flush=True)
    s=next(s for s in j.menu()['rows'] if tuple(s['key'])==('tuned_0321',17,.32,64));whole=whole_fixture(s)
    path=Path(__file__)
    result=dict(**j.provenance(),cache_helper_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                helper_source_status='new helper staged separately while existing refinement SOURCE remains unchanged; exact helper bytes identified by digest',
                same_compiler_A_passes=True,context_and_exact_parameter_cache=True,rows=rows,whole=whole,statistics=statistics(),passed=True)
    j.save_json(j.OUT/'CACHED_A_CHECKS.json',result);print('CACHED_A_PASS',rows,statistics(),flush=True)


if __name__=='__main__':validate_saved()
