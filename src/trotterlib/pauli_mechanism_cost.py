"""Saved-token P2 parts and Full descriptor checks. No sampler or signal API."""
import hashlib
import json
import math
import numpy as np
from .h4_d1_adapter import pauli_event_view, interaction_metrics
from .h4_d2_pauli_control import central_control_body, central_control_tokens
from .pauli_signal_adapter import append_pauli_evolution, append_pauli_event
from .fixed_signal_evaluation import hadamard_wrapper, default_compiler
from .rte_compiled_cost import _filtered_depth


class DescriptorRecorder:
    """Gate descriptor sink, never a QuantumCircuit or a synthesized gate."""
    def __init__(self,n):
        self.num_qubits=n
        self.data=[]

    def __getattr__(self,name):
        if name not in ('h','sdg','s','cx','crz','cy','cz','p'):
            raise AttributeError(name)
        def emit(*args):
            parameter_count=1 if name in ('crz','p') else 0
            self.data.append(dict(name=name,qubits=list(args[parameter_count:]),
                                  clbits=[],params=list(args[:parameter_count])))
        return emit


def recipe(law, result):
    cfg=law['config'];n=law['n_system'];q=cfg['q'];r=cfg['r']
    if len(result['events'])!=q*r or result['event_count']!=q*r:
        raise ValueError('SAVED_EVENT_CARDINALITY')
    events=tuple(pauli_event_view(x,law['components']) for x in result['events'])
    if any(x.taylor_order not in (0,2) for x in events):
        raise ValueError('SAVED_K2_SCOPE')
    return dict(n=n,q=q,r=r,delta=law['step_time'],
                retained=law['retained'],labels={str(i):x['label'] for i,x in enumerate(law['components'])},
                events=events,scalar_step=law['scalar_step_binary64'],
                saved_body_phase=result['body_phase'])


def source_tokens(spec, part):
    """Restore chronological source instructions; Full stays descriptor-only."""
    if part not in ('F','R','E'):
        raise ValueError('SOURCE_PART')
    tokens=[];phase=0.;n=spec['n'];wires=tuple(range(n))
    def phase_add(value):
        nonlocal phase
        phase=(phase+float(value))%(2*math.pi)
    def evolution(label,c,t):
        tokens.append(dict(kind='evolution',label=label,coefficient=c,time=t,
                           central_rz_angle=float(2*c*t),wires=wires))
    for step in range(spec['q']):
        if part in ('F','R'):
            for x in spec['retained']:
                evolution(x['label'],x['coefficient'],spec['delta']/2)
        if part in ('F','E'):
            for event in spec['events'][step*spec['r']:(step+1)*spec['r']]:
                for app in event.application_sequence:
                    label=spec['labels'][app.component_id]
                    if app.role=='product':
                        for qubit,axis in enumerate(reversed(label)):
                            if axis!='I':
                                tokens.append(dict(kind='product',axis=axis.lower(),wires=(qubit,)))
                        if app.coefficient_sign<0:
                            phase_add(math.pi)
                    else:
                        evolution(label,app.coefficient_sign,event.rotation_angle)
                phase_add(float(np.angle(event.phase)))
        if part in ('F','R'):
            for x in reversed(spec['retained']):
                evolution(x['label'],x['coefficient'],spec['delta']/2)
            phase_add(spec['scalar_step'])
    return tokens,phase


def semantic_program(circuit):
    return dict(num_qubits=circuit.num_qubits,num_clbits=circuit.num_clbits,
        global_phase=float(circuit.global_phase),
        instructions=[dict(name=item.operation.name,
            qubits=[circuit.find_bit(x).index for x in item.qubits],
            clbits=[circuit.find_bit(x).index for x in item.clbits],
            params=[float(x) for x in item.operation.params])
            for item in circuit.data])


def program_hash(program):
    return hashlib.sha256(json.dumps(program,sort_keys=True,separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def expected_full(spec,axis):
    tokens,phase=source_tokens(spec,'F')
    controlled=central_control_tokens(tokens,phase,spec['n'],
                                     circuit_factory=DescriptorRecorder).body
    n=spec['n'];h=dict(name='h',qubits=[n],clbits=[],params=[])
    records=[h,*controlled.data]
    if axis=='sine':
        records.append(dict(name='sdg',qubits=[n],clbits=[],params=[]))
    elif axis!='cosine':
        raise ValueError('AXIS')
    records.extend([h,dict(name='measure',qubits=[n],clbits=[0],params=[])])
    return dict(num_qubits=n+1,num_clbits=1,global_phase=0.,instructions=records)


def compare_programs(actual,expected,atol=1e-12):
    for k in ('num_qubits','num_clbits'):
        if actual[k]!=expected[k]:raise ValueError('FULL_'+k)
    if not all(math.isfinite(p['global_phase']) for p in (actual,expected)):
        raise ValueError('FULL_NONFINITE_PHASE')
    if abs(math.remainder(actual['global_phase']-expected['global_phase'],2*math.pi))>atol:
        raise ValueError('FULL_GLOBAL_PHASE')
    if len(actual['instructions'])!=len(expected['instructions']):
        raise ValueError('FULL_INSTRUCTION_COUNT')
    maximum=0.
    for index,(a,e) in enumerate(zip(actual['instructions'],expected['instructions'])):
        for k in ('name','qubits','clbits'):
            if a[k]!=e[k]:raise ValueError(f'FULL_INSTRUCTION_{index}_{k}')
        if len(a['params'])!=len(e['params']):raise ValueError('FULL_PARAMETER_COUNT')
        for av,ev in zip(a['params'],e['params']):
            if not math.isfinite(av) or not math.isfinite(ev):raise ValueError('FULL_NONFINITE_PARAMETER')
            error=abs(math.remainder(av-ev,2*math.pi)) if a['name']=='p' else abs(av-ev)
            maximum=max(maximum,error)
            if error>atol:raise ValueError(f'FULL_PARAMETER_{index}')
    return dict(instructions=len(actual['instructions']),maximum_parameter_difference=maximum,
                scientific_semantics_match=True,Full_circuits_rebuilt=0)


def build_part(spec,part):
    """Only R/E are constructed; Full is explicitly refused."""
    from qiskit import QuantumCircuit
    if part not in ('R','E'):
        raise ValueError('FULL_BUILD_FORBIDDEN')
    body=QuantumCircuit(spec['n'])
    for step in range(spec['q']):
        if part=='R':
            for x in spec['retained']:
                append_pauli_evolution(body,x['label'],x['coefficient'],spec['delta']/2)
            for x in reversed(spec['retained']):
                append_pauli_evolution(body,x['label'],x['coefficient'],spec['delta']/2)
            body.global_phase+=spec['scalar_step']
        else:
            for event in spec['events'][step*spec['r']:(step+1)*spec['r']]:
                append_pauli_event(body,event,spec['labels'])
    lift=central_control_body(body)
    tokens,phase=source_tokens(spec,part)
    expected=central_control_tokens(tokens,phase,spec['n'],
                                   circuit_factory=DescriptorRecorder).body
    compare_programs(semantic_program(lift.body),dict(num_qubits=spec['n']+1,
        num_clbits=0,global_phase=0.,instructions=expected.data))
    return lift.body,dict(source_tokens=len(tokens),aggregate_phase=phase,
                          phase_owner='scalar' if part=='R' else 'event_signs_and_order')


def part_wrappers(spec,part):
    from qiskit import QuantumCircuit
    if part=='0':
        body=QuantumCircuit(spec['n']+1)
        sidecar=dict(aggregate_phase=0.,phase_owner='none')
    else:
        body,sidecar=build_part(spec,part)
    return {a:hadamard_wrapper(body,a,already_controlled=True) for a in ('cosine','sine')},sidecar


def metrics(circuit):
    hist={str(k):int(v) for k,v in circuit.count_ops().items()}
    return dict(RZ=hist.get('rz',0),CX=hist.get('cx',0),
                rz_depth=_filtered_depth(circuit,'rz'),cx_depth=_filtered_depth(circuit,'cx'),
                total_depth=int(circuit.depth() or 0),circuit_size=int(circuit.size()),
                gate_histogram=hist,global_phase=float(circuit.global_phase))
