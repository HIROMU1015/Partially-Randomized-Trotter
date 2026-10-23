"""Saved-descriptor D1 adapter. No scientific work/imports occur on module import.

Original primitives/control/compiler are reused. No sampler, Hamiltonian,
preparation, state evolution or event-angle generator is called by this module.
"""
from dataclasses import asdict, fields
from collections import Counter
from types import SimpleNamespace as NS

_API = None

def api():
    global _API
    if _API is None:
        import numpy as np
        import qiskit
        from qiskit import QuantumCircuit, qpy
        from qiskit.circuit.library import RZGate, XXPlusYYGate, PhaseGate
        from .rte import RTEEvent, RTEEventApplication, BasisChangeOperation, BasisReuseInterval
        from .rte_event_audit import verify_join
        from .pauli_signal_adapter import append_pauli_evolution, append_pauli_event
        from .df_rte_tail import DFBasisRegistry
        from .df_rte_circuit import DFRTEComponentCircuitSpec, DFRTEIdentityCircuitSpec, DFRTEEventSequenceCircuitRequest
        from .df_rte_qiskit import QiskitDFRTEEventCircuitBuilder
        from .df_partial_s2 import QiskitDFPartialS2CircuitBuilder
        from .df_trotter.ops import DiagonalEvolutionPrimitives, append_diagonal_primitives
        from .fixed_signal_evaluation import hadamard_wrapper, default_compiler
        from .rte_compiled_cost import canonical_qiskit_circuit_fingerprint, transpile_and_measure_cost
        _API = NS(**locals())
    return _API

def require(value, reason):
    if not value:
        raise ValueError(reason)

def event_from_record(record):
    a = api()
    apps = []
    for saved in record['application_sequence']:
        part = dict(saved)
        part['basis_change_operations'] = tuple(a.BasisChangeOperation(
            name=o['name'],qubits=tuple(o['qubits']),parameters=tuple(o['parameters']),
            matrix_sha256=o['matrix_sha256']) for o in part['basis_change_operations'])
        if part['diagonal_pauli_support'] is not None:
            part['diagonal_pauli_support'] = tuple(part['diagonal_pauli_support'])
        apps.append(a.RTEEventApplication(**part))
    names = {f.name for f in fields(a.RTEEvent)}
    data = {k:v for k,v in record.items() if k in names}
    for name in ('product_component_ids','selected_component_ids','df_fragment_ids'):
        data[name] = tuple(data[name])
    data['basis_reuse_intervals'] = tuple(a.BasisReuseInterval(**p) for p in data['basis_reuse_intervals'])
    data['application_sequence'] = tuple(apps)
    data['phase'] = complex(data['phase']['real'],data['phase']['imag'])
    return a.RTEEvent(**data)

def restore_basis(registry, saved, n, *, basis_id=None):
    a = api()
    ops = []
    for op in saved['operations']:
        d = op['description']
        require(op['qubits']==d['qubits'],'BASIS_QUBIT_BINDING')
        params = tuple(float(v) for v in d['parameters'])
        if d['name']=='rz' and len(params)==1:
            gate = a.RZGate(*params)
        elif d['name']=='xx_plus_yy' and len(params)==2:
            gate = a.XXPlusYYGate(*params)
        else:
            raise ValueError('UNSUPPORTED_SAVED_GATE:'+d['name'])
        ops.append((gate,tuple(d['qubits'])))
    definition = registry.register(ops,num_system_qubits=n,basis_id=basis_id)
    require(definition.basis_hash==saved['basis_hash'],'RESTORED_BASIS_HASH')
    require([asdict(o) for o in definition.metadata.operations] ==
            [dict(o['description'],qubits=tuple(o['description']['qubits']),
                  parameters=tuple(o['description']['parameters'])) for o in saved['operations']],
            'RESTORED_LOCAL_GATE_MATRIX_HASH_OR_DESCRIPTION')
    return definition

def grouped_events(join, q, r):
    require(join['q']==q and join['r']==r,'Q_R_BINDING')
    if join['method']=='PF':
        require(not join['events'],'PF_HAS_EVENTS')
        return tuple(() for _ in range(q))
    require(len(join['events'])==q*r,'EVENT_CARDINALITY')
    groups=[]
    for step in range(q):
        records=join['events'][step*r:(step+1)*r]
        require(all(e['outer_step_index']==step and e['inner_RTE_position']==j
                    for j,e in enumerate(records)),'SAVED_EVENT_ORDER')
        groups.append(tuple(event_from_record(e) for e in records))
    return tuple(groups)

def build(model, method, join, part='full'):
    a=api()
    require(part in ('full','retained','event'),'D1_PART')
    case=model['cases'][method];n=model['n_qubits'];q=case['config']['q'];r=case['config']['r']
    require(n>=1 and case['config']['T']/q==model['step_time'],'STEP_TIME_BINDING')
    require(join['method']==method and join['representation']==model['representation'],'CASE_BINDING')
    groups=grouped_events(join,q,r)
    dt=model['step_time'];retained=part in ('full','retained');random=part in ('full','event') and method=='PR'
    tags=[];basis_counters=[];virtual=Counter()
    if model['representation']=='pauli_jw':
        body=a.QuantumCircuit(n)
        labels={str(j):c['label'] for j,c in enumerate(case['components'])}
        for step,events in enumerate(groups):
            for reverse in (False,True):
                if reverse and random:
                    start=len(body.data)
                    for event in events:a.append_pauli_event(body,event,labels)
                    tags.append(dict(step=step,role='event',start=start,stop=len(body.data),phase_owner='event'))
                if retained:
                    start=len(body.data)
                    for b in (reversed(case['blocks']) if reverse else case['blocks']):
                        a.append_pauli_evolution(body,b['label'],b['coefficient'],dt/2)
                    tags.append(dict(step=step,role='retained',direction='reverse' if reverse else 'forward',
                                     start=start,stop=len(body.data),kind='Pauli_evolution'))
            if retained:body.global_phase-=model['hamiltonian']['constant']*dt
        pre=dict(body.count_ops())
        circuit=a.hadamard_wrapper(body,'cosine',already_controlled=False)
        diag=dict(body_name=body.name,controlled_operation_name=circuit.data[1].operation.name,
                  base_gate_name=circuit.data[1].operation.base_gate.name,
                  global_phase=float(body.global_phase))
        return NS(circuit=circuit,body=body,sidecar=dict(tags=tags,tag_coordinates='global source body instruction ranges',body_before_control_histogram=pre,
            control_policy='original_generic_whole_body',basis_counters=[],naming=diag))
    require(model['representation']=='df_native','REPRESENTATION')
    registry=a.DFBasisRegistry()
    blocks=[]
    for b in case['blocks']:
        definition=restore_basis(registry,b['basis'],n)
        blocks.append((b,definition))
    # Component specs cover the stored applications only; no probability/law recomputation.
    specs={}
    for events in groups:
        for event in events:
            for app in event.application_sequence:
                if app.component_id in specs:continue
                if app.is_identity:
                    spec=a.DFRTEIdentityCircuitSpec(app.component_id,app.coefficient_abs,
                        app.coefficient_sign,n,df_fragment_id=app.df_fragment_id)
                else:
                    candidates=[c for c in case['components'] if
                        c['basis']['basis_hash']==app.basis_hash and
                        tuple(c['support'])==app.diagonal_pauli_support and
                        c['sign']==app.coefficient_sign and c['coefficient_abs']==app.coefficient_abs]
                    require(bool(candidates),'SAVED_DF_COMPONENT_JOIN')
                    definition=restore_basis(registry,candidates[0]['basis'],n,basis_id=app.basis_id)
                    require(definition.metadata.operations==app.basis_change_operations,'EVENT_BASIS_OPERATIONS')
                    spec=a.DFRTEComponentCircuitSpec(app.component_id,app.coefficient_abs,
                        app.coefficient_sign,app.df_fragment_id,app.basis_id,app.diagonal_pauli_support,
                        app.basis_change_operations,n,basis_hash=app.basis_hash)
                specs[app.component_id]=spec
    step_results=[]
    for step,events in enumerate(groups):
        forward=a.QuantumCircuit(n+1);reverse=a.QuantumCircuit(n+1)
        rte=a.QuantumCircuit(n+1)
        if retained:
            for phase in case['native_scalar_phases']:
                if phase!=0:forward.append(a.PhaseGate(phase),[n])
            for direction,target,ordered in [('forward',forward,blocks),('reverse',reverse,list(reversed(blocks)))]:
                for b,definition in ordered:
                    start=len(target.data)
                    proxy=NS(runtime_basis_operations=definition.runtime_operations)
                    a.QiskitDFPartialS2CircuitBuilder._append_basis(target,proxy,inverse=True)
                    prim=b['primitives']
                    saved=a.DiagonalEvolutionPrimitives(global_phase=prim['global_phase'],
                        rz=tuple(tuple(x) for x in prim['rz']),rzz=tuple(tuple(x) for x in prim['rzz']))
                    a.append_diagonal_primitives(target,saved,controlled=True,ancilla_qubit=n)
                    a.QiskitDFPartialS2CircuitBuilder._append_basis(target,proxy,inverse=False)
                    tags.append(dict(step=step,role='retained',direction=direction,block=b['operator_key'],
                        start=start,stop=len(target.data),nested_kinds=['basis','diagonal','central_control']))
                    virtual.update(g.name for g,qs in definition.runtime_operations)
                    virtual.update(g.name for g,qs in definition.runtime_operations)
                    virtual.update({'rz':len(prim['rz']),'rzz':len(prim['rzz'])})
                    if prim['global_phase']!=0:virtual.update({'source_global_phase':1})
        if random:
            request=a.DFRTEEventSequenceCircuitRequest(events=events,component_specs=tuple(specs.values()),
                controlled=True,ancilla_qubit=n,cancel_adjacent_equal_bases=True,occurrence_rte_steps=r)
            result=a.QiskitDFRTEEventCircuitBuilder(basis_registry=registry).build_sequence(request)
            rte=result.circuit
            basis_counters.append(dict(step=step,application_count=result.application_count,
                naive_basis_change_count=result.naive_basis_change_count,
                emitted_basis_change_count=result.emitted_basis_change_count,
                cancelled_basis_change_pairs=result.cancelled_basis_change_pairs,
                basis_switch_count=result.basis_switch_count,relative_ancilla_phase=result.relative_ancilla_phase))
            tags.append(dict(step=step,role='event',start=0,stop=len(rte.data),
                             nested_kinds=['basis','Z_ZZ','central_control'],
                             precontrol_histogram=None,missing_reason='NATIVE_CENTRAL_CONTROL_NO_SECOND_COUNTERFACTUAL_BODY'))
        step_circuit=a.QuantumCircuit(n+1)
        for piece in (forward,rte,reverse):step_circuit.compose(piece,qubits=tuple(range(n+1)),inplace=True)
        step_results.append(step_circuit)
    body=a.QuantumCircuit(n+1)
    for piece in step_results:body.compose(piece,qubits=tuple(range(n+1)),inplace=True)
    circuit=a.hadamard_wrapper(body,'cosine',already_controlled=True)
    return NS(circuit=circuit,body=body,sidecar=dict(tags=tags,tag_coordinates='relative to named forward/event/reverse piece within step; nested kinds overlap',basis_counters=basis_counters,
        control_policy='original_native_central_only',df_repetition_policy='raw_concatenation',
        virtual_retained_precontrol_histogram=dict(virtual),virtual_event_precontrol_histogram=None,
        virtual_event_missing_reason='NOT_AN_ACTUAL_GLOBAL_CONTROL_LIFT; see event sidecar and native basis counters',
        source_after_control_histogram=dict(body.count_ops()),naming=dict(body_name=body.name)))

def empty(n):
    a=api();c=a.QuantumCircuit(n+1,1);c.h(n);c.h(n);c.measure(n,0)
    return NS(circuit=c,body=None,sidecar=dict(tags=[],phase='none',control_policy='none'))

def save_input(built, directory, expected=None):
    from pathlib import Path
    import json
    a=api();d=Path(directory)
    fingerprint=a.canonical_qiskit_circuit_fingerprint(built.circuit)
    data=dict(input_fingerprint=fingerprint,expected_fingerprint=expected,
        input_fingerprint_matches=None if expected is None else fingerprint==expected,
        pretranspile_histogram=dict(built.circuit.count_ops()),sidecar=built.sidecar)
    with (d/'input.qpy').open('xb') as f:a.qpy.dump(built.circuit,f)
    if built.sidecar.get('control_policy')=='original_generic_whole_body':
        with (d/'body_before_control.qpy').open('xb') as f:a.qpy.dump(built.body,f)
    return data

def compile_saved(directory):
    from pathlib import Path
    a=api();d=Path(directory)
    with (d/'input.qpy').open('rb') as f:circuits=a.qpy.load(f)
    require(len(circuits)==1,'SINGLE_QPY_INPUT')
    circuit=circuits[0]
    cost=a.transpile_and_measure_cost(circuit,a.default_compiler())
    with (d/'output.qpy').open('xb') as f:a.qpy.dump(cost.transpiled_circuit,f)
    return dict(RZ=cost.rz_count,CX=cost.cx_count,depth=cost.total_depth,
        rz_depth=cost.rz_depth,cx_depth=cost.cx_depth,
        pretranspile_histogram=dict(cost.pretranspile_gate_counts),
        posttranspile_histogram=dict(cost.posttranspile_gate_counts),
        input_fingerprint=cost.actual_circuit_fingerprint,
        output_fingerprint=a.canonical_qiskit_circuit_fingerprint(cost.transpiled_circuit),
        compiler=asdict(cost.compiler),pass_trace=None,pass_trace_missing_reason='NO_CALLBACK_IN_FIXED_WRAPPER')

def interactions(metrics):
    result={}
    for rep in ('pauli_jw','df_native'):
        row={}
        for metric in ('RZ','CX'):
            f=metrics[rep+'_PR_full'][metric];r=metrics[rep+'_PR_retained'][metric]
            e=metrics[rep+'_PR_event'][metric];z=metrics['empty'][metric]
            row[metric]=dict(full=f,retained=r,event=e,empty=z,
                retained_increment=r-z,event_increment=e-z,isolated_prediction=r+e-z,
                Delta_interaction=f-r-e+z,contextual_event_increment=f-r,contextual_retained_increment=f-e)
        result[rep]=row
    return result
