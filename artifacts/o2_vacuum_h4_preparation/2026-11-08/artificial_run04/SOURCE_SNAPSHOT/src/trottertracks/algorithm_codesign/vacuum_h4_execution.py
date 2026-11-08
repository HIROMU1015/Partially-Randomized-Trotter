"""Deferred 2-path resource transaction; no science or loader runs on import."""
from dataclasses import asdict,dataclass
from fractions import Fraction as F
from pathlib import Path
import hashlib
import importlib.metadata
import io
import json
import platform
import resource
import subprocess
import time

from qiskit import qpy,transpile
from qiskit.transpiler import CouplingMap

from . import vacuum_h4_adapter as a
from . import vacuum_h4_numeric as numeric
from . import m_n0_metadata_execution as saved
from ..pr_dictionary_validation.fixture import CompilerProfile
from ..pr_dictionary_validation.gates import compiled_wrapper,compile_cache_key

METHODS=('controlled_PR','known_SE','candidate')
SOURCE_FILES=tuple(dict.fromkeys((*saved.SOURCE_FILES,
    'src/trottertracks/algorithm_codesign/vacuum_reference_cswap.py',
    'src/trottertracks/algorithm_codesign/vacuum_h4_adapter.py',
    'src/trottertracks/algorithm_codesign/vacuum_h4_numeric.py',
    'src/trottertracks/algorithm_codesign/vacuum_h4_execution.py',
    'src/trottertracks/algorithm_codesign/a1_new_target_n2.py',
    'src/trotterlib/rte_compiled_cost.py',
    'src/trottertracks/pr_dictionary_validation/fixture.py',
    'src/trottertracks/pr_dictionary_validation/gates.py',
    'src/trottertracks/algorithm_codesign/df_correction_comparison/intervals.py',
    'scripts/tracks/algorithm_codesign/run_vacuum_h4_limited.py')))


@dataclass(frozen=True)
class Profile(CompilerProfile):
    optimization_level:int=2

    def validate(self):
        expected=asdict(CompilerProfile());expected['optimization_level']=2
        if asdict(self)!=expected:raise ValueError('fixed level2 compiler profile mismatch')


def preflight(root,expected_source_commit):
    root=Path(root)
    git=lambda *args:subprocess.check_output(['git','-C',str(root),*args])
    if len(expected_source_commit)!=40 or git('rev-parse','HEAD').decode().strip()!=expected_source_commit:
        raise ValueError('full execution SOURCE commit/HEAD mismatch')
    hashes={}
    for name in SOURCE_FILES:
        data=(root/name).read_bytes()
        if data!=git('show',expected_source_commit+':'+name):raise ValueError('execution SOURCE changed')
        hashes[name]=hashlib.sha256(data).hexdigest()
    env={x:importlib.metadata.version(x) for x in ('numpy','qiskit','mpmath')}
    if env['numpy']!='1.26.4' or env['qiskit']!='1.3.0':
        raise ValueError('fixed compiler/numerical environment mismatch')
    return dict(execution_source_commit=expected_source_commit,source_sha256=hashes,
        environment=dict(python=platform.python_version(),**env),N1=saved.N1,N1_source=saved.N1_SOURCE,
        profile=asdict(Profile()),scope='2 fixed paths, X/Y, 3 methods, 2 topologies; retry0',
        no_state_Born_RNG_shots=True)


def metrics(c):
    source=a.module('rte_compiled_cost')
    counts=dict(source._gate_counts(c))
    return dict(qubits=c.num_qubits,native_gate_counts=counts,
        rz_count=counts.get('rz',0),cx_count=counts.get('cx',0),
        rz_depth=source._filtered_depth(c,'rz'),cx_depth=source._filtered_depth(c,'cx'),
        total_depth=c.depth(),circuit_size=c.size(),reset_count=counts.get('reset',0),
        measure_count=counts.get('measure',0),T_count='UNKNOWN_NO_FT_SYNTHESIS',
        physical_reset_readout_latency='UNKNOWN',state_preparation_excluded=True)


def _same_evolution(body,wrapper,axis):
    """Exact gate/parameter/wire identity to the reused fixed selected builder."""
    instructions=wrapper.data[1:-(2 if axis=='Y' else 1)]
    def signature(c,data):
        return [(i.operation.name,tuple(float(p).hex() for p in i.operation.params),
                 tuple(c.find_bit(q).index for q in i.qubits)) for i in data]
    if (signature(body,body.data)!=signature(wrapper,instructions)
        or float(body.global_phase)!=float(wrapper.global_phase)):
        raise ValueError('traced body vs fixed selected full mismatch')


def compare_records(records,output,call):
    """Explicit science entry, also tested on invented supplied records only."""
    out=Path(output);profile=Profile();cache={};routes={};gate_cache={}
    paths=call('path_selection',lambda:a.select_paths(records))
    context=call('restore_context',lambda:a.SavedPR(records,expected_hashes=a.record_hashes(records)))
    rows=[];builds=[]
    for pi,path in enumerate(paths):
        body,L,R,split=call('traced_body_build',lambda:context.body_with_boundaries(path['order'],tuple(path['labels'])))
        builds.append(dict(path=path,split=split,L=asdict(L),R=asdict(R)))
        for axis in ('X','Y'):
            old=call('selected_wrapper_build',lambda:context.selected_circuit(path['order'],tuple(path['labels']),axis))
            _same_evolution(body,old,axis)
            from qiskit import ClassicalRegister
            bit=ClassicalRegister(1,'readout');old.add_register(bit);old.measure(context.n,bit[0])
            ref,ledger=call('reference_wrapper_build',lambda:a.build_reference(L,R,axis))
            circuits=dict(controlled_PR=old,known_SE=ref,candidate=ref.copy())
            ctx=dict(path=path['identity'],axis=axis,preparation=context.prep['preparation_hash'],
                     policy=context.policy['policy_fingerprint'],resource='full_no_state_preparation')
            keys={m:compile_cache_key(c,profile,ctx) for m,c in circuits.items()}
            if keys['known_SE']!=keys['candidate']:raise ValueError('known SE/candidate IR mismatch')
            for method in METHODS:
                raw=circuits[method];token=keys[method];warm=token in cache;start=time.perf_counter()
                if not warm:
                    cache[token]=call('native_compile',lambda:compiled_wrapper(raw,profile))
                native=cache[token];native_seconds=time.perf_counter()-start
                for topology in ('all_to_all','line_fixed_layout'):
                    route_key=(token,topology);route_warm=route_key in routes
                    start=time.perf_counter()
                    if topology=='all_to_all':compiled=native
                    else:
                        if not route_warm:
                            routes[route_key]=call('routing_compile',lambda:transpile(native,
                                basis_gates=list(profile.basis_gates),optimization_level=2,
                                seed_transpiler=17,approximation_degree=1.,qubits_initially_zero=False,
                                coupling_map=CouplingMap.from_line(raw.num_qubits),
                                initial_layout=list(range(raw.num_qubits)),
                                layout_method='trivial',routing_method='sabre'))
                        compiled=routes[route_key]
                    routing_seconds=time.perf_counter()-start
                    proof_key=(token,topology,'proof')
                    if proof_key not in routes:
                        routes[proof_key]=call('operator_qualification',
                            lambda:numeric.qualify_compilation(raw,compiled,context.n,cache=gate_cache))
                    proof=routes[proof_key]
                    correction=F(0) if method=='controlled_PR' else F(ledger['correction_rounding_upper'])
                    row=dict(path=path,axis=axis,method=method,topology=topology,
                        IR_identity=token,compile_cache_hit=warm,routing_cache_hit=route_warm,
                        metrics=metrics(compiled),numerical_qualification=proof,
                        additional_raw_correction_error_upper=str(correction),
                        combined_coordinate_difference_upper=str(correction+F(proof['coordinate_error_upper'])),
                        error_allowance_adopted=False,known_SE_candidate_same_IR=True,
                        native_compile_seconds=native_seconds,routing_compile_seconds=routing_seconds,
                        phase_ledger=None if method=='controlled_PR' else ledger)
                    rows.append(row)
                    # Preserve each completed result before proceeding.
                    (out/'RESULT_ROWS.json').write_text(json.dumps(rows,indent=2)+'\n')
                    if method!='candidate':
                        stream=io.BytesIO();qpy.dump(compiled,stream)
                        (out/f'path{pi}_{axis}_{method}_{topology}.qpy').write_bytes(stream.getvalue())
        (out/'PATH_DESCRIPTORS.json').write_text(json.dumps(builds,indent=2)+'\n')
    (out/'LOCAL_GATE_CERTIFICATES.json').write_text(json.dumps(
        list(gate_cache.values()),default=str,indent=2)+'\n')
    if len(rows)!=24:raise ValueError('finite comparison rows incomplete')
    return dict(status='RESOURCE_COMPARISON_AND_UNIFORM_BOUNDS_RECORDED',
        rows=24,paths=paths,scientific_GO=False,expected_Q_cost='NOT_EVALUATED',
        error_allowance_adopted=False,gate_key_count=len(gate_cache))


CAPS=dict(input_load=1,path_selection=1,restore_context=1,traced_body_build=2,
          selected_wrapper_build=4,reference_wrapper_build=4,native_compile=8,
          routing_compile=8,operator_qualification=16)


def run(root,fresh_output,*,expected_source_commit):
    """One transaction only. A separate explicit science instruction is needed."""
    root,out=Path(root).resolve(),Path(fresh_output).resolve()
    if out.is_relative_to(root/'artifacts/o2_a1_new_target'):
        raise ValueError('output may not enter old scientific namespace')
    out.mkdir(parents=True,exist_ok=False)
    counts={k:0 for k in CAPS};result=None;start=time.perf_counter()
    def write(name,data):
        with (out/name).open('x') as f:json.dump(data,f,indent=2,default=str);f.write('\n')
    def journal(data):
        import os
        with (out/'CALLS.jsonl').open('a') as f:
            f.write(json.dumps(data,default=str)+'\n');f.flush();os.fsync(f.fileno())
    def call(name,function):
        counts[name]+=1
        if counts[name]>CAPS[name]:raise ValueError('call scope exceeded: '+name)
        journal(dict(event='CALL_START',call=name,index=counts[name]))
        try:value=function()
        except BaseException as exc:
            journal(dict(event='CALL_EXCEPTION',call=name,error=type(exc).__name__+': '+str(exc)));raise
        journal(dict(event='CALL_RETURN',call=name,index=counts[name]));return value
    try:
        source=preflight(root,expected_source_commit);write('SOURCE.json',source)
        records,inputs=call('input_load',lambda:saved.load_fixed_inputs(root));write('INPUTS.json',inputs)
        result=compare_records(records,out,call)
    except BaseException as exc:
        result=dict(status='STOP_INCOMPLETE',reason=type(exc).__name__+': '+str(exc),
                    unfinished='UNKNOWN',retry=0)
    finally:
        write('RESULT.json',result);write('CALL_COUNTS.json',counts)
        write('STOP.json',dict(status='MANDATORY_STOP',result_status=result['status'],
            retry=0,automatic_next_stage=False,elapsed_seconds=time.perf_counter()-start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return result
