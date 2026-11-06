"""Fixed saved m0 H4 trajectory: one native build/path, two streamed compiles.

No RNG, state/physical-H reconstruction, signal/shot evaluation or retry.
Reuse the previous input/trajectory/compiler helpers and all circuit emitters.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
import tracemalloc
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_b2_p1_compiled_cost as previous
from trotterlib.b2_p1_compiled_cost import request_instruction_bound
from trotterlib.b2_streamed_native_cost import stream_native
from trotterlib.pauli_signal_adapter import digest
from trotterlib.rte_compiled_cost import _numeric_parameter_payload
from qiskit.converters import circuit_to_dag

BASE='1bed45be1869f447cfd90cef443877fcaed7af6c'
SAVED='2d21c23879aaae063a796fa95c1433fb3d0b5738'
OLD_SOURCE='5aa5d061600382abe438707c14eaf665466ae383'
STREAM_SOURCE='c897389643a8d3b81ff63f9a0d71b250b17a565e'
PREFIX='artifacts/track_a_b2_p1_compiled_cost/2026-11-07/run01/'
TRAJECTORY='c3ea48da7dcd082f948b6df9879b3105fc44145c438342396a9777d0b409fbc5'
COMPILER='a0301472e2fb95ba82d3ce80dace0d6de9637ccc74b1887d8271ca1c3e5c04d8'
STREAM_PATH='src/trotterlib/b2_streamed_native_cost.py'
OWN='scripts/run_b2_m0_streaming_h4.py'
require=previous.require
dump=previous.dump


def json_value(value):
    return json.loads(json.dumps(value,allow_nan=False))


def instruction_rows(circuit):
    return [dict(name=x.operation.name,qubits=[circuit.find_bit(q).index for q in x.qubits],
        clbits=[circuit.find_bit(c).index for c in x.clbits],
        parameters=[_numeric_parameter_payload(v) for v in x.operation.params])
        for x in circuit.data]


def compare_native(reference,candidate,saved_fingerprint):
    left,right=instruction_rows(reference.circuit),instruction_rows(candidate.circuit)
    checks=dict(system_qubits_equal=reference.num_system_qubits==candidate.num_system_qubits,
        qubits_equal=reference.circuit.num_qubits==candidate.circuit.num_qubits,
        classical_bits_equal=reference.circuit.num_clbits==candidate.circuit.num_clbits==0,
        ordinary_control=reference.control_policy==candidate.control_policy=='ordinary',
        instruction_rows_equal=left==right,
        global_phase_equal=float(reference.circuit.global_phase).hex()==float(candidate.circuit.global_phase).hex(),
        fingerprint_equal=reference.circuit_fingerprint==candidate.circuit_fingerprint==saved_fingerprint)
    return dict(status='EXACT_MATCH' if all(checks.values()) else 'INCONCLUSIVE',checks=checks,
        reference_fingerprint=reference.circuit_fingerprint,candidate_fingerprint=candidate.circuit_fingerprint,
        saved_fingerprint=saved_fingerprint,instruction_count=len(left),
        instruction_rows_digest=digest(left),phase_fitting=False)


def compare_axis(record,saved):
    delta={name:record['metrics'][name]-saved['metrics'][name]
           for name in previous.RPE_COST_METRICS}
    identities={name:record[name]==saved[name] for name in (
        'actual_circuit_fingerprint','compiled_circuit_fingerprint','compiler_hash',
        'backend_fingerprint','scope','state_preparation_included','qubits')}
    return dict(status='EXACT_MATCH' if all(identities.values()) and not any(delta.values()) else 'INCONCLUSIVE',
        circuit_identity_checks=identities,reference_metrics=saved['metrics'],
        candidate_metrics=record['metrics'],signed_difference=delta,
        absolute_difference={k:abs(v) for k,v in delta.items()})


def rss():
    fields=Path('/proc/self/status').read_text().splitlines()
    current=int(next(v.split()[1] for v in fields if v.startswith('VmRSS:')))
    return dict(current_RSS_KiB=current,
                process_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def profile(function,*,trace_python=False):
    gc.collect();before=rss()
    if trace_python: tracemalloc.start()
    tick=time.perf_counter()
    try:
        value=function()
        measured=dict(elapsed_seconds=time.perf_counter()-tick,RSS_before=before,RSS_after=rss(),
            Python_traced_peak_bytes=tracemalloc.get_traced_memory()[1] if trace_python else None,
            compile_incremental_peak_RSS_KiB=None)
    finally:
        if trace_python: tracemalloc.stop()
    return value,measured


def run(private_dir,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    result=dict(schema_version='b2_m0_streaming_h4_v1',status='PREFLIGHT',mandatory_stop=True,
        started_UTC=datetime.now(timezone.utc).isoformat(),compile_calls_started=0,axes={},
        fixed_conditions=dict(model='Linear H4;1.00 Angstrom;STO-3G',target='H_phys',DF_rank=12,
            L_D=3,r=4,K=2,delta=.02,m=0,Q=1,state_role='original_H_ground_state',
            state_preparation_included=False),expected_prices=None,price_MC=None,total_RPE_cost=None,
        guarantee_scope='one saved m0 trajectory; no all-support or long-circuit certification')
    try:
        source=previous.git('rev-parse','HEAD').decode().strip()
        old,old_identity=previous.saved(SAVED,PREFIX+'result.json')
        stored,trajectory_identity=previous.saved(SAVED,PREFIX+'trajectory.json')
        correctness,correctness_identity=previous.saved(previous.CORRECT_COMMIT,previous.CORRECT_PATH)
        spec,spec_identity=previous.saved(previous.SPEC_COMMIT,previous.SPEC_PATH)
        require(old['status']=='PASS_SPECIFIED_TRAJECTORY_M0_ONLY' and
            old['source']['commit']==OLD_SOURCE and old['compile_calls_started']==2,'SAVED_PRICE_STATUS_SOURCE')
        source_paths=(*previous.CORE,STREAM_PATH,'scripts/run_b2_p1_compiled_cost.py',OWN)
        source_hashes={}
        for path in source_paths:
            raw=(ROOT/path).read_bytes()
            require(raw==previous.git('show',source+':'+path),'COMMITTED_SOURCE: '+path)
            if path in previous.CORE or path=='scripts/run_b2_p1_compiled_cost.py':
                require(hashlib.sha256(raw).hexdigest()==old['source']['sha256'][path],
                        'ORIGINAL_NATIVE_COMPILER_SOURCE: '+path)
            if path==STREAM_PATH:
                require(raw==previous.git('show',STREAM_SOURCE+':'+path),'FIXED_STREAM_SOURCE')
            source_hashes[path]=hashlib.sha256(raw).hexdigest()
        result['source']=dict(commit=source,sha256=source_hashes)
        compiler=previous.compiler_from_saved({'compiler':old['compiler']['settings']},
            importlib.metadata.version('qiskit'))
        provider=previous.MatchedNoPrepCostProvider(compiler)
        require(previous.compiler_settings_hash(compiler)==old['compiler']['settings_hash']==COMPILER
            and provider.backend_hash==old['compiler']['backend_fingerprint']=='no_backend','COMPILER_BACKEND')
        require(old['price']['circuit_cost_scope']==previous.RPE_HADAMARD_INTERROGATION_SCOPE and
                old['price']['state_preparation_included'] is False,'FULL_NO_PREP_SCOPE')
        checks=correctness['controlled_correctness']['checks']
        require(correctness['controlled_correctness']['status']=='PASS' and
                len(checks)==16 and all(c['passed'] for c in checks),'CONTROLLED_16_PASS')
        for key in ('original_hamiltonian','original_state','df_hamiltonian','sector','spin_orbital_order','state_role'):
            require(spec[key]==old['input']['scientific_identity'][key]==correctness['input_identity'][key],
                    'SCIENTIFIC_METADATA: '+key)
        private_files={name:Path(private_dir)/name for name in old['input']['private_file_sha256']}
        private_hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in private_files.items()}
        require(private_hashes==old['input']['private_file_sha256'],'PRIVATE_FILE_HASHES')
        raw_df=previous.git('show',previous.DF_COMMIT+':'+previous.DF_PATH)
        require(hashlib.sha256(raw_df).hexdigest()==previous.DF_SHA==old['input']['df_sha256'],'DF_SNAPSHOT_HASH')
        require(platform.node()=='GPUA100-gleap02','SERVER_2')
        result.update(input=old['input'],compiler=old['compiler'],
            saved_references=dict(price=old_identity,trajectory=trajectory_identity,
                                  controlled_correctness=correctness_identity,input_spec=spec_identity),
            controlled_evidence=dict(reused_checks=16,status='PASS',
                scope=correctness['controlled_correctness']['scope'],new_H4_operator_or_signal_simulation=False),
            runtime=dict(host=platform.node(),python=sys.version,available_cpus=len(os.sched_getaffinity(0)),
                load_average=os.getloadavg(),packages={p:importlib.metadata.version(p) for p in
                    ('qiskit','numpy','scipy','openfermion','qiskit-nature')}))
        with tempfile.TemporaryDirectory(prefix='b2-stream-fixed-df-') as temp:
            path=Path(temp)/'snapshot.npz';path.write_bytes(raw_df)
            ham,sector,unused_df_state,metadata=previous.load_snapshot(path)
            del unused_df_state
        require(ham.n_blocks==12 and metadata['hamiltonian_hash']==spec['df_hamiltonian'] and
            metadata['fragment_order']==list(range(12)) and sector.n_qubits==8 and
            spec['sector'].endswith(metadata['sector']['basis_indices_hash']),'DF12_SECTOR_ORDER')
        preparation=previous.connection.prepare_b2(ham,L_D=3)
        config,distribution=previous.configure_law(SimpleNamespace(tail=preparation.rte_preparation.symbolic_tail),
                                                  delta_time=.02,r=4,K=2)
        require(json_value(asdict(config))==old['law']['config'] and
                json_value(asdict(distribution))==old['law']['distribution'] and
                preparation.preparation_hash==old['law']['preparation_hash'] and
                preparation.constant_coefficient==old['law']['scalar'] and
                preparation.extracted_identity_coefficient==old['law']['tail_identity'],'CANONICAL_LAW_IDENTITY')
        context=previous.connection.RPEConnectionContext(spec['original_hamiltonian'],spec['df_hamiltonian'],
            spec['original_state'],spec['sector'],spec['spin_orbital_order'],source,compiler,
            provider.backend_hash,'rz_count',None,None,None,None)
        round_=previous.connection.B2RPERound(context,preparation,previous.RPERoundSpecification(0,.02),
                                            3,4,2,.02,1,config,distribution)
        events,selection=previous.fixed_events(round_,correctness)
        event_dicts=[e.to_dict() for e in events]
        require(event_dicts==stored['canonical_events']==old['price']['trajectory'] and
                digest(event_dicts)==old['price']['trajectory_identity']==TRAJECTORY and
                list(selection['representative_indices'])==[0,4,4,0] and
                [e.taylor_order for e in events]==[0,2,2,0],'FULL_SAVED_TRAJECTORY_REPLAY')
        result['trajectory']=dict(identity=TRAJECTORY,selection=selection,
            full_event_fields_match=True,orders=[e.taylor_order for e in events],
            probabilities=[e.event_probability for e in events],new_sampling=False)
        result['preflight_status']='PASS';dump(output/'preflight.json',result)
        def reference_build():
            request=round_.explicit_request(events)
            return round_.native(request,max_instructions=request_instruction_bound(request))
        reference,ref_performance=profile(reference_build,trace_python=True)
        candidate,candidate_performance=profile(lambda:stream_native(round_,iter(events)),trace_python=True)
        result['native_comparison']=compare_native(reference,candidate,old['price']['evolution_fingerprint'])
        result['build_performance']=dict(reference=ref_performance,stream=candidate_performance,
            protocol='same process; reference then stream; tracemalloc per build including fingerprint; reference circuit retained',
            each_path_build_count=1)
        dump(output/'result.json',result)
        require(result['native_comparison']['status']=='EXACT_MATCH','NATIVE_GATE_SEQUENCE_OR_IDENTITY')
        wrappers={};saved_axes={};result['wrappers']={}
        for axis in ('cosine','sine'):
            saved_axes[axis]=previous.saved(SAVED,PREFIX+axis+'.json')[0]
            old_wrapper=previous.connection.build_native_hadamard_wrapper(reference,axis=axis,
                max_instructions=len(reference.circuit.data)+4)
            wrappers[axis]=previous.connection.build_native_hadamard_wrapper(candidate,axis=axis,
                max_instructions=len(candidate.circuit.data)+4)
            fp=previous.canonical_qiskit_circuit_fingerprint(wrappers[axis])
            match=(fp==previous.canonical_qiskit_circuit_fingerprint(old_wrapper)==
                saved_axes[axis]['actual_circuit_fingerprint'] and
                instruction_rows(old_wrapper)==instruction_rows(wrappers[axis]))
            result['wrappers'][axis]=dict(status='EXACT_MATCH' if match else 'INCONCLUSIVE',
                fingerprint=fp,instructions=len(wrappers[axis].data),
                qubits=wrappers[axis].num_qubits,clbits=wrappers[axis].num_clbits,
                measurements=int(wrappers[axis].count_ops().get('measure',0)))
            require(match and wrappers[axis].num_qubits==9 and wrappers[axis].num_clbits==1
                    and wrappers[axis].count_ops().get('measure')==1,'WRAPPER_IDENTITY_SCOPE: '+axis)
        dump(output/'native_and_wrappers.json',dict(native=result['native_comparison'],wrappers=result['wrappers']))
        def compile_once(circuit,settings,**kwargs):
            require(result['compile_calls_started']<2,'MAXIMUM_TWO_COMPILER_CALLS')
            axis=('cosine','sine')[result['compile_calls_started']]
            require(circuit is wrappers[axis],'WRONG_AXIS_CIRCUIT')
            result['compile_calls_started']+=1
            cost,performance=profile(lambda:previous.transpile_and_measure_cost(circuit,settings,**kwargs))
            record=dict(axis=axis,SOURCE=source,scope=context.cost_scope,state_preparation_included=False,
                qubits=cost.qubit_count,actual_circuit_fingerprint=cost.actual_circuit_fingerprint,
                compiled_circuit_fingerprint=previous.canonical_qiskit_circuit_fingerprint(cost.transpiled_circuit),
                compiler_hash=cost.compiler_settings_hash,backend_fingerprint=cost.backend_fingerprint,
                metrics={k:getattr(cost,k) for k in previous.RPE_COST_METRICS},
                measurements=int(cost.transpiled_circuit.count_ops().get('measure',0)),
                pretranspile_gate_counts=dict(cost.pretranspile_gate_counts),
                posttranspile_gate_counts=dict(cost.posttranspile_gate_counts),
                compiled_global_phase=cost.global_phase,performance=performance,
                compiled_DAG_operations=len(circuit_to_dag(cost.transpiled_circuit).op_nodes()))
            record['comparison']=compare_axis(record,saved_axes[axis])
            result['axes'][axis]=record;dump(output/(axis+'.json'),record)
            dump(output/'result.json',result)
            require(record['measurements']==1 and record['comparison']['status']=='EXACT_MATCH',
                    'SAVED_COMPILED_PRICE_OR_IDENTITY: '+axis)
            print(axis+': all six metrics and compiled identity EXACT_MATCH',flush=True)
            return cost
        previous.connection.accounting._compile_pair(candidate,compiler,
            compile_function=compile_once,backend=None,wrapper_builder=lambda _,axis:wrappers[axis])
        for path,expected in source_hashes.items():
            require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,'CONSUMED_SOURCE_CHANGED')
        require({name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in private_files.items()}==private_hashes,
                'PRIVATE_INPUT_CHANGED')
        result['status']='PASS_M0_STREAMING_IDENTICAL'
    except Exception as exc:
        result.update(status='INCONCLUSIVE',stop_reason=type(exc).__name__+': '+str(exc))
    result['elapsed_seconds']=time.perf_counter()-started
    result['finished_UTC']=datetime.now(timezone.utc).isoformat()
    dump(output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--private-dir',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(args.private_dir,args.output)
    print(result['status'],result.get('stop_reason',''),flush=True)
    sys.exit(0 if result['status']=='PASS_M0_STREAMING_IDENTICAL' else 1)
