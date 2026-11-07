"""Fixed Q2 H4 comparison; one trajectory, two builds, four whole compiles.

Reuse m0 input/identity/measurement helpers and the unchanged B2 emitters.
No m0 evolution, RNG sampling, signals, state solve, alternative compiler or retry.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import run_b2_m0_streaming_h4 as m0

BASE='f6eea2f28be1f15b6ff5e2954b9c8fb48f3c5a0f'
M0_PATH='artifacts/track_a_b2_p1_m0_streaming_h4/2026-11-07/run02/result.json'
OWN='scripts/run_b2_q2_streaming_h4.py'
PATTERN=(0,4,4,0,4,0,0,4)
TRAJECTORY='0ecf28e7a5d2d367c6c609a8d8a4da544c2d03b955aef86b0656f0bf107b7ae8'
previous=m0.previous
require=previous.require
dump=previous.dump


def two_step_events(first):
    """Fixed user ordering; never reverse the palindromic first step."""
    first=tuple(first)
    require(len(first)==4 and first[0]==first[3] and first[1]==first[2],
            'FIXED_FIRST_STEP_STRUCTURE')
    e0,e4=first[:2]
    return (e0,e4,e4,e0,e4,e0,e0,e4)


def compile_both_paths(result,evolutions,wrappers,compiler,output,source,
                       compile_function=previous.transpile_and_measure_cost):
    """Use existing pair compiler, saving the new Q2 reference before comparison."""
    expected_slots=[(path,axis) for path in ('reference','stream') for axis in ('cosine','sine')]
    for path in ('reference','stream'):
        def compile_once(circuit,settings,**kwargs):
            count=result['compile_calls_started']
            require(count<4 and expected_slots[count][0]==path,'MAX_FOUR_FIXED_COMPILER_CALLS')
            axis=expected_slots[count][1]
            require(circuit is wrappers[path][axis],'FIXED_PATH_AXIS')
            result['compile_calls_started']+=1
            cost,performance=m0.profile(lambda:compile_function(circuit,settings,**kwargs))
            compiled=cost.transpiled_circuit
            record=dict(path=path,axis=axis,SOURCE=source,
                scope=previous.RPE_HADAMARD_INTERROGATION_SCOPE,state_preparation_included=False,
                qubits=cost.qubit_count,actual_circuit_fingerprint=cost.actual_circuit_fingerprint,
                compiled_circuit_fingerprint=previous.canonical_qiskit_circuit_fingerprint(compiled),
                compiler_hash=cost.compiler_settings_hash,backend_fingerprint=cost.backend_fingerprint,
                metrics={k:getattr(cost,k) for k in previous.RPE_COST_METRICS},
                measurements=int(compiled.count_ops().get('measure',0)),
                pretranspile_gate_counts=dict(cost.pretranspile_gate_counts),
                posttranspile_gate_counts=dict(cost.posttranspile_gate_counts),
                compiled_global_phase=cost.global_phase,performance=performance,
                compiled_DAG_operations=len(m0.circuit_to_dag(compiled).op_nodes()))
            require(record['compiler_hash']==m0.COMPILER and record['backend_fingerprint']=='no_backend'
                and record['measurements']==1 and compiled.num_clbits==1 and
                record['actual_circuit_fingerprint']==kwargs['actual_circuit_fingerprint'],
                'COMPILED_CONTEXT_SCOPE')
            if path=='stream':
                record['comparison']=m0.compare_axis(record,result['axes']['reference'][axis])
            result['axes'][path][axis]=record
            dump(output/(path+'_'+axis+'.json'),record);dump(output/'result.json',result)
            if path=='stream':
                require(record['comparison']['status']=='EXACT_MATCH','Q2_COMPILED_IDENTITY_OR_PRICE: '+axis)
            print(path+'/'+axis+': '+str(record['metrics']),flush=True)
            return cost
        previous.connection.accounting._compile_pair(evolutions[path],compiler,
            compile_function=compile_once,backend=None,wrapper_builder=lambda _,axis:wrappers[path][axis])


def run(private_dir,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    result=dict(schema_version='b2_q2_streaming_h4_v1',status='PREFLIGHT',mandatory_stop=True,
        started_UTC=datetime.now(timezone.utc).isoformat(),compile_calls_started=0,
        axes={'reference':{},'stream':{}},expected_prices=None,price_MC=None,total_RPE_cost=None,
        fixed_conditions=dict(model='Linear H4;1.00 Angstrom;STO-3G',target='H_phys',DF_rank=12,
            L_D=3,r=4,K=2,delta=.02,m=1,Q=2,T=.04,state_role='original_H_ground_state',
            state_preparation_included=False,controlled='ordinary'))
    try:
        source=previous.git('rev-parse','HEAD').decode().strip()
        last,last_identity=previous.saved(BASE,M0_PATH)
        stored,stored_identity=previous.saved(m0.SAVED,m0.PREFIX+'trajectory.json')
        correctness,correctness_identity=previous.saved(previous.CORRECT_COMMIT,previous.CORRECT_PATH)
        spec,spec_identity=previous.saved(previous.SPEC_COMMIT,previous.SPEC_PATH)
        require(last['status']=='PASS_M0_STREAMING_IDENTICAL' and
            last['source']['commit']=='db7f0daeffc9345b69fd8695283a20fea2f8e764','M0_INPUT_EVIDENCE')
        hashes={}
        for path in (*last['source']['sha256'],OWN):
            raw=(ROOT/path).read_bytes()
            require(raw==previous.git('show',source+':'+path),'COMMITTED_SOURCE: '+path)
            if path in last['source']['sha256']:
                require(hashlib.sha256(raw).hexdigest()==last['source']['sha256'][path],
                        'FIXED_NATIVE_STREAM_COMPILER_SOURCE: '+path)
            hashes[path]=hashlib.sha256(raw).hexdigest()
        result['source']=dict(commit=source,sha256=hashes)
        compiler=previous.compiler_from_saved({'compiler':last['compiler']['settings']},
                                              importlib.metadata.version('qiskit'))
        provider=previous.MatchedNoPrepCostProvider(compiler)
        require(previous.compiler_settings_hash(compiler)==last['compiler']['settings_hash']==m0.COMPILER
            and provider.backend_hash==last['compiler']['backend_fingerprint']=='no_backend','FIXED_COMPILER_BACKEND')
        require(all(a['scope']==previous.RPE_HADAMARD_INTERROGATION_SCOPE and
                    a['state_preparation_included'] is False for a in last['axes'].values()),'FULL_NO_PREP_SCOPE')
        require(correctness['controlled_correctness']['status']=='PASS' and
                len(correctness['controlled_correctness']['checks'])==16 and
                all(c['passed'] for c in correctness['controlled_correctness']['checks']),'CONTROLLED_EVIDENCE')
        for key in ('original_hamiltonian','original_state','df_hamiltonian','sector','spin_orbital_order','state_role'):
            require(spec[key]==last['input']['scientific_identity'][key]==correctness['input_identity'][key],
                    'SCIENTIFIC_METADATA: '+key)
        private_files={name:Path(private_dir)/name for name in last['input']['private_file_sha256']}
        private_hashes={name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in private_files.items()}
        require(private_hashes==last['input']['private_file_sha256'],'PRIVATE_INPUT_HASHES')
        raw_df=previous.git('show',previous.DF_COMMIT+':'+previous.DF_PATH)
        require(hashlib.sha256(raw_df).hexdigest()==previous.DF_SHA==last['input']['df_sha256'],'DF_SNAPSHOT_HASH')
        require(platform.node()=='GPUA100-gleap02','SERVER_2')
        result.update(input=last['input'],compiler=last['compiler'],
            reused_evidence=dict(m0_metadata_only=last_identity,base_event_metadata=stored_identity,
                                  controlled=correctness_identity,input_spec=spec_identity),
            controlled_evidence=dict(reused_checks=16,status='PASS',
                scope=correctness['controlled_correctness']['scope'],new_H4_signal_or_statevector_simulation=False),
            runtime=dict(python=sys.version,host=platform.node(),available_cpus=len(os.sched_getaffinity(0)),
                load_average=os.getloadavg(),packages={p:importlib.metadata.version(p) for p in
                    ('qiskit','numpy','scipy','openfermion','qiskit-nature')}))
        with tempfile.TemporaryDirectory(prefix='b2-q2-fixed-df-') as temp:
            path=Path(temp)/'snapshot.npz';path.write_bytes(raw_df)
            ham,sector,unused_df_state,metadata=previous.load_snapshot(path);del unused_df_state
        require(ham.n_blocks==12 and metadata['hamiltonian_hash']==spec['df_hamiltonian'] and
            metadata['fragment_order']==list(range(12)) and sector.n_qubits==8 and
            spec['sector'].endswith(metadata['sector']['basis_indices_hash']),'DF12_SECTOR_ORDER')
        p=previous.connection.prepare_b2(ham,L_D=3)
        config,distribution=previous.configure_law(SimpleNamespace(tail=p.rte_preparation.symbolic_tail),
                                                   delta_time=.02,r=4,K=2)
        old_price=previous.saved(m0.SAVED,m0.PREFIX+'result.json')[0]
        require(m0.json_value(asdict(config))==old_price['law']['config'] and
            m0.json_value(asdict(distribution))==old_price['law']['distribution'] and
            p.preparation_hash==old_price['law']['preparation_hash'] and
            p.constant_coefficient==old_price['law']['scalar'] and
            p.extracted_identity_coefficient==old_price['law']['tail_identity'],'FIXED_CANONICAL_LAW')
        context=previous.connection.RPEConnectionContext(spec['original_hamiltonian'],spec['df_hamiltonian'],
            spec['original_state'],spec['sector'],spec['spin_orbital_order'],source,compiler,
            provider.backend_hash,'rz_count',None,None,None,None)
        round_=previous.connection.B2RPERound(context,p,previous.RPERoundSpecification(1,.02),
                                            3,4,2,.02,2,config,distribution)
        first,selection=previous.fixed_events(round_,correctness)
        m0.saved_event_payload(first,stored['canonical_events'],m0.TRAJECTORY)
        events=two_step_events(first)
        payload=m0.json_value([e.to_dict() for e in events])
        e0,e4=stored['canonical_events'][:2]
        require(payload==[e0,e4,e4,e0,e4,e0,e0,e4] and m0.digest(payload)==TRAJECTORY
            and list(selection['representative_indices'])==[0,4,4,0],'FIXED_Q2_TRAJECTORY')
        previous.connection.accounting._canonical_events(round_,config,distribution,events,2)
        trajectory=dict(identity=TRAJECTORY,representative_indices=list(PATTERN),
            step_indices=[[0,4,4,0],[4,0,0,4]],canonical_events=payload,
            new_sampling=False,price_based_selection=False,round_binding=round_.binding)
        dump(output/'trajectory.json',trajectory)
        result['trajectory']=dict(identity=TRAJECTORY,representative_indices=list(PATTERN),
            step_orders=[[e.taylor_order for e in events[:4]],[e.taylor_order for e in events[4:]]],
            complete_payload_path='trajectory.json',canonical_validation='PASS',new_sampling=False)
        result['preflight_status']='PASS';dump(output/'preflight.json',result)
        def reference_build():
            request=round_.explicit_request(events)
            result['reference_step_requests']=dict(Q=request.repetition_count,
                event_component_ids=[[list(e.selected_component_ids) for e in o.events] for o in request.rte_occurrences],
                sampling_policy=request.sampling_policy)
            return round_.native(request,max_instructions=m0.request_instruction_bound(request))
        reference,ref_perf=m0.profile(reference_build,trace_python=True)
        candidate,stream_perf=m0.profile(lambda:m0.stream_native(round_,iter(events)),trace_python=True)
        comparison=m0.compare_native(reference,candidate,reference.circuit_fingerprint)
        comparison.pop('saved_fingerprint');comparison['reference_origin']='newly built Q2 native; not m0 extrapolation'
        result['native_comparison']=comparison
        result['build_performance']=dict(reference=ref_perf,stream=stream_perf,
            protocol='same process; reference then stream; each once; tracemalloc includes requests/native fingerprint; reference retained')
        dump(output/'result.json',result)
        require(comparison['status']=='EXACT_MATCH' and reference.q==candidate.q==2
            and reference.T==candidate.T==.04,'NATIVE_GATE_IDENTITY_Q2')
        evolutions={'reference':reference,'stream':candidate};wrappers={}
        result['wrappers']={}
        for path,evolution in evolutions.items():
            wrappers[path]={};result['wrappers'][path]={}
            for axis in ('cosine','sine'):
                w=previous.connection.build_native_hadamard_wrapper(evolution,axis=axis,
                    max_instructions=len(evolution.circuit.data)+4)
                require(w.num_qubits==9 and w.num_clbits==1 and w.count_ops().get('measure')==1,'WRAPPER_SCOPE')
                wrappers[path][axis]=w
                result['wrappers'][path][axis]=dict(fingerprint=previous.canonical_qiskit_circuit_fingerprint(w),
                    instructions=len(w.data),qubits=w.num_qubits,clbits=w.num_clbits,measurements=1)
        for axis in ('cosine','sine'):
            require(result['wrappers']['reference'][axis]['fingerprint']==result['wrappers']['stream'][axis]['fingerprint']
                and m0.instruction_rows(wrappers['reference'][axis])==m0.instruction_rows(wrappers['stream'][axis]),
                'WRAPPER_IDENTITY: '+axis)
        dump(output/'native_and_wrappers.json',dict(native=comparison,wrappers=result['wrappers']))
        compile_both_paths(result,evolutions,wrappers,compiler,output,source)
        for path,sha in hashes.items():require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,'SOURCE_CHANGED')
        require({name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in private_files.items()}==private_hashes,
                'PRIVATE_INPUT_CHANGED')
        result['status']='PASS_Q2_STREAMING_IDENTICAL'
    except Exception as exc:
        result.update(status='INCONCLUSIVE',stop_reason=type(exc).__name__+': '+str(exc))
    result['elapsed_seconds']=time.perf_counter()-started
    result['finished_UTC']=datetime.now(timezone.utc).isoformat();dump(output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--private-dir',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    result=run(args.private_dir,args.output)
    print(result['status'],result.get('stop_reason',''),flush=True)
    sys.exit(0 if result['status']=='PASS_Q2_STREAMING_IDENTICAL' else 1)
