"""Fixed real H4 Q2: one streamed native build and two whole compiles.

Replay saved inputs/events and compare against the immutable Q2 reference.
No reference rebuild, m0/m2+, RNG, signals, state solve, MC or retry.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
import run_b2_q2_streaming_h4 as q2
import benchmark_b2_fingerprint_pipeline as pipeline
from trotterlib.streamed_circuit_fingerprint import streamed_canonical_qiskit_circuit_fingerprint as streamed_hash

m0 = q2.m0
previous = q2.previous
require, dump = previous.require, previous.dump
REFERENCE = 'ce2e6736ca1178a4adafc4af344852d958951ea6'
PREFIX = 'artifacts/track_a_b2_p1_q2_streaming_h4/2026-11-07/run01/'
PIPELINE = 'c71bb8227000bd544aa0c4f946c0e6cf28a80bee'
OWN = 'scripts/run_b2_q2_fingerprint_h4.py'
OPT_IN = ('src/trotterlib/b2_streamed_native_cost.py',
          'src/trotterlib/o1_m1_resource_bridge.py',
          'src/trottertracks/resource_applicability/ax2a_native_df.py')
EXTRA = ('src/trotterlib/streamed_circuit_fingerprint.py',
         'scripts/benchmark_b2_fingerprint_pipeline.py',
         'scripts/benchmark_streamed_circuit_fingerprint.py')


def prepare_fixed(private_dir, output, result):
    """Reuse prior input/law/event helpers; never call an old runner's main."""
    source = previous.git('rev-parse', 'HEAD').decode().strip()
    reference, ref_id = previous.saved(REFERENCE, PREFIX+'result.json')
    stored, trajectory_id = previous.saved(REFERENCE, PREFIX+'trajectory.json')
    correctness, correctness_id = previous.saved(previous.CORRECT_COMMIT, previous.CORRECT_PATH)
    spec, spec_id = previous.saved(previous.SPEC_COMMIT, previous.SPEC_PATH)
    law = previous.saved(m0.SAVED, m0.PREFIX+'result.json')[0]['law']
    fixed = dict(model='Linear H4;1.00 Angstrom;STO-3G', target='H_phys', DF_rank=12,
        L_D=3, r=4, K=2, delta=.02, m=1, Q=2, T=.04, state_role='original_H_ground_state',
        state_preparation_included=False, controlled='ordinary')
    require(reference['status']=='PASS_Q2_STREAMING_IDENTICAL' and reference['compile_calls_started']==4
        and reference['source']['commit']=='e09fd13c5a2357dd001d9cfbe86d4e99c3a5620d'
        and reference['fixed_conditions']==fixed, 'FIXED_SAVED_Q2_REFERENCE')
    hashes = {}
    for path in (*reference['source']['sha256'], *EXTRA, OWN):
        raw = (ROOT/path).read_bytes()
        require(raw==previous.git('show', source+':'+path), 'COMMITTED_SOURCE: '+path)
        if path in OPT_IN or path in EXTRA:
            require(raw==previous.git('show', PIPELINE+':'+path), 'FIXED_OPT_IN_PIPELINE: '+path)
        elif path in reference['source']['sha256']:
            require(hashlib.sha256(raw).hexdigest()==reference['source']['sha256'][path],
                    'UNCHANGED_EMITTER_LAW_COMPILER: '+path)
        hashes[path] = hashlib.sha256(raw).hexdigest()
    compiler = previous.compiler_from_saved({'compiler':reference['compiler']['settings']},
                                            importlib.metadata.version('qiskit'))
    provider = previous.MatchedNoPrepCostProvider(compiler)
    require(previous.compiler_settings_hash(compiler)==reference['compiler']['settings_hash']==m0.COMPILER
        and provider.backend_hash==reference['compiler']['backend_fingerprint']=='no_backend', 'COMPILER_BACKEND')
    require(all(a['scope']==previous.RPE_HADAMARD_INTERROGATION_SCOPE and
        a['state_preparation_included'] is False for a in reference['axes']['reference'].values()), 'FULL_NO_PREP_SCOPE')
    checks = correctness['controlled_correctness']['checks']
    require(correctness['controlled_correctness']['status']=='PASS' and len(checks)==16
        and all(c['passed'] for c in checks), 'REUSED_CONTROLLED_16')
    for key in ('original_hamiltonian','original_state','df_hamiltonian','sector','spin_orbital_order','state_role'):
        require(spec[key]==reference['input']['scientific_identity'][key]==correctness['input_identity'][key],
                'SCIENTIFIC_METADATA: '+key)
    private_files = {name:Path(private_dir)/name for name in reference['input']['private_file_sha256']}
    private_hashes = {name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in private_files.items()}
    require(private_hashes==reference['input']['private_file_sha256'], 'PRIVATE_INPUT_HASHES')
    raw_df = previous.git('show', previous.DF_COMMIT+':'+previous.DF_PATH)
    require(hashlib.sha256(raw_df).hexdigest()==previous.DF_SHA==reference['input']['df_sha256'], 'DF_SNAPSHOT')
    require(platform.node()=='GPUA100-gleap02', 'SERVER_2')
    result.update(fixed_conditions=fixed, source=dict(commit=source, sha256=hashes),
        input=reference['input'], compiler=reference['compiler'],
        saved_references=dict(Q2=ref_id, trajectory=trajectory_id, controlled=correctness_id, input_spec=spec_id),
        controlled_evidence=dict(reference['controlled_evidence']),
        runtime=dict(host=platform.node(), python=sys.version, available_cpus=len(os.sched_getaffinity(0)),
            load_average=os.getloadavg(), packages={p:importlib.metadata.version(p) for p in
                ('qiskit','numpy','scipy','openfermion','qiskit-nature')},
            MemAvailable=next(l for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))))
    with tempfile.TemporaryDirectory(prefix='b2-q2-hash-fixed-df-') as temp:
        path = Path(temp)/'snapshot.npz'; path.write_bytes(raw_df)
        ham, sector, unused_df_state, metadata = previous.load_snapshot(path)
        del unused_df_state
    require(ham.n_blocks==12 and metadata['hamiltonian_hash']==spec['df_hamiltonian']
        and metadata['fragment_order']==list(range(12)) and sector.n_qubits==8
        and spec['sector'].endswith(metadata['sector']['basis_indices_hash']), 'DF12_SECTOR_ORDER')
    preparation = previous.connection.prepare_b2(ham, L_D=3)
    config, distribution = previous.configure_law(SimpleNamespace(tail=preparation.rte_preparation.symbolic_tail),
                                                 delta_time=.02, r=4, K=2)
    require(m0.json_value(asdict(config))==law['config'] and m0.json_value(asdict(distribution))==law['distribution']
        and preparation.preparation_hash==law['preparation_hash']
        and preparation.constant_coefficient==law['scalar']
        and preparation.extracted_identity_coefficient==law['tail_identity'], 'FIXED_CANONICAL_LAW')
    context = previous.connection.RPEConnectionContext(spec['original_hamiltonian'], spec['df_hamiltonian'],
        spec['original_state'], spec['sector'], spec['spin_orbital_order'], source, compiler,
        provider.backend_hash, 'rz_count', None, None, None, None)
    round_ = previous.connection.B2RPERound(context, preparation, previous.RPERoundSpecification(1,.02),
                                          3,4,2,.02,2,config,distribution)
    first, selection = previous.fixed_events(round_, correctness)
    events = q2.two_step_events(first)
    payload = m0.saved_event_payload(events, stored['canonical_events'], q2.TRAJECTORY)
    require(list(selection['representative_indices'])==[0,4,4,0]
        and stored['representative_indices']==list(q2.PATTERN)
        and stored['identity']==q2.TRAJECTORY, 'FIXED_Q2_EVENT_ORDER')
    previous.connection.accounting._canonical_events(round_,config,distribution,events,2)
    dump(output/'trajectory.json', dict(identity=q2.TRAJECTORY, representative_indices=list(q2.PATTERN),
        step_indices=[[0,4,4,0],[4,0,0,4]], canonical_events=payload, new_sampling=False,
        price_based_selection=False, round_binding=round_.binding))
    result['trajectory'] = dict(identity=q2.TRAJECTORY, representative_indices=list(q2.PATTERN),
        canonical_validation='PASS', full_saved_event_fields_match=True, new_sampling=False)
    result['preflight_status'] = 'PASS'; dump(output/'preflight.json',result)
    return round_, events, reference, compiler, private_files, private_hashes


def native_comparison(candidate, reference):
    saved = reference['native_comparison']
    fingerprint_matches = candidate.circuit_fingerprint==saved['reference_fingerprint']
    rows_digest = m0.digest(m0.instruction_rows(candidate.circuit))
    checks = dict(fingerprint_equal=fingerprint_matches,
        global_phase_bound_by_fingerprint=fingerprint_matches,
        instruction_count_equal=len(candidate.circuit.data)==saved['instruction_count'],
        instruction_rows_digest_equal=rows_digest==saved['instruction_rows_digest'],
        qubits_equal=candidate.circuit.num_qubits==reference['axes']['reference']['cosine']['qubits'],
        system_and_ancilla=candidate.num_system_qubits+1==candidate.circuit.num_qubits,
        classical_bits_zero=candidate.circuit.num_clbits==0, ordinary_control=candidate.control_policy=='ordinary',
        Q_equal=candidate.q==reference['fixed_conditions']['Q'],
        T_equal=float(candidate.T).hex()==float(reference['fixed_conditions']['T']).hex())
    return dict(status='EXACT_MATCH' if all(checks.values()) else 'INCONCLUSIVE', checks=checks,
        candidate_fingerprint=candidate.circuit_fingerprint, saved_fingerprint=saved['reference_fingerprint'],
        instruction_count=len(candidate.circuit.data), instruction_rows_digest=rows_digest,
        global_phase=float(candidate.circuit.global_phase), phase_fitting=False,
        reference_origin='saved Q2 reference; no reference circuit rebuilt')


def evaluate_candidate(result, round_, events, reference, compiler, output, source,
                       compile_function=previous.transpile_and_measure_cost):
    """One actual native construction, two wrappers, at most two compiles."""
    require(result['native_builds_started']==result['compile_calls_started']==0, 'FIXED_EXECUTION_START')
    result['native_builds_started'] += 1
    candidate = pipeline.measured_native(round_, events, streamed_hash, result['stages'])
    result['native_comparison'] = native_comparison(candidate, reference)
    dump(output/'result.json',result)
    require(result['native_comparison']['status']=='EXACT_MATCH', 'SAVED_NATIVE_IDENTITY')
    wrappers = {}
    for axis in ('cosine','sine'):
        w = pipeline.measured_wrapper(candidate, axis, streamed_hash, result['stages'])
        wrappers[axis] = w
        fp, result['stages'][axis+'_precompile_fingerprint'] = pipeline.profile(lambda:streamed_hash(w))
        record = dict(fingerprint=fp, instructions=len(w.data), qubits=w.num_qubits,
                      clbits=w.num_clbits, measurements=int(w.count_ops().get('measure',0)))
        saved = reference['wrappers']['reference'][axis]
        checks = {name:record[name]==saved[name] for name in record}
        checks['saved_axis_fingerprint'] = fp==reference['axes']['reference'][axis]['actual_circuit_fingerprint']
        checks['pretranspile_gate_counts'] = dict(w.count_ops())==reference['axes']['reference'][axis]['pretranspile_gate_counts']
        record.update(status='EXACT_MATCH' if all(checks.values()) else 'INCONCLUSIVE', checks=checks)
        result['wrappers'][axis] = record; dump(output/'result.json',result)
        require(record['status']=='EXACT_MATCH', 'SAVED_WRAPPER_IDENTITY: '+axis)
    dump(output/'native_and_wrappers.json',dict(native=result['native_comparison'],wrappers=result['wrappers']))
    def input_hash(circuit):
        axis = next(a for a in wrappers if wrappers[a] is circuit)
        fp, result['stages'][axis+'_pair_input_fingerprint'] = pipeline.profile(lambda:streamed_hash(circuit))
        require(fp==result['wrappers'][axis]['fingerprint'], 'WRAPPER_CHANGED_BEFORE_COMPILE')
        return fp
    def compile_once(circuit, settings, **kwargs):
        count = result['compile_calls_started']
        require(count<2, 'MAXIMUM_TWO_COMPILES')
        axis = ('cosine','sine')[count]
        require(circuit is wrappers[axis] and settings==compiler and kwargs['backend'] is None
            and kwargs['actual_circuit_fingerprint']==result['wrappers'][axis]['fingerprint'], 'FIXED_AXIS_COMPILER_INPUT')
        result['compile_calls_started'] += 1
        cost, perf = pipeline.profile(lambda:compile_function(circuit,settings,**kwargs))
        result['stages'][axis+'_whole_compile_and_metrics'] = perf
        compiled = cost.transpiled_circuit
        fp, result['stages'][axis+'_compiled_fingerprint'] = pipeline.profile(lambda:streamed_hash(compiled))
        dag, result['stages'][axis+'_compiled_DAG'] = pipeline.profile(lambda:pipeline.circuit_to_dag(compiled,copy_operations=True))
        operations = dag.size(); del dag; gc.collect()
        record = dict(axis=axis, SOURCE=source, scope=previous.RPE_HADAMARD_INTERROGATION_SCOPE,
            state_preparation_included=False, qubits=cost.qubit_count,
            actual_circuit_fingerprint=cost.actual_circuit_fingerprint, compiled_circuit_fingerprint=fp,
            compiler_hash=cost.compiler_settings_hash, backend_fingerprint=cost.backend_fingerprint,
            metrics={name:getattr(cost,name) for name in previous.RPE_COST_METRICS},
            measurements=int(compiled.count_ops().get('measure',0)), compiled_clbits=compiled.num_clbits,
            pretranspile_gate_counts=dict(cost.pretranspile_gate_counts),
            posttranspile_gate_counts=dict(cost.posttranspile_gate_counts), compiled_global_phase=cost.global_phase,
            compiled_DAG_operations=operations, performance=perf)
        record['comparison'] = m0.compare_axis(record,reference['axes']['reference'][axis])
        result['axes'][axis] = record; dump(output/(axis+'.json'),record); dump(output/'result.json',result)
        require(record['comparison']['status']=='EXACT_MATCH' and record['measurements']==1 and compiled.num_clbits==1
            and record['actual_circuit_fingerprint']==kwargs['actual_circuit_fingerprint']
            and float(record['compiled_global_phase']).hex()==float(reference['axes']['reference'][axis]['compiled_global_phase']).hex(),
            'SAVED_COMPILED_IDENTITY_OR_METRICS: '+axis)
        print(axis+': saved Q2 compiled identity and six metrics EXACT_MATCH',flush=True)
        return cost
    costs = previous.connection.accounting._compile_pair(candidate,compiler,compile_function=compile_once,
        backend=None,wrapper_builder=lambda value,axis:wrappers[axis],fingerprint_function=input_hash)
    require(len(costs)==2 and result['compile_calls_started']==2, 'EXACTLY_TWO_COMPLETED_AXES')
    gc.collect()
    result['new_pipeline_end_RSS'] = pipeline.rss()
    result['end_retention'] = dict(native=True,wrappers=2,compiled_circuits=2,diagnostic_DAGs=False)
    return candidate


def run(private_dir, output):
    output = Path(output); output.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    result = dict(schema_version='b2_q2_fingerprint_h4_v1',status='PREFLIGHT',mandatory_stop=True,
        started_UTC=datetime.now(timezone.utc).isoformat(),native_builds_started=0,compile_calls_started=0,
        reference_rebuilds=0,reference_recompiles=0,stages={},wrappers={},axes={},
        expected_prices=None,price_MC=None,total_RPE_cost=None,m17_feasibility=None)
    try:
        round_,events,reference,compiler,private_files,private_hashes = prepare_fixed(private_dir,output,result)
        candidate = evaluate_candidate(result,round_,events,reference,compiler,output,result['source']['commit'])
        # Technical same-circuit hash controls only; no additional build/compile.
        result['hash_control_RSS_before'] = pipeline.rss()
        controls = {}
        for name,hasher in [('legacy',previous.canonical_qiskit_circuit_fingerprint),('stream',streamed_hash)]:
            fp,perf = pipeline.profile(lambda:hasher(candidate.circuit))
            require(fp==candidate.circuit_fingerprint, 'POST_PIPELINE_NATIVE_HASH_CONTROL')
            controls[name] = dict(fingerprint=fp,performance=perf)
        result['same_native_hash_controls'] = dict(protocol='after pipeline; same built native circuit; legacy then stream; warmed definitions',
            extra_builds=0,extra_compiles=0,observations=controls)
        for path,sha in result['source']['sha256'].items():
            require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha, 'CONSUMED_SOURCE_CHANGED')
        require({name:hashlib.sha256(p.read_bytes()).hexdigest() for name,p in private_files.items()}==private_hashes,
                'PRIVATE_INPUT_CHANGED')
        result['status'] = 'PASS_Q2_OPT_IN_FINGERPRINT_IDENTICAL'
    except Exception as exc:
        result.update(status='INCONCLUSIVE',stop_reason=type(exc).__name__+': '+str(exc))
    result.update(elapsed_seconds=time.perf_counter()-start,finished_UTC=datetime.now(timezone.utc).isoformat())
    dump(output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--private-dir',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=run(args.private_dir,args.output)
    print(result['status'],result.get('stop_reason',''),flush=True)
    sys.exit(0 if result['status']=='PASS_Q2_OPT_IN_FINGERPRINT_IDENTICAL' else 1)
