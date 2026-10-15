"""Grant-only adapter over the pinned H6 port; no import-time science.

The coordinator imports this file only in admitted children. Preparations are
decoded from the old verified archive, not regenerated. No old grant is used.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
import resource
import time

from h6_fixed_delta_plan_v1 import cells, rotate_record


def modules():
    from trottertracks.resource_applicability import h6_matched_port_v1 as m
    from trottertracks.resource_applicability import h6_frozen_preparation_v1 as codec
    from trottertracks.resource_applicability.h6_roundoff_preparation_v1 import read_preparations
    return m, codec, read_preparations


def load_data(config):
    m, codec, read = modules()
    mapping = config['input_mapping']
    ham, sector, full, state, metadata, _ = m.load_h6_snapshot(
        Path(mapping['h6_input_snapshot.npz']), json.loads(Path(mapping['snapshot_receipt.json']).read_bytes()),
        json.loads(Path(mapping['df_receipt.json']).read_bytes()))
    if (ham.n_qubits, len(ham.lambdas), sector.dimension) != (12, 19, 400):
        raise ValueError('FIXED_H6_LAYOUT')
    preparations, archive = read(config['archive']['path'], config['archive']['sha256'])
    if archive['binding'] != config['archive']['binding']:
        raise ValueError('ARCHIVE_BINDING_CHANGED')
    return ham, sector, full, state, metadata, preparations


def plan(T, q):
    m, _, _ = modules()
    from trottertracks.resource_applicability.h6_matched_contract_v1 import plan as original
    from trottertracks.resource_applicability.ax2b_h6_contract import primitive_time_schedule
    p = original()
    p.update(T=T, cells=[c for c in cells(q) if c['method'] != 'B0I'], epsilons=[.05,.01,.005,.001,.0005])
    p['matrix_free']['num_threads'] = 1
    p['coverage']['cells'] = [dict(cell_id=c['id'], schedule=primitive_time_schedule(c,T=T)) for c in p['cells']]
    unique = {(('one' if i == 0 else str(i-1)),t) for c in p['cells']
        for i,t in primitive_time_schedule(c,T=T)['unique_primitive_times']}
    p['coverage']['primitive_actions'] = 3*len(unique)
    return p


def signal(config, task, writer, progress):
    m, codec, _ = modules()
    p = plan(task['T'], task['q'])
    port = m.MatchedPort(Path(config['provider_root']), dict(kind='H6_TECHNICAL',plan=p), writer, progress)
    port.phase('saved_input_and_shared_preparation')
    port.ham, port.sector, full, state, metadata, preparations = load_data(config)
    port.qindices, port.qstate = m.checked_basis_bridge(port.ham,port.sector,full,state,1e-12)
    port.state = state.copy(); port.saved_state = state.copy()
    port.preps = {c['id']:preparations[codec.key(c['method']=='B0',c['prefix'])] for c in p['cells']}
    # These caches are shared across all four signals at this T.
    port.spectra = {}; port.native_actions = {}; port.tails = {}
    port.oracle_tails = {}; port.truncated_references = {}
    bounds = m.actual_bounds(port.cells,port.preps,T=p['T']); m.check_bounds(bounds,port.caps)
    operator, _ = m.df_linear_operator(port.ham,port.sector,**p['matrix_free'])
    matrix, count = m.bounded_sector_matrix(lambda v:operator@v,port.sector.dimension,port.calls,
        per_action_cap=port.caps['reference_matvec_per_action'])
    error = max(float(m.np.linalg.norm(matrix[:,i]-m.occupation_column(
        port.ham,port.sector.basis_indices,int(index)))) for i,index in enumerate(port.sector.basis_indices))
    m.finite_scale_guard(log_B=0., intermediate_norm=float(m.np.linalg.norm(matrix)),absolute_discrepancy=error)
    port.reference, pair = m.reference_pair(matrix,state,p['T'])
    port.reference_error = error + pair['signal_difference']
    writer.write('reference.json',dict(T=p['T'],reference=m.complex_record(port.reference),
        independent_column_error=error,pair=pair,matvecs=count,metadata=metadata,regenerated=False))
    del matrix, operator
    port.primitives()
    port.correctness()
    saved = json.loads((writer.directory/'signal_summary.json').read_bytes())['signals']
    b0 = next(r for r in saved if r['cell']['method']=='B0')
    b0i = rotate_record(b0,p['T'],config['tail_scalar']['C'],config['tail_scalar']['C_error'])
    b0i['precision_rows'] = m.precision_rows(b0i,p)
    writer.write(b0i['cell']['id']+'_signal.json',b0i)
    writer.write('panel_signals.json',dict(T=p['T'],signals=saved+[b0i],calls=port.calls.used,
        empirical=True,certified=False,regenerated=False))


def cost(config, task, writer, progress, trace):
    m, codec, _ = modules()
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    c = task['cell']; raw = deepcopy(c)
    if c['method']=='B0I':
        raw.update(method='B0',id=c['id'].replace('B0I','B0'))
    p = plan(task['T'],c['q'])
    port = m.MatchedPort(Path(config['provider_root']),dict(kind='H6_TECHNICAL',plan=p),writer,progress)
    port.ham,port.sector,full,state,_,preparations = load_data(config)
    port.qindices,port.qstate = m.checked_basis_bridge(port.ham,port.sector,full,state,1e-12)
    prepared = preparations[codec.key(raw['method']=='B0',raw['prefix'])]
    port.preps[raw['id']] = prepared
    steps = port.sampled_steps(raw,task['seed']) if raw['R'] else None
    events = [[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None
    event_hash = m.digest(events)
    writer.write('trajectory.json',dict(task=task,events=events,event_digest=event_hash,
        quantum_shots_sampled=False,rare_events_fully_covered=False))
    import qiskit
    cp = p['compiler_proposed']
    compiler = m.CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],
        None,None,cp['seed_transpiler'],qiskit.__version__)
    scalar = prepared.constant_coefficient + (config['tail_scalar']['C'] if c['method']=='B0I' else 0.)
    with use_fingerprint_reuse() as hashes:
        with trace.stage('evolution_build'):
            evolution = (m.partial_native_from_step_requests(steps,control_policy='symmetric_directional',
                max_instructions=port.caps['untranspiled_instructions']) if steps else
                m.build_deterministic_native(prepared.deterministic_blocks,num_system_qubits=12,
                    T=task['T'],q=c['q'],formula=c['formula'],scalar=scalar,
                    control_policy='symmetric_directional',max_instructions=port.caps['untranspiled_instructions']))
        low = m.np.concatenate((port.qstate,m.np.zeros_like(port.qstate)))
        if task['validate']:
            with trace.stage('control_validation'):
                high = m.np.concatenate((m.np.zeros_like(port.qstate),port.qstate))
                port.calls.take('control_probe',2)
                a,b = m.simulate_statevector(evolution.circuit,low),m.simulate_statevector(evolution.circuit,high)
                z = complex(m.np.vdot(port.qstate,b[len(port.qstate):]))
                errors = [float(m.np.linalg.norm(a-low)),float(m.np.linalg.norm(b[:len(port.qstate)]))]
                if not steps:
                    expected = task['expected_signal']
                    errors.append(abs(z-complex(expected['real'],expected['imag'])))
                for axis,expected in (('cosine',z.real),('sine',z.imag)):
                    probe = m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=False,
                        max_instructions=port.caps['untranspiled_instructions'])
                    port.calls.take('control_probe')
                    errors.append(abs(m.hadamard_expectation(m.simulate_statevector(probe,low),12)-expected))
                    del probe
                m.finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=max(errors))
                writer.write('control_validation.json',dict(max_error=max(errors),certified=False))
        rows = []
        for axis in ('cosine','sine'):
            if m.digest([[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None)!=event_hash:
                raise ValueError('PAIRED_EVENTS_CHANGED')
            start = time.monotonic()
            # One evolution is reused by both axes; only one copied wrapper and
            # one compiled result are live. Fingerprint reuse remains content checked.
            wrapper = m.build_native_hadamard_wrapper(evolution,axis=axis,include_measurement=True,
                max_instructions=port.caps['untranspiled_instructions'])
            port.calls.take('compile')
            writer.write('compile_attempt_'+axis+'.json',dict(task=task,axis=axis,reserved=True))
            with trace.stage('fingerprint_transpile_metrics',axis=axis,instructions=len(wrapper.data)):
                cost_row = m.transpile_and_measure_cost(wrapper,compiler,
                    actual_circuit_fingerprint=m.canonical_qiskit_circuit_fingerprint(wrapper))
            if len(cost_row.transpiled_circuit.data)>port.caps['transpiled_instructions']:
                raise ValueError('TRANSPILED_INSTRUCTION_CAP')
            metrics = {k:getattr(cost_row,v) for k,v in dict(RZ='rz_count',CX='cx_count',
                RZ_depth='rz_depth',CX_depth='cx_depth',depth='total_depth',size='circuit_size').items()}
            row = dict(task=task,axis=axis,metrics=metrics,event_digest=event_hash,
                fingerprint=cost_row.actual_circuit_fingerprint,compiler_hash=cost_row.compiler_settings_hash,
                wall_seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            if row['compiler_hash']!=config['compiler_hash']:
                raise ValueError('COMPILER_CHANGED')
            writer.write('wrapper_'+axis+'.json',row); rows.append(row)
            del wrapper,cost_row;gc.collect()
        writer.write('hash_reuse.json',hashes.snapshot())
    writer.write('cost_result.json',dict(task=task,rows=rows,calls_attempted=port.calls.used,
        population_mean_certified=False))
