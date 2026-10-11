"""Cost-only supplementation port; never calls setup/correctness/signal paths.

The frozen cost-task numerical, trajectory, control-validation and row logic is
retained. Native construction runs again for the second axis using the SAME
sampled steps; it does not resample. Each compiled input has one owning circuit.
"""
from .h6_matched_port_v1 import (MatchedPort, load_h6_snapshot, safe_path,
    checked_basis_bridge, _prepare_discard, _prepare, actual_bounds, digest,
    finite_rte_distribution, CompilerSettings, partial_native_from_step_requests,
    build_deterministic_native, simulate_statevector, build_native_hadamard_wrapper,
    finite_scale_guard, hadamard_expectation, complex_record, json, np, time,
    resource, gc)
from .h6_matched_memory_v3 import compile_owned


def cost_task(root, manifest, task, writer, progress, trace):
    """Frozen numerical/sampling logic, single-owner wrapper lifetime."""
    port = MatchedPort(root,manifest,writer,progress)
    c = task['cell']; identity = manifest['input_identity']
    port.ham,port.sector,full,state,_,_ = load_h6_snapshot(safe_path(root,identity['snapshot_path']),
        json.loads((root/identity['snapshot_receipt_path']).read_text()),
        json.loads((root/identity['df_receipt_path']).read_text()))
    port.qindices,port.qstate = checked_basis_bridge(port.ham,port.sector,full,state,1e-12)
    p = (_prepare_discard if c['method']=='B0' else _prepare)(port.ham,c['prefix'])
    port.preps[c['id']] = p
    bounds = actual_bounds([c],port.preps,T=port.plan['T'])
    if any(v>port.caps['untranspiled_instructions'] for v in bounds['cells'][0]['wrapper_instruction_upper_bounds'].values()):
        raise RuntimeError('COST_STRUCTURAL_INSTRUCTION_CAP')
    steps = port.sampled_steps(c,task['seed']) if c['R'] else None
    events = [[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None
    event_hash = digest(events)
    distribution = finite_rte_distribution(p.exact_rte_lambda_r*port.plan['T']/c['R'],c['K']) if steps else None
    expected = next((x['partial_event_digest'] for x in manifest['reuse_registry']['missing'] if x['task']==task), None)
    if expected is not None and expected != event_hash:
        raise ValueError('PARTIAL_EVENT_REGENERATION_MISMATCH')
    writer.write('trajectory.json',dict(task=task,events=events,event_digest=event_hash,
        finite_distribution=distribution.to_dict() if distribution else None,
        observed_orders=[e.taylor_order for s in steps for e in s.rte_occurrence.events] if steps else [],
        quantum_shots_sampled=False,rare_events_fully_covered=False))
    import qiskit
    cp = port.plan['compiler_proposed']
    compiler = CompilerSettings(tuple(cp['basis_gates']),None,None,cp['optimization_level'],None,None,cp['seed_transpiler'],qiskit.__version__)
    branch_high = None;records=[]
    low = np.concatenate((port.qstate,np.zeros_like(port.qstate)))
    high = np.concatenate((np.zeros_like(port.qstate),port.qstate))
    for policy in ('symmetric_directional','ordinary') if task['ordinary'] else ('symmetric_directional',):
        def make_evolution():
            return (partial_native_from_step_requests(steps,control_policy=policy,max_instructions=port.caps['untranspiled_instructions'])
            if steps else build_deterministic_native(p.deterministic_blocks,num_system_qubits=port.ham.n_qubits,
                T=port.plan['T'],q=c['q'],formula=c['formula'],scalar=p.constant_coefficient,
                control_policy=policy,max_instructions=port.caps['untranspiled_instructions']))
        with trace.stage('evolution_build',control=policy):
            owner = [make_evolution()]
        z = None;checks=[]
        with trace.stage('control_validation',control=policy,instructions=len(owner[0].circuit.data)):
            if task['validate']:
                port.calls.take('control_probe',2)
                a,b = simulate_statevector(owner[0].circuit,low),simulate_statevector(owner[0].circuit,high)
                checks += [float(np.linalg.norm(a-low)),float(np.linalg.norm(b[:len(port.qstate)]))]
                z = complex(np.vdot(port.qstate,b[len(port.qstate):]))
                if branch_high is None:
                    branch_high = b
                else:
                    checks.append(float(np.linalg.norm(b-branch_high)))
                if not steps:
                    expected = task['expected_signal']
                    checks.append(abs(z-complex(expected['real'],expected['imag'])))
                for axis,expected in (('cosine',z.real),('sine',z.imag)):
                    probe = build_native_hadamard_wrapper(owner[0],axis=axis,include_measurement=False,
                                                         max_instructions=port.caps['untranspiled_instructions'])
                    port.calls.take('control_probe')
                    checks.append(abs(hadamard_expectation(simulate_statevector(probe,low),port.ham.n_qubits)-expected))
                    del probe
                finite_scale_guard(log_B=0.,intermediate_norm=1.,absolute_discrepancy=max(checks))
                writer.write('validation_'+policy+'.json',dict(max_error=max(checks),signal=complex_record(z),
                    probe='saved state branches and X/Y',random_signal_is_ensemble=False,certified=False))
        for axis in ('cosine','sine'):
            if digest([[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None)!=event_hash:
                raise ValueError('COST_PAIRED_EVENTS_CHANGED')
            start = time.monotonic()
            if not owner:
                with trace.stage('evolution_build',control=policy,axis=axis):
                    owner.append(make_evolution())
            def before_compile():
                port.calls.take('compile')
                writer.write('compile_attempt_%02d.json'%port.calls.used['compile'],
                    dict(task=task,axis=axis,control=policy,calls_attempted_before_fingerprint=port.calls.used.copy(),
                         actual_transpile_started_not_implied=True))
            cost = compile_owned(owner,compiler,axis=axis,
                max_instructions=port.caps['untranspiled_instructions'],
                transpiled_cap=port.caps['transpiled_instructions'],trace=trace,
                before_compile=before_compile)
            fields = dict(RZ='rz_count',CX='cx_count',RZ_depth='rz_depth',CX_depth='cx_depth',
                          depth='total_depth',size='circuit_size')
            row = dict(cell_id=c['id'],phase=task['phase'],replica=task['replica'],seed=task['seed'],
                axis=axis,control=policy,event_digest=event_hash,
                metrics={k:getattr(cost,v) for k,v in fields.items()},
                fingerprint=cost.actual_circuit_fingerprint,compiler_hash=cost.compiler_settings_hash,
                classical_wall_seconds=time.monotonic()-start,
                worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                scope='measured Hadamard wrapper; state preparation excluded')
            writer.write('wrapper_'+policy+'_'+axis+'.json',row);records.append(row)
            del cost;gc.collect()
        owner.clear();gc.collect()
    writer.write('cost_result.json',dict(task=task,rows=records,calls_attempted=port.calls.used,
                                       population_mean_certified=False))
