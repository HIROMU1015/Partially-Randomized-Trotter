"""Future H8 cost adapter with identical compilation and fewer duplicate checks.

No current runner is rewired. A future source/input/profile/task grant must
explicitly select this port and its branch-reuse validation method.
"""
from copy import deepcopy
import gc
import json
from pathlib import Path
import resource
import time
import numpy as np

from .h6_matched_port_v1 import MatchedPort
from .h8_signal_preflight_port_v1 import load_h8_snapshot, private_path
from .h8_hash_stream_reuse_v1 import use_streaming_hash_reuse
from .h8_control_reuse_v1 import validate_control
from .h48_preflight_contract_v1 import task_caps
from .ax2b_h4_science_v5 import checked_basis_bridge
from .ax2b_molecular_ports_v3 import actual_bounds, hadamard_expectation
from .ax2b_native_df_v5 import build_deterministic_native, partial_native_from_step_requests
from .h6_matched_memory_v3 import compile_owned
from .ax2b_h6_controller import finite_scale_guard
from .ax2a_preparation import digest
from trotterlib.pr2_matched_accuracy_m1_execution import _prepare, _prepare_discard
from trotterlib.df_trotter.circuit import simulate_statevector
from trotterlib.rte import finite_rte_distribution, CompilerSettings


def cost_task(root, manifest, task, writer, progress, trace):
    """Frozen trajectory/compiler order with explicit new validation telemetry."""
    port = MatchedPort(root, manifest, writer, progress)
    c = task['cell']; identity = manifest['input_identity']; root = Path(root)
    port.ham, port.sector, full, state, _, _ = load_h8_snapshot(
        private_path(root, identity['snapshot_path']),
        json.loads(Path(identity['snapshot_receipt_path']).read_text()),
        json.loads(Path(identity['df_receipt_path']).read_text()))
    port.qindices, port.qstate = checked_basis_bridge(port.ham, port.sector, full, state, 1e-12)
    prep = (_prepare_discard if c['method'] == 'B0' else _prepare)(port.ham, c['prefix'])
    port.preps[c['id']] = prep
    bounds = actual_bounds([c], port.preps, T=port.plan['T'])
    if any(v > port.caps['untranspiled_instructions'] for v in bounds['cells'][0]['wrapper_instruction_upper_bounds'].values()):
        raise RuntimeError('COST_STRUCTURAL_INSTRUCTION_CAP')
    steps = port.sampled_steps(c, task['seed']) if c['R'] else None
    events = [[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None
    event_hash = digest(events)
    distribution = finite_rte_distribution(prep.exact_rte_lambda_r*port.plan['T']/c['R'], c['K']) if steps else None
    expected_event = next((x['partial_event_digest'] for x in manifest.get('reuse_registry', {}).get('missing', []) if x['task'] == task), None)
    if expected_event is not None and expected_event != event_hash:
        raise ValueError('PARTIAL_EVENT_REGENERATION_MISMATCH')
    writer.write('trajectory.json', dict(task=task, events=events, event_digest=event_hash,
        finite_distribution=distribution.to_dict() if distribution else None,
        observed_orders=[e.taylor_order for s in steps for e in s.rte_occurrence.events] if steps else [],
        quantum_shots_sampled=False, rare_events_fully_covered=False))
    import qiskit
    cp = port.plan['compiler_proposed']
    compiler = CompilerSettings(tuple(cp['basis_gates']), None, None,
        cp['optimization_level'], None, None, cp['seed_transpiler'], qiskit.__version__)
    previous_high = None; records = []; simulations = 0
    for policy in ('symmetric_directional', 'ordinary') if task['ordinary'] else ('symmetric_directional',):
        def make_evolution():
            return (partial_native_from_step_requests(steps, control_policy=policy,
                    max_instructions=port.caps['untranspiled_instructions']) if steps else
                build_deterministic_native(prep.deterministic_blocks, num_system_qubits=port.ham.n_qubits,
                    T=port.plan['T'], q=c['q'], formula=c['formula'], scalar=prep.constant_coefficient,
                    control_policy=policy, max_instructions=port.caps['untranspiled_instructions']))
        with trace.stage('evolution_build', control=policy):
            owner = [make_evolution()]
        if task['validate']:
            with trace.stage('control_validation', control=policy, instructions=len(owner[0].circuit.data)):
                evidence, b = validate_control(owner[0], port.qstate, calls=port.calls,
                    simulate=simulate_statevector, expectation=hadamard_expectation,
                    expected_signal=task['expected_signal'] if steps is None else None,
                    previous_high=previous_high)
                finite_scale_guard(log_B=0., intermediate_norm=1., absolute_discrepancy=evidence['max_error'])
                simulations += evidence['full_evolution_simulations']
                if previous_high is None:
                    previous_high = b
                writer.write('validation_'+policy+'.json', evidence)
        for axis in ('cosine', 'sine'):
            if digest([[e.to_dict() for e in s.rte_occurrence.events] for s in steps] if steps else None) != event_hash:
                raise ValueError('COST_PAIRED_EVENTS_CHANGED')
            started = time.monotonic()
            if not owner:
                with trace.stage('evolution_build', control=policy, axis=axis):
                    owner.append(make_evolution())
            def before_compile():
                port.calls.take('compile')
                writer.write('compile_attempt_%02d.json'%port.calls.used['compile'],
                    dict(task=task, axis=axis, control=policy,
                         calls_attempted_before_fingerprint=port.calls.used.copy(),
                         actual_transpile_started_not_implied=True))
            measured = compile_owned(owner, compiler, axis=axis,
                max_instructions=port.caps['untranspiled_instructions'],
                transpiled_cap=port.caps['transpiled_instructions'], trace=trace,
                before_compile=before_compile)
            fields = dict(RZ='rz_count', CX='cx_count', RZ_depth='rz_depth', CX_depth='cx_depth',
                          depth='total_depth', size='circuit_size')
            row = dict(cell_id=c['id'], phase=task['phase'], replica=task['replica'], seed=task['seed'],
                axis=axis, control=policy, event_digest=event_hash,
                metrics={k: getattr(measured, v) for k, v in fields.items()},
                fingerprint=measured.actual_circuit_fingerprint, compiler_hash=measured.compiler_settings_hash,
                classical_wall_seconds=time.monotonic()-started,
                worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                scope='measured Hadamard wrapper; state preparation excluded')
            writer.write('wrapper_'+policy+'_'+axis+'.json', row); records.append(row)
            del measured; gc.collect()
        owner.clear(); gc.collect()
    writer.write('cost_result.json', dict(task=task, rows=records, calls_attempted=port.calls.used,
        population_mean_certified=False, validation_algorithm='reused evolution branches and X/Y suffix',
        full_evolution_simulations=simulations, expensive_full_wrapper_simulations=0))


def cost_port(root, manifest, task, writer, progress, trace):
    local = deepcopy(manifest)
    local['plan']['caps_proposed'] = task_caps(task, manifest['plan']['caps_proposed'])
    with use_streaming_hash_reuse() as cache:
        cost_task(root, local, task, writer, progress, trace)
        writer.write('hash_reuse_receipt.json', cache.snapshot())
