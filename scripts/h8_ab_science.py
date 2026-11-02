"""H8 conditional pilot: four independently controlled groups, paired native axes."""
import gc
import math
from pathlib import Path
import statistics
import h8_ab_contract as c
import h8_ab_native as native
import h8_ab_sampling as sampling
import h8_key_compile_parallel as pool
from h6_dba_science import checked_results, event_oracle


class Budget:
    def __init__(self, writer, limits):
        self.writer, self.limits = writer, limits
        self.used = {k: 0 for k in limits}

    def take(self, name, count=1):
        c.require(type(count) is int and count > 0 and name in self.limits, 'REGISTERED_CALL')
        c.require(self.used[name] + count <= self.limits[name], 'CALL_BUDGET:' + name)
        self.used[name] += count
        self.writer.write(f'reservation_{name}_{self.used[name]}.json',
                          dict(name=name, count=count, cumulative=self.used[name], nonrefundable=True))


def tasks(config):
    occupied = set(config['old_seed_values'])
    out = []
    for group in c.plan()['B']['groups']:
        for replica in range(8):
            value = c.seed(group['cell']['id'], group['stratum'], replica, config['binding']['input'])
            c.require(value not in occupied, 'EXISTING_MASTER_SEED_COLLISION')
            occupied.add(value)
            derived = [int(c.digest(['H8_AB_ORDER', value])[:16], 16)]
            derived += [int(c.digest(['H8_AB_COMPONENT', value, i])[:16], 16) for i in range(group['cell']['q'])]
            c.require(len(derived) == len(set(derived)) and all(s not in occupied for s in derived),
                      'DERIVED_SEED_COLLISION')
            occupied.update(derived)
            out.append(dict(id=group['id'] + '_' + str(replica), group=group, replica=replica, seed=value,
                            derived_seeds=derived))
    return out


def rare(config, grant, output):
    writer = c.Writer(output)
    m, n, ham, sector, state, qindices, qstate, preps, compiler = native.load(config, writer, (7,))
    prep = preps[7]
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    from trottertracks.resource_applicability.h8_control_reuse_v1 import validate_control
    tasks_list = tasks(config)
    writer.write('seed_registry.json', dict(tasks=tasks_list, old_registered_seed_count=len(config['old_seed_values']),
        collision_audit='saved exploration/confirmation, original trajectories and prior conditional pilot seeds',
        candidate_pairing_assumed=False, cost_sampling_not_quantum_shots=True))
    # Warm the tiny conditional tables before fork; all workers inherit them.
    from trotterlib.rte import finite_rte_distribution
    for group in c.plan()['B']['groups']:
        distribution = finite_rte_distribution(prep.exact_rte_lambda_r * .8 / group['cell']['R'], 4)
        probs = dict(zip(distribution.orders, distribution.order_probabilities, strict=True))
        sampling.cached_law(group['cell']['R'], tuple(probs.get(k, 0.) for k in (0, 2, 4)), group['stratum'])

    def callback(task, directory):
        local = c.Writer(directory); cell = task['group']['cell']
        budget = Budget(local, dict(trajectory=1, occurrence=cell['q'], RTE_event=cell['R'],
            native_evolution=1, compile=2, control_probe=4 if task['replica'] == 0 else 0,
            independent_event_oracle=1 if task['replica'] == 0 else 0))
        budget.take('trajectory'); budget.take('occurrence', cell['q']); budget.take('RTE_event', cell['R'])
        steps, receipt = sampling.steps(prep, cell, task['group']['stratum'], task['seed'])
        c.require([receipt['order_seed'], *receipt['component_seeds']] == task['derived_seeds'], 'REGISTERED_RNG_STREAMS')
        local.write('trajectory.json', receipt)
        budget.take('native_evolution')
        with use_fingerprint_reuse() as reuse:
            evolution = n.partial_native_from_step_requests(steps, control_policy='symmetric_directional',
                                                           max_instructions=4_000_000)
            if task['replica'] == 0:
                evidence, high = validate_control(evolution, qstate, calls=budget,
                    simulate=m.simulate_statevector, expectation=m.hadamard_expectation)
                norm_error = abs(float(m.np.linalg.norm(high)) - 1.)
                del high
                budget.take('independent_event_oracle')
                independent = event_oracle(steps, m, n, qstate)
                z = complex(evidence['signal']['real'], evidence['signal']['imag'])
                error = max(evidence['max_error'], abs(z - independent), norm_error)
                m.finite_scale_guard(log_B=0., intermediate_norm=1., absolute_discrepancy=error)
                c.require(math.isfinite(error) and error < 1e-8, 'INDEPENDENT_CONTROL_EVENT_MISMATCH')
                local.write('control.json', dict(evidence, status='PASS', independent_event_error=abs(z - independent),
                    high_branch_norm_error=norm_error, max_combined_error=error,
                    independent_of_event_circuit_lowering=True, independent_of_orbital_decomposition=False))
            rows = []
            # Same compiler and whole wrapper; a larger structural scope than H6.
            import time
            for axis in ('cosine', 'sine'):
                budget.take('compile')
                wrapper = n.build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=True,
                                                         max_instructions=4_000_000)
                fp = m.canonical_qiskit_circuit_fingerprint(wrapper)
                local.write('compile_' + axis + '_entered.json', dict(axis=axis, fingerprint=fp))
                start = time.monotonic()
                result = m.transpile_and_measure_cost(wrapper, compiler, actual_circuit_fingerprint=fp)
                c.require(result.compiler_settings_hash == c.COMPILER and result.actual_circuit_fingerprint == fp
                    and m.canonical_qiskit_circuit_fingerprint(wrapper) == fp, 'COMPILER_WRAPPER_IDENTITY')
                row = dict(axis=axis, RZ=result.rz_count, CX=result.cx_count, fingerprint=fp,
                    compiler_hash=result.compiler_settings_hash, compile_seconds=time.monotonic() - start)
                c.require(type(row['RZ']) is int and row['RZ'] > 0, 'POSITIVE_INTEGER_RZ')
                local.write('measured_' + axis + '.json', row); rows.append(row)
                del result, wrapper; gc.collect()
            local.write('fingerprint_reuse.json', reuse.snapshot())
        c.require(c.digest([e.to_dict() for step in steps for e in step.rte_occurrence.events])
                  == receipt['event_digest'], 'PAIRED_EVENT_SEQUENCE_CHANGED')
        c.require(budget.used == budget.limits, 'EXACT_TRAJECTORY_CALL_COMPLETENESS')
        return dict(task_id=task['id'], group=task['group']['id'], replica=task['replica'], seed=task['seed'],
            event_digest=receipt['event_digest'], n2=receipt['n2'], n4=receipt['n4'], rows=rows, actual_calls=budget.used)

    def batch(selected, name):
        return pool.map_owned(selected, callback, cpus=grant['worker_cpus'],
            limits=dict(max_workers=len(grant['worker_cpus'])), directory=output / name, deadline=math.inf,
            font_seed=config['font_seed'], reserve=lambda task: writer.write(
                'reservation_' + task['id'] + '.json', dict(task=task, nonrefundable=True)))

    first = batch([x for x in tasks_list if x['replica'] == 0], 'first_control_workers')
    rows = checked_results(first)
    writer.write('first_control_gate.json', dict(status='PASS', groups=4, native_reaped=True,
                                               control_probes=16, independent_event_oracles=4))
    remaining = batch([x for x in tasks_list if x['replica'] != 0], 'remaining_workers')
    rows += checked_results(remaining)
    summaries = []
    for group in c.plan()['B']['groups']:
        selected = [x for x in rows if x['group'] == group['id']]
        c.require(len(selected) == 8 and {x['replica'] for x in selected} == set(range(8)), 'REGISTERED_N8_COMPLETE')
        for axis in ('cosine', 'sine'):
            values = [next(a['RZ'] for a in x['rows'] if a['axis'] == axis) for x in selected]
            summaries.append(dict(group=group['id'], axis=axis, n=8, mean=statistics.mean(values),
                sample_SE=statistics.stdev(values) / math.sqrt(8), min=min(values), max=max(values),
                population_mean_certified=False))
    writer.write('worker_receipts.json', dict(first=first, remaining=remaining))
    calls = {k: sum(row['actual_calls'].get(k, 0) for row in rows) for k in rows[0]['actual_calls']}
    c.require(all(calls[k] == c.plan()['B'][k] for k in calls), 'EXACT_STAGE_CALL_COMPLETENESS')
    return dict(stage='B', status='COMPLETE_MANDATORY_STOP', groups=c.plan()['B']['groups'],
        trajectories=rows, summaries=summaries, actual_calls=calls, original_n32_mixed=False,
        unknown_cost_zero_filled=False, cost_reaccounting=False, population_mean_certified=False,
        new_signal=False, new_fit=False, statistical_estimand='conditional stratum cost at fixed H8 candidate')
