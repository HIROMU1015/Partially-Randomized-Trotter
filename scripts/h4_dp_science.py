"""Future authorized H4 kernels. Importing does not load scientific providers.

The runner authorizes before importing this module. Provider loading is lazy.
"""
from fractions import Fraction
from contextlib import nullcontext
import importlib
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import h4_dp_contract as c


class Writer:
    def __init__(self, directory):
        self.directory = Path(directory)

    def write(self, name, value):
        c.require(Path(name).name == name, 'EXCLUSIVE_OUTPUT_NAME')
        c.save(self.directory / name, value)


class Budget:
    def __init__(self, directory, allowed):
        self.directory, self.allowed, self.used = Path(directory), allowed, {k: 0 for k in allowed}

    def take(self, name, count=1):
        c.require(name in self.allowed and self.used[name] + count <= self.allowed[name], 'SCIENTIFIC_CALL_BUDGET')
        self.used[name] += count
        c.save(self.directory / 'calls_ledger.json', dict(allowed=self.allowed, reserved=self.used,
               nonrefundable=True, returned_success_not_implied_by_reservation=True))


def compiler(m, native):
    import qiskit
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    cp = native['compiler_options']
    c.require(qiskit.__version__ == '1.3.0', 'QISKIT_VERSION')
    value = m.CompilerSettings(tuple(cp['basis_gates']), None, None, cp['optimization_level'],
                               None, None, cp['seed_transpiler'], qiskit.__version__)
    c.require(compiler_settings_hash(value) == native['binding']['compiler'], 'COMPILER_HASH')
    return value


def load_context(spec, directory):
    """Read saved inputs and reconstruct original blocks; no signal/reference solver."""
    native = c.read(spec['native_config'])
    if spec['stage'] == 'P':
        config = c.read(spec['A_native_adapter'])
        for path in reversed(config['import_paths']):
            sys.path.insert(0, path)
        frozen = importlib.import_module('pr_cost_native')
        context = frozen.load_context('H4', config, spec['cells'], directory)
        context.frozen = frozen
        from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
        context.fingerprint_reuse = use_fingerprint_reuse
        return context
    sys.path.insert(0, str(Path(native['provider_root']) / 'src'))
    m = importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    from trottertracks.resource_applicability.h4_common_panel_port_v1 import h4_loader
    ham, sector, full, state, md, _ = h4_loader(Path(native['mapping']['snapshot']),
        c.read(native['mapping']['snapshot_receipt']), c.read(native['mapping']['df_receipt']))
    c.require((ham.n_qubits, len(ham.lambdas), sector.dimension) == (8, 12, 36), 'H4_LAYOUT')
    c.require(md['hamiltonian_hash'] == native['binding']['hamiltonian']
              and md['state_hash'] == native['binding']['state'], 'H4_SAVED_INPUT_IDENTITY')
    _, qstate = m.checked_basis_bridge(ham, sector, full, state, 1e-12)
    from trottertracks.resource_applicability.h6_hash_reuse_v1 import use_fingerprint_reuse
    return SimpleNamespace(m=m, ham=ham, qstate=qstate, preparations={}, fingerprint_reuse=use_fingerprint_reuse,
                           compiler=compiler(m, native), binding=native['binding'])


def deterministic_control(context, evolution, expected, writer):
    m, state = context.m, context.qstate
    n = len(state).bit_length() - 1
    low = m.np.concatenate((state, m.np.zeros_like(state)))
    high = m.np.concatenate((m.np.zeros_like(state), state))
    a = m.simulate_statevector(evolution.circuit, low)
    b = m.simulate_statevector(evolution.circuit, high)
    z = complex(m.np.vdot(state, b[len(state):]))
    errors = [float(m.np.linalg.norm(a-low)), float(m.np.linalg.norm(b[:len(state)])),
              abs(float(m.np.linalg.norm(b))-1), abs(z-complex(expected['real'], expected['imag']))]
    for axis, value in [('cosine', z.real), ('sine', z.imag)]:
        wrapper = m.build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=False,
                                                  max_instructions=1000000)
        errors.append(abs(m.hadamard_expectation(m.simulate_statevector(wrapper, low), n)-value))
    c.require(all(math.isfinite(x) for x in errors) and max(errors) < 1e-9, 'NATIVE_CONTROL_SCALAR_SIGNAL')
    writer.write('control.json', dict(status='PASS', control_probes=4, maximum_error=max(errors),
                                    expected_signal_recomputed=False))


def _compile_axes(context, evolution, writer, budget, seed=None, event_digest=None):
    rows = {}
    for axis in c.AXES:
        m = context.m
        wrapper = m.build_native_hadamard_wrapper(evolution, axis=axis, include_measurement=True,
                                                  max_instructions=1000000)
        fp = m.canonical_qiskit_circuit_fingerprint(wrapper)
        budget.take('wrapper_compile')
        result = m.transpile_and_measure_cost(wrapper, context.compiler, actual_circuit_fingerprint=fp)
        c.require(result.compiler_settings_hash == context.binding['compiler']
                  and result.actual_circuit_fingerprint == fp
                  and m.canonical_qiskit_circuit_fingerprint(wrapper) == fp, 'COMPILE_IDENTITY')
        row = dict(axis=axis, fingerprint=fp, compiler_hash=result.compiler_settings_hash,
            RZ=result.rz_count, CX=result.cx_count, RZ_depth=result.rz_depth, CX_depth=result.cx_depth,
            depth=result.total_depth, size=result.circuit_size, seed=seed, event_digest=event_digest)
        c.require(all(type(row[k]) is int and row[k] >= 0 for k in
                      ('RZ', 'CX', 'RZ_depth', 'CX_depth', 'depth', 'size')) and row['RZ'] > 0,
                  'INTEGER_COMPILED_METRICS')
        writer.write('measured_' + axis + '.json', row)
        rows[axis] = row
    return rows


def compile_axes(context, evolution, writer, budget, seed=None, event_digest=None):
    with getattr(context, 'fingerprint_reuse', nullcontext)():
        return _compile_axes(context, evolution, writer, budget, seed, event_digest)


def validate_discard_preparation(prep, expected_truncated_hash, prefix, num_system_qubits,
                                 expected_constant):
    """The discard builder binds its truncated H; full input is checked at load.

    Expected hash comes from the saved full Hamiltonian's first-prefix view,
    before invoking the builder. It is never copied from the observed prep.
    """
    c.require(prep.hamiltonian_hash == expected_truncated_hash, 'DISCARD_HAMILTONIAN_BINDING')
    c.require(prep.num_system_qubits == num_system_qubits and prep.ld == prefix
              and prep.deterministic_fragment_indices == tuple(range(prefix))
              and prep.randomized_block_indices == (), 'DISCARD_PARTITION_BINDING')
    c.require(prep.constant_coefficient == expected_constant and prep.exact_rte_lambda_r == 0.,
              'DISCARD_SCALAR_OR_EMPTY_TAIL_BINDING')


def discard_preparation(context, prefix):
    if prefix not in context.preparations:
        from trotterlib.df_partial_randomized_pf import df_hamiltonian_hash
        from trotterlib.pr2_s0_s1_validation import WEIGHT_RULE
        truncated = context.ham.select_blocks(tuple(range(prefix)))
        expected = df_hamiltonian_hash(truncated, weight_rule=WEIGHT_RULE)
        prep = context.m._prepare_discard(context.ham, prefix)
        validate_discard_preparation(prep, expected, prefix, context.ham.n_qubits,
                                     context.ham.constant)
        context.preparations[prefix] = prep
    return context.preparations[prefix]


def deterministic_task(context, task, directory):
    cell, record = task['cell'], task['record']
    writer = Writer(directory)
    budget = Budget(directory, dict(native_evolution=1, control_probe=4, wrapper_compile=2))
    prefix = cell['prefix']
    prep = discard_preparation(context, prefix)
    scalar = record['scalar_receipt']['scalar'] if cell['method'] == 'B0I' else prep.constant_coefficient
    budget.take('native_evolution')
    evolution = context.m.build_deterministic_native(prep.deterministic_blocks, num_system_qubits=8,
        T=.8, q=cell['q'], formula=cell['formula'], scalar=scalar,
        control_policy='symmetric_directional', max_instructions=1000000)
    budget.take('control_probe', 4)
    deterministic_control(context, evolution, record['signals']['corrected'], writer)
    rows = compile_axes(context, evolution, writer, budget)
    return dict(cell_id=cell['id'], axes=rows, calls=budget.used, new_signal=0)


def confirmation_task(context, task, directory):
    writer = Writer(directory)
    cell = task['cell']
    budget = Budget(directory, dict(trajectory=1, occurrence=cell['q'], RTE_event=cell['R'],
        native_evolution=1, wrapper_compile=2, control_probe=4, independent_event_oracle=1))
    budget.take('trajectory'); budget.take('occurrence', cell['q']); budget.take('RTE_event', cell['R'])
    prep = context.preparations[cell['prefix']]
    requests, distribution = context.frozen.steps(prep, cell, task['seed'])
    hierarchy = c.seed_hierarchy(task['seed'], cell['q'])
    c.require([r.seed for r in requests] == (hierarchy if cell['q'] == 1 else hierarchy[1:]), 'FROZEN_SEED_LAW')
    events = context.frozen.event_values(requests)
    event_digest = c.digest(events)
    log_b = cell['R'] * math.log(distribution.exact_finite_distribution)
    c.require(math.isfinite(log_b) and abs(log_b-task['saved_log_B']) <= task['log_B_tolerance'],
              'ORIGINAL_FINITE_RTE_NORMALIZATION')
    c.require(prep.preparation_hash == task['preparation_hash'], 'PREPARATION_IDENTITY')
    writer.write('trajectory.json', dict(cell=cell, seed=task['seed'], master_seed=task['seed'],
        step_seeds=[r.seed for r in requests], events=events,
        event_digest=event_digest, original_RTE_distribution=distribution.to_dict(),
        cohort='INDEPENDENT_H4_CONFIRMATION', conditional=False, quantum_weights_changed=False))
    budget.take('native_evolution')
    evolution = context.native.partial_native_from_step_requests(requests,
        control_policy='symmetric_directional', max_instructions=1000000)
    if task['replica'] == 0:
        budget.take('control_probe', 4); budget.take('independent_event_oracle')
        context.frozen.control(evolution, requests, context, writer, 1e-9)
    rows = compile_axes(context, evolution, writer, budget, task['seed'], event_digest)
    c.require(c.digest(context.frozen.event_values(requests)) == event_digest, 'PAIRED_EVENT_MUTATION')
    pair = dict(cell_id=cell['id'], replica=task['replica'], seed=task['seed'], event_digest=event_digest,
                axes=rows, cohort='INDEPENDENT_H4_CONFIRMATION', calls=budget.used,
                stratum=c.event_stratum(events))
    c.check_pair(pair, context.binding['compiler'])
    return pair


def deterministic_account(records, endpoints, measurements, previous=None):
    """One approved pass over fixed saved shots; no PR reranking or shot refit."""
    cells = {r['cell']['id']: r['cell'] for r in records}
    rows, winners = [], []
    for ep in endpoints:
        cid, ac = ep['cell_id'], ep['saved_accounting']
        row = dict(cell_id=cid, bias_mode=ep['bias_mode'], epsilon=ep['epsilon'], factor=1.,
                   eligible=ac['cost_acquisition_eligible'], G_exact=None, shots=None)
        if row['eligible']:
            c.require(cid in measurements and set(measurements[cid]) == set(c.AXES), 'MISSING_ELIGIBLE_PF_COST')
            row['shots'] = dict(cosine=ac['axes']['real']['shots'], sine=ac['axes']['imag']['shots'])
            row['G_exact'] = str(sum(row['shots'][axis]*int(measurements[cid][axis]['RZ']) for axis in c.AXES))
        rows.append(row)
    for mode in c.FIELDS:
        for eps in c.EPS:
            choices = [r for r in rows if r['bias_mode'] == mode and r['epsilon'] == eps and r['eligible']]
            winner = min(choices, key=lambda r: (Fraction(r['G_exact']), r['cell_id'])) if choices else None
            ties = [r['cell_id'] for r in choices if winner is not None and r['G_exact'] == winner['G_exact']]
            old = next((r for r in (previous or []) if r['bias_mode'] == mode and r['epsilon'] == eps), None)
            winners.append(dict(bias_mode=mode, epsilon=eps, best=winner, exact_ties=sorted(ties),
                cell=None if winner is None else cells[winner['cell_id']],
                boundary=None if winner is None else c.boundary_attributes(cells[winner['cell_id']], list(cells.values())),
                previous_subset=old, saved_subset_winner_changed=None if old is None or winner is None else
                    old['deterministic']['cell_id'] != winner['cell_id'],
                domain='REGISTERED_D96_PLUS_FULL8', strict_bias_certificate=None))
    excluded = [cid for cid in cells if not any(r['eligible'] for r in rows if r['cell_id'] == cid)]
    return dict(rows=rows, winners=winners, domain=len(cells), zero_fill=False,
                full_direct_compilation=set(cells) <= set(measurements), excluded_ids=excluded,
                global_optimum_certified=False, population_mean_certified=False)
