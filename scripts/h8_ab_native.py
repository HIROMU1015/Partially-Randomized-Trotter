"""Restore fixed H8 operations; extract exact tails without new Gaussian synthesis."""
from dataclasses import asdict
import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import h8_ab_contract as c


def prefix_preparation(ham, blocks, prefix, m):
    from trotterlib.df_partial_s2 import DFPartialS2Preparation, _partition_hash, _block_hash_payload
    from trotterlib.df_partial_randomized_pf import df_fragment_weight
    from trotterlib.df_rte_tail import extract_df_diagonal_tail, prepare_df_rte_event_inputs
    from trotterlib.pr2_matched_accuracy_m1_execution import generation_partition
    partition = generation_partition(ham, prefix)
    partition_hash = _partition_hash(cast_hash := m.df_hamiltonian_hash(ham), partition,
                                     partition_policy='explicit_ordered_partition')
    tail_blocks = blocks[prefix + 1:]
    c.require(tuple(b.original_fragment_index for b in tail_blocks) == tuple(range(prefix, len(ham.lambdas))),
              'SAVED_GENERATION_ORDER')
    proxy = math.fsum(df_fragment_weight(ham, i, weight_rule='lambda_frobenius_squared')
                     for i in range(prefix, len(ham.lambdas)))
    tail = extract_df_diagonal_tail(f'partial-s2-tail-{cast_hash[:16]}-ld-{prefix}',
        [SimpleNamespace(eta=m.np.asarray(b.diagonal_eta), lam=b.lam, U_ops=b.runtime_basis_operations)
         for b in tail_blocks], fragment_ids=[f'df-fragment-{i}' for i in range(prefix, len(ham.lambdas))],
        identity_policy='extract_identity_phase', coefficient_atol=0., ranking_proxy_lambda_r=proxy)
    selected = tuple(blocks[:prefix + 1])
    payload = dict(hamiltonian_hash=cast_hash, partition_hash=partition_hash,
        deterministic_blocks=[_block_hash_payload(b) for b in selected], tail_hash=tail.tail_hash,
        identity_policy='extract_identity_phase', coefficient_atol=0.,
        basis_construction_policy='fermionic_gaussian_jw', diagonal_sort='descending_abs',
        product_formula='2nd', preparation_hash_policy='df_partial_s2_preparation_v1')
    md = tail.extraction_metadata
    return DFPartialS2Preparation(hamiltonian_hash=cast_hash, partition_hash=partition_hash,
        preparation_hash=c.digest(payload), ld=prefix, num_system_qubits=ham.n_qubits,
        deterministic_blocks=selected, deterministic_block_order=tuple(b.block_id for b in selected),
        deterministic_fragment_indices=tuple(range(prefix)), randomized_block_indices=tuple(range(prefix, len(ham.lambdas))),
        constant_coefficient=float(ham.constant), tail_extraction=tail,
        rte_preparation=prepare_df_rte_event_inputs(tail), ranking_proxy_lambda_r=proxy,
        exact_rte_lambda_r=tail.rte_lambda_r, identity_policy='extract_identity_phase', coefficient_atol=0.,
        threshold_dropped_component_count=md.threshold_dropped_component_count,
        threshold_dropped_coefficient_l1=md.threshold_dropped_coefficient_l1,
        threshold_operator_error_bound=md.threshold_operator_error_bound,
        extracted_identity_coefficient=tail.deterministic_identity_coefficient,
        basis_construction_policy='fermionic_gaussian_jw', diagonal_sort='descending_abs')


def preparation_export(prep):
    # Match the independently saved H8 export schema exactly.
    from dataclasses import asdict
    tail = prep.rte_preparation.symbolic_tail
    return dict(components=[dict(component_id=x.component_id, probability=x.probability,
            coefficient_sign=x.coefficient_sign, basis_id=x.basis_id, basis_hash=x.basis_hash,
            is_identity=x.is_identity, diagonal_pauli_support=list(x.diagonal_pauli_support or ()))
            for x in tail.components],
        bases={d.basis_hash: json.loads(json.dumps([asdict(x) for x in d.operations]))
               for d in tail.basis_definitions},
        constant_hex=prep.constant_coefficient.hex(), extracted_hex=prep.extracted_identity_coefficient.hex(),
        lambda_r_hex=prep.exact_rte_lambda_r.hex(), preparation_hash=prep.preparation_hash)


def load(config, writer, prefixes):
    m = importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    loader = importlib.import_module('trottertracks.resource_applicability.h8_signal_preflight_port_v1')
    native = importlib.import_module('trottertracks.resource_applicability.ax2b_native_df_v5')
    from trotterlib.df_partial_s2 import DFDeterministicOneBodySpec, DFDeterministicFragmentSpec, df_hamiltonian_hash
    from trotterlib.df_rte_tail import describe_basis_change_operations
    from qiskit.circuit.library import RZGate, XXPlusYYGate
    import qiskit
    from trotterlib.rte_compiled_cost import compiler_settings_hash
    import h8_b0_anchor_adapter as restore
    import h8_strengthened_export as exported
    paths = config['mapping']
    ham, sector, full, state, md, _ = loader.load_h8_snapshot(Path(paths['h8_input_snapshot.npz']),
        c.read(paths['snapshot_receipt.json']), c.read(paths['df_receipt.json']))
    c.require((ham.n_qubits, len(ham.lambdas), sector.dimension) == (16, 28, 4900), 'H8_LAYOUT')
    c.require(md['hamiltonian_hash'] == config['binding']['hamiltonian']
              and md['state_hash'] == config['binding']['state'], 'H8_STATE_HAMILTONIAN')
    qindices, qstate = m.checked_basis_bridge(ham, sector, full, state, 1e-12)
    export = c.read(paths['h8_preparation_export'])
    exported.validate_export(export, expected_hashes=config['expected_preparation_hashes'])
    class RestorePermit:
        calls = 0
        def require(self, scope):
            from h8_b0_anchor_contract import SCOPE
            c.require(scope == SCOPE, 'RESTORE_UTILITY_SCOPE_ONLY')
        def take(self, name):
            c.require(name == 'saved_preparation_decode' and self.calls == 0, 'RESTORE_ONCE')
            self.calls += 1
    def inspect(ops):
        d = describe_basis_change_operations(ops)
        return d, json.loads(json.dumps([asdict(x) for x in d]))
    blocks = restore.restore(export, RestorePermit(), factory=lambda name, params:
        {'rz': RZGate, 'xx_plus_yy': XXPlusYYGate}[name](*params), inspect=inspect,
        one_type=DFDeterministicOneBodySpec, fragment_type=DFDeterministicFragmentSpec)
    c.require(all(float(ham.lambdas[i]).hex() == b.lam.hex() for i, b in enumerate(blocks[1:])), 'SAVED_BLOCK_LAMBDAS')
    # Pure reconstruction; the original builder/eigensolver is never called.
    bridge = SimpleNamespace(np=m.np, df_hamiltonian_hash=df_hamiltonian_hash)
    preps = {p: prefix_preparation(ham, blocks, p, bridge) for p in prefixes}
    for p, prep in preps.items():
        key = 'faithful:p' + str(p)
        if key in export:
            observed = preparation_export(prep)
            c.require(observed == {k: export[key][k] for k in observed}, 'EXACT_SAVED_PREFIX_COMPONENT_BASIS_BINDING')
    cp = config['compiler_options']
    compiler = m.CompilerSettings(tuple(cp['basis_gates']), None, None, cp['optimization_level'], None, None,
                                  cp['seed_transpiler'], qiskit.__version__)
    c.require(qiskit.__version__ == '1.3.0' and compiler_settings_hash(compiler) == c.COMPILER, 'LIVE_COMPILER_PROFILE')
    writer.write('preparation_binding.json', dict(saved_export_sha256=c.sha(paths['h8_preparation_export']),
        prefixes={str(p): dict(preparation_hash=v.preparation_hash, lambda_r=v.exact_rte_lambda_r,
            extracted_identity=v.extracted_identity_coefficient, constant=v.constant_coefficient,
            block_count=len(v.deterministic_blocks), saved_exact_export_match=('faithful:p' + str(p)) in export)
            for p, v in preps.items()}, new_Gaussian_operations=0, new_SCF_DF_state=0,
        compiler_sha256=c.COMPILER, profile_sha256=c.PROFILE))
    return m, native, ham, sector, state, qindices, qstate, preps, compiler
