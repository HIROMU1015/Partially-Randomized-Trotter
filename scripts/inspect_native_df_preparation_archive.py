"""Inspect existing frozen H6 metadata. No archive restoration or gate costs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_native_archive_descriptions',
    ROOT/'src/trotterlib/native_df_archive_metadata.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inspect_archive(path, expected_sha256):
    graph = module.FrozenMetadataGraph.from_file(path, expected_sha256=expected_sha256)
    summaries = {}
    for key, ref in graph.archive['codec']['roots'].items():
        f = graph.fields(ref, 'DFPartialS2Preparation')
        n = graph.scalar(f['num_system_qubits'])
        if n != 12:
            raise ValueError('This frozen schema was audited for H6 only')
        blocks = []
        for br in graph.sequence(f['deterministic_blocks']):
            node = graph.node(br)
            if node['type'] not in ('DFDeterministicOneBodySpec', 'DFDeterministicFragmentSpec'):
                raise ValueError('Unknown deterministic block')
            b = node['fields']
            operations = graph.basis_description(b['basis_change_operations'], n=n,
                basis_id=b['basis_id'], basis_hash=b['basis_hash'])
            graph.compare_runtime_descriptions(b['runtime_basis_operations'], operations)
            diagonal = b['diagonal_eigenvalues'] if node['type'] == 'DFDeterministicOneBodySpec' else b['diagonal_eta']
            if len(graph.sequence(diagonal)) != n:
                raise ValueError('Diagonal coefficient dimension mismatch')
            # Retain original hex scalar encodings; never produce RZ/RZZ lists.
            for value in graph.sequence(diagonal):
                graph.scalar(value)
            blocks.append(dict(block_id=b['block_id'], original_fragment_index=b['original_fragment_index'],
                order_index=b['order_index'], basis_id=b['basis_id'], basis_hash=b['basis_hash'],
                basis_description_node_sha256=b['basis_change_operations']['ref'],
                runtime_gate_list_node_sha256=b['runtime_basis_operations']['ref'],
                diagonal_coefficient_node_sha256=diagonal['ref'],
                runtime_description_exact=True, primitive_lists_already_saved=False))
        if [b['order_index'] for b in blocks] != list(range(len(blocks))):
            raise ValueError('Block order mismatch')
        rte = graph.fields(f['rte_preparation'], 'DFRTEEventPreparation')
        tail = graph.fields(rte['symbolic_tail'], 'SymbolicRTETail')
        catalog = []
        bases = {}
        for cr in graph.sequence(tail['components']):
            c = graph.fields(cr, 'RTEComponent')
            record = {k:graph.scalar(c[k]) for k in ('component_id','coefficient_abs',
                'coefficient_sign','probability','basis_id','basis_hash','df_fragment_id',
                'is_identity','diagonal_pauli_support')}
            if not record['is_identity']:
                operations = graph.basis_description(c['basis_change_operations'], n=n,
                    basis_id=record['basis_id'], basis_hash=record['basis_hash'])
                identity = (record['basis_id'], record['basis_hash'])
                if identity in bases and bases[identity] != operations:
                    raise ValueError('Conflicting component basis description')
                bases[identity] = operations
            catalog.append(record)
        if len({c['component_id'] for c in catalog}) != len(catalog):
            raise ValueError('Duplicate component ID')
        declared = graph.fields(tail['extraction_metadata'], 'DFTailExtractionMetadata')
        if len(catalog) != graph.scalar(declared['randomized_component_count']):
            raise ValueError('Component completeness declaration mismatch')
        registry = graph.node(rte['basis_registry'])
        if registry['type'] != 'DFBasisRegistry':
            raise ValueError('Missing full registry')
        defined = set()
        for definition in registry['definitions']:
            matches = [k for k in bases if k[0] == definition['basis_id']]
            if len(matches) != 1 or definition['num_system_qubits'] != n:
                raise ValueError('Registry/component identity mismatch')
            graph.compare_runtime_descriptions(definition['operations'], bases[matches[0]])
            defined.add(matches[0])
        if defined != set(bases):
            raise ValueError('Incomplete runtime basis registry')
        summaries[key] = dict(root_node_sha256=ref['ref'], preparation_hash=f['preparation_hash'],
            hamiltonian_hash=f['hamiltonian_hash'], partition_hash=f['partition_hash'],
            num_system_qubits=n, prefix=f['ld'],
            deterministic_fragment_indices=graph.scalar(f['deterministic_fragment_indices']),
            randomized_block_indices=graph.scalar(f['randomized_block_indices']),
            identity_policy=f['identity_policy'], coefficient_atol_hex=f['coefficient_atol']['float_hex'],
            constant_hex=f['constant_coefficient']['float_hex'],
            extracted_identity_hex=f['extracted_identity_coefficient']['float_hex'],
            exact_rte_lambda_r_hex=f['exact_rte_lambda_r']['float_hex'],
            blocks=blocks, tail_hash=tail['tail_hash'],
            complete_component_count=len(catalog), catalog_sha256=module.digest(catalog),
            component_probability_values_present=True, component_basis_registry_complete=True,
            component_law_recomputed=False, gate_matrices_recomputed=False,
            primitive_lists_generated=False, theoretical_cost_evaluated=False)
    return dict(status='EXISTING_ARCHIVE_DESCRIPTIONS_VERIFIED',archive_sha256=expected_sha256,
        codec_digest=graph.archive['codec_digest'],node_count=len(graph.nodes),roots=summaries,
        input_identity=graph.archive['binding']['input_identity']['hamiltonian_hash'],
        stored_profile_sha256=graph.archive['binding']['profile_sha256'],
        public_input_array_hashes_not_recomputed=True,scientific_imports=0,
        numerical_law_or_gate_matrix_certificate=False,runtime_authorization=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    args=parser.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    result=inspect_archive(args.archive,args.sha256)
    forbidden={'numpy','scipy','qiskit','qiskit_nature','pyscf','cupy','trottertracks'} & set(sys.modules)
    if forbidden:raise RuntimeError('Unexpected scientific import')
    print(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
