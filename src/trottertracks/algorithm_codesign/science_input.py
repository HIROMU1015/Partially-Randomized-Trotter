"""Future authorized snapshot loading; NEVER called by preparation/tests."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import numpy as np

from .input_contract import INPUT
from .numerics import Spectrum, gamma
from .pilot import Task
from .shared_cpu import module


def load_authorized_task():
    # Caller must already validate a separately reviewed authorization and
    # consume the source-bound one-shot marker. This function has no CLI.
    path = Path(INPUT['source_worktree_literal'])/INPUT['relative_path_literal']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    if digest.hexdigest() != INPUT['documented_raw_sha256']:
        raise ValueError("STOP_INPUT_IDENTITY_MISMATCH")
    loader = module('pr2_new_series_validation')
    source = module('pr2_s0_s1_validation')
    m1 = module('pr2_matched_accuracy_m1_execution')
    tail_source = module('df_rte_tail')
    decomposition = module('df_trotter.decompose')
    hamiltonian, sector, saved_state, sector_state, metadata, layout = loader._load_snapshot_once(path)
    for key, expected in (("hamiltonian_hash", INPUT['documented_hamiltonian_hash']),
                          ("state_hash", INPUT['documented_state_hash']),
                          ("state_vector_hash", INPUT['documented_state_vector_hash'])):
        if metadata[key] != expected:
            raise ValueError("STOP_INPUT_LAYER_IDENTITY_MISMATCH")
    if hamiltonian.n_qubits != 8 or hamiltonian.n_blocks != 12:
        raise ValueError("Unexpected BF-1 model dimension/rank")
    state = source._to_qiskit_state(saved_state, 8)
    one_body, blocks, reconstruction = m1._dense_block_operators(hamiltonian)
    identities, coefficients = [], []
    eigen_error = 0.
    # Pure DF diagonalization and the existing symbolic coefficient routine;
    # no basis-change circuit or dense_extracted_df_tail helper is called.
    for index in range(3, 12):
        g = np.asarray(hamiltonian.g_matrices[index])
        vectors, eta = decomposition.diag_hermitian(g, sort='descending_abs', assume_hermitian=True)
        items = tail_source.exact_df_diagonal_coefficients(eta, float(hamiltonian.lambdas[index]))
        identities.extend(value for support, value in items if not support)
        coefficients.extend(value for support, value in items if support and abs(value) > 0.)
        eigensystem_error = Spectrum(g, eigensystem=(eta, vectors)).matrix_error
        eigen_error += eigensystem_error*max(1., np.linalg.norm(g, 'fro')+eigensystem_error)*abs(hamiltonian.lambdas[index])*256
    extracted_identity = math.fsum(identities)
    scalar = float(hamiltonian.constant)+extracted_identity
    dimension = len(state)
    residual = sum(blocks[3:], np.zeros((dimension, dimension), dtype=complex))-extracted_identity*np.eye(dimension)
    matrices = dict(D0=one_body, D1=blocks[0], D2=blocks[1], D3=blocks[2], R=residual)
    assembly_scale = abs(float(hamiltonian.constant))+float(np.sum(np.abs(hamiltonian.one_body)))
    assembly_scale += math.fsum(abs(float(lam))*float(np.sum(np.abs(g)))**2
                               for lam, g in zip(hamiltonian.lambdas, hamiltonian.g_matrices))
    # Uniform forward budget on EACH assembled generator and scalar. The
    # operation allowance covers selected-minus-base, Hermitian symmetrizing,
    # tail summation and identity subtraction; it is not inferred solely
    # from the full-H reconstruction discrepancy. See the v3 derivation.
    assembly_bound = reconstruction['absolute_operator_norm_error']+gamma(65536)*math.sqrt(dimension)*max(1., assembly_scale)+eigen_error
    generator_bounds = dict.fromkeys(matrices, assembly_bound)
    return Task(matrices, scalar, math.fsum(abs(c) for c in coefficients), state, assembly_bound,
                generator_bounds, assembly_bound), dict(
        input=INPUT, input_raw_sha256=digest.hexdigest(), metadata_identities={k: metadata[k] for k in
          ('hamiltonian_hash', 'state_hash', 'state_vector_hash')}, layout=layout,
        native_generator_count=4, extracted_identity=extracted_identity,
        lambda_r=math.fsum(abs(c) for c in coefficients), scalar=scalar,
        matrix_assembly_bound=assembly_bound,
        assembly_budget_policy='uniform_per_generator_and_scalar_v1',
        generator_assembly_bounds=generator_bounds, scalar_assembly_bound=assembly_bound,
        full_target_generator_budget=math.fsum(generator_bounds.values())+assembly_bound,
        partition_indices=[0, 1, 2],
        source_reconstruction=reconstruction, new_state_solve=False,
        circuits_built=0, trajectories=0, gpu_queries=0)
