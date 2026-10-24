"""Fixed fixture recipe (stdlib) and separately guarded, lazy science loader.

The test helper itself is NEVER imported/called: it also builds RTE requests.
Unresolved numeric identities are None, never invented fixture hashes.
"""
import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from trottertracks.b2d_adapter.adapter import context_from_preparation
from .dense_backend import PreparedDenseInputs

BASE_COMMIT = "52ba88687f92d089921b207e60aec006f04141de"
FIXTURE_PATH = "tests/test_df_partial_s2_repeated.py"
FIXTURE_BLOB = "0c92ae9c1e8c5721d9c8629a83435b855f5d6043"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class InputRecipe:
    fixture_commit: str = BASE_COMMIT
    fixture_blob: str = FIXTURE_BLOB
    fixture_path: str = FIXTURE_PATH
    fixture_function: str = "_nontrivial_basis_case"
    n_qubits: int = 2
    ld: int = 2
    weight_rule: str = "lambda_frobenius_squared"
    n_electrons: int = 1
    occupied_modes: tuple[int, ...] = (0,)
    basis_order: str = "sorted_OpenFermion_JW_number_indices"
    identity_policy: str = "extract_identity_phase"
    coefficient_atol: int = 0

    def validate(self):
        if asdict(self) != asdict(InputRecipe()):
            raise ValueError("Only the proposed fixed recipe is connected; no automatic fallback.")

    @property
    def fingerprint(self):
        return fingerprint(asdict(self))


def read_fixture_definition(source_root, recipe=InputRecipe()):
    """Read fixed bytes/literals only; no import, ranking, H or matrix creation."""
    recipe.validate()
    data = (Path(source_root) / recipe.fixture_path).read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
    if blob != recipe.fixture_blob:
        raise ValueError("Fixture SOURCE blob changed.")
    function = next(n for n in ast.parse(data).body
                    if isinstance(n, ast.FunctionDef) and n.name == recipe.fixture_function)
    constructor = function.body[0].value
    if not isinstance(constructor, ast.Call) or not isinstance(constructor.func, ast.Name) or constructor.func.id != "DFHamiltonian":
        raise ValueError("Fixed fixture constructor missing.")
    def literal(node):
        if isinstance(node, ast.Call):
            if ast.unparse(node.func) != "np.asarray":
                raise ValueError("Unexpected fixture array constructor.")
            return ast.literal_eval(node.args[0])
        if isinstance(node, ast.Tuple):
            return tuple(literal(n) for n in node.elts)
        return ast.literal_eval(node)
    return {kw.arg: literal(kw.value) for kw in constructor.keywords}


def unresolved_identity(recipe=InputRecipe()):
    recipe.validate()
    return dict(recipe_fingerprint=recipe.fingerprint, hamiltonian_id=None,
                preparation_id=None, tail_id=None, ordered_d_block_ids=None,
                sector_id=None, sector_basis_indices=None, state_id=None,
                input_data_sha256=None, status="RESOLVE_ONLY_AFTER_SCIENCE_APPROVAL")


def connect_prepared(preparation, *, blocks, physical_hamiltonian, normalized_tail,
                     state, sector_basis_indices, recipe=InputRecipe()):
    """Metadata-only assembly; numerical checks stay in DenseBackend._validate.

    Array objects remain opaque here. This can be tested using plain objects.
    """
    recipe.validate()
    context = context_from_preparation(preparation)
    if preparation.num_system_qubits != recipe.n_qubits or preparation.ld != recipe.ld:
        raise ValueError("Preparation/recipe dimension or split differs.")
    if preparation.identity_policy != recipe.identity_policy or preparation.coefficient_atol != 0:
        raise ValueError("Wrong scalar/threshold policy.")
    if tuple(blocks) != context.block_ids:
        raise ValueError("D block ordering differs from preparation.")
    indices = tuple(int(i) for i in sector_basis_indices)
    # Static bit-pattern membership; no state or operator norm is evaluated.
    expected = tuple(i for i in range(1 << recipe.n_qubits)
                     if i.bit_count() == recipe.n_electrons)
    if indices != expected:
        raise ValueError("Sector basis must be the complete sorted number sector.")
    sector_id = fingerprint(dict(n_qubits=recipe.n_qubits, N=recipe.n_electrons,
                                 basis_indices=indices, ordering=recipe.basis_order))
    state_id = fingerprint(dict(sector_id=sector_id, occupied_modes=recipe.occupied_modes,
                                policy="physical_occupation_no_eigensolver"))
    return context, PreparedDenseInputs(
        context.preparation_id, context.tail_id, preparation.hamiltonian_hash,
        sector_id, state_id, dict(blocks), physical_hamiltonian, normalized_tail,
        preparation.rte_preparation.symbolic_tail, state)


def load_candidate(source_root, *, execution):
    """SCIENCE ONLY: must follow external approval, unused grant check and claim.

    Reuses existing basis construction (including internal Gaussian gates and
    tail basis circuits). This cost belongs to input scope, not to stage-2 PF
    resource compilation. It must be separately registered before execution.
    """
    from .preflight import require_claimed_run
    permit = require_claimed_run(execution)
    permit.begin_input(source_root, InputRecipe())
    definition = read_fixture_definition(source_root)
    import numpy as np
    from trotterlib.df_hamiltonian import DFHamiltonian, PhysicalSector
    from trotterlib.df_partial_randomized_pf import split_df_hamiltonian_by_ld
    from trotterlib.df_partial_s2 import prepare_df_partial_s2
    from trotterlib.finite_rte_signal_validation import (
        dense_df_operator_in_sector, _normalized_symbolic_tail_in_sector,
        _qiskit_to_openfermion_sector_permutation, _array_fingerprint,
    )
    h = DFHamiltonian(
        constant=definition["constant"],
        one_body=np.asarray(definition["one_body"], dtype=np.complex128),
        lambdas=np.asarray(definition["lambdas"], dtype=np.float64),
        g_matrices=tuple(np.asarray(g, dtype=np.complex128) for g in definition["g_matrices"]),
        metadata=definition["metadata"],
    )
    recipe = InputRecipe()
    split = split_df_hamiltonian_by_ld(h, recipe.ld, weight_rule=recipe.weight_rule)
    prep = prepare_df_partial_s2(h, split, identity_policy=recipe.identity_policy,
                                 coefficient_atol=0.0)
    sector = PhysicalSector.number_sector(n_qubits=h.n_qubits, n_electrons=1)
    physical = dense_df_operator_in_sector(h, sector)
    blocks = {}
    # select_blocks retains one-body and constant: isolate each operator explicitly.
    for block in prep.deterministic_blocks:
        if block.block_kind == "one_body":
            part = DFHamiltonian(0.0, h.one_body, np.empty(0), (), dict(h.metadata))
        elif block.block_kind == "df_fragment":
            j = block.original_fragment_index
            part = DFHamiltonian(0.0, np.zeros_like(h.one_body), h.lambdas[[j]],
                                 (h.g_matrices[j],), dict(h.metadata))
        else:
            raise ValueError("Unregistered deterministic block kind; STOP.")
        blocks[block.block_id] = dense_df_operator_in_sector(part, sector)
    permutation = _qiskit_to_openfermion_sector_permutation(sector)
    retained = _normalized_symbolic_tail_in_sector(prep, sector, permutation)
    if retained.leakage_frobenius_norm > permit.contract["numerical_atol"]:
        raise ValueError("Retained tail leaks outside registered sector; STOP.")
    # OpenFermion mode 0 is the most significant occupation bit.
    occupation = sum(1 << (h.n_qubits - 1 - m) for m in recipe.occupied_modes)
    state = np.zeros(sector.dimension, dtype=np.complex128)
    state[list(sector.basis_indices).index(occupation)] = 1
    context, inputs = connect_prepared(
        prep, blocks=blocks, physical_hamiltonian=physical,
        normalized_tail=retained.matrix, state=state,
        sector_basis_indices=sector.basis_indices)
    arrays = {**blocks, "physical": physical, "tail": retained.matrix, "state": state}
    digest = fingerprint({key: _array_fingerprint(value) for key, value in arrays.items()})
    provenance = dict(recipe=asdict(recipe), fixture_definition=definition,
        source_commit=permit.contract["source_commit"], hamiltonian_id=inputs.hamiltonian_id,
        preparation_id=inputs.preparation_id, tail_id=inputs.tail_id,
        ordered_d_block_ids=context.block_ids,
        deterministic_fragment_indices=prep.deterministic_fragment_indices,
        randomized_block_indices=prep.randomized_block_indices,
        scalar_terms=[asdict(t) for t in context.scalar_terms],
        exact_rte_lambda_r=prep.exact_rte_lambda_r,
        normalization="retained_symbolic_Pauli_L1_not_ranking_proxy",
        sector_id=inputs.sector_id, sector_basis_indices=tuple(map(int, sector.basis_indices)),
        state_id=inputs.state_id, occupied_modes=recipe.occupied_modes,
        input_data_sha256=digest, tail_leakage_norm=retained.leakage_frobenius_norm,
        array_shapes={key: value.shape for key, value in arrays.items()})
    permit.bind_input(context, inputs, provenance)
    return context, inputs, provenance
