"""Synthetic comparisons against the source-pinned H6 pricing primitives."""

from pathlib import Path
from types import SimpleNamespace
import math
import sys

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import XXPlusYYGate


@pytest.fixture(scope="module")
def original():
    sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent / "payload-bound")]
    import run_full_m7

    return run_full_m7.check()[1:3]


def preparation():
    from trotterlib.df_partial_s2 import DFDeterministicFragmentSpec, DFDeterministicOneBodySpec

    common = dict(basis_id="test", basis_hash="test", basis_change_operations=(), num_system_qubits=2,
                  runtime_basis_operations=((XXPlusYYGate(0.19, -0.23), (0, 1)),))
    one = DFDeterministicOneBodySpec(
        block_id="one", block_kind="one_body", original_fragment_index=None, rank=None,
        diagonal_eigenvalues=(0.5, -0.22), order_index=0, **common,
    )
    df = DFDeterministicFragmentSpec(
        block_id="df", block_kind="df_fragment", original_fragment_index=0, rank=2,
        diagonal_eta=(0.13, -0.79), lam=0.51, order_index=1, **common,
    )
    return SimpleNamespace(num_system_qubits=2, deterministic_blocks=(one, df),
                           preparation_hash="synthetic", rte_preparation=SimpleNamespace(basis_registry=None))


def wrapper_digests(evolution, fingerprints):
    from trottertracks.resource_applicability import ax2b_native_df_v5 as native

    return [
        fingerprints.fingerprint(native.build_native_hadamard_wrapper(
            evolution, axis=axis, max_instructions=len(evolution.circuit.data) + 4,
        ))
        for axis in ("cosine", "sine")
    ]


@pytest.mark.parametrize(("formula", "q", "unique"), [
    ("4th(m5_best)", 1, 6), ("4th(m5_best)", 3, 6), ("4th(m5_best)", 8, 6),
    ("4th", 2, 2), ("2nd", 4, 1),
])
def test_full_and_b0i_keep_flat_circuit(original, formula, q, unique):
    from h6_price_acceleration import CircuitPieces, PriceFingerprints, fingerprint_context

    helpers, old_context = original
    prep = preparation()
    case = dict(formula=formula, q=q, T=0.17 * q, R_outer=None)
    with old_context() as fingerprints:
        expected = helpers["compose"](case, prep, [], -0.31, fingerprints.fingerprint)
        expected_wrappers = wrapper_digests(expected, fingerprints)
    with fingerprint_context()() as fingerprints:
        pieces = CircuitPieces(prep)
        shared = PriceFingerprints(fingerprints)
        result = pieces.assemble(case, helpers["core"], [], -0.31, shared.prepare)
        assert list(shared.axes.values()) == expected_wrappers
        assert wrapper_digests(result, fingerprints) == expected_wrappers
    assert result.circuit_fingerprint == expected.circuit_fingerprint
    assert result.instruction_upper_bound == expected.instruction_upper_bound
    assert result.circuit.global_phase.hex() == expected.circuit.global_phase.hex()
    assert len(pieces.deterministic) == unique


def test_pr_reuses_only_flanks_across_paths(original, monkeypatch):
    from h6_price_acceleration import CircuitPieces, PriceFingerprints, fingerprint_context
    from trotterlib.df_rte_qiskit import QiskitDFRTEEventCircuitBuilder

    helpers, old_context = original
    prep = preparation()
    case = dict(formula="4th(m5_best)", q=2, T=0.37, R_outer=16)
    pieces = CircuitPieces(prep)
    steps = pieces.schedule(case, helpers["core"], random_tail=True)
    pieces.prepare_pr_flanks(case, helpers["core"])
    assert len(pieces.flanks) == 6
    cached_flanks = pieces.flanks.copy()

    generated = []

    def replay_event(builder, occurrence, *, basis_plan):
        generated.append((occurrence, basis_plan))
        tail = QuantumCircuit(3)
        tail.cp(math.sin(occurrence), 2, 0)
        tail.rz(math.cos(occurrence), 1)
        tail.global_phase = 0.11 * occurrence
        return SimpleNamespace(circuit=tail)

    monkeypatch.setattr(QiskitDFRTEEventCircuitBuilder, "build_sequence", replay_event)
    retained = []
    for path in range(2):
        requests = [
            SimpleNamespace(step_time=step["signed_time"], preparation=prep,
                            rte_occurrence=path * len(steps) + i + 1, rte_basis_plan=(path, i))
            for i, step in enumerate(steps)
        ]
        expected_events = [(request.rte_occurrence, request.rte_basis_plan) for request in requests]
        generated.clear()
        with old_context() as fingerprints:
            expected = helpers["compose"](case, prep, requests, 0.23, fingerprints.fingerprint)
            expected_wrappers = wrapper_digests(expected, fingerprints)
        assert generated == expected_events
        generated.clear()
        with fingerprint_context()() as fingerprints:
            shared = PriceFingerprints(fingerprints)
            result = pieces.assemble(case, helpers["core"], requests, 0.23, shared.prepare)
            assert list(shared.axes.values()) == expected_wrappers
            assert wrapper_digests(result, fingerprints) == expected_wrappers
        assert generated == expected_events
        assert result.circuit_fingerprint == expected.circuit_fingerprint
        assert result.instruction_upper_bound == expected.instruction_upper_bound
        assert result.circuit.global_phase.hex() == expected.circuit.global_phase.hex()
        assert all(pieces.flanks[key] is value for key, value in cached_flanks.items())
        retained.append(result.circuit_fingerprint)
    assert retained[0] != retained[1]


def test_rte_event_reuse_preserves_legacy_seed_and_law(original):
    import json
    from h6_price_acceleration import event_receipts, reuse_sampled_events
    from trotterlib import rte

    components = (rte.RTEComponent("a", .7, .7, 1), rte.RTEComponent("b", .3, .3, -1))
    cases = [(components, rte.finite_rte_distribution(.01, 2), 17),
             (components[::-1], rte.finite_rte_distribution(-.6, 4), 31)]
    def sample(table, law, seed):
        return rte.sample_rte_events(table, law, sample_count=256, seed=seed)

    expected = [sample(*case) for case in cases]
    original_make = rte._make_event
    with reuse_sampled_events():
        actual = [sample(*case) for case in cases]
    assert rte._make_event is original_make
    assert actual == expected
    assert len({id(event) for event in actual[0]}) < 5
    entries = event_receipts(actual[1])
    assert json.dumps(entries) == json.dumps([event.to_dict() for event in expected[1]])
    assert len({id(receipt) for receipt in entries}) == len({id(event) for event in actual[1]})


def test_same_compiler_metrics_and_receipts(original, tmp_path, monkeypatch):
    from h6_price_acceleration import CircuitPieces, PriceFingerprints, acquire_axis as acquire_shared, fingerprint_context
    from run_full_m7 import acquire_axis
    import run_h6_df19_price_v2_q4 as runner

    helpers, old_context = original
    prep = preparation()
    case = dict(formula="4th(m5_best)", q=2, T=0.34, R_outer=None)
    old_path, new_path = tmp_path / "old", tmp_path / "new"
    old_path.mkdir()
    new_path.mkdir()
    monkeypatch.setattr(runner, "price_axis", acquire_axis)
    with old_context() as fingerprints:
        expected = helpers["compose"](case, prep, [], -0.31, fingerprints.fingerprint)
        old = runner.parallel_axes(old_path, expected, fingerprints)
    with fingerprint_context()() as fingerprints:
        shared = PriceFingerprints(fingerprints)
        result = CircuitPieces(prep).assemble(case, helpers["core"], [], -0.31, shared.prepare)
        monkeypatch.setattr(runner, "price_axis", acquire_shared)
        new = runner.parallel_axes(new_path, result, shared)
    assert old.keys() == new.keys() == {"cosine", "sine"}
    for axis, after in new.items():
        for field in ("metrics", "fingerprint", "compiler"):
            assert old[axis][field] == after[field]
        assert (new_path / after["qpy"]).is_file()


def test_single_axis_and_empty_evolution(original):
    from h6_price_acceleration import shared_fingerprints, hadamard_wrapper

    _, old_context = original
    evolution = QuantumCircuit(2, global_phase=-0.37)
    with old_context() as fingerprints:
        assert shared_fingerprints(evolution, ("sine",)) == (
            fingerprints.fingerprint(evolution),
            {"sine": fingerprints.fingerprint(hadamard_wrapper(evolution, "sine"))},
        )
