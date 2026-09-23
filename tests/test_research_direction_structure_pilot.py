from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from trotterlib.research_direction_structure_pilot import (
    build_conjugated_pauli_rotation,
    build_conjugated_pauli_sequence,
    finalize_wp06a_artifact,
    maximum_operator_difference,
    support_restricted_unitary_completion,
    validate_wp06a_artifact,
)


def _orthogonal(size: int, seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q_matrix, r_matrix = np.linalg.qr(rng.normal(size=(size, size)))
    signs = np.where(np.diag(r_matrix) < 0.0, -1.0, 1.0)
    return (q_matrix @ np.diag(signs)).astype(np.complex128)


def test_support_completion_is_unitary_and_preserves_selected_columns() -> None:
    full = _orthogonal(6)
    restricted = support_restricted_unitary_completion(full, (1, 4))

    np.testing.assert_allclose(restricted.T @ restricted, np.eye(6), atol=1e-12)
    np.testing.assert_allclose(restricted[:, (1, 4)], full[:, (1, 4)], atol=1e-12)


def test_support_completion_preserves_controlled_conjugated_zz_and_phase() -> None:
    full = _orthogonal(4)
    support = (0, 2)
    restricted = support_restricted_unitary_completion(full, support)
    reference = build_conjugated_pauli_rotation(
        full,
        support=support,
        angle=0.27,
        controlled=True,
        scalar_phase=-0.13,
    )
    candidate = build_conjugated_pauli_rotation(
        restricted,
        support=support,
        angle=0.27,
        controlled=True,
        scalar_phase=-0.13,
    )
    missing_phase = build_conjugated_pauli_rotation(
        restricted,
        support=support,
        angle=0.27,
        controlled=True,
        scalar_phase=0.0,
    )

    assert maximum_operator_difference(
        reference, candidate, allow_global_phase=False
    ) < 1e-12
    assert maximum_operator_difference(
        reference, missing_phase, allow_global_phase=False
    ) > 1e-3


def test_shared_full_basis_and_per_event_support_bases_preserve_short_sequence() -> None:
    full = _orthogonal(4)
    supports = ((0, 1), (0, 2))
    angles = (0.21, -0.08)
    reference = build_conjugated_pauli_sequence(
        (full, full),
        supports=supports,
        angles=angles,
        controlled=True,
        scalar_phase=-0.05,
        fuse_shared_basis=True,
    )
    candidate = build_conjugated_pauli_sequence(
        tuple(
            support_restricted_unitary_completion(full, support)
            for support in supports
        ),
        supports=supports,
        angles=angles,
        controlled=True,
        scalar_phase=-0.05,
        fuse_shared_basis=False,
    )

    assert maximum_operator_difference(
        reference, candidate, allow_global_phase=False
    ) < 1e-12


def test_wp06a_artifact_is_tamper_evident_and_scope_guarded() -> None:
    body = {
        "scope": {
            "final_total_cost_evaluation_performed": False,
            "decision_grade": False,
        },
        "decision": {
            "eta_trigger_fired": True,
            "universal_support_restricted_replacement_adopted": False,
        },
        "checks": {"test": True},
        "overall_pass": True,
    }
    payload = finalize_wp06a_artifact(body, provenance={"test": True})
    validate_wp06a_artifact(payload)

    tampered = deepcopy(payload)
    tampered["decision"]["eta_trigger_fired"] = False
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        validate_wp06a_artifact(tampered)
