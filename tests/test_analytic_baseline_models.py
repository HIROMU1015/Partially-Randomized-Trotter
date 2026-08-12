from __future__ import annotations

import math

import pytest

from trotterlib.partial_randomized_pf import (
    LEGACY_REPO_MODEL,
    PR_PAPER_V2_MODEL,
    analytic_baseline_definition,
    randomized_prefactor_B,
    randomized_prefactor_B_for_model,
    randomized_prefactor_b0,
    randomized_prefactor_b0_for_model,
)


@pytest.mark.parametrize("method", ("qdrift", "rte"))
def test_legacy_model_preserves_unversioned_prefactor(method: str) -> None:
    assert randomized_prefactor_b0_for_model(
        LEGACY_REPO_MODEL, method, 1.7
    ) == randomized_prefactor_b0(method, 1.7)
    assert randomized_prefactor_B_for_model(
        2.3, LEGACY_REPO_MODEL, method, 1.7
    ) == randomized_prefactor_B(2.3, method, 1.7)


def test_v2_e22_model_uses_296_over_9_without_gamma() -> None:
    definition = analytic_baseline_definition(PR_PAPER_V2_MODEL, "rte")
    assert definition.coefficient == pytest.approx(296.0 / 9.0)
    assert definition.gamma is None
    assert definition.paper_version == "arXiv:2503.05647v2"
    assert definition.source_equation == "Appendix E, Eq. (E22)"
    assert randomized_prefactor_b0_for_model(
        PR_PAPER_V2_MODEL, "rte", 1.0
    ) == pytest.approx((296.0 / 9.0) * (0.1 * math.pi) ** 2)


def test_v2_e22_model_rejects_qdrift_label() -> None:
    with pytest.raises(ValueError, match="only for randomized_method='rte'"):
        analytic_baseline_definition(PR_PAPER_V2_MODEL, "qdrift")


def test_unknown_baseline_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported analytic baseline model"):
        analytic_baseline_definition("unknown", "rte")


@pytest.mark.parametrize("g_rand", (-1.0, math.inf, math.nan))
def test_versioned_prefactor_rejects_invalid_randomized_cost(g_rand: float) -> None:
    with pytest.raises(ValueError, match="g_rand"):
        randomized_prefactor_b0_for_model(LEGACY_REPO_MODEL, "rte", g_rand)


@pytest.mark.parametrize("kappa", (0.0, -1.0, math.inf, math.nan))
def test_versioned_prefactor_rejects_invalid_kappa(kappa: float) -> None:
    with pytest.raises(ValueError, match="kappa"):
        randomized_prefactor_B_for_model(
            kappa,
            PR_PAPER_V2_MODEL,
            "rte",
            1.0,
        )
