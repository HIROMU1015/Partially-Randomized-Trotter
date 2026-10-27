"""Artificial regression tests only; never reopen the scientific NPZ or run a solver."""
import dataclasses
import json
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.integration.validate_h4_one_case_equivalence import (
    array_comparison, match_class, fingerprint, make_context, run)
from trottertracks.resource_applicability.pipeline import compose, ScientificExecutionDisabled, PipelineError
from trottertracks.resource_applicability.h4_geometry_pipeline import H4GeometryAdapter, FrozenH4Input


def qualification():
    return {"P1_commit": "2c865c66171fc9d1972b67c1257c57e5684fe26c",
            "candidate_identity": {"input": "a"*64, "state": "b"*64}}


def test_exact_error_is_measured_without_tolerance_relaxation():
    check = array_comparison([1.0, 2.0], [1.0, 2.0], atol=0, rtol=0)
    assert check["passed"] and check["max_abs_error"] == 0
    fail = array_comparison([1.0001], [1.0], atol=1e-12, rtol=1e-12)
    assert not fail["passed"] and fail["atol"] == 1e-12


def test_shape_and_nonfinite_are_rejected():
    assert not array_comparison([1], [[1]], atol=1e-12, rtol=0)["passed"]
    assert not array_comparison([np.nan], [0], atol=1, rtol=1)["passed"]


def test_wiring_and_metadata_are_not_promoted_to_independent():
    assert match_class(passed=True, wiring=True) == "MATCH_WIRING_ONLY"
    assert match_class(passed=True, metadata=True) == "MATCH_METADATA_ONLY"
    assert match_class(passed=True) == "INCONCLUSIVE"
    assert match_class(passed=False, independent=True) == "MISMATCH"


def test_fingerprint_uses_exact_binary64_encoding():
    assert fingerprint("x", {"value": .8}) != fingerprint("x", {"value": ".8"})
    assert fingerprint("x", {"a": 1, "b": 2}) == fingerprint("x", {"b": 2, "a": 1})


def test_fixed_context_is_single_non_B3_case():
    c = make_context(qualification())
    H4GeometryAdapter().validate(c)
    assert (c.df_rank, c.ld, c.q, c.rte_steps, c.finite_taylor_order, c.evolution_time) == (12,3,1,4,4,.8)
    assert c.deterministic_indices == (0,1,2) and c.randomized_indices == tuple(range(3,12))
    assert c.source_stop_status == "PRESERVED_NOT_RESTARTED" and c.input_kind == "frozen_scientific"


@pytest.mark.parametrize("mode", ["scientific", "real", "resume"])
def test_public_cli_modes_stay_disabled_before_adapter_access(mode):
    class Poison:
        def validate(self, *_):
            raise AssertionError("must reject before adapter access")
    with pytest.raises(ScientificExecutionDisabled):
        compose(make_context(qualification()), Poison(), mode=mode)


def test_synthetic_composer_does_not_accept_real_frozen_input():
    with pytest.raises(ScientificExecutionDisabled):
        compose(make_context(qualification()), H4GeometryAdapter(), mode="synthetic")


def test_context_mix_rejected_without_hamiltonian_call():
    c = make_context(qualification())
    data = FrozenH4Input({}, c.input_id, c.state_id, None, ("other-cohort",))
    class Poison:
        def hamiltonian(self, **_):
            raise AssertionError("constructor must not be reached")
    with pytest.raises(PipelineError, match="binding"):
        H4GeometryAdapter().input(c, data, Poison())


def test_existing_output_refused_before_any_science(tmp_path):
    d = tmp_path / "existing"; d.mkdir()
    for name in ("locations", "contract", "qualification"):
        (tmp_path / name).write_text("{}")
    with pytest.raises(FileExistsError):
        run(tmp_path/"locations", tmp_path/"contract", tmp_path/"qualification", d)
    assert list(d.iterdir()) == []
