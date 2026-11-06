"""Artificial tiny-integral regression; no saved H4 input, circuit or signal."""
import numpy as np
import pytest
from test_o1_m2_residual_adapter import (
    adapter, fixture, pauli_matrix, physical_car, prepare,
)
from trottertracks.o1_m2_cutoff_free_pauli import map_integrals


@pytest.fixture(autouse=True)
def no_scientific_input(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved scientific inputs are outside synthetic scope")
    monkeypatch.setattr(np, "load", forbidden)


@pytest.mark.parametrize("scale", [1e-10, 1e-16, 1e-22])
def test_tiny_complete_residual_identity_and_one_body_are_retained(scale):
    h, g, constant, J, U = fixture()
    result = prepare(h*scale, g*scale, constant*scale, J*scale, U)
    physical, core, residual = result.pauli_parts()
    R = physical_car(result.residual.h, result.residual.g, 0.)
    H = physical_car(result.physical.h, result.physical.g, constant*scale)
    C = physical_car(result.core.h, result.core.g, constant*scale)
    # Residual-scale test: a zero mapper must fail even at 1e-22 input scale.
    tol = 128*np.finfo(float).eps*np.max(abs(R))
    assert np.max(abs(R)) > 0 and residual.terms and residual.constant != 0
    np.testing.assert_allclose(pauli_matrix(residual), R, rtol=0, atol=tol)
    np.testing.assert_allclose(pauli_matrix(core)+pauli_matrix(residual), H,
                               rtol=0, atol=128*np.finfo(float).eps*np.max(abs(H)))
    np.testing.assert_allclose(C+R, H, rtol=0, atol=tol)
    assert residual.constant == pytest.approx(np.trace(R).real/16, rel=2e-14, abs=0)
    assert result.residual.nuclear_constant == 0
    assert np.max(abs(result.residual.h)) > 0
    law = result.residual_rte(step_time=.125, r=2, K=2)
    assert law.tail == residual.terms
    assert law.hamiltonian.constant == residual.constant


def test_no_library_global_change_and_old_mapper_negative_control():
    import openfermion.config as config
    before = config.EQ_TOLERANCE
    packet = adapter.SpatialIntegrals(np.array([[3e-12]]), np.array([[[[-2e-12]]]]), 7e-13)
    fixed = map_integrals(packet, model_id="tiny_signed")
    old = adapter._pauli_module().PauliHamiltonian.from_spatial_integrals(
        packet.h, packet.g, nuclear_repulsion=packet.nuclear_constant)
    ref = physical_car(packet.h, packet.g, packet.nuclear_constant)
    assert config.EQ_TOLERANCE == before
    np.testing.assert_allclose(pauli_matrix(fixed), ref, rtol=0, atol=1e-26)
    assert np.max(abs(pauli_matrix(old)-ref)) > 1e-12
    assert fixed.metadata["coefficient_cutoff"] == 0


def test_zero_and_signed_scalar_are_exactly_accounted_once():
    for scalar in (0., -3e-20, 2.5):
        packet = adapter.SpatialIntegrals(np.zeros((1,1)), np.zeros((1,)*4), scalar)
        mapped = map_integrals(packet, model_id="scalar")
        assert mapped.constant == scalar and mapped.terms == ()
