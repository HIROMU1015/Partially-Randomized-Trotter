"""Invented two-orbital CAR/JW/native checks; no molecular files or circuits."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from trottertracks import o1_m2_residual_adapter as adapter


def fixture():
    # Signed, non-square general J with offdiagonals; nontrivial real U.
    J = np.array([[[.7, -.4], [-.4, -.2]], [[-.1, .3], [.3, .6]]])
    U = np.array([[[.8, -.6], [.6, .8]], [[0., 1.], [1., 0.]]])
    h = np.array([[.23, -.17], [-.17, -.31]])
    # Construct physical chemist integrals independently, not with reconstruct.
    A = np.array([[.5, .2], [.2, -.3]])
    B = np.array([[-.1, .4], [.4, .6]])
    g = .9 * np.multiply.outer(A, A) - .6 * np.multiply.outer(B, B)
    return h, g, 1.7, J, U


def prepare(*values, **kwargs):
    return adapter.prepare_residual(*(values or fixture()),
        source_identity="artificial-source", input_identity="artificial-input",
        core_input_identity=kwargs.pop("core_input_identity", "artificial-input"),
        expected_frames=kwargs.pop("expected_frames", 2), sector=kwargs.pop("sector", (1, 1)),
        **kwargs)


def annihilators(n):
    """Independent bit-occupation CAR; mode 0 is least-significant bit."""
    result = []
    for q in range(n):
        a = np.zeros((1 << n, 1 << n))
        for s in range(1 << n):
            if s & (1 << q):
                a[s ^ (1 << q), s] = (-1) ** ((s & ((1 << q) - 1)).bit_count())
        result.append(a)
    return result


def physical_car(h, g, constant):
    n = len(h)
    a = annihilators(2 * n)
    out = constant * np.eye(1 << (2 * n))
    for p, q in np.ndindex(n, n):
        for spin in range(2):
            out += h[p, q] * a[2*p+spin].T @ a[2*q+spin]
    for p, q, r, s in np.ndindex((n,) * 4):
        for sigma, tau in np.ndindex(2, 2):
            out += .5 * g[p, q, r, s] * (a[2*p+sigma].T @ a[2*r+tau].T
                                                        @ a[2*s+tau] @ a[2*q+sigma])
    return out


def density_core_car(model):
    n = len(model.spatial_one_body_tensor)
    a = annihilators(2 * n)
    E = {(p, q): sum(a[2*p+s].T @ a[2*q+s] for s in range(2))
         for p, q in np.ndindex(n, n)}
    out = model.constant * np.eye(1 << (2 * n))
    for p, q in np.ndindex(n, n):
        out = out + model.spatial_one_body_tensor[p][q] * E[p, q]
    for frame in model.frames:
        J = np.array(frame.spatial_J)
        U = np.array(frame.spatial_orbital_rotation)
        densities = [sum(U[p, i] * U[q, i].conjugate() * E[p, q]
                         for p, q in np.ndindex(n, n)) for i in range(n)]
        for i, j in np.ndindex(n, n):
            out = out + .5 * J[i, j] * densities[i] @ densities[j]
    return out


def pauli_matrix(packet):
    """No Qiskit matrix/reference shortcut: interpret strings directly."""
    axes = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
            "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    out = packet.constant * np.eye(1 << packet.num_qubits, dtype=complex)
    for word, coefficient in packet.terms:
        op = np.array([[1.]])
        for axis in word:
            op = np.kron(op, axes[axis])
        out += coefficient * op
    return out


@contextmanager
def actual_native_dependencies():
    # Bypass only eager package __init__; real NumPy/Qiskit/OpenFermion modules,
    # eigendecomposition, coefficients and native classes remain in use.
    names = ("trotterlib", "trotterlib.df_trotter")
    packages = {}
    for name in names:
        package = ModuleType(name)
        package.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
        packages[name] = package
    with patch.dict(sys.modules, packages):
        import qiskit
        with patch.object(qiskit, "QuantumCircuit", side_effect=AssertionError("No circuit build")):
            yield


def test_passive_import_and_explicit_array_entry_have_no_file_io():
    code = """
import builtins, sys
sys.path.insert(0, 'src')
import numpy as np
from unittest.mock import patch
with patch.object(np, 'load', side_effect=AssertionError('no NPZ decode')):
    from trottertracks import o1_m2_residual_adapter
assert 'trotterlib' not in sys.modules
assert 'qiskit' not in sys.modules
assert 'openfermion' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)
    with patch.object(np, "load", side_effect=AssertionError("No NPZ input")):
        result = prepare()
    assert result.general_core.original_outer_rank == 2


def test_signed_general_j_full_core_matches_independent_car_and_physical_sum():
    result = prepare()
    h, g, constant, J, U = fixture()
    physical = physical_car(h, g, constant)
    core = density_core_car(result.general_core)
    normal_core = physical_car(result.core.h, result.core.g, constant)
    residual = physical_car(result.residual.h, result.residual.g, 0.)
    np.testing.assert_allclose(core, normal_core, rtol=0, atol=1e-14)
    np.testing.assert_allclose(core + residual, physical, rtol=0, atol=1e-14)
    # Ensure the fixtures would catch old LD-tail, sign and omitted one-body bugs.
    assert np.linalg.norm(result.residual.h) > .01
    assert np.linalg.norm(physical - core) > .1
    assert np.linalg.det(J[0]) < 0  # cannot impersonate one positive square DF
    np.testing.assert_array_equal(result.general_core.frames[1].spatial_J, J[1])
    assert result.core.nuclear_constant == result.physical.nuclear_constant == 1.7
    assert result.residual.nuclear_constant == 0.
    # Independent element contraction establishes original-g, not fitted-g κ.
    expected = h.copy()
    for p, q in np.ndindex(2, 2):
        expected[p, q] -= sum(g[p, r, r, q] for r in range(2)) / 2
    np.testing.assert_allclose(result.general_core.spatial_one_body_tensor, expected, atol=1e-16)
    np.testing.assert_allclose(result.physical.g - result.core.g, result.residual.g, atol=0)
    sector = [s for s in range(16) if sum((s >> q) & 1 for q in (0, 2)) == 1
              and sum((s >> q) & 1 for q in (1, 3)) == 1]
    assert len(sector) == 4
    np.testing.assert_allclose((core + residual)[np.ix_(sector, sector)],
                               physical[np.ix_(sector, sector)], rtol=0, atol=1e-14)


def test_existing_jw_and_finite_rte_use_entire_residual_with_electronic_identity():
    result = prepare()
    physical, core, residual = result.pauli_parts()
    reference = physical_car(result.physical.h, result.physical.g, 1.7)
    reference_core = density_core_car(result.general_core)
    np.testing.assert_allclose(pauli_matrix(physical), reference, rtol=0, atol=1e-14)
    np.testing.assert_allclose(pauli_matrix(core), reference_core, rtol=0, atol=1e-14)
    np.testing.assert_allclose(pauli_matrix(residual), reference - reference_core, rtol=0, atol=1e-14)
    assert abs(residual.constant) > .01  # nuclear residual zero ≠ electronic I zero
    assert residual.constant == pytest.approx(np.trace(reference - reference_core).real / 16)
    assert residual.metadata["sector_projection"] is False
    law = result.residual_rte(step_time=.125, r=2, K=2)
    assert law.retained == ()
    assert law.tail == residual.terms
    assert law.hamiltonian.constant == residual.constant
    assert sum(c.probability for c in law.components) == pytest.approx(1.)
    assert law.lambda_r == pytest.approx(sum(abs(c) for _, c in residual.terms))
    for c, (_, coefficient) in zip(law.components, residual.terms, strict=True):
        assert c.coefficient_sign == (1 if coefficient > 0 else -1)


def test_actual_native_preparation_keeps_base_and_every_signed_frame():
    result = prepare()
    with actual_native_dependencies():
        native = result.native_core()
    assert len(native.blocks) == 3  # base + ALL frames; never old ld=2 selection
    assert tuple(b.block_id for b in native.blocks) == ("one_body", "F0", "F1")
    assert native.constant == 1.7
    base = native.blocks[0]
    diagonal = base.diagonal
    vacuum = -diagonal.global_phase + sum(angle/2 for _, angle in diagonal.rz)
    assert vacuum == pytest.approx(0., abs=2e-15)
    energies = []
    for occupied in range(4):
        energies.append(-diagonal.global_phase + sum(
            angle * (-1 if q == occupied else 1) / 2 for q, angle in diagonal.rz))
    rotation = np.asarray(base.rotation)
    reconstructed = (rotation * energies) @ rotation.conj().T
    np.testing.assert_allclose(reconstructed, result.general_core.one_body_tensor,
                               rtol=0, atol=1e-14)
    # Independently evaluate diagonal number occupations and emitted Z angles.
    for block, frame in zip(native.blocks[1:], result.general_core.frames, strict=True):
        K = np.asarray(frame.K)
        for s in range(16):
            occupation = np.array([(s >> q) & 1 for q in range(4)])
            z = 1 - 2 * occupation
            d = block.diagonal
            emitted = -d.global_phase + sum(angle*z[q]/2 for q, angle in d.rz)
            emitted += sum(angle*z[p]*z[q]/2 for p, q, angle in d.rzz)
            assert emitted == pytest.approx(occupation @ K @ occupation, abs=2e-15)


def test_exact_core_has_zero_residual_and_scalar_is_not_duplicated():
    h, _, constant, J, U = fixture()
    seed = prepare()
    result = prepare(h, seed.core.g, constant, J, U)
    np.testing.assert_array_equal(result.residual.g, np.zeros((2,) * 4))
    np.testing.assert_allclose(result.residual.h, 0., atol=1e-16)
    # Nuclear offset changes physical and core together, leaving residual fixed.
    shifted = prepare(h, seed.physical.g, constant + 9., J, U)
    np.testing.assert_array_equal(shifted.residual.h, seed.residual.h)
    assert shifted.residual.nuclear_constant == 0.


@pytest.mark.parametrize("case", ["identity", "rank", "shape", "J", "U", "h", "g", "complex", "nonfinite", "scalar", "sector"])
def test_reject_mismatched_or_unsupported_inputs(case):
    h, g, constant, J, U = fixture()
    kwargs = {}
    if case == "identity": kwargs["core_input_identity"] = "different-input"
    if case == "rank": kwargs["expected_frames"] = 8
    if case == "shape": U = U[:1]
    if case == "J": J[0, 0, 1] += .1
    if case == "U": U[0, 0, 0] += .1
    if case == "h": h[0, 1] += .1
    if case == "g": g[0, 1, 0, 0] += .1
    if case == "complex": U = U.astype(complex)
    if case == "nonfinite": g[0, 0, 0, 0] = np.nan
    if case == "scalar": constant = np.array([constant])
    if case == "sector": kwargs["sector"] = (3, 0)
    with pytest.raises(ValueError):
        prepare(h, g, constant, J, U, **kwargs)


def test_explicit_lazy_loader_binds_bytes_and_copies_arrays(tmp_path):
    h, g, constant, J, U = fixture()
    core_path, input_path = tmp_path / "artificial_core.npz", tmp_path / "artificial_input.npz"
    np.savez(core_path, J=J, U=U)
    np.savez(input_path, h=h, g=g, constant=constant)
    binding = adapter.SavedCoreBinding("artificial-source",
        hashlib.sha256(core_path.read_bytes()).hexdigest(),
        hashlib.sha256(input_path.read_bytes()).hexdigest(), 2, 2)
    result = adapter.load_saved_pair(core_path, input_path, binding=binding, sector=(1, 1))
    np.testing.assert_array_equal(result.physical.h, h)
    assert not result.physical.h.flags.writeable
    np.savez(input_path, h=h + 1., g=g, constant=constant)
    with patch.object(np, "load", side_effect=AssertionError("Mismatch must fail before decode")):
        with pytest.raises(ValueError, match="bytes differ"):
            adapter.load_saved_pair(core_path, input_path, binding=binding, sector=(1, 1))
