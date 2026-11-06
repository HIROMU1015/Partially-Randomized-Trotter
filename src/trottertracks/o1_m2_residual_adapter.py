"""Explicit whole-core / physical-residual bridge; passive import, no runner.

Only supplied arrays are contracted. File decoding, native preparation and JW
mapping are separate explicit calls. No fit, state, circuit, RNG or cost entry
is invoked here. Molecular calls require their own scientific execution scope.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType

from . import c1_general_k_adapter as general


def _pauli_module():
    # Existing private-package pattern (c1_stage1.legacy_df_module), restricted
    # to this helper so df_hamiltonian/PySCF and eager trotterlib are not loaded.
    namespace = "trottertracks._o1_m2_legacy"
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = [str(Path(__file__).resolve().parents[1] / "trotterlib")]
        sys.modules[namespace] = package
    return importlib.import_module(namespace + ".pauli_signal_adapter")


@dataclass(frozen=True)
class SavedCoreBinding:
    """Saved core's generating SOURCE/input; never the new execution SOURCE."""
    source_identity: str
    core_sha256: str
    input_sha256: str
    frames: int
    orbitals: int

    def __post_init__(self):
        if not self.source_identity:
            raise ValueError("Explicit scientific SOURCE identity required")
        for value in (self.core_sha256, self.input_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Expected lowercase scientific file SHA256")
        if any(type(v) is not int or v < 1 for v in (self.frames, self.orbitals)):
            raise ValueError("Positive frame/orbital counts required")


UPDATE005 = SavedCoreBinding(
    source_identity="980b514c81eb415fe053f558d7d3c0b8cfb5b9b4",
    core_sha256="938340b8e09c85f5f4021582501ba6d81fc79d5382c37c1824863939bdf6e56e",
    input_sha256="1d6087f60484c6c249f298be75437730e59a1e18e97d1529c19fdf24370603db",
    frames=8, orbitals=4,
)


@dataclass(frozen=True)
class SpatialIntegrals:
    """Real orthonormal MO h and chemist g; scalar is nuclear, not JW identity."""
    h: object
    g: object
    nuclear_constant: float


@dataclass(frozen=True)
class WholeCoreResidual:
    physical: SpatialIntegrals
    core: SpatialIntegrals
    residual: SpatialIntegrals
    general_core: general.GeneralKModel
    sector: tuple[int, int] | None

    def pauli_parts(self):
        """Explicit JW mapping for a future physical=core+residual check.

        Returns (physical, core, residual). Uses the existing mapper unchanged;
        its OpenFermion arithmetic tolerances remain a qualification condition.
        No matrix, state, time evolution or circuit is constructed.
        """
        mapper = _pauli_module().PauliHamiltonian
        return tuple(mapper.from_spatial_integrals(
            p.h, p.g, nuclear_repulsion=p.nuclear_constant,
            model_id="M2_" + role,
            metadata=dict(core_generation_source_identity=self.general_core.source_identity,
                          input_identity=self.general_core.input_identity,
                          partition="all_core_frames_plus_physical_minus_core",
                          sector=self.sector, sector_projection=False))
            for role, p in (("physical", self.physical), ("core", self.core),
                            ("residual", self.residual)))

    def residual_rte(self, *, step_time, r, K):
        """Existing full nonidentity Pauli tail; identity is retained once.

        This configures a residual-only finite law, not a composed D/R kernel.
        Sampling and circuit methods of the returned adapter are not called.
        """
        module = _pauli_module()
        p = self.residual
        residual = module.PauliHamiltonian.from_spatial_integrals(
            p.h, p.g, nuclear_repulsion=p.nuclear_constant,
            model_id="M2_physical_minus_whole_core")
        return module.PauliSignalAdapter(residual, "PR", 0, step_time, r, K)

    def native_core(self):
        """Existing generic CDF native preparation, with every frame retained.

        Explicit eigendecomposition/primitive preparation only; no builder.
        The strong baseline's spin-adapted precision/fusion policy is a later
        execution choice, not a performance claim of this generic bridge.
        """
        from .c1_stage2_native import prepare_cdf
        return prepare_cdf(self.general_core)


def _readonly(array):
    result = array.copy()
    result.setflags(write=False)
    return result


def prepare_residual(h, g, constant, J, U, *, source_identity, input_identity,
                     core_input_identity, expected_frames=None, sector=None):
    """Keep original-g kappa and all signed-J frames; return normal-order R.

    R.g = g - g_fit (opposite the old saved tensor_residual).
    Hcore.h = kappa_original + contract(g_fit)/2; R.h = h - Hcore.h.
    The shared nuclear scalar is in Hcore once, so R.nuclear_constant=0.
    Electronic identity produced by JW is never assumed to vanish.
    source_identity denotes the saved core's generating SOURCE. A future
    scientific runner must separately fix this adapter's execution commit.
    """
    import numpy as np
    from .c1_stage1 import corrected_kappa
    from .c1_h6_initialization import reconstruct

    if not input_identity or input_identity != core_input_identity:
        raise ValueError("Core and physical input identity differ")
    arrays = tuple(np.asarray(x) for x in (h, g, J, U))
    if any(np.iscomplexobj(a) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Finite real spatial integrals/J/U required")
    h, g, J, U = (np.array(a, dtype=np.float64, copy=True) for a in arrays)
    if h.ndim != 2 or h.shape[0] < 1 or h.shape[0] != h.shape[1]:
        raise ValueError("Square nonempty physical h required")
    n = len(h)
    if (g.shape != (n,) * 4 or J.ndim != 3 or len(J) < 1
            or J.shape[1:] != (n, n) or U.shape != J.shape):
        raise ValueError("Physical/core shape mismatch")
    if expected_frames is not None and (type(expected_frames) is not int
                                        or len(J) != expected_frames):
        raise ValueError("Whole core frame count differs")
    if (not np.allclose(h, h.T, rtol=0, atol=1e-12)
            or not np.allclose(g, g.transpose(1, 0, 2, 3), rtol=0, atol=1e-12)
            or not np.allclose(g, g.transpose(2, 3, 0, 1), rtol=0, atol=1e-12)):
        raise ValueError("Real Hermitian chemist integral convention required")
    if not np.allclose(U.transpose(0, 2, 1) @ U, np.eye(n), rtol=0, atol=1e-10):
        raise ValueError("Orthogonal U required; no polar repair")
    scalar = np.asarray(constant)
    if (scalar.shape != () or np.iscomplexobj(scalar)
            or not np.isfinite(scalar)):
        raise ValueError("Finite real nuclear scalar required")
    if sector is not None:
        sector = tuple(sector)
        if len(sector) != 2 or any(type(v) is not int or not 0 <= v <= n for v in sector):
            raise ValueError("Sector must be explicit (Nalpha,Nbeta)")
    kappa = corrected_kappa(h, g)
    model = general.from_ffsim_number_arrays(
        one_body_tensor=kappa, diag_coulomb_mats=J, orbital_rotations=U,
        constant=float(scalar), z_representation=False,
        frame_ids=tuple(f"F{i}" for i in range(len(J))),
        group_ids=tuple(f"G{i}" for i in range(len(J))),
        source_identity=source_identity, input_identity=input_identity,
        one_body_role="ffsim_corrected_kappa", enable_general_k=True)
    fitted = reconstruct(J, U)
    core_h = kappa + .5 * np.einsum("prrq->pq", fitted)
    residual_h, residual_g = h - core_h, g - fitted
    if not all(np.isfinite(a).all() for a in (fitted, core_h, residual_h, residual_g)):
        raise ValueError("Nonfinite contraction")
    packet = lambda a, b, c: SpatialIntegrals(_readonly(a), _readonly(b), c)
    return WholeCoreResidual(packet(h, g, float(scalar)),
                             packet(core_h, fitted, float(scalar)),
                             packet(residual_h, residual_g, 0.), model, sector)


def load_saved_pair(core_path, input_path, *, binding=UPDATE005, sector=(2, 2)):
    """Explicit lazy file entry; verify scientific bytes before NPZ decoding.

    Defaults bind only the saved update005/input pair. Paths are never inferred
    from CWD or searched. Synthetic tests supply their own binding and paths.
    """
    import io
    import numpy as np
    payloads = []
    for path, expected in ((core_path, binding.core_sha256),
                           (input_path, binding.input_sha256)):
        data = Path(path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Saved scientific file bytes differ from binding")
        payloads.append(data)
    with np.load(io.BytesIO(payloads[0]), allow_pickle=False) as core:
        if set(core.files) != {"J", "U"}:
            raise ValueError("Expected saved J/U-only core")
        J, U = core["J"], core["U"]
    with np.load(io.BytesIO(payloads[1]), allow_pickle=False) as physical:
        if set(physical.files) != {"h", "g", "constant"}:
            raise ValueError("Expected physical h/g/constant input")
        h, g, constant = physical["h"], physical["g"], physical["constant"]
    if h.shape != (binding.orbitals,) * 2:
        raise ValueError("Bound orbital count differs")
    identity = "sha256:" + binding.input_sha256
    return prepare_residual(h, g, constant, J, U,
        source_identity=binding.source_identity, input_identity=identity,
        core_input_identity=identity, expected_frames=binding.frames, sector=sector)
