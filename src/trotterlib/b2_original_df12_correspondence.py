"""Fixed saved-input operator comparison, without state solve or evolution.

See the result-prior PLAN. No scientific I/O or calculation runs on import.
The existing CAR/JW/occupation/DF kernels are reused without changing them.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.linalg import svdvals

from trottertracks.resource_applicability import ax2b_independent_reference as occupation

HDF_SHA = "a3c88f6afa8965a994acb4f6ede4b8fcea1d074755f282b292c790e4c573a958"
DF_SHA = "3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a"
DF_HASH = "de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424"
DF_COMMIT = "bf5d3b2405b2bec4f88dd0589b75dd737b13e052"
DF_PATH = "artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz"
GEN_COMMIT = "c644925b50587072784846df09bf02c39e8453e1"
ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_hash(value):
    """Existing S0 identity convention: dtype, shape JSON, C-order bytes."""
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(a.dtype.str.encode())
    h.update(json.dumps(list(a.shape), separators=(",", ":")).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def basic_metrics(delta, *, spectral=False):
    a = np.atleast_2d(np.asarray(delta, dtype=np.complex128))
    result = dict(max_abs_entry=float(np.max(abs(a), initial=0)),
                  frobenius=float(np.linalg.norm(a)),
                  induced_infinity_norm=float(np.max(np.sum(abs(a), axis=1), initial=0)))
    if spectral:
        result["spectral_norm"] = float(svdvals(a, check_finite=True)[0])
    return result


def arithmetic_rule(one, two, hamiltonian):
    n, rank = len(one), len(hamiltonian.lambdas)
    d = 1 << n
    scale = (abs(hamiltonian.constant) + np.sum(abs(one)) + np.sum(abs(two))
             + np.sum(abs(hamiltonian.one_body))
             + sum(abs(w) * np.sum(abs(g))**2 for w, g in
                   zip(hamiltonian.lambdas, hamiltonian.g_matrices, strict=True)))
    m = 4*n**4 + 8*d + 16*rank
    eps = np.finfo(float).eps
    gamma = m*eps/(1-m*eps)
    return dict(m=m, epsilon=eps, operand_scale=float(scale), gamma=gamma,
                max_entry_tolerance=float(gamma*max(1, scale)),
                MO_orthonormality_tolerance=1e-10, mp_dps=[80, 120],
                mp_stability_relative="1e-60", rigorous_roundoff_bound=False,
                model_error_acceptance_threshold=None)


def exact_jw(constant, one, two):
    """Use existing sparse JW without SymbolicOperator += coefficient pruning."""
    from openfermion import FermionOperator
    from openfermion.linalg import jordan_wigner_sparse
    op = FermionOperator()
    op.terms = {(): complex(constant)}
    for p, q in np.argwhere(one != 0):
        op.terms[((int(p), 1), (int(q), 0))] = complex(one[p, q])
    for p, q, r, s in np.argwhere(two != 0):
        op.terms[((int(p), 1), (int(q), 1), (int(r), 0), (int(s), 0))] = complex(two[p, q, r, s])
    return jordan_wigner_sparse(op, len(one)).toarray()


def occupation_original(constant, one, two, basis, *, convert=complex):
    """Minimal quartic extension of the existing big-endian occupation oracle."""
    n = len(one)
    positions = {int(index): j for j, index in enumerate(basis)}
    one = [[convert(x) for x in row] for row in one]
    terms = [(tuple(map(int, pqrs)), convert(two[tuple(pqrs)]))
             for pqrs in np.argwhere(two != 0)]
    out = [[convert(0) for _ in basis] for _ in basis]
    for column, index in enumerate(basis):
        values = occupation._one_body(one, {int(index): convert(1)})
        values[int(index)] = values.get(int(index), 0) + convert(constant)
        for (p, q, r, s), coefficient in terms:
            target, sign = int(index), 1
            for mode, create in ((s, False), (r, False), (q, True), (p, True)):
                step = occupation._ladder(target, mode, n, create)
                if step is None:
                    break
                target, factor = step
                sign *= factor
            else:
                values[target] = values.get(target, 0) + coefficient*sign
        for target, value in values.items():
            if target not in positions and value != 0:
                raise ValueError("ORIGINAL_REFERENCE_LEAVES_SECTOR")
            if target in positions:
                out[positions[target]][column] = value
    return out


def precision_task(payload):
    """No eigenvalues, eigenvectors, expm or signal: only supplied operators."""
    import mpmath as mp
    dps, constant, pairs, h_eff, weights, factors, basis = payload
    started = time.perf_counter()
    with mp.workdps(dps):
        def exact(x):
            z = complex(x)
            def real(v):
                a, b = v.as_integer_ratio()
                return mp.mpf(a)/b
            return mp.mpc(real(z.real), real(z.imag))
        matrices = {name: occupation_original(constant, one, two, basis, convert=exact)
                    for name, (one, two) in pairs.items()}
        matrices["H_DF12"] = occupation.occupation_df_matrix(
            constant, h_eff, weights, factors, basis, convert=exact)
        strings = {name: [[(mp.nstr(x.real, dps), mp.nstr(x.imag, dps)) for x in row]
                          for row in matrix] for name, matrix in matrices.items()}
    return dict(dps=dps, matrices=strings, elapsed_seconds=time.perf_counter()-started)


def precision_comparisons(low, high, binary64):
    import mpmath as mp
    with mp.workdps(140):
        def parse(result):
            return {k: mp.matrix([[mp.mpc(*v) for v in row] for row in matrix])
                    for k, matrix in result["matrices"].items()}
        a, b = parse(low), parse(high)
        stability, roundoff = {}, {}
        for name in b:
            residual = max(abs(x) for x in a[name]-b[name])
            limit = mp.mpf("1e-60")*max(1, max(abs(x) for x in b[name]))
            stability[name] = dict(max_abs_entry=mp.nstr(residual, 30),
                                   tolerance=mp.nstr(limit, 30), passed=residual <= limit)
            rounded = np.array(b[name].tolist(), dtype=complex)
            roundoff[name] = basic_metrics(binary64[name]-rounded)
        differences = {}
        for left, right in (("H_phys", "H_src"), ("H_src", "H_DF12"), ("H_phys", "H_DF12")):
            delta = b[left]-b[right]
            rounded_delta = np.array(delta.tolist(), dtype=complex)
            singular = float(svdvals(rounded_delta)[0])
            differences[left+"_minus_"+right] = dict(
                max_abs_entry=mp.nstr(max(abs(x) for x in delta), 40),
                frobenius=mp.nstr(mp.sqrt(sum(abs(x)**2 for x in delta)), 40),
                induced_infinity_norm=mp.nstr(max(sum(abs(delta[i, j]) for j in range(delta.cols))
                                                    for i in range(delta.rows)), 40),
                spectral_norm_of_high_precision_difference_binary64_SVD=singular)
    return dict(stability_80_vs_120=stability, binary64_vs_120=roundoff,
                model_differences_120=differences,
                precision_elapsed_seconds={str(low["dps"]): low["elapsed_seconds"],
                                           str(high["dps"]): high["elapsed_seconds"]},
                certified_interval=False,
                description="Exact integer-ratio lift of fixed binary64 coefficients; no original-H eigensolve")


def _reference_module():
    spec = importlib.util.spec_from_file_location(
        "b2_saved_car_reference", ROOT/"scripts/run_o1_m1_h4_operator_correctness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(hdf_path, output):
    import h5py
    import scipy
    import openfermion
    from openfermion import MolecularData, get_sparse_operator
    from openfermion.circuits.low_rank import get_chemist_two_body_coefficients
    from openfermion.config import EQ_TOLERANCE
    from .df_hamiltonian import PhysicalSector, df_linear_operator
    from .h4_molecular_inputs import spin_lift_unthresholded
    from .h2_non_df_pilot import reverse_bits
    from .pr2_s0_s1_validation import load_snapshot

    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    output, hdf_path = Path(output), Path(hdf_path)
    output.mkdir(parents=True, exist_ok=False)
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT)
    source_commit = git("rev-parse", "HEAD").decode().strip()
    source_paths = ["src/trotterlib/b2_original_df12_correspondence.py",
                    "scripts/run_b2_original_df12_correspondence.py",
                    "docs/research/b2_original_df12_correspondence_20261106/PLAN.md"]
    for path in source_paths:
        if git("show", source_commit+":"+path) != (ROOT/path).read_bytes():
            raise ValueError("COMMITTED_EXECUTION_SOURCE_REQUIRED: "+path)
    if sha(hdf_path.read_bytes()) != HDF_SHA:
        raise ValueError("FIXED_HDF5_HASH_MISMATCH")
    raw = git("show", DF_COMMIT+":"+DF_PATH)
    if sha(raw) != DF_SHA:
        raise ValueError("FIXED_DF12_HASH_MISMATCH")
    # Reuse the existing path-based snapshot loader on a private scratch copy.
    # Raw is already a public fixed DF blob, and is never added to this result.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="b2-fixed-df-loader-") as tmp:
        npz = Path(tmp)/"fixed.npz"
        npz.write_bytes(raw)
        ham, sector, unused_df_state, metadata = load_snapshot(npz)
        del unused_df_state
    if metadata["hamiltonian_hash"] != DF_HASH or ham.n_blocks != 12:
        raise ValueError("DF_IDENTITY_OR_RANK")
    if metadata["fragment_order"] != list(range(12)):
        raise ValueError("GENERATION_ORDER")
    old_source = {}
    for path in ("src/trotterlib/pr2_s0_s1_validation.py", "src/trotterlib/df_hamiltonian.py"):
        digest = sha(git("show", GEN_COMMIT+":"+path))
        if digest != metadata["source_hashes"][path]:
            raise ValueError("GENERATION_SOURCE_HASH")
        old_source[path] = digest

    # This constructor/load/getter path only reads saved integrals; no run_pyscf.
    molecule = MolecularData(filename=str(hdf_path.with_suffix("")))
    one_mo, eri_of = molecule.get_integrals()
    interaction = molecule.get_molecular_hamiltonian()
    one_phys, spin_two_phys = spin_lift_unthresholded(one_mo, eri_of.transpose(0, 3, 1, 2))
    two_phys = spin_two_phys/2
    one_src, two_src = interaction.one_body_tensor, interaction.two_body_tensor
    n, d = len(one_src), 1 << len(one_src)
    if (n, ham.n_qubits, sector.dimension, sector.nelec_alpha, sector.nelec_beta) != (8, 8, 36, 2, 2):
        raise ValueError("FIXED_SYSTEM_DIMENSION")
    expected_geometry = [("H", (0., 0., z)) for z in (-1.5, -.5, .5, 1.5)]
    geometry = [(atom, tuple(map(float, position))) for atom, position in molecule.geometry]
    if (geometry != expected_geometry or molecule.basis.lower() != "sto-3g"
            or molecule.n_electrons != 4 or molecule.charge != 0 or molecule.multiplicity != 1):
        raise ValueError("FIXED_MOLECULE_METADATA")
    if float(molecule.nuclear_repulsion).hex() != float(ham.constant).hex():
        raise ValueError("NUCLEAR_SCALAR_EXACT_MISMATCH")
    rule = arithmetic_rule(one_phys, two_phys, ham)
    tau, checks = rule["max_entry_tolerance"], []
    def check(name, left, right=0, tolerance=tau):
        diagnostic = basic_metrics(np.asarray(left)-right)
        checks.append(dict(name=name, **diagnostic, tolerance=tolerance,
                           passed=diagnostic["max_abs_entry"] <= tolerance))

    permutation = reverse_bits(n)
    indices = permutation[sector.basis_indices]
    expected_sector = PhysicalSector.spin_sector(n_qubits=8, nelec_alpha=2, nelec_beta=2)
    if not np.array_equal(sector.basis_indices, expected_sector.basis_indices):
        raise ValueError("SAVED_SECTOR_BASIS")
    ref = _reference_module()
    a = ref.annihilators(n)
    identity = np.eye(d, dtype=complex)
    original = lambda one, two: ham.constant*identity + ref.one_body(one, a).toarray() + ref.quartic(two, a).toarray()
    matrices = {"H_phys": original(one_phys, two_phys), "H_src": original(one_src, two_src)}
    retained = ref.one_body(ham.one_body, a).toarray()
    squares, fragment_checks = np.zeros_like(retained), []
    for index, (w, g) in enumerate(zip(ham.lambdas, ham.g_matrices, strict=True)):
        lifted = ref.one_body(g, a)
        squares += w*(lifted@lifted).toarray()
        check(f"factor_{index}_Hermiticity", g, g.conj().T)
        spatial = g[::2, ::2]
        check(f"factor_{index}_exact_spin_lift", g, np.kron(spatial, np.eye(2)))
        fragment_checks.append(dict(index=index, sign=int(np.sign(w)),
                                    saved_G_hash=metadata["g_matrix_hashes"][index]))
    matrices["H_DF12"] = ham.constant*identity + retained + squares
    for name, one, two in (("H_phys", one_phys, two_phys), ("H_src", one_src, two_src)):
        jw = exact_jw(ham.constant, one, two)[np.ix_(permutation, permutation)]
        check(name+"_CAR_vs_sparse_JW_full", matrices[name], jw)
        independent_sector = np.asarray(occupation_original(ham.constant, one, two, sector.basis_indices), dtype=complex)
        check(name+"_full_vs_independent_sector", matrices[name][np.ix_(indices, indices)], independent_sector)
    full_sector = PhysicalSector(n_qubits=n, basis_indices=np.arange(d))
    full_op, full_counter = df_linear_operator(ham, full_sector, backend="python")
    full_df = full_op.matmat(identity)[np.ix_(permutation, permutation)]
    check("DF_CAR_vs_existing_full_matvec", matrices["H_DF12"], full_df)
    sector_op, sector_counter = df_linear_operator(ham, sector, backend="python")
    sector_df = sector_op.matmat(np.eye(sector.dimension))
    check("DF_full_vs_existing_sector_matvec", matrices["H_DF12"][np.ix_(indices, indices)], sector_df)
    independent_df = np.asarray(occupation.occupation_df_matrix(
        ham.constant, ham.one_body, ham.lambdas, ham.g_matrices, range(d)), dtype=complex)
    check("DF_full_vs_independent_occupation", full_df, independent_df[np.ix_(permutation, permutation)])
    correction, _ = get_chemist_two_body_coefficients(two_src, spin_basis=True)
    expected_eff = .5*((one_src+correction)+(one_src+correction).conj().T)
    check("saved_one_body_vs_SOURCE_correction_and_Hermitianization", ham.one_body, expected_eff)
    quadratic = np.einsum("l,lpq,lqr->pr", ham.lambdas, np.asarray(ham.g_matrices), np.asarray(ham.g_matrices))
    quartic = -np.einsum("l,lpq,lrs->prqs", ham.lambdas, np.asarray(ham.g_matrices), np.asarray(ham.g_matrices))
    normal = original(ham.one_body+quadratic, quartic)
    check("DF_signed_squares_vs_normal_order_full", matrices["H_DF12"], normal)
    check("scalar_full_vacuum_once", np.array([matrices[k][0, 0] for k in matrices]), ham.constant)
    check("MO_orthonormality", molecule.canonical_orbitals.T @ molecule.overlap_integrals @ molecule.canonical_orbitals,
          np.eye(4), tolerance=rule["MO_orthonormality_tolerance"])
    outside = np.setdiff1d(np.arange(d), indices)
    na = np.array([sum((j >> q)&1 for q in range(0, n, 2)) for j in range(d)])
    nb = np.array([sum((j >> q)&1 for q in range(1, n, 2)) for j in range(d)])
    spin_raise = np.zeros((n, n))
    spin_raise[np.arange(0, n, 2), np.arange(1, n, 2)] = 1
    plus = ref.one_body(spin_raise, a)
    from scipy.sparse import diags
    spin_squared = .5*(plus@plus.conj().T+plus.conj().T@plus)+diags(.25*(na-nb)**2)
    for name, matrix in matrices.items():
        check(name+"_Hermiticity_full", matrix, matrix.conj().T)
        check(name+"_Hermiticity_sector", matrix[np.ix_(indices, indices)], matrix[np.ix_(indices, indices)].conj().T)
        for label, counts in (("N", na+nb), ("Nalpha", na), ("Nbeta", nb), ("Sz", .5*(na-nb))):
            check(name+"_commutator_"+label, matrix*(counts[None, :]-counts[:, None]))
        check(name+"_commutator_S_squared", spin_squared@matrix-(spin_squared.T@matrix.T).T)
        check(name+"_sector_leakage", matrix[np.ix_(outside, indices)])
    pairs = (("H_phys", "H_src"), ("H_src", "H_DF12"), ("H_phys", "H_DF12"))
    differences = {space: {left+"_minus_"+right: basic_metrics(
        (matrices[left]-matrices[right]) if space == "full_256" else
        (matrices[left]-matrices[right])[np.ix_(indices, indices)], spectral=True)
        for left, right in pairs} for space in ("full_256", "sector_36")}
    binary_sector = {name: matrix[np.ix_(indices, indices)] for name, matrix in matrices.items()}
    payload = (ham.constant, {"H_phys": (one_phys, two_phys), "H_src": (one_src, two_src)},
               ham.one_body, ham.lambdas, ham.g_matrices, sector.basis_indices.tolist())
    tasks = [(digits, *payload) for digits in (80, 120)]
    tasks.extend((digits, *payload[:-1], list(range(d))) for digits in (80, 120))
    with ProcessPoolExecutor(max_workers=min(len(tasks), len(os.sched_getaffinity(0)))) as pool:
        precision = list(pool.map(precision_task, tasks))
    mp_result = dict(sector_36=precision_comparisons(*precision[:2], binary_sector),
                     full_256=precision_comparisons(*precision[2:],
                         {name: matrix[np.ix_(permutation, permutation)] for name, matrix in matrices.items()}))
    mp_result["sector_36"]["comparison_basis"] = "saved OpenFermion sector order; binary64 uses bit-reversed Qiskit indices in this same row order"
    mp_result["full_256"]["comparison_basis"] = "OpenFermion MSB; binary64 Qiskit full matrix is bit-reversed before precision comparison"
    for space, diagnostic in mp_result.items():
        for name, comparison in diagnostic["binary64_vs_120"].items():
            checks.append(dict(name=name+"_binary64_vs_120_"+space, **comparison,
                               tolerance=tau, passed=comparison["max_abs_entry"] <= tau))
    symbolic = get_sparse_operator(interaction, n_qubits=n).toarray()[np.ix_(permutation, permutation)]
    package_paths = {"MolecularData": Path(openfermion.__file__).parent/"chem/molecular_data.py",
                     "low_rank_correction": Path(openfermion.__file__).parent/"circuits/low_rank.py"}
    reused_paths = ["scripts/run_o1_m1_h4_operator_correctness.py",
                    "src/trottertracks/resource_applicability/ax2b_independent_reference.py",
                    "src/trotterlib/pr2_s0_s1_validation.py", "src/trotterlib/df_hamiltonian.py",
                    "src/trotterlib/h4_molecular_inputs.py", "src/trotterlib/h2_non_df_pilot.py"]
    result = dict(schema_version="b2_original_df12_operator_v1", source_commit=source_commit,
        started_at_utc=started_at,
        status="OPERATOR_CHECKS_PASS_MODEL_DIFFERENCES_RECORDED" if
            all(c["passed"] for c in checks) and all(c["passed"] for diagnostic in mp_result.values()
                                                    for c in diagnostic["stability_80_vs_120"].values())
            else "STOP_OPERATOR_VALIDATION_INELIGIBLE",
        identity=dict(HDF5_SHA256=HDF_SHA, DF12_NPZ_SHA256=DF_SHA, DF12_commit=DF_COMMIT,
                      DF12_path=DF_PATH, saved_DF_hamiltonian_hash=DF_HASH,
                      generation_SOURCE_commit=GEN_COMMIT, generation_SOURCE_hashes=old_source,
                      execution_SOURCE_hashes={p: sha((ROOT/p).read_bytes()) for p in source_paths+reused_paths},
                      original_integral_hashes={"one_MO": array_hash(one_mo), "two_MO_OF": array_hash(eri_of),
                                               "MO": array_hash(molecule.canonical_orbitals),
                                               "AO_overlap": array_hash(molecule.overlap_integrals)},
                      operator_matrix_hashes={k: array_hash(v) for k, v in matrices.items()},
                      complete_historical_provenance_proven=False),
        conditions=dict(model="linear_H4", distance_angstrom=1., basis="STO-3G", spatial_orbitals=4,
                        spin_orbitals=n, electrons=4, nelec_alpha=2, nelec_beta=2,
                        full_dimension=d, sector_dimension=36, DF_rank=12,
                        generation_order=list(range(12)), B2_L_D=3, delta_window=None,
                        validation_scope="operator reconstruction only; no evolution or state evaluation",
                        tensor="W[p,q,r,s] a†p a†q a_r a_s; W=spin_lift(g_OF)/2; g_OF=(ps|qr)",
                        H_src="spinorb_from_spatial cutoff before 1/2; InteractionOperator supplied to DF",
                        state_order="Qiskit LSB; bit_reverse(OpenFermion MSB)",
                        original_RPE_target_selected=None, nuclear_scalar_hex=float(ham.constant).hex()),
        numerical_rule=rule, checks=checks, model_differences=differences, high_precision=mp_result,
        factor_identity_checks=fragment_checks,
        component_diagnostics=dict(one_body_correction_vs_saved=basic_metrics((ham.one_body-one_src)-correction),
            normal_order_one_body_vs_source=basic_metrics(ham.one_body+quadratic-one_src),
            normal_order_quartic_vs_source=basic_metrics(ref.quartic(quartic-two_src, a).toarray(), spectral=True)),
        cutoff_diagnostics=dict(EQ_TOLERANCE=EQ_TOLERANCE,
            removed_spin_one_coefficients=int(np.count_nonzero((one_phys != 0)&(one_src == 0))),
            removed_spin_two_coefficients=int(np.count_nonzero((two_phys != 0)&(two_src == 0))),
            symbolic_get_sparse_vs_H_src=basic_metrics(symbolic-matrices["H_src"], spectral=True),
            symbolic_conversion_is_target_definition=False),
        resources=dict(cpu_affinity=len(os.sched_getaffinity(0)), gpu_available=False,
                       precision_workers=min(len(tasks), len(os.sched_getaffinity(0))),
                       DF_backend="python excitation tables", CAR_backend="SciPy sparse",
                       BLAS_env={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
                       DF_action_counts=dict(full=full_counter["count"], sector=sector_counter["count"])),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         packages={k: importlib.metadata.version(k) for k in
                                   ("numpy", "scipy", "h5py", "openfermion", "openfermionpyscf", "qiskit", "mpmath")},
                         package_SOURCE_hashes={k: sha(v.read_bytes()) for k, v in package_paths.items()}),
        limitations=["No rigorous interval certification or historical h/g-to-DF receipt",
                     "No target adoption or RPE model-error budget qualification",
                     "No original-H ground state computed; saved DF state was not used"],
        forbidden_work_executed=dict(SCF=0, DF_decomposition=0, ground_state_solve=0,
                                     time_evolution=0, circuit_build_compile=0, sampling=0, shot_cost=0),
        mandatory_stop=True, elapsed_seconds=time.perf_counter()-started)
    if sha(hdf_path.read_bytes()) != HDF_SHA or sha(raw) != DF_SHA:
        raise ValueError("INPUT_CHANGED_DURING_RUN")
    with (output/"result.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result
