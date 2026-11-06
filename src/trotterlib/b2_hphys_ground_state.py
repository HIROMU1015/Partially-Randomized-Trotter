"""One fixed H_phys sector solve; private state, public diagnostics only.

No scientific I/O or solve on import. Existing construction, sector solver,
phase rule, hash and independent occupation reference are reused.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from . import b2_original_df12_correspondence as operators

ROOT = Path(__file__).resolve().parents[2]
PRIOR_COMMIT = "275813d0a12a4bca62a8c972312c80386829de22"
PRIOR_SOURCE = "126b9d4480c7937b39c8be3aef7fbecda48cb556"
PRIOR_PATH = "artifacts/b2_original_df12_correspondence/2026-11-06/run03/result.json"
RAW_HASH = "bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2"
NORM_TOL = 1e-12
RESIDUAL_TOL = 1e-10
HASH_DEFINITION = "sha256(dtype.str UTF8 + compact JSON(shape) UTF8 + C-order contiguous array bytes); S0 _array_hash convention"


def phase_only(vector, pivot):
    result = np.array(vector, dtype=complex, copy=True)
    return result*np.exp(-1j*np.angle(result[pivot]))


def solve_checked(raw, indices, independent_raw_sector):
    """Call the unchanged existing sector solver once; return all diagnostics."""
    from .h2_non_df_pilot import canonical_ground
    raw = np.asarray(raw, dtype=complex)
    indices = np.asarray(indices, dtype=np.int64)
    solve = .5*(raw+raw.conj().T)
    checks = []
    def check(name, error, limit):
        error = float(error)
        checks.append(dict(name=name, error=error, tolerance=limit,
                           passed=bool(np.isfinite(error) and error <= limit)))
    symmetry = dict(raw_full=operators.basic_metrics(raw-raw.conj().T, spectral=True),
                    solve_full=operators.basic_metrics(solve-solve.conj().T, spectral=True),
                    solve_minus_raw=operators.basic_metrics(solve-raw, spectral=True))
    raw_sector = raw[np.ix_(indices, indices)]
    solve_sector = solve[np.ix_(indices, indices)]
    symmetry["raw_sector"] = operators.basic_metrics(raw_sector-raw_sector.conj().T, spectral=True)
    symmetry["solve_sector"] = operators.basic_metrics(solve_sector-solve_sector.conj().T, spectral=True)
    symmetry["independent_raw_sector_difference"] = operators.basic_metrics(
        raw_sector-independent_raw_sector, spectral=True)
    for name, record in symmetry.items():
        check(name+"_spectral_norm", record["spectral_norm"], RESIDUAL_TOL)
    if not all(c["passed"] for c in checks):
        raise ValueError("PRE_SOLVE_OPERATOR_CORRECTNESS_FAILED")

    energy, state, gap = canonical_ground(solve, indices)
    sector_state = state[indices]
    check("full_state_norm", abs(np.linalg.norm(state)-1), NORM_TOL)
    check("sector_state_norm", abs(np.linalg.norm(sector_state)-1), NORM_TOL)
    for name, matrix, psi in (("solve_full_residual", solve, state),
                              ("solve_sector_residual", solve_sector, sector_state),
                              ("raw_full_residual", raw, state),
                              ("raw_sector_residual", raw_sector, sector_state),
                              ("independent_raw_sector_residual", independent_raw_sector, sector_state)):
        check(name, np.linalg.norm(matrix@psi-energy*psi), RESIDUAL_TOL)
    outside = np.ones(len(state), dtype=bool)
    outside[indices] = False
    check("state_leakage_outside_sector", np.linalg.norm(state[outside]), NORM_TOL)
    pivot = int(np.argmax(abs(state)))
    check("phase_pivot_imaginary", abs(state[pivot].imag), NORM_TOL)
    checks.append(dict(name="phase_pivot_real_positive", passed=bool(state[pivot].real > 0)))
    check("phase_reapplication_l2", np.linalg.norm(phase_only(state, pivot)-state), NORM_TOL)
    check("global_phase_recovery_l2", np.linalg.norm(phase_only(np.exp(.371j)*state, pivot)-state), NORM_TOL)
    rayleigh = complex(np.vdot(state, raw@state)/np.vdot(state, state))
    check("raw_Rayleigh_energy_real", abs(rayleigh.real-energy), RESIDUAL_TOL)
    check("raw_Rayleigh_energy_imaginary", abs(rayleigh.imag), RESIDUAL_TOL)
    report = dict(energy_Ha=energy, same_sector_gap_Ha=gap, checks=checks,
                  symmetry=symmetry, solve_operator_hash=operators.array_hash(solve),
                  phase_pivot_Qiskit_index=pivot,
                  phase_policy="numpy first argmax(abs(full_Qiskit_vector)); multiply exp(-i angle(pivot)); L2 normalize (existing canonical_ground)",
                  state_norm=float(np.linalg.norm(state)), sector_state_norm=float(np.linalg.norm(sector_state)),
                  raw_Rayleigh=dict(real=rayleigh.real, imag=rayleigh.imag),
                  state_solve_calls=1, eigh_calls=1,
                  solver="h2_non_df_pilot.canonical_ground -> scipy.linalg.eigh on supplied sector; values[0], vectors[:,0]",
                  lowest_in="supplied Nalpha=Nbeta=2 sector only; not full Fock minimum",
                  numerical_Hermitianization="(raw H_phys + raw H_phys.conj().T)/2; target remains raw H_phys",
                  status="HPHYS_SECTOR_GROUND_STATE_CORRECTNESS_PASS" if all(c["passed"] for c in checks)
                         else "STOP_STATE_CORRECTNESS_FAILED")
    return state, sector_state, report


def run(hdf_path, public_output, private_output):
    from openfermion import MolecularData
    from .h4_molecular_inputs import spin_lift_unthresholded
    from .h2_non_df_pilot import reverse_bits
    from .df_hamiltonian import PhysicalSector

    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).isoformat()
    hdf_path, public_output, private_output = map(Path, (hdf_path, public_output, private_output))
    git = lambda *a: subprocess.check_output(["git", *a], cwd=ROOT)
    source_commit = git("rev-parse", "HEAD").decode().strip()
    new_paths = ["src/trotterlib/b2_hphys_ground_state.py", "scripts/run_b2_hphys_ground_state.py",
                 "docs/research/b2_hphys_ground_state_20261106/PLAN.md"]
    for path in new_paths:
        if git("show", source_commit+":"+path) != (ROOT/path).read_bytes():
            raise ValueError("EXECUTION_SOURCE_NOT_COMMITTED: "+path)
    prior_bytes = git("show", PRIOR_COMMIT+":"+PRIOR_PATH)
    prior = json.loads(prior_bytes)
    if prior["source_commit"] != PRIOR_SOURCE or not prior["status"].startswith("OPERATOR_CHECKS_PASS"):
        raise ValueError("PRIOR_OPERATOR_RECORD")
    reused = ["src/trotterlib/b2_original_df12_correspondence.py",
              "scripts/run_o1_m1_h4_operator_correctness.py",
              "src/trotterlib/h4_molecular_inputs.py", "src/trotterlib/h2_non_df_pilot.py",
              "src/trottertracks/resource_applicability/ax2b_independent_reference.py"]
    for path in reused:
        if operators.sha((ROOT/path).read_bytes()) != prior["identity"]["execution_SOURCE_hashes"][path]:
            raise ValueError("REUSED_SOURCE_CHANGED: "+path)
    for path, digest in prior["identity"]["generation_SOURCE_hashes"].items():
        if operators.sha(git("show", operators.GEN_COMMIT+":"+path)) != digest:
            raise ValueError("HISTORICAL_GENERATION_SOURCE_HASH: "+path)
    if operators.sha(hdf_path.read_bytes()) != operators.HDF_SHA:
        raise ValueError("FIXED_HDF5_HASH")
    df_bytes = git("show", operators.DF_COMMIT+":"+operators.DF_PATH)
    if operators.sha(df_bytes) != operators.DF_SHA:
        raise ValueError("FIXED_DF12_FILE_HASH")
    with np.load(io.BytesIO(df_bytes), allow_pickle=False) as saved:
        of_indices = np.asarray(saved["sector_basis_indices"], dtype=np.int64)
        df_metadata = json.loads(str(saved["metadata_json"].item()))
    if df_metadata["hamiltonian_hash"] != operators.DF_HASH:
        raise ValueError("DF12_METADATA_IDENTITY")
    expected = PhysicalSector.spin_sector(n_qubits=8, nelec_alpha=2, nelec_beta=2)
    if expected.dimension != 36 or not np.array_equal(of_indices, expected.basis_indices):
        raise ValueError("FIXED_SECTOR_BASIS")
    if operators.array_hash(of_indices) != df_metadata["sector"]["basis_indices_hash"]:
        raise ValueError("SAVED_SECTOR_HASH")

    public_output.mkdir(parents=True, exist_ok=False)
    if private_output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("PRIVATE_PAYLOAD_MUST_BE_OUTSIDE_PUBLIC_WORKTREE")
    private_output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (private_output/".gitignore").write_text("*\n")
    subprocess.run(["git", "-C", str(private_output), "check-ignore", "--quiet", "state.npz"], check=True)

    molecule = MolecularData(filename=str(hdf_path.with_suffix("")))
    one_mo, g_of = molecule.get_integrals()
    one, lifted_g = spin_lift_unthresholded(one_mo, g_of.transpose(0, 3, 1, 2))
    two, constant = lifted_g/2, float(molecule.nuclear_repulsion)
    for name, value in (("one_MO", one_mo), ("two_MO_OF", g_of),
                        ("MO", molecule.canonical_orbitals), ("AO_overlap", molecule.overlap_integrals)):
        if operators.array_hash(value) != prior["identity"]["original_integral_hashes"][name]:
            raise ValueError("FIXED_ORIGINAL_ARRAY_HASH: "+name)
    ref = operators._reference_module()
    a = ref.annihilators(8)
    raw = constant*np.eye(256, dtype=complex)+ref.one_body(one, a).toarray()+ref.quartic(two, a).toarray()
    raw_hash = operators.array_hash(raw)
    if raw_hash != RAW_HASH or raw_hash != prior["identity"]["operator_matrix_hashes"]["H_phys"]:
        raise ValueError("FIXED_RAW_HPHYS_OPERATOR_IDENTITY")
    permutation = reverse_bits(8)
    qiskit_indices = permutation[of_indices]
    independent = np.asarray(operators.occupation_original(constant, one, two, of_indices), dtype=complex)
    solve_started = time.perf_counter()
    state, sector_state, validation = solve_checked(raw, qiskit_indices, independent)
    solve_and_validation_seconds = time.perf_counter()-solve_started
    of_full = state[permutation]
    embedded = np.zeros(256, dtype=complex)
    embedded[qiskit_indices] = sector_state
    order_ok = (np.array_equal(of_full[of_indices], sector_state)
                and np.array_equal(of_full[permutation], state)
                and np.array_equal(embedded, state))
    validation["checks"].append(dict(name="OF_Qiskit_and_embedding_exact_roundtrip", passed=order_ok))
    if not order_ok:
        validation["status"] = "STOP_STATE_CORRECTNESS_FAILED"
    state_hash, sector_state_hash = operators.array_hash(state), operators.array_hash(sector_state)
    result = dict(schema_version="b2_hphys_sector_ground_state_v1", target="H_phys",
        state_role="original_H_ground_state", source_commit=source_commit, started_at_utc=timestamp,
        status=validation["status"], conditions=dict(model="linear_H4", distance_angstrom=1., basis="STO-3G",
            spatial_orbitals=4, spin_orbitals=8, nelec_alpha=2, nelec_beta=2, sector_dimension=36,
            full_vector_dimension=256, spin_order="2p=alpha,2p+1=beta", coefficient_cutoff=0.,
            nuclear_scalar_hex=constant.hex(), DF12_rank=12, B2_L_D=3, delta_window=None,
            state_preparation_cost_included=False),
        provenance=dict(HDF5_SHA256=operators.HDF_SHA, DF12_NPZ_SHA256=operators.DF_SHA,
            DF12_hamiltonian_hash=operators.DF_HASH, DF12_commit=operators.DF_COMMIT,
            prior_result_commit=PRIOR_COMMIT, prior_result_path=PRIOR_PATH,
            prior_result_SHA256=operators.sha(prior_bytes), prior_comparison_SOURCE=PRIOR_SOURCE,
            raw_H_phys_operator_hash=raw_hash, solve_operator_hash=validation["solve_operator_hash"],
            sector_basis_hash=operators.array_hash(of_indices), Qiskit_indices_hash=operators.array_hash(qiskit_indices),
            execution_SOURCE_hashes={p: operators.sha((ROOT/p).read_bytes()) for p in new_paths+reused},
            historical_generation_SOURCE=operators.GEN_COMMIT,
            historical_generation_SOURCE_hashes=prior["identity"]["generation_SOURCE_hashes"]),
        state_identity=dict(state_vector_hash=state_hash, sector_state_vector_hash=sector_state_hash,
            hash_definition=HASH_DEFINITION, state_dtype="<c16", index_dtype="<i8",
            full_vector_order="Qiskit little-endian", sector_vector_order="saved OpenFermion MSB sector basis order",
            phase_policy=validation["phase_policy"]),
        validation=validation, numerical_criteria=dict(state_norm=NORM_TOL, residual_Ha=RESIDUAL_TOL,
            Hermiticity_spectral_Ha=RESIDUAL_TOL, phase_L2=NORM_TOL, certified_interval=False),
        environment=dict(python=platform.python_version(), platform=platform.platform(),
            packages={n: importlib.metadata.version(n) for n in ("numpy", "scipy", "openfermion", "openfermionpyscf", "h5py", "qiskit")},
            cpu_affinity=len(os.sched_getaffinity(0)), gpu_used=False,
            algorithm="SciPy sparse CAR construction + single dense36 Hermitian eigh; default BLAS settings",
            BLAS_env={k: os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}),
        timings=dict(solve_and_validation_seconds=solve_and_validation_seconds),
        forbidden_work_executed=dict(SCF=0, integral_generation=0, DF_decomposition=0,
            other_Hamiltonian_state_solve=0, DF_state_use=0, simulation=0, circuit_build_compile=0,
            RPE_round_shot_cost=0, sampling=0, scientific_review=0),
        private_payload=dict(filename="state.npz", published=False), mandatory_stop=True)
    state_file = private_output/"state.npz"
    with os.fdopen(os.open(state_file, os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600), "wb") as stream:
        np.savez_compressed(stream, state_vector=state, sector_state_vector=sector_state,
            sector_basis_indices=of_indices, qiskit_full_vector_indices=qiskit_indices,
            energy=np.asarray(validation["energy_Ha"]),
            metadata_json=np.asarray(json.dumps(result, sort_keys=True, allow_nan=False)))
    with np.load(state_file, allow_pickle=False) as stored:
        for key, value in (("state_vector", state), ("sector_state_vector", sector_state),
                           ("sector_basis_indices", of_indices), ("qiskit_full_vector_indices", qiskit_indices)):
            if not np.array_equal(stored[key], value) or operators.array_hash(stored[key]) != operators.array_hash(value):
                raise ValueError("PRIVATE_SAVE_RELOAD_IDENTITY")
    result["private_payload"].update(file_SHA256=operators.sha(state_file.read_bytes()),
                                     save_reload_arrays_exact=True, file_mode=oct(state_file.stat().st_mode&0o777),
                                     directory_mode=oct(private_output.stat().st_mode&0o777))
    result["timings"]["total_seconds"] = time.perf_counter()-started
    if operators.sha(hdf_path.read_bytes()) != operators.HDF_SHA:
        raise ValueError("HDF5_CHANGED_DURING_EXECUTION")
    binding = dict(schema_version="b2_original_state_input_spec_v1", target="H_phys",
        state_role="original_H_ground_state", original_hamiltonian="H_phys:S0_array_hash:"+raw_hash,
        original_state="S0_array_hash:"+state_hash, df_hamiltonian=operators.DF_HASH,
        sector="Nalpha=2,Nbeta=2;OF_basis_S0_hash:"+operators.array_hash(of_indices),
        spin_orbital_order="2p=alpha,2p+1=beta;Qiskit full index=sum(n_q*2**q)",
        source=source_commit, state_preparation_included=False,
        NPZ_fields=dict(state_vector=[256], sector_state_vector=[36], sector_basis_indices=[36],
                        qiskit_full_vector_indices=[36], energy=[]),
        private_payload_SHA256=result["private_payload"]["file_SHA256"], state_hash_definition=HASH_DEFINITION,
        model_error_bound=None, energy_precision=None, success_probability=None, alias_branch_conditions=None,
        compiler=None, backend_hash=None, cost_metric=None,
        execution_status="INPUT_SPEC_ONLY; no RPEConnectionContext or scientific RPE execution")
    for path, value in ((public_output/"result.json", result), (public_output/"agent_b_input_spec.json", binding),
                         (private_output/"provenance.json", result)):
        with path.open("x") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return result
