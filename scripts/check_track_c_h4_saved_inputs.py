"""Compare saved, pre-DF H4 integrals with legacy DF/state; no SCF or DF build.

Run in a repository containing the fixed input objects:
    python3 scripts/check_track_c_h4_saved_inputs.py OUTPUT_DIRECTORY

Reuses the original DF excitation table/matvec and H4 spin lift, loading only
those definitions from fixed git objects so molecular/compilation runners and
their optional dependencies are never imported. Diagnostic tolerances come
from the saved H4 input receipt; they do not adopt an RPE pilot contract.
"""

import ast
from dataclasses import dataclass
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INPUT_REF = "e22d7718c47358ff650dbc04e8e72f6ab1c8bf41"
DF_REF = "4591b8d75a2a6a55c74a51225f0c61648023c8ee"
ORIGINAL = "artifacts/non_df_h4_pilot/2026-10-23/run01/input_run/input_generation/data/"
INPUTS = {
    "original": (INPUT_REF, ORIGINAL + "snapshot.npz",
                 "8a9c45429dab53cb2a4def7f38ccb2872b211011166423c281e92549b2deaf50"),
    "legacy_df": (DF_REF, "artifacts/rte_connected_cluster_cost_validation/"
                  "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz",
                  "13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b"),
    "legacy_state": (INPUT_REF, "artifacts/rpe_fixed_request_measurement_b/2026-11-03/run01/physical_state.npz",
                     "793300c86417f52569289ce86980c5f9a87292e130199355a0f4f7056225c9c4"),
}


def git_bytes(ref, path):
    return subprocess.check_output(["git", "show", f"{ref}:{path}"], cwd=ROOT)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selected_definitions(ref, path, names, namespace):
    source = git_bytes(ref, path)
    tree = ast.parse(source, filename=path)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return dict(commit=ref, path=path, sha256=sha(source), definitions=list(names))


def interaction_matrix(constant, one, two, basis, n=8):
    """CAR action: constant + h[p,q] a†p aq + W[p,q,r,s] a†p a†q ar as.

    OpenFermion big-endian determinant indices; no coefficient cutoff.
    """
    out = np.eye(len(basis), dtype=complex) * constant
    positions = {int(b): j for j, b in enumerate(basis)}
    terms = [(complex(one[p, q]), ((q, False), (p, True)))
             for p, q in np.argwhere(one != 0)]
    terms += [(complex(two[p, q, r, s]), ((s, False), (r, False), (q, True), (p, True)))
              for p, q, r, s in np.argwhere(two != 0)]
    for column, determinant in enumerate(basis):
        for coefficient, operators in terms:
            state, sign = int(determinant), 1
            for mode, create in operators:
                bit = 1 << (n - 1 - int(mode))
                if bool(state & bit) == create:
                    break
                before = ((1 << n) - 1) ^ ((1 << (n - int(mode))) - 1)
                sign *= -1 if (state & before).bit_count() % 2 else 1
                state ^= bit
            else:
                if state in positions:
                    out[positions[state], column] += coefficient * sign
    return out


def rotate_eri(eri, u):
    return np.einsum("ap,bq,cr,ds,abcd->pqrs", u, u, u, u, eri, optimize=True)


def main(output):
    output.mkdir(parents=True, exist_ok=False)
    provenance, arrays = {}, {}
    for name, (ref, path, expected) in INPUTS.items():
        data = git_bytes(ref, path)
        assert sha(data) == expected, f"Changed input: {name}"
        provenance[name] = dict(commit=ref, path=path, sha256=expected)
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            arrays[name] = {k: z[k] for k in z.files}
    receipt_bytes = git_bytes(INPUT_REF, ORIGINAL + "receipt.json")
    receipt = json.loads(receipt_bytes)
    provenance["original_receipt"] = dict(commit=INPUT_REF, path=ORIGINAL + "receipt.json", sha256=sha(receipt_bytes))
    assert receipt["snapshot_sha256"] == INPUTS["original"][2]
    assert receipt["artificial_only"] is False and receipt["num_qubits"] == 8
    assert receipt["basis"].lower() == "sto-3g"
    assert receipt["geometry"] == [["H", [0., 0., float(j)]] for j in range(4)]

    ns = dict(np=np, dataclass=dataclass, Any=Any, Sequence=Sequence, __name__=__name__)
    reused = [selected_definitions(DF_REF, "src/trotterlib/df_hamiltonian.py",
        ["DFHamiltonian", "PhysicalSector", "_ExcitationTable", "_before_mode_masks",
         "_build_excitation_table", "_apply_one_body", "_df_matvec_python"], ns)]
    reused.append(selected_definitions(INPUT_REF, "src/trotterlib/h4_molecular_inputs.py",
                                      ["spin_lift_unthresholded"], ns))
    original, df, saved = (arrays[k] for k in ("original", "legacy_df", "legacy_state"))
    basis, state = saved["basis_indices"], saved["state"]
    assert saved["n_qubits"].item() == 8 and saved["n_electrons"].item() == 4
    assert np.array_equal(basis, [i for i in range(256) if i.bit_count() == 4])
    assert df["hamiltonian_hash"].item() == saved["hamiltonian_hash"].item()
    assert len(df["lambdas"]) == 12
    for package in (original, df, saved):
        assert all(np.isfinite(a).all() for a in package.values() if a.dtype.kind in "fc")

    # The retained DF one-body includes the normal-ordering correction.
    g, lam = df["g_matrices"], df["lambdas"]
    bare_one = df["one_body"] + np.einsum("l,lpq,lqr->pr", lam, g, g)
    df_two = np.einsum("l,lps,lqr->pqrs", lam, g, g)
    target_spatial_two = 2 * df_two[::2, 1::2, 1::2, ::2]
    eh_source, vh_source = np.linalg.eigh(original["one_mo"])
    eh_target, vh_target = np.linalg.eigh(bare_one[::2, ::2])
    matrix_tol = receipt["gates"]["DF_full_H_agreement"]["tolerance"]
    state_tol = receipt["gates"]["state_residual"]["tolerance"]
    orbital_tol = receipt["gates"]["orbitals"]["tolerance"]
    # Core eigenvalues are nondegenerate. Enumerate their 16 real sign choices;
    # molecular symmetries can give multiple equivalent maps. Prefer near-I.
    candidates = []
    for signs in itertools.product((-1, 1), repeat=4):
        u = (vh_source * np.asarray(signs)) @ vh_target.conj().T
        eri = rotate_eri(original["eri_chemist"], u)
        error = float(np.max(abs(eri.transpose(0, 2, 3, 1) - target_spatial_two)))
        candidates.append((error, float(np.linalg.norm(u - np.eye(4))), signs, u, eri))
    viable = [c for c in candidates if c[0] <= orbital_tol]
    if not viable:
        raise ValueError("No supported orbital map; do not generate replacement inputs")
    fit = min(viable, key=lambda c: c[1])
    tensor_error, _, signs, u, eri = fit
    rotated_one = u.conj().T @ original["one_mo"] @ u
    assert max(np.max(abs(u.imag)), np.max(abs(rotated_one.imag)), np.max(abs(eri.imag))) < orbital_tol
    spin_one, spin_two_unscaled = ns["spin_lift_unthresholded"](rotated_one.real, eri.real)
    spin_two = spin_two_unscaled / 2
    constant = original["nuclear_repulsion"].item()
    original_in_legacy = interaction_matrix(constant, spin_one, spin_two, basis)

    sector = ns["PhysicalSector"](n_qubits=8, basis_indices=basis, n_electrons=4)
    hdf = ns["DFHamiltonian"](constant=df["constant"].item(), one_body=df["one_body"],
            lambdas=lam, g_matrices=tuple(g), metadata=json.loads(df["metadata_json"].item()))
    table = ns["_build_excitation_table"](sector)
    legacy = np.column_stack([ns["_df_matvec_python"](hdf, v, table, len(basis)) for v in np.eye(len(basis))])
    df_from_normal_order = interaction_matrix(df["constant"].item(), bare_one, df_two, basis)

    source_one, source_two = ns["spin_lift_unthresholded"](original["one_mo"], original["eri_chemist"])
    source_full = interaction_matrix(constant, source_one, source_two / 2, np.arange(256))
    reverse = np.array([int(f"{i:08b}"[::-1], 2) for i in range(256)])
    saved_source_full = original["non_DF_matrix"][np.ix_(reverse, reverse)]
    source_sector = source_full[np.ix_(basis, basis)]
    # Exterior-power map: target (legacy) determinants -> original saved MOs.
    uspin = np.zeros((8, 8), dtype=complex)
    uspin[::2, ::2] = u
    uspin[1::2, 1::2] = u
    occupations = [[q for q in range(8) if int(b) & (1 << (7 - q))] for b in basis]
    wedge = np.array([[np.linalg.det(uspin[np.ix_(a, b)]) for b in occupations] for a in occupations])
    source_state = original["state"][reverse][basis]
    eigenvalues, eigenvectors = np.linalg.eigh((original_in_legacy + original_in_legacy.conj().T) / 2)
    energy, ground = float(eigenvalues[0]), eigenvectors[:, 0]
    ground *= np.exp(-1j * np.angle(ground[np.argmax(abs(ground))]))
    norm2 = lambda x: float(np.linalg.norm(x, 2))
    metrics = {
        "constant_difference_Ha": abs(constant - df["constant"].item()),
        "core_eigenvalue_max_difference_Ha": float(np.max(abs(eh_source - eh_target))),
        "core_min_eigenvalue_spacing_Ha": float(min(np.diff(eh_source))),
        "direct_unaligned_one_body_max_difference_Ha": float(np.max(abs(original["spin_one"] - bare_one))),
        "direct_unaligned_two_body_max_difference_Ha": float(np.max(abs(original["spin_two"] - df_two))),
        "orbital_orthogonality_norm2": norm2(u.conj().T @ u - np.eye(4)),
        "orbital_distance_to_identity_norm2": norm2(u - np.eye(4)),
        "rotated_one_body_max_difference_Ha": float(np.max(abs(spin_one - bare_one))),
        "rotated_two_body_max_difference_Ha": float(np.max(abs(spin_two - df_two))),
        "tensor_fit_max_difference_Ha": tensor_error,
        "original_spin_one_rebuild_max_difference_Ha": float(np.max(abs(source_one - original["spin_one"]))),
        "original_spin_two_rebuild_max_difference_Ha": float(np.max(abs(source_two / 2 - original["spin_two"]))),
        "original_full_saved_matrix_rebuild_fro_Ha": float(np.linalg.norm(source_full - saved_source_full)),
        "df_matvec_normal_order_agreement_norm2_Ha": norm2(legacy - df_from_normal_order),
        "physical_H_vs_legacy_DF_sector_norm2_Ha": norm2(original_in_legacy - legacy),
        "physical_H_hermiticity_norm2_Ha": norm2(original_in_legacy - original_in_legacy.conj().T),
        "DF_hermiticity_norm2_Ha": norm2(legacy - legacy.conj().T),
        "wedge_unitarity_norm2": norm2(wedge.conj().T @ wedge - np.eye(70)),
        "original_basis_intertwining_norm2_Ha": norm2(source_sector @ wedge - wedge @ original_in_legacy),
        "ground_energy_Ha": energy,
        "ground_gap_Ha": float(eigenvalues[1] - eigenvalues[0]),
        "ground_reference_residual_l2_Ha": norm2(original_in_legacy @ ground - energy * ground),
        "saved_state_norm_error": abs(norm2(state) - 1),
        "saved_state_original_energy_Ha": float(np.vdot(state, original_in_legacy @ state).real),
        "saved_state_original_ground_energy_difference_Ha": abs(float(np.vdot(state, original_in_legacy @ state).real) - energy),
        "saved_state_original_ground_residual_l2_Ha": norm2(original_in_legacy @ state - energy * state),
        "saved_state_DF_residual_l2_Ha": norm2(legacy @ state - np.vdot(state, legacy @ state) * state),
        "saved_state_original_ground_infidelity_abs": abs(1 - abs(np.vdot(ground, state)) ** 2),
        "saved_source_state_norm_error": abs(norm2(source_state) - 1),
        "saved_source_state_mapped_infidelity_abs": abs(1 - abs(np.vdot(source_state, wedge @ state)) ** 2),
        "ground_vs_saved_FCI_energy_difference_Ha": abs(energy - original["fci_energy"].item()),
    }
    gates = {
        "operator_agreement": metrics["physical_H_vs_legacy_DF_sector_norm2_Ha"] <= matrix_tol,
        "original_source_rebuild": metrics["original_full_saved_matrix_rebuild_fro_Ha"] <= matrix_tol,
        "DF_convention": metrics["df_matvec_normal_order_agreement_norm2_Ha"] <= matrix_tol,
        "basis_intertwining": metrics["original_basis_intertwining_norm2_Ha"] <= matrix_tol,
        "orbital_map": metrics["orbital_orthogonality_norm2"] <= orbital_tol and tensor_error <= orbital_tol,
        "ground_reference": metrics["ground_reference_residual_l2_Ha"] <= state_tol and metrics["ground_gap_Ha"] > state_tol,
        "saved_state_ground": metrics["saved_state_original_ground_residual_l2_Ha"] <= state_tol
            and metrics["saved_state_original_ground_infidelity_abs"] <= matrix_tol,
        "saved_source_state": metrics["saved_source_state_mapped_infidelity_abs"] <= matrix_tol,
        "state_norm": metrics["saved_state_norm_error"] <= receipt["gates"]["state_norm"]["tolerance"],
        "hermiticity": metrics["physical_H_hermiticity_norm2_Ha"] <= state_tol,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    np.savez(output / "ground_reference.npz", state=ground, basis_indices=basis,
             original_H_in_legacy_basis=original_in_legacy, energy=energy,
             orbital_map_source_to_legacy=u, eigenvalues=eigenvalues)
    result = dict(status="PASS" if all(gates.values()) else "FAIL", host=socket.gethostname(),
        scope="saved H4 original-integral/legacy-DF/state numerical check only",
        input_provenance=provenance, reused_source=reused,
        implementation_sha256=sha(Path(__file__).read_bytes()),
        environment=dict(python=platform.python_version(), numpy=np.__version__),
        physical_sector=dict(n_qubits=8, n_electrons=4, dimension=70, ordering="OpenFermion big-endian, sorted number-sector indices"),
        hamiltonian_hash=df["hamiltonian_hash"].item(), state_fingerprint=saved["fingerprint"].item(),
        orbital_map=dict(method="nondegenerate bare one-body eigensystems + 16 sign choices; tensor check; near-identity tie-break",
                         signs=list(signs), maps_within_orbital_tolerance=len(viable), matrix=u.real.tolist()),
        diagnostic_tolerances=dict(matrix_agreement_Ha=matrix_tol, state_residual_Ha=state_tol, orbital=orbital_tol,
                                  source="original saved H4 receipt, not RPE contract adoption"),
        metrics=metrics, gates=gates,
        ground_reference_sha256=sha((output / "ground_reference.npz").read_bytes()),
        no_new_SCF_or_DF=True, circuits_built_or_compiled=0, pilots_executed=0,
        limitations=["Numerical equivalence after an inferred orbital map, not byte identity with the legacy pre-DF integral file.",
                     "No new PySCF FCI call; FCI comparison uses the existing saved scalar.",
                     "No PF/RPE signal, shot, synthesis or resource-cost evaluation; pilot contract remains unadopted."],
        mandatory_STOP=True)
    (output / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], metrics=metrics, gates=gates), indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_track_c_h4_saved_inputs.py NEW_OUTPUT_DIRECTORY")
    main(Path(sys.argv[1]))
