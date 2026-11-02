"""H4 molecular input implementation; called only AFTER authorization/claim.

No molecular data are generated on import. The H2 function remains unchanged.
This keeps its integral conventions, checks and tolerance values, generalized
explicitly to four spatial orbitals and Nalpha=Nbeta=2 (dimension 36).
"""
import math
import json
import numpy as np
from scipy.linalg import expm
from .h2_non_df_pilot import check, reverse_bits, canonical_ground
from .fixed_signal_evaluation import FixedSignalConfig, prepare_signal_adapter
from .pauli_signal_adapter import PauliHamiltonian, array_hash
from .non_df_h4_readiness import structural_prefix
from .isolated_qiskit_execution import sha, save, read


def spin_lift_unthresholded(one, eri):
    """Same convention as Pauli.from_spatial_integrals, with zero cutoff."""
    n=len(one);spin_one=np.zeros((2*n,2*n));spin_two=np.zeros((2*n,)*4)
    for a in range(2):
        spin_one[a::2,a::2]=one
        for b in range(2):
            spin_two[a::2,b::2,b::2,a::2]=eri.transpose(0,2,3,1)
    return spin_one,spin_two


def dimension_gate(mo):
    a = np.asarray(mo)
    if a.shape != (4,4) or not np.isfinite(a).all():
        raise ValueError("H4_FOUR_SPATIAL_EIGHT_SPIN_REQUIRED")
    from .df_hamiltonian import PhysicalSector
    sector = PhysicalSector.spin_sector(n_qubits=8, nelec_alpha=2, nelec_beta=2)
    if sector.dimension != 36:
        raise ValueError("H4_PHYSICAL_SECTOR_DIMENSION")
    return sector


def resolve_prefixes(pauli, df):
    from .df_partial_randomized_pf import rank_df_fragments
    from .fixed_signal_evaluation import DFSignalAdapter
    if pauli.num_qubits != 8 or df.n_qubits != 8:
        raise ValueError("H4_HAMILTONIAN_DIMENSION")
    ranked = rank_df_fragments(df)
    p = structural_prefix([abs(c) for _,c in pauli.terms])
    d = structural_prefix([x.weight for x in ranked])
    prep = DFSignalAdapter(df, 'PR', d, .1, 4, 4)
    if not prep.randomized or prep.lambda_r <= 0:
        raise ValueError("H4_POSITIVE_RANDOM_TAIL_REQUIRED")
    return dict(pauli_jw=dict(L_D=p, retained_labels=[s for s,c in pauli.terms[:p]],
        tail_labels=[s for s,c in pauli.terms[p:]], nonidentity_term_count=len(pauli.terms)),
        df_native=dict(L_D=d, rank=df.n_blocks,
            retained_indices=list(prep.preparation.deterministic_fragment_indices),
            generation_indices=[f.original_index for f in ranked], one_body_always_retained=True))


def build_h4_inputs(destination, plan):
    """One authorized SCF + one DF from the exact same generated integrals."""
    from pyscf import ao2mo, fci, gto, lib, scf
    from openfermion import InteractionOperator, get_fermion_operator, get_sparse_operator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from .df_hamiltonian import PhysicalSector, df_hamiltonian_from_integrals, df_linear_operator

    settings = plan["system"]
    destination.mkdir()
    # Only this process's PySCF scratch/checkpoint/log destinations are changed.
    lib.param.TMPDIR = str(destination)
    molecule = gto.Mole()
    molecule.atom = settings["geometry_Angstrom"]
    molecule.unit = "Angstrom"
    molecule.basis = settings["basis"]
    molecule.charge = settings["charge"]
    molecule.spin = settings["spin_2S"]
    molecule.symmetry = settings["symmetry"]
    molecule.output = str(destination/"pyscf.log")
    molecule.verbose = 4
    molecule.build()
    mf = scf.RHF(molecule)
    mf.conv_tol = settings["SCF"]["conv_tol"]
    mf.max_cycle = settings["SCF"]["max_cycle"]
    mf.init_guess = settings["SCF"]["init_guess"]
    mf.chkfile = str(destination/"scf.chk")
    hf_energy = float(mf.kernel())
    if not mf.converged:
        raise ValueError("SCF_NOT_CONVERGED_NO_RETRY")
    mo = np.asarray(mf.mo_coeff)
    dimension_gate(mo)
    if np.iscomplexobj(mo) and np.max(np.abs(mo.imag)) != 0:
        raise ValueError("H4_REAL_MO_REQUIRED")
    mo = np.asarray(mo.real, dtype=float)
    one = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.kernel(molecule, mo), mo.shape[1])
    enuc = float(molecule.energy_nuc())
    gates = dict(orbitals=check(np.linalg.norm(mo.T@mf.get_ovlp()@mo-np.eye(len(one))),
                               plan["numerical"]["orthonormality"], "MO_ORTHONORMALITY"))
    metadata = dict(model_id="H4_1.00A_STO-3G_FIXED_PILOT", molecule_type=4,
                    distance=1., basis="STO-3G", charge=0, multiplicity=1,
                    geometry=settings["geometry_Angstrom"], hf_energy=hf_energy,
                    SCF_converged=True, input_origin="new_one_shot_SCF_same_integrals_for_Pauli_and_DF")
    pauli = PauliHamiltonian.from_spatial_integrals(one, eri, nuclear_repulsion=enuc,
                                                   model_id=metadata["model_id"], metadata=metadata)
    # Independent legacy OpenFermion spin lift, preserving its original behavior.
    legacy_spin_one, legacy_spin_two = spinorb_from_spatial(one, eri.transpose(0, 2, 3, 1))
    interaction = InteractionOperator(enuc, legacy_spin_one, legacy_spin_two/2)
    operator = jordan_wigner(get_fermion_operator(interaction))
    permutation = reverse_bits(pauli.num_qubits)
    legacy = get_sparse_operator(operator, n_qubits=pauli.num_qubits).toarray()[np.ix_(permutation, permutation)]
    full = pauli.sparse_operator().toarray()
    # The independent legacy crosscheck above retains its original behavior.
    # Production DF uses exactly the same UNTHRESHOLDED MO tensor as Pauli.
    # OpenFermion's spinorb_from_spatial has a hidden EQ_TOLERANCE truncation.
    spin_one,spin_two=spin_lift_unthresholded(one,eri)
    df = df_hamiltonian_from_integrals(constant=enuc, one_body=spin_one, two_body=spin_two/2,
                                      df_rank=None, df_tol=1e-8, metadata=metadata)
    if df.n_qubits != 8 or df.n_blocks < 2:
        raise ValueError("H4_DF_LAYOUT_OR_NONEMPTY_TAIL")
    df_full = prepare_signal_adapter(df, FixedSignalConfig("df_native", T=plan["evolution"]["T"], q=plan["evolution"]["q"])).operators()[0].toarray()
    tolerance = plan["numerical"]
    gates["Hermiticity"] = check(np.linalg.norm(full-full.conj().T, 2), tolerance["Hermiticity"], "HERMITICITY")
    gates["legacy_JW_agreement"] = check(np.linalg.norm(full-legacy, 2), tolerance["Hamiltonian_operator_agreement"], "PAULI_LEGACY_JW")
    gates["DF_full_H_agreement"] = check(np.linalg.norm(full-df_full, 2), tolerance["Hamiltonian_operator_agreement"], "DF_NON_DF_FULL_H")
    sector = PhysicalSector.spin_sector(n_qubits=8, nelec_alpha=2, nelec_beta=2)
    if sector.dimension != 36:
        raise ValueError("H4_PHYSICAL_SECTOR_DIMENSION")
    indices = permutation[sector.basis_indices]
    energy, state, gap = canonical_ground(full, indices)
    df_energy, df_state, df_gap = canonical_ground(df_full, indices)
    for name, error, limit in [
        ("state_norm", abs(np.linalg.norm(state)-1), tolerance["state_norm"]),
        ("state_residual", np.linalg.norm(full@state-energy*state), tolerance["state_residual"]),
        ("DF_state_energy", abs(energy-df_energy), tolerance["state_energy_agreement"]),
        ("DF_state_overlap", abs(1-abs(np.vdot(state, df_state))), tolerance["state_energy_agreement"]),
    ]:
        gates[name] = check(error, limit, name)
    production, counter = df_linear_operator(df, sector, backend="python")
    df_sector = np.column_stack([production@v for v in np.eye(sector.dimension)])
    gates["existing_DF_sector_operator"] = check(np.linalg.norm(df_sector-df_full[np.ix_(indices, indices)], 2),
        tolerance["Hamiltonian_operator_agreement"], "EXISTING_DF_OPERATOR")
    fci_energy, ci = fci.direct_spin1.kernel(one, eri, len(one), (2, 2), tol=1e-12,
                                            max_cycle=100, ecore=enuc)
    gates["PySCF_FCI_energy"] = check(abs(float(fci_energy)-energy), tolerance["state_energy_agreement"], "FCI_ENERGY")
    T = plan["evolution"]["T"]
    z = complex(np.vdot(state, expm(-1j*T*full)@state))
    zd = complex(np.vdot(state, expm(-1j*T*df_full)@state))
    gates["DF_exact_reference_signal"] = check(abs(z-zd), tolerance["reference_signal_agreement"], "DF_REFERENCE_SIGNAL")
    with (destination/"snapshot.npz").open("xb") as stream:
        np.savez(stream, mo_coeff=mo, one_mo=one, eri_chemist=eri, nuclear_repulsion=enuc,
                 spin_one=spin_one, spin_two=spin_two/2, df_one_body=df.one_body,
                 df_lambdas=df.lambdas, df_g_matrices=np.asarray(df.g_matrices),
                 state=state, DF_state=df_state, non_DF_matrix=full, DF_matrix=df_full,
                 physical_qiskit_indices=indices, energy=energy, fci_energy=fci_energy,
                 fci_vector=ci, hf_energy=hf_energy)
    receipt = dict(status="PASS", gates=gates, geometry=settings["geometry_Angstrom"], basis="STO-3G",
                   energy=energy, df_energy=df_energy, fci_energy=float(fci_energy), hf_energy=hf_energy,
                   gap=gap, df_gap=df_gap, num_qubits=8, num_nonidentity_Pauli_terms=len(pauli.terms),
                   state_hash=array_hash(state), df_state_hash=array_hash(df_state),
                   Pauli_hamiltonian_hash=pauli.hamiltonian_hash, DF_metadata=df.metadata,
                   sector_indices=indices.tolist(), sector_operator_actions=counter["count"],
                   SCF_calls=1, DF_calls=1, FCI_crosscheck_calls=1,
                   independent_reference_signal=dict(real=z.real, imag=z.imag),
                   DF_independent_reference_signal=dict(real=zd.real, imag=zd.imag),
                   same_L_D_is_same_partition=False)
    receipt.update(physical_sector_dimension=36, spatial_orbitals=4, spin_orbitals=8,
        prefixes=resolve_prefixes(pauli, df), all_gates_passed=True)
    return pauli, df, state, receipt
