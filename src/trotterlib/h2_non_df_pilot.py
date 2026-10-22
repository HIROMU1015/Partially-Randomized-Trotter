"""Predeclared H2 non-DF fixed-signal pilot and small-system cross-checks.

No science runs on import. The one-shot runner fixes SOURCE and the contract
before calling build_h2_inputs or evaluate_case. No search or retry path.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
import hashlib
import math

import numpy as np
from scipy.linalg import eigh, expm

from .fixed_signal_accounting import AXES, fixed_signal_shots
from .fixed_signal_evaluation import (
    FixedSignalConfig, _statistics, evaluate_fixed_signal, prepare_signal_adapter,
)
from .pauli_signal_adapter import PauliHamiltonian, array_hash, sparse_mean_action


def contract():
    return dict(
        schema="h2_non_df_fixed_signal_pilot_contract_v1",
        system=dict(molecule="H2", geometry_Angstrom=[["H", [0., 0., 0.]], ["H", [0., 0., 1.]]],
                    basis="sto-3g", charge=0, spin_2S=0, symmetry=False,
                    orbital_policy="PySCF canonical real RHF MOs; no active-space/frozen-core",
                    SCF=dict(conv_tol=1e-12, max_cycle=100, init_guess="minao"),
                    integral_policy="real orthonormal spatial MO; chemist (pq|rs); unthresholded supplied arrays",
                    spin_order="2*p+alpha(0)/beta(1)", qubit_order="qiskit_little_endian",
                    state_policy="exact lowest eigenvector in Nalpha=Nbeta=1; largest amplitude positive real",
                    reference="each representation's own full-H on the same non-DF physical-sector state"),
        DF=dict(rank_policy="tol_only_1e-8_no_final_rank", expected_rank=3,
                generation_order="OpenFermion low_rank_two_body_decomposition",
                split_order="existing weight_ranked_prefix", PR_L_D=1,
                identity_policy="extract_identity_phase", coefficient_atol=0.),
        Pauli=dict(coefficient_cutoff=0., split_order="descending_abs_then_label",
                   PR_L_D_policy="floor(nonidentity_term_count/2)", identity="retained_once",
                   DF_rank=None),
        evaluation=dict(T=.8, q=4, delta=.2, r=2, K=4, product_formula="S2",
                        methods=["PF", "PR"], representations=["pauli_jw", "df_native"],
                        epsilon_sig=[.05, .01, .005, .001, .0005],
                        epsilon_axis_policy="epsilon_sig/sqrt(2)", alpha_axis=.025,
                        statistical_model="existing fresh-IID bounded +/-1 corrected by finite-RTE B; Hoeffding sufficient N",
                        state_preparation_cost=0., state_preparation_excluded=True,
                        PR_cost_samples=64, adaptive_sampling=False,
                        seed_rule="sha256(H2_NON_DF_PILOT_20261022/<representation>/<method>)[:8] little-endian",
                        cost_uncertainty="paired axes; nominal Student-t 95%; no population or rare-event guarantee",
                        compiled_signal_check="every sampled full cosine/sine wrapper",
                        independent_trajectory_operator_checks=3),
        compiler=dict(basis_gates=["rz", "sx", "x", "cx"], optimization_level=1,
                      transpiler_seed=17, coupling_map=None, backend=None,
                      cost_scope="full Hadamard wrapper ordinary diag(I,U), phases and measurement; no preparation"),
        numerical=dict(Hermiticity=1e-10, orthonormality=1e-10,
                       Hamiltonian_operator_agreement=1e-9, state_energy_agreement=1e-9,
                       state_residual=1e-10, state_norm=1e-12,
                       reference_signal_agreement=1e-10, law_mean_agreement=1e-10,
                       trajectory_unitary_agreement=1e-9, compiled_probability_agreement=1e-9,
                       allowance_policy="existing empirical_allowance formula; no certification",
                       allowance_settings=dict(roundoff_multiplier=64., safety_factor=10., floor=1e-12),
                       maximum_u_over_headroom=.01,
                       failure_policy="any technical gate failure stops run; ineligible precision row remains null without changing parameters"),
        execution=dict(run_namespace="artifacts/non_df_h2_pilot/2026-10-22/run01",
                       one_shot=True, automatic_retry=False, resume=False,
                       new_SCF_calls=1, new_DF_decompositions=1, FCI_crosscheck_calls=1,
                       fixed_evaluations=4, candidate_search=False, new_fits=False,
                       H4_or_larger=False, new_scientific_model=False, scientific_review=False,
                       shared_environment_changes=False, push=False, mandatory_STOP=True),
        environment={"numpy": "1.26.4", "scipy": "1.14.1", "qiskit": "1.3.0",
                     "qiskit-nature": "0.7.2", "openfermion": "1.6.1",
                     "openfermionpyscf": "0.5", "pyscf": "2.7.0"},
    )


def check(error, tolerance, name):
    value = float(error)
    if not math.isfinite(value) or value > tolerance:
        raise ValueError(f"{name}: {value} > {tolerance}")
    return dict(error=value, tolerance=tolerance, PASS=True)


def reverse_bits(n):
    return np.asarray([int(f"{i:0{n}b}"[::-1], 2) for i in range(1 << n)], dtype=int)


def canonical_ground(matrix, indices):
    values, vectors = eigh(matrix[np.ix_(indices, indices)])
    vector = np.zeros(len(matrix), dtype=complex)
    vector[indices] = vectors[:, 0]
    pivot = int(np.argmax(np.abs(vector)))
    vector *= np.exp(-1j*np.angle(vector[pivot]))
    vector /= np.linalg.norm(vector)
    return float(values[0]), vector, float(values[1]-values[0])


def pauli_matrix(label):
    matrices = dict(I=np.eye(2), X=np.array([[0, 1], [1, 0]]),
                    Y=np.array([[0, -1j], [1j, 0]]), Z=np.diag([1, -1]))
    result = np.ones((1, 1), dtype=complex)
    for axis in label:
        result = np.kron(result, matrices[axis])
    return result


def build_h2_inputs(destination, plan):
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
    mo = np.asarray(mf.mo_coeff, dtype=float)
    one = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.kernel(molecule, mo), mo.shape[1])
    enuc = float(molecule.energy_nuc())
    gates = dict(orbitals=check(np.linalg.norm(mo.T@mf.get_ovlp()@mo-np.eye(len(one))),
                               plan["numerical"]["orthonormality"], "MO_ORTHONORMALITY"))
    metadata = dict(model_id="H2_1.00A_STO-3G_NON_DF_PILOT", molecule_type=2,
                    distance=1., basis="STO-3G", charge=0, multiplicity=1,
                    geometry=settings["geometry_Angstrom"], hf_energy=hf_energy,
                    SCF_converged=True, input_origin="new_one_shot_SCF_same_integrals_for_Pauli_and_DF")
    pauli = PauliHamiltonian.from_spatial_integrals(one, eri, nuclear_repulsion=enuc,
                                                   model_id=metadata["model_id"], metadata=metadata)
    # Independent legacy OpenFermion spin lift, preserving its original behavior.
    spin_one, spin_two = spinorb_from_spatial(one, eri.transpose(0, 2, 3, 1))
    interaction = InteractionOperator(enuc, spin_one, spin_two/2)
    operator = jordan_wigner(get_fermion_operator(interaction))
    permutation = reverse_bits(pauli.num_qubits)
    legacy = get_sparse_operator(operator, n_qubits=pauli.num_qubits).toarray()[np.ix_(permutation, permutation)]
    full = pauli.sparse_operator().toarray()
    df = df_hamiltonian_from_integrals(constant=enuc, one_body=spin_one, two_body=spin_two/2,
                                      df_rank=None, df_tol=1e-8, metadata=metadata)
    if df.n_blocks != plan["DF"]["expected_rank"]:
        raise ValueError("DF_EXPECTED_RANK_MISMATCH")
    df_full = prepare_signal_adapter(df, FixedSignalConfig("df_native", T=.8, q=4)).operators()[0].toarray()
    tolerance = plan["numerical"]
    gates["Hermiticity"] = check(np.linalg.norm(full-full.conj().T, 2), tolerance["Hermiticity"], "HERMITICITY")
    gates["legacy_JW_agreement"] = check(np.linalg.norm(full-legacy, 2), tolerance["Hamiltonian_operator_agreement"], "PAULI_LEGACY_JW")
    gates["DF_full_H_agreement"] = check(np.linalg.norm(full-df_full, 2), tolerance["Hamiltonian_operator_agreement"], "DF_NON_DF_FULL_H")
    sector = PhysicalSector.spin_sector(n_qubits=4, nelec_alpha=1, nelec_beta=1)
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
    fci_energy, ci = fci.direct_spin1.kernel(one, eri, len(one), (1, 1), tol=1e-12,
                                            max_cycle=100, ecore=enuc)
    gates["PySCF_FCI_energy"] = check(abs(float(fci_energy)-energy), tolerance["state_energy_agreement"], "FCI_ENERGY")
    T = plan["evaluation"]["T"]
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
                   gap=gap, df_gap=df_gap, num_qubits=4, num_nonidentity_Pauli_terms=len(pauli.terms),
                   state_hash=array_hash(state), df_state_hash=array_hash(df_state),
                   Pauli_hamiltonian_hash=pauli.hamiltonian_hash, DF_metadata=df.metadata,
                   sector_indices=indices.tolist(), sector_operator_actions=counter["count"],
                   SCF_calls=1, DF_calls=1, FCI_crosscheck_calls=1,
                   independent_reference_signal=dict(real=z.real, imag=z.imag),
                   DF_independent_reference_signal=dict(real=zd.real, imag=zd.imag),
                   same_L_D_is_same_partition=False)
    return pauli, df, state, receipt


def dense_law_mean(adapter, *, q):
    """Independent dense Horner reference, H2 only, no sampled estimate."""
    full, blocks, tail, scalar = adapter.operators()
    n = full.shape[0]
    unit = np.eye(n, dtype=complex)
    for block in blocks:
        unit = expm(-.5j*adapter.step_time*block.toarray())@unit
    if adapter.randomized:
        x = -1j*adapter.step_time/adapter.r*tail.toarray()
        poly = np.eye(n, dtype=complex)
        for degree in range(adapter.K+1, 0, -1):
            poly = np.eye(n)+x@poly/degree
        unit = np.linalg.matrix_power(poly, adapter.r)@unit
    for block in reversed(blocks):
        unit = expm(-.5j*adapter.step_time*block.toarray())@unit
    unit *= np.exp(-1j*scalar*adapter.step_time)
    return np.linalg.matrix_power(unit, q)


def seed_for(representation, method):
    value = f"H2_NON_DF_PILOT_20261022/{representation}/{method}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def case_config(hamiltonian, representation, method, plan):
    e = plan["evaluation"]
    ld = (len(hamiltonian.terms)//2 if representation == "pauli_jw" else plan["DF"]["PR_L_D"]) if method == "PR" else None
    return FixedSignalConfig(representation, method=method, T=e["T"], q=e["q"], r=e["r"], K=e["K"],
                             L_D=ld, seed=seed_for(representation, method), cost_samples=e["PR_cost_samples"])


def numerical_evidence(adapter, state, config, plan):
    from scipy.sparse.linalg import expm_multiply
    full = adapter.operators()[0].toarray()
    ref_dense = complex(np.vdot(state, expm(-1j*config.T*full)@state))
    ref_sparse = complex(np.vdot(state, expm_multiply(-1j*config.T*adapter.operators()[0], state)))
    mean = dense_law_mean(adapter, q=config.q)
    corrected = complex(np.vdot(state, mean@state))
    raw = complex(np.vdot(state, sparse_mean_action(adapter, state, q=config.q)))
    logB = config.q*adapter.log_B_step
    discrepancy = abs(raw*math.exp(logB)-corrected)
    closure = abs(raw-math.exp(-logB)*corrected)
    tol = plan["numerical"]
    gates = dict(reference=check(abs(ref_dense-ref_sparse), tol["reference_signal_agreement"], "DENSE_SPARSE_REFERENCE"),
                 law_mean=check(discrepancy, tol["law_mean_agreement"], "HORNER_FORWARD_LAW_MEAN"),
                 raw_normalization=check(closure, tol["law_mean_agreement"], "RAW_CORRECTED_NORMALIZATION"))
    # Existing empirical_allowance formula, with a declared conservative count
    # of scalar-operation proxies for these tiny matrix computations.
    work = config.q*(2*len(adapter.operators()[1])*len(state)**3 +
                     config.r*(config.K+2)*len(state)**3)+len(state)**3
    maximum = max(1., float(np.linalg.norm(mean, 2)), float(np.linalg.norm(mean@state)))
    s = tol["allowance_settings"]
    rounding = s["roundoff_multiplier"]*np.finfo(float).eps*(1+work)*maximum
    agreement = abs(ref_dense-ref_sparse)+discrepancy+closure
    u = s["safety_factor"]*max(s["floor"], rounding, agreement)
    du = s["safety_factor"]*max(s["floor"], rounding, closure)
    return dict(gates=gates, corrected=dict(real=corrected.real, imag=corrected.imag),
                reference=dict(real=ref_dense.real, imag=ref_dense.imag),
                allowance=dict(cosine=u, sine=u), log_B_margin=du,
                rounding_proxy=rounding, work=work, maximum_norm=maximum,
                agreement=agreement, certified=False, kind="EMPIRICAL",
                scope="existing allowance formula, declared dense/sparse operation proxy; local diagnostic only")


def precision_accounting(result, epsilon, plan):
    b = result["signal"]["signed_bias"]
    budget = fixed_signal_shots(epsilon_sig=epsilon,
        bias=dict(cosine=abs(b["real"]), sine=abs(b["imag"])),
        allowance={a:result["shots"]["axes"][a]["allowance"] for a in AXES},
        log_B_upper=result["normalization"]["log_B_upper"], evidence_kind="EMPIRICAL",
        evidence_ref=result["shots"]["evidence_ref"],
        alpha_axes={a:plan["evaluation"]["alpha_axis"] for a in AXES})
    for a in AXES:
        row = budget["axes"][a]
        if row["shots"] is not None and row["allowance"]/row["headroom"] > plan["numerical"]["maximum_u_over_headroom"]:
            row.update(status="UNDETERMINED", shots=None)
    budget["usable"] = all(budget["axes"][a]["shots"] is not None for a in AXES)
    budget["N_total"] = sum(budget["axes"][a]["shots"] for a in AXES) if budget["usable"] else None
    total = None
    if budget["usable"]:
        values = [sum(budget["axes"][a]["shots"]*row["RZ"][a] for a in AXES)
                  for row in result["circuit_cost"]["samples"]]
        total = _statistics(values)
        if result["circuit_cost"]["kind"] == "deterministic":
            total.update(standard_error=0., nominal95_halfwidth=0.)
        expected = sum(budget["axes"][a]["shots"]*Fraction(result["circuit_cost"]["axes"][a]["mean_exact"]) for a in AXES)
        if Fraction(total["mean_exact"]) != expected:
            raise ValueError("SAVED_PRECISION_G_IDENTITY")
    return dict(epsilon_sig=epsilon, shots=budget, total_RZ=total)


def evaluate_case(hamiltonian, state, representation, method, plan, provenance):
    config = case_config(hamiltonian, representation, method, plan)
    adapter = prepare_signal_adapter(hamiltonian, config)
    evidence = numerical_evidence(adapter, state, config, plan)
    result = evaluate_fixed_signal(hamiltonian, state, config,
        epsilon_sig=plan["evaluation"]["epsilon_sig"][0],
        numerical_allowance=evidence["allowance"], log_B_margin=evidence["log_B_margin"],
        evidence_kind="EMPIRICAL", evidence_ref=f"{representation}_{method}/numerical_evidence.json",
        state_metadata=dict(method=plan["system"]["state_policy"], molecule="H2", physical_sector="Nalpha=Nbeta=1"),
        source_provenance=provenance, verify_compiled=True)
    # Record limited molecular execution scope without upgrading the general
    # library's artificial-only validation claims or certifying N.
    result["pilot_validation"] = dict(molecular_system="H2", fixed_case_only=True,
        numerical_certificate=False, scientific_superiority_claimed=False,
        existing_DF_experiment_cohort_reproduced=False, new_Monte_Carlo_cohort=True)
    rows = [precision_accounting(result, eps, plan) for eps in plan["evaluation"]["epsilon_sig"]]
    if not rows[0]["shots"]["usable"]:
        # Precision is a result, not a reason to change settings or retry.
        result["primary_precision_status"] = "INELIGIBLE_OR_UNRESOLVED_NO_RETRY"
    else:
        result["primary_precision_status"] = "CONDITIONALLY_ELIGIBLE"
    return config, adapter, evidence, result, rows
