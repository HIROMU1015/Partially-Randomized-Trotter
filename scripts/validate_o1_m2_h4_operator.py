"""Fixed update005 H4 operator-only check; no dynamics, states or circuits.

Science SOURCE = fixed adapter commit + this driver's recorded SHA256.
The saved core's generating SOURCE is separate historical provenance.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ADAPTER_COMMIT = "471b8587535e46732d4ebf0f31e4900ea14b9346"
RESULT_COMMIT = "aced1647b81e8d28091058cdc7b6e3e1ee54896a"
RESULT_DIR = "artifacts/track_a_o1_m2_h4_operator_correctness/2026-11-06"
CORE_PATH = "artifacts/research_c_pr_integrated/2026-11-02/run01/update005_candidate.npz"
INPUT_PATH = "artifacts/research_c_pr_aware_df/2026-10-27/run01/prepare/input.npz"


class CARReference:
    """Existing independent bit CAR, with sparse cached monomial products.

    Extends the earlier artificial test's direct h/g and density-product
    references to full256, avoiding repeated dense 256x256 multiplications.
    """
    def __init__(self, spatial):
        import numpy as np
        from scipy.sparse import csr_matrix, eye
        from trottertracks.pr_dictionary_validation.reference import car_annihilator

        class Context:
            @staticmethod
            def matrix(rows, columns):
                return np.zeros((rows, columns), dtype=np.complex128)

        self.n = spatial
        self.dimension = 1 << (2 * spatial)
        self.identity = eye(self.dimension, format="csr", dtype=complex)
        self.zero = csr_matrix((self.dimension, self.dimension), dtype=complex)
        self.annihilators = tuple(csr_matrix(car_annihilator(Context, 2*spatial, p))
                                 for p in range(2*spatial))
        self.creators = tuple(a.conj().T.tocsr() for a in self.annihilators)
        self.cache = {(): self.identity}
        self.E = {(p, q): sum((self.monomial(((2*p+s, 1), (2*q+s, 0)))
                               for s in range(2)), self.zero.copy())
                  for p, q in np.ndindex(spatial, spatial)}

    def monomial(self, term):
        if term not in self.cache:
            product = self.identity
            for q, creation in term:
                product = product @ (self.creators[q] if creation else self.annihilators[q])
            self.cache[term] = product.tocsr()
        return self.cache[term]

    def integrals(self, packet):
        import numpy as np
        out = packet.nuclear_constant * self.identity
        for p, q in np.ndindex(self.n, self.n):
            out = out + packet.h[p, q] * self.E[p, q]
        for p, q, r, s in np.ndindex((self.n,) * 4):
            coefficient = .5 * packet.g[p, q, r, s]
            if coefficient == 0.:  # exact zero only
                continue
            for sigma, tau in np.ndindex(2, 2):
                term = ((2*p+sigma, 1), (2*r+tau, 1), (2*s+tau, 0), (2*q+sigma, 0))
                out = out + coefficient * self.monomial(term)
        return out.toarray()

    def density_core(self, model):
        import numpy as np
        out = model.constant * self.identity
        for p, q in np.ndindex(self.n, self.n):
            out = out + model.spatial_one_body_tensor[p][q] * self.E[p, q]
        for frame in model.frames:  # every saved frame, never an LD2 tail
            J, U = np.asarray(frame.spatial_J), np.asarray(frame.spatial_orbital_rotation)
            densities = [sum((U[p, i] * U[q, i].conjugate() * self.E[p, q]
                              for p, q in np.ndindex(self.n, self.n)), self.zero.copy())
                         for i in range(self.n)]
            for i in range(self.n):
                weighted = sum((J[i, j] * densities[j] for j in range(self.n)), self.zero.copy())
                out = out + .5 * (densities[i] @ weighted)
        return out.toarray()


def packet_scale(packet):
    import numpy as np
    # Every spin-summed one-body and quartic CAR monomial has norm <=1.
    return abs(packet.nuclear_constant) + 2*float(np.sum(abs(packet.h))) + 2*float(np.sum(abs(packet.g)))


def core_scale(model):
    import numpy as np
    scale = abs(model.constant) + 2*float(np.sum(abs(np.asarray(model.spatial_one_body_tensor))))
    for frame in model.frames:
        U, J = np.asarray(frame.spatial_orbital_rotation), np.asarray(frame.spatial_J)
        s = np.sum(abs(U), axis=0)**2
        scale += 2*float(np.sum(abs(J)*s[:, None]*s[None, :]))
    return scale


def pauli_matrix(packet):
    """Independent Qiskit-label bit action; no Qiskit matrix or circuit call."""
    import numpy as np
    d = 1 << packet.num_qubits
    result = packet.constant * np.eye(d, dtype=complex)
    for label, coefficient in packet.terms:
        for column in range(d):
            row, phase = column, 1.+0j
            for q, axis in enumerate(reversed(label)):
                occupied = (column >> q) & 1
                if axis in "XY":
                    row ^= 1 << q
                if axis == "Y":
                    phase *= -1j if occupied else 1j
                elif axis == "Z" and occupied:
                    phase = -phase
            result[row, column] += coefficient * phase
    return result


def mapper_small_term_audit(packet, reference, car_matrix, mapped_matrix):
    """Observe existing OF's actual fermion conversion loss; do not patch it."""
    import numpy as np
    from openfermion import InteractionOperator, get_fermion_operator
    from openfermion.config import EQ_TOLERANCE
    from trottertracks.c1_stage1 import spin_lift_unthresholded
    one, two = spin_lift_unthresholded(packet.h, packet.g)
    interaction = InteractionOperator(packet.nuclear_constant, one, two/2)
    raw = {term: interaction[term] for term in interaction if interaction[term] != 0}
    retained = get_fermion_operator(interaction).terms
    lost = {term: value for term, value in raw.items() if term not in retained}
    tiny = [abs(v) for v in raw.values() if 0 < abs(v) < EQ_TOLERANCE]
    removed = reference.zero.copy()
    for term, coefficient in lost.items():
        removed = removed + coefficient * reference.monomial(term)
    removed = removed.toarray()
    residual_after_fermion_loss = mapped_matrix - (car_matrix - removed)
    return dict(eq_tolerance=float(EQ_TOLERANCE), raw_nonzero_fermion_terms=len(raw),
        retained_fermion_terms=len(retained), tiny_nonzero_terms=len(tiny),
        tiny_coefficient_l1=math.fsum(float(v) for v in tiny),
        lost_fermion_terms=len(lost),
        lost_coefficient_l1=math.fsum(float(abs(v)) for v in lost.values()),
        lost_fermion_operator_max_entry=float(np.max(abs(removed))),
        mapper_minus_retained_CAR_max_entry=float(np.max(abs(residual_after_fermion_loss))),
        mapper_minus_full_CAR_max_entry=float(np.max(abs(mapped_matrix-car_matrix))),
        interpretation="Residual after actual fermion loss includes JW cutoff and binary64 rounding; not isolated as exact arithmetic.")


def run(core_path, input_path, output):
    import numpy as np
    from trottertracks import c1_stage1 as stage1
    from trottertracks import o1_m2_residual_adapter as adapter

    start = time.monotonic()
    adapter_path = ROOT / "src/trottertracks/o1_m2_residual_adapter.py"
    fixed = subprocess.check_output(["git", "show", ADAPTER_COMMIT+":src/trottertracks/o1_m2_residual_adapter.py"], cwd=ROOT)
    if adapter_path.read_bytes() != fixed:
        raise ValueError("Fixed adapter SOURCE differs")
    if output.exists():
        raise FileExistsError("Fresh operator result path required")
    result = dict(scope="M2_H4_OPERATOR_CORRECTNESS_ONLY", started_utc=datetime.now(timezone.utc).isoformat(),
        science_source=dict(adapter_commit=ADAPTER_COMMIT,
            driver_path=str(Path(__file__).relative_to(ROOT)),
            driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256(fixed).hexdigest(),
            core_generation_source=adapter.UPDATE005.source_identity),
        saved_inputs=dict(result_commit=RESULT_COMMIT, original_core_path=CORE_PATH,
            original_input_path=INPUT_PATH, binding=asdict(adapter.UPDATE005),
            local_core_path=str(core_path.relative_to(ROOT)), local_input_path=str(input_path.relative_to(ROOT))),
        runtime=dict(python=sys.version, host=platform.node(), precision="float64/complex128",
            packages={k: importlib.metadata.version(k) for k in ("numpy", "scipy", "qiskit", "openfermion")}),
        command=shlex.join([sys.executable, *sys.argv]),
        conditions=dict(molecule="linear H4", spacing_angstrom=1.0, basis="STO-3G",
            orbitals=4, modes=8, full_dimension=256, sector=[2, 2], sector_dimension=36,
            core_rank=8, retained_frames=list(range(8)), LD2_tail_used=False,
            spin_order="mode 2p=alpha,2p+1=beta", fock_index="sum occupation[q]*2**q",
            pauli_label="q[n-1]...q[0]", sector_projection=False,
            state_or_eigensolver=False, dynamics_or_sampling=False, circuits=False, shots_or_cost=False))
    pair = adapter.load_saved_pair(core_path, input_path)

    # The formula, operation count and guard are fixed in SOURCE before any
    # operator difference. Only input operand scales enter the numerical value.
    lengths = stage1.operation_lengths(4, 256, 8)
    scales = dict(physical=packet_scale(pair.physical), core_density=core_scale(pair.general_core),
                  core_integrals=packet_scale(pair.core), residual=packet_scale(pair.residual))
    length = max(lengths.values())
    tolerance = 8*stage1.gamma(length)*max(1., math.fsum(scales.values()))
    result["tolerance"] = dict(formula="8*gamma(max(operation_lengths(4,256,8)))*max(1,sum(operand_scales))",
        unit_roundoff=2.**-53, rounding_guard=8, operation_lengths=lengths,
        path_length=length, operand_scales=scales, absolute_max_entry=tolerance, rtol=0,
        interpretation="Predeclared binary64 operation-scale engineering allowance; no rigorous enclosure or energy guarantee. Not inherited from synthetic 1e-14.")
    print(json.dumps(dict(predeclared_tolerance=result["tolerance"])), flush=True)

    reference = CARReference(4)
    car = {name: reference.integrals(getattr(pair, name)) for name in ("physical", "core", "residual")}
    density = reference.density_core(pair.general_core)
    pauli_packets = dict(zip(("physical", "core", "residual"), pair.pauli_parts(), strict=True))
    mapped = {name: pauli_matrix(packet) for name, packet in pauli_packets.items()}
    sector = stage1.sector_indices(4, 2, 2)
    outside = np.setdiff1d(np.arange(256), sector)
    checks = []

    def check(name, left, right):
        difference = left-right
        for space, error in (("full256", difference), ("sector36", difference[np.ix_(sector, sector)])):
            maximum = float(np.max(abs(error)))
            checks.append(dict(check=name, space=space, max_entry=maximum,
                frobenius=float(np.linalg.norm(error.ravel())), tolerance=tolerance,
                status="PASS" if maximum <= tolerance else "FAIL"))

    check("density_core_vs_normal_order_core", density, car["core"])
    check("density_core_plus_R_vs_physical_CAR", density+car["residual"], car["physical"])
    check("normal_order_core_plus_R_vs_physical_CAR", car["core"]+car["residual"], car["physical"])
    for name in ("physical", "core", "residual"):
        check(name+"_Pauli_vs_CAR", mapped[name], car[name])
        check(name+"_CAR_Hermitian", car[name], car[name].conj().T)
    check("Pauli_core_plus_R_vs_physical_CAR", mapped["core"]+mapped["residual"], car["physical"])
    check("Pauli_core_plus_R_vs_physical_Pauli", mapped["core"]+mapped["residual"], mapped["physical"])
    check("Pauli_R_vs_physical_minus_density_core", mapped["residual"], car["physical"]-density)
    result["checks"] = checks

    # Independent element sums, not the adapter's einsum contraction.
    kappa = np.array([[pair.physical.h[p, q] - .5*math.fsum(float(pair.physical.g[p, r, r, q]) for r in range(4))
                       for q in range(4)] for p in range(4)])
    contraction = np.array([[.5*math.fsum(float(pair.residual.g[p, r, r, q]) for r in range(4))
                             for q in range(4)] for p in range(4)])
    bookkeeping_errors = dict(original_g_kappa=float(np.max(abs(np.asarray(pair.general_core.spatial_one_body_tensor)-kappa))),
        residual_one_body_contraction=float(np.max(abs(pair.residual.h-contraction))),
        residual_g_sign=float(np.max(abs(pair.residual.g-(pair.physical.g-pair.core.g)))))
    result["bookkeeping"] = dict(errors=bookkeeping_errors,
        residual_h_maxabs=float(np.max(abs(pair.residual.h))),
        residual_g_maxabs=float(np.max(abs(pair.residual.g))),
        nuclear_constants={name: float(getattr(pair, name).nuclear_constant) for name in car},
        electronic_identity={name: dict(Pauli_constant=float(pauli_packets[name].constant),
            CAR_trace_over_dimension=float(np.trace(car[name]).real/256),
            identity_difference=float(abs(pauli_packets[name].constant-np.trace(car[name])/256))) for name in car},
        wrong_residual_sign_max_entry=float(np.max(abs(density-car["residual"]-car["physical"]))),
        sector_indices=sector.tolist(),
        sector_leakage={name: float(np.max(abs(matrix[np.ix_(outside, sector)]))) for name, matrix in {**car, "density_core": density}.items()})
    result["small_term_audit"] = {name: mapper_small_term_audit(getattr(pair, name), reference, car[name], mapped[name]) for name in car}
    result["Pauli_nonidentity_terms"] = {name: len(packet.terms) for name, packet in pauli_packets.items()}
    result["reference"] = dict(method="Existing car_annihilator bit signs; cached CSR monomials; direct normal-ordered quartic map; independent raw J/U density products; independent label bit action",
                               cached_monomials=len(reference.cache), no_input_cutoff=True,
                               no_symbolic_reference_threshold=True)
    scalar_ok = (pair.physical.nuclear_constant == pair.core.nuclear_constant and pair.residual.nuclear_constant == 0.)
    leakage_ok = max(result["bookkeeping"]["sector_leakage"].values()) <= tolerance
    identity_ok = max(v["identity_difference"] for v in result["bookkeeping"]["electronic_identity"].values()) <= tolerance
    result["status"] = "PASS" if (all(c["status"] == "PASS" for c in checks)
        and max(bookkeeping_errors.values()) <= tolerance and scalar_ok and leakage_ok and identity_ok) else "FAIL"
    result["elapsed_seconds"] = time.monotonic()-start
    result["stopped_after"] = "operator-only result; no PF/RTE law/dynamics/state/circuit/shot/cost"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], maximum=max(c["max_entry"] for c in checks),
        tolerance=tolerance, output=str(output), elapsed_seconds=result["elapsed_seconds"])), flush=True)
    return 0 if result["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=ROOT/RESULT_DIR/"inputs/update005_candidate.npz")
    parser.add_argument("--input", type=Path, default=ROOT/RESULT_DIR/"inputs/input.npz")
    parser.add_argument("--output", type=Path, default=ROOT/RESULT_DIR/"result.json")
    args = parser.parse_args()
    return run(args.core.resolve(), args.input.resolve(), args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
