"""Prepared M2 H4 operator-only entry. Run only after separate user permission.

No default scientific paths: --help/import do not load H4. Reuses the previous
fixed driver's independent sparse CAR and raw density-product references.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import validate_o1_m2_h4_operator as previous


def run(core_path, input_path):
    import math
    import numpy as np
    from trottertracks import c1_stage1 as stage1
    from trottertracks import o1_m2_residual_adapter as adapter
    pair = adapter.load_saved_pair(core_path, input_path)
    # Fix formulas before operator deltas; keep residual scale, no max(1,S_R).
    length = max(stage1.operation_lengths(4, 256, 8).values())
    sr = previous.packet_scale(pair.residual)
    scales = [previous.packet_scale(pair.physical), previous.packet_scale(pair.core),
              previous.core_scale(pair.general_core), sr]
    tau_full = 8*stage1.gamma(length)*math.fsum(scales)
    tau_residual = 8*stage1.gamma(length)*sr
    ref = previous.CARReference(4)
    car = {name: ref.integrals(getattr(pair, name)) for name in ("physical", "core", "residual")}
    density = ref.density_core(pair.general_core)
    packets = dict(zip(car, pair.pauli_parts(), strict=True))
    mapped = {name: previous.pauli_matrix(packet) for name, packet in packets.items()}
    sector = stage1.sector_indices(4, 2, 2)
    checks = []

    def check(name, left, right, tol):
        delta = left-right
        for space, matrix in (("full256", delta), ("sector36", delta[np.ix_(sector, sector)])):
            error = float(np.max(abs(matrix)))
            checks.append(dict(name=name, space=space, max_entry=error,
                frobenius=float(np.linalg.norm(matrix)), absolute_tolerance=tol, passed=error <= tol))

    check("density_core_plus_R_vs_H", density+car["residual"], car["physical"], tau_full)
    check("density_core_vs_integral_core", density, car["core"], tau_full)
    for name in car:
        tol = tau_residual if name == "residual" else tau_full
        check(name+"_Pauli_vs_independent_CAR", mapped[name], car[name],
              tol)
        check(name+"_CAR_Hermitian",car[name],car[name].conj().T,tol)
        check(name+"_mapped_Hermitian",mapped[name],mapped[name].conj().T,tol)
    check("mapped_core_plus_R_vs_original_H", mapped["core"]+mapped["residual"], car["physical"], tau_full)
    residual_size = float(np.max(abs(car["residual"])))
    # A criterion incapable of distinguishing R from zero is not qualified.
    resolution_ok = sr > 0 and tau_residual < .01*residual_size
    identity = {name: dict(mapped=packet.constant,
        car_trace=float(np.trace(car[name]).real/256), nuclear=getattr(pair, name).nuclear_constant)
        for name, packet in packets.items()}
    residual_identity_error = abs(identity["residual"]["mapped"]-identity["residual"]["car_trace"])
    outside = np.setdiff1d(np.arange(256), sector)
    leakage = {name: float(np.max(abs(matrix[np.ix_(outside, sector)]))) for name, matrix in mapped.items()}
    kappa = np.array([[pair.physical.h[p,q]-.5*math.fsum(float(pair.physical.g[p,r,r,q]) for r in range(4))
                       for q in range(4)] for p in range(4)])
    rh = np.array([[.5*math.fsum(float(pair.residual.g[p,r,r,q]) for r in range(4))
                    for q in range(4)] for p in range(4)])
    bookkeeping = dict(kappa_error=float(np.max(abs(kappa-pair.general_core.spatial_one_body_tensor))),
        contraction_error=float(np.max(abs(rh-pair.residual.h))),
        g_sign_error=float(np.max(abs(pair.residual.g-(pair.physical.g-pair.core.g)))),
        residual_h_max=float(np.max(abs(pair.residual.h))),
        residual_g_max=float(np.max(abs(pair.residual.g))))
    passed = (all(c["passed"] for c in checks) and resolution_ok
              and residual_identity_error <= tau_residual and bool(packets["residual"].terms)
              and max(leakage.values()) <= tau_full
              and max(bookkeeping[k] for k in ("kappa_error", "contraction_error", "g_sign_error")) <= tau_full
              and pair.residual.nuclear_constant == 0.
              and pair.physical.nuclear_constant == pair.core.nuclear_constant)
    paths = ("src/trottertracks/o1_m2_residual_adapter.py", "src/trottertracks/o1_m2_cutoff_free_pauli.py",
             "scripts/validate_o1_m2_h4_operator.py", "scripts/validate_o1_m2_retained_residual.py")
    return dict(status="PASS" if passed else "NOT_QUALIFIED", scope="H4_OPERATOR_ONLY",
        utc=datetime.now(timezone.utc).isoformat(), host=platform.node(),
        source=dict(base="a21be7b589bfe43ec7495e9872a961c84858ff91",
                    files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}),
        inputs=dict(binding=previous.asdict(adapter.UPDATE005), core=str(core_path), physical=str(input_path)),
        runtime={k:importlib.metadata.version(k) for k in ("numpy","scipy","openfermion","qiskit")},
        tolerance=dict(L=length, guard=8, operand_scales=scales, full=tau_full, residual=tau_residual,
                       rtol=0, residual_uses_no_unit_floor=True,
                       rationale="Predeclared binary64 operand-scale test, not a rigorous enclosure; H and R criteria separate."),
        checks=checks, identity=identity, residual_identity_error=residual_identity_error,
        residual_size=residual_size, residual_resolution_qualified=resolution_ok,
        zero_mapper_negative_control_error=residual_size,
        wrong_sign_negative_control_error=2*residual_size,
        bookkeeping=bookkeeping, sector_leakage=leakage,
        nonidentity_terms={name:len(p.terms) for name,p in packets.items()},
        metadata={name:p.metadata for name,p in packets.items()},
        stopped_after="operator-only; no dynamics, circuit, state, sampling, shot or cost")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Fresh result path required; old results are protected")
    payload = run(args.core.resolve(), args.input.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
