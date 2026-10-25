"""Finite matched B0 evaluation; reuses D2, C/Arb and saved PR accounting.

No scientific work on import. The dedicated runner owns the fixed inputs and
finite call scope. Historical entrypoints, contracts and results are untouched.
"""
from fractions import Fraction as Q
import math

AXES = ("cosine", "sine")
EPSILONS = ("0.05", "0.01", "0.005", "0.001", "0.0005")


def validate_retained_body(body, model, prefix, metadata):
    from .h4_d2_pauli_control import _tokens
    tokens, phase = _tokens(body)
    n = model["n_qubits"]; case = model["cases"]["PR"]
    q = case["config"]["q"]; dt = model["step_time"]
    blocks = case["blocks"]
    if (body.num_qubits != n or case["config"]["T"] / q != dt
            or prefix["L_D"] != len(blocks)
            or prefix["retained_labels"] != [b["label"] for b in blocks]):
        raise ValueError("RETAINED_PARTITION_OR_TIME_MISMATCH")
    expected = []
    for _ in range(q):
        for ordered in (blocks, reversed(blocks)):
            for b in ordered:
                expected.append(dict(kind="evolution", label=b["label"],
                                     coefficient=float(b["coefficient"]), time=dt / 2,
                                     central_rz_angle=2 * b["coefficient"] * (dt / 2),
                                     wires=tuple(range(n))))
    expected_phase = 0.
    for _ in range(q):
        expected_phase = (expected_phase - model["hamiltonian"]["constant"] * dt) % (2 * math.pi)
    if tokens != expected or phase != metadata["sidecar"]["naming"]["global_phase"] or phase != expected_phase:
        raise ValueError("RETAINED_SOURCE_OPERAND_ORDER_OR_PHASE_MISMATCH")
    return dict(status="PASS", retained_count=len(blocks), instruction_count=len(tokens),
                aggregate_phase=phase, q=q, T=case["config"]["T"],
                body_newly_generated=False, event_applications=0)


def _interval(saved, C):
    lo, hi = Q(saved["lower"]), Q(saved["upper"])
    if lo > hi:
        raise ValueError("REVERSED_INTERVAL")
    return C.real((lo + hi) / 2) + C.arb(0, C.real((hi - lo) / 2).upper())


def center_radius(interval):
    lo, hi = Q(interval["lower"]), Q(interval["upper"])
    if lo > hi:
        raise ValueError("REVERSED_INTERVAL")
    return (lo + hi) / 2, (hi - lo) / 2


def target_enclosure(model, reference, bits):
    """One complex retained-state action, giving both measurement axes.

    Reuse the C operands and phase, the PR0 exact Pauli action, and the Arb
    bridge's unitary L2 recentering. No reference evolution or PR event occurs.
    The circuit's accumulated binary64 phase belongs to rho, not u_enc.
    """
    from . import h4_arb_bridge as B
    from .h4_arb_pr0_preparation import _pauli_map, _pauli_action
    C = B._backend()
    if bits not in (64, 96, 128):
        raise ValueError("PRECISION_SCHEDULE")
    n = model["n_qubits"]; raw = model["state"]
    case = model["cases"]["PR"]; q = case["config"]["q"]
    if len(raw) != 1 << n or model["qubit_order"] != "qiskit_little_endian":
        raise ValueError("STATE_LAYOUT")
    with C.ctx.workprec(bits):
        nu = sum((C.rational(x)**2 + C.rational(y)**2 for x, y in raw), Q(0))
        if nu <= 0:
            raise ValueError("RAW_NORM_NOT_POSITIVE")
        if "state_norm" in reference and not (Q(reference["state_norm"]["lower"]) <= nu <= Q(reference["state_norm"]["upper"])):
            raise ValueError("SAVED_REFERENCE_NORM_MISMATCH")
        bra = [C.complex_ball(z) for z in raw]; vector = list(bra)
        if not all(z.real.is_exact() and z.imag.is_exact() for z in vector):
            raise ValueError("INITIAL_BINARY64_STATE_NOT_EXACT")
        error = Q(0); maps = {}; trig = {}
        delta = C.rational(model["step_time"])
        blocks = [(b["label"], C.rational(b["coefficient"]) * delta / 2) for b in case["blocks"]]
        for label, angle in blocks:
            maps[label] = _pauli_map(label, 1)
            if angle not in trig:
                a = C.real(angle); trig[angle] = (a.cos(), C.acb(0, -a.sin()))
        scalar = C.phase(C.real(-C.rational(model["hamiltonian"]["constant"]) * delta))
        for _ in range(q):
            for ordered in (blocks, reversed(blocks)):
                for label, angle in ordered:
                    cosine, sine = trig[angle]
                    pv = _pauli_action(vector, maps[label], C)
                    vector, error = B._recenter([cosine*z + sine*w for z, w in zip(vector, pv)], error, C)
            vector, error = B._recenter([scalar*z for z in vector], error, C)
        dot = sum((a.conjugate()*z for a, z in zip(bra, vector)), C.acb(0))
        transfer = (C.real(nu).sqrt().upper() * C.real(error)).upper()
        raw_signal = C.acb(dot.real + C.arb(0, transfer), dot.imag + C.arb(0, transfer))
        raw_reference = C.acb(_interval(reference["reference"]["real"], C),
                              _interval(reference["reference"]["imag"], C))
        norm = C.real(nu)
        physical = raw_signal / norm
        bias = (raw_signal - raw_reference) / norm
        if not physical.is_finite() or not bias.is_finite():
            raise ValueError("NONFINITE_TARGET_OR_BIAS")
        record = dict(bits=bits, state_actions=1, reference_state_actions=0,
                      raw_norm_exact=str(nu), state_renormalized=False,
                      retained_rotations=2*q*len(blocks), scalar_applications=q,
                      target_operand_kind="EXACT_PRODUCTS_OF_SAVED_BINARY64_OPERANDS",
                      raw_signal=C.serialize(raw_signal), physical_signal=C.serialize(physical),
                      saved_full_reference=reference["reference"], bias=C.serialize(bias),
                      state_l2_error_upper_exact=str(error),
                      inner_product_error_upper_exact=str(C.dyadic(transfer)),
                      mathematical_B_exact="1", log_B_exact="0",
                      actual_rho=None, formal_certificate_accepted=False)
        record["axes"] = {}
        for axis, key in zip(AXES, ("real", "imag")):
            bounds = record["bias"][key]; center, radius = center_radius(bounds)
            record["axes"][axis] = dict(bias_interval=bounds, signed_bias_exact=str(center),
                                        u_enc_exact=str(radius), interval_width_exact=str(2*radius))
        return record


def precision_decisions(target):
    """Classify width ambiguity without evaluating shot numbers."""
    decisions = []
    for eps in EPSILONS:
        epsilon_axis = Q(float(eps) / math.sqrt(2))
        for axis in AXES:
            row = target["axes"][axis]
            center, radius = Q(row["signed_bias_exact"]), Q(row["u_enc_exact"])
            h = epsilon_axis - abs(center) - radius
            if h > 0 and radius / h <= Q(1, 100):
                status = "GUARD_MET"
            elif max(Q(0), abs(center)-radius) >= epsilon_axis:
                status = "BIAS_EXCEEDS_TOLERANCE"
            else:
                status = "WIDTH_AMBIGUITY"
            decisions.append(dict(epsilon_sig=eps, axis=axis, status=status))
    return decisions


def conditional_results(target, costs, saved_PR):
    from .h4_conditional_cng import scale_bounds, base_axis, shot
    scale = scale_bounds(1, 0)
    outputs = []; calls = 0
    for eps in EPSILONS:
        axes = {}; low_counts = {}; high_counts = {}
        for axis in AXES:
            t = target["axes"][axis]
            a = base_axis(eps, t["signed_bias_exact"], t["u_enc_exact"], scale)
            zero = shot(a, 0); calls += 1
            cap = Q(a["rho_guard_exact"])
            guard = shot(a, cap) if a["domain_nonempty"] else dict(N=None, reason="EMPTY_GUARD_DOMAIN")
            calls += int(a["domain_nonempty"])
            axes[axis] = dict(a, rho_zero=zero, guard_face=guard)
            low_counts[axis] = zero["N"]; high_counts[axis] = guard["N"]
        old_rows = [r for r in saved_PR if r["case"] == "pauli_jw_PR" and r["epsilon_sig"] == eps]
        if len(old_rows) != 1:
            raise ValueError("SAVED_PR_PRECISION_ROW")
        old = old_rows[0]
        def total(counts):
            return None if any(counts[a] is None for a in AXES) else sum(counts[a]*costs[a]["RZ"] for a in AXES)
        g0, g1 = total(low_counts), total(high_counts)
        pr0 = Q(old["G_exact"])
        prlo, prhi = map(Q, old["G_guard_range_exact"])
        nominal_order = "UNDETERMINED" if g0 is None else "B0_LOWER" if g0 < pr0 else "PR_LOWER" if g0 > pr0 else "TIE"
        guard_order = "UNDETERMINED"
        if g0 is not None and g1 is not None:
            guard_order = "B0_LOWER_THROUGHOUT_INDEPENDENT_GUARDS" if g1 < prlo else "PR_LOWER_THROUGHOUT_INDEPENDENT_GUARDS" if g0 > prhi else "GUARD_RANGES_OVERLAP"
        outputs.append(dict(epsilon_sig=eps, case="pauli_jw_B0_B0I", axes=axes,
                            N_cos=low_counts["cosine"], N_sin=low_counts["sine"],
                            N_total=None if None in low_counts.values() else sum(low_counts.values()),
                            G_exact=None if g0 is None else str(g0),
                            G_guard_range_exact=None if g0 is None or g1 is None else [str(g0), str(g1)],
                            saved_PR={k:old[k] for k in ("case", "N_cos", "N_sin", "N_total", "G_exact", "G_guard_range_exact", "cohort_n", "hypothetical_rho_new")},
                            rho_zero_order=nominal_order, independent_guard_order=guard_order,
                            B0_over_saved_PR_G_exact=None if g0 is None else str(Q(g0)/pr0),
                            condition="HYPOTHETICAL_RHO_ZERO; SAVED_PR_COHORT_MEAN_PRICE",
                            actual_rho=None, actual_N=None, actual_G=None,
                            formal_certificate_accepted=False))
    if calls > 20:
        raise ValueError("N_SCALAR_CALL_SCOPE")
    return dict(rows=outputs, scale=scale, N_scalar_evaluations=calls,
                PR_N_G_recomputations=0, PR_cost_population_mean_guaranteed=False)
