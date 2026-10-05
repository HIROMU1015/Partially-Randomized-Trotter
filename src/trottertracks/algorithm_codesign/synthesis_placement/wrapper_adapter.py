"""SP-1 stored-sequence adapter: exact population accounting, point signal check.

Never invokes a synthesizer, SP-0.5 scoring, a sampler, or shared trotterlib.
Physical bias bounds use saved primitive operator guards and exact rational
coefficient enclosures. Small-matrix signal arithmetic is diagnostic only.
Cost is fusion-normalized additive synthesized-primitive T cost; no cross-string
Clifford+T simplification or actual whole-wrapper compilation is performed.
"""
from __future__ import annotations

from decimal import Decimal, localcontext, ROUND_FLOOR, ROUND_CEILING
from fractions import Fraction as F
import hashlib
import json

import mpmath as mp

from .wrapper_accounting import Gate, Path, population_profile, axis_budget
from .wrapper_sequence import Angle, selected

COST_METRIC = "fusion-normalized additive synthesized-primitive T cost"


def bounds(value):
    value = F(value)
    out = {}
    for key, mode in (("lo", ROUND_FLOOR), ("hi", ROUND_CEILING)):
        with localcontext() as ctx:
            ctx.prec = 70
            ctx.rounding = mode
            out[key] = str(Decimal(value.numerator)/Decimal(value.denominator))
    return out


def interval(raw):
    lo, hi = F(raw["lo"]), F(raw["hi"])
    if lo > hi:
        raise ValueError("reversed saved enclosure")
    return lo, hi


def exact_catalogue_index(angle):
    if angle.rad:
        return None
    scaled = 4*angle.pi
    return scaled.numerator % 8 if scaled.denominator == 1 else None


def catalogue_sequence(k):
    return "t" if k == 7 else "S"*(k//2)+("T" if k % 2 else "")


def source_angle(raw):
    unit = raw["unit"]
    v = F(raw["numerator"], raw["denominator"])
    if unit not in ("pi", "rad"):
        raise ValueError("invalid saved angle")
    return Angle(pi=v) if unit == "pi" else Angle(rad=v)


class StoredLibrary:
    """Use only the input bytes bound in contract; do not recompute PAI or J."""
    def __init__(self, raw_bytes, expected_sha):
        if hashlib.sha256(raw_bytes).hexdigest() != expected_sha:
            raise PermissionError("SP-0.5 saved-input identity mismatch")
        raw = json.loads(raw_bytes)
        self.sequences, self.angle_keys, self.interpolations = {}, {}, {}
        for row in raw["synthesis_rows"]:
            sequence = row["sequence"]
            if (hashlib.sha256(sequence.encode()).hexdigest() != row["sequence_sha256"]
                    or sequence.count("T")+sequence.count("t") != row["T_count"]
                    or row.get("error_pass") is not True
                    or any(g not in "HTtSXW" for g in sequence)):
                raise PermissionError("saved sequence / count / guard failed")
            eta = F(row["error_guard"]["projective_operator_upper"])
            if not 0 <= eta <= F(1, 10**6):
                raise PermissionError("saved operator error exceeds fixed epsilon")
            a = source_angle(row["angle"])
            k = exact_catalogue_index(a)
            if k is not None and sequence != catalogue_sequence(k):
                raise PermissionError("exact fast-path sequence proof does not match")
            self.sequences[row["key"]] = row
            self.angle_keys.setdefault(a, []).append(row["key"])
        for row in raw["economics_rows"]:
            # Stored J/classification/moments are intentionally not used.
            for a_raw, interp in zip(row["native_angles"], row["interpolations"], strict=True):
                a = source_angle(a_raw)
                previous = self.interpolations.setdefault(a, interp)
                if previous != interp:
                    raise PermissionError("duplicate saved PAI enclosures disagree")
        if len(self.sequences) != 23:
            raise PermissionError("saved key inventory changed")
        for k in range(8):
            if f"pi:{k}/4" not in self.sequences:
                raise PermissionError("exact catalogue incomplete")

    def baseline_key(self, angle):
        k = exact_catalogue_index(angle)
        if k is not None:
            return f"pi:{k}/4"  # Channel periodicity only AFTER joint lowering/fusion.
        if angle not in self.angle_keys:
            raise ValueError("unregistered native angle would require new synthesis")
        return sorted(self.angle_keys[angle])[0]

    def delta(self, key):
        row = self.sequences[key]
        if exact_catalogue_index(source_angle(row["angle"])) is not None:
            return F(0)  # Proven exact projective channel, common to NONE and PAI.
        guard = row["error_guard"]
        # Rounding saved eta upward must never make 2*eta exceed the chosen delta.
        return max(F(guard["channel_diamond_upper"]), 2*F(guard["projective_operator_upper"]))

    def spec(self, native, mask):
        choose = selected(native, mask)
        if native.generator == "II":
            # The pre-lowering system phase is never discarded. Only this full
            # joint-space global factor acts as the identity channel.
            coefficients, keys, coefficient_error, gamma_upper = (F(1),), (None,), F(0), F(1)
            saved = None
        elif not choose or exact_catalogue_index(native.angle) is not None:
            coefficients, keys = (F(1),), (self.baseline_key(native.angle),)
            coefficient_error, gamma_upper, saved = F(0), F(1), None
        else:
            saved = self.interpolations[native.angle]
            enclosures = [interval(b) for b in saved["g"]]
            mids = [(lo+hi)/2 for lo, hi in enclosures]
            # Trace preservation and p/weight consistency are exact. g3 is
            # corrected by the known sum(g)=1 identity, and its displacement
            # from the saved true-coefficient enclosure is fully charged to u.
            coefficients = (mids[0], mids[1], 1-mids[0]-mids[1])
            coefficient_error = sum(max(abs(g-lo), abs(g-hi)) for g, (lo, hi)
                                    in zip(coefficients, enclosures))
            gamma_upper = max(sum(max(abs(lo), abs(hi)) for lo, hi in enclosures),
                              sum(abs(g) for g in coefficients))
            keys = tuple(f"pi:{k}/4" for k in saved["notch_indices"])
        costs = tuple(self.sequences[k]["T_count"] if k is not None else 0 for k in keys)
        deltas = tuple(self.delta(k) if k is not None else F(0) for k in keys)
        gate = Gate(coefficients, costs, deltas)
        return {"gate": gate, "keys": keys, "generator": native.generator,
                "coefficient_error": coefficient_error, "gamma_upper": gamma_upper,
                "saved_interpolation": saved, "placement_selected": choose}


def spec_record(spec):
    gate = spec["gate"]
    gamma = sum(abs(g) for g in gate.coefficients)
    return {"generator": spec["generator"], "placement_selected": spec["placement_selected"],
            "stored_sequence_keys": spec["keys"],
            "coefficients_exact": [str(g) for g in gate.coefficients],
            "canonical_probabilities_exact": [str(abs(g)/gamma) for g in gate.coefficients],
            "gamma": bounds(gamma), "T_counts_additive": list(map(int, gate.t_counts)),
            "channel_delta_upper": [bounds(e)["hi"] for e in gate.diamond_errors],
            "coefficient_L1_error_upper": bounds(spec["coefficient_error"])["hi"],
            "true_and_implemented_gamma_upper": bounds(spec["gamma_upper"])["hi"],
            "saved_true_coefficient_interpolation": spec["saved_interpolation"]}


def numerical_bias(paths, path_specs):
    total = F(0)
    for path, specs in zip(paths, path_specs, strict=True):
        gamma_max_product = F(1)
        for spec in specs:
            gamma_max_product *= spec["gamma_upper"]
        error = gamma_max_product*sum(spec["coefficient_error"]/spec["gamma_upper"] for spec in specs)
        total += path["probability"]*abs(path["outer_weight"])*error
    return total


def conditional_record(path, specs, contract):
    gamma = F(1)
    for spec in specs:
        gamma *= sum(abs(g) for g in spec["gate"].coefficients)
    conditional = population_profile([Path(F(1), path["outer_weight"],
        tuple(s["gate"] for s in specs), fixed_t=contract["metric"]["per_shot_fixed_T"])])
    return {"Gamma": bounds(gamma),
        "abs_outer_weight_times_Gamma": bounds(abs(path["outer_weight"])*gamma),
        "conditional_weight_second_moment": bounds(conditional["second_moment"]),
        "conditional_expected_C_T_add": bounds(conditional["expected_T_count"]),
        "conditional_E_W2_C_T_add": bounds(conditional["joint_weighted_T"]),
        "conditional_synthesis_bias_upper": bounds(conditional["synthesis_bias_upper"])["hi"]}


def resource_row(paths, path_specs, contract):
    if (96*F(contract["metric"]["alpha_axis"]) != F(contract["metric"]["alpha_familywise"])
            or 2*F(contract["metric"]["epsilon_axis_lower"])**2 > F(contract["metric"]["epsilon_complex"])**2):
        raise ValueError("complex accuracy / familywise confidence allocation inconsistent")
    population = [Path(p["probability"], p["outer_weight"],
                       tuple(s["gate"] for s in specs), fixed_t=contract["metric"]["per_shot_fixed_T"])
                  for p, specs in zip(paths, path_specs, strict=True)]
    profile = population_profile(population)
    u = numerical_bias(paths, path_specs)
    if u > F(contract["metric"]["numerical_bias_cap"]):
        raise ArithmeticError("numerical coefficient bound exceeds fixed cap; no retry")
    epsilon = F(contract["metric"]["epsilon_axis_lower"])
    alpha = F(contract["metric"]["alpha_axis"])
    record = axis_budget(profile, epsilon, alpha, contract["caps"]["shot_cap_per_axis"], numerical_bias=u)
    return profile, u, record


def classify_ratio(raw):
    lo, hi = interval(raw)
    if hi <= F(19, 20):
        return "MATERIAL_GAIN"
    if lo >= F(21, 20):
        return "MATERIAL_LOSS"
    if F(19, 20) < lo <= hi < F(21, 20):
        return "NO_MATERIAL_SEPARATION"
    return "NUMERIC_INCONCLUSIVE"


def _mp(value):
    value = F(value)
    return mp.mpf(value.numerator)/value.denominator


def angle_value(angle):
    return _mp(angle.pi)*mp.pi + _mp(angle.rad)


def dagger(matrix):
    return matrix.transpose_conj()


def tensor(a, b):
    return mp.matrix([[a[i//b.rows, j//b.cols]*b[i % b.rows, j % b.cols]
                       for j in range(a.cols*b.cols)] for i in range(a.rows*b.rows)])


def pauli(name):
    return {"I": mp.eye(2), "X": mp.matrix([[0, 1], [1, 0]]),
            "Y": mp.matrix([[0, -mp.j], [mp.j, 0]]), "Z": mp.matrix([[1, 0], [0, -1]])}[name]


def ideal_native(native):
    theta = angle_value(native.angle)
    q = tensor(pauli(native.generator[0]), pauli(native.generator[1]))
    return mp.cos(theta/2)*mp.eye(4)-mp.j*mp.sin(theta/2)*q


def sequence_matrix(sequence):
    w = mp.exp(mp.j*mp.pi/4)
    gates = {"H": mp.matrix([[1, 1], [1, -1]])/mp.sqrt(2),
        "T": mp.diag([1, w]), "t": mp.diag([1, 1/w]), "S": mp.diag([1, mp.j]),
        "X": pauli("X"), "W": w*mp.eye(2)}
    u = mp.eye(2)
    for letter in sequence:
        u = u*gates[letter]  # Stored pygridsynth written-product order.
    return u


def embedded_sequence(generator, sequence):
    """Fixed Clifford conjugation of a stored Rz string to a joint Pauli."""
    if generator == "II":
        return mp.eye(4)
    if generator not in ("IX", "IY", "IZ", "ZX", "ZY", "ZZ", "ZI"):
        raise ValueError("no reviewed Clifford lowering for generator")
    u = sequence_matrix(sequence)
    if generator == "ZI":
        return tensor(u, mp.eye(2))
    h, s = sequence_matrix("H"), sequence_matrix("S")
    basis = {"Z": mp.eye(2), "X": h, "Y": s*h}[generator[1]]
    full_basis = tensor(mp.eye(2), basis)
    cnot = mp.matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    middle = tensor(mp.eye(2), u)
    if generator[0] == "Z":
        middle = cnot*middle*cnot
    return full_basis*middle*dagger(full_basis)


def initial_density():
    # |+>_ancilla |0>_system, represented without an approximate sqrt.
    return mp.matrix([[mp.mpf("0.5") if i in (0, 2) and j in (0, 2) else 0
                       for j in range(4)] for i in range(4)])


def signal(density):
    re = sum((tensor(pauli("X"), mp.eye(2))*density)[i, i] for i in range(4))
    im = sum((tensor(pauli("Y"), mp.eye(2))*density)[i, i] for i in range(4))
    return mp.mpc(mp.re(re), mp.re(im))


def diagnostic_signal(paths, path_specs, library, cache):
    """Point mpmath checks; NEVER used for shots, materiality, or selection."""
    ideal = finite = mp.mpc(0)
    trace_residual = mp.mpf(0)
    for path, specs in zip(paths, path_specs, strict=True):
        rho_ideal, rho_finite = initial_density(), initial_density()
        for native, spec in zip(path["post"], specs, strict=True):
            u = ideal_native(native)
            rho_ideal = u*rho_ideal*dagger(u)
            accumulated = mp.zeros(4)
            for coefficient, key in zip(spec["gate"].coefficients, spec["keys"], strict=True):
                if coefficient == 0:
                    continue
                cache_key = (native.generator, key)
                if cache_key not in cache:
                    cache[cache_key] = (mp.eye(4) if key is None else
                        embedded_sequence(native.generator, library.sequences[key]["sequence"]))
                v = cache[cache_key]
                accumulated += _mp(coefficient)*(v*rho_finite*dagger(v))
            rho_finite = accumulated
        scale = _mp(path["probability"]*path["outer_weight"])
        ideal += scale*signal(rho_ideal)
        finite += scale*signal(rho_finite)
        trace_residual = max(trace_residual, abs(sum(rho_finite[i, i] for i in range(4))-1))
    return {"ideal": ideal, "finite": finite, "trace_residual": trace_residual,
            "axis_residual": {"Re": abs(mp.re(finite-ideal)), "Im": abs(mp.im(finite-ideal))}}
