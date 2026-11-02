"""L local-gate production kernel, using frozen C ball/rational helpers.

No molecular/reference/law/bias/sampling/compiler APIs. Real input is passed
only by the separately authorized L worker. Artificial tests supply tiny data.
All ideal gates use exact binary64 real parameters, never decimal substitution.
"""
from fractions import Fraction as Q
import hashlib
import importlib.util
import math
from pathlib import Path
import struct


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = _load("local_frozen_C_ball", "h4_ball_evidence.py")
P = _load("local_frozen_rational", "h4_enclosure_prototype.py")
require = C.require
PRECISION = 128
LIMITS = dict(forward=148, inverse=148, basis=11, support=324)


def finite_float(value):
    require(type(value) in (str, float, int), "FINITE_BINARY64_PARAMETER")
    value = float(value)
    require(math.isfinite(value), "NONFINITE_PARAMETER")
    return value


def matrix_hash(raw):
    require(type(raw) is list and len(raw) in (2, 4) and all(type(r) is list and len(r) == len(raw) for r in raw), "LOCAL_MATRIX_SHAPE")
    packed = []
    for row in raw:
        for pair in row:
            require(type(pair) is list and len(pair) == 2 and all(type(x) in (int, float) for x in pair), "COMPLEX_BINARY64_PAIR")
            packed.append(struct.pack("<dd", *(finite_float(x) for x in pair)))
    return hashlib.sha256(b"".join(packed)).hexdigest()


def ideal_gate(descriptor, *, inverse=False):
    """Qiskit1.3 RZ and XXPlusYY definitions, including beta's endian sign."""
    require(type(inverse) is bool, "INVERSE_FLAG")
    name, params = descriptor["name"], descriptor["parameters"]
    require(name in ("rz", "xx_plus_yy") and len(params) == (1 if name == "rz" else 2), "UNSUPPORTED_LOCAL_GATE")
    theta = C.real(finite_float(params[0])) * (-1 if inverse else 1)
    if name == "rz":
        return C.acb_mat([[C.phase(-theta / 2), 0], [0, C.phase(theta / 2)]])
    beta = C.real(finite_float(params[1]))
    cos, sin = (theta / 2).cos(), (theta / 2).sin()
    return C.acb_mat([[1, 0, 0, 0], [0, cos, C.acb(0, -sin) * C.phase(-beta), 0],
                      [0, C.acb(0, -sin) * C.phase(beta), cos, 0], [0, 0, 0, 1]])


def discrepancy(descriptor, raw, *, inverse=False):
    require(C.ctx.prec == PRECISION, "FIXED_128_BIT_CONTEXT")
    matrix_hash(raw)
    ideal = ideal_gate(descriptor, inverse=inverse)
    require(len(raw) == ideal.nrows(), "DESCRIPTOR_MATRIX_DIMENSION")
    bound = C.frobenius(C.matrix(raw) - ideal)
    require(bound.is_finite() and bound.lower() >= 0, "UNPROVED_LOCAL_BOUND")
    return C.dyadic(bound.upper())


def product_bound(errors):
    """Frozen rational telescoping, outward rounded at each fixed128 step.

    Avoid exponentially growing denominators in long product bounds. Exact
    two-factor rational composition precedes each Arb outward rounding.
    """
    require(all(type(x) is Q and x >= 0 for x in errors), "NONNEGATIVE_EXACT_BOUND")
    require(C.ctx.prec == PRECISION, "FIXED_128_BIT_CONTEXT")
    bound = Q(0)
    for error in errors:
        exact = P.product_error_bound([Q(1), Q(1)], [bound, error])
        bound = C.dyadic(C.real(exact).upper())
    return bound


def conjugation_bound(forward, inverse):
    # ||F D J - U D U*|| <= f+i+f*i for any unitary diagonal D.
    return product_bound([inverse, forward])


def signed_controlled_identity(sign, *, controlled=True):
    """Artificial/formal branch oracle: a system sign becomes relative phase."""
    require(type(sign) is int and sign in (-1, 1) and type(controlled) is bool, "SIGNED_CONTROL_DOMAIN")
    return dict(zero_branch=1, one_branch=sign) if controlled else dict(system_global_phase=sign)


def coverage_ledger(records):
    """Preserve aggregate C total; refuse unproved overlaps or cancellation."""
    require(type(records) is list and records, "COVERAGE_REQUIRED")
    seen, owners = set(), {}
    for record in records:
        require(set(record) == {"id", "covers", "bound_ref"}, "COVERAGE_SCHEMA")
        require(type(record["id"]) is str and record["id"] not in seen, "DUPLICATE_COVERAGE_RECORD")
        seen.add(record["id"])
        covers = record["covers"]
        require(type(covers) is list and covers and all(type(x) is str for x in covers)
                and len(set(covers)) == len(covers) and bool(record["bound_ref"]), "MISSING_COVERAGE_OR_BOUND")
        for primitive in covers:
            owners.setdefault(primitive, []).append(record["id"])
    overlap = {k: v for k, v in owners.items() if len(v) > 1}
    required = {"bias_distance", "basis_reuse", "phase_accumulation", "pi_rounding", "local_forward", "local_inverse"}
    missing = sorted(required - set(owners))
    return dict(records=records, overlapping_primitives=overlap, uncovered_primitives=missing,
                status="COMBINATION_BLOCKED" if overlap or missing else "DISJOINT_COVERAGE_ONLY_NO_ACCEPTANCE",
                simple_addition_permitted=not overlap and not missing,
                aggregate_C_total_modified=False, numerical_combination_performed=False,
                formal_certificate_accepted=False)


def plan_from_saved(model, registry, *, artificial=False):
    """Static descriptor/byte membership checks; no ball operations here."""
    require(model["representation"] == "df_native" and model["qubit_order"] == "qiskit_little_endian", "DF_MODEL_SCOPE")
    n = model["n_qubits"]
    require(type(n) is int and n > 0, "MODEL_QUBITS")
    gates, bases, supports = {}, {}, []
    case_bases, block_counts = {}, {}
    for method in ("PF", "PR"):
        case = model["cases"][method]
        case_bases[method] = set()
        block_counts[method] = len(case["blocks"])
        for item in case["blocks"] + case["components"]:
            basis = item["basis"]
            key = basis["basis_hash"]
            operations = []
            for op in basis["operations"]:
                d = op["description"]
                require(d["name"] in ("rz", "xx_plus_yy"), "UNSUPPORTED_LOCAL_GATE")
                require(d["qubits"] == op["qubits"] and len(set(op["qubits"])) == len(op["qubits"])
                        and all(type(q) is int and 0 <= q < n for q in op["qubits"]), "BASIS_SUPPORT")
                require(len(op["qubits"]) == (1 if d["name"] == "rz" else 2), "GATE_ARITY")
                require(len(d["parameters"]) == (1 if d["name"] == "rz" else 2), "GATE_PARAMETERS")
                for p in d["parameters"]:
                    finite_float(p)
                h = matrix_hash(op["matrix"])
                require(h == d["matrix_sha256"] and h in registry["local_gates"], "FORWARD_HASH_REGISTRY")
                matrix_hash(op["inverse_matrix"])
                require(len(op["inverse_matrix"]) == len(op["matrix"]), "INVERSE_SHAPE")
                expected = registry["local_gates"][h]
                require(expected["name"] == d["name"] and expected["parameters"] == d["parameters"], "REGISTRY_DESCRIPTOR")
                record = dict(descriptor={"name": d["name"], "parameters": d["parameters"]},
                              forward=op["matrix"], inverse=op["inverse_matrix"])
                if h in gates:
                    require(gates[h] == record, "HASH_ALIAS_DIFFERENT_DESCRIPTOR_OR_INVERSE")
                else:
                    gates[h] = record
                operations.append(dict(gate_hash=h, qubits=op["qubits"]))
            canonical = C.canonical(dict(num_system_qubits=n, operations=[x["description"] for x in basis["operations"]],
                                         hash_policy="ordered_local_basis_operations_v1"))
            require(canonical == key, "ORDERED_BASIS_HASH")
            if key in bases:
                require(bases[key] == operations, "BASIS_ALIAS_MISMATCH")
            else:
                bases[key] = operations
            case_bases[method].add(key)
        if method == "PR":
            for index, comp in enumerate(case["components"]):
                support = comp["support"]
                require(type(comp["sign"]) is int and comp["sign"] in (-1, 1), "COMPONENT_SIGN")
                require(type(support) is list and len(support) in (0, 1, 2) and len(set(support)) == len(support)
                        and all(type(q) is int and 0 <= q < n for q in support), "SYMBOLIC_COMPONENT_SUPPORT")
                supports.append(dict(index=index, basis_hash=comp["basis"]["basis_hash"], support=support, sign=comp["sign"]))
    require(set(gates) == set(registry["local_gates"]), "LOCAL_GATE_COVERAGE")
    if not artificial:
        require((n, len(gates), len(bases), len(supports)) == (8, 148, 11, 324), "FIXED_H4_SCOPE_COUNTS")
    return dict(domain="ARTIFICIAL" if artificial else "SAVED_H4", n=n, gates=gates, bases=bases,
                supports=supports, case_bases={k: sorted(v) for k, v in case_bases.items()}, block_counts=block_counts)


def diagnose(plan, budget, *, emit, authorize_before_call):
    """Exact fixed128 local/basis/support work; reservation is never refunded."""
    gate_results, basis_results = {}, {}
    with C.ctx.workprec(PRECISION):
        for h, gate in sorted(plan["gates"].items()):
            values = {}
            for direction in ("forward", "inverse"):
                authorize_before_call()
                budget.take(direction)
                values[direction] = discrepancy(gate["descriptor"], gate[direction], inverse=direction == "inverse")
                emit("local_" + h + "_" + direction + ".json",
                     dict(kind="LOCAL_GATE_DISCREPANCY", matrix_hash=h, direction=direction,
                          discrepancy_upper_exact=str(values[direction]), precision_bits=PRECISION,
                          ideal_semantics="Qiskit1.3 exact real gate on encoded binary64 parameters",
                          literal_decomposition_or_compiler_proved=False))
            gate_results[h] = values
        for key, operations in sorted(plan["bases"].items()):
            authorize_before_call()
            budget.take("basis")
            forward = product_bound([gate_results[x["gate_hash"]]["forward"] for x in operations])
            inverse = product_bound([gate_results[x["gate_hash"]]["inverse"] for x in reversed(operations)])
            pair = conjugation_bound(forward, inverse)
            basis_results[key] = pair
            emit("basis_" + key + ".json", dict(kind="ORDERED_BASIS_AND_CONTROL_BOUND", basis_hash=key,
                 forward_upper_exact=str(forward), inverse_upper_exact=str(inverse),
                 control_zero_upper_exact=str(pair), control_one_upper_exact=str(pair),
                 forward_gate_order=[x["gate_hash"] for x in operations],
                 inverse_gate_order=[x["gate_hash"] for x in reversed(operations)],
                 proof="Ideal factors unitary; telescoping spectral bound; same bound for F*D*J with D=I or unitary signed diagonal.",
                 scope="Rounded basis versus ideal gate symbols; diagonal/control synthesis and compiler not certified"))
        for comp in plan["supports"]:
            authorize_before_call()
            budget.take("support")
            require(comp["basis_hash"] in basis_results, "UNCOVERED_COMPONENT_BASIS")
            signed_controlled_identity(comp["sign"])
            emit("support_" + str(comp["index"]) + ".json", dict(kind="SYMBOLIC_SIGNED_Z_SUPPORT_COVERAGE", **comp,
                 ideal_diagonal_involution=True, DF_fragment_lambda_A_squared_is_involution=False))
        uniform = {}
        for method in ("PF", "PR"):
            count = 8 * (2 * plan["block_counts"][method] + (4 * 5 if method == "PR" else 0))
            maximum = max((basis_results[h] for h in plan["case_bases"][method]), default=Q(0))
            error = product_bound([maximum] * count)
            uniform[method] = dict(maximum_conjugations=count, local_basis_transfer_upper_exact=str(error),
                                   event_orders=[0, 2, 4] if method == "PR" else [], axes_reused=["cosine", "sine"],
                                   precisions_reused=["0.05", "0.01", "0.005", "0.001", "0.0005"],
                                   phase_scope="Fixed same real angles/signs; global and relative phases retained, not synthesized numerically here",
                                   bound_excludes="New diagonal parameter rounding, literal gate decomposition, compiler transforms, physical preparation and shot accounting")
            uniform[method]["measurement_scale_B_not_applied"] = True
    ledger = coverage_ledger([
        dict(id="saved_C_total", covers=["bias_distance", "basis_reuse", "phase_accumulation", "pi_rounding"], bound_ref="Existing C axes[].total_error_upper"),
        dict(id="new_L_uniform_basis", covers=["local_forward", "local_inverse", "basis_reuse"], bound_ref="L uniform basis bound")])
    return dict(status="LOCAL_ENCLOSURES_COMPLETE_COMBINATION_BLOCKED", domain=plan["domain"],
                precision_bits=PRECISION, uniform_basis_bounds=uniform, coverage_ledger=ledger,
                formal_certificate_accepted=False, N_G_authorized=False, automatic_retry=False,
                mandatory_STOP=True, compiler_transfer="MISSING_EVIDENCE")
