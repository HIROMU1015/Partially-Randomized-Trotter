"""Authorized Q1/Q2-L only: saved DF preparation and state-free scalar bounds.

No RNG, state solver, full interrogation, compilation, pilot, or shot allocation.
Reuse the frozen sampler/CV connection and existing rational enclosures.
"""
from dataclasses import asdict
from fractions import Fraction as F
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

from .a1_connections import connect_sampler_inputs, proposed_control_banks
from .parent_allocation_native import interval_angle
from .return_aggregation import Interval, root_interval
from .shared_cpu import module as cpu_module


SNAPSHOT = ("artifacts/rte_connected_cluster_cost_validation/"
            "h4_sto3g_d100_rank12_ld3_dt0p1_ref4_k2_connected_"
            "pilot30_max1500_hold1500_rare375_v1.hamiltonian.npz")
SNAPSHOT_SHA = "13e4b10d2347ed900fe4aa4b2238128ebe735a48988e321b9e4fabac4d092a3b"
H_HASH = "56e4df83655aa2f2f8132126f2635996516dc2cfb1bf4cb8d95490184ad631e5"
PARTITION_HASH = "9034397f5bde1f20cb712578d14245f4dac1236413d34642a75d83d867b52086"
PREPARATION_HASH = "50c77f5455cc1a3c0710ed3c7588efc957b9cead7436bd0a6d851af61738a4bc"
TAIL_HASH = "c51e35da56f3de1646896339f38da3cb40dcb8405a24d781cef134a654bab642"
CONTROL_A, CONTROL_B = "df-fragment-3:Z0Z1", "df-fragment-6:Z0"


def fraction_record(q):
    q = F(q)
    upper = float(q)
    if F.from_float(upper) < q:
        upper = math.nextafter(upper, math.inf)
    return {"exact": str(q), "float_upper_preview": upper}


def interval_record(x):
    return {"lo": str(x.lo), "hi": str(x.hi)}


def state_free_bias(connection, coefficient_abs, *, lambda_r, tau, normalization,
                    root_bits=160, angle_dps=80):
    """Bound actual grid/rounded-angle mean versus the SOURCE polynomial target.

    Target = P3(-i*tau*A)/B_f, A=sum(c_i P_i)/lambda_f, with exact binary64
    operands interpreted as rationals and ideal sign-absorbed involutions P_i.
    Let S=sum |c_i| and s=tau*S/lambda_f. The normalized exact-coefficient
    reference mixture equals P3(-i*tau*A)/B_*(s). Distribution, angle, and B
    differences are bounded separately. Outer common unitary factors have norm1.
    This does NOT certify DF-to-circuit basis equivalence or native execution.
    """
    coefficients = tuple(F.from_float(float(x)) for x in coefficient_abs)
    if (len(coefficients) != len(connection.law.labels.labels)
            or not coefficients or min(coefficients) <= 0):
        raise ValueError("positive complete retained coefficient list required")
    lam, tau_f, b_f = map(lambda x: F.from_float(float(x)),
                          (lambda_r, tau, normalization))
    if min(lam, tau_f, b_f) <= 0:
        raise ValueError("positive m0 scalar inputs required")
    total = sum(coefficients, F(0))
    s = tau_f * total / lam
    w0 = root_interval(1 + s*s, root_bits)
    r2 = root_interval(1 + (s/3)**2, root_bits)
    w2 = Interval(s*s*r2.lo/2, s*s*r2.hi/2)
    b = Interval(w0.lo+w2.lo, w0.hi+w2.hi)
    reference = tuple(Interval(w.lo/b.hi, w.hi/b.lo) for w in (w0, w2))
    reference_labels = tuple(c/total for c in coefficients)
    label_l1 = sum((abs(q-p) for q, p in zip(connection.law.labels.masses,
                                            reference_labels)), F(0))
    order_l1 = sum((max(abs(q-p.lo), abs(q-p.hi)) for q, p in zip(
        connection.law.orders.masses, reference)), F(0))
    expected_labels_upper = reference[0].hi + 3*reference[1].hi
    law_bound = order_l1 + expected_labels_upper*label_l1
    angles, angle_bound = [], F(0)
    for n, pn in zip((0, 2), reference):
        # Same binary64 division/atan as frozen rte._make_event; no event/RNG.
        actual = math.atan(float(tau)/(n+1))
        ideal = interval_angle(s/(n+1), dps=angle_dps)
        error = max(abs(F.from_float(actual)-ideal["lo"]),
                    abs(F.from_float(actual)-ideal["hi"]))
        angle_bound += pn.hi*error
        angles.append(dict(order=n, actual_hex=actual.hex(),
                           reference_atan={"lo": str(ideal["lo"]), "hi": str(ideal["hi"])},
                           error_upper=fraction_record(error)))
    normalization_bound = max(abs(b.lo/b_f-1), abs(b.hi/b_f-1))
    bound = law_bound + angle_bound + normalization_bound
    return dict(
        reference="P3(-i*tau_f*sum(c_i*P_i)/lambda_f)/B_f; exact recorded operands",
        exact_coefficient_l1=str(total), reference_s=str(s),
        reference_order_intervals=[interval_record(x) for x in reference],
        reference_normalizer=interval_record(b), root_bits=root_bits,
        angle_interval_dps=angle_dps, angles=angles,
        label_l1_upper=fraction_record(label_l1),
        order_l1_upper=fraction_record(order_l1),
        expected_labels_upper=fraction_record(expected_labels_upper),
        grid_law_bias_upper=fraction_record(law_bound),
        angle_bias_upper=fraction_record(angle_bound),
        normalization_bias_upper=fraction_record(normalization_bound),
        conditional_coordinate_bias_upper=fraction_record(bound),
        assumptions=["IID uniform 53-bit random-word model", "ideal Hermitian involutions",
                     "same ideal outer unitaries and scalar phase",
                     "selected policy implements the same abstract path operator"],
        unresolved=["independent numerical DF/basis-to-selected-circuit error bound",
                    "native/backend effects excluded from this ideal model",
                    "Q3 state payload and Q4 full prices not acquired"],
        full_numerical_bias_closed=False, measurement_authorized=False)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")


def run(output_dir, *, root=None):
    root = Path(root or Path(__file__).resolve().parents[3]).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_paths = ["src/trottertracks/algorithm_codesign/"+n for n in
                    ("a1_law_validation.py", "a1_connections.py", "a1_control_variates.py",
                     "shared_cpu.py", "return_aggregation.py", "parent_allocation_native.py")]
    source_paths += ["src/trotterlib/"+n for n in
                     ("rte.py", "df_partial_s2.py", "df_rte_tail.py",
                      "df_partial_randomized_pf.py", "rte_connected_cluster_cost_validation.py")]
    source_paths += ["scripts/tracks/algorithm_codesign/run_a1_q1_q2l.py"]
    source_paths += ["src/trottertracks/algorithm_codesign/"+n for n in
                     ("parent_allocation.py", "rte_reallocation/numeric.py",
                      "synthesis_placement/economics.py")]
    source = dict(schema="a1_q1_q2l_source_v1", source_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256={p: _sha(root/p) for p in source_paths},
        snapshot_path=SNAPSHOT, snapshot_sha256=_sha(root/SNAPSHOT),
        python_version=platform.python_version(),
        packages={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "qiskit", "mpmath")},
        scope=dict(model="linear_H4", distance_angstrom=1., basis="STO-3G", df_rank=12,
                   sector_electrons=4, system_qubits=8, ld=3, m=0, q=1,
                   delta=.02, r=1, K=2, coefficient_atol=0., axes=["X", "Y"]),
        excluded=["Q2-T", "Q3", "Q4", "Q5", "RNG/sampling", "state/ground solve",
                  "full interrogation build/compile", "variance/N/cost evaluation"])
    _write(output/"SOURCE.json", source)
    try:
        if source["snapshot_sha256"] != SNAPSHOT_SHA:
            raise ValueError("fixed snapshot mismatch; no replacement/retry")
        if source["packages"]["numpy"] != "1.26.4" or source["packages"]["qiskit"] != "1.3.0":
            raise ValueError("fixed numerical profile mismatch")
        rte = cpu_module("rte")
        loader = cpu_module("rte_connected_cluster_cost_validation")
        split = cpu_module("df_partial_randomized_pf")
        partial = cpu_module("df_partial_s2")
        h = loader.load_connected_cluster_hamiltonian_snapshot(root/SNAPSHOT)
        prep = partial.prepare_df_partial_s2(h, split.split_df_hamiltonian_by_ld(h, 3),
                 identity_policy="extract_identity_phase", coefficient_atol=0.)
        actual = (prep.hamiltonian_hash, prep.partition_hash, prep.preparation_hash,
                  prep.rte_preparation.symbolic_tail.tail_hash)
        if actual != (H_HASH, PARTITION_HASH, PREPARATION_HASH, TAIL_HASH):
            _write(output/"IDENTITY_MISMATCH.json", dict(actual=actual,
                   expected=[H_HASH, PARTITION_HASH, PREPARATION_HASH, TAIL_HASH]))
            raise ValueError("fixed preparation identity mismatch; no replacement/retry")
        tail = prep.rte_preparation.symbolic_tail
        distribution0 = rte.finite_rte_distribution(prep.exact_rte_lambda_r*.02, 2)
        tolerance = max(math.nextafter(distribution0.step_truncation_residual_bound, math.inf),
                        math.ulp(0.))
        config, distribution = rte.make_rte_config(tail, evolution_time=.02, rte_steps=1,
                            truncation_tolerance=tolerance, finite_taylor_order=2)
        connection = connect_sampler_inputs(tail.components, distribution,
            source=source["source_commit"]+":"+TAIL_HASH, validate_components=rte._validate_components)
        catalog = []
        for component, q in zip(tail.components, connection.law.labels.masses):
            record = asdict(component)
            record.update(probability_hex=component.probability.hex(),
                          coefficient_abs_hex=component.coefficient_abs.hex(),
                          grid_mass=fraction_record(q))
            catalog.append(record)
        _write(output/"CATALOG.json", dict(schema="a1_opt2_catalog_v1", identities=actual,
               component_count=len(catalog), components=catalog,
               lambda_r=prep.exact_rte_lambda_r, lambda_r_hex=prep.exact_rte_lambda_r.hex(),
               identity_coefficient=prep.extracted_identity_coefficient,
               identity_coefficient_hex=prep.extracted_identity_coefficient.hex(),
               distribution=distribution.to_dict(), config=asdict(config),
               order_choice_input_hex=[p.hex() for p in connection.order_choice_input],
               label_choice_input_hex=[p.hex() for p in connection.label_choice_input],
               order_cdf_hex=[p.hex() for p in connection.order_cdf],
               label_cdf_hex=[p.hex() for p in connection.label_cdf],
               component_probability_sum_hex=connection.component_probability_sum.hex(),
               order_grid_masses=[fraction_record(q) for q in connection.law.orders.masses]))
        cv, order, strata = proposed_control_banks(connection, law_id=TAIL_HASH,
                                                  a=CONTROL_A, b=CONTROL_B)
        patterns = ((1,1,0,0), (1,0,0,0), (0,0,0,0), (0,0,1,0), (0,0,1,1))
        masses = tuple(strata.pattern_mass(p) for p in patterns)
        assert min(masses) > 0 and sum(masses) == 1
        assert cv.joint_mean((0,1)) == cv.joint_mean((0,2)) == 0
        assert cv.joint_mean((1,2)) == cv.means[2]
        positive = sum(q > 0 for q in connection.law.labels.masses)
        zero_ids = [label for label, q in zip(connection.law.labels.labels,
                                             connection.law.labels.masses) if not q]
        q1 = dict(schema="a1_q1_result_v1", status="PASS", component_count=len(catalog),
                  positive_grid_label_count=positive, zero_grid_label_ids=zero_ids,
                  controls=dict(a=CONTROL_A, b=CONTROL_B, means=[fraction_record(q) for q in cv.means],
                    joint_moments=[[fraction_record(cv.joint_mean((i,j))) for j in range(3)]
                                   for i in range(3)]),
                  order_stratum_masses=[fraction_record(order.pattern_mass((i,))) for i in (0,1)],
                  five_strata=[dict(pattern=p, mass=fraction_record(q)) for p,q in zip(patterns,masses)],
                  exact_centering_in_grid_model=True, rng_calls=0, measurement_calls=0)
        _write(output/"Q1.json", q1)
        q2 = state_free_bias(connection, [c.coefficient_abs for c in tail.components],
              lambda_r=prep.exact_rte_lambda_r, tau=config.dimensionless_step_time,
              normalization=config.distribution_normalization)
        _write(output/"Q2L.json", q2)
        runtime_sources = {}
        for name, loaded in tuple(sys.modules.items()):
            filename = getattr(loaded, "__file__", None)
            if filename and (name.startswith("_bf1_shared_cpu") or name.startswith("trottertracks")):
                path = Path(filename).resolve()
                if path.is_relative_to(root/"src") and path.suffix == ".py":
                    runtime_sources[str(path.relative_to(root))] = _sha(path)
        _write(output/"RUNTIME_SOURCES.json", runtime_sources)
        bias = F(q2["conditional_coordinate_bias_upper"]["exact"])
        reference_epsilon = F.from_float(.266146579872968)
        if bias >= reference_epsilon:
            raise ValueError("no positive conditional arithmetic residual")
        result = dict(schema="a1_q1_q2l_result_v1", status="Q1_PASS_Q2L_CONDITIONAL_STOP",
            source_commit=source["source_commit"], q1_pass=True,
            conditional_coordinate_bias_upper=fraction_record(bias),
            reference_coordinate_epsilon_hex=float(reference_epsilon).hex(),
            conditional_arithmetic_residual=fraction_record(reference_epsilon-bias),
            full_numerical_residual_certified=False, next_science_started=False,
            elapsed_seconds=time.monotonic()-started,
            artifact_sha256={p.name:_sha(p) for p in sorted(output.glob("*.json"))})
        _write(output/"RESULT.json", result)
        _write(output/"STOP.json", dict(status="MANDATORY_STOP", completed=["Q1", "Q2-L scalar bound"],
                 pending=q2["unresolved"], no_progression_to=["Q2-T", "Q3", "Q4", "Q5"]))
        return result
    except Exception as exc:
        _write(output/"STOP.json", dict(status="STOPPED", error_type=type(exc).__name__,
                 error=str(exc), no_retry=True, elapsed_seconds=time.monotonic()-started))
        raise
