"""Fixed, state-independent B2 normalization screen; no circuit or signal calls."""
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from types import SimpleNamespace


SOURCE = "66a41ad9219f12377e7d325e272c1a90d66ab7e5"
INPUT_COMMIT = "bf5d3b2405b2bec4f88dd0589b75dd737b13e052"
INPUT_PATH = "artifacts/pr2_s0_s1_validation/2026-09-28/h4_1p00_rank12_development_v1.npz"
INPUT_SHA = "3bc92e92c595a50eadf97c80ed8641adbb214b14e6e94b7a28ac08e8c2e0f80a"
DF_HASH = "de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424"
ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/trotterlib/rte.py", "src/trotterlib/df_rte_tail.py",
    "src/trotterlib/df_trotter/decompose.py", "src/trotterlib/df_trotter/ops.py",
    "src/trotterlib/pr2_matched_accuracy_m1_execution.py",
    "src/trotterlib/rpe_target_round_horizon_validation.py",
    "src/trotterlib/b2_rpe_connection.py",
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def resources():
    return dict(available_cpus=len(os.sched_getaffinity(0)),
        memory={line.split(":")[0]: line.split(":", 1)[1].strip()
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith(("MemTotal:", "MemAvailable:"))})


def main(output):
    import numpy as np
    from trotterlib.df_trotter.decompose import diag_hermitian
    from trotterlib.df_rte_tail import exact_df_diagonal_coefficients
    from trotterlib.pr2_matched_accuracy_m1_execution import _explicit_cutoff_tolerance
    from trotterlib.rpe_target_round_horizon_validation import required_rpe_round_horizon
    from trotterlib.rte import RTEComponent, make_rte_config, finite_rte_attenuation

    if output.exists():
        raise FileExistsError("Fresh output directory required")
    execution_head = git("rev-parse", "HEAD").decode().strip()
    source_hashes = {}
    for path in SOURCE_PATHS:
        data = (ROOT/path).read_bytes()
        if data != git("show", SOURCE+":"+path):
            raise ValueError("Changed scientific source: "+path)
        source_hashes[path] = sha(data)
    before = resources()
    started = datetime.now(timezone.utc).isoformat()
    raw = git("show", INPUT_COMMIT+":"+INPUT_PATH)
    if sha(raw) != INPUT_SHA:
        raise ValueError("Fixed input SHA mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        # Never decode state, sector-state, or one-body data for this screen.
        metadata = json.loads(str(saved["metadata_json"].item()))
        weights = saved["lambdas"].copy()
        factors = saved["g_matrices"].copy()
    if (metadata["hamiltonian_hash"] != DF_HASH or weights.shape != (12,)
            or factors.shape != (12, 8, 8) or not metadata["hamiltonian_metadata"]["hermitized"]):
        raise ValueError("DF12 context mismatch")

    # Same numerical kernels/order as build_df_blocks -> extract_df_diagonal_tail,
    # stopping before Gaussian circuit generation. No fake executable basis.
    coefficients, identities, fragments = [], [], []
    exact_zeros = 0
    for index in range(3, 12):
        _rotation, eta = diag_hermitian(factors[index], sort="descending_abs", assume_hermitian=True)
        entries = exact_df_diagonal_coefficients(eta, weights[index])
        fragments.append(dict(index=index, signed_weight=float(weights[index]), eta=eta.tolist()))
        for support, coefficient in entries:
            item = dict(fragment=index, support=list(support), coefficient=coefficient)
            if coefficient == 0.0:
                exact_zeros += 1
            elif not support:
                identities.append(item)
            else:
                coefficients.append(item)
    coefficients.sort(key=lambda v: (f"df-fragment-{v['fragment']}", len(v["support"]), v["support"]))
    identities.sort(key=lambda v: f"df-fragment-{v['fragment']}")
    lam = math.fsum(abs(v["coefficient"]) for v in coefficients)
    identity = math.fsum(v["coefficient"] for v in identities)
    projection_hash = sha(json.dumps(dict(input=INPUT_SHA, L_D=3, coefficients=coefficients,
        identity_policy="extract_identity_phase"), sort_keys=True).encode())
    components = tuple(RTEComponent(
        component_id=f"df-fragment-{v['fragment']}:"+"".join(f"Z{k}" for k in v["support"]),
        probability=abs(v["coefficient"])/lam, coefficient_abs=abs(v["coefficient"]),
        coefficient_sign=1 if v["coefficient"] > 0 else -1,
        df_fragment_id=f"df-fragment-{v['fragment']}", diagonal_pauli_support=tuple(v["support"]))
        for v in coefficients)
    tail = SimpleNamespace(tail_id="B2_LD3_normalization_projection_only", tail_hash=projection_hash,
                           lambda_r=lam, components=components)
    tau = lam*.02/4
    config, distribution = make_rte_config(tail, evolution_time=.02, rte_steps=4,
        finite_taylor_order=2, truncation_tolerance=_explicit_cutoff_tolerance(tau, 2), seed=0)
    assert config.dimensionless_step_time == tau
    assert distribution.orders == (0, 2)
    horizon = required_rpe_round_horizon(target_energy_precision=.00015936001019904,
                                        beta_rpe=.4, delta_time=.02)
    b = distribution.exact_finite_distribution
    log_b = math.log(b)
    rows = []
    with localcontext() as ctx:
        ctx.prec = 80
        t = Decimal.from_float(tau)
        b_formula = (1+t*t).sqrt() + t*t/2*(1+(t/3)**2).sqrt()
        log_b_formula = b_formula.ln()
        log_b_fixed = Decimal.from_float(b).ln()
        for m in range(horizon["maximum_round_index_M"]+1):
            q, r = 1 << m, 4
            n = q*r
            log_B = n*log_b
            A = finite_rte_attenuation(config, tail_evolutions=q)
            B = math.exp(log_B) if log_B <= math.log(sys.float_info.max) else None
            log_A_decimal = -Decimal(n)*log_b_formula
            a_decimal = log_A_decimal.exp()
            a_fixed = (-Decimal(n)*log_b_fixed).exp()
            rows.append(dict(m=m, Q_m=q, T_m=q*.02, r_m=r, K_m=2, lambda_R=lam, tau_m=tau,
                event_count=n, normalization_per_event=b, normalization_total=B,
                attenuation=A, log_attenuation=-log_B, log_normalization_total=log_B,
                attenuation_decimal80=str(a_decimal), normalization_total_decimal80=str((-log_A_decimal).exp()),
                log_attenuation_decimal80=str(log_A_decimal),
                attenuation_decimal80_fixed_binary64_b=str(a_fixed),
                relative_difference_vs_decimal_formula=float(abs(Decimal.from_float(A)-a_decimal)/a_decimal),
                relative_difference_vs_decimal_fixed_b=float(abs(Decimal.from_float(A)-a_fixed)/a_fixed),
                attenuation_binary64_underflow=A == 0.0,
                attenuation_binary64_subnormal=0 < A < sys.float_info.min,
                normalization_binary64_overflow=B is None))
        precision = dict(scalar_crosscheck="Decimal precision 80; same binary64 tau, no high-precision DF solve",
            b_decimal_formula=str(b_formula), b_binary64_hex=b.hex(), tau_binary64_hex=tau.hex(),
            max_relative_attenuation_difference_vs_formula=max(x["relative_difference_vs_decimal_formula"] for x in rows),
            max_relative_attenuation_difference_vs_fixed_b=max(x["relative_difference_vs_decimal_fixed_b"] for x in rows),
            interval_certification=False)
    if not horizon["minimal_power_of_two_horizon"]:
        raise ValueError("Nonminimal horizon")
    if not (len(coefficients) == 324 and math.isfinite(lam) and lam > 0
            and math.isclose(math.fsum(c.probability for c in components), 1., rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Fixed B2 coefficient extraction inconsistent")
    if not math.isclose(lam, .5832704246339983, rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("Coefficient-only extraction differs from saved B2 lambda_R")
    result = dict(schema="b2_fixed_rpe_attenuation_screen_v1", status="FIXED_SCREEN_COMPLETE",
        mandatory_stop=True, started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        source=dict(library_commit=SOURCE, execution_head=execution_head, source_files_sha256=source_hashes,
                    runner_path="scripts/run_b2_rpe_attenuation_screen.py", runner_sha256=sha(Path(__file__).read_bytes())),
        input=dict(commit=INPUT_COMMIT, path=INPUT_PATH, sha256=INPUT_SHA, df_hamiltonian_hash=DF_HASH,
                   decoded_members=["metadata_json", "lambdas", "g_matrices"], state_decoded=False),
        conditions=dict(system="linear H4 / 1.00 Angstrom / STO-3G", DF_rank=12,
            L_D=3, deterministic_fragments=[0,1,2], randomized_fragments=list(range(3,12)),
            retained_one_body=True, nuclear_scalar_hex=metadata["constant_hex"],
            identity_policy="extract_identity_phase", coefficient_atol=0., PF="partial_S2",
            r=4, K=2, delta=.02, epsilon_E_screen_only=.00015936001019904,
            success_probability_screen_only=.95, beta_RPE_screen_only=.4),
        coefficient_projection=dict(lambda_R=lam, extracted_tail_identity=identity,
            identities=identities, components=coefficients, fragments=fragments,
            zero_coefficients_excluded=exact_zeros, probability_sum=math.fsum(c.probability for c in components),
            projection_hash=projection_hash, executable_tail_hash=False,
            basis_operations_generated=False, state_independent=True),
        canonical_law=dict(taylor_distribution=config.taylor_distribution, orders=list(distribution.orders),
            order_weights=list(distribution.unnormalized_order_weights),
            order_probabilities=list(distribution.order_probabilities), normalization_per_event=b,
            corrected_polynomial="I - i*tau*h - tau^2*h^2/2 + i*tau^3*h^3/6; h=H_R/lambda_R",
            normalization_formula="b_2(tau)=sqrt(1+tau^2)+(tau^2/2)*sqrt(1+(tau/3)^2)",
            scalar_event_phase={"0":"+1", "2":"-1"}, sampling_performed=False,
            truncation_tolerance_role="existing fixed-K constructor consistency only; no phase-error allocation"),
        horizon=horizon, rounds=rows, precision=precision,
        environment=dict(host=socket.gethostname(), python=sys.version,
            executable=sys.executable, versions={p:importlib.metadata.version(p) for p in
                ("numpy","scipy","qiskit","openfermion","openfermionpyscf")},
            platform=platform.platform(), resources_before=before, resources_after=resources(),
            numerical_workers="one small scalar/matrix workload; no resource cap or worker pool"),
        not_evaluated=["original_H/state correspondence", "energy_interval/alias/branch",
            "PF/RTE/stat/model/implementation phase allocation", "signal/radius", "quantum_shots",
            "circuit_build/compile", "sampling/prices", "total_RPE_cost", "energy_success_guarantee"],
        guarantee_status="normalization_diagnostic_only_not_energy_or_success_certified")
    output.mkdir(parents=True)
    (output/"result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    with (output/"rounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(status=result["status"], horizon=horizon, lambda_R=lam, tau=tau,
        normalization_per_event=b, first_round=rows[0], last_round=rows[-1], precision=precision), indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: run_b2_rpe_attenuation_screen.py FRESH_OUTPUT_DIRECTORY")
    main(Path(sys.argv[1]).resolve())
