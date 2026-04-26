from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from qiskit.circuit import Parameter

from trotterlib.df_hamiltonian import build_df_h_d_from_molecule, solve_df_ground_state
from trotterlib.df_gpu_statevector import build_parameterized_gpu_template
from trotterlib.df_partial_randomized_pf import (
    _collect_df_perturbation_errors,
    _fit_errors,
    _set_df_time_worker_template,
    _simulate_df_time_task,
    _to_qiskit_state_order,
    build_df_hd_trotter_blocks,
    default_perturbation_t_values,
    rank_df_fragments,
    select_df_h_d,
    split_df_hamiltonian_by_ld,
)
from trotterlib.df_trotter.circuit import build_df_trotter_circuit


def timed(label: str, fn):
    t0 = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - t0


def main() -> int:
    molecule_type = 5
    pf_label = "8th(Morales)"
    ld = 5
    gpu_ids = ("0", "1", "2", "3", "4")
    t_values = default_perturbation_t_values(molecule_type, pf_label)

    timings: dict[str, float] = {}

    (hamiltonian, sector), timings["build_df_hamiltonian_s"] = timed(
        "build_df_hamiltonian",
        lambda: build_df_h_d_from_molecule(molecule_type),
    )
    ranked, timings["rank_fragments_s"] = timed(
        "rank_fragments",
        lambda: rank_df_fragments(hamiltonian),
    )
    partition, timings["split_by_ld_s"] = timed(
        "split_by_ld",
        lambda: split_df_hamiltonian_by_ld(
            hamiltonian,
            ld,
            ranked_fragments=ranked,
        ),
    )
    h_d, timings["select_h_d_s"] = timed(
        "select_h_d",
        lambda: select_df_h_d(hamiltonian, partition),
    )
    ground_state, timings["solve_ground_state_s"] = timed(
        "solve_ground_state",
        lambda: solve_df_ground_state(h_d, sector, tol=1e-10, expand_state=True),
    )
    state_flat, timings["bit_reverse_state_s"] = timed(
        "bit_reverse_state",
        lambda: _to_qiskit_state_order(ground_state.state_vector, h_d.n_qubits),
    )
    blocks, timings["build_trotter_blocks_s"] = timed(
        "build_trotter_blocks",
        lambda: build_df_hd_trotter_blocks(h_d),
    )
    time_parameter = Parameter("t")
    template_qc = build_df_trotter_circuit(
        blocks,
        time=time_parameter,
        num_qubits=h_d.n_qubits,
        pf_label=pf_label,
        energy_shift=h_d.constant,
    )
    template, timings["build_parameterized_gpu_template_s"] = timed(
        "build_parameterized_gpu_template",
        lambda: build_parameterized_gpu_template(
            template_qc,
            state_flat,
            time_parameter_name=time_parameter.name,
            gpu_ids=gpu_ids,
            optimization_level=0,
        ),
    )

    task_args = [
        (
            float(t_value),
            tuple(blocks),
            int(h_d.n_qubits),
            pf_label,
            float(h_d.constant),
            state_flat,
            gpu_ids[idx % len(gpu_ids)],
            1,
            0,
            False,
        )
        for idx, t_value in enumerate(t_values)
    ]

    t0 = time.perf_counter()
    import multiprocessing as mp

    _set_df_time_worker_template(template)
    try:
        with mp.get_context("fork").Pool(
            processes=len(task_args),
            initializer=_set_df_time_worker_template,
            initargs=(template,),
        ) as pool:
            raw_results = list(pool.map(_simulate_df_time_task, task_args, chunksize=1))
    finally:
        _set_df_time_worker_template(None)
    timings["gpu_time_points_parallel_wall_s"] = time.perf_counter() - t0

    raw_results.sort(key=lambda item: item[0])
    final_state_list = [(time_value, evolved) for time_value, evolved, _ in raw_results]
    profiles = [dict(profile) for _, _, profile in raw_results]
    times_out, errors = _collect_df_perturbation_errors(
        final_state_list,
        float(np.real(ground_state.energy)),
        state_flat,
    )
    coeff, fixed_coeff, fit_slope, fit_coeff = _fit_errors(
        pf_label=pf_label,
        times_out=times_out,
        perturbation_errors=errors,
    )

    profile_stage_sums: dict[str, float] = {}
    for profile in profiles:
        for key, value in profile.items():
            if key.startswith("template_prepare_"):
                continue
            if key.endswith("_s") and isinstance(value, (int, float)):
                profile_stage_sums[key] = profile_stage_sums.get(key, 0.0) + float(value)

    result = {
        "molecule_type": molecule_type,
        "pf_label": pf_label,
        "ld": ld,
        "df_rank_actual": hamiltonian.n_blocks,
        "num_qubits": hamiltonian.n_qubits,
        "sector_dim": sector.dimension,
        "lambda_r": partition.lambda_r,
        "t_values": list(times_out),
        "errors": list(errors),
        "c_gs_d": coeff,
        "fixed_order_coeff": fixed_coeff,
        "fit_slope": fit_slope,
        "fit_coeff": fit_coeff,
        "ground_state_energy": float(ground_state.energy),
        "ground_state_residual_norm": float(ground_state.residual_norm),
        "timings": timings,
        "parameterized_template_profile": template.prepare_profile,
        "gpu_profile_stage_sums": profile_stage_sums,
        "gpu_profiles": profiles,
    }
    out_path = PROJECT_ROOT / "artifacts" / "partial_randomized_pf" / "H5_8th_Morales_ld5_cgs_profile.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
