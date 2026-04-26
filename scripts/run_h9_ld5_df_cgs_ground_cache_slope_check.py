from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from trotterlib.df_hamiltonian import build_df_h_d_from_molecule  # noqa: E402
from trotterlib.df_partial_randomized_pf import (  # noqa: E402
    fit_df_cgs_with_perturbation,
    rank_df_fragments,
    split_df_hamiltonian_by_ld,
)


def main() -> int:
    molecule_type = 9
    ld = 5
    gpu_ids = ("0", "1", "2", "3", "4")
    out_dir = PROJECT_ROOT / "artifacts" / "partial_randomized_pf"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "H9_ld5_df_cgs_ground_cache_slope_check.json"
    partial_path = out_dir / "H9_ld5_df_cgs_ground_cache_slope_check.partial.json"

    hamiltonian, sector = build_df_h_d_from_molecule(molecule_type)
    ranked = rank_df_fragments(hamiltonian)
    partition = split_df_hamiltonian_by_ld(
        hamiltonian,
        ld,
        ranked_fragments=ranked,
    )

    rows = []
    for pf_label in ("2nd", "4th", "8th(Morales)"):
        result = fit_df_cgs_with_perturbation(
            hamiltonian,
            sector,
            partition,
            pf_label,
            evolution_backend="gpu",
            gpu_ids=gpu_ids,
            parallel_times=True,
            use_parameterized_template=True,
            use_ground_state_cache=True,
        )
        row = {
            "molecule_type": molecule_type,
            "df_rank_actual": result.df_rank_actual,
            "ld": result.ld,
            "pf_label": result.pf_label,
            "order": result.order,
            "c_gs_d": result.coeff,
            "fit_slope": result.fit_slope,
            "fit_coeff": result.fit_coeff,
            "errors": list(result.perturbation_errors),
            "t_values": list(result.t_values),
            "ground_state_cache": result.metadata.get("ground_state_cache"),
            "use_parameterized_template": result.metadata.get(
                "use_parameterized_template"
            ),
            "template_transpile_s": (
                result.metadata.get("parameterized_template_profile") or {}
            ).get("transpile_s"),
            "gpu_wall_sum_s": sum(
                float(profile.get("total_s", 0.0))
                for profile in result.simulation_profiles
            ),
            "per_t_has_transpile": [
                "transpile_s" in profile for profile in result.simulation_profiles
            ],
        }
        rows.append(row)
        partial_path.write_text(
            json.dumps(rows, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(row, sort_keys=True), flush=True)

    final_path.write_text(
        json.dumps(rows, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"wrote {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
