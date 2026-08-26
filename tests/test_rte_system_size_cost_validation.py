from __future__ import annotations

import numpy as np
import qiskit

from trotterlib.df_hamiltonian import DFHamiltonian
from trotterlib.rte import CompilerSettings
from trotterlib.rte_system_size_cost_validation import (
    validate_system_size_paired_cluster_models,
    validate_system_size_paired_payload,
    write_system_size_paired_validation,
)


def _hamiltonian() -> DFHamiltonian:
    return DFHamiltonian(
        constant=0.11,
        one_body=np.asarray([[0.2]], dtype=np.complex128),
        lambdas=np.asarray([0.7]),
        g_matrices=(np.asarray([[1.0]], dtype=np.complex128),),
        metadata={"name": "system-size-cost-validation-toy"},
    )


def _compiler() -> CompilerSettings:
    return CompilerSettings(
        basis_gates=("rz", "sx", "x", "cx"),
        backend_name=None,
        coupling_map=None,
        optimization_level=1,
        layout_method=None,
        routing_method=None,
        transpiler_seed=17,
        qiskit_version=qiskit.__version__,
    )


def test_system_size_grid_pairs_k3_and_k4_trajectories(tmp_path) -> None:
    payload = validate_system_size_paired_cluster_models(
        _hamiltonian(),
        ld=0,
        reference_delta_time=0.08,
        reference_rte_steps=4,
        compiler=_compiler(),
        common_sample_count=2,
        single_rare_sample_count=2,
        seed=411,
        sequence_lengths=(4,),
        cluster_lengths=(3, 4),
        maximum_workers=1,
        persistent_cache_path=tmp_path / "cache.sqlite",
        checkpoint_directory=tmp_path / "checkpoints",
        provenance={"test": True},
    )

    validate_system_size_paired_payload(payload)
    assert set(payload["model_summaries"]) == {"k1_k3", "k1_k4"}
    assert len(list((tmp_path / "checkpoints").glob("*.json"))) == 10
    k3 = payload["paired_strata_by_model"]["k1_k3"]["4"]
    k4 = payload["paired_strata_by_model"]["k1_k4"]["4"]
    for key in k3:
        assert (
            k3[key]["event_stream_rolling_digest"]
            == k4[key]["event_stream_rolling_digest"]
        )
        assert k3[key]["actual_statistics"] == k4[key]["actual_statistics"]
    output = tmp_path / "system-size.json"
    write_system_size_paired_validation(payload, output)
    assert output.exists()
