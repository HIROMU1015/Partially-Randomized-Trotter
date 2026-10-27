"""Explicit artificial fixture and mocked existing APIs for P1 wiring tests.

Only zero-array fixtures are allocated; no operator/state/law/circuit/price is evaluated.
The callbacks exercise df_core_stages calls instead of a separate demo pipeline.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

from .pipeline import CompilerProfile, EvidenceLocator, ScientificContext, StageProviders, StageResult
from .h4_geometry_pipeline import FrozenH4Input, DONOR_COMMIT, GEOMETRY_PATH
from . import df_core_stages as stages


def synthetic_context():
    return ScientificContext(
        study_id="P1_ARTIFICIAL_H4", cohort_id="SYNTHETIC_ONLY_NO_RESEARCH_COHORT",
        result_version="p1-wiring-v1", source=EvidenceLocator(
            DONOR_COMMIT,
            "artifacts/resource_applicability/track_a_h4_lazy_identity_run05/2026-10-06/source_freeze_v1.json",
            "SOURCE"),
        evidence=(EvidenceLocator(DONOR_COMMIT,
                  GEOMETRY_PATH + "inputs.py", "DONOR_INPUT_CONTRACT_NOT_NEW_RESULT"),),
        representation="DF", geometry="linear-H4:0.90Angstrom", basis="STO-3G",
        sector="N=4,Sz=0;8-spin-orbitals", state_id="ARTIFICIAL_STATE_TOKEN",
        input_id="ARTIFICIAL_INPUT_TOKEN", df_rank=12,
        rank_policy="frozen_geometry_rank12_generated_order", ld=3,
        deterministic_indices=(0, 1, 2), randomized_indices=tuple(range(3, 12)),
        formula="2nd", q=1,
        stage_order=("deterministic_half", "finite_rte", "reverse_deterministic_half"),
        evolution_time=0.8, rte_steps=1, finite_taylor_order=2,
        rte_law="rte_even_taylor_paired", normalization_policy="once_per_complete_occurrence",
        seed=0, precision="ARTIFICIAL_NO_NUMERICAL_EVALUATION",
        applicable_domain="P1_MOCK_WIRING_ONLY", estimand="fixed_complex_signal",
        metric="rz_count", cost_scope="single_hadamard_interrogation_without_state_preparation",
        compiler=CompilerProfile(("rz", "sx", "x", "cx"), None, None, 1, None, None, 17,
                                 "MOCK_NOT_QISKIT"),
        source_stop_status="PRESERVED_NOT_RESTARTED", input_kind="synthetic")


def synthetic_fixture(context=None):
    import numpy as np  # artificial array allocation only, never a scientific kernel
    c = context or synthetic_context()
    partition = SimpleNamespace(ld=c.ld, deterministic_block_indices=c.deterministic_indices,
                                randomized_block_indices=c.randomized_indices)
    # Correct input array types/shapes, with no physical H4 input or operator evaluation.
    arrays = {"one": np.zeros((8, 8), dtype=np.complex128), "nuclear": 0.0,
              "lambdas": np.zeros(12, dtype=np.float64),
              "G": tuple(np.zeros((8, 8), dtype=np.complex128) for _ in range(12))}
    for value in (arrays["one"], arrays["lambdas"], *arrays["G"]):
        value.setflags(write=False)
    data = FrozenH4Input(arrays,
                         c.input_id, c.state_id, partition, c.binding)
    preparation = SimpleNamespace(rte_preparation=SimpleNamespace(
        symbolic_tail="ARTIFICIAL_TAIL", component_specs=("ARTIFICIAL_SPEC",)),
        num_system_qubits=8)
    config = SimpleNamespace(tail_id="ARTIFICIAL_TAIL_ID", tail_hash="ARTIFICIAL_TAIL_HASH")
    core = stages.CoreBindings(
        hamiltonian=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
        prepare_s2=Mock(return_value=preparation),
        configure_rte=Mock(return_value=(config, "ARTIFICIAL_DISTRIBUTION")),
        event_sequence=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
        step_request=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
        s2_builder=Mock(return_value=SimpleNamespace(build_step=Mock(
            return_value=SimpleNamespace(circuit="ARTIFICIAL_EVOLUTION_NO_CIRCUIT")))),
        compile_cost=Mock(return_value=SimpleNamespace(rz_count=None)),
        compiler_settings=Mock(side_effect=lambda **kw: SimpleNamespace(**kw)),
        rpe_accounting=Mock(side_effect=AssertionError("wrong estimand")),
        signal_shots=Mock(return_value=None),
        signal_wrapper=Mock(return_value="ARTIFICIAL_WRAPPER_NO_CIRCUIT"))

    def prepare(ctx, previous):
        p = previous.payload
        value = stages.prepare(core, p["hamiltonian"], p["partition"])
        return StageResult("pf_pr", ctx.binding, "SYNTHETIC",
                           {"formula": ctx.formula, "stage_order": ctx.stage_order},
                           payload={"preparation": value})

    def finite(ctx, previous):
        preparation = previous.payload["preparation"]
        config, distribution = stages.finite_rte(core, ctx, preparation,
                                                truncation_tolerance=1e-10)
        events = tuple("ARTIFICIAL_EVENT" for _ in range(ctx.rte_steps))
        request = stages.event_request(core, ctx, preparation, config, distribution, events=events)
        return StageResult("finite_rte_event", ctx.binding, "SYNTHETIC",
                           {"rte_law": ctx.rte_law, "normalization": None,
                            "sampling_performed": False}, payload={"request": request})

    def accounting(ctx, previous):
        # The dependency-injected shots call is a mock; no bias/signal is evaluated.
        record = stages.account_fixed_signal(core, ctx, normalization=1.0,
                                              bias_by_axis={"cosine": 0.0, "sine": 0.0},
                                              epsilon=0.01)
        return StageResult("accounting", ctx.binding, "SYNTHETIC", record,
                           "mock calls only; N/G remain None and unqualified",
                           payload=previous.payload)

    def compiled(ctx, previous):
        cost = stages.compile_axis(core, ctx, previous.payload["request"], axis="cosine")
        return StageResult("compiled_cost", ctx.binding, "SYNTHETIC",
                           {"rz_count": cost.rz_count, "N": None, "G": None,
                            "scientific_qualification": "NOT_ACQUIRED",
                            "compiler_profile": ctx.compiler.qiskit_version},
                           "mock compile boundary; no Qiskit build/transpile")

    return data, StageProviders("synthetic", core, prepare, finite, accounting, compiled)
