"""One new H4 B2/S2 path, adapted from the pinned geometry implementation.

This is not the old geometry runner and does not implement its launch contract.
Only existing frozen arrays and an explicit already-fixed partition are accepted.
No generation, eigensolver, state validation, rank selection or sampling occurs here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .pipeline import PipelineError, ScientificContext, StageResult

DONOR_COMMIT = "821cc0f6ced57168293ea166644024867b064ffb"
GEOMETRY_PATH = "src/trottertracks/resource_applicability/h4_geometry/"
H4_DISTANCES = ("0.70", "0.80", "0.90", "1.10", "1.40", "1.60")


@dataclass(frozen=True)
class FrozenH4Input:
    arrays: Mapping[str, Any]
    input_id: str
    state_id: str
    fixed_partition: Any
    binding: tuple[str, ...]


class H4GeometryAdapter:
    """B2 non-B3, explicit-order, second-order fixed-signal boundary only."""

    def validate(self, c: ScientificContext) -> None:
        if c.representation != "DF" or c.df_rank != 12 or c.basis != "STO-3G":
            raise PipelineError("H4 geometry requires DF rank12 / STO-3G")
        if c.geometry not in tuple("linear-H4:" + x + "Angstrom" for x in H4_DISTANCES):
            raise PipelineError("geometry outside the pinned H4 distance scope")
        if c.sector != "N=4,Sz=0;8-spin-orbitals" or c.formula != "2nd":
            raise PipelineError("H4 sector or S2 formula mismatch")
        if c.q != 1:
            raise PipelineError("P1 connects one S2 step only; repeated q is not connected")
        if c.stage_order != ("deterministic_half", "finite_rte", "reverse_deterministic_half"):
            raise PipelineError("H4 stage order mismatch")
        if c.rank_policy != "frozen_geometry_rank12_generated_order":
            raise PipelineError("H4 ordering must stay explicit; no reranking")
        if (c.deterministic_indices != tuple(range(c.ld)) or
                c.randomized_indices != tuple(range(c.ld, c.df_rank))):
            raise PipelineError("H4 requires its frozen generated-order prefix")
        if not c.randomized_indices or c.rte_steps < 1:
            raise PipelineError("this adapter is the B2 randomized path only")
        if c.rte_law != "rte_even_taylor_paired" or c.normalization_policy != "once_per_complete_occurrence":
            raise PipelineError("H4 law or normalization policy mismatch")
        if c.estimand != "fixed_complex_signal" or c.metric != "rz_count":
            raise PipelineError("H4 signal accounting cannot silently become phase-RPE accounting")
        if c.cost_scope != "single_hadamard_interrogation_without_state_preparation":
            raise PipelineError("H4 requires explicit full-wrapper P=0 cost scope")
        if c.source_stop_status != "PRESERVED_NOT_RESTARTED":
            raise PipelineError("old STOP cannot be cleared or resumed by this adapter")

    def input(self, c: ScientificContext, data: FrozenH4Input, core) -> StageResult:
        self.validate(c)
        if not isinstance(data, FrozenH4Input) or data.binding != c.binding:
            raise PipelineError("frozen input/cohort binding mismatch")
        if data.input_id != c.input_id or data.state_id != c.state_id:
            raise PipelineError("input or raw-state identity mismatch")
        required = {"one", "nuclear", "lambdas", "G"}
        if not required <= data.arrays.keys():
            raise PipelineError("missing frozen H4 arrays; generation is not a fallback")
        a = data.arrays
        if len(a["lambdas"]) != c.df_rank or len(a["G"]) != c.df_rank:
            raise PipelineError("frozen rank mismatch")
        p = data.fixed_partition
        if (p.ld != c.ld or tuple(p.deterministic_block_indices) != c.deterministic_indices
                or tuple(p.randomized_block_indices) != c.randomized_indices):
            raise PipelineError("fixed partition mismatch; no split selection is performed")
        # Same represented sum as donor inputs.py:191-202 / signal.py:96.
        # 'one' already includes the correction; nuclear is applied exactly once.
        h = core.hamiltonian(constant=a["nuclear"], one_body=a["one"],
                             lambdas=a["lambdas"], g_matrices=tuple(a["G"]),
                             metadata={"input_id": c.input_id, "state_id": c.state_id,
                                       "donor_commit": DONOR_COMMIT,
                                       "partition_policy": "explicit_ordered_partition"})
        return StageResult("input", c.binding, "SYNTHETIC" if c.input_kind == "synthetic"
                           else "UNQUALIFIED", {"input_id": c.input_id, "state_id": c.state_id},
                           "conversion only; numerical/operator equivalence not verified",
                           payload={"hamiltonian": h, "partition": p})


# Selective port of donor signal.py:200-203, preserving its existing signature/body.
# This function is bound for a later authorized stage, never evaluated in P1 tests.
def shots(B, bias, epsilon):
    import math
    if not (math.isfinite(B) and B >= 1 and math.isfinite(bias) and bias >= 0
            and math.isfinite(epsilon) and epsilon > 0):
        raise PipelineError("shot inputs")
    allowance = epsilon / math.sqrt(2) - bias
    return None if allowance <= 0 else math.ceil(2 * B * B / (allowance * allowance) * math.log(2 / 0.025))


# Selective port of donor circuits.py:290-298. No import/build on module import.
def wrapper(evolution, axis):
    from qiskit import QuantumCircuit
    if axis not in ("cosine", "sine"):
        raise PipelineError("axis")
    qc = QuantumCircuit(evolution.num_qubits, 1)
    anc = evolution.num_qubits - 1
    qc.h(anc)
    qc.compose(evolution, inplace=True)
    if axis == "sine":
        qc.sdg(anc)
    qc.h(anc)
    qc.measure(anc, 0)
    return qc
