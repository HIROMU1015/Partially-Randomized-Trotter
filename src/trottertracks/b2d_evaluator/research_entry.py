"""Research B development entry: fixed locators and artificial DAG transport only.

The historical scientific stage planners/backends remain independent and paused.
This entry never loads physical input or dispatches those stage implementations.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path

from trottertracks.b2d_adapter.adapter import DContext, BlockExponential, actual_adjoint
from .evaluation import Settings, Graph, Node, evaluate

FROZEN_BASE = "e87375fa9a998dce80792be0f15597c747f00d83"
STAGES = {'A0': {'stage': 'A0', 'module': 'trottertracks.b2d_evaluator.a0_reference', 'law': 'reference', 'source_commit': '1e3921ba16e5d531e9a3010cffa9ff717609a93a', 'result_commit': '31662b49157b15f2d1c03eef1c2faa07e2f905e7', 'report': 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/31662b49157b15f2d1c03eef1c2faa07e2f905e7/docs/research/research_b_outer_a0_execution_20261025.md', 'stop': 'artifacts/research_b_outer_a0/20261025/independent-once/STOP.json'}, 'B0': {'stage': 'B0', 'module': 'trottertracks.b2d_evaluator.b0_native', 'law': 'native', 'source_commit': 'b190fe28b98d2f94e1509e19cd3d36a63b2bb99e', 'result_commit': '0db9b3ed189eb6cd9d66e070993dbd0e72d29104', 'report': 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/0db9b3ed189eb6cd9d66e070993dbd0e72d29104/docs/research/research_b_outer_b0_recovery_20261025.md', 'stop': 'artifacts/research_b_outer_b0/20261025/corrected-2/STOP.json'}, 'B1': {'stage': 'B1', 'module': 'trottertracks.b2d_evaluator.b1_native', 'law': 'IID', 'source_commit': '11e2532a62d54a65f8fad5489be6b135f826011b', 'result_commit': 'bef63edbd8cc564444b44e4e8cf23baa90f05fe5', 'report': 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/bef63edbd8cc564444b44e4e8cf23baa90f05fe5/docs/research/research_b_outer_b1_expected_native_cost_20261025.md', 'stop': 'artifacts/research_b_outer_b1/20261025/iid256-once/STOP.json'}, 'Outer': {'stage': 'Outer', 'module': 'trottertracks.b2d_evaluator.outer', 'law': 'stage_specific', 'source_commit': 'b85dd328423de5b1159f11939835f04db90f3187', 'result_commit': 'b1d9f46ee583d86666deb433a4fc4c20dad20225', 'report': 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/b1d9f46ee583d86666deb433a4fc4c20dad20225/docs/research/research_b_outer_splitting_stage1_execution_20261025.md', 'stop': 'artifacts/research_b_outer_splitting/20261025/once-16rows/STOP.json'}, 'Exact-R': {'stage': 'Exact-R', 'module': 'trottertracks.b2d_evaluator.exact_r', 'law': 'exact_R', 'source_commit': 'e76c0474806dca42598fa92a6564cded10896f98', 'result_commit': '58502f5549d350a8da8cbc23a6e674c4670feb9b', 'report': 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/58502f5549d350a8da8cbc23a6e674c4670feb9b/docs/research/research_b_bnext3_exact_r_execution_20261026.md', 'stop': 'artifacts/research_b_outer_exact_r/20261026/exact-r-once/STOP.json'}, 'Full-DF': {'stage': 'Full-DF', 'module': 'trottertracks.b2d_evaluator.full_df', 'law': 'deterministic_PF', 'source_commit': '8811ef089d588a529aca001f9e40cef1b2f6e1c7', 'result_commit': '3244f2af9c4017c57c180b9057b22e3eb19c62b1', 'report': 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/3244f2af9c4017c57c180b9057b22e3eb19c62b1/docs/research/research_b_full_deterministic_df_pf_execution_20261026.md', 'stop': 'artifacts/research_b_outer_full_df_pf/20261026/full-df-once/STOP.json'}, 'Full-DF-q4': {'stage': 'Full-DF-q4', 'module': 'trottertracks.b2d_evaluator.full_df_q4', 'law': 'Yoshida4', 'source_commit': '620448b14235d2d9e0e76d5d0ba7273ce887a628', 'result_commit': '266b8f887eccef57e85e0eb9866e90f4f06a0b32', 'report': 'https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/266b8f887eccef57e85e0eb9866e90f4f06a0b32/docs/research/research_b_full_df_yoshida4_q4_execution_20261026.md', 'stop': 'artifacts/research_b_outer_full_df_y4_q4/20261026/q4-once/STOP.json'}}


class EntryError(ValueError):
    """Invalid technical binding; no scientific stage was started."""


def plan(stage, *, mode="plan"):
    if mode not in ("plan", "mock"):
        raise EntryError("Only plan/mock; scientific/real/resume execution is closed")
    if stage not in STAGES:
        raise EntryError("Unknown stage")
    return {"schema": "b2d_research_entry_v1", "stage": stage, "mode": mode,
            "locator": dict(STAGES[stage]), "locator_commit": FROZEN_BASE,
            "input_recipe": "trottertracks.b2d_evaluator.input_package:InputRecipe",
            "evaluator": "trottertracks.b2d_evaluator.evaluation:evaluate",
            "scientific_binding": "DEFERRED_REQUIRES_SEPARATE_SCIENTIFIC_EXECUTION",
            "historical_state": "FINAL_PAUSE_PRESERVED", "scientific_GO": None,
            "qualifications": {"u_num": None, "gamma": None, "rare_event": None,
                               "native_price": None},
            "stop": "MANDATORY_STOP_NO_AUTOMATIC_EXTENSION"}


@dataclass(frozen=True)
class MockRequest:
    stage: str
    source: str
    cohort: str
    law: str
    method: str
    signed_order: tuple[BlockExponential, ...]
    processor: tuple[BlockExponential, ...]
    call_budget: int = 3
    status: str = "READY"

    def validate(self):
        plan(self.stage, mode="mock")
        if self.source != "ARTIFICIAL_ONLY" or self.cohort != "ARTIFICIAL_ONLY":
            raise EntryError("Mock source/cohort must be ARTIFICIAL_ONLY")
        if self.law != STAGES[self.stage]["law"] or self.method not in ("S2", "Yoshida4"):
            raise EntryError("Explicit stage law and supported method required")
        if self.stage == "Full-DF-q4" and self.method != "Yoshida4":
            raise EntryError("Full-DF-q4 retains Yoshida4")
        if type(self.call_budget) is not int or self.call_budget != 3:
            raise EntryError("Artificial transport has exactly three dispatches")
        if self.status != "READY":
            raise EntryError("STOP/FAILED/MISSING/UNQUALIFIED cannot be resumed")
        for sequence in (self.signed_order, self.processor):
            if type(sequence) is not tuple or not sequence:
                raise EntryError("Explicit nonempty ordered descriptor tuple required")
            for op in sequence:
                if (type(op) is not BlockExponential or type(op.evolution_time) is not Fraction
                        or not op.block_id.startswith("mock-") or op.controlled
                        or op.ancilla_qubit is not None):
                    raise EntryError("Only artificial uncontrolled rational descriptors")


def artificial_request(stage):
    p = plan(stage)
    # Artificial signed descriptors, not coefficients of a scientific formula.
    order = (BlockExponential("mock-D", Fraction(1, 2), False, None),
             BlockExponential("mock-R", Fraction(-1, 3), False, None))
    return MockRequest(stage, "ARTIFICIAL_ONLY", "ARTIFICIAL_ONLY", p["locator"]["law"],
                       "Yoshida4" if stage == "Full-DF-q4" else "S2", order, order[:1])


def _settings():
    return Settings("ARTIFICIAL_ONLY", "ARTIFICIAL_ONLY", "ARTIFICIAL_ONLY",
                    "mock-hamiltonian-label", "mock-sector", "mock-state", "ARTIFICIAL_ONLY",
                    DContext("mock-preparation", "mock-tail", ("mock-D",), 1),
                    (Fraction(1),), (1,), (1,), 1, 2, 0, 1e-8, Fraction(0),
                    "extract_identity_phase", "analytic_moments_no_sampling", "complex128",
                    1e-8, None, None, "descriptive_only_no_scientific_go", False)


class MockTransport:
    """Opaque descriptor transport. No numerical backend or callback loader."""
    is_mock = True

    def __init__(self):
        self.calls = []

    def perform(self, kind, dependencies, parameters):
        self.calls.append(kind)
        if kind == "mock_descriptor":
            return parameters[0]
        if kind in ("mock_transport", "mock_record"):
            return dependencies[0]
        raise EntryError("Unknown artificial node")


def run_mock(request, *, mode="mock", backend=None):
    plan(request.stage, mode=mode)
    if mode != "mock":
        raise EntryError("run_mock requires mock mode")
    request.validate()
    if backend is None:
        backend = MockTransport()
    if type(backend) is not MockTransport or backend.calls:
        raise EntryError("A fresh built-in mock transport is required")
    def descriptors(ops):
        return [{**asdict(op), "evolution_time": str(op.evolution_time)} for op in ops]
    payload = {"source": request.source, "cohort": request.cohort, "stage": request.stage,
               "law": request.law, "method": request.method,
               "signed_order": descriptors(request.signed_order),
               "actual_adjoint": descriptors(actual_adjoint(request.signed_order)),
               "processor": descriptors(request.processor), "call_budget": request.call_budget}
    graph = Graph(_settings(),
                  (Node("mock_descriptor", (), (payload,)),
                   Node("mock_transport", (0,), ()), Node("mock_record", (1,), ())),
                  ({"stage": request.stage, "outputs": {"descriptor_record": 2}},),
                  {"descriptor_binding": 0})
    result = evaluate(graph, lambda _: backend, mock_only=True)
    if len(backend.calls) != request.call_budget:
        raise EntryError("Artificial call budget mismatch")
    return {**plan(request.stage, mode="mock"), "technical_status": "MOCK_TRANSPORT_ONLY",
            "scientific_stage_evaluated": False, "evaluation": result}


def write_record(directory, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "record.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    return target
