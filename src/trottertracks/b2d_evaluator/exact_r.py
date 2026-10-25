"""Four-case Exact-R pilot; imports/inspection are metadata only.

The existing B0 input, DF block, wrapper, compiler and evidence paths are reused.
Actual basis construction and numerical references are lazy, execution-only.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction as F
from functools import lru_cache
import argparse
import json
import math
import os
from pathlib import Path
import platform
import sys

from . import b0_native as b0
from .followup_io import (SNAPSHOT, SNAPSHOT_COMMIT, registered_scope,
                          preflight as followup_preflight, run_prepared)
from .launcher import _encode, rejection
from .outer import DStage, INNER, OuterCase, mathematical_stages
from .scope import Stage1Failure, STOP

OUTPUT_PARENT = "artifacts/research_b_outer_exact_r"
AXES = ("cosine", "sine")
BASE = "4a8ef63081011c890d68bfd1cab78515a3d01f02"


def cases():
    return tuple(dict(case_id="E"+str(i+1), baseline_candidate_id=baseline,
        tail_mode="exact-r", case=asdict(OuterCase(method, time, 2, "A2", 1)))
        for i, (baseline, method, time) in enumerate((
            ("B3", "outer-Yoshida4", F(9, 50)),
            ("B3", "outer-Yoshida4", F(-9, 50)),
            ("B4", "SABA2", F(9, 50)),
            ("B4", "SABA2", F(-9, 50)))))


def validate_case(row):
    if row not in cases():
        raise Stage1Failure("EXACT_R_CASE_SCOPE_MISMATCH")


@lru_cache(maxsize=1)
def metadata_context():
    return b0.candidate_graph("EXACT_R_METADATA").settings.context


def plan(row, context=None):
    validate_case(row)
    # Only the old ordered D/R metadata is reused, never an RTE config/event.
    old = dict(candidate_id=row["baseline_candidate_id"], case=row["case"],
               rte_parameters=dict(r_rte=2, K=2))
    return b0.outer_plan(old, metadata_context() if context is None else context)


@dataclass(frozen=True)
class ExactRStage:
    fragment_index: int
    tail_id: str
    block: object
    duration: F
    extracted_identity: float
    tail_mode: str = "exact-r"

    def __post_init__(self):
        if (self.tail_mode != "exact-r" or self.fragment_index != self.block.original_fragment_index
                or self.block.block_kind != "df_fragment" or not self.tail_id
                or not math.isfinite(self.extracted_identity)):
            raise ValueError("Exact-R fragment/identity binding differs")
        if isinstance(self.duration, (float, bool)):
            raise ValueError("Exact signed rational duration required")
        object.__setattr__(self, "duration", F(self.duration))
        if not self.duration:
            raise ValueError("Nonzero Exact-R duration required")


def append_exact_r(circuit, stage, builder, *, controlled=False, ancilla_qubit=None):
    """Full-Fock DF exponent plus +c_ext*t; global scalar stays outside."""
    if controlled and ancilla_qubit is None:
        raise ValueError("Controlled Exact-R needs its ancilla")
    builder.append_block_evolution(circuit, stage.block, float(stage.duration),
        controlled=controlled, ancilla_qubit=ancilla_qubit)
    phase = float(stage.duration)*stage.extracted_identity
    if controlled:
        circuit.p(phase, ancilla_qubit)
    else:
        circuit.global_phase += phase


def inventory():
    blocks, tails, exps = {}, {}, {}
    for row in cases():
        word = plan(row)
        for stage in word.body:
            if isinstance(stage, DStage):
                for op in stage.append_operations():
                    label = "D_"+str(stage.context.block_ids.index(op.block_id))
                    for controlled in (False, True):
                        blocks[b0.block_key(label, op.evolution_time, controlled)] = (label, op.evolution_time, controlled)
            else:
                time = stage.evolution_time
                for controlled in (False, True):
                    tails[b0.block_key("R", time, controlled)] = (time, controlled)
        # Independent, unfused mathematical product, not a traversal of body.
        for kind, weight in mathematical_stages(word.case.method):
            time = word.case.total_time*weight/word.case.step_count
            if kind == "R":
                exps[b0.exp_key("R", time)] = ("R", time)
            else:
                for w in map(F, b0.symmetric_s2_sequence(INNER["A2"]())):
                    for label in ("D_0", "D_1", "D_2", "D_1", "D_0"):
                        t = time*w/(1 if label == "D_2" else 2)
                        exps[b0.exp_key(label, t)] = (label, t)
        exps[b0.exp_key("physical", word.case.total_time)] = ("physical", word.case.total_time)
    return dict(blocks=blocks, tails=tails, exps=exps)


def scope():
    inv = inventory()
    calls = [("exact_r."+k, "input") for k in (
        "fixture", "split", "prepare", "full_inputs", "input_gate", "tail_prepare", "tail_identity_gate")]
    for group, kind in (("blocks", "block_build"), ("tails", "tail_build"),
                        ("tails", "tail_gate"), ("exps", "exponential")):
        calls.extend(("exact_r."+kind, key) for key in inv[group])
    for row in cases():
        key = row["case_id"]
        calls.extend(("exact_r."+k, key) for k in (
            "system_build", "controlled_build", "reference", "system_gate", "controlled_gate",
            "signals", "saved_baseline"))
        for axis in AXES:
            calls.extend(("exact_r."+k, key+":"+axis) for k in (
                "wrapper_build", "wrapper_reference", "raw_wrapper_gate", "raw_expectation",
                "compile", "compiled_gate", "compiled_expectation", "measure"))
    return registered_scope("exact_r", calls, cases=list(cases()), new_exact_r_cases=4,
        existing_finite_rte_cases=4, system_builds=4, controlled_builds=4, wrappers=8,
        compile_requests=8, cost_requests=8, tail_builds=len(inv["tails"]),
        operator_gates=len(inv["tails"])+24, expectations=16,
        precision="complex128", atol=1e-12, rtol=0, phase_fit=False,
        compiler=b0.PROFILE, normalization=1, sampling=False,
        target="physical_complex_coherent_signal_absolute_error",
        saved_signal_commit=SNAPSHOT_COMMIT,
        comparison="saved finite-r2-K2 corrected means; no new finite-RTE execution",
        exclusions=["shots", "G", "MC", "rare_costs", "gamma", "A1_enclosure", "full_DF_PF", "n_D_search"])


class Native(b0.Native):
    """Small B0 extension; the old Native class and RTE renderer are unchanged."""
    def __init__(self, run, ledger):
        super().__init__(run, ledger)
        self.tail_block = self.call("tail_prepare", "input", self.prepare_tail)
        self.tail_diagnostics = self.call("tail_identity_gate", "input", self.tail_identity_gate)

    def call(self, kind, key, function):
        try:
            return self.ledger.call("exact_r."+kind, function, identity=key,
                summary=(lambda x: x) if kind.endswith("_gate") else None)
        except BaseException as error:
            if not getattr(error, "recorded", False):
                self.ledger.rows.append(dict(failed_boundary="exact_r."+kind, identity=key,
                    error_type=type(error).__name__, message=str(error),
                    diagnostics=getattr(error, "diagnostics", None), metrics=None))
                error.recorded = True
            raise

    def prepare_tail(self):
        from trotterlib.df_partial_randomized_pf import df_hamiltonian_to_model, rank_df_fragments
        from trotterlib.df_partial_s2 import DFDeterministicFragmentSpec
        from trotterlib.df_rte_tail import DFBasisRegistry
        from trotterlib.df_trotter.ops import build_df_blocks
        if tuple(self.prep.randomized_block_indices) != (2,):
            raise Stage1Failure("EXACT_R_RESIDUAL_IDENTITY_MISMATCH")
        block = build_df_blocks(df_hamiltonian_to_model(self.h.select_blocks((2,))), sort="descending_abs")[0]
        basis = DFBasisRegistry().register(block.U_ops, num_system_qubits=2)
        rank = next(r.rank for r in rank_df_fragments(self.h) if r.original_index == 2)
        return DFDeterministicFragmentSpec("df-fragment-2", "df_fragment", 2, rank,
            basis.basis_id, basis.basis_hash, basis.metadata.operations,
            tuple(float(v) for v in block.eta), float(block.lam), 2, 3, tuple(block.U_ops))

    def tail_identity_gate(self):
        if (self.tail_block.lam != float(self.h.lambdas[2]) or self.tail_block.num_system_qubits != 2
                or self.prep.identity_policy != "extract_identity_phase" or self.prep.coefficient_atol != 0):
            raise Stage1Failure("EXACT_R_FRAGMENT_DEFINITION_MISMATCH")
        commutator = self.matrices["D"]@self.matrices["R"]-self.matrices["R"]@self.matrices["D"]
        norm = float(self.np.linalg.norm(commutator, ord=2))
        if not self.np.isfinite(norm) or norm <= 1e-12:
            raise Stage1Failure("EXACT_R_REQUIRED_NONCOMMUTATIVITY_FAIL")
        return dict(DR_commutator_norm=norm, fragment_index=2, lam=self.tail_block.lam,
            extracted_identity=self.prep.extracted_identity_coefficient,
            scalar_once=self.scalar, precision="EMPIRICAL_COMPLEX128")

    def exact_stage(self, time):
        return ExactRStage(2, self.context.tail_id, self.tail_block, time,
                           self.prep.extracted_identity_coefficient)

    def tail(self, time, controlled):
        def build():
            qc = self.QC(3 if controlled else 2)
            append_exact_r(qc, self.exact_stage(time), self.block_builder,
                           controlled=controlled, ancilla_qubit=2 if controlled else None)
            return qc
        return self.cached("tail_build", b0.block_key("R", time, controlled), build)

    def render_exact(self, row, controlled):
        word = plan(row, self.context)
        qc = self.QC(3 if controlled else 2)
        for stage in word.body:
            if isinstance(stage, DStage):
                for op in stage.append_operations():
                    label = "D_"+str(self.context.block_ids.index(op.block_id))
                    qc.compose(self.block(label, op.evolution_time, controlled), inplace=True)
            else:
                qc.compose(self.tail(stage.evolution_time, controlled), inplace=True)
        phase = -float(word.case.total_time)*self.scalar
        if controlled:
            qc.p(phase, 2)
        else:
            qc.global_phase += phase
        return qc

    def exact_reference(self, row):
        """CAR/expm, in mathematical order; no circuit, basis or descriptor use."""
        validate_case(row)
        case = OuterCase(**row["case"])
        value = self.eye.copy()
        for kind, weight in mathematical_stages(case.method)*case.step_count:
            time = case.total_time*weight/case.step_count
            if kind == "R":
                value = value@self.exp("R", time)
            else:
                for w in map(F, b0.symmetric_s2_sequence(INNER["A2"]())):
                    for label in ("D_0", "D_1", "D_2", "D_1", "D_0"):
                        value = value@self.exp(label, time*w/(1 if label == "D_2" else 2))
        return self.np.exp(-1j*float(case.total_time)*self.scalar)*value

    def signals(self, row, reference):
        physical = self.exp("physical", row["case"]["total_time"])
        z = complex(self.np.vdot(self.state, reference@self.state))
        target = complex(self.np.vdot(self.state, physical@self.state))
        delta = reference-physical
        return dict(X=z.real, Y=z.imag, radius=abs(z), coherent_signal=z,
            physical_signal=target, physical_complex_coherent_signal_absolute_error=abs(z-target),
            X_absolute_error=abs(z.real-target.real), Y_absolute_error=abs(z.imag-target.imag),
            radius_absolute_error=abs(abs(z)-abs(target)),
            full_Fock_operator_error=float(self.np.linalg.norm(delta, ord=2)),
            N1_sector_operator_error=float(self.np.linalg.norm(delta[self.np.ix_([2, 1], [2, 1])], ord=2)),
            normalization=1, u_num=None, gamma=None, qualification="DESCRIPTIVE_NOT_CERTIFIED")

    def saved_baseline(self, row, reference):
        with (self.run["root"]/SNAPSHOT/"result.json").open() as stream:
            saved = json.load(stream)
        old = next(r for r in saved["rows"] if r["candidate_id"] == row["baseline_candidate_id"]
                   and r["case"]["total_time"] == str(row["case"]["total_time"]))
        expected = {**row["case"], "total_time": str(row["case"]["total_time"])}
        if old["case"] != expected or old["rte_parameters"] != dict(r_rte=2, K=2):
            raise Stage1Failure("EXACT_R_SAVED_COMPARISON_SCOPE_MISMATCH")
        # Old ideal-tail is a consistency check, never the circuit's oracle.
        saved_signals = old["outputs"]["signals"]
        matches = {}
        for name, matrix in (("ideal_tail", reference),
                             ("physical", self.exp("physical", row["case"]["total_time"]))):
            z = complex(self.np.vdot(self.state, matrix@self.state))
            old_z = complex(saved_signals[name]["X"], saved_signals[name]["Y"])
            matches[name] = self.check(self.np.asarray([[z]]), self.np.asarray([[old_z]]),
                                       "EXACT_R_SAVED_IDEAL_OR_PHYSICAL_SIGNAL_MISMATCH")
        return dict(commit=SNAPSHOT_COMMIT, candidate_id=old["candidate_id"], case=old["case"],
            rte_parameters=old["rte_parameters"], outputs=old["outputs"],
            saved_signal_consistency=matches,
            cost_status="No new finite-RTE costs; B1 estimates are S0-conditional, B0 representatives are not means")

    def provenance(self):
        block = self.tail_block
        return dict(preparation_id=self.context.preparation_id, tail_id=self.context.tail_id,
            fragment_index=block.original_fragment_index, lam=block.lam, eta=block.diagonal_eta,
            basis_id=block.basis_id, basis_hash=block.basis_hash,
            basis_operations=len(block.runtime_basis_operations),
            gaussian_convention="inverse then diagonal then forward; same block primitive as B0",
            scalar_once=self.scalar, extracted_identity=self.prep.extracted_identity_coefficient,
            compiler=b0.PROFILE, occupied_modes=[0], ancilla=2,
            sector_mapping="OpenFermion [1,2] to Qiskit [2,1]",
            python=sys.version, executable=sys.executable, platform=platform.platform())


def execute(run, ledger, evidence, *, native_factory=Native):
    native = native_factory(run, ledger)
    inv = inventory()
    for label, time, controlled in inv["blocks"].values():
        native.block(label, time, controlled)
    for label, time in inv["exps"].values():
        native.exp(label, time)
    for key, (time, controlled) in inv["tails"].items():
        circuit = native.tail(time, controlled)
        reference = native.exp("R", time)
        native.call("tail_gate", key, lambda: native.circuit_gate(circuit,
                    native.controlled(reference) if controlled else reference))
    outputs = []
    compiled = []
    for row in cases():
        key = row["case_id"]
        word = plan(row, native.context)
        schedule = [dict(kind="D", duration=s.duration, tau=s.tau, repetitions=s.repetitions)
                    if isinstance(s, DStage) else dict(kind="exact-r", duration=s.evolution_time)
                    for s in word.body]
        record = dict(**row, correctness="IN_PROGRESS", append_schedule=schedule,
                      signed_R_time_sum=word.cost_metadata["R_time"], normalization=1)
        ledger.rows.append(record)
        system = native.call("system_build", key, lambda: native.render_exact(row, False))
        controlled = native.call("controlled_build", key, lambda: native.render_exact(row, True))
        ref = native.call("reference", key, lambda: native.exact_reference(row))
        record["operator_gates"] = dict(
            system=native.call("system_gate", key, lambda: native.circuit_gate(system, ref)),
            controlled=native.call("controlled_gate", key, lambda: native.circuit_gate(controlled, native.controlled(ref))))
        record["axes"] = {}
        for axis in AXES:
            aid = key+":"+axis
            wrapper = native.call("wrapper_build", aid, lambda: native.wrapper(controlled, axis))
            wr = native.call("wrapper_reference", aid, lambda: native.wrapper_reference(ref, axis))
            gate = native.call("raw_wrapper_gate", aid, lambda: native.circuit_gate(wrapper, wr))
            raw = native.call("raw_expectation", aid, lambda: native.expectation(wrapper, wr))
            circuit = native.call("compile", aid, lambda: native.compile(wrapper))
            cgate = native.call("compiled_gate", aid, lambda: native.circuit_gate(circuit, wr, raw=wrapper))
            csignal = native.call("compiled_expectation", aid, lambda: native.expectation(circuit, wr))
            record["axes"][axis] = dict(raw_operator_error=gate, compiled_operator_error=cgate,
                raw_expectation=raw, compiled_expectation=csignal, metrics=None)
            compiled.append((aid, circuit, record["axes"][axis]))
        record["correctness"] = "PASS_EMPIRICAL_NO_PHASE_FIT"
        outputs.append((row, record, ref))
    # All four cases, both axes and every compiled gate pass before comparisons/costs.
    for row, record, ref in outputs:
        record["saved_finite_rte"] = native.call("saved_baseline", row["case_id"], lambda: native.saved_baseline(row, ref))
    for row, record, ref in outputs:
        record["signals"] = native.call("signals", row["case_id"], lambda: native.signals(row, ref))
    for aid, circuit, record in compiled:
        record["metrics"] = native.call("measure", aid, lambda: native.measure(circuit))
    return dict(study="B-Next3_Exact_R", rows=ledger.rows, source=run["source"],
        provenance=native.provenance(), input_diagnostics=native.input_diagnostics,
        tail_diagnostics=native.tail_diagnostics, compile_requests=len(compiled),
        actual_transpile_count=len(native.compile_cache), normalization=1,
        scientific_status="DESCRIPTIVE_ONLY", classification="EXACT_R_PILOT_NOT_TOTAL_RESOURCE_COMPARISON",
        shots=None, G=None, gamma=None, stop=STOP)


def preflight(root, output, source):
    run = followup_preflight(root, output, source, "exact_r", scope())
    if run["versions"]["qiskit"] != b0.PROFILE["qiskit"]:
        raise ValueError("Exact-R compiler version mismatch")
    from .input_package import read_fixture_definition
    read_fixture_definition(root)
    run["resources"] = dict(affinity_CPUs=len(os.sched_getaffinity(0)),
        load_average=os.getloadavg(),
        memory_available=next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemAvailable:")),
        acceleration="shared signed block/exponential/compile caches; native-v5 defaults; no artificial resource or wall-time cap",
        workload="4 small full-Fock cases/8 wrappers; no extra large worker framework")
    return run


def main():
    parser = argparse.ArgumentParser(description="Four Exact-R cases; science needs explicit --execute")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--source-commit")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.inspect:
        print(_encode(dict(status="METADATA_ONLY_NO_SCIENCE", scope=scope())))
        return 0
    if not args.source_commit or not args.output:
        parser.error("--preflight/--execute requires --source-commit and --output")
    try:
        run = preflight(Path(__file__).resolve().parents[3], args.output, args.source_commit)
        status = (dict(status="PREFLIGHT_ONLY_NO_SCIENCE", source=run["source"],
                       versions=run["versions"], scope=run["scope"], fresh_output=True)
                  if args.preflight else run_prepared(run, execute))
    except BaseException as error:
        status = rejection(error)
    print(_encode(status))
    return 0 if status["status"] in ("PREFLIGHT_ONLY_NO_SCIENCE", "COMPLETED_DESCRIPTIVE_MANDATORY_STOP") else 2
