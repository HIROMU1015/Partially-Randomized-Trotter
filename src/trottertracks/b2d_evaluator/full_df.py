"""Fixed Full DF-PF pilot, metadata-only until explicit scientific execution.

Reuse AX-2B controlled Full PF, the library system builder and B0 prepared
wrappers/compiler. No old acquisition runtime or approval machinery is loaded.
"""
from dataclasses import asdict
from fractions import Fraction as F
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from . import b0_native as b0
from . import exact_r
from ._full_df_schedule import deterministic_control_plan, _get_w_list
from .followup_io import registered_scope, preflight as base_preflight, run_prepared
from .input_package import read_fixture_definition
from .launcher import _encode, rejection
from .scope import Stage1Failure, STOP

PROVIDER_COMMIT = "31cdff47f3282d2898c153a1af22f90e500be4c6"
EXACT_COMMIT = "58502f5549d350a8da8cbc23a6e674c4670feb9b"
EXACT_PATH = "artifacts/research_b_outer_exact_r/20261026/exact-r-once"
OUTPUT_PARENT = "artifacts/research_b_outer_full_df_pf"
LABELS = ("B0", "B1", "B2", "B3")  # one-body, DF0, DF1, DF2; not candidate IDs.
AXES = ("cosine", "sine")


def cases():
    return tuple(dict(case_id="FD"+str(i+1), formula=formula, q=2, total_time=time,
                      method="full-deterministic-df-pf", control_policy="ordinary")
        for i, (formula, time) in enumerate((("2nd", F(9,50)), ("2nd", F(-9,50)),
                                            ("4th", F(9,50)), ("4th", F(-9,50)))))


def validate_case(row):
    if row not in cases():
        raise Stage1Failure("FULL_DF_CASE_SCOPE_MISMATCH")


def schedule(row):
    validate_case(row)
    return deterministic_control_plan(4, float(row["total_time"]), row["q"], row["formula"])


def reference_pieces(row):
    validate_case(row)
    weights = _get_w_list(row["formula"])
    return (1.,) if row["formula"] == "2nd" else (weights[1], weights[0], weights[1])


def inventory():
    probes, exps = {}, {}
    for row in cases():
        for stage in schedule(row):
            for controlled in (False, True):
                key = b0.block_key(LABELS[stage.term], stage.time, controlled)
                probes[key] = (stage.term, stage.time, controlled)
            exps[b0.exp_key(LABELS[stage.term], stage.time)] = (LABELS[stage.term], stage.time)
        for w in reference_pieces(row):
            time = float(row["total_time"])*w/row["q"]
            for i in (0,1,2,3,2,1,0):
                t = time if i == 3 else time/2
                exps[b0.exp_key(LABELS[i], t)] = (LABELS[i], t)
        exps[b0.exp_key("physical", row["total_time"])] = ("physical", row["total_time"])
    return dict(probes=probes, exps=exps)


def scope():
    inv = inventory()
    calls = [("full_df."+k, "input") for k in (
        "fixture", "split", "prepare", "full_inputs", "input_gate", "full_prepare", "definition_gate", "saved_comparisons")]
    for kind in ("block_build", "block_gate"):
        calls.extend(("full_df."+kind, key) for key in inv["probes"])
    calls.extend(("full_df.exponential", key) for key in inv["exps"])
    for row in cases():
        key = row["case_id"]
        calls.extend(("full_df."+k, key) for k in (
            "system_build", "controlled_build", "reference", "system_gate", "controlled_gate", "signals"))
        for axis in AXES:
            calls.extend(("full_df."+k, key+":"+axis) for k in (
                "wrapper_build", "wrapper_reference", "raw_wrapper_gate", "raw_expectation",
                "compile", "compiled_gate", "compiled_expectation", "measure"))
    return registered_scope("full_df_pf", calls, cases=list(cases()), new_cases=4,
        wrappers=8, compile_requests=8, cost_requests=8, system_builds=4, controlled_builds=4,
        block_probes=len(inv["probes"]), exponential_keys=len(inv["exps"]),
        operator_gates=len(inv["probes"])+24, expectations=16,
        precision="complex128", atol=1e-12, rtol=0, phase_fit=False, normalization=1,
        compiler=b0.PROFILE, provider_source=PROVIDER_COMMIT, scalar_policy="original_c0_once",
        target="physical_complex_coherent_signal_absolute_error",
        saved_comparisons="four finite-RTE and four Exact-R cases;no recalculation",
        exclusions=["D_R_outer_plan", "RTE_config_events", "sampling", "shots", "G", "rare_costs",
                    "A1_enclosure", "gamma", "q_nD_search", "molecules", "new_PF_coefficients"])


class Native(b0.Native):
    def __init__(self, run, ledger):
        super().__init__(run, ledger)
        from . import _full_df_native as provider
        self.provider = provider
        self.full_blocks = self.call("full_prepare", "input", self.prepare_full)
        self.full_scalar = float(self.h.constant)
        self.full_diagnostics = self.call("definition_gate", "input", self.definition_gate)

    def call(self, kind, key, function):
        try:
            return self.ledger.call("full_df."+kind, function, identity=key,
                summary=(lambda x:x) if kind.endswith("_gate") else None)
        except BaseException as error:
            if not getattr(error, "recorded", False):
                self.ledger.rows.append(dict(failed_boundary="full_df."+kind, identity=key,
                    error_type=type(error).__name__, message=str(error),
                    diagnostics=getattr(error, "diagnostics", None), metrics=None))
                error.recorded = True
            raise

    def prepare_full(self):
        # Shared split preparation binds the old input only. No split evolution.
        tail_block = exact_r.Native.prepare_tail(self)
        self.matrices.update(zip(LABELS, (self.blocks["D_0"], self.blocks["D_1"],
            self.blocks["D_2"], self.matrices["R"]+self.prep.extracted_identity_coefficient*self.eye)))
        return self.prep.deterministic_blocks+(tail_block,)

    def definition_gate(self):
        if (len(self.full_blocks) != 4 or self.full_blocks[0].block_kind != "one_body"
                or tuple(b.original_fragment_index for b in self.full_blocks[1:]) != (0,1,2)
                or any(b.lam != float(self.h.lambdas[i]) for i,b in enumerate(self.full_blocks[1:]))):
            raise Stage1Failure("FULL_DF_BLOCK_IDENTITY_MISMATCH")
        value = self.full_scalar*self.eye+sum((self.matrices[k] for k in LABELS), self.np.zeros((4,4),complex))
        return dict(physical_identity_error=self.check(value, self.matrices["physical"], "FULL_DF_SCALAR_IDENTITY_FAILURE"),
                    scalar=self.full_scalar, scalar_policy="c0_only_no_extracted_identity",
                    ordered_fragments=[None,0,1,2])

    def block_probe(self, index, time, controlled):
        def build():
            qc = self.QC(3 if controlled else 2)
            self.provider._append_block(qc, self.full_blocks[index], time,
                                        "ORDINARY" if controlled else "UNCONTROLLED", 2)
            return qc
        return self.cached("block_build", b0.block_key(LABELS[index],time,controlled), build)

    def system(self, row):
        validate_case(row)
        from trotterlib.df_trotter.circuit import build_df_trotter_circuit
        from trotterlib.df_trotter.model import Block, DFBlock, OneBodyGaussianBlock
        blocks = []
        for b in self.full_blocks:
            blocks.append(Block.from_one_body_gaussian(OneBodyGaussianBlock(b.runtime_basis_operations,
                self.np.asarray(b.diagonal_eigenvalues))) if b.block_kind == "one_body" else
                Block.from_df(DFBlock(b.runtime_basis_operations,self.np.asarray(b.diagonal_eta),b.lam)))
        step = build_df_trotter_circuit(blocks, float(row["total_time"])/row["q"], 2, row["formula"], energy_shift=0.)
        qc = self.QC(2)
        for _ in range(row["q"]): qc.compose(step, inplace=True)
        qc.global_phase -= float(row["total_time"])*self.full_scalar
        return qc

    def controlled_system(self, row):
        validate_case(row)
        # The old builder's instruction bound, derived from these actual blocks.
        pieces = 1 if row["formula"] == "2nd" else 3
        bound = row["q"]*pieces*sum(self.provider.block_instruction_bound(b,"ORDINARY")
                                     for b in self.full_blocks for _ in (0,1))+1
        result = self.provider.build_deterministic_native(self.full_blocks, num_system_qubits=2,
            T=float(row["total_time"]), q=row["q"], formula=row["formula"], scalar=self.full_scalar,
            control_policy="ordinary", max_instructions=bound)
        return result.circuit

    def full_reference(self, row):
        """Canonical mathematical S2, independent of native/merged stage lists."""
        validate_case(row)
        value = self.eye.copy()
        for _ in range(row["q"]):
            for w in reference_pieces(row):
                time = float(row["total_time"])*w/row["q"]
                for i in (0,1,2,3,2,1,0):
                    value = value@self.exp(LABELS[i], time if i == 3 else time/2)
        return self.np.exp(-1j*float(row["total_time"])*self.full_scalar)*value

    def signals(self, row, reference):
        return exact_r.Native.signals(self, {"case":{"total_time":row["total_time"]}}, reference)

    def saved_comparisons(self):
        from .followup_io import SNAPSHOT, SNAPSHOT_COMMIT
        old = json.loads((self.run["root"]/SNAPSHOT/"result.json").read_text())
        exact = json.loads((self.run["root"]/EXACT_PATH/"result.json").read_text())
        if exact["source"] != "e76c0474806dca42598fa92a6564cded10896f98":
            raise Stage1Failure("FULL_DF_SAVED_EXACT_SOURCE_MISMATCH")
        finite = [r for r in old["rows"] if r["candidate_id"] in ("B3","B4")]
        if len(finite) != 4 or [r["case_id"] for r in exact["rows"]] != ["E1","E2","E3","E4"]:
            raise Stage1Failure("FULL_DF_SAVED_COMPARISON_SCOPE_MISMATCH")
        for r in finite:
            target = self.exp("physical", F(r["case"]["total_time"]))
            z = complex(self.np.vdot(self.state,target@self.state))
            s = r["outputs"]["signals"]["physical"]
            self.check(self.np.asarray([[z]]),self.np.asarray([[complex(s["X"],s["Y"])]]),"FULL_DF_SAVED_PHYSICAL_MISMATCH")
        return dict(finite_rte_commit=SNAPSHOT_COMMIT, finite_rte=finite,
                    exact_r_commit=EXACT_COMMIT, exact_r=exact["rows"],
                    policy="Unchanged saved cells;no cost ratios or total-resource ranking")

    def provenance(self):
        return dict(provider_commit=PROVIDER_COMMIT,
            provider_reuse="unchanged selected AX-2A/AX-2B definitions;adapted imports only",
            system_builder="df_trotter.circuit.build_df_trotter_circuit",
            scalar=self.full_scalar, scalar_policy="original_c0_once", input_preparation=self.context.preparation_id,
            blocks=[dict(block_id=b.block_id,fragment=b.original_fragment_index,basis_hash=b.basis_hash,
                         basis_operations=len(b.runtime_basis_operations)) for b in self.full_blocks],
            compiler=b0.PROFILE, occupation=[0], ancilla=2,
            python=sys.version, executable=sys.executable, platform=platform.platform())


def execute(run, ledger, evidence, *, native_factory=Native):
    native = native_factory(run, ledger)
    inv = inventory()
    for label,time in inv["exps"].values(): native.exp(label,time)
    for key,(index,time,controlled) in inv["probes"].items():
        qc = native.block_probe(index,time,controlled); ref = native.exp(LABELS[index],time)
        native.call("block_gate",key,lambda: native.circuit_gate(qc,native.controlled(ref) if controlled else ref))
    outputs, compiled = [], []
    for row in cases():
        key = row["case_id"]
        record = dict(**row, correctness="IN_PROGRESS", axes={}, normalization=1,
                      append_schedule=[dict(asdict(s), mode="ORDINARY", source_plan_mode=s.mode)
                                       for s in schedule(row)])
        ledger.rows.append(record)
        system = native.call("system_build",key,lambda:native.system(row))
        control = native.call("controlled_build",key,lambda:native.controlled_system(row))
        ref = native.call("reference",key,lambda:native.full_reference(row))
        record["operator_gates"] = dict(system=native.call("system_gate",key,lambda:native.circuit_gate(system,ref)),
            controlled=native.call("controlled_gate",key,lambda:native.circuit_gate(control,native.controlled(ref))))
        for axis in AXES:
            aid = key+":"+axis
            wrapper = native.call("wrapper_build",aid,lambda:native.wrapper(control,axis))
            wr = native.call("wrapper_reference",aid,lambda:native.wrapper_reference(ref,axis))
            raw_gate = native.call("raw_wrapper_gate",aid,lambda:native.circuit_gate(wrapper,wr))
            raw = native.call("raw_expectation",aid,lambda:native.expectation(wrapper,wr))
            qc = native.call("compile",aid,lambda:native.compile(wrapper))
            gate = native.call("compiled_gate",aid,lambda:native.circuit_gate(qc,wr,raw=wrapper))
            measured = native.call("compiled_expectation",aid,lambda:native.expectation(qc,wr))
            record["axes"][axis] = dict(raw_operator_error=raw_gate,compiled_operator_error=gate,
                raw_expectation=raw,compiled_expectation=measured,metrics=None)
            compiled.append((aid,qc,record["axes"][axis]))
        record["correctness"] = "PASS_EMPIRICAL_NO_PHASE_FIT"
        outputs.append((row,ref,record))
    saved = native.call("saved_comparisons","input",native.saved_comparisons)
    for row,ref,record in outputs:
        record["signals"] = native.call("signals",row["case_id"],lambda:native.signals(row,ref))
    for aid,qc,record in compiled:
        record["metrics"] = native.call("measure",aid,lambda:native.measure(qc))
    return dict(study="B-Next4_Full_DF_PF",source=run["source"],rows=ledger.rows,
        input_diagnostics=native.input_diagnostics,full_diagnostics=native.full_diagnostics,
        provenance=native.provenance(),saved_comparisons=saved,compile_requests=8,
        actual_transpile_count=len(native.compile_cache),scientific_status="DESCRIPTIVE_ONLY",
        classification="FIXED_Q_PILOT_NOT_MATCHED_RESOURCE_OPTIMIZATION",normalization=1,
        shots=None,G=None,gamma=None,stop=STOP)


def preflight(root,output,source):
    run = base_preflight(root,output,source,"full_df_pf",scope())
    if run["versions"]["qiskit"] != b0.PROFILE["qiskit"]: raise ValueError("Full DF compiler version mismatch")
    read_fixture_definition(root)
    blobs = {}
    for name in ("result.json","inputs.json","binding.json"):
        path = EXACT_PATH+"/"+name
        expected = subprocess.check_output(["git","rev-parse",EXACT_COMMIT+":"+path],cwd=root,text=True).strip()
        actual = subprocess.check_output(["git","hash-object","--",path],cwd=root,text=True).strip()
        if expected != actual: raise Stage1Failure("FULL_DF_EXACT_SNAPSHOT_MISMATCH")
        blobs[name] = expected
    inputs = json.loads((Path(root)/EXACT_PATH/"inputs.json").read_text())
    if any(inputs[k] != run["inputs"][k] for k in ("hamiltonian_id","sector_id","state_id","input_data_sha256")):
        raise Stage1Failure("FULL_DF_SAVED_INPUT_IDENTITY_MISMATCH")
    run["saved_exact_blobs"] = blobs
    run["resources"] = dict(affinity_CPUs=len(os.sched_getaffinity(0)),load_average=os.getloadavg(),
        memory_available=next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')),
        acceleration="shared CAR exponentials/probes;existing compiler cache;case-ordered fail-fast pilot,no artificial resource cap")
    return run


def main():
    parser = argparse.ArgumentParser(description="Full DF-S2/Yoshida4 four-case preparation or explicit execution")
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("inspect","preflight","execute"): mode.add_argument("--"+name,action="store_true")
    parser.add_argument("--source-commit"); parser.add_argument("--output")
    args = parser.parse_args()
    if args.inspect:
        print(_encode(dict(status="METADATA_ONLY_NO_SCIENCE",scope=scope()))); return 0
    if not args.source_commit or not args.output: parser.error("--preflight/--execute requires --source-commit and --output")
    try:
        run = preflight(Path(__file__).resolve().parents[3],args.output,args.source_commit)
        status = (dict(status="PREFLIGHT_ONLY_NO_SCIENCE",source=run["source"],versions=run["versions"],scope=run["scope"],fresh_output=True)
                  if args.preflight else run_prepared(run,execute))
    except BaseException as error: status = rejection(error)
    print(_encode(status))
    return 0 if status["status"] in ("PREFLIGHT_ONLY_NO_SCIENCE","COMPLETED_DESCRIPTIVE_MANDATORY_STOP") else 2
