"""Two fixed Full DF-Yoshida4 q4 cases, using the existing Full DF-PF path.

No circuit, PF coefficient, numerical reference or compiler is reimplemented.
Inspection/preflight use metadata only; science requires explicit --execute.
"""
import csv
from fractions import Fraction as F
import json
import math
from pathlib import Path
import subprocess

from . import full_df as core
from .followup_io import SNAPSHOT
from .scope import Stage1Failure

STUDY = "full_df_y4_q4"
OUTPUT_PARENT = "artifacts/research_b_outer_"+STUDY
Q2_COMMIT = "3244f2af9c4017c57c180b9057b22e3eb19c62b1"
Q2_PATH = "artifacts/research_b_outer_full_df_pf/20261026/full-df-once"
B1_COMMIT = "bef63edbd8cc564444b44e4e8cf23baa90f05fe5"
B1_PATH = "artifacts/research_b_outer_b1/20261025"
BIAS_BUDGET = 5e-10
PR_CANDIDATES = ("D1", "R2", "R4")


def cases():
    return tuple(dict(case_id=key, formula="4th", q=4, total_time=time,
                      method="full-deterministic-df-pf", control_policy="ordinary")
                 for key,time in (("FD5",F(9,50)),("FD6",F(-9,50))))


def validate_case(row):
    if row not in cases():
        raise Stage1Failure("FULL_DF_Q4_CASE_SCOPE_MISMATCH")


def scope():
    result = core.scope(cases(), validator=validate_case, study=STUDY)
    result.update(result_study="Full_DF_Yoshida4_q4", epsilon=1e-9,
        observed_bias_budget=BIAS_BUDGET, u_num=None, gamma=None,
        saved_comparisons="unchanged q2 FD3/4;PR D1/R2/R4 saved means and B1 finite-IID statistics;B3/B4 and Exact-R context;no recalculation")
    return result


def observed_bias_classification(bias):
    """Point-value comparison only; no numerical enclosure or certificate."""
    if bias is None or not math.isfinite(bias) or bias < 0:
        return "INCONCLUSIVE_NUMERICAL_IDENTIFIABILITY"
    return ("WITHIN_OBSERVED_BIAS_BUDGET" if bias <= BIAS_BUDGET
            else "OUTSIDE_OBSERVED_BIAS_BUDGET")


class Native(core.Native):
    case_validator = staticmethod(validate_case)

    def signals(self, row, reference):
        result = super().signals(row, reference)
        result.update(epsilon=1e-9, observed_bias_budget=BIAS_BUDGET,
            observed_bias_classification=observed_bias_classification(
                result["physical_complex_coherent_signal_absolute_error"]),
            budget_qualification="OBSERVED_B_ONLY_U_NUM_GAMMA_NULL_NOT_CERTIFIED")
        return result

    def saved_comparisons(self):
        result = super().saved_comparisons()
        root = self.run["root"]
        old = json.loads((root/SNAPSHOT/"result.json").read_text())
        q2 = json.loads((root/Q2_PATH/"result.json").read_text())
        pr = [r for r in old["rows"] if r["candidate_id"] in PR_CANDIDATES]
        q2_rows = [r for r in q2["rows"] if r["case_id"] in ("FD3","FD4")]
        with (root/B1_PATH/"STATISTICS.csv").open(newline="") as file:
            statistics = [r for r in csv.DictReader(file)
                          if r["group"].split(":")[0] in PR_CANDIDATES
                          and r["metric"] != "measurement_count"]
        expected = {key+":"+time+":"+axis for key in PR_CANDIDATES
                    for time in ("9/50","-9/50") for axis in core.AXES}
        metrics = {"rz_count","cx_count","rz_depth","cx_depth","total_depth","circuit_size"}
        if (len(pr) != 6 or len(q2_rows) != 2 or
                {(r["group"],r["metric"]) for r in statistics} !=
                {(g,m) for g in expected for m in metrics} or len(statistics) != 72):
            raise Stage1Failure("FULL_DF_Q4_SAVED_COMPARISON_SCOPE_MISMATCH")
        result.update(full_y4_q2_commit=Q2_COMMIT, full_y4_q2=q2_rows,
            high_precision_pr=pr, b1_commit=B1_COMMIT, b1_statistics=statistics,
            b1_qualification="FINITE_IID256_S0_CONDITIONAL_RARE_COSTS_UNKNOWN_NOT_UNCONDITIONAL_MEAN")
        return result


def preflight(root, output, source):
    run = core.preflight(root, output, source, study=STUDY, call_scope=scope())
    root = Path(root)
    blobs = {}
    for commit,path in ((Q2_COMMIT,Q2_PATH),(B1_COMMIT,B1_PATH+"/iid256-once")):
        for name in ("result.json","inputs.json","binding.json"):
            # B1's large result is unnecessary: the published statistics suffice.
            if commit == B1_COMMIT and name == "result.json":
                continue
            relative = path+"/"+name
            expected = subprocess.check_output(["git","rev-parse",commit+":"+relative],cwd=root,text=True).strip()
            actual = subprocess.check_output(["git","hash-object","--",relative],cwd=root,text=True).strip()
            if actual != expected:
                raise Stage1Failure("FULL_DF_Q4_SAVED_SNAPSHOT_MISMATCH")
            blobs[relative] = expected
        inputs = json.loads((root/path/"inputs.json").read_text())
        binding = json.loads((root/path/"binding.json").read_text())
        if (any(inputs[k] != run["inputs"][k] for k in (
                "hamiltonian_id","sector_id","state_id","input_data_sha256")) or
                binding["scope"]["compiler"] != json.loads(core._encode(core.b0.PROFILE))):
            raise Stage1Failure("FULL_DF_Q4_SAVED_CONDITIONS_MISMATCH")
    relative = B1_PATH+"/STATISTICS.csv"
    expected = subprocess.check_output(["git","rev-parse",B1_COMMIT+":"+relative],cwd=root,text=True).strip()
    if subprocess.check_output(["git","hash-object","--",relative],cwd=root,text=True).strip() != expected:
        raise Stage1Failure("FULL_DF_Q4_B1_STATISTICS_MISMATCH")
    blobs[relative] = expected
    run["saved_q2_b1_blobs"] = blobs
    return run


def execute(run, ledger, evidence, *, native_factory=Native):
    return core.execute(run, ledger, evidence, native_factory=native_factory,
                        rows=cases(), validator=validate_case)


def main():
    return core.main(scope_factory=scope, preflight_function=preflight, body=execute,
                     description="Fixed Full DF-Yoshida4 q4; FD5/FD6 only, one-shot")
