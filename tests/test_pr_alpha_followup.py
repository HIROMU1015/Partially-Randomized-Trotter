"""Artificial fixtures only. Never loads saved molecular numbers or runs A again."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pr_alpha_followup_runtime as runtime
core = runtime.core


def toy(method="B2", bias=0., cost=1):
    pairs = [dict(replica=i, seed=i+1, event_digest=hashlib.sha256(str(i).encode()).hexdigest(),
                  compiler_hash="a"*64, real=1, imag=1) for i in range(32)]
    r = dict(contract=core.CONTRACT, system="H6", cell_id="SYNTHETIC_"+method,
        method=method, epsilon=math.sqrt(2), bias_kind="USUAL_SIGNED_SAVED", allowance_factor=1.,
        bias=dict(real=bias, imag=bias), allowance=dict(real=0., imag=0.),
        log_B=0., log_B_margin=0., evidence_kind="EMPIRICAL", evidence_ref="SYNTHETIC_ONLY",
        normalization_role="B_CORR_IS_ESTIMATOR_MULTIPLIER", budget_role="B_BUDGET_IS_BOUND_ONLY",
        certificate_verified=False, population_winner_certified=False,
        probability_evidence="FRESH_IID_HADAMARD_MODEL", fresh_iid=True,
        axis_sign_convention="BIT0_PLUS1_COS_REAL_SDG_SIN_IMAG",
        corrected=dict(real=0., imag=0.), raw=dict(real=0., imag=0.), B_corr=1., B_budget=1.)
    r["baseline_accounting"] = core.frozen_rule().u_aware_shots(r["bias"], r["allowance"],
        epsilon=r["epsilon"], log_B_upper=0., evidence_kind="EMPIRICAL", evidence_ref="SYNTHETIC_ONLY")
    r["baseline_accounting"]["cost_acquisition_eligible"] = True
    r["cost"] = (dict(kind="PR_PAIRED_MEASURED", phase="confirmation", pairs=pairs, sample_count=32)
        if method == "B2" else dict(kind="DETERMINISTIC_MEASURED", real=str(cost), imag=str(cost)))
    return r


def exact_tail(n, p, low, high):
    value = sum((F(math.comb(n,k)) * p**k * (1-p)**(n-k)
                 for k in range(n+1) if k <= low or k >= high), F(0))
    return dict(probability=float(value), exact=str(value), numerical_status="SYNTHETIC_RATIONAL_ORACLE", underflow=False)


class BinomialTests(unittest.TestCase):
    def axis(self, **kw):
        args=dict(n=4, corrected=0., raw=0., B_corr=1., B_budget=1., headroom=.5,
                  allowance=0., backend=exact_tail)
        args.update(kw)
        return core.binomial_axis(**args)

    def test_ge_and_gt_integer_boundary(self):
        self.assertEqual(self.axis()["binomial"]["exact"], "5/8")
        self.assertEqual(self.axis(event="gt")["binomial"]["exact"], "1/8")

    def test_no_float_boundary_snapping(self):
        b=core.binomial_boundaries(4,0,1,F(1,2)+F(1,2**80))
        self.assertEqual((b["lower_last"],b["upper_first"]),(0,4))

    def test_normalization_budget_cannot_replace_correction(self):
        good=self.axis(corrected=1.,raw=.5,B_corr=2.,B_budget=3.)
        self.assertEqual(good["p_rational"],"3/4")
        with self.assertRaisesRegex(ValueError,"RAW_CORRECTED"):
            self.axis(corrected=1.,raw=.5,B_corr=3.,B_budget=3.)

    def test_bad_probability_and_nonfinite_rejected(self):
        for value in (2.,math.inf,math.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.axis(corrected=value,raw=value)

    def test_budget_must_bound_estimator(self):
        with self.assertRaisesRegex(ValueError,"NORMALIZATION_BOUND"):
            self.axis(B_corr=2.,B_budget=1.)

    def test_scipy_matches_exact_small_distribution(self):
        for n,p in ((4,F(1,2)),(17,F(1,4)),(43,F(3,4))):
            for low,high in ((0,n),(-1,n+1),(n//3,2*n//3+1)):
                got=core.scipy_tail(n,p,low,high)["probability"]
                self.assertAlmostEqual(got,exact_tail(n,p,low,high)["probability"],places=13)

    def test_degenerate_probability_is_exact(self):
        self.assertEqual(core.scipy_tail(10,F(0),0,11)["probability"],1.)
        self.assertEqual(core.scipy_tail(10,F(1),-1,11)["probability"],0.)

    def test_synthetic_large_n_no_enumeration(self):
        r=self.axis(n=10**12,headroom=1e-5,backend=core.scipy_tail)
        self.assertEqual(r["binomial"]["numerical_status"],"EXACT_DISTRIBUTION_FLOAT_EVALUATION")
        self.assertFalse(r["floating_probability_is_interval_certificate"])

    def test_nonempty_underflow_not_reported_as_zero(self):
        r=self.axis(backend=lambda *a:dict(probability=None,underflow=True))
        self.assertEqual(r["alpha_compliance"],"UNDETERMINED_NUMERICAL")

    def test_binomial_probability_not_hoeffding_bound(self):
        r=self.axis()
        self.assertNotEqual(r["binomial"]["probability"],r["hoeffding_bound"])

    def test_union_without_independent_axis_product(self):
        r=core.binomial_case(toy(),backend=exact_tail)
        self.assertIsNone(r["exact_joint_failure_probability"])
        self.assertEqual(r["union_bound_kind"],"SUM_OF_MARGINALS_NO_AXIS_INDEPENDENCE_ASSUMED")

    def test_wrong_axis_convention_rejected(self):
        r=toy();r["axis_sign_convention"]="S_THEN_H"
        with self.assertRaisesRegex(ValueError,"SIGN_CONVENTION"):
            core.binomial_case(r)

    def test_cost_se_not_measurement_evidence(self):
        r=toy();r["probability_evidence"]="SMALL_COST_SE"
        with self.assertRaisesRegex(ValueError,"COST_SE"):
            core.binomial_case(r)

    def test_saved_headroom_and_budget_binding(self):
        for field in ("headroom", "B_budget"):
            r=toy()
            if field=="headroom":r["baseline_accounting"]["axes"]["real"][field]+=.1
            else:r[field]=2.
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"BINDING"):
                core.binomial_case(r,backend=exact_tail)

    def test_old_rpe_contract_not_transferred(self):
        r=toy();r["contract"]="OLD_H4_RPE_BETA_ALLOCATION"
        with self.assertRaisesRegex(ValueError,"NO_RPE_OR_H4"):
            core.binomial_case(r)


class SensitivityTests(unittest.TestCase):
    def test_baseline_uses_frozen_rule_and_equal_split(self):
        r=toy();out=core.reaccount(r,"0.05")
        self.assertEqual(out["accounting"]["axes"],r["baseline_accounting"]["axes"])
        self.assertEqual(out["alpha_real"]+out["alpha_imag"],out["alpha_total"])

    def test_baseline_mismatch_fails_closed(self):
        r=toy();r["baseline_accounting"]["axes"]["real"]["shots"]+=1
        with self.assertRaisesRegex(ValueError,"BASELINE_REPRODUCTION"):
            core.reaccount(r,"0.05")

    def test_paired_cost_sample_count_and_covariance(self):
        r=toy()
        for i,p in enumerate(r["cost"]["pairs"]):p.update(real=1+2*(i%2),imag=3-2*(i%2))
        out=core.reaccount(r,.05)
        self.assertEqual(out["paired_samples"],32);self.assertEqual(out["cost_SE"],0.)
        self.assertFalse(out["cost_uncertainty_is_alpha_evidence"])

    def test_exploration_not_pooled_with_confirmation(self):
        r=toy();r["cost"]["phase"]="exploration"
        with self.assertRaisesRegex(ValueError,"CONFIRMATION_COST"):
            core.reaccount(r,.01)

    def test_missing_cost_not_zero(self):
        r=toy("B1");r["cost"]={"kind":"MISSING"}
        out=core.reaccount(r,.01)
        self.assertEqual(out["status"],"MISSING");self.assertIsNone(out["G_exact"])

    def test_model_cost_preserves_class(self):
        r=toy("B1");r["cost"]["kind"]="DETERMINISTIC_MODEL"
        self.assertEqual(core.reaccount(r,.05)["cost_class"],"DETERMINISTIC_MODEL")

    def test_mixed_axis_cost_keeps_measured_and_model(self):
        r=toy("B1");r["cost"].update(kind="DETERMINISTIC_MIXED",axis_classes=dict(real="MEASURED",imag="MODEL"))
        self.assertEqual(core.reaccount(r,.01)["axis_cost_classes"],r["cost"]["axis_classes"])

    def test_pair_compiler_mismatch_rejected(self):
        r=toy();r["cost"]["pairs"][0]["compiler_hash"]="b"*64
        with self.assertRaisesRegex(ValueError,"COMPILER_CHANGED"):core.reaccount(r,.01)

    def test_minimum_shot_and_certified_caller_not_certificate(self):
        r=toy();r.update(epsilon=100.,evidence_kind="CERTIFIED")
        out=core.reaccount(r,.01)
        self.assertEqual(out["accounting"]["N_total"],2)
        self.assertTrue(out["accounting"]["allowance_certification_claimed_by_caller"])
        self.assertFalse(out["accounting"]["certificate_verified_by_this_function"])

    def test_log_domain_keeps_no_cost(self):
        r=toy();r["log_B"]=100.
        out=core.reaccount(r,.01)
        self.assertEqual(out["status"],"LOG_DOMAIN_ONLY");self.assertIsNone(out["G_exact"])

    def test_ineligible_state_keeps_no_cost(self):
        r=toy(bias=2.);out=core.reaccount(r,.01)
        self.assertEqual(out["status"],"INELIGIBLE");self.assertIsNone(out["G_exact"])

    def test_unknown_evidence_remains_undetermined(self):
        r=toy();r["evidence_kind"]="UNKNOWN"
        out=core.reaccount(r,.01)
        self.assertEqual(out["status"],"UNDETERMINED")

    def test_comparison_rejects_unequal_precision(self):
        rows=[core.reaccount(toy(),.05),core.reaccount(toy("B1"),.05)]
        rows[1]["epsilon"]*=2
        with self.assertRaisesRegex(ValueError,"UNEQUAL_PR_PF"):
            core.compare(rows)

    def test_unresolved_candidate_blocks_full_domain_minimum(self):
        rows=[core.reaccount(toy(),.05),core.reaccount(toy("B1"),.05)]
        missing=deepcopy(rows[1]);missing.update(cell_id="SYNTHETIC_MISSING",status="MISSING",G_exact=None)
        r=core.compare(rows+[missing]);self.assertEqual(r["rank"],"UNRESOLVED")

    def test_integer_rounding_can_change_point_rank(self):
        pr=toy(bias=.4);pf=toy("B1",bias=.3,cost=F(34,25))
        a=core.compare([core.reaccount(x,.05) for x in (pr,pf)])
        b=core.compare([core.reaccount(x,.1) for x in (pr,pf)])
        self.assertNotEqual(a["rank"],b["rank"])
        self.assertFalse(b["population_winner_certified"])


class AdmissionTests(unittest.TestCase):
    def plan_grant(self, directory, stage="B1"):
        path=Path(directory)/"input.json";path.write_text("{}")
        plan=dict(schema="PR_ALPHA_FOLLOWUP_PLAN_V1",stage=stage,sealed=True,source_commit="a"*40,
            source_files=[],inputs_bound=True,input=dict(path=str(path),bytes=2,sha256=runtime.digest_bytes(b"{}")),
            payload_kind="SYNTHETIC",mandatory_stop=True,cross_stage_dispatch=False,
            scope=dict(contract=core.CONTRACT,allocation="EQUAL_TWO_AXES",alpha_total_grid=["0.01","0.05","0.10"]))
        output=Path(directory)/(stage+"_synthetic")
        grant=dict(stage=stage,approved_by_user=True,runtime_authorization=True,
            source_commit=plan["source_commit"],plan_sha256=runtime.digest_bytes(runtime.canonical(plan)),
            input_sha256=plan["input"]["sha256"],one_shot=True,retry=False,auto_next_stage=False,
            output=str(output),marker=str(output)+".oneshot")
        return plan,grant

    def test_unapproved_rejected_before_input_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p,g=self.plan_grant(d);g["approved_by_user"]=False;seen=[]
            with self.assertRaisesRegex(ValueError,"SEPARATE_APPROVAL"):
                runtime.execute(p,g,"B1",loader=lambda _:seen.append(1))
            self.assertEqual(seen,[])

    def test_inputs_must_be_bound(self):
        with tempfile.TemporaryDirectory() as d:
            p,g=self.plan_grant(d);p["inputs_bound"]=False;g["plan_sha256"]=runtime.digest_bytes(runtime.canonical(p))
            with self.assertRaisesRegex(ValueError,"INPUTS_NOT_BOUND"):runtime.validate_grant(p,g,"B1")

    def test_stage_grants_independent(self):
        with tempfile.TemporaryDirectory() as d:
            p,g=self.plan_grant(d)
            with self.assertRaisesRegex(ValueError,"STAGE_SCOPE"):runtime.validate_grant(p,g,"B2")

    def test_retry_and_scope_expansion_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p,g=self.plan_grant(d)
            for key in ("retry","auto_next_stage"):
                bad=dict(g);bad[key]=True
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,"NO_RETRY"):
                    runtime.validate_grant(p,bad,"B1")

    def test_failed_stage_consumes_only_own_marker(self):
        with tempfile.TemporaryDirectory() as d:
            p,g=self.plan_grant(d)
            def broken(*args):raise ValueError("SYNTHETIC_STOP")
            with self.assertRaisesRegex(ValueError,"SYNTHETIC_STOP"):
                runtime.execute(p,g,"B1",source_check=lambda p:None,loader=lambda p:{},calculator=broken)
            self.assertTrue(Path(g["marker"]).exists())
            p2,g2=self.plan_grant(d,"B2")
            self.assertFalse(Path(g2["marker"]).exists())
            self.assertEqual(runtime.validate_grant(p2,g2,"B2")[0],Path(g2["output"]))

    def test_synthetic_success_then_one_shot_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            p,g=self.plan_grant(d)
            out=runtime.execute(p,g,"B1",source_check=lambda p:None,loader=lambda p:{},calculator=lambda *a:{"fixture":"SYNTHETIC"})
            self.assertEqual(out["status"],"COMPLETE_MANDATORY_STOP")
            self.assertFalse(out["real_scientific_stage_executed"])
            with self.assertRaisesRegex(ValueError,"ALREADY_USED"):
                runtime.execute(p,g,"B1",source_check=lambda p:None,loader=lambda p:{})

    def test_partial_scientific_stop_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p,g=self.plan_grant(d)
            def broken(*args):raise runtime.ScientificStop("SYNTHETIC_EXCEEDS",dict(rows=[dict(status="EXCEEDS_BUDGET")]))
            with self.assertRaises(runtime.ScientificStop):
                runtime.execute(p,g,"B1",source_check=lambda p:None,loader=lambda p:{},calculator=broken)
            output=Path(g["output"])
            self.assertEqual(runtime.read_json(output/"terminal.json")["status"],"STOP")
            self.assertEqual(runtime.read_json(output/"partial_result.json")["rows"][0]["status"],"EXCEEDS_BUDGET")

    def test_changed_input_rejected_without_calculation(self):
        with tempfile.TemporaryDirectory() as d:
            p,g=self.plan_grant(d);Path(p["input"]["path"]).write_text("changed")
            with self.assertRaisesRegex(ValueError,"USED_INPUT_CHANGED"):
                runtime.load_bound_input(p)

    def test_source_scope_and_runtime_reject_only_used_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"used.py").write_bytes(b"used")
            p=dict(source_commit="a"*40,source_files=[dict(path="used.py",sha256=runtime.digest_bytes(b"used"))],
                   runtime=dict(python_version=sys.version.split()[0],python_executable_sha256=runtime.digest_bytes(Path(sys.executable).read_bytes()),packages=[]))
            with patch.object(runtime,"ROOT",root),patch.object(runtime.subprocess,"check_output",return_value=b"used"):
                runtime.source_gate(p)
                (root/"unrelated.txt").write_bytes(b"changed elsewhere")
                runtime.source_gate(p)
                (root/"used.py").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"USED_SOURCE_CHANGED"):runtime.source_gate(p)
            with self.assertRaisesRegex(ValueError,"PYTHON_RUNTIME_CHANGED"):
                runtime.runtime_gate(dict(python_version="0.0",packages=[]))
            with self.assertRaisesRegex(ValueError,"PYTHON_EXECUTABLE_CHANGED"):
                runtime.runtime_gate(dict(python_version=sys.version.split()[0],python_executable_sha256="0"*64,packages=[]))

    def test_missing_rows_block_minimum_and_baseline_is_first(self):
        records=[toy(),toy("B1")]
        records[1]["missing_fields"]=["synthetic_missing_cost"]
        p=dict(stage="B2",scope=dict(case_keys=[list(runtime.case_key(r)) for r in records],alpha_total_grid=["0.01","0.05","0.10"]),
               budget=dict(input_rows=2,accounting_rows=6))
        with patch.object(core,"reaccount",wraps=core.reaccount) as spy:
            out=runtime.compute(p,dict(records=records))
            self.assertEqual(spy.call_args_list[0].args[1],"0.05")
        self.assertEqual(out["accounting_rows"],6)
        self.assertTrue(all(c["rank"]=="UNRESOLVED" for c in out["comparisons"]))
        with self.assertRaisesRegex(ValueError,"EXACT_CANDIDATE"):
            runtime.scope_gate(p,records+[toy("B0")])

    def test_budget_checked_before_backend(self):
        r=toy();p=dict(stage="B1",scope=dict(case_keys=[list(runtime.case_key(r))]),
            budget=dict(input_rows=1,binomial_axis_evaluations=0))
        with patch.object(core,"binomial_case") as spy,self.assertRaisesRegex(ValueError,"CALL_BUDGET"):
            runtime.compute(p,dict(records=[r]))
        spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
