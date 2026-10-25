"""Independent dense C check of the small-system contraction bridge."""
from fractions import Fraction as Q
import copy
import importlib
import json
from pathlib import Path
import sys
import types
import unittest

pkg=types.ModuleType("pr_mean_test");pkg.__path__=[str(Path(__file__).resolve().parents[1]/"src/trotterlib")];sys.modules[pkg.__name__]=pkg
M=importlib.import_module(pkg.__name__+".h4_pauli_pr_mean")
B=importlib.import_module(pkg.__name__+".h4_arb_bridge")
FIXTURES=Path(__file__).resolve().parents[1]/"artifacts/non_df_h4_pauli_pr_search_preparation/2026-10-25/artificial_fixtures.json"


class MeanTests(unittest.TestCase):
    def test_dense_C_operator_oracle_contained(self):
        C=B._backend()
        for f in json.loads(FIXTURES.read_text()):
            d=f["law"];raw=f["raw"];cfg=d["config"]
            result=M.artificial_mean(d,raw,f["reference"],64)
            with C.ctx.workprec(192):
                I=C.identity(4);components=[C.pauli(c["label"])*c["sign"] for c in d["components"]]
                rotations=[[I*C.real(a).cos()+P*C.acb(0,-C.real(a).sin()) for P in components] for a in d["angles"]]
                pis=list(map(Q,d["order_mapping"]["exact_masses"]))
                # Frozen C has fixed orders (0,2,4). K2 embeds exactly by
                # zero mass at order4; the old numerical SOURCE is unchanged.
                if cfg["K"]==2:
                    rotations.append([I for _ in components]);pis.append(Q(0))
                F=C.finite_implemented_mean(components,rotations,list(map(Q,d["component_mapping"]["exact_masses"])),pis)
                retained=[]
                for b in d["retained"]:
                    a=C.real(C.rational(b["coefficient"])*C.rational(d["step_time"])/2)
                    retained.append(I*a.cos()+C.pauli(b["label"])*C.acb(0,-a.sin()))
                state=C.acb_mat([[C.complex_ball(z)] for z in raw])
                out=C.evolve(state,retained,F,C.phase(C.real(d["scalar_step_binary64"])),cfg["q"],cfg["r"])
                z=C.signal(state,out)*C.real(d["B_saved_binary64"])/C.signal(state,state).real
                for key,ball in (("real",z.real),("imag",z.imag)):
                    bounds=result["physical_corrected_mean"][key]
                    self.assertLessEqual(Q(bounds["lower"]),C.dyadic(ball.lower()))
                    self.assertGreaterEqual(Q(bounds["upper"]),C.dyadic(ball.upper()))
            self.assertEqual(result["state_actions"],1);self.assertEqual(result["B_applied_count"],1)
            self.assertFalse(result["state_renormalized"])

    def test_raw_ray_scale_and_bias_reference_whole_interval(self):
        f=json.loads(FIXTURES.read_text())[1]
        ref=dict(reference=dict(real=dict(lower="1/10",upper="1/5"),imag=dict(lower="-1/10",upper="1/10")))
        a=M.artificial_mean(f["law"],f["raw"],ref,64)
        raw2=[[2*x,2*y] for x,y in f["raw"]]
        b=M.artificial_mean(f["law"],raw2,f["reference"],64)
        self.assertEqual(Q(b["raw_norm_exact"]),4*Q(a["raw_norm_exact"]))
        for key in ("real","imag"):
            u=a["physical_corrected_mean"][key];v=b["physical_corrected_mean"][key]
            self.assertLessEqual(max(Q(u["lower"]),Q(v["lower"])),min(Q(u["upper"]),Q(v["upper"])))
        self.assertGreater(Q(a["axes"]["cosine"]["interval_width_exact"]),Q(1,40))

    def test_bad_law_cannot_use_contraction_radius(self):
        f=json.loads(FIXTURES.read_text())[0];d=copy.deepcopy(f["law"])
        d["order_mapping"]["exact_masses"][0]="2"
        with self.assertRaises(ValueError):M.artificial_mean(d,f["raw"],f["reference"])

    def test_real_H4_dispatch_absent(self):
        with self.assertRaisesRegex(ValueError,"NO_REAL_H4"):
            M.artificial_mean(dict(n_system=8),[],{})


if __name__=="__main__":unittest.main()
