"""Artificial finite-selection and paired-price checks."""
import importlib.util
from pathlib import Path
import unittest

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("toy_pr_summary",root/"scripts/summarize_h4_pauli_pr_search.py")
T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)


class SelectionTests(unittest.TestCase):
    def test_neighbors_remain_one_coordinate_within_scope(self):
        p=dict(L_D=8,q=2,r=4,K=4);cid=T.R.spec_id(p)
        rows=[dict(candidate_id=cid,parameters=p,epsilon_sig=e,cohort="exploration",G_exact="10") for e in T.F.EPSILONS]
        candidates=T.proposals(rows,[cid],32)
        self.assertTrue(candidates);self.assertLessEqual(len(candidates),32)
        self.assertEqual(len({c["candidate_id"] for c in candidates}),len(candidates))
        for c in candidates:
            self.assertEqual(sum(c[k]!=p[k] for k in p),1)
            self.assertTrue(0<=c["L_D"]<=183)
        self.assertEqual(T.proposals(rows,[cid],32),candidates)

    def test_null_unmeasured_and_confirm_not_pooled(self):
        rows=[dict(candidate_id="bad",epsilon_sig="0.05",cohort="exploration",G_exact=None),
              dict(candidate_id="discovery",epsilon_sig="0.05",cohort="exploration",G_exact="9"),
              dict(candidate_id="confirmation",epsilon_sig="0.05",cohort="confirmation",G_exact="1")]
        self.assertEqual([r["candidate_id"] for r in T.ranked(rows,"0.05")],["discovery"])

    def test_paired_G_uses_covariance(self):
        p=[dict(cosine=10,sine=20),dict(cosine=20,sine=10)]
        g=T.A.paired(p,dict(cosine=2,sine=2),T.statistics())
        self.assertEqual(g["mean_exact"],"60");self.assertEqual(g["standard_error"],0)


if __name__=="__main__":unittest.main()
