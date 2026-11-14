"""Artificial metadata tests; no real price arithmetic/fit/compile."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import admit_pr_df_ld1_m4_holdout as C


def fixture():
    # Reported error values are intentionally supplied, never recomputed here.
    predicted=dict(axis='X',predicted_RZ=dict(numerator=101,denominator=2))
    rows=[dict(index=i,axis='X',cohort='artificial') for i in range(4)]
    unit=dict(m=4,axis='X',n_paths=4,P=predicted['predicted_RZ'],O=dict(numerator=52,denominator=1),
        signed_error={'supplied':'signed'},absolute_error={'supplied':'absolute'},
        signed_relative_error={'supplied':'relative'},absolute_relative_error={'supplied':'absolute relative'},
        valid=True,reasons=[],receipts=rows)
    accepted=[dict(status='ACCEPTED_SAVED_RECEIPT') for i in range(4)]
    return [unit,deepcopy(unit),deepcopy(predicted),deepcopy(rows),accepted]


class Tests(unittest.TestCase):
    def test_copies_saved_exact_metrics_without_recomputing(self):
        d=fixture();out=C.check_comparison(*d)
        self.assertEqual(out,{k:d[0][k] for k in C.EXACT_KEYS})
        self.assertIsNot(out['P'],d[0]['P'])

    def test_prediction_and_axis_mismatch_rejected(self):
        d=fixture();d[2]['predicted_RZ']['numerator']=102
        with self.assertRaisesRegex(ValueError,'frozen exact'):C.check_comparison(*d)
        d=fixture();d[2]['axis']='Y'
        with self.assertRaisesRegex(ValueError,'identity'):C.check_comparison(*d)

    def test_four_paths_and_accepted_receipts_required(self):
        d=fixture();d[3].pop()
        with self.assertRaisesRegex(ValueError,'Four original'):C.check_comparison(*d)
        d=fixture();d[4][0]['status']='HELD'
        with self.assertRaisesRegex(ValueError,'four axis'):C.check_comparison(*d)

    def test_receipt_binding_and_source_comparison_must_match(self):
        d=fixture();d[1]['O']['numerator']=999
        with self.assertRaisesRegex(ValueError,'consistency'):C.check_comparison(*d)
        d=fixture();d[0]['receipts'][0]['axis']='Y';d[1]=deepcopy(d[0])
        d[3]=[dict(index=i,axis='X',cohort='artificial') for i in range(4)]
        with self.assertRaisesRegex(ValueError,'these original receipts'):C.check_comparison(*d)

    def test_entrypoint_never_calls_numerical_science(self):
        tree=ast.parse(Path(C.__file__).read_text())
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {'fit','predict','calculate','mean','evaluate','sample_requests','transpile','compile','cost'})
        comparison=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check_comparison')
        self.assertFalse(any(isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Sub,ast.Div)) for n in ast.walk(comparison)))


if __name__=='__main__':unittest.main()
