"""Artificial metadata corruption checks; no molecular arrays or numerical stats."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from fractions import Fraction as F

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('protocol_check',ROOT/'scripts/check_track_a_protocol_draft.py')
k=importlib.util.module_from_spec(s);s.loader.exec_module(k)


class StructureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for name in (*k.DOCS,'track_a_protocol_compliance_matrix.csv',k.BUNDLE+'/evidence_catalog.json',k.BUNDLE+'/requirements.json'):
            p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,p)

    def tearDown(self):self.tmp.cleanup()

    def rows(self,mutate):
        path=self.root/'track_a_protocol_compliance_matrix.csv'
        with path.open(newline='') as f:rows=list(csv.DictReader(f))
        mutate(rows)
        with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

    def test_current_structure(self):self.assertEqual(k.validate(self.root,verify_git=False)['status'],'PASS')

    def test_reject_missing_matrix_row(self):
        self.rows(lambda rows:rows.pop())
        with self.assertRaisesRegex(ValueError,'COVERAGE'):k.validate(self.root,verify_git=False)

    def test_reject_duplicate_matrix_row(self):
        self.rows(lambda rows:rows.append(rows[0].copy()))
        with self.assertRaisesRegex(ValueError,'COVERAGE'):k.validate(self.root,verify_git=False)

    def test_reject_adopted_or_execution_flags(self):
        self.rows(lambda rows:rows[0].update(new_science_authorized='TRUE'))
        with self.assertRaisesRegex(ValueError,'NO_SCIENCE'):k.validate(self.root,verify_git=False)

    def test_reject_union_as_fixed_independent_P(self):
        def change(rows):
            row=next(r for r in rows if r['series_id']=='H8_TOL_ONLY_DF28' and r['requirement_id']=='P_FIXED')
            row['cohort']='OLD_CONFIRMATION_UNION6_N32'
        self.rows(change)
        with self.assertRaisesRegex(ValueError,'NOT_UNION_SELECTION'):k.validate(self.root,verify_git=False)

    def test_reject_model_relabelled_direct(self):
        self.rows(lambda rows:next(r for r in rows if r['series_id']=='H6_TOL_ONLY_DF19').update(cost_evidence='ALL_DIRECT'))
        with self.assertRaisesRegex(ValueError,'MODEL_CANNOT'):k.validate(self.root,verify_git=False)

    def test_reject_legacy_relabelled_tol_only(self):
        self.rows(lambda rows:next(r for r in rows if r['series_id']=='H4_LEGACY_DF12' and r['requirement_id']=='C_DF').update(DF_policy='TOL_ONLY_1E-8'))
        with self.assertRaisesRegex(ValueError,'LEGACY_CANNOT'):k.validate(self.root,verify_git=False)

    def test_reject_missing_evidence(self):
        self.rows(lambda rows:rows[0].update(evidence_ids='E_NOT_OBTAINED'))
        with self.assertRaisesRegex(ValueError,'EVIDENCE_REFERENCE'):k.validate(self.root,verify_git=False)

    def test_reject_commit_mismatch(self):
        self.rows(lambda rows:rows[0].update(evidence_commits='0'*40))
        with self.assertRaisesRegex(ValueError,'COMMIT_BINDING'):k.validate(self.root,verify_git=False)

    def test_exact_two_axis_cost_identity_artificial_only(self):
        Np=(3,7);Nd=(11,2);Cp=(F(5,2),F(9,2));Cd=(F(7),F(3))
        Gp=sum(n*c for n,c in zip(Np,Cp));Gd=sum(n*c for n,c in zip(Nd,Cd))
        RN=F(sum(Np),sum(Nd));RC=(Gp/sum(Np))/(Gd/sum(Nd))
        self.assertEqual(Gp/Gd,RN*RC)
        self.assertEqual(Gp/Gd,sum(F(n*d,Gd)*F(p,n)*(c/d) for p,n,c,d in zip(Np,Nd,Cp,Cd)))
        self.assertNotEqual(Gp/Gd,RN*(sum(Cp)/sum(Cd)))


if __name__=='__main__':unittest.main()
