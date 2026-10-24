"""Technical evidence guards only; no scientific runner or kernel imports."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import adopt_h46_standard_y4_confirmation as adoption


class AdoptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = adoption.render()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        dest = self.root / adoption.OUT
        dest.mkdir(parents=True)
        for name, content in self.files.items():
            (dest / name).write_bytes(content)

    def reject_change(self, filename, change):
        path = self.root / adoption.OUT / filename
        if filename.endswith(".json"):
            data = json.loads(path.read_text())
            change(data)
            path.write_text(json.dumps(data, ensure_ascii=False))
        else:
            path.write_bytes(change(path.read_bytes()))
        try:
            with self.assertRaisesRegex(ValueError, "changed"):
                adoption.validate(self.root)
        finally:
            path.write_bytes(self.files[filename])

    def test_fixed_literals_and_cross_index_transcription(self):
        self.assertEqual(adoption.validate(self.root)["status"], "PASS")
        record = adoption.json_saved(self.files["adoption_record.json"])
        self.assertEqual(record["result_commit"], adoption.RESULT)
        self.assertEqual(record["saved_result_decision"], "REPRODUCED_APPROXIMATE")
        refs = adoption.json_saved(self.files["source_catalog.json"])["sources"]
        for entry in record["groups_and_comparisons_saved"]:
            original = adoption.json_saved(adoption.git_bytes(adoption.RESULT, adoption.REC + "results.json"))
            self.assertEqual(entry["result_saved"], original["results"][entry["system"]])
        role_rows = adoption.csv_saved(self.files["evidence_roles.csv"])
        ledger_rows = adoption.csv_saved(self.files["evidence_ledger_v5.csv"])
        ids = {r["evidence_row_id"] for r in ledger_rows}
        self.assertTrue(all(r["evidence_id"] in ids and r["source_id"] in refs for r in role_rows))
        self.assertEqual(len({r["identity"] for r in role_rows}), len(role_rows))

    def test_historical_csv_bytes_and_cells_unchanged(self):
        for new, old, count in [("claim_evidence_matrix_v3.csv", "claim_evidence_matrix_v2.csv", 21),
                                ("evidence_ledger_v5.csv", "evidence_ledger_v4.csv", 391),
                                ("paper_data_manifest.csv", "paper_data_manifest.csv", 9)]:
            original = adoption.git_bytes(adoption.PAPER, adoption.PRIOR + old)
            self.assertTrue(self.files[new].startswith(original))
            self.assertEqual(adoption.csv_saved(self.files[new])[:count], adoption.csv_saved(original))

    def test_presentation_whitespace_is_not_scientific_evidence(self):
        path = self.root / adoption.OUT / "adoption_record.json"
        path.write_text(json.dumps(json.loads(path.read_text()), ensure_ascii=False))
        self.assertEqual(adoption.validate(self.root)["status"], "PASS")

    def test_reject_changed_mean(self):
        self.reject_change("adoption_record.json", lambda d: d["groups_and_comparisons_saved"][0]["groups_saved"][0].update(G_mean="0"))

    def test_reject_changed_SE_or_approximate_interval(self):
        for field in ["difference_SE", "approximate_interval_low"]:
            with self.subTest(field=field):
                (self.root / adoption.OUT / "adoption_record.json").write_bytes(self.files["adoption_record.json"])
                self.reject_change("adoption_record.json", lambda d: d["groups_and_comparisons_saved"][0]["comparison_saved"].update({field: "0"}))

    def test_reject_source_or_input_or_contract_substitution(self):
        for field in ["SOURCE", "input_manifest_sha256", "contract_sha256", "result_commit"]:
            with self.subTest(field=field):
                (self.root / adoption.OUT / "adoption_record.json").write_bytes(self.files["adoption_record.json"])
                self.reject_change("adoption_record.json", lambda d: d.update({field: "WRONG"}))

    def test_reject_pooling_and_77_as_comparison_family(self):
        self.reject_change("adoption_record.json", lambda d: d.update(cohort_pooling=True))
        self.reject_change("adoption_record.json", lambda d: d["family_policy_saved"].update(family_size=77))

    def test_reject_m5_best_or_exploration_transfer(self):
        for field, value in [("candidate", "m5_best"), ("cohort", "EXPLORATION")]:
            with self.subTest(field=field):
                (self.root / adoption.OUT / "adoption_record.json").write_bytes(self.files["adoption_record.json"])
                self.reject_change("adoption_record.json", lambda d: d["groups_and_comparisons_saved"][1]["groups_saved"][1].update({field: value}))

    def test_reject_epsilon_q_prefix_mixup(self):
        for field in ["epsilon", "q", "prefix", "n"]:
            with self.subTest(field=field):
                (self.root / adoption.OUT / "adoption_record.json").write_bytes(self.files["adoption_record.json"])
                self.reject_change("adoption_record.json", lambda d: d["groups_and_comparisons_saved"][0]["groups_saved"][0].update({field: "999"}))

    def test_reject_rigorous_guarantee_zero_tail_or_final_adoption(self):
        self.reject_change("adoption_record.json", lambda d: d.update(rare_tail_bound=0, final_evidence_freeze=True))
        self.reject_change("paper_manifest.json", lambda d: d.update(population_mean_guarantee=True, final_paper_claim_adoption=True))

    def test_reject_changed_same_q_negative_control(self):
        self.reject_change("adoption_record.json", lambda d: d["groups_and_comparisons_saved"][0]["same_q_controls_saved"][0].update(reversal="True"))

    def test_reject_old_claim_change(self):
        self.reject_change("claim_evidence_matrix_v3.csv", lambda b: b.replace(b"CE1,", b"CE1_CHANGED,", 1))

    def test_reject_failed_attempt_erasure(self):
        self.reject_change("adoption_record.json", lambda d: d.update(failed_attempt_result_commit=adoption.RESULT))

    def test_reject_index_alias_as_new_evidence(self):
        self.reject_change("three_environment_overlay.json", lambda d: d.update(independent_evidence_increment_from_indexing=77))

    def test_reject_original_main_table_change(self):
        manifest = adoption.json_saved(self.files["paper_manifest.json"])
        saved = manifest["main_table"]
        path = self.root / saved["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"altered disposable copy")
        # All fixed Git blobs have already been cached; only the disposable
        # preservation target changes. The real original is never written.
        with patch.object(adoption, "ROOT", self.root):
            with self.assertRaisesRegex(ValueError, "protected original changed"):
                adoption.render()

    def test_reject_new_scientific_call_or_compare_release(self):
        self.reject_change("STOP.json", lambda d: d["new_scientific_calls"].update(compile=1))
        self.reject_change("STOP.json", lambda d: d.update(old_strict_COMPARE="READY"))

    def test_only_metadata_modules_imported(self):
        modules = [Path(adoption.__file__), Path(adoption.__file__).with_name("integrate_h468_paper_evidence.py")]
        allowed = {"argparse", "csv", "functools", "hashlib", "io", "json", "pathlib", "subprocess", "integrate_h468_paper_evidence"}
        for module in modules:
            tree = ast.parse(module.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(n.name in allowed for n in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, allowed)


if __name__ == "__main__":
    unittest.main()
