"""Evidence integrity tests, not scientific re-analysis or an independent review."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("integration", ROOT / "scripts/integrate_h468_paper_evidence.py")
integration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(integration)


class EvidenceStructure(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h468-structure-")
        self.root = Path(self.temp.name)
        paths = [r["path"] for r in integration.read_json(integration.BUNDLE + "upstream_preservation.json")["files"]]
        paths += [str(p.relative_to(ROOT)) for p in (ROOT / integration.BUNDLE).iterdir() if p.is_file()]
        for path in set(paths):
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(ROOT / path)

    def tearDown(self):
        self.temp.cleanup()

    def change(self, name, mutate):
        path = self.root / integration.BUNDLE / name
        obj = json.loads(path.read_text())
        mutate(obj)
        path.unlink()  # unlink only the disposable symlink, never its target
        path.write_text(json.dumps(obj, ensure_ascii=False))

    def test_saved_evidence_passes(self):
        self.assertEqual(integration.validate(self.root)["main_table_rows"], 6)

    def test_reject_cohort_pooling(self):
        self.change("main_table_proposed.json", lambda x: x["rows"][0].update(cohorts_pooled=True))
        with self.assertRaisesRegex(ValueError, "pooling"):
            integration.validate(self.root)

    def test_reject_false_all_reproduced_claim(self):
        self.change("main_table_proposed.json", lambda x: x["rows"][0].update(saved_direction_status="REPRODUCED"))
        with self.assertRaisesRegex(ValueError, "direction mismatch"):
            integration.validate(self.root)

    def test_reject_h8_double_count(self):
        self.change("main_table_proposed.json", lambda x: x["rows"].append(dict(x["rows"][-1])))
        with self.assertRaisesRegex(ValueError, "six unique"):
            integration.validate(self.root)

    def test_reject_h8_audit_certification(self):
        self.change("main_table_proposed.json", lambda x: x["rows"][4].update(audit_status="PASS"))
        with self.assertRaisesRegex(ValueError, "public audit gap"):
            integration.validate(self.root)

    def test_reject_invented_h4_ratio(self):
        self.change("main_table_proposed.json", lambda x: x["rows"][0].update(saved_reduction_percent_exact="0"))
        with self.assertRaisesRegex(ValueError, "ratio forbidden"):
            integration.validate(self.root)

    def test_reject_changed_saved_se(self):
        self.change("main_table_proposed.json", lambda x: x["rows"][0].update(paired_SE="0"))
        with self.assertRaisesRegex(ValueError, "saved token changed"):
            integration.validate(self.root)

    def test_reject_unseen_variance_zero(self):
        self.change("main_table_proposed.json", lambda x: x["rows"][1].update(unseen_variance=0))
        with self.assertRaisesRegex(ValueError, "zero-fill"):
            integration.validate(self.root)

    def test_reject_old_factor_transfer(self):
        self.change("main_table_proposed.json", lambda x: x["rows"][2].update(mechanism_for_this_cohort="OLD_N8_FACTORS"))
        with self.assertRaisesRegex(ValueError, "cohort transfer"):
            integration.validate(self.root)

    def test_reject_final_freeze(self):
        self.change("adoption_record.json", lambda x: x.update(final_evidence_freeze=True))
        with self.assertRaisesRegex(ValueError, "freeze forbidden"):
            integration.validate(self.root)

    def test_reject_original_figure_change(self):
        receipt = integration.read_json(integration.BUNDLE + "upstream_preservation.json")
        path = next(r["path"] for r in receipt["files"] if r["path"].startswith(integration.FIG))
        dest = self.root / path
        dest.unlink()
        dest.write_text("altered disposable copy")
        with self.assertRaisesRegex(ValueError, "upstream bytes changed"):
            integration.validate(self.root)


if __name__ == "__main__":
    unittest.main()
