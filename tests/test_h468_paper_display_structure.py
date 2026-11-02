"""Mutation checks for manuscript metadata; no science or statistics execution."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("display_check", ROOT / "scripts/check_h468_paper_display.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class DisplayStructure(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="h468-display-")
        self.root = Path(self.temp.name)
        paths = [x["path"] for x in json.loads((ROOT / mod.B / "preservation.json").read_text())["files"]]
        paths += [str(p.relative_to(ROOT)) for p in (ROOT / mod.B).iterdir() if p.is_file()]
        for path in set(paths):
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(ROOT / path)

    def tearDown(self):
        self.temp.cleanup()

    def edit_json(self, path, change):
        dest = self.root / mod.B / path
        obj = json.loads(dest.read_text())
        change(obj)
        dest.unlink()
        dest.write_text(json.dumps(obj, ensure_ascii=False))

    def test_valid_package(self):
        self.assertEqual(mod.check(self.root)["main_table_rows"], 6)

    def test_reject_old_figure_relabelling(self):
        self.edit_json("layout_decision.json", lambda d: d.update(Figure2A="H4_DF10_INDEPENDENT_P"))
        with self.assertRaisesRegex(ValueError, "cohort relabelling"):
            mod.check(self.root)

    def test_reject_h8_double_count(self):
        self.edit_json("layout_decision.json", lambda d: d.update(H8_B_is_same_evidence_as_Table1=False))
        with self.assertRaisesRegex(ValueError, "double counting"):
            mod.check(self.root)

    def test_reject_numerical_processing(self):
        self.edit_json("publication_manifest.json", lambda d: d.update(new_numerical_analysis=True))
        with self.assertRaisesRegex(ValueError, "scientific scope"):
            mod.check(self.root)

    def test_reject_complete_input_identity(self):
        self.edit_json("publication_manifest.json", lambda d: d.update(input_procedures_fully_identical=True))
        with self.assertRaisesRegex(ValueError, "input_procedures"):
            mod.check(self.root)

    def test_reject_population_guarantee(self):
        self.edit_json("publication_manifest.json", lambda d: d.update(population_mean_certified=True))
        with self.assertRaisesRegex(ValueError, "population_mean"):
            mod.check(self.root)

    def test_reject_stale_a82_status(self):
        self.edit_json("publication_manifest.json", lambda d: d["figure2"].update(A82_current="MISSING"))
        with self.assertRaisesRegex(ValueError, "stale A82"):
            mod.check(self.root)

    def test_reject_new_p_factor_transfer(self):
        self.edit_json("publication_manifest.json", lambda d: d["figure2"].update(new_factors_transferred_to_H4_H6_P=True))
        with self.assertRaisesRegex(ValueError, "factor transfer"):
            mod.check(self.root)

    def test_reject_changed_table_cost(self):
        dest = self.root / mod.B / "main_table.csv"
        text = dest.read_text().replace("39405103093/554", "0")
        dest.unlink()
        dest.write_text(text)
        with self.assertRaisesRegex(ValueError, "numeric bytes changed"):
            mod.check(self.root)

    def test_reject_final_freeze(self):
        self.edit_json("publication_manifest.json", lambda d: d.update(final_evidence_freeze=True))
        with self.assertRaisesRegex(ValueError, "final_evidence_freeze"):
            mod.check(self.root)

    def test_reject_h8_pr_audit_upgrade_even_with_new_document_hash(self):
        dest = self.root / mod.B / "h8_public_evidence_status.csv"
        text = dest.read_text().replace("REPORTED_LOCAL_PASS_NOT_REVERIFIED_FROM_PUBLIC_RECEIPTS", "PUBLIC_SELECTED_PF_RECEIPT_VERIFIED")
        dest.unlink()
        dest.write_text(text)
        self.edit_json("publication_manifest.json", lambda d: d["H8_public_evidence"].update(sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        with self.assertRaisesRegex(ValueError, "PR audit gap erased"):
            mod.check(self.root)

    def test_reject_cem_link_mismatch_even_with_new_document_hash(self):
        dest = self.root / mod.B / "evidence_linkage.csv"
        text = dest.read_text().replace("CE_H4_DF10_P_1", "CE_DOES_NOT_EXIST")
        dest.unlink()
        dest.write_text(text)
        self.edit_json("publication_manifest.json", lambda d: d["display_evidence_linkage"].update(sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
        with self.assertRaisesRegex(ValueError, "unknown CEM claim"):
            mod.check(self.root)

    def test_reject_original_artwork_change(self):
        dest = self.root / mod.FIG / "figures/figure2A.svg"
        dest.unlink()
        dest.write_text("disposable altered artwork")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            mod.check(self.root)


if __name__ == "__main__":
    unittest.main()
