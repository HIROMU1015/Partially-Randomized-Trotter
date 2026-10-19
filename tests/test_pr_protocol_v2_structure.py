"""Draft-boundary and evidence-integrity mutations, not scientific re-analysis."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protocol_check", ROOT / "scripts/check_pr_protocol_v2_draft.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


class ProtocolDraftStructure(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="protocol-v2-")
        self.root = Path(self.temp.name)
        paths = [r["path"] for r in json.loads((ROOT / mod.B / "preservation.json").read_text())["files"]]
        paths += [str(p.relative_to(ROOT)) for p in (ROOT / mod.B).iterdir() if p.is_file()]
        paths += [mod.PROTO, "docs/research/track_a_pr_resource_protocol_v2_changes_20261019.md"]
        for path in set(paths):
            dest = self.root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(ROOT / path)

    def tearDown(self):
        self.temp.cleanup()

    def edit(self, name, mutate, manifest_key=None):
        path = self.root / mod.B / name
        obj = json.loads(path.read_text())
        mutate(obj)
        path.unlink()
        path.write_text(json.dumps(obj, ensure_ascii=False))
        if manifest_key:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.edit("protocol_manifest.json", lambda d: d[manifest_key].update(sha256=digest))

    def edit_stage(self, system, stage, status):
        path = self.root / mod.B / "stage_status.csv"
        rows = list(csv.DictReader(io.StringIO(path.read_text())))
        next(r for r in rows if r["system"] == system and r["stage"] == stage)["status"] = status
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        path.unlink()
        path.write_text(stream.getvalue())
        self.edit("protocol_manifest.json", lambda d: d["stage_status"].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def test_valid(self):
        self.assertFalse(mod.check(self.root)["v2_adopted"])

    def test_reject_adoption(self):
        self.edit("draft_status.json", lambda d: d.update(status="ADOPTED"))
        with self.assertRaisesRegex(ValueError, "silently adopted"):
            mod.check(self.root)

    def test_reject_h2_a_stale_status_even_with_updated_hash(self):
        self.edit_stage("H2", "A", "NOT_EXECUTED_PREPARED_ONLY")
        with self.assertRaisesRegex(ValueError, "H2 completed stage lost"):
            mod.check(self.root)

    def test_reject_h2_p_n_inferred(self):
        self.edit("H2_saved_condition_projection.json", lambda d: d["P_contract_saved"].update(n=48), "H2_saved_condition_projection")
        with self.assertRaisesRegex(ValueError, "P unsealed field"):
            mod.check(self.root)

    def test_reject_h2_proposal_silently_adopted(self):
        self.edit("H2_latest_condition_projection.json", lambda d: d["P_draft_saved"].update(P_n_final={"candidate": 80}), "H2_latest_condition_projection")
        with self.assertRaisesRegex(ValueError, "P unsealed field"):
            mod.check(self.root)

    def test_reject_h2_history_overwrite(self):
        self.edit("H2_status_history.json", lambda d: d["H2_stages"].update(A="COMPLETE"), "H2_status_history")
        with self.assertRaisesRegex(ValueError, "H2 A history lost"):
            mod.check(self.root)

    def test_reject_h2_rare_zero_fill(self):
        self.edit("H2_latest_condition_projection.json", lambda d: d["S_STATS"]["saved_rare_information"][0].update(unobserved_strata_zero_filled=True), "H2_latest_condition_projection")
        with self.assertRaisesRegex(ValueError, "unseen strata zero filled"):
            mod.check(self.root)

    def test_reject_fp_n1_pooling(self):
        self.edit("sampling_and_seed_status.json", lambda d: d["H2"].update(FP_A_pooled=True), "sampling_seed_status")
        with self.assertRaisesRegex(ValueError, "pooling"):
            mod.check(self.root)

    def test_reject_old_h2_sector(self):
        self.edit("H2_saved_condition_projection.json", lambda d: d["sector"].update(dimension=6), "H2_saved_condition_projection")
        with self.assertRaisesRegex(ValueError, "old H2 input"):
            mod.check(self.root)

    def test_reject_h2_paper_inclusion(self):
        self.edit("draft_status.json", lambda d: d.update(H2_paper_main_table_member=True))
        with self.assertRaisesRegex(ValueError, "H2_paper"):
            mod.check(self.root)

    def test_reject_h8_audit_upgrade(self):
        self.edit_stage("H8", "P", "COMPLETE_FULL_PUBLIC_AUDIT")
        with self.assertRaisesRegex(ValueError, "audit gap"):
            mod.check(self.root)

    def test_reject_h8_s_stats_inference(self):
        self.edit_stage("H8", "S_STATS", "COMPLETE")
        with self.assertRaisesRegex(ValueError, "H8 S_STATS inferred"):
            mod.check(self.root)

    def test_reject_contract_change_authorization(self):
        self.edit("draft_status.json", lambda d: d.update(existing_contracts_changed=True))
        with self.assertRaisesRegex(ValueError, "existing_contracts"):
            mod.check(self.root)

    def test_reject_input_equivalence_claim(self):
        self.edit("system_conditions.json", lambda d: d["systems"][0].update(all_input_procedures_identical=True), "system_conditions")
        with self.assertRaisesRegex(ValueError, "input equivalence inferred"):
            mod.check(self.root)

    def test_reject_v1_mutation(self):
        dest = self.root / mod.V1
        dest.unlink()
        dest.write_text("altered disposable v1")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            mod.check(self.root)

    def test_reject_paper_numeric_mutation(self):
        dest = self.root / mod.B / "H468_independent_confirmations.csv"
        text = dest.read_text().replace("39405103093/554", "0")
        dest.unlink()
        dest.write_text(text)
        with self.assertRaises(ValueError):
            mod.check(self.root)


if __name__ == "__main__":
    unittest.main()
