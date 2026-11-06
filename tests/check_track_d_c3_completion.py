"""Synthetic preservation/regression checks for interrupted C3 completion."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import complete_track_d_c3_run02 as completion


class CompletionTests(unittest.TestCase):
    def test_partial_copy_preserves_original_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); old = root/"old"; new = root/"new"
            (old/"q8/circuits").mkdir(parents=True)
            data = b"synthetic saved compiled bytes\x00\xff"
            (old/"q8/circuits/reference_Y_compiled.qpy").write_bytes(data)
            (old/"SOURCE.json").write_text('{}\n')
            packet = dict(files=completion.inventory(old))
            completion.materialize(old, new, packet)
            self.assertEqual((new/"q8/circuits/reference_Y_compiled.qpy").read_bytes(), data)
            self.assertEqual(completion.inventory(old), packet["files"])
            self.assertFalse((new/"SOURCE.json").exists())
            with self.assertRaises(FileExistsError):
                completion.materialize(old, new, packet)
            (old/"q8/circuits/reference_Y_compiled.qpy").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "INTERRUPTED_INPUT_CHANGED"):
                completion.validate_inputs(old, packet)

if __name__ == "__main__":
    unittest.main()
