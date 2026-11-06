"""Synthetic C2 orchestration checks; no quantum/scientific imports or data."""
import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("c2_runner",ROOT/"scripts/run_track_d_c2_q2.py")
c2=importlib.util.module_from_spec(spec);spec.loader.exec_module(c2)


class C2Tests(unittest.TestCase):
    def test_archive_mapping_keeps_fixed_raw_paths_and_rejects_unrelated_files(self):
        self.assertEqual(c2.member_destination("C1/circuits/reference_X_raw.qpy"),Path("circuits/reference_X_raw.qpy"))
        self.assertEqual(c2.member_destination("C1/references.npz"),Path("references.npz"))
        for name in ("../other.qpy","C1/../x.qpy","/C1/circuits/A.qpy","B1/result.json","C1/circuits/a.txt"):
            with self.assertRaises(ValueError):c2.member_destination(name)

    def test_four_wrapper_paired_prices_do_not_sum_q1_or_qubit_widths(self):
        compiled={}
        for arm,offset in (("symmetric_directional",10),("reference",7)):
            for axis in c2.AXES:
                compiled[arm+"_"+axis]=dict(metrics={k:offset+i for i,k in enumerate(c2.METRICS)})
        saved=[dict(sample="0",axis="X",symmetric_qubit_count="9",reference_qubit_count="17")]
        value=c2.comparisons(compiled,saved)
        self.assertEqual(len(compiled),4);self.assertEqual(len(value["q2_paired"]),2)
        self.assertEqual(value["q2_paired"][0]["delta"]["cx_count"],-3)
        self.assertEqual(value["q2_paired"][0]["reference"]["qubit_count"],12)
        self.assertEqual(value["q1_saved_individual_rows"],saved)
        self.assertFalse(value["sum_of_q1_wrapper_prices_is_q2_price"])


if __name__=="__main__":unittest.main()
