"""Metadata-only regression for the observed completion-order failure."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('v1_complete',Path(__file__).resolve().parents[1]/'scripts/complete_h4_df10_price_v1.py')
completion=importlib.util.module_from_spec(spec);spec.loader.exec_module(completion)


class CompletionTests(unittest.TestCase):
    def test_completion_order_is_not_path_identity(self):
        files=['/synthetic/'+str(i)+'/result.json' for i in [0,3,7,2,1,4,10,6,8,5,9,*range(11,33)]]
        actual=completion.canonical_trial_files(files)
        self.assertEqual(set(files),set(actual))
        self.assertEqual([int(Path(p).parent.name) for p in actual],list(range(33)))

    def test_missing_or_duplicated_path_is_rejected(self):
        for ids in [list(range(32)),[0,*range(32)]]:
            with self.assertRaises(ValueError):
                completion.canonical_trial_files(['/synthetic/'+str(i)+'/result.json' for i in ids])


if __name__=='__main__':unittest.main()
