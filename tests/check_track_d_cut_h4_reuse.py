"""Guard differences only; invented records, never the molecular archive."""
import unittest
from trottertracks.track_d_cut_h4_reuse import checked_errors,check_request


class ReuseTests(unittest.TestCase):
    def test_wrong_path_stops(self):
        with self.assertRaisesRegex(ValueError,"CENTRAL_FIXED_PATH_ONLY"):
            check_request(dict(sample_id="other",q=4,sample=0),dict(selected=2),[])

    def test_noncentral_is_not_silently_reused(self):
        with self.assertRaisesRegex(ValueError,"CENTRAL_FIXED_PATH_ONLY"):
            check_request(dict(sample_id="q4_sample00",q=4,sample=0),dict(selected=1),[])

    def test_incomplete_old_wrappers_stop(self):
        with self.assertRaisesRegex(ValueError,"FOUR_SAVED_WRAPPERS_REQUIRED"):
            check_request(dict(sample_id="q4_sample00",q=4,sample=0),dict(selected=2),[])

    def test_old_error_rules(self):
        row=dict(label="compiled_vs_CAR_observable",tolerance=1e-10,rtol=0,max_entry=1e-13,frobenius=1e-12)
        checked_errors([row])
        with self.assertRaisesRegex(ValueError,"OLD_OBSERVABLE_CHECK"):checked_errors([dict(row,frobenius=3e-10)])
        with self.assertRaisesRegex(ValueError,"OLD_COMPONENT_CHECK"):checked_errors([dict(row,rtol=1e-5)])
