"""Artificial D-seal boundary checks; no saved molecular data is processed."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_tol_main_contract as c
import seal_h4_tol_d as s


def fixture(directory):
    draft={name:'synthetic' for name in s.IMMUTABLE}
    draft.update(stage='D',SOURCE=s.SCIENCE_SOURCE,binding=deepcopy(c.BINDING),plan=c.plan('D'),
        approved_by_user=False,runtime_authorization=False,actual_scientific_calls=0,
        next_stage_authorized=False,automatic_retry=False,output=str(directory/'output'),lock=str(directory/'ONE_SHOT'))
    return draft


class SealBoundary(unittest.TestCase):
    def test_fixed_D_contract_stays_unapproved(self):
        with tempfile.TemporaryDirectory() as d:
            draft=fixture(Path(d));s.check_scope(deepcopy(draft),draft)

    def test_A_cannot_be_sealed_or_activated(self):
        with tempfile.TemporaryDirectory() as d:
            draft=fixture(Path(d));contract=dict(draft,stage='A')
            with self.assertRaisesRegex(ValueError,'D_ONLY'):s.check_scope(contract,draft)

    def test_candidate_budget_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            draft=fixture(Path(d));contract=deepcopy(draft)
            contract['plan']['calls']['wrapper_compile_maximum']=373
            with self.assertRaisesRegex(ValueError,'FIXED_D_SOURCE_PLAN'):s.check_scope(contract,draft)

    def test_rank_state_compiler_change_rejected(self):
        for key,value in [('rank',12),('state','x'*64),('compiler','y'*64)]:
            with tempfile.TemporaryDirectory() as d:
                draft=fixture(Path(d));contract=deepcopy(draft);contract['binding'][key]=value
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'BINDING'):s.check_scope(contract,draft)

    def test_runtime_source_and_input_rebinding_rejected(self):
        for key in ('runtime_hashes','provider_hashes','inputs','native_config','FP_evidence'):
            with tempfile.TemporaryDirectory() as d:
                draft=fixture(Path(d));contract=deepcopy(draft);contract[key]='changed'
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'UNCHANGED_PHASE0'):s.check_scope(contract,draft)

    def test_seal_cannot_issue_user_approval_or_retry(self):
        for key in ('approved_by_user','runtime_authorization','automatic_retry','next_stage_authorized'):
            with tempfile.TemporaryDirectory() as d:
                draft=fixture(Path(d));contract=dict(draft,**{key:True})
                with self.subTest(key=key),self.assertRaises(ValueError):s.check_scope(contract,draft)

    def test_existing_one_shot_or_output_rejected(self):
        for key in ('lock','output'):
            with tempfile.TemporaryDirectory() as d:
                draft=fixture(Path(d));Path(draft[key]).touch()
                with self.subTest(key=key),self.assertRaisesRegex(ValueError,'UNUSED_D_NAMESPACE'):s.check_scope(draft,draft)

    def test_direct_witness_refuse_before_science_and_leave_no_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);contract=fixture(d);c.save(d/'contract',contract);c.save(d/'grant',c.invalid_grant('D',contract))
            rows=s.refuse(d/'contract',d/'grant',d)
            self.assertEqual(len(rows),2);self.assertTrue(all(r['refused_before_science'] for r in rows))
            self.assertFalse(Path(contract['output']).exists());self.assertFalse(Path(contract['lock']).exists())


if __name__=='__main__':unittest.main()
