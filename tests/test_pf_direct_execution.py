"""Artificial-only checks for the production scope, bounds and admission gates."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import pf_direct_compile_contract as c
import pf_direct_execution as e


class ProductionBoundaryTests(unittest.TestCase):
    def test_current_exact_scope(self):
        _,rows=e.expected_tasks('current_4series_722')
        self.assertEqual(len(rows),722)
        self.assertEqual({r['system'] for r in rows},{'H2_DF3','H4_DF10','H6_DF19','H8_DF28'})

    def test_legacy_exact_scope(self):
        _,rows=e.expected_tasks('legacy_h4_df12_66')
        self.assertEqual(len(rows),66)
        self.assertEqual({r['system'] for r in rows},{'H4_LEGACY_DF12'})

    def test_disjoint_complete_scope(self):
        allrows,left=e.expected_tasks(e.GROUPS[0]);_,right=e.expected_tasks(e.GROUPS[1])
        a={r['task_id'] for r in left};b={r['task_id'] for r in right}
        self.assertFalse(a&b);self.assertEqual(a|b,{r['task_id'] for r in allrows})

    def test_no_grant_rejected_before_scope_access(self):
        with patch.object(e,'expected_tasks',side_effect=AssertionError('scope accessed')):
            with self.assertRaisesRegex(ValueError,'EXPLICIT_SCIENTIFIC_GRANT_REQUIRED'):
                e.authorize({'contract_id':e.GROUPS[0]}, {})

    def test_publication_does_not_authorize_science(self):
        with self.assertRaisesRegex(ValueError,'EXPLICIT_SCIENTIFIC_GRANT_REQUIRED'):
            e.authorize({'contract_id':e.GROUPS[0]},dict(action='PUBLICATION',approved_by_user=True,runtime_authorization=True))

    def test_invalid_digest_rejected(self):
        with self.assertRaisesRegex(ValueError,'EXACT_GRANT_BINDING'):
            e.authorize({'contract_id':e.GROUPS[0]},dict(action='PF_DIRECT_COMPILE_SCIENTIFIC_EXECUTION',
                approved_by_user=True,runtime_authorization=True,spec_digest='wrong'))

    def test_structural_bound_no_fixed_ceiling(self):
        class Native:
            @staticmethod
            def block_instruction_bound(block,mode):return block*(2 if mode=='DIRECTIONAL' else 1)
        self.assertEqual(e.instruction_bound(Native,[5,7],16,'4th'),16*3*36+1)
        self.assertEqual(e.instruction_bound(Native,[5,7],256,'4th'),256*3*36+1)

    def test_s2_s4_bound(self):
        class Native:
            @staticmethod
            def block_instruction_bound(block,mode):return block
        self.assertEqual(e.instruction_bound(Native,[2],1,'2nd'),5)
        self.assertEqual(e.instruction_bound(Native,[2],1,'4th'),13)

    def test_admit_beyond_historical_worker_limits(self):
        pending=[dict(memory_estimate=1) for _ in range(80)]
        self.assertEqual(len(e.choose_admissions(pending,[],dict(cpu_slots=64,effective_available_memory=64))),64)

    def test_memory_reserved_minus_actual_rss(self):
        pending=[dict(memory_estimate=6),dict(memory_estimate=4)]
        self.assertEqual(e.choose_admissions(pending,[dict(estimate=8,rss=3)],dict(cpu_slots=4,effective_available_memory=10)),[pending[1]])

    def test_no_memory_admission(self):
        self.assertFalse(e.choose_admissions([dict(memory_estimate=11)],[],dict(cpu_slots=64,effective_available_memory=10)))

    def test_existing_workers_not_killed_by_admission(self):
        active=[dict(estimate=20,rss=15)]
        before=deepcopy(active)
        self.assertFalse(e.choose_admissions([dict(memory_estimate=1)],active,dict(cpu_slots=64,effective_available_memory=0)))
        self.assertEqual(active,before)

    def test_pair_scopes_exact_call_counts(self):
        for group,axes in zip(e.GROUPS,(722,66)):
            _,rows=e.expected_tasks(group);caps=c.call_caps(rows)
            self.assertEqual(caps['wrapper_compile'],axes)
            self.assertEqual(caps['control_probe'],2*axes)
            for forbidden in c.FORBIDDEN_CALLS:self.assertEqual(caps[forbidden],0)


if __name__=='__main__':unittest.main()
