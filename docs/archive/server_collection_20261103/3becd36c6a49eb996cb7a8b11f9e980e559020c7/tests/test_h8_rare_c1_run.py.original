"""Artificial driver checks: symbolic providers only, no scientific imports."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from h8_rare_c1_run import (scope_gate,execute_batches,runtime_probe,
    require_safe_parent,library_record)
from h8_rare_c05.accounting import Budget,Permit,atomic,read
from h8_rare_c05.adapter import Adapter
from h8_rare_c05.sampling import register_streams,digest
from h8_rare_c1_contract import task_plan,budget
from test_h8_rare_c05 import ToyProvider,contract,ToyLaw,log_receipt


class ToyChecked(ToyProvider):
    def validate(self,evolution,requests,directory):
        atomic(Path(directory)/'control.json',dict(status='PASS',tolerance=1e-8,
            max_error=0.,control_probes=4,independent_event_oracles=1,
            full_Fock_scalar_once=True,reference_rotated=False))


class ToyAdmission:
    def observe(self,rows):pass
    def capacity(self):return dict(cpu=4,memory=10**9,io=4)
    def demand(self,task,c):return dict(cpu=2,memory=2**20,io=1,axis_width=2)


def prepare(root,q=2,fail=False):
    candidate='B2_p8_q2_2nd_r4_K4' if q==2 else 'B2_p7_q16_2nd_r4_K4'
    tasks=task_plan(candidate,q,43);c=contract(tasks[0]);c.update(condition='H8_0.05' if q==2 else 'H8_0.0005',
        q=q,r=4,K=4,n_by_stratum=dict(S2_SINGLE=43,S2_MULTI=None,S4_PRESENT=None),
        frozen_scientific_budget=budget(q,43),cost_cache_enabled=False,production_authorized=False,
        shots=dict(cosine=3,sine=5),pilot_numeric_accounting=True,law_compatibility_probe=True,
        canonical_order_probability_reference=['.5','.3','.2'])
    reg=register_streams(tasks,'MOCK_SOURCE','MOCK_BINDING','MOCK_EPOCH')
    c['private_task_registry_digest']=digest(dict(tasks=tasks,streams=reg))
    b=Budget(root/'budget',c['frozen_scientific_budget'])
    b.take('saved_preparation_decode',1,'saved_preparation_decode')
    b.take('prefix_reconstruction',1,'prefix_reconstruction')
    provider=ToyChecked('sine' if fail else None)
    a=Adapter(provider,Permit('MOCK','',c['condition']),c,b);a.initialize_law(0)
    return a,tasks,reg


class DriverTests(unittest.TestCase):
    def test_full_43_both_q_pair_accounting_and_unknowns(self):
        for q in (2,16):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);a,ts,reg=prepare(root,q)
                summary=execute_batches(a,ts,reg,root,ToyAdmission())
                self.assertEqual(summary['groups']['S2_SINGLE']['n'],43)
                self.assertEqual(summary['groups']['S2_SINGLE']['mean'],'88')
                self.assertIsNone(summary['groups']['S2_MULTI']['mean'])
                self.assertIsNone(summary['groups']['S4_PRESENT']['variance'])
                self.assertEqual(a.budget.used(),budget(q,43))
                receipts=read(root/'batch_receipts.json')
                self.assertEqual([len(r['workers']) for r in receipts],[1,42])
                for r in receipts:log_receipt(r)
                self.assertTrue(read(root/'completion.json')['all_owned_reaped'])

    def test_failed_representative_never_submits_remaining(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a,ts,reg=prepare(root,fail=True)
            with self.assertRaises(ValueError):execute_batches(a,ts,reg,root,ToyAdmission())
            self.assertFalse((root/'remaining').exists())
            self.assertEqual(a.budget.used()['trajectory'],1)
            receipt=read(root/'representative/native_receipt.json');log_receipt(receipt)
            self.assertTrue(receipt['all_reaped'])
            self.assertFalse((root/'pilot_summary.json').exists())

    def test_wrong_budget_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as d:
            a,ts,reg=prepare(Path(d));a.c['frozen_scientific_budget']['wrapper_compile']=87
            with self.assertRaisesRegex(ValueError,'FIXED_CALL_BUDGET'):scope_gate(a.c,ts,reg)

    def test_changed_registry_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a,ts,reg=prepare(Path(d));reg[ts[0]['id']]['order']+=1
            with self.assertRaisesRegex(ValueError,'PRIVATE_REGISTRY_SHA'):scope_gate(a.c,ts,reg)

    def test_runtime_failure_no_grant_or_science(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'probe'
            with patch('h8_rare_c1_run.bind_imports',side_effect=ValueError('UNSAFE_MULTITHREADED_FORK_PARENT')):
                self.assertEqual(runtime_probe({},out),2)
            x=read(out/'probe.json');self.assertEqual(x['science_calls'],0)
            self.assertFalse(x['valid_grant_consumed'])

    def test_multithread_parent_is_stop(self):
        with patch('h8_rare_c1_run.single_thread',return_value=False):
            with self.assertRaisesRegex(ValueError,'UNSAFE_MULTITHREADED'):require_safe_parent()

    def test_existing_library_metadata_only(self):
        key=NS(name='mock',num_qubits=1)
        circ=NS(parameters=[],global_phase=0,num_qubits=1,num_clbits=0,data=[])
        lib=NS(keys=lambda:[key],_get_equivalences=lambda k:[NS(params=[],circuit=circ)])
        x=library_record(lib);self.assertEqual(x['keys'][0]['name'],'mock')
        self.assertEqual(x['keys'][0]['entries'][0]['instructions'],[])

    def test_no_real_imports(self):
        self.assertFalse(any(n=='numpy' or n.startswith('qiskit') for n in sys.modules))


if __name__=='__main__':unittest.main()
