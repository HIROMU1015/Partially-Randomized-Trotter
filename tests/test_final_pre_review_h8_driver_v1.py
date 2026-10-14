"""Artificial JSON/CSV, protocol mutants, no real reaccounting."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

spec=importlib.util.spec_from_file_location('final_fixtures',Path(__file__).with_name('final_pre_review_fixtures_v1.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


class H8DriverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding,cls.payloads,cls.old_costs=f.synthetic_h8()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='h8-driver-artificial-',dir=os.environ['TMPDIR'])
        self.base=Path(self.tmp.name);self.root=self.base/'source';self.output=self.base/'output'
        self.contract=f.base_contract(f.driver.SCOPE,self.root,self.output,self.binding)
        self.mapping={}
        for name,data in self.payloads.items():
            p=self.base/name;p.write_bytes(data);self.mapping[name]=str(p)
        self.contract.update(input_receipts=[dict(logical_name=n,bytes=len(v),sha256=f.g.sha(v))
            for n,v in self.payloads.items()],expected_old_precision_cases=944,expected_old_map_cases=self.old_costs)
        self.refresh()

    def tearDown(self):self.tmp.cleanup()
    def refresh(self):self.auth=f.authorization(self.contract,self.output)
    def run_driver(self,loader=None):
        return f.driver.run(self.contract,self.auth,contract_sha=f.d0.digest(self.contract),
            authorization_sha=f.d0.digest(self.auth),mapping_loader=loader or (lambda:self.mapping),
            output=self.output,root=self.root,synthetic=True,monitor_factory=f.FakeMonitor)
    def read(self,name):return json.loads((self.output/name).read_text())

    def test_complete_fixture_gate_phase_b_p_and_union(self):
        self.assertEqual(self.run_driver(),0)
        equality=self.read('original_equality_v1.json');self.assertEqual(equality['precision_cases'],944)
        self.assertEqual(equality['cost_cases'],self.old_costs)
        records=self.read('candidate_accounting_v1.json')
        for scope,n in zip(f.driver.SCOPES,(920,260)):self.assertEqual(len(records[scope]),n)
        cid=self.binding['selection_digest']  # not reused as a scientific measurement.
        self.assertTrue(cid)
        union=f.d0.strict_json(self.payloads['selection_before_confirmation.json'])['selected_ids']
        for scope,n in zip(f.driver.SCOPES,(8,32)):
            rows=[r for r in records[scope] if r['cell_id']==union[0]]
            self.assertTrue(all(r['sample_count']==n for r in rows))
        coverage=self.read('confirmation_coverage_v1.json')
        self.assertTrue(any(v['status']=='OUTSIDE_ORIGINAL_UNION6' for v in coverage.values()))
        sensitivity=self.read('preparation_sensitivity_v1.json')
        self.assertEqual(sum(len(v) for v in sensitivity.values()),120)
        self.assertTrue(all(c['lower_envelope']['domain']=='P>=0' for v in sensitivity.values() for c in v.values()))
        rare=self.read('rare_cost_status_v1.json')['layers']['confirmation'][-1]
        self.assertEqual(rare['status'],'UNOBSERVED');self.assertIsNone(rare['population_layer_cost'])
        terminal=self.read('terminal_v1.json');self.assertEqual(terminal['status'],'COMPLETE_MANDATORY_STOP')
        self.assertIsNone(terminal['native_exit_code']);self.assertFalse(terminal['automatic_retry'])

    def test_absent_unapproved_authorization_never_loads_inputs(self):
        loader=Mock(side_effect=AssertionError('INPUT MUST NOT BE READ'))
        for auth in ({},dict(self.auth,approved_by_user=False),dict(self.auth,runtime_authorization=False)):
            self.auth=auth
            with self.assertRaisesRegex(f.g.Stop,'EXECUTION_AUTHORIZATION'):self.run_driver(loader)
        self.assertFalse(loader.called);self.assertFalse(self.output.exists())

    def test_draft_contract_never_loads_inputs(self):
        self.contract['approved']=False;self.refresh();loader=Mock()
        with self.assertRaises(f.g.Stop):self.run_driver(loader)
        self.assertFalse(loader.called)

    def test_contract_tamper_authorization_digest_refused(self):
        self.contract['carry']['actual']=1;loader=Mock()
        with self.assertRaises(f.g.Stop):self.run_driver(loader)
        self.assertFalse(loader.called)

    def test_source_tamper_terminal_saved_before_input_load(self):
        (self.root/'synthetic_source.txt').write_text('TAMPERED')
        loader=Mock();self.assertEqual(self.run_driver(loader),2);self.assertFalse(loader.called)
        self.assertIn('SOURCE_BYTES_SHA',self.read('terminal_v1.json')['first_stop']['reason'])

    def test_input_sha_tamper_stops_before_accounting(self):
        Path(self.mapping['signal_summary.json']).write_bytes(b'{}')
        with patch.object(f.driver.d0,'accounting',side_effect=AssertionError('NO NUMERIC CALL')) as calculate:
            self.assertEqual(self.run_driver(),2)
        self.assertFalse(calculate.called)
        self.assertIn('INPUT_BYTES_SHA',self.read('terminal_v1.json')['first_stop']['reason'])

    def test_complete_old_mismatch_never_extends(self):
        name='confirmation_resource_map.json';old=f.d0.strict_json(self.payloads[name]);old['rows'][0]['G_point']+=1
        raw=json.dumps(old).encode();Path(self.mapping[name]).write_bytes(raw)
        for r in self.contract['input_receipts']:
            if r['logical_name']==name:r.update(bytes=len(raw),sha256=f.g.sha(raw))
        self.refresh();original=f.driver.d0.accounting
        def gated(record,epsilon,factor):
            self.assertNotEqual(epsilon,.0005);return original(record,epsilon,factor)
        with patch.object(f.driver.d0,'accounting',side_effect=gated):self.assertEqual(self.run_driver(),2)
        self.assertFalse((self.output/'candidate_accounting_v1.json').exists())
        self.assertIn('ORIGINAL_MAP_VALUE_MISMATCH',self.read('terminal_v1.json')['first_stop']['reason'])

    def test_axis_statistics_mismatch_is_not_hidden_by_same_total(self):
        name='exploration_resource_map.json';old=f.d0.strict_json(self.payloads[name]);old['rows'][0]['axes']['cosine']['max']+=1
        raw=json.dumps(old).encode();Path(self.mapping[name]).write_bytes(raw)
        for r in self.contract['input_receipts']:
            if r['logical_name']==name:r.update(bytes=len(raw),sha256=f.g.sha(raw))
        self.refresh();self.assertEqual(self.run_driver(),2)
        self.assertIn('AXIS_STATISTICS_MISMATCH',self.read('terminal_v1.json')['first_stop']['reason'])

    def test_missing_eligible_cost_refused(self):
        signals=f.d0.strict_json(self.payloads['signal_summary.json'])['signals']
        cfg=f.driver.specs(self.binding,['B2_p7_q2_2nd_r4_K2']) [f.driver.SCOPES[0]]
        records=f.d0.analyze_scope(signals,[],cfg,f.a.ORIGINAL_EPSILONS,f.a.FACTORS)
        with self.assertRaisesRegex(f.g.Stop,'MISSING_ELIGIBLE_COST'):
            f.driver.complete_old_map(records,{'rows':[]},signals,[],cfg,lambda:None)

    def test_selection_tamper_recomputed_union_refused(self):
        records=[dict(cell_id=f'x{i}',method='B2',status='DERIVED_SAVED_SCALAR',epsilon=e,factor='1.0',
            metrics={'RZ':{'G_point':i}}) for e in f.a.ORIGINAL_EPSILONS for i in range(3)]
        with self.assertRaisesRegex(f.g.Stop,'TOP2_RZ_SELECTION'):
            f.driver.verify_old_selection(records,{'selected_ids':['x0','x2']})

    def test_p_envelope_exact_arithmetic_not_grid(self):
        lines=[dict(cell_id='A',intercept=0,slope=10),dict(cell_id='B',intercept=6,slope=4)]
        envelope=f.d0.lower_envelope(lines)
        self.assertEqual(envelope['segments'][0]['P_end'],{'numerator':'1','denominator':'1'})
        self.assertEqual(envelope['vertices'][-1]['tied_cells'],['A','B'])

    def test_one_shot_reexecution_refused_without_numeric_retry(self):
        self.contract['source_hashes']['synthetic_source.txt']='0'*64;self.refresh()
        self.assertEqual(self.run_driver(),2)
        with patch.object(f.driver.d0,'accounting',side_effect=AssertionError('NO RETRY')):
            with self.assertRaisesRegex(f.g.Stop,'ONE_SHOT_ALREADY_USED'):self.run_driver()

    def test_seed_source_axis_pair_identity_mutants(self):
        rows=f.d0.parse_wrappers(self.payloads['wrapper_metrics.csv']);signals=f.d0.strict_json(self.payloads['signal_summary.json'])['signals']
        selection=f.d0.strict_json(self.payloads['selection_before_confirmation.json'])
        for field,value in [('seed',rows[0]['seed']+1),('source','0'*40),('axis','Z')]:
            changed=copy.deepcopy(rows);changed[0][field]=value
            with self.assertRaises(ValueError):f.a.validate_inputs(signals,changed,selection,self.binding)

    def test_native_receipt_requires_real_same_parent_wait_and_absent_identity(self):
        self.contract['source_hashes']['synthetic_source.txt']='0'*64;self.refresh();self.assertEqual(self.run_driver(),2)
        kwargs=dict(contract_sha=f.d0.digest(self.contract),authorization_sha=f.d0.digest(self.auth),output=self.output)
        with self.assertRaisesRegex(f.g.Stop,'STILL_PRESENT'):
            f.driver.record_external_exit(self.contract,self.auth,wait_status=2,**kwargs)
        ticket=f.d0.strict_json(f.g.read_bytes(f.g.marker_path(self.output)))
        def exited(pid=None):
            return ticket['parent_identity'] if pid==os.getppid() else None
        with patch.object(f.driver.g,'identity',side_effect=exited):
            f.driver.record_external_exit(self.contract,self.auth,wait_status=2,**kwargs)
            with self.assertRaises(FileExistsError):f.driver.record_external_exit(self.contract,self.auth,wait_status=2,**kwargs)
        receipt=self.read('native_exit_v1.json');self.assertEqual(receipt['shell_wait_status'],2)
        self.assertIsNone(receipt['exact_exit_timestamp'])


if __name__=='__main__':unittest.main()
