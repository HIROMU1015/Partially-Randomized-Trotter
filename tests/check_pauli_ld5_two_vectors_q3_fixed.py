"""LD5 passive source binding and actual sequential entry with artificial risk."""
from copy import deepcopy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
spec=importlib.util.spec_from_file_location('_ld5_two_test',ROOT/'scripts/run_pauli_ld5_two_vectors_q3_fixed.py')
C=importlib.util.module_from_spec(spec);sys.modules[spec.name]=C;spec.loader.exec_module(C)
from check_rpe_pauli_integration import forbid_science
from check_pf_shortlist_q3_caller import artificial,MOCK


class LD5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with forbid_science(),patch.object(C.C,'generate',side_effect=AssertionError('No generation')):
            cls.inputs=C.build_inputs();cls.binding,cls.view=C.metadata(cls.inputs['cases'][0])

    def test_two_fixed_vectors_exact_sums_and_original_contract(self):
        x=self.inputs;old=C.C.document(dict(commit=C.BASE,path=C.L.INPUT))
        self.assertEqual([c['comparison_name'] for c in x['cases']],['FULL_PF__B_PRICE','VP'])
        self.assertEqual(x['maximum_scientific_calls'],2);self.assertEqual(x['allocation_generation_calls'],0)
        self.assertEqual([C.C.rational(c['conditional_cost']) for c in x['cases']],
                         [C.C.rational('3792207339/7'),C.C.rational('6105690627/7')])
        for case,(name,vector) in zip(x['cases'],C.VECTORS):
            self.assertEqual([s['N_X'] for s in case['allocation']['shots']],vector)
            self.assertEqual([s['N_Y'] for s in case['allocation']['shots']],vector)
            self.assertEqual(sum(C.C.rational(t['N_times_C_ref']) for t in case['price_terms']),C.C.rational(case['conditional_cost']))
        for k in ('options','statistical_contract','adopted_evaluation','acceleration'):
            self.assertEqual(x[k],old[k])
        self.assertNotEqual(x['cases'][0]['total_configuration_id'],old['cases'][0]['total_configuration_id'])

    def test_report_internal_fraction_and_threshold_headroom_display(self):
        spec=importlib.util.spec_from_file_location('_ld5_display_test',ROOT/'scripts/summarize_pauli_ld5_two_vectors_q3_fixed.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertEqual(module.display(Fraction(1,8)),module.display(dict(numerator=1,denominator=8)))
        self.assertEqual(module.display(Fraction(-1,8)),'-0.125')
        with self.assertRaises(ValueError):module.display(0.125)

    def test_A_helper_and_raw_normalization_SOURCE_target_preserved(self):
        h=C.C.document(C.A.LD5_HANDOFF);packet=self.view['signal_packet'];origin=packet['origin_signal_packet']
        self.assertEqual(origin['configuration_id'],C.CID)
        self.assertEqual(origin['dictionary_identity'],h['packets'][0]['dictionary_identity'])
        self.assertEqual(self.inputs['dictionary_helper']['commit'],C.A.LD3_HELPER_SOURCE)
        self.assertFalse(packet['target']['ground_state_certified'])
        for row,mean in zip(h['packets'][0]['rounds'],packet['means']):
            rec=row['mean_record']
            for k in ('raw','corrected','reference'):self.assertEqual(mean[k],rec['signal'][k])
            for k in ('B','log_B','attenuation'):self.assertEqual(mean['normalization'][k],rec['signal'][k])
            self.assertEqual(mean['source']['commit'],row['mean_source']['commit'])
            self.assertEqual(mean['implementation_source'],rec['source']);self.assertTrue(rec['scalar_once'])
            self.assertEqual(mean['qualification'],'UNQUALIFIED')
        self.assertTrue(all(packet[k] is None for k in ('sufficient_N','global_RMSE_bound','G','winner')))

    def test_wrong_physical_price_source_and_mixed_LD_rejected(self):
        table=C.C.document(C.PRICE)
        with forbid_science():
            for k,value in [('physical_id','LD3'),('C_ref_exact','-1'),('q',999)]:
                bad=deepcopy(table);bad['rows'][0][k]=value
                with self.assertRaises(ValueError):C.joined_prices(bad,self.binding,self.view)
            bad=deepcopy(self.inputs['cases'][0]);bad['configuration']=C.C.document(dict(commit=C.BASE,path=C.L.INPUT))['cases'][0]['configuration']
            with self.assertRaises(ValueError):C.metadata(bad)
            packet=deepcopy(self.view['signal_packet']['origin_signal_packet']);packet['means'][0]['raw']=[0,0]
            with self.assertRaises(ValueError):C.T.bind_signal(ROOT,self.binding,packet)

    def test_second_runs_after_model_pass_or_model_fail_but_not_gate_failure(self):
        for first in (C.C.PASS,C.C.FAIL):
            seen=[]
            def visit(case,i):seen.append(i);return dict(status=first if i==1 else C.C.PASS,scientific_evaluation_calls=1)
            result=C.run_cases(self.inputs,visit)
            self.assertEqual(seen,[1,2]);self.assertEqual(result['new_scientific_calls'],2)
        result=C.run_cases(self.inputs,lambda c,i:dict(status='UNKNOWN',scientific_evaluation_calls=1))
        self.assertEqual(result['new_scientific_calls'],1);self.assertEqual(len(result['unevaluated']),1)

    def test_actual_two_call_entry_persists_and_observes_both_artificial_jobs(self):
        case,x,b,v=artificial();second=deepcopy(case);second['label']='ARTIFICIAL_SECOND'
        x.update(cases=[case,second],concurrent_configurations=1)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/C.DIRECTORY).mkdir(parents=True);(root/C.INPUT).write_text(json.dumps(x))
            (root/C.CALLER).parent.mkdir();(root/C.CALLER).write_text('ARTIFICIAL_ONLY')
            for n in ('rpe_pauli_signal_intake','rpe_c_interfaces'):
                path=root/'src/trottertracks'/f'{n}.py';path.parent.mkdir(parents=True,exist_ok=True);path.write_text('ARTIFICIAL_ONLY')
            with patch.object(C,'ROOT',root),patch.object(C,'check_source'),patch.object(C,'build_inputs',return_value=x),\
                    patch.object(C.subprocess,'check_output',return_value='ARTIFICIAL_SOURCE'),\
                    patch.object(C.R,'blob',side_effect=lambda sha,path:(root/path).read_bytes()),\
                    patch.object(C,'metadata',return_value=(b,v)),patch.dict(C.C.validate_result.__globals__,ROOT=root):
                C.execute()
            run=root/C.DIRECTORY/'run01';saved=json.loads((run/'RESULT_INDEX.json').read_text())
            self.assertEqual(saved['new_scientific_calls'],2);self.assertFalse(saved['unevaluated'])
            for i,c in enumerate(x['cases'],1):
                folder=run/f'{i:02}'/c['label']
                for file in ('API_RETURN.json','RESULT.json','RESOURCE_FINAL.json','SEGMENTS.jsonl','PERSISTED.json'):
                    self.assertTrue((folder/file).exists(),file)
                self.assertEqual(saved['history'][i-1]['status'],C.C.PASS)


if __name__=='__main__':unittest.main(verbosity=2)
