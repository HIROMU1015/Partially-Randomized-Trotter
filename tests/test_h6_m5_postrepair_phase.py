"""Synthetic-only checks for log completeness and causal claim boundaries."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import inspect_h6_m5_postrepair_phase as tool


def rows():
    return [dict(count=i,pass_name=name,pass_seconds=0.01,global_phase_binary64_hex=phase.hex(),
        gate_counts={'rz':3,'sx':2,'measure':1},unexpected_names=[],
        scalar_observation='BASIS_NAMES_ONLY_NOT_GATE_IDENTITY_OR_CAUSAL_PROOF')
        for i,(name,phase) in enumerate([('BasisTranslator',1.7),('Optimize1qGatesDecomposition',1.8),('InverseCancellation',1.8)])]


class LogTests(unittest.TestCase):
    def test_phase_change_not_error_attribution(self):
        r=tool.summarize_callbacks(rows(),1.7.hex())
        self.assertEqual(r['phase_bit_change_pass_counts'],{'Optimize1qGatesDecomposition':1})
        self.assertEqual(r['first_phase_bit_change'],1)
        self.assertIsNone(r['first_numerical_error_pass'])
        self.assertFalse(r['causal_attribution_established'])

    def test_unchanged_phase_is_not_numeric_PASS(self):
        data=rows()
        for r in data:r['global_phase_binary64_hex']=1.7.hex()
        r=tool.summarize_callbacks(data,1.7.hex())
        self.assertEqual(r['phase_bit_change_records'],0)
        self.assertIsNone(r['first_phase_bit_change'])
        self.assertNotIn('PASS',r)
        self.assertIsNone(r['first_numerical_error_pass'])

    def test_missing_duplicate_reordered_or_boolean_indices_rejected(self):
        for indices in ([0,2],[0,1,1],[1,0,2],[False,1,2]):
            data=rows()[:len(indices)]
            for r,i in zip(data,indices):r['count']=i
            with self.subTest(indices=indices),self.assertRaises(ValueError):
                tool.summarize_callbacks(data,1.7.hex())

    def test_nonfinite_and_noncanonical_phase_rejected(self):
        for phase in ('nan','inf','-inf','0x01.b333333333333p+0'):
            data=rows();data[1]['global_phase_binary64_hex']=phase
            with self.subTest(phase=phase),self.assertRaises(ValueError):
                tool.summarize_callbacks(data,1.7.hex())

    def test_empty_log_rejected(self):
        with self.assertRaises(ValueError):tool.summarize_callbacks([],1.7.hex())

    def test_unknown_basis_and_false_causal_labels_rejected(self):
        for field,value in [('gate_counts',{'unknown':1}),('unexpected_names',['unknown']),('scalar_observation','PROVED_CAUSE')]:
            data=rows();data[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):tool.summarize_callbacks(data,1.7.hex())

    def test_invalid_count_or_time_rejected(self):
        for field,value in [('gate_counts',{'rz':True}),('gate_counts',{'rz':-1}),('pass_seconds',float('nan')),('pass_seconds',-1)]:
            data=rows();data[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):tool.summarize_callbacks(data,1.7.hex())

    def test_input_objects_not_mutated(self):
        data=rows();old=copy.deepcopy(data)
        result=tool.summarize_callbacks(data,1.7.hex());result['transitions'][0]['gate_counts']['rz']=999
        self.assertEqual(data,old)

    def test_artificial_probe_guard(self):
        with self.assertRaises(ValueError):tool.artificial_scalar_probe()

    def test_installed_scalar_loss_and_control(self):
        r=tool.artificial_scalar_probe(artificial=True)
        self.assertTrue(r['exact_sum_matches_original_base'])
        self.assertEqual(r['installed_empty_DAG_phase'],r['sequential_model_phase'])
        self.assertNotEqual(r['signed_scalar_error'],0.)
        self.assertEqual(r['single_assignment_phase'],r['initial_phase'])
        self.assertFalse(r['H6_error_magnitude_explained'])

    def test_default_CLI_has_no_scientific_execution(self):
        command=[sys.executable,'-B',str(Path(tool.__file__))]
        r=subprocess.run(command,capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(r.stdout)['status'],'STATIC_SCOPE_ONLY')

    def test_bound_manifest_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source='a'*40;plan='b'*64;inputs={}
            fixture={'callback_log':rows(),
                'lowering_receipt':dict(SOURCE=source,plan_sha256=plan,stored_global_phase_binary64_hex=1.7.hex()),
                'compiled_receipt':dict(SOURCE=source,plan_sha256=plan,schema=dict(global_phase=1.8,gate_counts=rows()[-1]['gate_counts'])),
                'completion_witness':dict(SOURCE=source,plan_sha256=plan,status='STOP'),
                'worker_STOP':dict(SOURCE=source,plan_sha256=plan,reason='NUMERIC_DISAGREEMENT:compiled_vs_lowered')}
            for role,data in fixture.items():
                path=root/(role+('.jsonl' if role=='callback_log' else '.json'))
                path.write_text('\n'.join(json.dumps(r) for r in data) if role=='callback_log' else json.dumps(data))
                inputs[role]=dict(path=str(path),bytes=path.stat().st_size,sha256=tool.sha(path))
            manifest=dict(origin_SOURCE=source,origin_plan_sha256=plan,inputs=inputs)
            self.assertEqual(tool.inspect_manifest(manifest)['callback_records'],3)
            Path(inputs['callback_log']['path']).write_text('{}\n')
            with self.assertRaisesRegex(ValueError,'INPUT_SHA'):tool.inspect_manifest(manifest)


if __name__=='__main__':unittest.main()
