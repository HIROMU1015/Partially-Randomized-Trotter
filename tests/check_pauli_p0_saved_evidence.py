"""Saved JSON consistency only: no H arrays, operator reconstruction or new norm.

Copies existing scalar diagnostics; coefficient min/max are inventory values,
not an estimate of dropped coefficients or a new operator-error calculation.
"""
from pathlib import Path
import json
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
SOURCE='9fa9525b3'
DIRECTORY='docs/research/b2_pauli_split_p0_20261116/'
OUTPUT=ROOT/'docs/research/pauli_p0_gate_diagnosis_20261116/SAVED_EVIDENCE.json'


def read(path):
    return json.loads(subprocess.check_output(['git','show',SOURCE+':'+DIRECTORY+path],cwd=ROOT))


def inventory():
    rows=[];binding=None
    for ld in (4,6,7,8,9):
        base='LD%d/'%ld
        result=read(base+'result.json');dictionary=read(base+'LD%d_DICTIONARY.json'%ld)
        if binding is None:binding=result['binding']
        assert result['binding']==dictionary['binding']==binding
        jw=read(base+'LD%d_JW_COEFFICIENTS.json'%ld)
        record=next(r for r in result['prefixes'] if r['prefix']==ld)
        coefficients={}
        for term in jw:
            label=['I']*8
            for q,axis in term['term']:label[7-q]=axis
            assert term['imag']==0
            coefficients[''.join(label)]=term['real']
        identity=coefficients.pop('I'*8,0.)
        actual={t['label_qiskit']:t['coefficient'] for t in dictionary['terms']}
        assert len(actual)==len(dictionary['terms'])
        rows.append(dict(L_D=ld,source=dict(commit=SOURCE,path=DIRECTORY+base),
            scientific_source=result['scientific_source'],status=result['status'],
            fixed_P0_source=result['fixed_P0_source'],stage1_extension=result['stage1_extension'],
            record=record,checks=[c for c in result['checks'] if c['name'].startswith('LD%d_'%ld)],
            control_formally_completed=result.get('control',{}).get('PASS'),
            coefficient_policy_saved=result.get('coefficient_policy'),runtime_saved=result.get('runtime'),
            operator_frobenius_norms_saved=None,
            coefficient_inventory=dict(JW_terms=len(jw),dictionary_terms=len(actual),
                JW_to_dictionary_coefficients_identical=coefficients==actual,
                JW_to_dictionary_identity_identical=identity==dictionary['pauli_identity'],
                min_saved_abs_coefficient=min(abs(c) for c in actual.values()),
                max_saved_abs_coefficient=max(abs(c) for c in actual.values()),
                saved_terms_below_1e_minus_8=sum(0<abs(c)<1e-8 for c in actual.values())),
            collector_metadata=dictionary['collector_metadata'],error=result.get('error'),
            automatic_retry=result['automatic_retry'],mandatory_stop=result['mandatory_stop']))
    return dict(schema='P0_SAVED_JSON_INSPECTION_V1',
        source=subprocess.check_output(['git','rev-parse',SOURCE],cwd=ROOT,text=True).strip(),
        binding=binding,
        scope='JSON_RECORD_AND_COEFFICIENT_IDENTITY_INSPECTION_ONLY',
        H_array_loads=0,new_operator_norms=0,new_P0_runs=0,rows=rows)


class SavedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence=inventory()

    def test_inspection_matches_published_json_report(self):
        self.assertEqual(self.evidence,json.loads(OUTPUT.read_text()))

    def test_raw_JW_is_preserved_by_dictionary_for_every_split(self):
        for row in self.evidence['rows']:
            self.assertTrue(row['coefficient_inventory']['JW_to_dictionary_coefficients_identical'])
            self.assertTrue(row['coefficient_inventory']['JW_to_dictionary_identity_identical'])
            self.assertEqual(row['collector_metadata']['coefficient_cutoff'],0.)

    def test_gate_failure_and_late_unexecuted_checks_are_not_relabelled(self):
        for row in self.evidence['rows']:
            ld=row['L_D'];checks=row['checks'];last=checks[-1]
            self.assertTrue(row['mandatory_stop']);self.assertFalse(row['automatic_retry'])
            self.assertEqual(row['fixed_P0_source'],'fd111f989b9cd2f052d7189e4b7b6fab10aa1a9f')
            self.assertTrue(all(c['threshold']==1e-9 for c in checks))
            if ld in (4,6):
                self.assertTrue(all(c['PASS'] for c in checks))
                self.assertTrue(row['control_formally_completed'])
            else:
                self.assertEqual(last['name'],'LD%d_Pauli_tail_full_fro'%ld)
                self.assertFalse(last['PASS']);self.assertTrue(all(c['PASS'] for c in checks[:-1]))
                self.assertIsNone(row['control_formally_completed'])
                self.assertEqual(last['value'],row['record']['identity_metrics']['Pauli_tail_full_fro'])
                self.assertEqual(row['error']['message'],'P0_NUMERICAL_GATE: '+last['name'])

    def test_no_new_comparator_norm_or_relative_error_is_invented(self):
        for row in self.evidence['rows']:
            self.assertIsNone(row['operator_frobenius_norms_saved'])
            self.assertEqual(set(row['record']['tail']),
                             {'hermiticity_fro','sector_escape_fro','sector_return_fro'})


if __name__=='__main__':unittest.main(verbosity=2)
