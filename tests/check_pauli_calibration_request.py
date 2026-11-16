"""Saved LD3/LD5 inputs and an artificial unregistered split through real metadata entry.

No real H array, path, build/compile, fit, model evaluation or Q3 is invoked.
"""
from contextlib import ExitStack
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import prepare_pauli_calibration as RUN
from trottertracks import pauli_calibration_request as C
from trottertracks import pauli_candidate_preparation as A

E = A.E
PRICES = {3: dict(commit='f25a05f8a92aab43cfbe30fc80b974054156aac1',
                  path='docs/research/a_pauli_ld3_cal_price_20261116/PRICE_26AXIS.json'),
          5: dict(commit='fb0d60b0a1e7c31aee73b2b31f802dda7b2f0125',
                  path='docs/research/a_pauli_ld5_cal_price_20261116/PRICE_26AXIS.json')}


def saved_price_projection(root, loc):
    """Copy exact saved values; this is not a newly evaluated PriceRules quote."""
    old = E.read_ref(root, loc); rows = []
    for i, r in enumerate(old['rows']):
        rows.append(dict(m=r['m'], axis=r['axis'], physical_id=r['physical_id'], q=r['q'], R=r['R'],
            unit_price_RZ=deepcopy(r['C_ref_exact']), classification='MODEL_REFERENCE',
            source=dict(loc, selector=['rows', i, 'C_ref_exact']), qualified_E_C=False, CI=None))
    return dict(schema='PR_PRICE_APPLICABILITY_V1', configuration_id=old['configuration_id'], policy='MODEL',
                method='PR_PAULI', L_D=3 if 'ld3' in loc['path'] else 5, rows=rows, complete=True,
                fixture_role='SAVED_PRICE_CARRIER_PROJECTION_NO_NEW_QUOTE', source=loc, Q3_executed=False)


def missing(config):
    return dict(schema='PR_PRICE_APPLICABILITY_V1', configuration_id=config['id'], policy='MODEL',
        method='PR_PAULI', L_D=config['spec']['L_D'], complete=False, fixture_role='ARTIFICIAL_NEEDS_NO_QUOTE',
        rows=[dict(m=r['m'], axis=a, physical_id=r['candidate']['id'], q=r['q_PF'], R=r['R_outer'],
            unit_price_RZ=None, classification='NEEDS_PRICE_EVIDENCE', needs=dict(
                reasons=['No matching physical family'], missing_axes=[a],
                calibration=dict(rule='A + H*q + E*qR', coefficients=['A','H','E'],
                                 reuse='Same-candidate matching receipts only'), execution_authorized=False))
            for r in config['rounds'] for a in ('X','Y')])


def prepared(config):
    return dict(configuration=config, configuration_id=config['id'], configuration_source=None,
        dictionary_source=config['spec']['split']['tail_dictionary']['source'],
        price_handoff=dict(request=dict(implementation=None)),
        schedule=[dict(m=r['m'], q_RPE=r['q_RPE'], T=r['T'], q_PF=r['q_PF'], actual_delta_pf=r['actual_delta_pf'],
            R_outer=r['R_outer'], physical_id=r['candidate']['id'], structure=r['structure'], mapping=r['mapping'],
            law_binding=r['law_binding']) for r in config['rounds']])


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        f=ROOT/'docs/research/pauli_candidate_preparation_20261116/examples/RESPONSE_SAVED_AND_PENDING.json.gz'
        cls.preps={r['L_D']:r for r in json.loads(gzip.decompress(f.read_bytes()))['items'] if r.get('configuration')}
        cls.d=C.driver(str(ROOT)); cls.B1=cls.d.D3.legacy(); cls.P=cls.B1.B0.connection()

    def no_science(self):
        stack=ExitStack()
        for obj, name in ((np,'load'),(E.p0,'load_saved'),(self.B1,'acquire_path'),
                          (self.d,'input_preflight'),(self.d,'worker'),(self.d,'execute')):
            stack.enter_context(patch.object(obj,name,side_effect=AssertionError('No H4 scientific leaf')))
        return stack

    def entry(self, prep, quote, tmp, inspect=False):
        a,b,out=(Path(tmp)/x for x in ('A.json','D2.json','OUT.json'))
        a.write_text(json.dumps(prep)); b.write_text(json.dumps(quote))
        args=['--preparation',str(a),'--price-result',str(b),'--output',str(out)]
        if inspect:args+=['--inspect-saved-cell']
        with patch('sys.stdout',new=io.StringIO()):result=RUN.main(args)
        self.assertEqual(result,json.loads(out.read_text()))
        return result

    def test_saved_prices_are_copied_without_calibration_or_quote(self):
        for p in (3,5):
            quote=saved_price_projection(ROOT,PRICES[p])
            with self.no_science(), tempfile.TemporaryDirectory() as tmp:
                result=self.entry(self.preps[p],quote,tmp)
            self.assertEqual(result['status'],'REUSE_SUPPLIED_PRICES_NO_CALIBRATION',result.get('needs'))
            self.assertEqual(result['reused_prices'],quote['rows'])
            self.assertIsNone(result['driver_request'])
            self.assertFalse(result['supplied_D2_response_recomputed'])

    def test_LD3_LD5_reproduce_published_calibration_scientific_inputs(self):
        for p,loc in zip((3,5),C.SAVED_BINDINGS):
            quote=saved_price_projection(ROOT,PRICES[p]); old=E.read_ref(ROOT,loc)
            with self.no_science(), tempfile.TemporaryDirectory() as tmp:
                result=self.entry(self.preps[p],quote,tmp,inspect=True)
                reqfile=Path(result['driver_request_file'])
                with patch.object(sys,'argv',['cal','--request',str(reqfile),'--metadata']), \
                     patch('sys.stdout',new=io.StringIO()) as output:
                    self.d.main()
                actual=json.loads(output.getvalue())
            self.assertEqual(result['status'],'REUSE_SAVED_CALIBRATION',result.get('needs'))
            for key in ('request','dictionary_identity','step_seeds','scalar','operators','cohort','path_indices',
                        'n_paths','axes','compile_maximum','m12_canonical','configuration'):
                self.assertEqual(actual[key],old[key],(p,key))
            self.assertTrue(result['driver_request']['driver_spec']['reuse_only'])
            self.assertIsNone(actual['request']['m'])

    def test_partial_price_reuse_and_existing_calibration_do_not_launch_new_paths(self):
        quote=saved_price_projection(ROOT,PRICES[3])
        for i in (2,3):
            quote['rows'][i].update(unit_price_RZ=None,classification='NEEDS_PRICE_EVIDENCE',
                                   needs=missing(self.preps[3]['configuration'])['rows'][i]['needs'])
        quote['complete']=False
        with self.no_science():result=C.prepare(ROOT,self.preps[3],quote)
        self.assertEqual(result['status'],'REUSE_SAVED_CALIBRATION',result.get('needs'))
        self.assertEqual(len(result['reused_prices']),24)
        self.assertEqual(len(result['missing_prices']),2)
        self.assertEqual(result['path_calls'],0)
        with self.assertRaisesRegex(ValueError,'reuse-only'):
            self.d.select_job_spec(dict(prefix=3,spec=result['driver_request']['driver_spec']))

    def test_out_of_scope_formula_actual_step_K_or_contract_returns_needs(self):
        base=self.preps[3]['configuration']; b=E.p1.b_api(ROOT)
        for key,value,reason in [('formula','4th(m5_best)','FORMULA_OR_K'),
                                 ('finite_K',4,'FORMULA_OR_K'),
                                 ('delta_pf',A.encoded(.05),'SAME_STEP')]:
            s=deepcopy(base['spec']);s[key]=value;cfg=b.generate(s)
            with self.no_science():result=C.prepare(ROOT,prepared(cfg),missing(cfg))
            self.assertEqual(result['status'],'NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION')
            self.assertTrue(any(reason in r for r in result['needs']))
            self.assertIsNone(result['driver_request'])
        prep=deepcopy(self.preps[3]);prep['price_handoff']['request']['implementation']=dict(lowering_id='OTHER')
        with self.no_science():result=C.prepare(ROOT,prep,missing(base))
        self.assertEqual(result['status'],'NEEDS_IMPLEMENTATION_CONTRACT')
        with self.no_science():result=C.prepare(ROOT,prep,saved_price_projection(ROOT,PRICES[3]))
        self.assertEqual(result['status'],'NEEDS_IMPLEMENTATION_CONTRACT')
        self.assertEqual(result['reused_prices'],[])

    def test_identity_schedule_quote_and_missing_source_are_not_substituted(self):
        quote=saved_price_projection(ROOT,PRICES[3])
        for kind in ('configuration_id','axis','physical_id','q'):
            q=deepcopy(quote)
            if kind=='configuration_id':q[kind]='OTHER'
            elif kind=='axis':q['rows'][0]['axis']='Y'
            elif kind=='physical_id':q['rows'][0][kind]='OTHER'
            else:q['rows'][0][kind]+=1
            with self.no_science():r=C.prepare(ROOT,self.preps[3],q)
            self.assertEqual(r['status'],'NEEDS_INPUT_OR_FIXED_IMPLEMENTATION_SOURCE')
        prep=deepcopy(self.preps[3]);prep['schedule'][0]['R_outer']+=1
        with self.no_science():r=C.prepare(ROOT,prep,quote)
        self.assertIsNone(r['driver_request'])
        q=deepcopy(quote);q['rows'][0]['classification']='DIRECT_FINITE_COHORT'
        with self.no_science():r=C.prepare(ROOT,self.preps[3],q)
        self.assertEqual(r['status'],'NEEDS_INPUT_OR_FIXED_IMPLEMENTATION_SOURCE')
        with self.no_science(), patch.object(E,'dictionary_item',side_effect=FileNotFoundError('candidate P0 SOURCE')):
            r=C.prepare(ROOT,self.preps[3],missing(self.preps[3]['configuration']))
        self.assertEqual(r['status'],'NEEDS_INPUT_OR_FIXED_IMPLEMENTATION_SOURCE')
        self.assertIn('candidate P0 SOURCE',r['needs'][0])
        self.assertIsNone(r['driver_request'])

    def test_unregistered_artificial_split_passes_real_driver_and_worker_input_checks(self):
        item=deepcopy(E.dictionary_item(ROOT,self.preps[3]['configuration']))
        item['payload'].update(prefix=4,lambda_P=.25,saved_DF_scalar=.125,
                               terms=[dict(label_qiskit='ZIIIIIII',coefficient=.25)])
        item['dictionary_sha256']=E.L.canonical_sha(item['payload'])
        item['source']=dict(commit='ARTIFICIAL',path='toy/LD4_DICTIONARY.json')
        item['P0_result_source']=dict(commit='ARTIFICIAL',path='toy/result.json')
        item['P0_metrics']=dict(artificial_fixture=True)
        td=E.L.dictionary_identity(item);s=deepcopy(self.preps[3]['configuration']['spec'])
        s['L_D']=4;s['split'].update(L_D=4,tail_dictionary=td,lambda_R=A.encoded(.25),
                                   lambda_source=dict(item['source'],selector=['lambda_P']),scalar=.125)
        s['law']['tail_dictionary']=td;s['law']['source']['dictionary']=item['source']
        cfg=E.p1.b_api(ROOT).generate(s);prep=prepared(cfg)
        orig_blob=E.p0.blob;orig_load=self.P.load_dictionary
        def blob(root,commit,path):
            return b'ARTIFICIAL_OPERATORS_NOT_LOADED' if commit=='ARTIFICIAL' else orig_blob(root,commit,path)
        def load(root,identity):
            return deepcopy(item['payload']) if identity==td else orig_load(root,identity)
        with self.no_science(), patch.object(E,'dictionary_item',return_value=item), \
             patch.object(E.p0,'blob',side_effect=blob), patch.object(self.P,'load_dictionary',side_effect=load), \
             tempfile.TemporaryDirectory() as tmp:
            result=self.entry(prep,missing(cfg),tmp)
            self.assertEqual(result['status'],'CALIBRATION_REQUEST_READY_NOT_EXECUTED',result.get('needs'))
            reqfile=Path(result['driver_request_file'])
            with patch.object(sys,'argv',['cal','--request',str(reqfile),'--metadata']), \
                 patch('sys.stdout',new=io.StringIO()) as output:
                self.d.main()
            actual=json.loads(output.getvalue())
            spec=self.d.load_request_spec(reqfile)
            selected=self.d.select_job_spec(dict(prefix=4,spec=spec,binding=actual))
            self.assertEqual(selected['prefix'],4)
            self.assertNotIn(4,self.d.SPECS)
            self.assertEqual(actual['request']['lambda_P'],.25)
            self.assertEqual(actual['request']['R'],cfg['rounds'][12]['R_outer'])
            self.assertEqual(actual['request']['physical_cell']['profile'],cfg['spec']['profile'])
            self.assertEqual(actual['operators']['ld4']['commit'],'ARTIFICIAL')
            self.assertEqual(result['candidate_implementation_binding']['qualification'],
                             'SOURCE_BOUND_RECIPE_PENDING_NUMERICAL_INPUT_PREFLIGHT')
            self.assertEqual(len(actual['step_seeds']),2)
            with self.assertRaisesRegex(ValueError,'prefix differs'):
                self.d.select_job_spec(dict(prefix=5,spec=spec,binding=actual))
            with self.assertRaisesRegex(ValueError,'binding prefix differs'):
                self.d.select_job_spec(dict(prefix=4,spec=spec,binding=E.read_ref(ROOT,C.SAVED_BINDINGS[0])))
            bad=deepcopy(spec);bad['cell']['R']+=1
            with self.assertRaises(ValueError):C.validate_driver_spec(ROOT,bad)
            self.assertEqual(result['compile_calls'],0)
        self.assertEqual(self.d.select_job_spec(dict(prefix=5)),self.d.SPECS[5])


if __name__=='__main__':unittest.main(verbosity=2)
