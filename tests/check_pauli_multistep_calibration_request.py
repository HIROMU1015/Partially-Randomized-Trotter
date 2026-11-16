"""Canonical saved schedules plus artificial D2 responses; no new prices/signals."""
from copy import deepcopy
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import check_pauli_calibration_request as OLD
from check_pauli_calibration_request import prepared,missing,E,A,C,PRICES,ROOT
import prepare_pauli_calibration as CLI


class MultiStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the original saved fixture and science-forbidden leaf patches.
        OLD.CalibrationTests.setUpClass()
        cls.saved=OLD.CalibrationTests.preps[5];cls.B=E.p1.b_api(ROOT)

    def forbid(self):return OLD.CalibrationTests.no_science(self.helper)

    def setUp(self):
        self.helper=OLD.CalibrationTests();self.helper.d=OLD.CalibrationTests.d;self.helper.B1=OLD.CalibrationTests.B1

    def config(self,width):
        spec=deepcopy(self.saved['configuration']['spec']);spec['delta_pf']=A.encoded(width)
        return self.B.generate(spec)

    def synthetic_quote(self,config,reuse=()):
        q=missing(config)
        for row in q['rows']:
            if row['m']in reuse:
                row.update(classification='MODEL_REFERENCE',unit_price_RZ=dict(numerator=1234,denominator=1),
                    source=dict(fixture='ARTIFICIAL_SAME_PHYSICAL_PRICE_NOT_SCIENCE',m=row['m'],axis=row['axis']))
        q['complete']=all(r['unit_price_RZ']is not None for r in q['rows'])
        return q

    def test_binary64_two_tenths_reuses_early_round_and_requests_one_new_step(self):
        config=self.config(.2);quote=self.synthetic_quote(config,[0]);before=deepcopy((config,quote))
        with self.forbid():out=C.prepare(ROOT,config,quote)
        self.assertEqual(out['status'],'CALIBRATION_REQUEST_READY_NOT_EXECUTED',out['needs'])
        self.assertEqual(len(out['reused_prices']),2)
        self.assertEqual(out['round_actions'][0]['action'],'REUSE_SUPPLIED_PRICES')
        self.assertEqual(len(out['calibration_requests']),1)
        g=out['calibration_requests'][0];spec=g['driver_request']['driver_spec']
        self.assertEqual(spec['covered_rounds'],list(range(1,13)))
        self.assertEqual(C.frac(spec['calibration_domain']['actual_step']),Fraction(.2))
        self.assertEqual(spec['cell']['T'],1.6)
        self.assertEqual(spec['cell']['R'],config['rounds'][12]['R_outer'])
        self.assertEqual(g['driver_metadata']['request']['step_time'],.2)
        OLD.CalibrationTests.d.select_job_spec(dict(prefix=5,spec=spec,binding=g['driver_metadata']))
        self.assertEqual(before,(config,quote))
        self.assertEqual(out['science_calls'],0)

    def test_multiple_actual_steps_group_rounds_without_rewriting_h_or_T(self):
        config=self.config(1.6)
        with self.forbid():out=C.prepare(ROOT,prepared(config),self.synthetic_quote(config,[0]))
        self.assertEqual(out['status'],'CALIBRATION_GROUPS_READY_NOT_EXECUTED',out['needs'])
        self.assertEqual(len(out['calibration_requests']),4)
        self.assertEqual(len({g['driver_request']['driver_spec']['base']for g in out['calibration_requests']}),4)
        self.assertEqual(len({g['driver_request']['driver_spec']['private']for g in out['calibration_requests']}),4)
        self.assertEqual([C.frac(g['proposed_applicability']['actual_step'])for g in out['calibration_requests']],
                         [Fraction(.2),Fraction(.4),Fraction(.8),Fraction(1.6)])
        last=out['calibration_requests'][-1]
        self.assertEqual(last['covered_rounds'],list(range(4,13)))
        self.assertEqual([r['T']for r in config['rounds']],[r['T']for r in self.saved['configuration']['rounds']])
        for g in out['calibration_requests']:
            spec=g['driver_request']['driver_spec'];binding=OLD.CalibrationTests.d.metadata(spec)
            self.assertEqual(C.frac(binding['calibration_domain']['actual_step']),Fraction(binding['request']['step_time']))
            self.assertEqual(binding['request']['m'],None)
            self.assertEqual(binding['compile_maximum'],4)
        with self.forbid():bad=C.prepare(ROOT,config,self.synthetic_quote(config,[0]),
            execution=dict(base='same-output-for-all-steps',private='same-private-for-all-steps'))
        self.assertEqual(bad['status'],'NEEDS_INPUT_OR_FIXED_IMPLEMENTATION_SOURCE')
        self.assertIn('DISTINCT_OUTPUT_LOCATIONS_REQUIRED',bad['needs'][0])
        self.assertEqual(bad['calibration_requests'],[])
        self.assertIsNone(bad['driver_request'])

    def test_exact_fifth_has_different_q_and_returns_specific_blocked_rounds(self):
        config=self.config(Fraction(1,5));binary=self.config(.2)
        with self.forbid():out=C.prepare(ROOT,config,self.synthetic_quote(config,[0,1]))
        self.assertEqual((config['rounds'][1]['q_PF'],binary['rounds'][1]['q_PF']),(2,1))
        self.assertEqual(out['status'],'NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION')
        self.assertEqual(out['calibration_requests'],[])
        blocked=[r for r in out['round_actions']if r['action']=='NEEDS_RULE_OR_IMPLEMENTATION_EXTENSION']
        self.assertEqual([r['m']for r in blocked],list(range(2,13)))
        self.assertTrue(all('EXACT_STEP_OR_T_NOT_BINARY64_DRIVER_REPRESENTABLE' in r['needs'][0]for r in blocked))
        self.assertEqual(len(out['reused_prices']),4)

    def test_same_physical_cell_deduplicates_distinct_requested_width_candidates(self):
        a=self.config(.2);b=self.config(math.nextafter(.2,math.inf))
        self.assertNotEqual(a['id'],b['id'])
        inputs=[dict(configuration=c,quote=self.synthetic_quote(c,[0]))for c in (a,b)]
        with self.forbid():out=C.prepare_many(ROOT,inputs)
        self.assertEqual(out['unique_calibration_requests'],1)
        self.assertEqual(out['maximum_new_paths'],2);self.assertEqual(out['maximum_new_compiles'],4)
        self.assertEqual(len(out['calibration_requests'][0]['consumers']),2)
        self.assertEqual({r['configuration_id']for r in out['results']},{a['id'],b['id']})

    def test_changed_width_and_batch_use_real_CLI_driver_metadata_path(self):
        config=self.config(1.6);quote=self.synthetic_quote(config,[0])
        with self.forbid(),tempfile.TemporaryDirectory()as tmp:
            base=Path(tmp);(base/'config.json').write_text(json.dumps(config));(base/'quote.json').write_text(json.dumps(quote))
            with patch('sys.stdout',new=io.StringIO()):
                out=CLI.main(['--preparation',str(base/'config.json'),'--price-result',str(base/'quote.json'),
                    '--output',str(base/'result.json')])
            self.assertEqual(len(out['calibration_requests']),4)
            for group in out['calibration_requests']:
                spec=OLD.CalibrationTests.d.load_request_spec(group['driver_request_file'])
                with patch.object(sys,'argv',['cal','--request',group['driver_request_file'],'--metadata']), \
                        patch('sys.stdout',new=io.StringIO())as capture:
                    OLD.CalibrationTests.d.main()
                self.assertEqual(json.loads(capture.getvalue())['calibration_domain'],spec['calibration_domain'])
            batch=[dict(configuration=self.config(.2),quote=self.synthetic_quote(self.config(.2),[0]))]*2
            (base/'batch.json').write_text(json.dumps(batch))
            with patch('sys.stdout',new=io.StringIO()):
                out=CLI.main(['--batch',str(base/'batch.json'),'--output',str(base/'batch_result.json')])
            self.assertEqual(out['unique_calibration_requests'],1)

    def test_step_R_and_h_mismatch_do_not_become_new_calibration(self):
        config=self.config(.2)
        with self.forbid():out=C.prepare(ROOT,config,self.synthetic_quote(config,[0]))
        spec=deepcopy(out['driver_request']['driver_spec']);spec['calibration_domain']['actual_step']=dict(numerator=1,denominator=5)
        with self.assertRaisesRegex(ValueError,'EXACT_STEP'):C.validate_driver_spec(ROOT,spec)
        spec=deepcopy(out['driver_request']['driver_spec']);spec['covered_rounds']=[0]
        with self.assertRaises(ValueError):C.validate_driver_spec(ROOT,spec)
        wrong=self.synthetic_quote(config);wrong['rows'][0]['q']+=1
        with self.forbid():bad=C.prepare(ROOT,config,wrong)
        self.assertEqual(bad['status'],'NEEDS_INPUT_OR_FIXED_IMPLEMENTATION_SOURCE')
        spec=deepcopy(config['spec']);spec['h']=A.encoded(.2);different=self.B.generate(spec)
        with self.forbid():bad=C.prepare(ROOT,different,missing(different))
        self.assertIn('FIXED_RPE_H_REQUIRED',bad['needs'][0])


if __name__=='__main__':unittest.main(verbosity=2)
