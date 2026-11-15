"""Artificial exact prices only; no D1 observation reads or real totals."""
import unittest
from copy import deepcopy
from fractions import Fraction
from unittest.mock import patch

import account_full_h_reference_scenario as C
import hashlib
import json


def fixture():
    rows=[dict(m=m,axis=a,N=1,C_ref=C.exact(1),status='ACCEPTED_SAVED_REFERENCE')
          for m in range(13) for a in ('X','Y')]
    rows[0].update(N=2,C_ref=C.exact(7));rows[1].update(N=3,C_ref=C.exact(11))
    return rows,[dict(label='mock_low',T_PF=50),dict(label='mock_high',T_PF=100)]


def stopped_fixture():
    unit=dict(physical_id='mock',original_configuration_id='mock_configuration',total_configuration_id='mock_total',
        original_candidate_id='mock_candidate',total_candidate_id='mock_decoder',B_join={'mock':'join'})
    rule=dict(physical_candidate={'id':'mock'},profile={'compiler':'mockcompiler'},implementation={'lowering':'mock'},
        proposed_acquisition=dict(cohort='mock',seed_u64=17),target_preparation_hash='mockprep',
        scalar={'total':-1},H_state_snapshot={'state':'mockstate'},
        units=[dict(unit,axis=axis) for axis in ('X','Y')])
    mapping={k:dict(commit='b'*40,path=k) for k in ('observed','job','result','stop','prices')}
    mapping.update(rule_source=C.RULE,execution_source='a'*40)
    job=dict(rule_source=C.RULE,row=rule['physical_candidate'],profile=rule['profile'],implementation=rule['implementation'],
        cohort='mock',index=0,seeds=[17],independent_paths=1)
    source={'commit':'a'*40}
    result=dict(source=source,status='STOP',mandatory_stop=True,child={'context_check':dict(status='PASS',
        saved_context=dict(preparation_hash='mockprep',scalar=-1,binding={'state':'mockstate'}),
        compiler_profile=rule['profile'],implementation=rule['implementation'])})
    stop=dict(accepted_axis_prices=0,held_axis_prices=2,C_0X=None,C_0Y=None,failed_requirement='mock required gate')
    prices=[];observed=[];records={}
    for axis in ('X','Y'):
        record=dict(compiler='mockcompiler',fingerprint=axis,metrics={'rz_count':7})
        records[axis]=json.dumps(record).encode()
        prices.append(dict(axis=axis,m=0,physical_id='mock',per_shot_RZ=None,
            acceptance='HELD_NULL_REQUIRED_EVIDENCE_INCOMPLETE',source=source))
        observed.append(dict(axis=axis,physical_id='mock',path_index=0,cohort='mock',seed=17,source=source,
            status='OBSERVED_COMPILE_ONLY_NOT_ACCEPTED_REFERENCE',per_shot_RZ=None,fingerprint=axis,
            observed_compiled_RZ=7,measurement=dict(axis=axis,include_measurement=True),
            compile_record=dict(path=axis,sha256=hashlib.sha256(records[axis]).hexdigest())))
    evidence=dict(job=job,result=result,stop=stop,prices=prices,observed=observed)
    return rule,mapping,evidence,records


class Tests(unittest.TestCase):
    def test_weight_and_ratio_and_requested_difference_sign(self):
        r=C.account(*fixture())
        self.assertEqual(r['m0_partial'],C.exact(47));self.assertEqual(r['T'],C.exact(71))
        self.assertEqual(r['m0_fraction'],C.exact(Fraction(47,71)))
        self.assertEqual(r['m0_percent'],C.exact(Fraction(4700,71)))
        self.assertEqual([v['T_minus_T_PF'] for v in r['comparisons']],[C.exact(21),C.exact(-29)])

    def test_one_missing_axis_never_sums_or_substitutes(self):
        rows,b=fixture();rows[1].update(status='HELD',C_ref=None,reason='Missing Y receipt')
        with patch.object(C,'rational',side_effect=AssertionError('No arithmetic before admission')):
            r=C.account(rows,b)
        self.assertIsNone(r['T']);self.assertIsNone(r['m0_partial'])
        self.assertEqual(r['missing'],[dict(m=0,axis='Y',reason='Missing Y receipt')])

    def test_duplicate_or_negative_quote_is_rejected_without_repair(self):
        rows,b=fixture();rows[-1]=deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'Unique'):C.account(rows,b)
        rows,b=fixture();rows[0]['C_ref']=C.exact(-1)
        with self.assertRaisesRegex(ValueError,'Nonnegative'):C.account(rows,b)

    def test_actual_reader_route_preserves_producer_stop_and_never_sums(self):
        rule,mapping,evidence,records=stopped_fixture()
        with patch.object(C.G,'evidence',side_effect=lambda root,ref:evidence[ref['path']]), \
             patch.object(C.G.C,'blob',side_effect=lambda root,commit,path:records[path]):
            m0,admissions=C.admit_m0(None,rule,mapping)
        self.assertEqual([r['status'] for r in admissions],['HELD_D1_REQUIRED_GATE_STOP']*2)
        rows,benchmarks=fixture();rows[:2]=m0
        with patch.object(C,'rational',side_effect=AssertionError('No summing held path')):
            result=C.account(rows,benchmarks)
        self.assertIsNone(result['T']);self.assertEqual(len(result['missing']),2)
        self.assertTrue(all(r['C_ref'] is None for r in m0))
        self.assertEqual([r['original_observation']['observed_compiled_RZ'] for r in admissions],[7,7])

    def test_held_reader_does_not_repair_scalar_source_or_axis(self):
        for mutation in ('scalar','producer','axis'):
            rule,mapping,evidence,records=stopped_fixture()
            if mutation=='scalar': evidence['result']['child']['context_check']['saved_context']['scalar']=0
            if mutation=='producer': mapping['execution_source']='c'*40
            if mutation=='axis': evidence['observed'][1]['measurement']['axis']='X'
            with self.assertRaises(ValueError):
                C.held_m0(rule,mapping,**evidence,compiled_records={a:json.loads(v) for a,v in records.items()})

    def test_successful_publication_requires_its_own_admission_not_stop_reader(self):
        rule,mapping,evidence,records=stopped_fixture();evidence['result']['status']='M0_REFERENCE_COMPLETE'
        with self.assertRaisesRegex(ValueError,'Published D1 STOP'):
            C.held_m0(rule,mapping,**evidence,compiled_records={a:json.loads(v) for a,v in records.items()})


if __name__=='__main__':unittest.main()
