"""Artificial batch/packet fixtures only; no saved-H science or old reanalysis."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import df_head_pauli_tail_completion as F

spec=importlib.util.spec_from_file_location('completion_fixture',ROOT/'tests/check_df_head_pauli_tail_p1.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
OLD='1'*40;NEW='2'*40;RAW='3'*40;SOURCE=dict(commit=NEW,fixture='ARTIFICIAL_ONLY')


def record(job):
    return dict(status=F.LOCAL,m=job['m'],B_candidate_id=job['B_candidate_id'],B_configuration_id=job['B_configuration_id'],
        B_identity=deepcopy(job['B_identity']),B_law_binding=deepcopy(job['B_law_binding']),
        candidate=deepcopy(job['physical_context']['physical_candidate']),
        source=deepcopy(job['physical_context']['numerical_recipe']['source']),
        numerical=dict(gate_pass=True),signal=dict(raw=dict(real=.25,imag=.125)),
        N_X=None,N_Y=None,C=None,G=None,global_RMSE_bound=None,scientific_PASS_FAIL=None)


def setup():
    base=dict(packets=[]);jobs={}
    for prefix in F.PREFIXES:
        b,_,_,c,_,_=fixture.setup(prefix=prefix,M=12)
        p=dict(prefix=prefix,configuration=c,scientific_source=OLD,evaluated_rounds=[0,6],rounds=[])
        for canonical in c['rounds']:
            m=canonical['m'];j=F.A.job(c,m,SOURCE,b)
            v=record(j) if m in F.REUSED else None
            if v:v['source']=dict(commit=OLD,fixture='ARTIFICIAL_OLD_RECORD')
            p['rounds'].append(dict(m=m,candidate_id=canonical['candidate']['id'],identity=canonical['identity'],
                law_binding=canonical['law_binding'],mean_record=v,
                mean_source=None if v is None else dict(path='OLD_RELATIVE_SOURCE'),
                status=F.LOCAL if v else 'NOT_EVALUATED',N_X=None,N_Y=None,nominal_N_X=None,nominal_N_Y=None,
                sufficient_N_X=None,sufficient_N_Y=None,global_RMSE_bound=None,C=None,G=None))
            if m in F.MISSING:jobs[prefix,m]=j
        base['packets'].append(p)
    return base,jobs


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.jobs=setup()

    def setUp(self):
        guard=patch.object(np,'load',side_effect=AssertionError('No real arrays'))
        guard.start();self.addCleanup(guard.stop)

    def test_only_22_missing_called_and_no_P1_or_full_mean(self):
        seen=[];products={};result=dict(records=[])
        def evaluate(p,m,j):seen.append((p,m));return record(j)
        F.run_missing(self.jobs,evaluate,lambda n,v:products.update({n:deepcopy(v)}),result)
        self.assertEqual(result['status'],F.SUCCESS);self.assertEqual(len(seen),22)
        self.assertFalse(any(m in F.REUSED for p,m in seen))
        self.assertEqual(seen,[(p,m) for m in F.MISSING for p in F.PREFIXES])
        self.assertTrue(all(v['space']=='sector36' for v in result['records']))

    def test_first_numeric_FAIL_preserved_and_no_followup(self):
        result=dict(records=[]);seen=[];products={}
        def evaluate(p,m,j):
            seen.append((p,m));v=record(j)
            if len(seen)==3:v.update(status='NUMERICAL_FAILURE',numerical=dict(gate_pass=False),signal=None)
            return v
        ok=F.run_missing(self.jobs,evaluate,lambda n,v:products.update({n:deepcopy(v)}),result)
        self.assertFalse(ok);self.assertEqual(len(seen),3);self.assertEqual(len(result['records']),3)
        self.assertEqual(result['records'][-1]['record']['status'],'NUMERICAL_FAILURE')
        self.assertIn('LD1_m2.json',products)

    def test_returned_source_or_identity_mismatch_stops_immediately(self):
        result=dict(records=[])
        def evaluate(p,m,j):v=record(j);v['B_candidate_id']='FOREIGN';return v
        self.assertFalse(F.run_missing(self.jobs,evaluate,lambda *args:None,result))
        self.assertEqual(len(result['records']),1)
        self.assertIn('MISMATCH',result['status'])

    def test_join_preserves_old_values_ids_law_and_per_round_SOURCE(self):
        new=[dict(prefix=p,m=m,record=record(j)) for (p,m),j in self.jobs.items()]
        out=F.join_packets(self.base,new,raw_commit=RAW,science_source=NEW)
        self.assertEqual(out['status'],F.SUCCESS)
        for old,packet in zip(self.base['packets'],out['packets']):
            self.assertEqual(packet['configuration'],old['configuration'])
            self.assertEqual(packet['evaluated_rounds'],list(range(13)))
            for a,b in zip(old['rounds'],packet['rounds']):
                self.assertEqual(a['identity'],b['identity']);self.assertEqual(a['law_binding'],b['law_binding'])
                if a['m'] in F.REUSED:
                    self.assertEqual(a['mean_record'],b['mean_record'])
                    self.assertEqual(b['mean_source']['commit'],F.P1)
                    self.assertEqual(b['mean_record']['source']['commit'],OLD)
                else:self.assertEqual(b['mean_source']['commit'],RAW)
                for k in ('N_X','N_Y','nominal_N_X','nominal_N_Y','sufficient_N_X','sufficient_N_Y','global_RMSE_bound','C','G'):
                    self.assertIsNone(b[k])

    def test_partial_packet_and_unauthorized_or_duplicate_items(self):
        p,m=next(iter(self.jobs));one=dict(prefix=p,m=m,record=record(self.jobs[p,m]))
        out=F.join_packets(self.base,[one],raw_commit=RAW,science_source=NEW)
        self.assertIn('INCOMPLETE',out['status'])
        self.assertIsNone(out['packets'][0]['rounds'][2]['mean_record'])
        for bad in ([one,one],[dict(one,m=0)]):
            with self.assertRaises(ValueError):F.join_packets(self.base,bad,raw_commit=RAW,science_source=NEW)
        wrong=deepcopy(one);wrong['record']['source']['commit']=OLD
        with self.assertRaises(ValueError):F.join_packets(self.base,[wrong],raw_commit=RAW,science_source=NEW)

    def test_existing_engine_artificial_m12_without_new_N_or_decoder(self):
        b,_,_,config,contexts,ref=fixture.setup(M=12)
        job=F.A.job(config,12,SOURCE,b);ctx=contexts[0]
        value=F.A.provider.evaluate(ctx,job,ref(ctx,job),root=ROOT)
        self.assertEqual(value['status'],F.LOCAL)
        self.assertEqual(value['candidate']['q'],4096)
        self.assertTrue(value['raw_corrected_independent_propagation'])
        self.assertTrue(value['numerical']['gate_pass']);self.assertIsNone(value['N_X'])


if __name__=='__main__':unittest.main()
