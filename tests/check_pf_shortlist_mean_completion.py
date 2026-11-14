"""Completion metadata and artificial C.execute/numerical fixtures only."""
from copy import deepcopy
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pf_shortlist_mean_completion as F
import run_truncated_fullh_signals as TR
SOURCE=dict(fixture='ARTIFICIAL_COMPLETION_ONLY')


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared=F.prepare(ROOT,SOURCE)

    def setUp(self):
        p=patch.object(np,'load',side_effect=AssertionError('No saved molecular arrays in tests'))
        p.start();self.addCleanup(p.stop)

    def test_exact_48_30_scope_and_actual_early_widths(self):
        p=self.prepared
        self.assertEqual((len(p['new']),len(p['saved_sources']),len(p['steps'])),(48,30,10))
        self.assertEqual(sum(r['original_status']=='S2L_QUALIFICATION_PENDING' for r in p['new']),4)
        self.assertFalse(any(r['m'] in (0,1,4,12) for r in p['new']))
        first=p['schedules'][0]['request']['pending']['requests']
        self.assertEqual([r['physical_candidate']['q'] for r in first[:5]],[1,1,2,4,7])
        self.assertEqual(first[3]['physical_candidate']['delta'],.2)
        self.assertNotEqual(first[3]['physical_candidate']['delta'],float(F.S.CANDIDATES[0]['width']))

    def test_staged_30_values_source_origins_and_normalization_unchanged(self):
        for schedule in self.prepared['schedules']:
            label=schedule['label'];old=F.read(ROOT,F.STAGING,F.STAGEDIR+'STAGED_SIGNAL_INPUT_'+label+'.json')
            for record in schedule['saved']:
                m=record['domain']['reference']['m'];req=schedule['request']['pending']['requests'][m]
                rebound=F.C.b_mean(req,record)
                for key in ('raw','corrected','reference','normalization'):
                    self.assertEqual(rebound[key],old['means'][m][key])
                self.assertEqual(record['source'],SOURCE)
                self.assertEqual(record['input_source']['saved_mean'],old['means'][m]['provider_binding']['input_source']['saved_mean'])

    def test_48_only_provider_calls_30_saved_copies_no_extra_rounds(self):
        called=[];api=F.I.providers(str(ROOT))
        reader=TR.reference_reader(ROOT,self.prepared['shared'],api)
        def mock_evaluate(ctx,req,ref,*,source):
            called.append((req['physical_candidate']['method'],req['physical_candidate']['id']))
            return dict(domain=deepcopy(req['domain']),source=source,status='NUMERICAL_MEAN_AVAILABLE',
                signal=dict(raw=dict(real=.25,imag=.125),corrected=dict(real=.25,imag=.125),reference=ref['reference'],
                    B=1.,log_B=0.,attenuation=1.,total_signed_bias=None),numerical=dict(gates_passed=True),input_source=SOURCE)
        with patch.object(F.N,'evaluate',side_effect=mock_evaluate),tempfile.TemporaryDirectory() as tmp:
            for schedule in self.prepared['schedules']:
                packet=F.I.produce_signals(ROOT,schedule['request'],Path(tmp)/schedule['label'],
                    context_factory=lambda: type('Mock',(),dict(binding=self.prepared['shared']['binding']))(),
                    reference_provider=reader,saved=schedule['saved'])
                self.assertEqual(len(packet['means']),13);F.S.require_complete(packet)
                self.assertEqual(packet['producer_result']['saved_copies'],len(schedule['saved']))
        self.assertEqual(len(called),48);self.assertEqual(len(set(called)),48)

    def test_saved_step_context_rejects_outside_recipe_and_scalar(self):
        s=self.prepared['schedules'][0];entries=[x for x in self.prepared['new'] if x['label']==s['label']]
        e=entries[0];d=e['step_source'];state=np.array([1.,0.],complex);op=np.eye(2)
        ctx=F.SavedStepContext(ROOT,s['request'],dict(state=state,constant=2.29310124732,tail_scalar=.03,source=SOURCE),
            entries,{(d['commit'],d['path']):op})
        c=s['request']['pending']['requests'][e['m']]['physical_candidate']
        self.assertIs(ctx.pf_step(c,e['structure']['signed_weights']),op)
        bad=deepcopy(c);bad['q']+=1
        with self.assertRaises(ValueError):ctx.pf_step(bad,e['structure']['signed_weights'])
        ctx.constant+=.1
        with self.assertRaises(ValueError):ctx.pf_step(c,e['structure']['signed_weights'])

    def test_full_and_B0I_scalar_once_on_artificial_step(self):
        t=.4;theta=.07;state=np.array([1.,0.],complex);step=np.diag(np.exp(-1j*np.array([theta,-theta])))
        api=F.I.providers(str(ROOT));kernel=F.N.kernels(str(ROOT))
        for s in (self.prepared['schedules'][0],self.prepared['schedules'][3]):
            req=s['request']['pending']['requests'][2]
            context=type('Toy',(),dict(root=str(ROOT),state=state,constant=.17,metadata=dict(source=SOURCE),
                pf_step=lambda self,c,w:step,tail_scalar=lambda self,p:.03))()
            ref=dict(deepcopy(req['domain']['reference']),reference=dict(real=.4,imag=.1),
                independent_reference=dict(real=.4,imag=.1),signal_difference=0.)
            actual=F.N.evaluate(context,req,ref,source=SOURCE)
            self.assertEqual(actual['status'],'NUMERICAL_MEAN_AVAILABLE')
            scalar=.17+(.03 if req['physical_candidate']['method']=='B0I' else 0.)
            expected=np.exp(-1j*(req['physical_candidate']['q']*theta+t*scalar))
            z=actual['signal']['raw'];self.assertLess(abs(complex(z['real'],z['imag'])-expected),2e-14)
            self.assertEqual(actual['signal']['raw'],actual['signal']['corrected'])

    def test_numeric_failure_keeps_partial_and_does_not_request_next_mean(self):
        schedule=self.prepared['schedules'][0];seen=[]
        reader=TR.reference_reader(ROOT,self.prepared['shared'],F.I.providers(str(ROOT)))
        def failure(ctx,req,ref,*,source):
            seen.append(req['m'])
            return dict(domain=deepcopy(req['domain']),source=source,status='NUMERICAL_UNRESOLVED',
                signal=None,numerical=dict(gates_passed=False),reason='ARTIFICIAL_GATE_FAIL')
        with patch.object(F.N,'evaluate',side_effect=failure),tempfile.TemporaryDirectory() as tmp:
            packet=F.I.produce_signals(ROOT,schedule['request'],Path(tmp)/'run',
                context_factory=lambda:type('Mock',(),dict(binding=self.prepared['shared']['binding']))(),
                reference_provider=reader,saved=schedule['saved'])
            self.assertEqual(seen,[2]);self.assertEqual(packet['producer_result']['status'],'STOP_NUMERICAL_UNRESOLVED')
            self.assertEqual(packet['means'][2]['status'],'NUMERICAL_UNRESOLVED')
            with self.assertRaises(ValueError):F.S.require_complete(packet)


if __name__=='__main__':unittest.main()
