"""Invented laws/word tapes/mock outcomes only; never run a real campaign."""
from dataclasses import replace
from concurrent.futures import Future
from fractions import Fraction as F
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import json
import unittest
from unittest.mock import Mock,patch

from trottertracks.algorithm_codesign import a1_n3_campaign as c
from trottertracks.algorithm_codesign import a1_n3_connection as a
from trottertracks.algorithm_codesign import a1_control_variates as cv


def fixture():
    law=cv.OccurrenceLaw(cv.CategoricalLaw((0,2),(F(3,4),F(1,4)),'toy'),
        cv.CategoricalLaw(('a','b','z'),(F(1,4),F(3,4),F(0)),'toy'))
    sampler=a.ConditionalSampler(law,'toy','a','b')
    scopes={axis:cv.Scope('toy-target','toy','toy-wrapper','m0',axis,F(1)) for axis in ('X','Y')}
    facade=a.MeasurementFacade(lambda *args:F(1,2),scopes)
    price=NS(price=Mock(return_value=(NS(rz_count=11,rz_depth=2,cx_count=3,cx_depth=2,total_depth=9,
        circuit_size=20,qubit_count=3,compiler_settings_hash='toy-c',actual_circuit_fingerprint='toy-f'),True)))
    services=(facade,price,sampler,scopes,dict(total_engine_coordinate_upper_exact='1/10000'))
    return services


def streams(domain,path_words=(0,0,0,0),bit=0):
    book=a.Streams(64);tape=iter(path_words)
    book.bind(domain,lambda:next(tape),lambda:bit)
    return book


class CampaignDriverTests(unittest.TestCase):
    def test_fixed_domain_count_seed_order_and_no_duplicate(self):
        ds=c.domains();self.assertEqual(len(ds),60);self.assertEqual(len(set(ds)),60)
        self.assertEqual(ds,tuple(sorted(ds,key=lambda d:d.token)))
        self.assertFalse(any(d.method=='B0' and d.stage=='train' for d in ds))
        self.assertEqual(sum(d.method=='A1' and d.stage=='train' for d in ds),10)

    def test_journal_precedes_path_generation_and_records_word_tape(self):
        services=fixture();d=a.Domain('B0','Y','cal','all')
        with TemporaryDirectory() as tmp:
            out=Path(tmp);j=out/'journal';w=out/'words'
            job=c.prepare_trial(services[2],streams(d),d,0,j,w)
            rows=[json.loads(x) for x in j.read_text().splitlines()]
            self.assertEqual([x['event'] for x in rows],['TRIAL_PLANNED','PATH_FIXED'])
            self.assertEqual(job['path'].order,0)
            self.assertEqual(json.loads(w.read_text())['measurement_word'],0)
            self.assertEqual(len(json.loads(w.read_text())['path_words']),2)

    def test_pinned_replay_draws_no_additional_random_words(self):
        s=fixture()[2];p=a.PathChoice(2,('a','b','b'));pinned=c.PinnedSampler(s,p,'B1-OL','4')
        no_word=Mock(side_effect=AssertionError('no extra RNG'))
        self.assertEqual(pinned.draw('B1-OL','4',no_word),p);no_word.assert_not_called()
        with self.assertRaisesRegex(ValueError,'replay'):pinned.draw('B0','all',no_word)

    def test_worker_preserves_bit_before_mock_price_and_exports_no_oracle(self):
        services=fixture();d=a.Domain('B0','X','cal','all')
        with TemporaryDirectory() as tmp:
            out=Path(tmp);log=out/'worker_toy.jsonl';job=c.prepare_trial(services[2],streams(d),d,0,out/'JOURNAL.jsonl',out/'engine_internal_words.jsonl')
            with patch.object(c,'_SERVICES',services),patch.object(c,'_WORKER',(out,log,None,None)),patch.object(c,'_STOP_EVENT',None):
                result=c.worker_trial(job)
            self.assertNotIn('reason',result);self.assertEqual(result['record'].bit,0)
            self.assertNotIn('probability',result);self.assertNotIn('state',result)
            events=[json.loads(x)['event'] for x in log.read_text().splitlines()]
            self.assertLess(events.index('BIT_SAVED'),events.index('PRICE_SAVED'))
            methods=c.reconcile(out);self.assertEqual(methods['B0']['bits_saved'],1)
            self.assertEqual(methods['B0']['realized_RZ_priced'],11)

    def test_price_failure_keeps_bit_and_UNKNOWN_missing_price(self):
        services=fixture();services[1].price.side_effect=ValueError('price failure')
        d=a.Domain('B0','X','cal','all')
        with TemporaryDirectory() as tmp:
            out=Path(tmp);job=c.prepare_trial(services[2],streams(d),d,0,out/'JOURNAL.jsonl',out/'engine_internal_words.jsonl')
            with patch.object(c,'_SERVICES',services),patch.object(c,'_WORKER',(out,out/'worker_toy.jsonl',None,None)),patch.object(c,'_STOP_EVENT',None):
                result=c.worker_trial(job)
            self.assertEqual(result['reason'],'price failure');self.assertTrue(result['bit_saved'])
            summary=c.reconcile(out)['B0'];self.assertEqual(summary['unpriced_bits'],1)
            self.assertFalse(summary['realized_RZ_complete'])
            self.assertEqual(summary['results']['X']['status'],'INCOMPLETE_NO_PRECISION_CLAIM')

    def test_reconcile_rejects_duplicate_trial_bit(self):
        with TemporaryDirectory() as tmp:
            out=Path(tmp);row=dict(event='BIT_SAVED',ticket='same',domain=dict(method='B0',axis='X',stage='cal',cell='all'),record={},engine_seconds=0)
            c.append(out/'worker_toy.jsonl',row);c.append(out/'worker_toy.jsonl',row)
            with self.assertRaisesRegex(ValueError,'duplicate'):c.reconcile(out)

    def test_shared_stop_starts_no_new_probability(self):
        services=fixture();d=a.Domain('B0','X','cal','all')
        with TemporaryDirectory() as tmp:
            out=Path(tmp);job=c.prepare_trial(services[2],streams(d),d,0,out/'JOURNAL.jsonl',out/'engine_internal_words.jsonl')
            stop=NS(is_set=lambda:True)
            with patch.object(c,'_SERVICES',services),patch.object(c,'_WORKER',(out,out/'worker_toy.jsonl',None,None)),patch.object(c,'_STOP_EVENT',stop):
                result=c.worker_trial(job)
            self.assertIn('before engine',result['reason']);self.assertEqual(result['counts'],{})

    def test_fixed_training_serialization_and_guard_match_existing_adapter(self):
        services=fixture();s=services[2];scope=services[3]['X'];batches=[]
        for i,_ in s.cells('B1-OL'):
            d=a.Domain('A1','X','train',i);path=s.draw('B1-OL',i,lambda:0)
            records=tuple(cv.Measurement(d.token,str(j),(path.event,),j%2,F(1),'p','toy-wrapper') for j in range(4))
            batches.append(cv.Batch(scope,'train',records))
        model,rank=a.freeze(s.bank,a.merge_training(batches),4)
        with TemporaryDirectory() as tmp:
            campaign=c.Campaign(tmp,services,None,None,1)
            with patch.object(campaign,'group',return_value=([(str(i),b) for i,b in enumerate(batches)],None)):
                actual=campaign.learn('X')
            saved=json.loads((Path(tmp)/'A1_X_FROZEN.json').read_text())
            self.assertEqual(actual.coefficients,model.coefficients);self.assertEqual(saved['rank'],rank)

    def test_word_budget_failure_still_saves_partial_tape(self):
        services=fixture();d=a.Domain('B0','X','cal','all');book=a.Streams(1)
        book.bind(d,lambda:0,lambda:0)
        with TemporaryDirectory() as tmp:
            out=Path(tmp)
            with self.assertRaisesRegex(ValueError,'budget'):c.prepare_trial(services[2],book,d,0,out/'j',out/'w')
            row=json.loads((out/'w').read_text());self.assertEqual(len(row['path_words']),1);self.assertIsNone(row['measurement_word'])

    def test_parallel_group_reconciles_fixed_trial_order_and_full_RZ_charge(self):
        services=fixture();d=a.Domain('B0','X','cal','all')
        book=a.Streams(64);book.bind(d,lambda:0,lambda:0)
        class ImmediatePool:
            def submit(self,fn,job):
                f=Future();f.set_result(fn(job));return f
        with TemporaryDirectory() as tmp:
            out=Path(tmp);campaign=c.Campaign(out,services,book,ImmediatePool(),2)
            with patch.object(c,'_SERVICES',services),patch.object(c,'_WORKER',(out,out/'worker_toy.jsonl',None,None)),patch.object(c,'_STOP_EVENT',None):
                rows,prices=campaign.group('B0','X','cal',[('all',4)])
            self.assertEqual([r.trial_id for r in rows[0][1].records],['0','1','2','3'])
            self.assertEqual(prices,(F(11),));self.assertEqual(sum(x.quantum[2] for x in campaign.costs),44)

    def test_forbidden_requalification_fails_before_any_numeric_evaluation(self):
        with c.forbidden_science():
            with self.assertRaisesRegex(RuntimeError,'forbids'):c.kernel.SavedEngine.qualify(None,0)


if __name__=='__main__':unittest.main()
