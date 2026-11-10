"""Artificial signature/replay regression; never invokes the continuation."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import complete_h4_df10_price_v2_independent as complete


class CompletionTests(unittest.TestCase):
    def test_four_argument_native_sampler_callback(self):
        key=('artificial',3)
        with patch.dict(complete.RUNNER.SAMPLED,{key:('requests','seeds','law')},clear=True):
            value=complete.saved_requests({'id':'artificial'},object(),'ARTIFICIAL',3)
        self.assertEqual(value,('requests','seeds','law'))

    def test_existing_decoder_replays_without_rng(self):
        import h46_long_time_native as native
        cfg=native.study.install()
        import trottertracks
        trottertracks.__path__.insert(0,str(Path(cfg['provider_root'])/'src/trottertracks'))
        spec=importlib.util.spec_from_file_location('artificial_replay_fixture',ROOT/'tests/test_h46_new2_native.py')
        fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
        fixture.NativeTests.setUpClass();f=fixture.NativeTests
        row=dict(system='ARTIFICIAL',method='PR',formula='2nd',id='artificial',q=16,T=1.6,R_outer=4,K=2,prefix=2)
        reqs,seeds,laws=native.sample_requests(row,f.prep,'ARTIFICIAL_ONLY',0)
        raw=dict(candidate=row,index=0,seeds=seeds,laws=laws,
                 events=[[e.to_dict() for e in r.rte_occurrence.events] for r in reqs])
        import numpy as np
        with patch.object(np.random,'default_rng',side_effect=AssertionError('no RNG on replay')):
            restored,actual_seeds,actual_laws=complete.replay_requests(f.prep,raw)
        self.assertEqual(actual_seeds,seeds);self.assertEqual(actual_laws,laws)
        self.assertEqual([[e.to_dict() for e in r.rte_occurrence.events] for r in restored],raw['events'])


if __name__=='__main__':unittest.main()
