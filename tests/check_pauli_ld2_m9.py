"""Synthetic/provider toy and metadata only. No H4 build/sampling/price."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy

ROOT=Path(__file__).resolve().parents[1]
PIN_ROOT=ROOT.parent/'c2-pauli-m9-source-13ab'
sys.path[:0]=[str(PIN_ROOT/'scripts'),str(PIN_ROOT/'tests')]
import check_pauli_ld1_m0_pilot_circuits as T  # provider-first original toy fixture
spec=importlib.util.spec_from_file_location('c2_m9',ROOT/'scripts/run_pauli_ld2_m9.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_fixed_metadata_and_old_validator_not_extended(self):
        result=json.loads(subprocess.check_output([sys.executable,'-B',str(ROOT/M.SCRIPT),'--metadata'],cwd=ROOT))
        self.assertEqual(result['request']['candidate_id'],M.CID)
        self.assertEqual([len(row) for row in result['step_seeds']],[512,512])
        self.assertTrue(result['seed_checks']['unique_new'])
        self.assertFalse(result['seed_checks']['old_cohort_overlap'])
        for field in ('cohort','legacy_source'):
            bad=deepcopy(result);bad[field]='WRONG'
            with self.assertRaises(ValueError):M.check_binding(bad,result)
        for target in ('candidate_id','q'):
            bad=deepcopy(result);bad['request'][target]='WRONG'
            with self.assertRaises(ValueError):M.check_binding(bad,result)
        bad=deepcopy(result);bad['step_seeds'][0][0]+=1
        with self.assertRaises(ValueError):M.check_binding(bad,result)
        B=M.legacy();plan=B.proposal(2,'B1-CAL');plan['cells'][-1]['cell_id']='M9_ROUND'
        with self.assertRaises(ValueError):B.validate(plan)

    def test_compare_exact_missing_zero_axes_separate(self):
        pre=dict(predictions=[dict(axis=a,predicted_RZ=M.enc(p)) for a,p in [('X',10),('Y',20)]])
        units={a:[dict(valid=True,compiled_RZ=2),dict(valid=True,compiled_RZ=6)] for a in M.AXES}
        r=M.compare(pre,units)
        self.assertEqual(r['rows'][0]['O'],M.enc(4))
        self.assertEqual(r['rows'][0]['signed_error'],M.enc(6))
        self.assertEqual(r['rows'][1]['signed_error'],M.enc(16))
        self.assertEqual(r['rows'][0]['signed_relative_error'],M.enc(1.5))
        units['X'].pop();r=M.compare(pre,units)
        self.assertIsNone(r['rows'][0]['O']);self.assertEqual(r['campaign'],'INCOMPLETE_OR_INVALID')
        units={a:[dict(valid=True,compiled_RZ=0),dict(valid=True,compiled_RZ=0)] for a in M.AXES}
        r=M.compare(pre,units);self.assertIsNone(r['rows'][0]['signed_relative_error'])
        self.assertEqual(r['rows'][0]['reason'],'ZERO_OBSERVED_PRICE')
        units['X'][0]['valid']=False
        self.assertIsNone(M.compare(pre,units)['rows'][0]['signed_error'])

    def test_acceleration_toy_receipt_and_rejection(self):
        import h46_long_time_native as native
        B=M.legacy();P=B.B0.connection()
        ctx,d,ops,blocks,req=T.toy()
        # repeated steps test event cache reuse and all original vector gates.
        req=dict(req,q=2,T=.2)
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);native.PRIVATE=folder
            cache=folder/'compiled_cache';cache.mkdir()
            def fake(wrapper,fp,representative):
                record=dict(fingerprint=fp,compiler=native.COMPILER_HASH,metrics={k:1 for k in native.METRICS})
                calls.append(fp);(cache/(fp+'.json')).write_text(json.dumps(record))
                for suffix in ('_wrapper.qpy.gz','_compiled.qpy.gz'):(cache/(fp+suffix)).write_bytes(b'ARTIFICIAL')
                return record,wrapper.copy(),False
            base=B.acquire_path(native,ctx,d,req,cohort=M.COHORT,index=0,seeds=[11,11],operators=ops,blocks=blocks,compile_cached=fake)
            calls.clear();log=M.Progress(folder/'progress.jsonl')
            with M.accelerated_observed(P,log) as stats:
                fast=B.acquire_path(native,ctx,d,req,cohort=M.COHORT,index=0,seeds=[11,11],operators=ops,blocks=blocks,compile_cached=fake)
            self.assertEqual(len(calls),2)
            for k in ('prices','fingerprints','event_metadata','events','scalar','seeds','checks','bit_order'):
                self.assertEqual(base[k],fast[k],k)
            self.assertGreater(stats['event_cache_hits'],0)
            dictionary_identity=dict(artificial=True)
            path=dict(index=0,cohort=M.COHORT,request=req,dictionary_identity=dictionary_identity,result=fast)
            binding=dict(request=req,dictionary_identity=dictionary_identity,step_seeds=[[11,11]],domain={a:dict(profile=dict(compiler=native.COMPILER_HASH)) for a in M.AXES})
            source=dict(commit='0'*40,caller='ARTIFICIAL',cell='M9_ROUND')
            receipts=B.receipts().compile_receipts([path],source,dictionary_identity,folder)
            M.check_result_path(path,binding,receipts,source)
            for key in ('axis','execution_source','seeds'):
                bad=deepcopy(receipts);bad['p0X'][key]='WRONG'
                with self.assertRaises(ValueError):M.check_result_path(path,binding,bad,source)
            stages=[json.loads(l) for l in (folder/'progress.jsonl').read_text().splitlines()]
            self.assertTrue(any(s['stage']=='simulate_1' and s['state']=='DONE' for s in stages))
            self.assertTrue(any(s['stage']=='compose' and s['state']=='DONE' for s in stages))

    def test_progress_failure_and_observer_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'progress.jsonl';log=M.Progress(p)
            with self.assertRaises(ValueError):
                with log.stage('artificial_failure'):raise ValueError('fixture')
            self.assertEqual(json.loads(p.read_text().splitlines()[-1])['state'],'STOP')
            stream=(Path(tmp)/'child.txt').open('w')
            proc=subprocess.Popen([sys.executable,'-c','import time; time.sleep(.15); raise SystemExit(7)'],stdout=stream)
            states=M.observe([(0,proc,stream)],log,poll_seconds=.02)
            self.assertEqual(states[0]['exit_code'],7)
            self.assertIsNone(states[0]['signal'])
            self.assertEqual(json.loads(p.read_text().splitlines()[-1])['stage'],'worker_exit')


if __name__=='__main__':unittest.main(verbosity=2)
