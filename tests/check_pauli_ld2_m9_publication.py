"""Artificial bytes/receipts and process overlap; no scientific sampling/compile."""
from pathlib import Path
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import sys
import unittest
from copy import deepcopy
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_pauli_ld2_m9 as Q
M=Q.M
BARRIER=None


def concurrent_fixture(task):
    BARRIER.wait(timeout=20)
    i=task[1]['index']
    return dict(index=i,result=dict(prices=dict(cosine=dict(rz_count=2+i),sine=dict(rz_count=4+i)),scalar=.25)),dict(index=i,worker_pid=os.getpid())


def fixture():
    binding=dict(request=dict(artificial=True),dictionary_identity=dict(artificial=True),step_seeds=[[11],[12]],
                 domain={a:dict(profile=dict(compiler='ARTIFICIAL')) for a in M.AXES})
    source=dict(commit='0'*40,caller='ARTIFICIAL',cell='M9_ROUND')
    r=dict(status='PASS',seeds=[11],sampling='ORIGINAL_NORMAL_SAMPLER',bit_order=dict(pass_=True),
           checks=[dict(check=k,error=0.,tolerance=1e-9) for k in M.legacy().receipts().CHECKS],n=2,
           fingerprints={a:dict(fingerprint=a,measurement=True,wrapper_qubits=3) for a in M.AXES.values()},
           prices={a:dict(rz_count=10) for a in M.AXES.values()},scalar=.25)
    r['bit_order']={'pass':True}
    r['compile_receipts']={a:dict(fingerprint=a,compiler='ARTIFICIAL',metrics=r['prices'][a]) for a in M.AXES.values()}
    path=dict(index=0,cohort=M.COHORT,request=binding['request'],dictionary_identity=binding['dictionary_identity'],result=r)
    receipts={'p0'+a:dict(axis=a,index=0,cohort=M.COHORT,request=binding['request'],seeds=[11],dictionary_identity=binding['dictionary_identity'],
                         execution_source=source,record=r['compile_receipts'][key]) for a,key in M.AXES.items()}
    raw=gzip.compress(json.dumps(path).encode());manifest=dict(index=0,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    return raw,manifest,binding,receipts,source


class Tests(unittest.TestCase):
    def test_bytes_binding_receipts_and_compact_output(self):
        args=fixture();result,check=Q.checked_raw(*args)
        self.assertEqual(result['result']['prices']['cosine']['rz_count'],10)
        self.assertEqual(set(result['result']),{'prices','scalar'})
        self.assertEqual(check['status'],'RAW_AND_RECEIPTS_CHECKED')
        for key in ('bytes','sha256','index'):
            bad=deepcopy(args[1]);bad[key]='WRONG'
            with self.assertRaises(ValueError):Q.checked_raw(args[0],bad,*args[2:])
        for key in ('axis','execution_source','seeds'):
            bad=deepcopy(args[3]);bad['p0X'][key]='WRONG'
            with self.assertRaises(ValueError):Q.checked_raw(*args[:3],bad,args[4])
        bad=deepcopy(args[3]);del bad['p0Y']
        with self.assertRaises(KeyError):Q.checked_raw(*args[:3],bad,args[4])

    def test_actual_process_overlap_order_and_exact_rows(self):
        global BARRIER
        BARRIER=mp.get_context('fork').Barrier(2)
        tasks=[('ARTIFICIAL',dict(index=i),None,None,None) for i in (1,0)]
        paths,checks=Q.parallel_paths(tasks,loader=concurrent_fixture)
        self.assertEqual([p['index'] for p in paths],[0,1])
        self.assertEqual(len({x['worker_pid'] for x in checks}),2)
        pre=dict(predictions=[dict(axis=a,predicted_RZ=M.enc(10)) for a in M.AXES])
        units={a:[dict(valid=True,compiled_RZ=p['result']['prices'][key]['rz_count']) for p in paths] for a,key in M.AXES.items()}
        self.assertEqual(M.compare(pre,units)['rows'][0]['O'],M.enc(2.5))
        self.assertEqual(M.compare(pre,units)['rows'][1]['O'],M.enc(4.5))
        with self.assertRaises(ValueError):Q.parallel_paths(tasks[:1],loader=concurrent_fixture)


if __name__=='__main__':unittest.main(verbosity=2)
