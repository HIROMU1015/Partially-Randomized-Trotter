"""Synthetic-only regression for original recurrence observation definitions."""
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import pf_common_run_contract as c
import h6_pf_metadata_completion as m

class Observations(unittest.TestCase):
    def test_original_recurrences_signed_phase_and_norm_observation(self):
        import numpy as np
        from scipy.linalg import eigh,expm
        for f in ('2nd','4th'):
            for q in (1,2,4,8):
                state=np.array([1,1j],complex)/math.sqrt(2)
                matrices=(np.array([[.2,.1j],[-.1j,-.3]]),np.array([[.05,.2],[.2,-.1]]))
                actions=[lambda v,t,a=a:expm(-1j*t*a)@v for a in matrices]
                forward=[]
                for a in matrices:
                    w,v=eigh(a);forward.append(lambda x,t,w=w,v=v:v@(np.exp(-1j*t*w)*(v.conj().T@x)))
                cell=dict(method='B0',prefix=1,q=q,formula=f,id='synthetic',K=None,R=None,r=None)
                counts=[];r=m.observe_paths(state,actions,forward,cell,.17,.08,on_action=counts.append)
                self.assertLess(r['forward_vector_error'],1e-12)
                self.assertGreaterEqual(r['maximum_intermediate_norm'],1.)
                self.assertEqual(r['raw_corrected_vector_closure'],float(np.linalg.norm(r['raw']-r['corrected'])))
                self.assertEqual(counts.count('raw'),counts.count('corrected'));self.assertEqual(counts.count('forward'),counts.count('corrected'))
                self.assertAlmostEqual(abs(r['phase']),1.)
    def test_vector_closure_is_measured_not_zero_filled(self):
        import numpy as np
        calls=[0]
        def changing(v,t):
            calls[0]+=1;return v*(1+calls[0]*1e-10)
        cell=dict(method='B0',prefix=1,q=1,formula='2nd',id='toy',K=None,R=None,r=None)
        r=m.observe_paths(np.array([1.,0.],complex),[changing,changing],[lambda v,t:v,lambda v,t:v],cell,0.,0.)
        self.assertGreater(r['raw_corrected_vector_closure'],0.)
        self.assertGreater(r['maximum_intermediate_norm'],1.)
    def test_metadata_new_grant_and_one_shot(self):
        with tempfile.TemporaryDirectory() as d,patch.object(c,'verify'):
            c.save(Path(d)/'plan_APPROVED.json',dict(SOURCE='SYNTHETIC',system='H6'))
            s=c.Stage(d,'METADATA',dict(metadata_candidate=1));s.take('metadata_candidate','toy')
            with self.assertRaisesRegex(ValueError,'GLOBAL_CAP'):s.take('metadata_candidate','extra')
            with self.assertRaisesRegex(ValueError,'ALREADY_ENTERED'):c.Stage(d,'METADATA',dict(metadata_candidate=1))
    def test_original_NPY_scalar_shape_and_strict_content_identity(self):
        import numpy as np,hashlib
        from pf_common_run_native import array_receipt_matches
        for value in (np.array(.17),np.array('metadata'),np.array([.17,.3])):
            record=dict(shape=list(value.shape),dtype=value.dtype.str,sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
            self.assertTrue(array_receipt_matches(value,record,original_shape=True))
            if value.ndim==0:
                self.assertFalse(array_receipt_matches(value,record,original_shape=False))
            self.assertFalse(array_receipt_matches(value,dict(record,sha256='0'*64),original_shape=True))
            self.assertFalse(array_receipt_matches(value,dict(record,dtype='<i8'),original_shape=True))

    def test_cached_expm_preserves_exact_matvec(self):
        import threading,numpy as np
        from scipy.linalg import expm
        from types import SimpleNamespace
        import pf_common_run_native as native
        ctx=SimpleNamespace(forward_cache={},forward_cache_lock=threading.RLock());calls=[]
        stage=SimpleNamespace(take=lambda *args:calls.append(args));a=np.diag([.3,-.2]).astype(complex)
        x=native.forward_propagator(stage,ctx,0,-.7,a,'toy');y=native.forward_propagator(stage,ctx,0,-.7,a,'toy')
        self.assertTrue(np.array_equal(x,expm(-.5j*(-.7)*a)));self.assertIs(x,y);self.assertEqual(len(calls),1)

if __name__=='__main__':unittest.main()
