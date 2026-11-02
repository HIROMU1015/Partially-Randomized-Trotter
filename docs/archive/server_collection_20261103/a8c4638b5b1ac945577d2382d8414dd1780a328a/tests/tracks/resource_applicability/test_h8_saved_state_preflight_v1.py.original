"""Only scalar policies and artificial coefficient matrices; no real inputs."""
import unittest
import numpy as np
from trottertracks.resource_applicability.h48_preflight_contract_v1 import h8_state_plan,phase_resources
from trottertracks.resource_applicability.h8_saved_projection_v1 import accept,policy

class SavedStateTests(unittest.TestCase):
    def test_complete_mock_state_port_no_molecule_or_decomposer(self):
        import tempfile,json
        from pathlib import Path
        from trottertracks.resource_applicability.h8_saved_state_port_v1 import SavedStatePort
        from trottertracks.resource_applicability.ax2b_supplement_records_v1 import AtomicWriter
        from trottertracks.resource_applicability.ax2b_h6_input_generation_watchdog_v1 import InputProgress
        from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);parent=b/'parent';parent.mkdir();out=b/'out';out.mkdir();f=self.fixture(1e-8)
            np.savez(parent/'integrals.npz',one_body=f['one_body'],two_body=f['two_body'],constant=np.array(0.))
            np.savez(parent/'raw_df.npz',weights=f['lambdas'],blocks=f['g_matrices'],correction=f['correction'],truncation=np.array(0.))
            m={'plan':h8_state_plan(),'source_commit':'a'*40,'saved_parent':{'directory':str(parent),'file_hashes':{n:file_hash(parent/n) for n in ('integrals.npz','raw_df.npz')}}}
            writer=AtomicWriter(out,byte_cap=32*2**20);progress=InputProgress(writer,cap=256)
            class Identity:
                def __matmul__(self,v):return v.copy()
            def solver(op,**kwargs):
                x=kwargs['v0'];np.testing.assert_array_equal(op@x,x);return np.array([1.]),x[:,None]
            port=SavedStatePort(m,writer,progress,grant_sha256='b'*64,solver=solver,operator_factory=lambda h,s:Identity())
            port.execute()
            r=json.loads((out/'snapshot_receipt.json').read_text())
            self.assertTrue(r['loader_roundtrip_checked']);self.assertEqual(r['metadata']['sector']['dimension'],4900)
            self.assertEqual(r['calls_attempted']['integral_build'],0);self.assertEqual(r['calls_attempted']['df_decomposition'],0)
            self.assertEqual(r['calls_attempted']['solver_matvec'],2)
    def test_scalar_order_does_not_replace_independent_coefficients(self):
        from trottertracks.resource_applicability.h8_saved_state_port_v1 import scalar_summary,independent_coefficients
        from trottertracks.resource_applicability.h8_saved_projection_v1 import reconstruct,norm
        f=self.fixture(1e-8);c=f['one_body']+f['correction'];g=f['g_matrices'];w=f['lambdas'];post=(g+g.conj().transpose(0,2,1))/2
        d=scalar_summary(f['one_body'],f['two_body'],c,w,g,post);a,b=reconstruct(c,w,g)
        self.assertEqual(d['raw_normal_order_one_body_residual_frobenius'],norm(a-f['one_body']))
        x,y=independent_coefficients(c,w,g)
        self.assertLess(norm(a-x),1e-13);self.assertLess(norm(b-y),1e-13)
    def test_zero_input_reexecution_budget(self):
        p=h8_state_plan();self.assertEqual([p['caps_proposed'][k] for k in ('integral_build','df_decomposition','compile','trajectory')],[0,0,0,0])
        self.assertEqual(p['caps_proposed']['solver_matvec'],10000)
        self.assertFalse(p['H8_full_campaign_authorized'])
    def test_parallel_job_cpu_and_memory_separation(self):
        a,b=phase_resources('H4_COST'),phase_resources('H8_STATE')
        self.assertTrue(set(a['assigned_cpus']).isdisjoint(b['assigned_cpus']))
        self.assertEqual(b['cost_cpu_sets'],[[30]])
        self.assertLessEqual(a['aggregate_rss_bytes']+b['worker_rss_stop_bytes']+b['coordinator_rss_reserve_bytes']+b['aggregate_margin_bytes'],96*2**30)
    def fixture(self,weight):
        one=np.zeros((16,16),dtype=complex);g=np.zeros((2,16,16),dtype=complex);g[0]=np.eye(16)/4
        for a,b in ((0,2),(1,3)):g[0,a,b]=1e-6;g[0,b,a]=-1e-6
        return dict(constant=0.,one_body=one,two_body=np.zeros((16,)*4,dtype=complex),
            lambdas=np.array([weight,0.],dtype='<f8'),g_matrices=g,correction=one.copy(),truncation=0.)
    def test_weighted_gate_and_primary_N16(self):
        arrays,r,_=accept(**self.fixture(1e-8));self.assertEqual(r['policy']['primary_N'],16)
        self.assertLess(float(r['eta_N_hartree_decimal']['16']),9.9e-11)
        np.testing.assert_array_equal(arrays['lambdas'],np.array([1e-8,0.]))
        self.assertGreater(r['fragments'][0]['difference_frobenius'],1e-10)
        self.assertFalse(r['representation_error_certified'])
    def test_reject_material_projection(self):
        with self.assertRaisesRegex(ValueError,'PROJECTION_WEIGHTED_BUDGET'):accept(**self.fixture(1.))

if __name__=='__main__':unittest.main()
