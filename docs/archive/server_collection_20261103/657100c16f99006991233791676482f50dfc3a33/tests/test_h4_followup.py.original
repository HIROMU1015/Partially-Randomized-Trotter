"""Synthetic scope, law, source/input refusal and numerical adapter checks."""
from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import h4_followup_contract as c


class Scope(unittest.TestCase):
    def test_boundary_has_no_native_panel_duplicate(self):
        rows=c.plan()['R']['cells'];self.assertEqual(len(rows),12)
        self.assertEqual(len({x['id'] for x in rows}),12)
        self.assertTrue(all(x['r']==1 or x['q']==16 for x in rows))
        self.assertEqual({x['K'] for x in rows},{2,4})

    def test_fixed_delta_reuse(self):
        rows=c.plan()['T']['times'];self.assertEqual([x['T'] for x in rows],[.4,.8,1.6])
        self.assertTrue(all(x['T']/x['q']==.2 for x in rows))
        self.assertEqual([x['T'] for x in rows if x['reuse']],[.8])
        self.assertEqual(c.plan()['T']['new_compile'],2*2*(4+32))

    def test_rare_exact_support_and_count(self):
        p=c.plan()['B'];self.assertEqual(len(p['groups']),4)
        self.assertEqual({x['stratum'] for x in p['groups']},{'s2_multi','s4'})
        self.assertEqual(p['compile'],2*len(p['groups'])*p['n'])

    def test_policy_panel_actual_rank_rule(self):
        rows=c.panel(4,10,policy=True)
        self.assertEqual([x['prefix'] for x in rows],[8,8,10,10,3])
        self.assertEqual(rows[-1]['K'],4)
        self.assertEqual(c.plan()['FP']['compile'],2*(4+32))
        self.assertFalse(c.plan()['F']['final_rank_supplied'])

    def test_seed_scope_no_alias(self):
        seeds={c.seed(stage,[.8,'fake'],i,'binding') for stage in ('B','T','FP') for i in range(32)}
        self.assertEqual(len(seeds),96)
        self.assertNotEqual(c.seed('T',[.4,'fake'],0,'binding'),c.seed('T',[1.6,'fake'],0,'binding'))
        with self.assertRaises(ValueError):c.seed('R','fake',0,'binding')

    def test_grant_and_cross_stage_refused_before_input(self):
        for stage in ('R','B','T','F','FP'):
            with self.assertRaisesRegex(ValueError,'APPROVAL'):c.verify({},dict(stage=stage,approved_by_user=False),stage)
        with self.assertRaisesRegex(ValueError,'STAGE'):c.verify({},dict(stage='B'),'T')

    def test_config_mutation_refused(self):
        grant=dict(stage='R',approved_by_user=True,one_shot=True,retry=False,next_stage_authorized=False,
            plan_digest=c.digest(c.plan()),config_digest=c.digest({}))
        with self.assertRaisesRegex(ValueError,'PLAN_CONFIG'):c.verify(dict(changed=True),grant,'R')

    def test_input_mutation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input';p.write_text('good')
            config=dict(source_hashes={},inputs=[dict(path=str(p),sha256=c.sha(p),bytes=4,role='fixture')])
            grant=dict(stage='R',approved_by_user=True,one_shot=True,retry=False,next_stage_authorized=False,
                plan_digest=c.digest(c.plan()),config_digest=c.digest(config),SOURCE='0'*40)
            p.write_text('evil')
            with self.assertRaisesRegex(ValueError,'INPUT_CHANGED'):c.verify(config,grant,'R',runtime=False)

    def test_conditional_membership_normalization(self):
        import h6_dba_sampling as sampling
        import numpy as np
        for stratum in ('s2_multi','s4'):
            law=sampling.counts.conditional_counts(16,(.8,.15,.05),stratum)
            self.assertAlmostEqual(math.fsum(law.probabilities),1.)
            orders=sampling.draw(law,np.random.Generator(np.random.PCG64(71)))
            self.assertTrue(sampling.counts.in_stratum(orders.count(2),orders.count(4),stratum))


class Numerical(unittest.TestCase):
    def test_native_paired_compile_adapter_at_new_times_and_conditional_law(self):
        import numpy as np
        import qiskit
        import h4_native_science as h
        import h4_followup_science as s
        from trotterlib.df_hamiltonian import DFHamiltonian,PhysicalSector
        from trotterlib.rte_compiled_cost import compiler_settings_hash
        m=h.provider();one=np.diag(np.repeat([.13,-.12,.03,-.04],2)).astype(complex)
        gs=tuple(np.diag(np.repeat(v,2)).astype(complex) for v in ([.2,.3,-.2,.1],[.3,-.2,.1,.4],[-.2,.1,.3,-.3]))
        ham=DFHamiltonian(.17,one,np.array([.4,-.3,.2]),gs,{})
        sector=PhysicalSector.spin_sector(n_qubits=8,nelec_alpha=2,nelec_beta=2)
        state=np.ones(36,dtype=complex)/6;full=np.zeros(256,dtype=complex);full[sector.basis_indices]=state
        _,qstate=m.checked_basis_bridge(ham,sector,full,state,1e-12)
        cp=dict(basis_gates=['rz','sx','x','cx'],optimization_level=1,seed_transpiler=17)
        compiler=m.CompilerSettings(tuple(cp['basis_gates']),None,None,1,None,None,17,qiskit.__version__)
        config=dict(compiler_options=cp,binding=dict(compiler=compiler_settings_hash(compiler)))
        cell=c.cell('B2',1,2,r=2,K=4);prep=m._prepare(ham,1)
        with tempfile.TemporaryDirectory() as tmp:
            for i,(T,stratum) in enumerate(((.4,None),(1.6,None),(.8,'s2_multi'),(.8,'s4'))):
                directory=Path(tmp)/str(i);directory.mkdir();p=h.accounting_plan();p.update(T=T,cells=[cell])
                p['caps_proposed'].update(trajectory=1,occurrence=2)
                port=m.MatchedPort(Path(tmp),dict(kind='SYNTHETIC_ONLY',plan=p),c.Writer(directory),h.Progress())
                port.ham=ham;port.preps[cell['id']]=prep;port.qstate=qstate
                task=dict(cell=cell,id='synthetic',replica=0,seed=100+i,stratum=stratum)
                row=s.compile_task(port,config,task,directory)
                self.assertEqual(len(row['rows']),2);self.assertEqual(row['control_probes'],4)
                self.assertTrue(all(r['metrics']['RZ']>0 for r in row['rows']))

    def test_projection_preserves_signed_weights_all_fragments(self):
        import numpy as np
        from h4_tol_only_input import projected
        one=np.eye(8,dtype=complex);g=np.eye(8,dtype=complex)
        weights=np.array([.4,-.3]);before=weights.copy()
        post,gs,receipt=projected(one,weights,np.array([g,2*g]),np.zeros((8,8),dtype=complex))
        self.assertTrue(np.array_equal(weights,before));self.assertEqual(len(gs),2)
        self.assertEqual(__import__('decimal').Decimal(receipt['eta_hartree_decimal']),0)
        self.assertFalse(receipt['representation_error_certified'])

    def test_projection_rejects_spin_and_large_change(self):
        import numpy as np
        from h4_tol_only_input import projected
        one=np.eye(8,dtype=complex);g=np.eye(8,dtype=complex);blocks=np.array([g,g]);w=np.array([1.,-1.])
        blocks[0,0,1]=.1
        with self.assertRaisesRegex(ValueError,'SPIN'):projected(one,w,blocks,np.zeros_like(one))
        blocks=np.array([g,g]);blocks[0,0,2]=blocks[0,1,3]=.1
        with self.assertRaisesRegex(ValueError,'BUDGET'):projected(one,w,blocks,np.zeros_like(one))

    def test_time_overlay_leaves_reference_and_changes_phase(self):
        import numpy as np
        import h4_native_science as h
        import h4_followup_science as s
        from test_h4_native_validation import record
        ham=SimpleNamespace(g_matrices=(np.diag([.3,.6]),),lambdas=(.4,),constant=.17)
        for T in (.4,1.6):
            r=record(prefix=0);before=deepcopy(r)
            port=SimpleNamespace(plan=dict(h.accounting_plan(),T=T),ham=ham,reference=complex(.9,.15))
            out=s.overlay(r,port);C=out['scalar_receipt']['C']
            self.assertEqual(out['reference'],r['reference']);self.assertEqual(r,before)
            z=complex(.8,.1)*np.exp(-1j*T*C)
            self.assertAlmostEqual(out['signals']['corrected']['real'],z.real)
            self.assertAlmostEqual(out['signals']['corrected']['imag'],z.imag)
            self.assertEqual(out['cell']['method'],'B0I');self.assertFalse(out['allowance']['certified'])


if __name__=='__main__':unittest.main()
