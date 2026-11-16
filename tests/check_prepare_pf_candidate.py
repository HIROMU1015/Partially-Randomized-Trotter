"""Finite selected-provider entry tests; artificial inputs, no H4 science."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import prepare_pf_candidate as D
import run_pf_m0_prices as M

class Entry(unittest.TestCase):
    def test_real_canonical_method_is_pf_with_route_in_split(self):
        for original,route in [('FULL','FULL_PF'),('B0I','TRUNCATED_PF')]:
            cfg={'spec':{'method':'PF','split':{'original_PF_method':original}}}
            self.assertEqual(D.comparison_method(cfg),route)
        with self.assertRaises(ValueError):D.comparison_method({'spec':{'method':'PR'}})

    def test_native_selection_preserves_full_vs_b0i(self):
        for p,original,method in [(10,'FULL','FULL'),(0,'B0I','B0I'),(8,'B0I','B0I')]:
            cfg={'spec':{'method':'PF','L_D':p,'split':{'original_PF_method':original}}}
            physical=dict(system='H4',method=method,prefix=p,formula='2nd',T=3.2,q=16,delta=.2,id='ARTIFICIAL')
            self.assertEqual(D.native_row(cfg,physical),dict(physical,R_outer=None,K=None))
            with self.assertRaises(ValueError):D.native_row(cfg,dict(physical,method='B0I' if p==10 else 'FULL'))

    def test_new_prefix_order_and_historical_hash_gate(self):
        for p in (0,1,8,9):
            prep=SimpleNamespace(preparation_hash='OWN_'+str(p),deterministic_fragment_indices=tuple(range(p)))
            job=dict(route='TRUNCATED_PF',preparation_hash=None,prefix_specific_preparation=True)
            M.check_preparation(prep,job,p)
            with self.assertRaises(ValueError):M.check_preparation(prep,dict(job,route='FULL_PF'),p)
        prep=SimpleNamespace(preparation_hash='OLD',deterministic_fragment_indices=tuple(range(7)))
        M.check_preparation(prep,dict(route='TRUNCATED_PF',preparation_hash='OLD'),7)
        with self.assertRaises(ValueError):M.check_preparation(prep,dict(route='TRUNCATED_PF',preparation_hash='OTHER'),7)
        with self.assertRaises(ValueError):M.check_preparation(prep,dict(route='TRUNCATED_PF',preparation_hash=None,prefix_specific_preparation=True),7)
    def test_scope_compile_preflight_only_requested_missing_cells(self):
        from trottertracks import pf_evidence_preparation as E
        value=dict(configuration={'id':'SYNTHETIC'},direct_prices=[dict(m=0,axis=a) for a in ('X','Y')],new_compile_cells=[dict(m=5)],prepared={'pending':{'requests':[dict(m=m,physical_candidate={'id':str(m)}) for m in range(13)]}})
        observed=[]
        def duplicate(rounds,selected):
            observed.extend(selected)
            return dict(rows=[dict(x,action='NEW_COMPILE_MISSING_CONFIRMED') for x in selected])
        with TemporaryDirectory() as tmp,patch.object(D,'ROOT',ROOT),patch.object(D,'head',return_value='HEAD'),patch.object(M,'duplicate_check',side_effect=duplicate):
            D.preflight(value,Path(tmp))
        self.assertEqual([(x['m'],x['axis']) for x in observed],[(5,'X'),(5,'Y')])
    def test_registry_reuse_does_not_reenter_science(self):
        with TemporaryDirectory(dir=ROOT,prefix='.pf_fixture_') as tmp:
            root=Path(tmp);cid='ARTIFICIAL';registry=root/'registry.json'
            D.write(registry,dict(schema='D_PF_EVIDENCE_REGISTRY_V1',configurations={cid:dict(signal_packet={'saved':1},prices={'saved':2},q3_inputs={'saved':3})}))
            request=root/'request.json';out=root/'response.json'
            D.write(request,dict(selected=[dict(configuration_body={'id':cid,'spec':{'method':'FULL_PF'}})],output_root=str(root/'outputs'),registry_file=str(registry)))
            with patch.object(sys,'argv',['prepare','--request',str(request),'--output',str(out),'--mode','complete']),patch.object(D,'complete',side_effect=AssertionError('No repeat science')):
                self.assertEqual(D.main(),0)
            self.assertEqual(D.load(out)['items'][0]['status'],'REUSED_COMPLETED_REGISTRY')

if __name__=='__main__':unittest.main(verbosity=2)
