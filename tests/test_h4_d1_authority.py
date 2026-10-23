"""Artificial D1 scope, nonrefundable budget and one-shot authority tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('d1_private');pkg.__path__=[str(ROOT/'src/trotterlib')]
sys.modules['d1_private']=pkg
from d1_private import h4_d1_control as D
C=D.C

def active():
    c=C.read(D.PREP/'contract_DRAFT.json')
    c.update(kind='H4_D1_ACTIVE_ONE_SHOT_V1',execution_authorized=True,grant_valid=True,
        dispatch_enabled=True,call_limits=D.LIMITS,maximum_scope=[v['id'] for v in c['variants']],
        mandatory_STOP=True,automatic_retry=False,D2_D3_authorized=False,N_G_authorized=False,
        science_SOURCE='TOY_SOURCE')
    return c

class AuthorityTests(unittest.TestCase):
    def test_draft_cannot_dispatch(self):
        with self.assertRaises(ValueError):D.scope(C.read(D.PREP/'contract_DRAFT.json'))

    def test_scope_cannot_expand(self):
        c=active();D.scope(c)
        for k,value in [('execution_authorized',False),('maximum_cost_transpile_calls',10),
            ('D2_D3_authorized',True),('N_G_authorized',True),('automatic_retry',True)]:
            b=copy.deepcopy(c);b[k]=value
            with self.subTest(key=k),self.assertRaises(ValueError):D.scope(b)
        c['fixed']['q']=9
        with self.assertRaises(ValueError):D.scope(c)

    def test_missing_grant_rejects_before_science_import(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)):
            with self.assertRaises(FileNotFoundError):D.authority()
            self.assertNotIn('qiskit',sys.modules)

    def test_budget_is_nonrefundable_and_fixed(self):
        with tempfile.TemporaryDirectory() as td:
            b=C.CallBudget(td,dict(cost_transpile=1));b.take('cost_transpile')
            with self.assertRaises(ValueError):b.take('cost_transpile')
            with self.assertRaises(ValueError):b.take('sampling')
            self.assertEqual(C.read(Path(td)/'call_ledger.json')['used']['cost_transpile'],1)

    def test_bound_grant_consumed_once_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td,patch.object(D,'BASE',Path(td)),patch.object(D,'verify_seal'):
            ns=Path(td)/'run01';ns.mkdir();(Path(td)/'approval.txt').write_text('TOY_APPROVAL')
            c=active();s=dict(SOURCE='TOY_SOURCE',approval_sha256=C.sha(Path(td)/'approval.txt'))
            C.save(Path(td)/'seal.json',s);c.update(seal_sha256=C.sha(Path(td)/'seal.json'),approval_sha256=s['approval_sha256'])
            C.save(Path(td)/'active_contract.json',c)
            g=dict(kind='H4_D1_SCIENCE_ONE_SHOT',SOURCE='TOY_SOURCE',namespace=str(ns),
                contract_sha256=C.sha(Path(td)/'active_contract.json'),seal_sha256=C.sha(Path(td)/'seal.json'),
                approval_sha256=s['approval_sha256'],call_limits=D.LIMITS,one_shot=True,automatic_retry=False)
            g['grant_id']=C.digest(g);C.save(ns/'grant.json',g)
            D.authority(consume=True)
            with self.assertRaises(ValueError):D.authority(consume=True)
            (Path(td)/'approval.txt').write_text('CHANGED')
            with self.assertRaises(ValueError):D.authority()

if __name__=='__main__':
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuthorityTests))
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(dict(tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),scope='ARTIFICIAL_AUTHORITY_ONLY'),indent=2)+'\n')
    sys.exit(0 if r.wasSuccessful() else 1)
