"""Synthetic saved-quote recovery; no H4 worker/science."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import unittest
import importlib.util
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_method_optimizer as O

class SavedQuotes(unittest.TestCase):
    def test_own_configuration_and_all_saved_totals_required(self):
        for bad in(None,'total','identity','partial_axes','conflict'):
            with self.subTest(bad=bad):
                config=dict(id='ARTIFICIAL_VARIANT',spec=dict(kappa='17'))
                c=dict(name='own',family='PR',method='PR_PAULI',key=dict(L_D=5),configuration_id=config['id'],
                    generated=False,prices=None,price_status='CONFLICTING_SAVED_UNIT_PRICES'if bad=='conflict'else'NO_SAVED_26_AXIS_PRICE',needs=['PRICE_26AXIS'],ready=False)
                p=dict(configuration_id=config['id'],configuration_source={'commit':'fixed','path':'own.json'},price_axes=[],N_X=[2]*13,N_Y=[3]*13,conditional_cost=130 if bad!='total'else 131)
                problem=dict(price_rules=dict(policy='MODEL',implementation_source='fixed'))
                def quotes(root,pr,request,output):
                    req=json.loads(request.read_text());items=[]
                    for item in req['items']:
                        self.assertEqual(item['configuration'],config)
                        rows=[dict(m=m,axis=a,unit_price_RZ=2,classification='MODEL_REFERENCE')for m in range(13)for a in O.AXES]
                        if bad=='partial_axes':rows.pop()
                        items.append(dict(item_id='own',registry='PAULI_LD5',configuration_id='OTHER'if bad=='identity'else config['id'],complete=True,status='QUOTED',rows=rows))
                    output.write_text(json.dumps(dict(registry={'PAULI_LD5':dict(configuration_id='ARTIFICIAL_BASE')},items=items)))
                with patch.object(O.Reader,'read',return_value=config),patch.object(O,'run_quotes',side_effect=quotes):
                    if bad=='identity':
                        with self.assertRaisesRegex(ValueError,'ANOTHER_CONFIGURATION'):O.quote_prices(ROOT,problem,[c],dict(candidates=[p]))
                    else:
                        O.quote_prices(ROOT,problem,[c],dict(candidates=[p]))
                        self.assertEqual(c['ready'],bad is None)
                        self.assertEqual(c['prices']is not None,bad is None)

    def test_df_conflict_is_scoped_but_requested_record_equality_stays_strict(self):
        spec=importlib.util.spec_from_file_location('_scoped_df_service',ROOT/'scripts/pr_candidate_services.py')
        service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
        for wanted in('own','conflict'):
            with self.subTest(wanted=wanted):
                docs={'a':{'value':1},'b':{'value':2}};calls=[]
                class Entry:
                    saved=['packet1','packet2'];_index=None
                    runtime=SimpleNamespace(jobs=lambda config:[{'key':wanted}])
                    reader=SimpleNamespace(read=lambda loc:docs[loc])
                    def physical_key(self,job):return job['key']
                    def saved_index(self):
                        calls.append(self.saved[0]);return {'own':{'source':'a'},'conflict':{'source':'a'if self.saved[0]=='packet1'else'b'}}
                entry=Entry()
                if wanted=='conflict':
                    with self.assertRaisesRegex(ValueError,'Conflicting saved means'):service.scope_df_saved_index(entry,{})
                    self.assertIsNone(entry._index)
                else:
                    service.scope_df_saved_index(entry,{})
                    self.assertEqual(entry._index,{'own':{'source':'a'}})
                self.assertEqual(entry.saved,['packet1','packet2']);self.assertEqual(calls,['packet1','packet2'])

if __name__=='__main__':unittest.main(verbosity=2)
