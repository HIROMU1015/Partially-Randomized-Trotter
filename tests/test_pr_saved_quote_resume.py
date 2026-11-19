"""Synthetic saved-quote recovery; no H4 worker/science."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
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

    def test_df_inventory_rejects_pauli_envelopes_before_fixed_validator(self):
        spec=importlib.util.spec_from_file_location('_typed_df_service',ROOT/'scripts/pr_candidate_services.py')
        service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
        source=dict(commit='1'*40,path='saved-packet.json')
        def forbidden():
            raise AssertionError('A wrong-method packet reached DF validation')
        for packet in (dict(packets=[dict(rounds=[])]),
                       dict(method='PR_PAULI',configuration_source={},means=[]),
                       dict(schema='PAULI_SIGNAL_PACKET_V1',configuration_source={},means=[]),
                       dict(configuration_source={},means=[None])):
            with self.subTest(packet=packet),tempfile.TemporaryDirectory() as cache:
                entry=SimpleNamespace(reader=SimpleNamespace(read=lambda loc:packet),saved_index=forbidden)
                with self.assertRaisesRegex(ValueError,'DF_SAVED_BUNDLE_EXPECTED'):
                    service.validated_packet_index(entry,source,cache,'fixed')
                self.assertEqual(list(Path(cache).iterdir()),[])
        entry=SimpleNamespace(reader=SimpleNamespace(read=lambda loc:dict(
            configuration_source=source,means=[dict(source=source)])),saved_index=lambda:{'physical':'original'})
        self.assertEqual(service.validated_packet_index(entry,source,None,'fixed'),{'physical':'original'})

    def test_df_alias_reuses_only_strictly_matching_scientific_records(self):
        spec=importlib.util.spec_from_file_location('_scoped_df_service',ROOT/'scripts/pr_candidate_services.py')
        service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)
        base=dict(B_configuration_id='A',B_candidate_id='A0',B_identity={'kappa':1},
                  B_law_binding={'configuration_id':'A'},signal={'raw':{'real':0.5}},
                  numerical={'gate_pass':True},reference_source={'commit':'fixed'})
        alias=dict(base,B_configuration_id='B',B_candidate_id='B0',B_identity={'kappa':2},
                   B_law_binding={'configuration_id':'B'},reuse={'source':'A'})

        for field, changed in ((None,None),('signal',{'raw':{'real':0.6}}),
                               ('numerical',{'gate_pass':False}),('reference_source',{'commit':'other'})):
            with self.subTest(field=field):
                docs={'a':base,'b':alias if field is None else dict(alias,**{field:changed})}
                calls=[]
                class Entry:
                    saved=['packet1','packet2']
                    _index=None
                    runtime=SimpleNamespace(jobs=lambda configuration:[dict(key=key,m=m)
                                             for m,key in enumerate(configuration)])
                    reader=SimpleNamespace(read=lambda loc: dict(configuration_source={},means=[])
                        if loc in ('packet1','packet2') else docs[loc])

                    def physical_key(self,job):return job['key']

                    def saved_index(self):
                        calls.append(self.saved[0])
                        return {'own':{'source':'a'},'shared':{'source':'a' if self.saved[0]=='packet1' else 'b'}}

                    def plan(self,configuration):
                        rows=[dict(m=m,decision='REUSE',saved=self._index[key]) if key in self._index
                              else dict(m=m,decision='REQUEST',request={'fixture':True})
                              for m,key in enumerate(configuration)]
                        return dict(rounds=rows,counts={'REUSE':sum(r['decision']=='REUSE' for r in rows),
                                                       'REQUEST':sum(r['decision']=='REQUEST' for r in rows),'UNSUPPORTED':0})

                entry=Entry()
                configuration=['own','shared','unobserved']
                conflicts=service.scope_df_saved_index(entry,configuration)
                self.assertEqual(entry.saved,['packet1','packet2'])
                self.assertEqual(calls,['packet1','packet2'])
                self.assertEqual(entry._index['own'],{'source':'a'})
                with tempfile.TemporaryDirectory() as tmp:
                    def locator(path):
                        return dict(commit='1'*40,path=path)

                    class FixedEntry(Entry):
                        def saved_index(self):
                            name=self.saved[0]['path']
                            calls.append(name)
                            return {'own':{'source':'a'},'shared':{'source':'a' if name=='packet1' else 'b'}}

                    cached=FixedEntry()
                    cached.reader=SimpleNamespace(read=lambda loc: docs[loc] if isinstance(loc,str) else
                        dict(configuration_source=locator('configuration'),means=[dict(source=locator('raw'))]))
                    cached.saved=[locator('packet1')]
                    calls.clear()
                    self.assertFalse(service.scope_df_saved_index(cached,configuration,cache_dir=tmp,scope='fixed'))
                    cached.saved.append(locator('packet2'))
                    self.assertEqual(service.scope_df_saved_index(cached,configuration,cache_dir=tmp,scope='fixed'),conflicts)
                    self.assertEqual(cached._index,entry._index)
                    self.assertEqual(calls,['packet1','packet2'])
                    self.assertEqual(service.scope_df_saved_index(cached,configuration,cache_dir=tmp,scope='fixed'),conflicts)
                    self.assertEqual(calls,['packet1','packet2'])
                    service.scope_df_saved_index(cached,configuration,cache_dir=tmp,scope='changed-descriptor')
                    self.assertEqual(calls,['packet1','packet2','packet1','packet2'])
                if field is None:
                    self.assertFalse(conflicts)
                    self.assertEqual(entry._index['shared'],{'source':'a'})
                else:
                    self.assertEqual(conflicts,{1:dict(m=1,physical_key='shared',differing_fields=[field],
                                                       sources=[{'source':'a'},{'source':'b'}])})
                    self.assertNotIn('shared',entry._index)
                    service.bind_df_conflicts(entry,conflicts)
                    plan=entry.plan(configuration)
                    self.assertEqual([r['decision'] for r in plan['rounds']],['REUSE','UNSUPPORTED','REQUEST'])
                    self.assertNotIn('request',plan['rounds'][1])
                    self.assertEqual(plan['counts'],{'REUSE':1,'UNSUPPORTED':1,'REQUEST':1})
                    self.assertEqual(plan['saved_signal_conflicts'],[conflicts[1]])
                    self.assertFalse(service.scope_df_saved_index(Entry(),['own']))


if __name__=='__main__':unittest.main(verbosity=2)
