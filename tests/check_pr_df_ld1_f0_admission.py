"""Artificial metadata only; opaque saved TV values are copied, never calculated."""
import ast
from copy import deepcopy
from pathlib import Path
import unittest

import admit_pr_df_ld1_f0 as C


def fixture():
    store={}
    def put(ref,value):
        store[(ref['commit'],ref['path'])]=value
    def read(ref):
        value=store[(ref['commit'],ref['path'])]
        for key in ref.get('selector',[]): value=value[key]
        return deepcopy(value)
    assumptions=['artificial conditional grid assumption']
    components=[dict(id='artificial_component', probability_hex='0x1.0p+0')]
    comp_source=dict(commit='a'*40,path='artificial/preparation.json',selector=[])
    put(comp_source,dict(hash='artificial_preparation',components=components))
    inputs=dict(plan=C.PLAN,target_cost_law=C.TARGET,tables=[dict(id='components',kind='component',source=comp_source,
        signature=dict(preparation_hash='artificial_preparation',components=components))])
    h=dict(execution_source=C.EXEC,raw_publication_commit=C.RAW,inputs=C.INPUTS,adopted_plan=C.PLAN,
        status='F0_DIAGNOSTIC_COMPLETE',counts=dict(complete=14),tables=[],rows=[],assumptions=assumptions,
        stage_summary=dict(saved='support changes retained'),conditional_path_scope=dict(saved='conditional only'),remaining_evidence=['actual PCG64 joint unknown'])
    result=dict(status=h['status'],counts=h['counts'],tables=[])
    prior=dict(rows=[]); raw_bounds=dict(rows=[])
    for i in range(14):
        name='components' if i==0 else 'order_m'+str(i-1)
        if i:
            m=i-1;pid='artificial_'+str(m);q=m+1;R=m+2
            source=dict(commit='b'*40,path='artificial/order'+str(m)+'.json',selector=[])
            signature=dict(m=m,candidate_id=pid,q_PF=q,R_outer=R,event_count='saved_count')
            put(source,deepcopy(signature))
            inputs['tables'].append(dict(id=name,kind='order',source=source,signature=signature,
                row=dict(m=m,axes=['X','Y'],physical_id=pid,q=q,R_outer=R,saved_event_positions='saved_count')))
        inp=inputs['tables'][i]
        diag_ref=dict(commit=C.RAW,path=C.D_DIR+'run01/'+name+'.json',selector=[])
        table=dict(id=name,input=C.at(C.INPUTS,'tables',i),diagnostic=diag_ref)
        h['tables'].append(table)
        diagnostic=dict(id=name,kind=inp['kind'],source=inp['source'],status='DIAGNOSTIC_COMPLETE',map=C.MAP,
            PRNG_independence_proved=False,physical_law_distance=None,qualified_for_compiled_price=False,
            TV={'opaque_saved_exact':'no numeric conversion'},triangle_bound={'saved':'bound'},
            support_changes=[dict(stage='artificial',change='POSITIVE_TO_ZERO')])
        put(diag_ref,diagnostic); result['tables'].append(dict(id=name,status=diagnostic['status'],path=diag_ref['path']))
        if not i: continue
        bound=dict(m=m,physical_id=pid,axes=['X','Y'],event_positions='saved_count',source=source,
            order_table=name+'.json',component_table='components.json',assumptions=assumptions,
            event_TV_upper={'opaque':'saved event bound'},path_TV_upper={'opaque':'saved path bound'},
            actual_PCG64_joint_distance=None,price_oscillation=None,price_shift=None,qualified_E_C=None)
        raw_bounds['rows'].append(bound)
        bound_ref=dict(commit=C.RAW,path=C.D_DIR+'run01/ROUND_BOUNDS.json',selector=['rows',m])
        for axis in ('X','Y'):
            sr=dict(commit='c'*40,path='artificial/law'+str(m)+'.json',selector=[])
            put(sr,dict(definition=dict(model=C.TARGET,A_declaration=dict(candidate_id=pid,q_PF=q,R_outer=R,K=2))))
            old=dict(m=m,axis=axis,physical_id=pid,
                physical_candidate=dict(system='H4',method='PR',formula='2nd',prefix=1,delta=0.1,K=2,id=pid,q=q,R_outer=R),
                structural=dict(compiled_price=False,law_source=sr))
            old.update({k:'artificial_'+k for k in ('configuration_id','total_configuration_id','original_canonical_id','total_candidate_id','alignment','Q3','original_law','total_law')})
            prior['rows'].append(old)
            h['rows'].append(dict(m=m,axis=axis,physical_id=pid,input=table['input'],component_diagnostic=h['tables'][0]['diagnostic'],
                order_diagnostic=diag_ref,conditional_round_bound=bound_ref,expectation_target=C.TARGET,
                actual_PCG64_joint_distance=None,price_correction=None,qualified_E_C=None))
    for ref,value in [(C.HANDOFF,h),(C.INPUTS,inputs),(C.RESULT,result),(C.PRIOR,prior),(C.FACTS,dict(preparation_hash='artificial_preparation'))]:
        put(ref,value)
    put(dict(commit=C.RAW,path=C.D_DIR+'run01/ROUND_BOUNDS.json'),raw_bounds)
    return read,store


class Tests(unittest.TestCase):
    def test_entry_flow_counts_shared_bounds_and_copies_saved_values(self):
        read,store=fixture();out=C.build(None,read)
        self.assertEqual(out['status'],'SAVED_DIAGNOSTICS_CONNECTED')
        for key,n in [('diagnostic_tables',14),('round_bound_pairs',13),('candidate_axes',26)]:
            self.assertEqual(out['counts'][key],dict(requested=n,accepted=n,held=0))
        self.assertEqual(out['rows'][0]['conditional_bound'],out['rows'][1]['conditional_bound'])
        self.assertEqual(out['tables'][0]['saved_summary']['TV'],{'opaque_saved_exact':'no numeric conversion'})
        self.assertEqual(out['round_bounds'][12]['saved_bound']['path_TV_upper'],{'opaque':'saved path bound'})
        self.assertIsNone(out['qualified_E_C']);self.assertEqual(out['price_provider_records_created'],0)
        ref=out['tables'][0]['support_changes'];self.assertEqual(read(ref)[0]['change'],'POSITIVE_TO_ZERO')

    def test_wrong_axis_physical_or_law_is_held_not_repaired(self):
        for key,value in [('physical_id','other'),('expectation_target','other_law')]:
            read,store=fixture();store[(C.HANDOFF['commit'],C.HANDOFF['path'])]['rows'][0][key]=value
            out=C.build(None,read)
            self.assertEqual(out['counts']['candidate_axes']['held'],1)
            self.assertEqual(out['rows'][0]['status'],'HELD')
        read,store=fixture();store[('c'*40,'artificial/law0.json')]['definition']['A_declaration']['R_outer']=999
        self.assertEqual(C.build(None,read)['counts']['candidate_axes']['held'],2)

    def test_table_source_mismatch_holds_shared_dependents(self):
        read,store=fixture()
        key=(C.RAW,C.D_DIR+'run01/order_m4.json');store[key]['source']['commit']='d'*40
        out=C.build(None,read)
        self.assertEqual(out['counts']['diagnostic_tables']['held'],1)
        self.assertEqual(out['counts']['round_bound_pairs']['held'],1)
        self.assertEqual(out['counts']['candidate_axes']['held'],2)

    def test_unproved_prng_and_price_unknowns_cannot_be_promoted(self):
        read,store=fixture();store[(C.RAW,C.D_DIR+'run01/components.json')]['PRNG_independence_proved']=True
        self.assertEqual(C.build(None,read)['counts']['candidate_axes']['held'],26)
        read,store=fixture();store[(C.RAW,C.D_DIR+'run01/ROUND_BOUNDS.json')]['rows'][0]['price_shift']=0
        self.assertEqual(C.build(None,read)['counts']['round_bound_pairs']['held'],1)

    def test_no_scientific_engine_or_probability_arithmetic(self):
        tree=ast.parse(Path(C.__file__).read_text())
        calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
        self.assertFalse(calls & {'fit','predict','mean','sample_requests','transpile','compile','cost','diagnose','Fraction','float'})
        # Path / 'ADMISSION.json' in main is a filesystem operation.
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('check_table','check_round','check_axis','build'):
                self.assertFalse(any(isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Div,ast.Pow)) for n in ast.walk(node)))


if __name__=='__main__':unittest.main()
