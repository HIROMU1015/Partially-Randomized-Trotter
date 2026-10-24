"""Small artificial PR1-signature systems; real H4 propagation never called."""
import ast
import copy
from fractions import Fraction as Q
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ReusedProductionScope(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            node.value=(node.value.replace('run_h4_arb_pr0_cosine.py','run_h4_arb_pr1_cosine.py')
                .replace('H4_PR0_COSINE','H4_PR1_COSINE').replace('PR0_cosine','PR1_cosine')
                .replace("r['native_witness']","r['controller']()['native_witness']"))
        return node

    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if any(isinstance(t,ast.Name) and t.id=='runner' for t in node.targets):
            extra=ast.parse("runner.update(runner['controller']())").body[0]
            return [node,extra]
        return node


# Reuse the 13 authority/pair/interval tests, scoped to the new runner/contract.
tree=ReusedProductionScope().visit(ast.parse((ROOT/'tests/test_h4_arb_pr0_cosine.py').read_text()))
tree.body=[n for n in tree.body if not isinstance(n,ast.If)]
exec(compile(ast.fix_missing_locations(tree),'<reused PR0 production tests, PR1 scope>','exec'),globals())

F=runpy.run_path(str(ROOT/'tests/test_h4_arb_pr0_preparation.py'))
P=A.P
backend=B._backend()


def saved_pattern_fixture(n=2):
    """43(+),4(-),72(-), with the real noncommutation signature on 2/3 qubits."""
    labels=('IX','IZ','YI') if n==2 else ('IYX','IIZ','XYI')
    components=[dict(label='I'*n,sign=1,coefficient_abs=0.) for _ in range(73)]
    for index,label,sign in zip((43,4,72),labels,(1,-1,-1)):
        components[index]=dict(label=label,sign=sign,coefficient_abs=.2)
    orders=(0,)*30+(2,0)
    model,join,_=F['fixture'](n,orders=orders,q=8,r=4,components=components,
        scalar=(.037,),angles=(.08880799816476508,.029672025283184623,.01780655959824602))
    for i,e in enumerate(join['events']):
        products=['43','4'] if i==30 else [];ids=products+['72']
        e.update(product_component_ids=products,rotation_component_id='72',selected_component_ids=ids)
        e['application_sequence']=[dict(application_index=j,component_id=cid,role='product' if j<len(products) else 'rotation',
            coefficient_abs=components[int(cid)]['coefficient_abs'],coefficient_sign=components[int(cid)]['sign'],
            is_identity=False,basis_id=None,basis_hash=None,df_fragment_id=None,basis_change_operations=[]) for j,cid in enumerate(ids)]
    plan=P.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_PR1_ORDER2')
    return model,join,plan


class PR1Order2ArbTests(unittest.TestCase):
    def test_saved_signature_full_toy_schedule_contains_dense_C_reference(self):
        for n in (2,3):
            model,join,plan=saved_pattern_fixture(n);raw=F['raw'](n)
            A.R1.check_event_semantics(model,join)
            ref=F['dense_selected'](model,join,raw)
            for bits in (64,96,128):
                signal,record=P._selected_unitary_enclosure(plan,raw,bits)
                self.assertTrue(signal.contains(ref),(n,bits,signal,ref))
                self.assertEqual(record['product_applications'],2);self.assertEqual(record['native_scalar_applications'],8)
                self.assertEqual(record['real_H4_state_actions'],0);self.assertFalse(record['normalization_B_applied_to_target'])

    def test_noncommuting_products_negative_signs_rotation_phase_analytic(self):
        # C event = -rotation(-YI) @ (-IZ) @ (+IX), scalar exp(-i/4).
        model,join,_=saved_pattern_fixture(2)
        event=copy.deepcopy(join['events'][30]);event.update(event_position=0,outer_step_index=0,inner_RTE_position=0)
        join['events']=[event];join.update(q=1,r=1)
        case=model['cases']['PR'];case['config'].update(q=1,r=1,T=.1);case['blocks']=[];case['native_scalar_phases']=[.25]
        plan=P.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_PR1_ORDER2');raw=F['raw'](2)
        actual,_=P._selected_unitary_enclosure(plan,raw,64)
        with backend.ctx.workprec(256):
            dim=4;I=backend.identity(dim);theta=backend.real(Q(event['rotation_angle']))
            p43,p4,p72=backend.pauli('IX'),-backend.pauli('IZ'),-backend.pauli('YI')
            rotation=I*theta.cos()+p72*backend.acb(0,-theta.sin())
            correct=-rotation*p4*p43;phase=backend.phase(backend.real('1/4'))
            psi=backend.acb_mat([[backend.complex_ball(z)] for z in raw]);nu=sum((Q(x)**2+Q(y)**2 for x,y in raw),Q(0))
            signal=lambda op:(backend.signal(psi,(op*psi)*phase).real/backend.real(nu))
            self.assertTrue(actual.contains(signal(correct)))
            for wrong in (-correct,-rotation*p43*p4,-rotation*(-p4)*p43,
                    -(I*theta.cos()-p72*backend.acb(0,-theta.sin()))*p4*p43):
                self.assertFalse(actual.overlaps(signal(wrong)))

    def test_saved_metadata_guards_reject_product_sign_phase_order_and_angle_corruption(self):
        for kind in ('sign','phase','products','rotation_sign','position','angle'):
            model,join,_=saved_pattern_fixture();e=join['events'][30]
            if kind=='sign':e['application_sequence'][1]['coefficient_sign']=1
            if kind=='rotation_sign':e['application_sequence'][2]['coefficient_sign']=1
            if kind=='phase':e['phase']['real']=1.
            if kind=='products':e['product_component_ids'].reverse()
            if kind=='position':e['event_position']=29
            if kind=='angle':e['rotation_angle']=.4
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                P.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_PR1_ORDER2')

    def test_real_scope_uses_only_PR1_metadata_and_old_artificial_entry_stays_closed(self):
        model,join,_=saved_pattern_fixture();join.update(trajectory_id=A.TID,trajectory_index=1)
        # A real domain must also have the fixed H4 shape and authenticated join.
        with self.assertRaises(ValueError):P.build_plan(model,join)
        plan=copy.deepcopy(F['fixture']()[2]);plan['domain']='SAVED_PR1_NO_EXECUTION'
        with self.assertRaises(ValueError):P._target_artificial(plan,F['raw'](2),64,'TOY_PR0')

    def test_selected_numeric_ast_diff_is_only_real_scope_label(self):
        tree=ast.parse((ROOT/'src/trotterlib/h4_arb_pr0_preparation.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_selected_unitary_enclosure')
        expected=A.ScopeStrings().visit(copy.deepcopy(node));actual=A.PlanScope().visit(copy.deepcopy(node))
        self.assertEqual(ast.dump(expected),ast.dump(actual))

    def test_profile_rejects_other_index_event_position_and_reversed_product(self):
        _,join,_=saved_pattern_fixture();self.assertTrue(A._pr1_orders(join['events']))
        for variant in ('order0','position','reverse'):
            bad=copy.deepcopy(join['events'])
            if variant=='order0':bad[30]['taylor_order']=0
            if variant=='position':bad[29],bad[30]=bad[30],bad[29]
            if variant=='reverse':bad[30]['product_component_ids'].reverse()
            self.assertFalse(A._pr1_orders(bad))

    def test_PR0_and_PF_or_extra_axis_contracts_cannot_dispatch(self):
        for key,value in [('target_key','PR0_cosine'),('trajectory_id',A.P0.TID),('cases',['pauli_jw_PF']),
                ('axes',['cosine','sine']),('max_state_actions',8)]:
            c=contract();c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):A.validate_contract(c)


if __name__=='__main__':unittest.main(verbosity=2)
