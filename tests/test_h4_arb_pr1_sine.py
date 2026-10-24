"""PR1 sine artificial tests; never propagate a saved H4 state."""
import ast
import copy
from fractions import Fraction as Q
import math
from pathlib import Path
import runpy
import unittest

ROOT=Path(__file__).resolve().parents[1]


class AuthorityScope(ast.NodeTransformer):
    def visit_Constant(self,node):
        if isinstance(node.value,str):
            if node.value in ('cosine','sine'):node.value={'cosine':'sine','sine':'cosine'}[node.value]
            else:node.value=(node.value.replace('run_h4_arb_pr0_cosine.py','run_h4_arb_pr1_sine.py')
                .replace('H4_PR0_COSINE','H4_PR1_SINE').replace('PR0_cosine','PR1_sine')
                .replace("r['native_witness']","r['controller']()['native_witness']"))
        return node

    def visit_Assign(self,node):
        node=self.generic_visit(node)
        if any(isinstance(t,ast.Name) and t.id=='runner' for t in node.targets):
            return [node,ast.parse("runner.update(runner['controller']())").body[0]]
        return node

    def visit_ClassDef(self,node):
        node=self.generic_visit(node)
        omitted=('test_shared_selected_kernel_same_as_guarded_artificial_interface',
            'test_parallel_artificial_pair_native_exits_common_norm_and_finite_scope')
        node.body=[n for n in node.body if not isinstance(n,ast.FunctionDef) or n.name not in omitted]
        return node


# Eleven existing authority/schedule/B-scale/native tests, with sine binding.
tree=AuthorityScope().visit(ast.parse((ROOT/'tests/test_h4_arb_pr0_cosine.py').read_text()))
tree.body=[n for n in tree.body if not isinstance(n,ast.If)]
exec(compile(ast.fix_missing_locations(tree),'<reused PR production authority tests, sine>','exec'),globals())
F1=runpy.run_path(str(ROOT/'tests/test_h4_arb_pr1_cosine.py'))
F=F1['F'];N=F['N'];backend=B._backend()


# Independent dense-matrix C route: select its imaginary output, retaining its
# 256bit matrices and original operands, rather than the vector kernel under test.
tree=ast.parse((ROOT/'tests/test_h4_arb_pr0_preparation.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dense_selected')
node.name='dense_imag'
for item in ast.walk(node):
    if (isinstance(item,ast.Attribute) and item.attr=='real' and isinstance(item.value,ast.Call)
            and isinstance(item.value.func,ast.Attribute) and item.value.func.attr=='signal'):item.attr='imag'
oracle_env=dict(F)
exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<C dense imag oracle>','exec'),oracle_env)
dense_imag=oracle_env['dense_imag']


def fixture(n=2):
    model,join,_=F1['saved_pattern_fixture'](n)
    return model,join,A.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_SINE')


def phase_fixture(phi=.25):
    model,join,_=F['fixture'](orders=(0,),q=1,r=1,blocks=False,scalar=(phi,),
        components=[dict(label='IZ',sign=1,coefficient_abs=.2)],angles=(0.,0.,0.))
    return model,join,A.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_SINE')


class SineEnclosureTests(unittest.TestCase):
    def test_two_three_qubit_order2_imag_contains_independent_dense_C(self):
        for n in (2,3):
            model,join,plan=fixture(n);raw=F['raw'](n);reference=dense_imag(model,join,raw)
            for bits in (64,96,128):
                signal,r=A.P._selected_unitary_enclosure(plan,raw,bits)
                self.assertTrue(signal.contains(reference),(n,bits,signal,reference))
                self.assertEqual(r['axis'],'sine');self.assertFalse(r['cosine_evaluated'])
                self.assertEqual(r['product_applications'],2);self.assertFalse(r['PR_law_mean_used'])

    def test_analytic_pure_phase_is_positive_imag_not_real_or_negative_imag(self):
        raw=F['raw'](2)
        for phi in (.25,-.25):
            _,_,plan=phase_fixture(phi);signal,r=A.P._selected_unitary_enclosure(plan,raw,64)
            with backend.ctx.workprec(256):
                a=backend.real(Q(phi));expected=a.sin()
                self.assertTrue(signal.contains(expected));self.assertFalse(signal.overlaps(-expected))
                self.assertFalse(signal.overlaps(a.cos()))
            self.assertEqual(r['real_H4_state_actions'],0)

    def test_native_Sdg_pair_positive_imag_common_norm_scale_and_native_reap(self):
        model,join,_=F['fixture'](orders=(0,),q=1,r=1,blocks=False,scalar=(.25,),components=[dict(label='IZ',sign=1,coefficient_abs=.2)])
        plan=A.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_SINE');raw=F['raw'](2)
        def h():return [N.gate('rz',2,theta=math.pi/2),N.gate('sx',2),N.gate('rz',2,theta=math.pi/2)]
        gates=[*h(),N.gate('rz',0,theta=.125),N.gate('cx',2,0),N.gate('rz',0,theta=-.125),N.gate('cx',2,0),
            N.gate('rz',2,theta=.25),N.gate('rz',2,theta=-math.pi/2),*h()]
        program=N.program(3,gates,phase=.125,axis='sine')
        with tempfile.TemporaryDirectory() as folder:
            ns=Path(folder);pair=A.action_pair(ns,plan,program,raw,64,'TOY_SOURCE',C.CallBudget(ns,A.LIMITS))
            target,compiled=pair['target']['record'],pair['compiled']['record']
            self.assertEqual(target['raw_norm_exact'],compiled['raw_norm_exact'])
            rho=A.reduce_pair(target,compiled,bits=64,budget_exact='1/1000000',scale_exact='3/2')
            self.assertEqual(rho['budget_relation'],'UPPER_WITHIN_BUDGET');self.assertTrue(rho['B_applied_exactly_once'])
            for role in ('target','compiled'):
                native=C.read(ns/(role+'_64_native_exit.json'))
                self.assertEqual(native['exit_code'],0);self.assertTrue(native['native_wait'] and native['reaped'])
            wrong=copy.deepcopy(program);wrong['instructions'][8]['params_exact']=[str(Q(math.pi/2))]
            _,bad=B._native_unitary_enclosure(wrong,raw,64)
            self.assertEqual(A.reduce_pair(target,bad,bits=64,budget_exact='1/1000000',scale_exact='3/2')['budget_relation'],'LOWER_EXCEEDS_BUDGET')

    def test_order2_sign_phase_order_corruption_rejected(self):
        for kind in ('sign','phase','products','rotation','angle','position'):
            model,join,_=fixture();event=join['events'][30]
            if kind=='sign':event['application_sequence'][1]['coefficient_sign']=1
            if kind=='phase':event['phase']['real']=1.
            if kind=='products':event['product_component_ids'].reverse()
            if kind=='rotation':event['application_sequence'][2]['coefficient_sign']=1
            if kind=='angle':event['rotation_angle']=.4
            if kind=='position':event['event_position']=29
            with self.subTest(kind=kind),self.assertRaises(ValueError):A.build_plan(model,join,domain='ARTIFICIAL',fixture_id='TOY_SINE')

    def test_common_exact_norm_raw_unchanged_zero_norm_rejected(self):
        _,_,plan=phase_fixture();raw=F['raw'](2);before=copy.deepcopy(raw)
        a,ra=A.P._selected_unitary_enclosure(plan,raw,64)
        b,rb=A.P._selected_unitary_enclosure(plan,[[2*x,2*y] for x,y in raw],64)
        self.assertTrue(a.overlaps(b));self.assertEqual(Q(rb['raw_norm_exact']),4*Q(ra['raw_norm_exact']))
        self.assertEqual(raw,before);self.assertFalse(ra['state_renormalized'])
        with self.assertRaisesRegex(ValueError,'POSITIVE_RAW_NORM'):A.P._selected_unitary_enclosure(plan,[[0.,0.]]*4,64)

    def test_sine_budget_axis_trajectory_and_old_artificial_guards(self):
        self.assertNotEqual(A.GUARD,A.CO.GUARD)
        for key,value in [('rho_budget_exact',A.CO.GUARD),('axes',['cosine']),('target_key','PR1_cosine'),('trajectory_id',A.P0.TID),('cases',['pauli_jw_PF'])]:
            c=contract();c[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):A.validate_contract(c)
        _,_,plan=phase_fixture();plan['domain']='SAVED_PR1_SINE_NO_EXECUTION'
        with self.assertRaises(ValueError):A.P._target_artificial(plan,F['raw'](2),64,'TOY_SINE')

    def test_only_final_dot_component_changes_in_unitary_numeric_AST(self):
        tree=ast.parse((ROOT/'src/trotterlib/h4_arb_pr0_preparation.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_selected_unitary_enclosure')
        baseline=A.CO.PlanScope().visit(copy.deepcopy(node));adapted=A.SineScope().visit(copy.deepcopy(baseline))
        # Revert the documented scope strings and sole component projection.
        for n in ast.walk(adapted):
            if isinstance(n,ast.Constant) and isinstance(n.value,str):
                n.value=n.value.replace('SAVED_PR1_SINE_NO_EXECUTION','SAVED_PR1_NO_EXECUTION').replace('SINE','COSINE').replace('sine','cosine')
            if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='dot' and n.attr=='imag':n.attr='real'
        self.assertEqual(ast.dump(baseline),ast.dump(adapted))


if __name__=='__main__':unittest.main(verbosity=2)
