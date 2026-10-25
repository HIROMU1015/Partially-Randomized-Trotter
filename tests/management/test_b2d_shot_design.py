"""Pure artificial scalar checks for the proposed single-complex-signal design.

No research data are opened and no candidate normalization/shot/G is evaluated.
No new production adapter or execution entry point is introduced here.
"""
import ast,math,sys,unittest
from fractions import Fraction as F
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks.pr_dictionary_validation.intervals import scalar_context,rational_interval,endpoints
from trottertracks.pr_dictionary_validation.gates import shot_count_from_enclosure

def extracted(path,names,namespace):
    # Avoid importing trotterlib/__init__ and its unrelated physics dependencies.
    tree=ast.parse((ROOT/path).read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(functions)!=len(names):raise ValueError('Expected scalar definitions changed')
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(path),'exec'),namespace)
    return namespace

class DesignChecks(unittest.TestCase):
    def tearDown(self):
        for module in ('numpy','scipy','qiskit','pyscf','trotterlib'):
            self.assertNotIn(module,sys.modules)

    def test_complex_coordinate_factor_matches_existing_scalar_Hoeffding(self):
        helper=extracted(Path('src/trotterlib/rpe_resource_accounting.py'),
            {'_hoeffding_shots'},dict(math=math))['_hoeffding_shots']
        # Deliberately unrelated toy inputs: no fixed research epsilon or N.
        bound,stat,alpha=F(7,5),F(1,9),F(1,13)
        ctx=scalar_context();b=rational_interval(ctx,bound);s=rational_interval(ctx,stat)
        expr=4*b*b/(s*s)*ctx.ln(rational_interval(ctx,2/alpha))
        expected=shot_count_from_enclosure(*endpoints(expr))
        coordinate=float(stat)/(math.sqrt(2)*float(bound))
        self.assertEqual(helper(coordinate,float(alpha)),expected)
        self.assertEqual(F(2)*bound**2/(stat**2/2),F(4)*bound**2/stat**2)

    def test_exact_large_integer_ceil_and_straddling_rejection(self):
        value=(1<<90)+117
        self.assertEqual(float(value),float(value+1))
        self.assertEqual(shot_count_from_enclosure(F(value)-F(1,7),F(value)),value)
        with self.assertRaisesRegex(ValueError,'SHOT_COUNT_UNRESOLVED'):
            shot_count_from_enclosure(F(value),F(value)+F(1,7))

    def test_saved_budget_definition_keeps_unqualified_values_null(self):
        ns=extracted(Path('src/trottertracks/b2d_evaluator/outer_rte_execution.py'),
            {'budget_metadata'},dict())
        policy=ns['budget_metadata']()
        self.assertEqual(sum(F(str(policy[k])) for k in
            ('bias_plus_numerical_fraction','implementation_fraction','statistical_fraction')),1)
        self.assertEqual(F(str(policy['X_alpha']))+F(str(policy['Y_alpha'])),F(str(policy['alpha'])))
        self.assertEqual(policy['target'],'physical_complex_coherent_signal_absolute_error')
        for k in ('u_num','implementation_gamma','statistical_error','shots'):
            self.assertIsNone(policy[k])
        self.assertEqual(policy['familywise_status'],'UNALLOCATED_NO_SIMULTANEOUS_CLAIM')

    def test_paired_cost_variance_requires_cross_axis_covariance(self):
        xs,ys=(3,5,9),(10,8,4);mx,my=5,7;n=3
        mean=lambda values:F(sum(values),len(values))
        vx=mean(xs);vy=mean(ys)
        sample_var=lambda values:sum((F(v)-mean(values))**2 for v in values)/(n-1)
        cov=sum((F(x)-vx)*(F(y)-vy) for x,y in zip(xs,ys))/(n-1)
        totals=[mx*x+my*y for x,y in zip(xs,ys)]
        self.assertEqual(sample_var(totals)/n,
            (mx*mx*sample_var(xs)+my*my*sample_var(ys)+2*mx*my*cov)/n)
        self.assertNotEqual(sample_var(totals)/n,
            (mx*mx*sample_var(xs)+my*my*sample_var(ys))/n)
        self.assertEqual(len(totals),n)

    def test_shot_weighting_changes_affine_rare_cost_boundary(self):
        # Two arbitrary scalar models; no actual candidate/rare probability.
        pp,pn=F(1,10),F(1,5);MP,MN=11,7;GP,GN=F(30),F(20)
        k=(1-pp)*GP-(1-pn)*GN
        slope=pp*MP/(pn*MN);intercept=k/(pn*MN)
        self.assertNotEqual(slope,pp/pn)
        up=F(17);un=intercept+slope*up
        self.assertEqual(k+pp*MP*up-pn*MN*un,0)

if __name__=='__main__':unittest.main()
