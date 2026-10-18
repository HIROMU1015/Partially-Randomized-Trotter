"""Artificial only: no H4/H6 evidence, scientific imports, draws or transpile."""
import copy
from fractions import Fraction as F
import importlib.util
import itertools
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def source_hashes():
    return {n:m.file_hash(ROOT/n) for n in ['src/trotterlib/pr_theory_compiled.py',
        'src/trotterlib/native_df_gate_model.py','scripts/run_pr_theory_compiled_h46.py']}
spec=importlib.util.spec_from_file_location('prtc',ROOT/'src/trotterlib/pr_theory_compiled.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
spec2=importlib.util.spec_from_file_location('prtc_runner',ROOT/'scripts/run_pr_theory_compiled_h46.py')
runner=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(runner)


def identity():
    x={k:'synthetic-'+k for k in m.IDENTITY_FIELDS}
    x.update(system='H6',prefix=5,q=2,r=2,K=4,formula='2nd',T=.8,P=0,
        control='symmetric_directional',wrapper='full_hadamard',numeric_envelope='VERIFIED',critical_metadata_bound=True)
    return x


def row():
    return dict(theory_identity=identity(),compiled_identity=identity(),model='A',axis='cosine',
        theory_axis='cosine',compiled_axis='cosine',metric='RZ',theory_metric='RZ',compiled_metric='RZ',
        theory_kind='LAW_EXPECTATION',all_positive_orders_included=True,sample_size=8,
        theory_lower='100',theory_upper='110',compiled_mean='80',compiled_observation_id='synthetic-observation',
        lowering_profile=m.counter.PROFILE)


class Alignment(unittest.TestCase):
    def test_exact_ratio_and_interval_no_midpoint(self):
        x=m.compare_row(row());self.assertEqual((x['rho_lower'],x['rho_upper']),('8/11','4/5'))
        self.assertEqual((x['residual_lower'],x['residual_upper']),('-3/8','-1/4'))
    def test_all_critical_identity_mismatches_rejected(self):
        for k in m.IDENTITY_FIELDS:
            x=row();x['compiled_identity'][k]='different'
            with self.subTest(k=k),self.assertRaises(ValueError):m.compare_row(x)
    def test_missing_identity_rejected(self):
        x=row();del x['theory_identity']['profile']
        with self.assertRaisesRegex(ValueError,'MISSING_IDENTITY'):m.compare_row(x)
    def test_axis_mixing_rejected(self):
        x=row();x['compiled_axis']='sine'
        with self.assertRaisesRegex(ValueError,'AXIS_BINDING'):m.compare_row(x)
    def test_metric_mixing_rejected(self):
        x=row();x['theory_metric']='CX'
        with self.assertRaisesRegex(ValueError,'METRIC_BINDING'):m.compare_row(x)
    def test_phase_mixing_rejected(self):
        x=row();x['compiled_identity']['sampling_phase']='confirmation'
        with self.assertRaisesRegex(ValueError,'sampling_phase'):m.compare_row(x)
    def test_profile_mixing_rejected(self):
        x=row();x['compiled_identity']['profile']='another-profile'
        with self.assertRaisesRegex(ValueError,'profile'):m.compare_row(x)
    def test_conditional_is_explicit_and_not_promoted(self):
        x=row();x['theory_identity']['numeric_envelope']='NOT_VERIFIED'
        with self.assertRaises(ValueError):m.compare_row(x)
        self.assertEqual(m.compare_row(x,conditional=True)['alignment'],'CONDITIONALLY_ALIGNED')
    def test_missing_critical_binding_is_not_conditional(self):
        x=row();x['theory_identity']['numeric_envelope']='NOT_VERIFIED';x['theory_identity']['critical_metadata_bound']=False
        with self.assertRaises(ValueError):m.compare_row(x,conditional=True)
    def test_same_events_and_different_events(self):
        x=row();x.update(theory_kind='PAIRED_EVENT_SAMPLE',theory_event_digest='same',compiled_event_digest='same')
        self.assertIn('PAIRED',m.compare_row(x)['interpretation']);x['compiled_event_digest']='different'
        with self.assertRaisesRegex(ValueError,'EVENT'):m.compare_row(x)
    def test_expectation_sample_difference_not_causally_attributed(self):
        self.assertEqual(m.compare_row(row())['attribution'],'UNATTRIBUTED')
        self.assertIn('NOT_COMPILER_ONLY',m.compare_row(row())['interpretation'])
    def test_rare_order_dropping_rejected(self):
        x=row();x['all_positive_orders_included']=False
        with self.assertRaisesRegex(ValueError,'RARE'):m.compare_row(x)
    def test_zero_missing_and_nonfinite_rejected(self):
        for k in ['theory_lower','theory_upper','compiled_mean']:
            for v in [0,None,'nan','inf','-inf',-1,True]:
                x=row();x[k]=v
                with self.subTest(k=k,v=v),self.assertRaises(ValueError):m.compare_row(x)
    def test_depth_and_ft_conversion_unimplemented(self):
        for metric in ['RZ_depth','T','Toffoli']:
            x=row();x['metric']=metric
            with self.assertRaises(ValueError):m.compare_row(x)
    def test_endpoint_reuse_not_double_counted(self):
        x=row();y=copy.deepcopy(x);y['model']='B'
        self.assertEqual(m.unique_observations([x,y]),1)
        y['compiled_mean']=81
        with self.assertRaisesRegex(ValueError,'CONFLICTING'):m.unique_observations([x,y])
    def test_descriptive_statistics_and_duplicate_sample_rejection(self):
        s=m.paired_statistics([('a',1),('b',3)]);self.assertEqual(s['mean'],2);self.assertEqual(s['standard_error'],1)
        self.assertFalse(s['population_mean_certified'])
        with self.assertRaises(ValueError):m.paired_statistics([('a',1),('a',3)])
    def test_shots_are_separate_from_unit_cost(self):
        self.assertEqual(m.weighted_cost({'cosine':2,'sine':3},{'cosine':10,'sine':20}),80)
        a=identity();a.update(epsilon=.05,bias_policy='fixed',accounting_source='saved',costs={'cosine':2,'sine':3},shots={'cosine':10,'sine':20})
        b=copy.deepcopy(a);b.update(costs={'cosine':4,'sine':6},shots={'cosine':5,'sine':10})
        self.assertEqual(m.pf_pr_comparison(a,b),1)
        b['epsilon']=.01
        with self.assertRaisesRegex(ValueError,'PF_PR_CONTEXT'):m.pf_pr_comparison(a,b)


class MixedCounter(unittest.TestCase):
    def setup_parts(self):
        c=m.counter
        b=c.Basis('fake','fake-hash',(c.Gate('h',(0,)),c.Gate('rz',(0,),.3)))
        bases=(b,b)
        factory=lambda block,t:NS(rz=((0,t),),rzz=(),global_phase=.1*t)
        components=(c.Component('fake-z',F(1),b,(0,),1),)
        return bases,factory,components
    def test_mixed_is_prefix_A_tail_B_not_interpolated(self):
        c=m.counter;b,f,cs=self.setup_parts();orders={0:F(3,4),2:F(1,4)}
        args=dict(bases=b,primitive_factory=f,T=.8,q=2,ancilla=1,scalar_phase=-.31,components=cs,orders=orders,r=2,tau=.3)
        results={k:m.native_expectation(model=k,**args) for k in m.MODELS}
        prefix=c.partial_prefix_circuit(b,f,.8,2,1,'symmetric_directional',-.31,'A')
        tail=c.expected_rte_occurrence(cs,orders,2,.3,1,'B')
        self.assertEqual(results['NATIVE_MIXED']['cosine'],c.expected_partial_trajectory(tail,2,prefix,1,'X'))
        self.assertNotEqual(results['A']['cosine'].lower,results['NATIVE_MIXED']['cosine'].lower)
        self.assertNotEqual(results['B']['cosine'].lower,results['NATIVE_MIXED']['cosine'].lower)
    def test_pf_native_is_A_and_scalar_wrapper_counted_once(self):
        b,f,_=self.setup_parts();args=dict(bases=b,primitive_factory=f,T=.8,q=2,ancilla=1,scalar_phase=-.31,formula='4th')
        self.assertEqual(m.native_expectation(model='A',**args),m.native_expectation(model='NATIVE_MIXED',**args))
        d=m.native_expectation(model='A',**args);self.assertEqual(d['sine'].rz-d['cosine'].rz,1)
    def test_all_orders_including_unseen_rare_four_are_retained(self):
        b,f,cs=self.setup_parts();p=F(1,10**18)
        d=m.native_expectation(model='NATIVE_MIXED',bases=b,primitive_factory=f,T=.8,q=1,ancilla=1,scalar_phase=.2,
            components=cs,orders={0:1-p,2:F(0),4:p},r=1,tau=.3)
        z=m.native_expectation(model='NATIVE_MIXED',bases=b,primitive_factory=f,T=.8,q=1,ancilla=1,scalar_phase=.2,
            components=cs,orders={0:F(1)},r=1,tau=.3)
        self.assertNotEqual(d['cosine'].upper,z['cosine'].upper)
    def test_identity_phase_uncertainty_is_not_zero_filled(self):
        b,f,cs=self.setup_parts();c=m.counter;cs=(c.Component('identity',F(1,2),None),c.Component('z',F(1,2),b[0],(0,)))
        d=m.native_expectation(model='NATIVE_MIXED',bases=b,primitive_factory=f,T=.8,q=2,ancilla=1,scalar_phase=.2,
            components=cs,orders={0:F(1)},r=1,tau=.3)['cosine']
        self.assertFalse(d.exact);self.assertEqual(d.upper.rz-d.lower.rz,2)
    def test_mixed_expectation_matches_exhaustive_event_words(self):
        b,f,_=self.setup_parts();c=m.counter
        cs=(c.Component('plus',F(2,3),b[0],(0,),1),c.Component('minus',F(1,3),b[0],(0,),-1))
        orders={0:F(3,4),2:F(1,4)};patterns=[]
        for k,p in orders.items():
            for seq in itertools.product(cs,repeat=k+1):
                mass=p
                for x in seq:mass*=x.probability
                patterns.append(((k,seq[:-1],seq[-1]),mass))
        total=c.Cost()
        for (e1,p1),(e2,p2) in itertools.product(patterns,repeat=2):
            total+=c.count(c.explicit_rte_occurrence((e1,e2),.3,1,'B'))*(p1*p2)
        fixed=c.count(c.hadamard_wrapper(c.partial_prefix_circuit(b,f,.8,2,1,'symmetric_directional',-.31,'A'),1,'X'))
        actual=m.native_expectation(model='NATIVE_MIXED',bases=b,primitive_factory=f,T=.8,q=2,ancilla=1,
            scalar_phase=-.31,components=cs,orders=orders,r=2,tau=.3)['cosine']
        self.assertEqual(actual.lower,fixed+2*total);self.assertEqual(actual.lower,actual.upper)


class Grant(unittest.TestCase):
    def plan_grant(self):
        p=dict(SOURCE='1'*40,namespace=m.NAMESPACE,sealed=True,input_sha256='2'*64,
            mandatory_stop=True,next_stage_authorized=False,operations=['COMPARE_SAVED_VALUES'])
        g=dict(status='APPROVED',approved=True,runtime_authorization=True,SOURCE=p['SOURCE'],namespace=m.NAMESPACE,
            plan_digest=m.digest(p),input_sha256=p['input_sha256'],one_shot=True,automatic_retry=False)
        return p,g
    def test_invalid_draft_refused(self):
        p,g=self.plan_grant();g['status']='INVALID'
        with self.assertRaisesRegex(ValueError,'NEW_APPROVED'):m.verify_grant(p,g,source=p['SOURCE'],input_sha=p['input_sha256'])
    def test_source_plan_input_binding(self):
        p,g=self.plan_grant();m.verify_grant(p,g,source=p['SOURCE'],input_sha=p['input_sha256'])
        for k in ['SOURCE','plan_digest','input_sha256','namespace']:
            x=copy.deepcopy(g);x[k]='changed'
            with self.subTest(k=k),self.assertRaises(ValueError):m.verify_grant(p,x,source=p['SOURCE'],input_sha=p['input_sha256'])
    def test_retry_and_next_stage_refused(self):
        p,g=self.plan_grant();g['automatic_retry']=True
        with self.assertRaises(ValueError):m.verify_grant(p,g,source=p['SOURCE'],input_sha=p['input_sha256'])
    def test_invalid_grant_rejects_before_numeric_input_access(self):
        p,g=self.plan_grant();g['approved']=False
        with patch.object(runner.subprocess,'check_output',return_value=p['SOURCE']),patch.object(runner.lib,'file_hash',side_effect=AssertionError('input opened')):
            with self.assertRaisesRegex(ValueError,'NEW_APPROVED'):runner.run(p,g,'DO_NOT_OPEN','DO_NOT_CREATE')
    def test_artificial_one_shot_completion_and_refused_reuse(self):
        with tempfile.TemporaryDirectory(prefix='prtc-artificial-',dir='/home/AbeHiromu') as tmp:
            home=Path(tmp);inp=home/'input.json';inp.write_text(json.dumps([row()]));out=home/'output'
            p,g=self.plan_grant();p.update(output=str(out),input_sha256=m.file_hash(inp),rows=[m.digest(row())],models=['A'],metrics=['RZ'],
                source_hashes=source_hashes())
            g.update(plan_digest=m.digest(p),input_sha256=p['input_sha256'])
            with patch.object(runner.subprocess,'check_output',return_value=p['SOURCE']):
                t=runner.run(p,g,inp,out);self.assertEqual(t['status'],'COMPLETE_MANDATORY_STOP')
                with self.assertRaises(FileExistsError):runner.run(p,g,inp,out)
    def test_first_failure_preserves_partial_and_stops(self):
        with tempfile.TemporaryDirectory(prefix='prtc-artificial-',dir='/home/AbeHiromu') as tmp:
            home=Path(tmp);inp=home/'input.json';bad=row();bad['compiled_mean']=0
            inp.write_text(json.dumps([bad,row()]));p,g=self.plan_grant();out=home/'output'
            p.update(output=str(out),input_sha256=m.file_hash(inp),rows=[m.digest(bad),m.digest(row())],models=['A'],metrics=['RZ'],source_hashes=source_hashes())
            # Distinct IDs keep the zero-cost check as the first failure.
            second=row();second['compiled_observation_id']='other';inp.write_text(json.dumps([bad,second]));p.update(input_sha256=m.file_hash(inp),rows=[m.digest(bad),m.digest(second)])
            g.update(plan_digest=m.digest(p),input_sha256=p['input_sha256'])
            with patch.object(runner.subprocess,'check_output',return_value=p['SOURCE']),self.assertRaises(ValueError):runner.run(p,g,inp,out)
            t=json.loads((out/'terminal.json').read_text());self.assertEqual(t['rows_completed'],0);self.assertIn('NONPOSITIVE',t['first_stop'])

if __name__=='__main__':unittest.main(verbosity=2)
