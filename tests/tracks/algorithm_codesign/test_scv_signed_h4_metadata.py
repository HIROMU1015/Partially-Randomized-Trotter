"""Invented dyadic bytes only; never read the molecular input or call run_fixed."""
from fractions import Fraction as F
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from trottertracks.algorithm_codesign import scv_signed_h4_metadata as a
from trottertracks.algorithm_codesign import a1_n3_connection as n3
from trottertracks.algorithm_codesign import a1_control_variates as cv

PATTERNS=[[1,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,1,1]]


def fixture(constant=False):
    law=dict(uniform_grid_size=2**53,application_label_order='products_then_rotation',orders=[0,2],
        order_masses=[dict(exact='3/4'),dict(exact='1/4')],
        label_masses=[dict(label=l,exact=p) for l,p in zip(('a','b','c','zero'),('1/4','1/4','1/2','0'))],
        five_patterns=PATTERNS,five_strata=[dict(exact=p) for p in ('3/16','9/16','3/16','3/64','1/64')])
    cat=dict(hamiltonian_hash='invented-H',preparation_hash='invented-prep',components=[
        dict(component_id=l,df_fragment_id='invented-'+l,coefficient_sign=s,diagonal_pauli_support=p)
        for l,s,p in zip(('a','b','c','zero'),(-1,1,-1,1),((0,),(0,1),(1,2),(2,)))])
    if constant:
        for r in cat['components']:r.update(coefficient_sign=1,diagonal_pauli_support=[0])
    return law,cat


def setup(root,law,cat):
    settings=dict(input_commit='invented',feature_version=a.signed.FEATURE,a='a',b='b',system_qubits=3,
        patterns=PATTERNS,identity=dict(hamiltonian_hash='invented-H',preparation_hash='invented-prep'))
    settings['inputs']={}
    for k,record in (('law',law),('catalog',cat)):
        data=json.dumps(record).encode();(root/k).write_bytes(data)
        settings['inputs'][k]=dict(path=k,sha256=a.digest(data),git_blob='invented')
    return settings


class MetadataEntryTests(unittest.TestCase):
    def perform(self,law=None,cat=None,constant=False,settings_change=None):
        if law is None:law,cat=fixture(constant)
        with TemporaryDirectory() as tmp:
            root=Path(tmp);s=setup(root,law,cat)
            if settings_change:settings_change(s)
            result=a.execute_once(root,root/'out',s,dict(source_commit='invented-source',scope={'model':'ARTIFICIAL ONLY'}))
            files={p.name:p.read_text() for p in (root/'out').iterdir()}
            parsed={k:json.loads(v) for k,v in files.items() if k.endswith('.json')}
            return result,parsed,files

    def test_nonconstant_exact_result_all_cells_saved_and_STOP(self):
        r,p,files=self.perform()
        self.assertEqual(r['metadata_structural_eligibility'],'ELIGIBLE');self.assertTrue(r['h1_nonconstant'])
        self.assertEqual([c['mean_exact'] for c in r['cells']],[F(-1,2),F(-1,3),F(-3,8),F(-3,8),F(-3,8)])
        self.assertEqual([c['second_moment_exact'] for c in r['cells']],[F(1,4),F(1),F(13,16),F(13,16),F(13,16)])
        self.assertEqual([c['variance_exact'] for c in r['cells']],[F(0),F(8,9),F(43,64),F(43,64),F(43,64)])
        saved=p['METADATA_RESULT.json'];self.assertEqual(saved['cells'][1]['variance_exact'],'8/9')
        for row,live in zip(saved['cells'],r['cells']):
            for k in ('mass_exact','mean_exact','second_moment_exact','variance_exact'):self.assertEqual(F(row[k]),live[k])
        self.assertEqual(p['STOP.json']['status'],'MANDATORY_STOP');self.assertFalse(r['science_ready'])
        self.assertFalse(r['state_loaded']);self.assertFalse(r['engine_loaded']);self.assertFalse(r['old_measurement_data_used'])

    def test_constant_classified_ineligible_without_fallback(self):
        r,p,_=self.perform(constant=True)
        self.assertEqual(r['metadata_structural_eligibility'],'INELIGIBLE_CONSTANT_H1')
        self.assertFalse(r['h1_nonconstant']);self.assertEqual(r['cells'][1]['variance_exact'],0)
        self.assertFalse(p['STOP.json']['next_stage_started'])

    def test_observed_counts_and_static_expectations_separate(self):
        r,p,files=self.perform();counts=p['CALL_COUNTS.json'];observed=counts['observed_top_level']
        for key in ('read_LAW','read_catalog','metadata_attempt','signed_binder','external_cells','signed_control','moment_snapshot','h1_classification'):
            self.assertEqual(observed[key],dict(started=1,returned=1,exceptions=0))
        i=counts['observed_internal_entries']
        for key,n in dict(cells=2,pattern_mass=10,joint_mean=80,json_law=2,json_catalog=2,signed_control_post_init=1,component=4,control_bank=3,categorical_law=2).items():self.assertEqual(i[key],n)
        self.assertEqual(counts['forbidden_calls_attempted'],{});self.assertEqual(counts['retry'],0)
        journal=[json.loads(x) for x in files['CALLS.jsonl'].splitlines()]
        self.assertEqual(sum(x['event']=='CALL_START' for x in journal),len(observed))
        self.assertEqual(sum(x['event']=='CALL_RETURN' for x in journal),len(observed))

    def test_each_input_hash_mismatch_before_scientific_attempt(self):
        for k in ('law','catalog'):
            def change(s):s['inputs'][k]['sha256']='bad'
            r,p,_=self.perform(settings_change=change)
            self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR');self.assertIsNone(r['h1_nonconstant'])
            self.assertNotIn('metadata_attempt',p['CALL_COUNTS.json']['observed_top_level'])
            self.assertEqual(p['STOP.json']['attempt_started'],0)

    def test_reordered_catalog_stops_before_moments(self):
        law,cat=fixture();cat['components'].reverse()
        r,p,_=self.perform(law,cat)
        self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')
        self.assertNotIn('signed_control',p['CALL_COUNTS.json']['observed_top_level'])

    def test_missing_duplicate_or_unknown_label_rejected(self):
        for mode in ('missing','duplicate','unknown'):
            law,cat=fixture()
            if mode=='missing':cat['components'].pop()
            if mode=='duplicate':cat['components'][-1]=copy.deepcopy(cat['components'][0])
            if mode=='unknown':cat['components'][-1]['component_id']='absent'
            r,p,_=self.perform(law,cat)
            self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR');self.assertEqual(r['cells'],[])

    def test_bad_sign_support_domain_rejected(self):
        for change in (dict(coefficient_sign=0),dict(coefficient_sign=True),dict(diagonal_pauli_support=[0,0]),
                       dict(diagonal_pauli_support=[0,1,2]),dict(diagonal_pauli_support=[3]),dict(diagonal_pauli_support=[-1])):
            law,cat=fixture();cat['components'][1].update(change)
            r,p,_=self.perform(law,cat);self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')
            self.assertNotIn('signed_control',p['CALL_COUNTS.json']['observed_top_level'])

    def test_wrong_H_preparation_identity_rejected(self):
        for field in ('hamiltonian_hash','preparation_hash'):
            law,cat=fixture();cat[field]='other'
            r,p,_=self.perform(law,cat);self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')

    def test_saved_strata_patterns_mass_and_schema_rejected(self):
        for mode in ('patterns','mass','grid','normalization'):
            law,cat=fixture();law=copy.deepcopy(law)
            if mode=='patterns':law['five_patterns'].reverse()
            if mode=='mass':law['five_strata'][1]['exact']='1/2'
            if mode=='grid':law['uniform_grid_size']=32
            if mode=='normalization':law['label_masses'][0]['exact']='1/2'
            r,p,_=self.perform(law,cat);self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')
            self.assertNotIn('signed_control',p['CALL_COUNTS.json']['observed_top_level'])

    def test_input_bytes_unchanged_and_only_read_once(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);s=setup(root,*fixture());before={k:(root/k).read_bytes() for k in ('law','catalog')}
            original=Path.read_bytes;seen=[]
            def read(p):seen.append(p.name);return original(p)
            with patch.object(Path,'read_bytes',read):r=a.execute_once(root,root/'out',s,{})
            self.assertEqual(seen,['law','catalog']);self.assertEqual(r['metadata_structural_eligibility'],'ELIGIBLE')
            for k,v in before.items():self.assertEqual((root/k).read_bytes(),v)

    def test_output_collision_and_resume_refused_no_input_read(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);s=setup(root,*fixture());out=root/'out';out.mkdir();(out/'STOP.json').write_text('original STOP')
            with patch.object(Path,'read_bytes',side_effect=AssertionError('no read')):
                with self.assertRaises(FileExistsError):a.execute_once(root,out,s,{})
            self.assertEqual((out/'STOP.json').read_text(),'original STOP')
        with TemporaryDirectory() as tmp:
            root=Path(tmp);s=setup(root,*fixture());out=root/'out';a.execute_once(root,out,s,{})
            with self.assertRaises(FileExistsError):a.execute_once(root,out,s,{})

    def test_exception_after_moment_preserves_partial_and_STOP(self):
        with patch.object(a,'check_moments',side_effect=KeyboardInterrupt('artificial interruption')):
            r,p,files=self.perform()
        self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR');self.assertIsNone(r['h1_nonconstant'])
        self.assertEqual(p['METADATA_RESULT.json']['cells'][1]['variance_exact'],'8/9')
        counts=p['CALL_COUNTS.json']['observed_top_level']
        self.assertEqual(counts['signed_control']['returned'],1);self.assertEqual(counts['metadata_attempt']['exceptions'],1)
        self.assertFalse(p['STOP.json']['next_stage_started']);self.assertIn('CALL_EXCEPTION',files['CALLS.jsonl'])

    def test_forbidden_scientific_calls_blocked_before_body(self):
        def prohibited(control):cv.fit_and_freeze(None,None)
        with patch.object(a,'check_moments',prohibited):r,p,_=self.perform()
        self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')
        self.assertEqual(sum(p['CALL_COUNTS.json']['forbidden_calls_attempted'].values()),1)
        self.assertIn('forbidden call blocked',r['reason'])

    def test_normal_path_never_loads_services_RNG_draw_or_fit(self):
        with patch.object(n3,'load_certified_services',side_effect=AssertionError('state/engine')), \
             patch.object(n3,'pcg64_streams',side_effect=AssertionError('RNG')), \
             patch.object(n3.ConditionalSampler,'draw',side_effect=AssertionError('sampling')), \
             patch.object(cv,'fit_and_freeze',side_effect=AssertionError('OLS')):
            r,p,_=self.perform()
        self.assertEqual(r['metadata_structural_eligibility'],'ELIGIBLE')
        self.assertEqual(p['CALL_COUNTS.json']['forbidden_calls_attempted'],{})

    def test_internal_overcall_FAIL_CLOSED(self):
        original=a.check_law
        def extra(md,record,settings,cells):
            original(md,record,settings,cells);md.sampler.cells('B1-OL')
        with patch.object(a,'check_law',extra):r,p,_=self.perform()
        self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')
        self.assertNotIn('signed_control',p['CALL_COUNTS.json']['observed_top_level'])
        self.assertEqual(p['CALL_COUNTS.json']['observed_internal_entries']['cells'],3)

    def test_result_save_failure_classified_UNKNOWN_no_science_retry(self):
        original=a.save;failures=[]
        def once(path,value):
            if Path(path).name=='METADATA_RESULT.json' and not failures:
                failures.append(1);raise OSError('artificial result save failure')
            return original(path,value)
        with patch.object(a,'save',once):r,p,_=self.perform()
        self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')
        self.assertEqual(p['METADATA_RESULT.json']['metadata_structural_eligibility'],'UNKNOWN_ERROR')
        self.assertEqual(p['CALL_COUNTS.json']['observed_top_level']['signed_control']['started'],1)
        self.assertTrue(p['STOP.json']['no_retry'])

    def test_call_entry_budget_no_second_function_invocation(self):
        with TemporaryDirectory() as tmp:
            calls=a.Calls(tmp);seen=[];calls.call('toy',lambda:seen.append(1))
            with self.assertRaises(RuntimeError):calls.call('toy',lambda:seen.append(2))
            self.assertEqual(seen,[1])

    def test_wrong_feature_version_before_science(self):
        r,p,_=self.perform(settings_change=lambda s:s.update(feature_version='indicator'))
        self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')
        self.assertNotIn('metadata_attempt',p['CALL_COUNTS.json']['observed_top_level'])

    def test_finalization_failure_is_UNKNOWN_and_STOP(self):
        original=a.save
        def fail_counts(path,value):
            if Path(path).name=='CALL_COUNTS.json':raise OSError('artificial counts save failure')
            return original(path,value)
        with patch.object(a,'save',fail_counts):r,p,_=self.perform()
        self.assertEqual(r['metadata_structural_eligibility'],'UNKNOWN_ERROR')
        self.assertEqual(p['METADATA_RESULT.json']['metadata_structural_eligibility'],'UNKNOWN_ERROR')
        self.assertEqual(p['STOP.json']['result_status'],'UNKNOWN_ERROR')
        self.assertTrue(p['STOP.json']['finalizer_errors'])

    def test_preflight_source_environment_hash_and_collision_guards(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);s=setup(root,*fixture())
            s.update(output='out',contract_commit='invented',kernel_commit='invented',scope={'model':'ARTIFICIAL'},references={},attempt_id='toy')
            (root/a.DOC).mkdir(parents=True);(root/a.SETTINGS).write_text(json.dumps(s))
            (root/'invented_source.py').write_text('invented')
            lock=dict(environment={'toy':1},kernel_paths=[],source_sha256={'invented_source.py':a.digest(b'invented')})
            (root/a.PROFILE).write_text(json.dumps(lock))
            def git(root,*args):
                if args==('rev-parse','HEAD'):return 'toy-source'
                if args==('status','--porcelain=v1'):return ''
                return 'invented'
            with patch.object(a,'git',git),patch.object(a,'environment',return_value={'toy':1}):
                settings,source=a.preflight(root,'toy-source')
                self.assertEqual(settings,s);self.assertEqual(source['source_commit'],'toy-source')
                with self.assertRaisesRegex(ValueError,'SOURCE commit'):a.preflight(root,'different')
                with patch.object(a,'environment',return_value={'toy':2}):
                    with self.assertRaisesRegex(ValueError,'environment'):a.preflight(root,'toy-source')
                (root/'invented_source.py').write_text('changed')
                with self.assertRaisesRegex(ValueError,'SOURCE hash'):a.preflight(root,'toy-source')
                (root/'invented_source.py').write_text('invented');(root/'out').mkdir()
                with self.assertRaises(FileExistsError):a.preflight(root,'toy-source')


if __name__=='__main__':unittest.main()
