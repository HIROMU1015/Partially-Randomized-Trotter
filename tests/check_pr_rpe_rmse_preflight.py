"""Artificial plans and read-only saved schema fixtures; no real N/RMSE math."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pr_rpe_rmse_preflight as f
from trottertracks import pr_rpe_rmse as model
from trottertracks import pr_r_dependent_n as old
from demo_pr_rpe_rmse import artificial_design
from check_pr_rpe_rmse import proposal


def fixtures(*,method='PR',M=3,design=None):
    d=deepcopy(design or artificial_design());source=dict(kind='ARTIFICIAL_ONLY')
    d['profile']=dict(f.p.PROFILE,compiler='MOCK');d['binding'].update(decomposition='MOCK_D',source=source)
    s=proposal(M=M,design=d)
    spec=dict(binding=d['binding'],profile=d['profile'],method=method,M=M,h=d['base_time'],
        L_D=d['prefix'] if method=='PR' else d['PF']['prefix'],delta_pf=d['delta'],formula=d['formula'],
        split=dict(L_D=d['prefix'] if method=='PR' else d['PF']['prefix'],source=source,
            deterministic_scope='PREFIX' if method=='PR' else 'FULL_H',
            randomized_scope='TAIL_AFTER_PREFIX' if method=='PR' else 'NONE',lambda_R=.5,lambda_source=source),
        kappa=dict(kind='PAPER_TOTAL_ROTATIONS',value=2,definition_source=source) if method=='PR' else None,
        event_mapping=dict(kind=model.MAPPING if method=='PR' else 'NONE',source=source),
        finite_K=2 if method=='PR' else None,
        law=dict(model='ideal_finite_RTE_law' if method=='PR' else 'deterministic',source=source),
        rmse_policy=dict(family='PR_PAPER_RMSE',id='MOCK_RMSE',source=source,error_contract='MOCK_EXPLICIT_CONTRACT',
            scientific_status='PROPOSED_NOT_ADOPTED',target_rmse_ha=2))
    request=next(r for r in s['rows'] if f.p.arm(r['identity']['candidate'])==method)
    spec['law']['source']=deepcopy(request['law_binding']['source'])
    return f.b_metadata(ROOT).generate(spec),s


def inventory(schedule):
    means=[];refs=[]
    for r in schedule['rows']:
        i=r['identity']
        means.append(dict(candidate=deepcopy(i['candidate']),binding=deepcopy(i['binding']),
            profile=deepcopy(i['profile']),structure=deepcopy(i['structure']),source=dict(kind='MOCK'),
            fields=list(f.MEAN_FIELDS),error_fields=list(f.ERROR_FIELDS),saved_status='CONDITIONAL_ROUND_PASS'))
        refs.append(dict(binding=deepcopy(i['binding']),T=i['candidate']['T'],source=dict(kind='MOCK'),fields=list(f.REFERENCE_FIELDS)))
    return dict(means=means,references=refs)


class PreflightTests(unittest.TestCase):
    def test_complete_metadata_never_qualifies_N_or_RMSE(self):
        b,a=fixtures();out=f.dry_run(b,a,inventory(a),root=ROOT)
        self.assertEqual(out['status'],'TECHNICAL_PREFLIGHT_COMPLETE_NOT_SCIENCE')
        self.assertFalse(out['N_packet']['rmse_evidence']['covers_state_target_alias_prior_decoder_all_rounds'])
        self.assertEqual(out['N_packet']['status'],'NOT_EVALUATED')
        self.assertIsNone(out['N_packet']['rmse_evidence']['rmse_bound_ha'])
        for row,brow in zip(out['N_packet']['rounds'],b['rounds']):
            self.assertEqual(row['identity'],brow['identity']);self.assertEqual(row['law_binding'],brow['law_binding'])
            self.assertIsNone(row['N_X']);self.assertIsNone(row['N_Y'])
        for row in out['rounds']:
            self.assertIsNone(row['nominal_N_X']);self.assertEqual(row['additional_scientific_calls_if_approved']['finite_mean_candidates_upper_bound'],0)

    def test_missing_input_and_reference_calls_are_explicit_never_executed(self):
        b,a=fixtures();out=f.dry_run(b,a,dict(means=[],references=[]),root=ROOT)
        for r in out['rounds']:
            calls=r['additional_scientific_calls_if_approved']
            self.assertEqual((calls['finite_mean_candidates_upper_bound'],calls['exact_tail_internal_controls_upper_bound'],calls['independent_reference_actions']),(1,1,2))
        self.assertEqual(out['science_calls'],0)

    def test_numeric_leaves_cannot_be_called_by_preflight(self):
        b,a=fixtures()
        with patch.object(model,'nominal_axis_shots',side_effect=AssertionError('N math')),patch.object(model,'existing_finite_mean',side_effect=AssertionError('real action')):
            f.dry_run(b,a,inventory(a),root=ROOT)

    def test_null_components_and_partial_saved_allowance_are_missing(self):
        self.assertEqual(f.present({'raw':{'real':None,'imag':0}},('raw',)),[])
        b,a=fixtures();inv=inventory(a);inv['means'][1]['error_fields'].remove('exact_tail_norm_error')
        out=f.dry_run(b,a,inv,root=ROOT)
        self.assertEqual(out['rounds'][0]['missing_mean_fields'],['exact_tail_norm_error'])
        self.assertTrue(out['rounds'][0]['additional_scientific_calls_if_approved']['recover_saved_diagnostics_first'])

    def test_all_round_identity_and_whole_schedule_mutation_rejected(self):
        b,a=fixtures();b['rounds'].pop()
        with self.assertRaises(ValueError):f.dry_run(b,a,{},root=ROOT)
        b,a=fixtures();a['rows'][1]['identity']['binding']['state']='FOREIGN'
        with self.assertRaises(ValueError):f.dry_run(b,a,{},root=ROOT)

    def test_binding_mismatch_blocks_instead_of_scheduling_science(self):
        b,a=fixtures();spec=deepcopy(b['spec']);spec['binding']['state']='OTHER'
        out=f.dry_run(f.b_metadata(ROOT).generate(spec),a,inventory(a),root=ROOT)
        self.assertIn('BINDING_OR_SOURCE_MISMATCH',out['global_issues'])
        self.assertTrue(all(r['additional_scientific_calls_if_approved'] is None for r in out['rounds']))

    def test_saved_SOURCE_mismatch_is_not_a_cache_miss_retry(self):
        b,a=fixtures();inv=inventory(a);inv['means'][1]['binding']['state']='WRONG'
        out=f.dry_run(b,a,inv,root=ROOT)
        self.assertIn('SAVED_SOURCE_IDENTITY_MISMATCH',out['rounds'][0]['issues'])
        self.assertIsNone(out['rounds'][0]['additional_scientific_calls_if_approved'])

    def test_saved_FAIL_or_numeric_unresolved_is_preserved(self):
        for status in ('CONDITIONAL_ROUND_FAIL','NUMERICAL_UNRESOLVED','MANDATORY_STOP','INFEASIBLE'):
            b,a=fixtures();inv=inventory(a);inv['means'][1]['saved_status']=status
            out=f.dry_run(b,a,inv,root=ROOT)
            self.assertIn('SAVED_FAILURE_OR_UNRESOLVED_NO_AUTOMATIC_RETRY',out['rounds'][0]['issues'])

    def test_kappa_and_tail_SOURCE_are_not_interchangeable(self):
        self.assertTrue(f.same_source(dict(commit='71ba49bb28b295aa28e51101bc3b1bf07d500767',path='src/trotterlib/rte.py'),
                                     dict(commit=f.A_SOURCE,path='src/trotterlib/rte.py'),ROOT))
        b,a=fixtures();spec=deepcopy(b['spec']);spec['kappa']['value']=3
        out=f.dry_run(f.b_metadata(ROOT).generate(spec),a,{},root=ROOT)
        self.assertIn('KAPPA_DEFINITION_OR_MAPPING_MISMATCH',out['global_issues'])
        spec=deepcopy(b['spec']);spec['split']['lambda_source']={'other':'MOCK'}
        out=f.dry_run(f.b_metadata(ROOT).generate(spec),a,{},root=ROOT)
        self.assertIn('TAIL_NORM_OR_SOURCE_MISMATCH',out['global_issues'])
        spec=deepcopy(b['spec']);spec['law']['source']={'unknown':'SAMPLING_LAW'}
        out=f.dry_run(f.b_metadata(ROOT).generate(spec),a,{},root=ROOT)
        self.assertIn('LAW_SOURCE_JOIN_NOT_CONFIRMED',out['rounds'][0]['issues'])

    def test_H4_B0_cannot_be_relabelled_B_full_H_PF(self):
        d=artificial_design();d['PF']=dict(method='B0',prefix=7)
        b,a=fixtures(method='PF',design=d)
        out=f.dry_run(b,a,{},root=ROOT)
        self.assertTrue(all('PF_B0_IS_NOT_B_FULL_H_BASELINE' in r['issues'] for r in out['rounds']))

    def test_H6_early_special_and_signed_schedule_block_default_B(self):
        d=model.design_from_task(old.saved_task(ROOT,'H6'))
        # Physical recipe metadata only; no real lambda or normalizer evaluated.
        b,a=fixtures(M=4,design=d)
        out=f.dry_run(b,a,{},root=ROOT)
        for r in out['rounds'][:4]:self.assertIn('B_FIXED_L_D_CANNOT_REPRESENT_H6_EARLY_OVERRIDE',r['issues'])
        self.assertTrue(any(w<0 for w in a['rows'][-1]['identity']['structure']['signed_weights']))
        self.assertEqual(sum(a['rows'][-1]['identity']['structure']['allocation']),a['rows'][-1]['identity']['candidate']['R_outer'])

    def test_old_policy_N_is_ignored_even_if_present_in_inventory(self):
        b,a=fixtures();inv=inventory(a)
        for v in inv['means']:v.update(plan={'N_X':999},N_X=999,nominal_N_X=999)
        out=f.dry_run(b,a,inv,root=ROOT)
        # Presence descriptors must be sanitized, not echo unrelated numeric payloads.
        self.assertIsNone(out['N_packet']['rounds'][0]['N_X'])
        self.assertIsNone(out['rounds'][0]['nominal_N_X'])
        self.assertTrue(all('N_X' not in v and 'plan' not in v for r in out['rounds'] for v in r['saved_mean_candidates']))

    def test_pinned_B_fixture_validates_without_prices_or_numeric_results(self):
        doc=f.p.SavedArtifacts(ROOT).read(dict(commit=f.B_SOURCE,path='docs/research/pr_rpe_rmse_controller_20261112/SYNTHETIC_IO_EXAMPLE.json'))
        f.b_metadata(ROOT).validate(doc['configuration'])

    def test_saved_schema_inventory_never_imports_numeric_backend(self):
        code="from pathlib import Path;from trottertracks import pr_rpe_rmse_preflight as f;v=f.saved_inventory(Path.cwd());import sys;assert v['science_calls']==0;assert not any(k in sys.modules for k in ('numpy','scipy','qiskit'));assert all('plan' not in r and 'N_X' not in r for r in v['means'])"
        env=__import__('os').environ.copy();env['PYTHONPATH']='src'
        subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,env=env,check=True)


if __name__=='__main__':unittest.main()
