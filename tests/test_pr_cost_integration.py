"""Artificial-only cost, cohort, source/input and execution-refusal regression."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src/trotterlib')]
import pr_cost_contract as contract
import pr_cost_integration as c
import run_pr_cost_integration as runner
import pr_cost_saved as saved


def binding():
    return dict(system='ARTIFICIAL',geometry='synthetic',basis='synthetic',rank=2,rank_policy='synthetic',
        T=.8,hamiltonian='h',state='s',snapshot='i',compiler='c',profile='p',control='symmetric_directional')


def accounting(eligible=True):
    return dict(cost_acquisition_eligible=eligible,eligibility_under_declared_allowance='ELIGIBLE' if eligible else 'INELIGIBLE',
        axes=dict(real=dict(shots=3),imag=dict(shots=7)),N_total=10)


def evidence(kind='DIRECT_MEASURED',axis='cosine',value=11,cell='det'):
    return dict(kind=kind,axis=axis,RZ=value,cell_id=cell,binding=binding(),SOURCE='a'*40,result_sha256='b'*64,
        preparation_identity='prep',scalar_policy='original_plus_tail_once',native_exit_code=0,reaped=True,
        final_gate='PASS',model_sha256='model',model_validation='ACCEPTED_FIXED_MODEL')


def pairs(cohort='EXPLORATION_N8'):
    return [dict(cell_id='pr',replica=i,seed=100+i,event_digest='event'+str(i),binding=binding(),cohort=cohort,
        axes={a:dict(seed=100+i,event_digest='event'+str(i),RZ=(10+i if a=='cosine' else 30-i)) for a in c.AXES})
        for i in range(32 if cohort=='CONFIRMATION_N32' else 8)]


def data():
    rows=[dict(cell_id=cid,method=method,bias_mode='USUAL_SIGNED',epsilon=.01,factor=1.,binding=binding(),
        accounting=accounting(),cohort='EXPLORATION_N8' if cid=='pr' else None)
        for cid,method in [('det','B0I'),('pr','B2')]]
    return dict(kind='ARTIFICIAL',binding=binding(),saved_accounting=rows,
        cost_evidence=[evidence(axis=a,value=11 if a=='cosine' else 20) for a in c.AXES],
        paired_PR=dict(pr=pairs()),deterministic_sets=dict(old=['det'],strengthened=['det']),PR_ids=['pr'])


class Accounting(unittest.TestCase):
    def test_direct_measurement_precedes_fixed_model(self):
        result=c.choose_unit_cost([evidence('FIXED_MODEL',value='100/3'),evidence()],binding(),'det','cosine')
        self.assertEqual(result['RZ'],11)

    def test_each_axis_keeps_its_own_price(self):
        result=c.account_rows(data(),c.Permit('D_ACCOUNTING',synthetic=True))
        self.assertEqual(result[0]['G_RZ_exact'],'173')

    def test_model_is_explicit_when_direct_cost_missing(self):
        d=data();d['cost_evidence'][0]=evidence('FIXED_MODEL',value='100/3')
        r=c.account_rows(d,c.Permit('D_ACCOUNTING',synthetic=True))
        self.assertEqual(r[0]['status'],'MODEL_CONDITIONAL_POINT')
        self.assertEqual(F(r[0]['G_RZ_exact']),240)

    def test_cost_missing_is_null_and_cannot_claim_full_set_winner(self):
        d=data();d['cost_evidence']=[]
        r=c.account_rows(d,c.Permit('D_ACCOUNTING',synthetic=True))
        self.assertIsNone(r[0]['G_RZ_exact'])
        result=c.compare(r,['det'],['pr'])
        self.assertIsNone(result['full_registered_set_PR_reduction_percent_exact'])
        self.assertFalse(result['comparison_complete'])

    def test_ineligible_candidate_not_zero(self):
        d=data();d['saved_accounting'][0]['accounting']=accounting(False)
        r=c.account_rows(d,c.Permit('D_ACCOUNTING',synthetic=True))
        self.assertIsNone(r[0]['G_RZ_exact'])

    def test_all_identity_dimensions_reject_mixing(self):
        for key in c.IDENTITY:
            e=evidence();e['binding'][key]='different'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'BINDING'):
                c.choose_unit_cost([e],binding(),'det','cosine')

    def test_source_profile_preparation_acceptance_required(self):
        for key,value in [('SOURCE',None),('result_sha256',None),('preparation_identity',None),
                          ('native_exit_code',2),('reaped',False),('final_gate','STOP')]:
            e=evidence();e[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):c.choose_unit_cost([e],binding(),'det','cosine')

    def test_conflicting_actual_cost_does_not_fall_back_to_model(self):
        with self.assertRaisesRegex(ValueError,'CONFLICTING'):
            c.choose_unit_cost([evidence(value=11),evidence(value=12),evidence('FIXED_MODEL')],binding(),'det','cosine')

    def test_conditional_sample_not_used_as_unit_cost(self):
        self.assertIsNone(c.choose_unit_cost([evidence('CONDITIONAL')],binding(),'det','cosine'))

    def test_paired_SE_uses_weighted_trajectory_totals(self):
        value=c.paired_statistics(pairs(),dict(cosine=3,sine=7),cohort='EXPLORATION_N8')
        self.assertEqual(F(value['mean_exact']),226)
        # Totals = 240-4i, whose sample variance is 96; SE=sqrt(96/8).
        self.assertAlmostEqual(value['sample_SE'],12**.5)

    def test_pairing_corruption_rejected(self):
        for key in ('seed','event_digest'):
            p=pairs();p[0]['axes']['sine'][key]='wrong'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'PAIRED_TRAJECTORY'):
                c.paired_statistics(p,dict(cosine=3,sine=7),cohort='EXPLORATION_N8')

    def test_n8_n32_and_conditional_pooling_refused(self):
        for cohort in ('CONFIRMATION_N32','CONDITIONAL_N8'):
            p=pairs();p[0]['cohort']=cohort
            with self.subTest(cohort=cohort),self.assertRaisesRegex(ValueError,'MIXING'):
                c.paired_statistics(p,dict(cosine=3,sine=7),cohort='EXPLORATION_N8')

    def test_partial_or_duplicate_trajectory_coverage_rejected(self):
        p=pairs();p[-1]=deepcopy(p[0])
        with self.assertRaises(ValueError):c.paired_statistics(p,dict(cosine=3,sine=7),cohort='EXPLORATION_N8')
        with self.assertRaises(ValueError):c.paired_statistics(p[:-1],dict(cosine=3,sine=7),cohort='EXPLORATION_N8')

    def test_normal_and_noncancellation_are_separate_endpoints(self):
        d=data();a=deepcopy(d['saved_accounting'][0]);a['bias_mode']='EMPIRICAL_NONCANCELLATION'
        d['saved_accounting'].append(a)
        r=c.account_rows(d,c.Permit('D_ACCOUNTING',synthetic=True))
        with self.assertRaisesRegex(ValueError,'SAME_ENDPOINT'):c.compare([r[1],r[2]],['det'],['pr'])

    def test_input_immutable_and_fixed_shots(self):
        d=data();before=deepcopy(d)
        runner.account(d,c.Permit('D_ACCOUNTING',synthetic=True))
        self.assertEqual(d,before)

    def test_incorrect_saved_shot_sum_rejected(self):
        d=data();d['saved_accounting'][0]['accounting']['N_total']=11
        with self.assertRaisesRegex(ValueError,'SHOTS'):c.account_rows(d,c.Permit('D_ACCOUNTING',synthetic=True))

    def test_synthetic_permission_cannot_read_real_values(self):
        d=data();d['kind']='SAVED_REAL_VALUES'
        with self.assertRaisesRegex(ValueError,'PERMIT'):c.account_rows(d,c.Permit('D_ACCOUNTING',synthetic=True))

    def test_actual_scope_does_not_authorize_other_stage(self):
        with self.assertRaisesRegex(ValueError,'PERMIT'):c.account_rows(data(),c.Permit('A_COST',synthetic=True))


class ProjectionsAndClosures(unittest.TestCase):
    def test_missing_accounting_is_not_ineligibility_or_zero(self):
        d=data();d['saved_accounting'][0]['accounting']=None
        rows=c.account_rows(d,c.Permit('D_ACCOUNTING',synthetic=True))
        self.assertEqual(rows[0]['status'],'ACCOUNTING_MISSING')
        self.assertFalse(c.compare(rows,['det'],['pr'])['comparison_complete'])

    def test_D_schema_projection_preserves_integer_prices_and_profile(self):
        proof=dict(native_exit_code=0,reaped=True,final_gate='PASS',SOURCE='a'*40,
            result_sha256='b'*64,preparation_identities=dict(det='prep'))
        body=dict(status='COMPLETE_MANDATORY_STOP',cell_id='det',rows=[dict(axis=a,RZ=12,
            compiler_hash='c',fingerprint='fp'+a) for a in c.AXES])
        rows=saved.measured_axes(body,binding(),proof)
        self.assertEqual([r['RZ'] for r in rows],[12,12])
        body['rows'][0]['profile_sha256']='WRONG'
        with self.assertRaisesRegex(ValueError,'PROFILE'):saved.measured_axes(body,binding(),proof)

    def test_saved_modes_and_missing_multiplier10_are_not_recomputed(self):
        record=dict(cell=dict(id='x',method='B0'),precision_rows={str(e):{'1.0':accounting()} for e in c.EPS})
        rows=saved.signal_accounting([record],binding(),{})
        self.assertEqual(len(rows),20)
        self.assertTrue(all(r['accounting'] is None for r in rows if r['factor']==10.))
        self.assertTrue(all(r['accounting'] is None for r in rows if r['bias_mode']=='EMPIRICAL_NONCANCELLATION'))

    def test_streaming_input_bytes_and_size_refuse_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input';path.write_bytes(b'good')
            rows=[dict(role='artificial',path=str(path),bytes=4,sha256=contract.sha(path))]
            contract.verify_byte_rows(rows);path.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError,'INPUT_BYTES_CHANGED'):contract.verify_byte_rows(rows)

    def test_source_blob_and_working_bytes_both_required(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.py';path.write_bytes(b'good')
            hashes={'source.py':hashlib.sha256(b'good').hexdigest()}
            with patch.object(contract.subprocess,'check_output',return_value=b'wrong'):
                with self.assertRaisesRegex(ValueError,'GIT_BINDING'):contract.verify_source(Path(d),'a'*40,hashes)
            path.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError,'SOURCE_BYTES_CHANGED'):contract.verify_source(Path(d),'a'*40,hashes)


class Conditional(unittest.TestCase):
    def fixture(self):
        groups=[dict(stratum=s,probability=p,mean_RZ=v,binding=binding(),cell_id='pr',law_id='law',
            cohort='CONDITIONAL_N8',mass_kind='ORIGINAL_RTE_LAW')
            for s,p,v in [('zero','1/2',10),('single','1/4',20),('multi','1/8',30),('s4','1/8',40)]]
        return dict(kind='ARTIFICIAL',binding=binding(),cell_id='pr',law_id='law',groups=groups,
            required_strata=[x['stratum'] for x in groups],partition_proof=dict(law_id='law',
                disjoint_exhaustive=True,strata=[x['stratum'] for x in groups],SOURCE='a'*40,sha256='b'*64))

    def test_complete_partition_weights_by_original_mass(self):
        r=c.conditional_diagnostic(self.fixture(),c.Permit('B_ANALYSIS',synthetic=True))
        self.assertEqual(F(r['unconditional_mean_RZ_exact']),F(75,4))
        self.assertIsNone(r['unconditional_SE'])

    def test_missing_stratum_not_zero_filled(self):
        d=self.fixture();d['groups'].pop()
        r=c.conditional_diagnostic(d,c.Permit('B_ANALYSIS',synthetic=True))
        self.assertIsNone(r['unconditional_mean_RZ']);self.assertEqual(r['missing_strata'],['s4'])

    def test_missing_original_mass_not_estimated(self):
        d=self.fixture();d['groups'][0]['probability']=None
        self.assertIsNone(c.conditional_diagnostic(d,c.Permit('B_ANALYSIS',synthetic=True))['unconditional_mean_RZ'])

    def test_incompatible_K_or_candidate_rejected(self):
        for key in ('cell_id','law_id'):
            d=self.fixture();d['groups'][1][key]='other'
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'ORIGINAL_LAW'):
                c.conditional_diagnostic(d,c.Permit('B_ANALYSIS',synthetic=True))

    def test_conditional_renormalized_masses_refused(self):
        d=self.fixture();d['groups'][0]['mass_kind']='CONDITIONAL_PROBABILITY'
        with self.assertRaisesRegex(ValueError,'RENORMALIZATION'):
            c.conditional_diagnostic(d,c.Permit('B_ANALYSIS',synthetic=True))


class Gates(unittest.TestCase):
    def test_nine_independent_invalid_grants(self):
        for s in ('H4','H6','H8'):
            for stage in contract.STAGES:
                grant=contract.invalid_grant(s,stage)
                with self.subTest(system=s,stage=stage),self.assertRaisesRegex(ValueError,'APPROVAL'):
                    contract.authorize(grant,s,stage)

    def test_other_stage_or_system_approval_rejected(self):
        grant=contract.invalid_grant('H4','A_COST')
        for s,t in [('H4','B_ANALYSIS'),('H6','A_COST'),('H4','D_ACCOUNTING')]:
            with self.assertRaisesRegex(ValueError,'INDEPENDENT'):contract.authorize(grant,s,t)

    def test_real_cli_rejects_before_config_science_or_output(self):
        with tempfile.TemporaryDirectory() as d:
            grant=Path(d)/'grant.json';contract.save(grant,contract.invalid_grant('H8','A_COST'))
            p=subprocess.run([sys.executable,'-P','-B',str(ROOT/'scripts/run_pr_cost_integration.py'),
                '--system','H8','--stage','A_COST','--run','--grant',str(grant),'--config',str(Path(d)/'absent')],
                capture_output=True,text=True)
            self.assertEqual(p.returncode,2);self.assertIn('APPROVAL_REQUIRED',p.stderr)
            self.assertEqual([x.name for x in Path(d).iterdir()],['grant.json'])

    def test_budget_no_refund_and_double_reservation_refused(self):
        with tempfile.TemporaryDirectory() as d:
            budget=contract.Budget(d,dict(wrapper_compile=2));budget.take('wrapper_compile');budget.take('wrapper_compile')
            with self.assertRaisesRegex(ValueError,'BUDGET'):budget.take('wrapper_compile')
            self.assertEqual(len(list(Path(d).iterdir())),2)

    def test_first_stop_preserved_and_other_scope_not_touched(self):
        with tempfile.TemporaryDirectory() as d:
            contract.first_stop(d,'FIRST');contract.first_stop(d,'SECOND')
            self.assertEqual(contract.read(Path(d)/'STOP.json')['reason'],'FIRST')

    def test_duplicate_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'file'
            for text in ['{"x":1,"x":2}','{"x":NaN}','{"x":1e999}']:
                p.write_text(text)
                with self.assertRaises(ValueError):contract.read(p)

    def test_home_only_output(self):
        with self.assertRaisesRegex(ValueError,'HOME_ONLY'):contract.home_path('/tmp/forbidden')

    def test_A_identity_and_saved_flags(self):
        cell=dict(id='B2_p1_q2_2nd_r4_K2',method='B2',prefix=1,q=2,r=4,K=2,R=8,formula='2nd')
        tables={str(e):{'1.0':dict(cost_acquisition_eligible=True)} for e in c.EPS}
        signal=dict(cell=cell,precision_rows=tables,empirical_noncancellation_precision_rows=tables)
        self.assertEqual(len(c.acquisition_candidates([signal],[cell])['selected']),1)
        wrong=deepcopy(cell);wrong['r']=8
        with self.assertRaises(ValueError):c.acquisition_candidates([signal],[wrong])

    def test_seed_is_semantic_and_source_scoped(self):
        import pr_cost_native as n
        a=n.seed('H4',binding(),dict(id='artificial'),0,'a'*40)
        self.assertEqual(a,n.seed('H4',binding(),dict(id='artificial'),0,'a'*40))
        self.assertNotEqual(a,n.seed('H4',binding(),dict(id='artificial'),0,'b'*40))

    def test_one_shot_consumed_marker_prevents_retry(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'lock';contract.claim_attempt(path,'a'*40,'ARTIFICIAL_A_COST')
            with self.assertRaises(FileExistsError):contract.claim_attempt(path,'a'*40,'ARTIFICIAL_A_COST')
            self.assertTrue(contract.read(path)['nonrefundable_attempt'])

    def test_H4_original_builder_proposal_is_explicit_not_input_generation(self):
        p=contract.plan('H4','A_COST')['H4_preparation_policy']
        self.assertEqual(p['prefix_reconstructions_proposed'],3)
        self.assertEqual(p['SCF_DF_state_solver'],0)
        self.assertFalse(contract.invalid_grant('H4','A_COST')['runtime_authorization'])


if __name__=='__main__':unittest.main()
