"""Two-time metadata and mock-result tests, no real signal/N/RMSE analysis."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import pr_rpe_rmse_pilot as pilot
from demo_pr_rpe_rmse import artificial_design,mock_provider
from check_pr_rpe_rmse_preflight import inventory as previous_inventory, fixtures as legacy_fixture
A=pilot.A;p=pilot.p
MOCK=dict(fixture='SYNTHETIC_ONLY_NO_REAL_HAMILTONIAN')


def inventory(schedule):
    out=previous_inventory(schedule)
    byid={r['identity']['candidate']['id']:r for r in schedule['rows']}
    for entry in out['means']:
        c=entry['candidate'];request=byid[c['id']]
        entry['law_binding']=dict(definition=dict(model=request['identity']['structure']['law_model'],sampling='IDEAL_FINITE_RTE_MEAN',source=MOCK),
                                  candidate_id=c['id'],R_outer=c['R_outer'],K=c['K'],q_PF=c['q'])
        entry['law_implementation_source']=deepcopy(request['law_binding']['source'])
        entry['law_source_evidence']=MOCK
    return out


def inputs(system='H4',M=None,kappa='1'):
    # Reuse unchanged B's artificial early/signed recipe factory, not its prices.
    path='tests/check_pr_rpe_rmse_integration.py'
    data=subprocess.check_output(['git','show',pilot.B_SOURCE+':'+path],cwd=ROOT)
    node=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='inputs')
    scope=dict(A=A,p=p,SOURCE=MOCK,artificial_design=artificial_design,deepcopy=deepcopy)
    exec(compile(ast.Module(body=[node],type_ignores=[]),path,'exec'),scope)
    args=scope['inputs'](M=(6 if system=='H4' else 8) if M is None else M,kappa=kappa,system=system)
    if system=='H4':args['design']['PF']=dict(method='B0',prefix=7)
    return args


def fixture(system='H4',**kwargs):
    args=inputs(system,**kwargs);schedule=A.schedule(**args);b=pilot.b_metadata(ROOT)
    config=b.generate(b.a_spec(args))
    return config,schedule,inventory(schedule)


def supplied_result(schedule,m):
    r=deepcopy(next(v for v in A.evaluate(schedule)['records'] if v['identity']['m']==m and p.arm(v['identity']['candidate'])=='PR'))
    r.update(nominal_N_X=123,nominal_N_Y=123,status='UNQUALIFIED_MODEL_ASSUMPTIONS')
    return r


class PilotTests(unittest.TestCase):
    def test_two_PR_times_all_round_nulls_and_B_identity(self):
        b,a,i=fixture();out=pilot.prepare_pilot(b,a,i,root=ROOT)
        self.assertEqual(out['pilot_rounds'],[0,6]);self.assertEqual(len(out['rounds']),2)
        self.assertEqual(len(out['N_packet']['rounds']),7)
        self.assertEqual(out['status'],'TECHNICALLY_PREPARED_NOT_SCIENTIFICALLY_QUALIFIED')
        for n,r in zip(out['N_packet']['rounds'],b['rounds']):
            self.assertEqual(n['identity'],r['identity']);self.assertEqual(n['law_binding'],r['law_binding'])
            self.assertIsNone(n['N_X']);self.assertIsNone(n['nominal_N_X'])
        self.assertIsNone(out['N_packet']['rmse_evidence']['rmse_bound_ha'])
        self.assertFalse(out['N_packet']['rmse_evidence']['covers_state_target_alias_prior_decoder_all_rounds'])

    def test_H6_early_and_signed_pieces_preserved_in_B_f09(self):
        b,a,i=fixture('H6');out=pilot.prepare_pilot(b,a,i,root=ROOT)
        self.assertEqual(out['pilot_rounds'],[4,8])
        for r in b['rounds'][:4]:
            self.assertEqual((r['candidate']['prefix'],r['R_outer'],r['q_PF']),(5,32,1))
            self.assertFalse(r['mapping']['kappa_applied'])
        for r in out['rounds']:
            self.assertTrue(any(w<0 for w in r['structure']['signed_weights']))
            self.assertEqual(len(r['structure']['allocation']),11)
            self.assertEqual(sum(r['structure']['allocation']),r['candidate']['R_outer'])

    def test_H4_B0_not_promoted_to_full_H_PF(self):
        args=inputs();b=pilot.b_metadata(ROOT);pf=b.generate(b.a_spec(args,'PF'))
        self.assertEqual(pf['spec']['split']['deterministic_scope'],'A_SOURCE_BOUND_PF_BASELINE')
        self.assertTrue(all(r['physical_candidate']['method']=='B0' and r['candidate']['physical_method']=='B0' for r in pf['rounds']))
        self.assertIsNone(pf['spec']['kappa'])
        with self.assertRaises(ValueError):pilot.prepare_pilot(pf,A.schedule(**args),{},root=ROOT)

    def test_missing_fixed_design_or_horizon_not_defaulted(self):
        with self.assertRaises(ValueError):pilot.prepare_pilot(None,None,{},root=ROOT)
        b,a,i=fixture(M=5)
        with self.assertRaises(ValueError):pilot.prepare_pilot(b,a,i,root=ROOT)

    def test_whole_schedule_M_kappa_and_SOURCE_mismatch_rejected(self):
        b,a,i=fixture()
        for other in (inputs(M=7),inputs(kappa='2')):
            with self.assertRaises(ValueError):pilot.prepare_pilot(b,A.schedule(**other),i,root=ROOT)
        a=deepcopy(a);a['design']['source']={'FOREIGN':'SOURCE'}
        with self.assertRaises(ValueError):pilot.prepare_pilot(b,a,i,root=ROOT)
        b,a,i=fixture();b['rounds'][6]['round_law_definition']['source']={'FOREIGN':'LAW'}
        with self.assertRaises(ValueError):pilot.prepare_pilot(b,a,i,root=ROOT)

    def test_missing_mean_and_pair_plan_only_no_science(self):
        b,a,i=fixture();out=pilot.prepare_pilot(b,a,dict(means=[],references=[]),root=ROOT)
        for r in out['rounds']:
            c=r['future_calls_upper_bound']
            self.assertEqual((c['finite_mean_candidates'],c['exact_tail_internal_controls'],c['independent_reference_actions']),(1,1,2))
        self.assertEqual(out['science_calls'],0)

    def test_saved_FAIL_numeric_unresolved_or_SOURCE_blocks_retry(self):
        b,a,i=fixture()
        for state in ('CONDITIONAL_ROUND_FAIL','NUMERICAL_UNRESOLVED'):
            bad=deepcopy(i);bad['means'][1]['saved_status']=state
            out=pilot.prepare_pilot(b,a,bad,root=ROOT)
            self.assertEqual(out['status'],'BLOCKED_SAVED_EVIDENCE');self.assertIsNone(out['rounds'][0]['future_calls_upper_bound'])
        bad=deepcopy(i);bad['means'][1]['binding']['state']='FOREIGN'
        self.assertIn('SAVED_SOURCE_IDENTITY_MISMATCH',pilot.prepare_pilot(b,a,bad,root=ROOT)['rounds'][0]['issues'])
        bad=deepcopy(i);bad['means'][1]['law_binding']['K']=4
        self.assertIn('SAVED_LAW_SOURCE_MISMATCH',pilot.prepare_pilot(b,a,bad,root=ROOT)['rounds'][0]['issues'])
        bad=deepcopy(i);bad['means'][1].pop('law_binding')
        self.assertIn('SAVED_LAW_EVIDENCE_MISSING_RECOVER_BEFORE_SCIENCE',pilot.prepare_pilot(b,a,bad,root=ROOT)['rounds'][0]['issues'])

    def test_supplied_mock_results_stay_nominal_and_outside_times_null(self):
        b,a,i=fixture();plan=pilot.prepare_pilot(b,a,i,root=ROOT)
        out=pilot.handoff_supplied_results(plan,b,a,[supplied_result(a,m) for m in (0,6)],MOCK,root=ROOT,evidence_domain='SYNTHETIC_ONLY')
        self.assertFalse(out['nominal_promoted_to_sufficient'])
        for r in out['rounds']:
            self.assertIsNone(r['N_X']);self.assertEqual(r['nominal_N_X'],123 if r['m'] in (0,6) else None)
        self.assertFalse(out['pilot_scope']['whole_schedule_RMSE_qualified'])
        self.assertIsNone(out['rmse_evidence']['rmse_bound_ha'])
        partial=pilot.handoff_supplied_results(plan,b,a,[supplied_result(a,0)],MOCK,root=ROOT,evidence_domain='SYNTHETIC_ONLY')
        self.assertIsNone(partial['rounds'][6]['nominal_N_X'])

    def test_foreign_round_PF_HP_or_sufficient_N_cannot_enter(self):
        b,a,i=fixture();plan=pilot.prepare_pilot(b,a,i,root=ROOT)
        bad=supplied_result(a,1)
        with self.assertRaises(ValueError):pilot.handoff_supplied_results(plan,b,a,[bad],MOCK,root=ROOT)
        for change in ({'policy_id':'HOEFFDING'},{'sufficient_N_X':100},{'N_X':100},{'rmse_guaranteed':True}):
            bad=supplied_result(a,0);bad.update(change)
            with self.assertRaises(ValueError):pilot.handoff_supplied_results(plan,b,a,[bad],MOCK,root=ROOT)
        pf=next(v for v in A.evaluate(a)['records'] if p.arm(v['identity']['candidate'])=='PF')
        with self.assertRaises(ValueError):pilot.handoff_supplied_results(plan,b,a,[pf],MOCK,root=ROOT)

    def test_no_scientific_leaf_or_provider_called(self):
        b,a,i=fixture()
        with patch.object(A,'nominal_axis_shots',side_effect=AssertionError('N')),patch.object(A,'existing_finite_mean',side_effect=AssertionError('action')):
            plan=pilot.prepare_pilot(b,a,i,root=ROOT)
            pilot.handoff_supplied_results(plan,b,a,[],MOCK,root=ROOT)
        self.assertFalse(any(k in sys.modules for k in ('numpy','scipy','qiskit')))

    def test_supplied_geometry_unknown_errors_not_zero_or_HP_N(self):
        self.assertIsNone(pilot.supplied_disk_geometry(.5,None,error_source=MOCK)['radius_lower'])
        self.assertIsNone(pilot.supplied_disk_geometry(.5,.1,error_source=None)['beta_deterministic'])
        out=pilot.supplied_disk_geometry(.5,.1,error_source=MOCK)
        self.assertAlmostEqual(out['radius_lower'],.4)
        self.assertAlmostEqual(out['beta_deterministic'],__import__('math').asin(.2))
        self.assertFalse(out['RMSE_qualified']);self.assertIsNone(out['N_X'])
        self.assertEqual(pilot.supplied_disk_geometry(.5,.6,error_source=MOCK)['status'],'INFEASIBLE_SUPPLIED_RADIUS')

    def test_existing_CLI_pilot_and_legacy_mode_unchanged(self):
        b,a,i=fixture()
        with tempfile.TemporaryDirectory(prefix='rmse-pilot-mock-') as tmp:
            t=Path(tmp)
            for name,v in [('b',b),('a',a),('i',i)]:(t/(name+'.json')).write_text(json.dumps(v))
            subprocess.run([sys.executable,'-B','scripts/preflight_pr_rpe_rmse.py','--two-time-pilot','--configuration',str(t/'b.json'),'--schedule',str(t/'a.json'),'--inventory',str(t/'i.json'),'--out',str(t/'out.json')],cwd=ROOT,check=True)
            self.assertEqual(json.loads((t/'out.json').read_text())['pilot_rounds'],[0,6])
            oldb,olda=legacy_fixture(M=0)
            for name,v in [('b',oldb),('a',olda),('i',inventory(olda))]:(t/(name+'.json')).write_text(json.dumps(v))
            subprocess.run([sys.executable,'-B','scripts/preflight_pr_rpe_rmse.py','--configuration',str(t/'b.json'),'--schedule',str(t/'a.json'),'--inventory',str(t/'i.json'),'--out',str(t/'legacy.json')],cwd=ROOT,check=True)
            self.assertEqual(json.loads((t/'legacy.json').read_text())['schema'],'A_RMSE_PREFLIGHT_V1')

    def test_saved_P1_metadata_ids_and_reference_presence_only(self):
        d=pilot.pilot_inventory(ROOT)
        means=[v for v in d['means'] if v['source']['commit']=='71ba49bb28b295aa28e51101bc3b1bf07d500767']
        self.assertEqual(len(means),12)
        self.assertTrue(all(v.get('law_binding') and v.get('law_source_evidence') for v in means))
        expected={('H4',.1):{2,4,8},('H4',6.4):{4,128,256},('H6',1.6):{11,16,32},('H6',25.6):{16,128,256}}
        for (s,T),rs in expected.items():
            matched=[v for v in means if v['candidate']['system']==s and v['candidate']['T']==T]
            self.assertEqual({v['candidate']['R_outer'] for v in matched},rs)
            self.assertTrue(all(v['candidate']['id']==p.candidate_id(v['candidate']) for v in matched))
            self.assertTrue(any(v['binding']['system']==s and v['T']==T and set(pilot.previous.REFERENCE_FIELDS)<=set(v['fields']) for v in d['references']))


if __name__=='__main__':unittest.main()
