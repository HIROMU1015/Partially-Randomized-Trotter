import pathlib,sys,tempfile,json,unittest,io,argparse
from copy import deepcopy
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks.rpe_local0_search import Gateway,RequestError,resolve,number,write,load,digest
from rpe_local0_acceptance_fixture import build,seed

def fixture_config(directory):
    p=directory/'problem.json';r=directory/'round.json'
    problem=dict(problem=dict(system='ARTIFICIAL_LOCAL0_SCOPE_ACCEPTANCE_ONLY',hamiltonian_data=dict(expected=dict(fragment_count=10))),
        accuracy=dict(epsilon_E_decimal='1'),target=dict(energy_ha=0),rpe=dict(M=12),
        methods=dict(PR_PAULI=dict(family='PR',L_D_range=dict(min=1,max_offset_from_fragment_count=-1),caps=[4]*13,generation={})),
        search=dict(decrement_steps=[1,2],large_step_headroom='9/10',seed_rules=['B_PRICE'],max_steps_per_method=10,max_new_q3_per_method=0),
        execution_routes=dict(PR_PAULI=[dict(kind='artificial',interpreter=sys.executable,configuration_ids='*',artificial=dict(weights=[1]*13,scale='1',tau='1'))]))
    spec=dict(round_id='ARTIFICIAL_BASE',methods=['PR_PAULI'],science_approved=False,max_new_q3_per_method=dict(PR_PAULI=0),initial='all',executor=dict(prior_run_dirs=[]))
    write(p,problem);write(r,spec)
    import subprocess
    return dict(problem=str(p),round=str(r),entry='method',state_directory=str(directory/'state'),code_root=str(ROOT),
        code_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),computation_source='ARTIFICIAL_FIXED_SOURCE',python=sys.executable,plan_template=False)

def request(lo,hi,budget=0):
    return dict(action='start',instruction='Artificial acceptance: specified range and start',scope=dict(methods=['PR_PAULI'],per_method=dict(PR_PAULI=dict(L_D=dict(min=lo,max=hi))),shots='auto',stop=dict(max_new_q3_per_method=budget)))

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.home=pathlib.Path(self.tmp.name);self.config=fixture_config(self.home)
    def test_partial_overlap_uses_normal_entry_and_retains_unseen_point(self):
        gateway=build(self.config);old=gateway.handle(request(4,6));old_run=pathlib.Path(old['run_directory'])
        prior=seed(old_run,self.config);old_round=(old_run/'ROUND.json').read_bytes()
        current=gateway.handle(request(5,7,1));new_run=pathlib.Path(current['run_directory']);result=load(new_run/'ROUND_RESULT.json')['methods']['PR_PAULI']
        calls=load(pathlib.Path(self.config['state_directory'])/'ARTIFICIAL_CALLS.json')
        self.assertNotEqual(old['run_id'],current['run_id']);self.assertEqual(old_round,(old_run/'ROUND.json').read_bytes())
        self.assertEqual(digest(load(pathlib.Path(prior['job'])/'OUTCOME.json')),prior['outcome_sha256'])
        self.assertEqual(current['preflight']['reused']['PR_PAULI'],1)
        self.assertEqual(result['prior_rounds']['loaded'],1)
        self.assertEqual(sum(c['configuration_id']=='ARTIFICIAL_LD5' for c in calls),0)
        self.assertEqual(len(calls),1);self.assertEqual(calls[0]['configuration_id'],'ARTIFICIAL_LD6')
        self.assertIn('ARTIFICIAL_LD7',result['ready_candidates'])
        self.assertFalse(any(e['configuration_id']=='ARTIFICIAL_LD7' for e in result['evaluated']))
        self.assertIsNotNone(result['pending_request'])
        before=len(calls);again=gateway.handle(request(5,7,1));self.assertEqual(again['run_id'],current['run_id'])
        bare=gateway.handle(dict(action='start',instruction='探索を開始して'));self.assertEqual(bare['run_id'],current['run_id'])
        self.assertEqual(len(load(pathlib.Path(self.config['state_directory'])/'ARTIFICIAL_CALLS.json')),before)
        evidence=dict(status='PASS',old_run=old['run_id'],new_run=current['run_id'],normal_entry='run_rpe_method_optimizer.round_mode',
            reused_at_preflight=current['preflight']['reused'],reused_at_normal_start=result['prior_rounds']['loaded'],same_q3_relaunches=0,
            artificial_new_calls=len(calls),unseen_candidate='ARTIFICIAL_LD7',unseen_remains=True,old_config_and_receipt_unchanged=True,
            same_scope_reuses_run=True,real_scientific_calls=0)
        write(pathlib.Path(self.config['state_directory'])/'OVERLAP_ACCEPTANCE.json',evidence)
    def test_exact_fraction_and_binary64_are_different(self):
        exact=number(dict(kind='exact',value='1/5'),'width');binary=number(dict(kind='binary64',value='0.2'),'width')
        self.assertNotEqual(exact[1]['exact'],binary[1]['exact']);self.assertEqual(exact[1]['exact'],'1/5')
    def test_unsupported_kappa_modes_report_specific_field_and_create_no_run(self):
        for mode in ('absolute','multiplier'):
            q=request(4,7);q['scope']['per_method']['PR_PAULI']['kappa']=dict(mode=mode,min='2',max='64')
            with self.assertRaises(RequestError) as caught:build(self.config).handle(q)
            self.assertIn('kappa.'+mode,caught.exception.field)
        self.assertFalse((pathlib.Path(self.config['state_directory'])/'runs').exists())
    def test_omitted_settings_and_explicit_scope_start_grant(self):
        q=request(4,7);problem,spec,scope=resolve(self.config,q)
        self.assertTrue(spec['science_approved']);self.assertEqual(spec['max_steps_per_method'],10)
        self.assertEqual(problem['methods']['PR_PAULI']['caps'],[4]*13)
    def test_missing_confirmed_target_is_not_guessed(self):
        with self.assertRaisesRegex(RequestError,'no confirmed target'):build(self.config).handle(dict(action='start',instruction='探索を開始して'))
    def test_unknown_fields_do_not_silently_shrink_scope(self):
        q=request(4,7);q['scope']['per_method']['PR_PAULI']['unimplemented_bound']=7
        with self.assertRaises(RequestError):build(self.config).handle(q)
    def test_preview_and_start_have_same_effective_run_id(self):
        g=build(self.config);q=request(4,6);q['action']='preview';preview=g.handle(q)
        self.assertFalse((pathlib.Path(self.config['state_directory'])/'runs').exists())
        q['action']='start';started=g.handle(q);self.assertEqual(preview['run_id'],started['run_id'])
    def test_normal_scope_uses_central_preparation_defaults_and_keeps_explicit_zero(self):
        template = load(self.config['round'])
        template['executor']['distributed'] = dict(preparation=dict(budgets=dict(signal=0, compile=0, audit=0)))
        write(self.config['round'], template)
        self.config['preparation_budget_defaults'] = dict(signal='UNLIMITED', compile='UNLIMITED', audit=3)
        q = request(4, 6)
        q['action'] = 'preview'
        gateway = Gateway(self.config)
        preview = gateway.handle(q)
        self.assertEqual(preview['applied']['preparation_budgets'], self.config['preparation_budget_defaults'])
        self.assertFalse(preview['started'])
        q['scope']['preparation_budgets'] = dict(compile=0)
        limited = gateway.handle(q)
        self.assertEqual(limited['applied']['preparation_budgets'], dict(signal='UNLIMITED', compile=0, audit=3))
        self.assertNotEqual(preview['run_id'], limited['run_id'])
        q['action'] = 'start'
        _, spec, _ = resolve(self.config, q)
        self.assertEqual(spec['executor']['distributed']['preparation']['budgets'], limited['applied']['preparation_budgets'])
        self.assertEqual(spec['max_new_q3_per_method'], {'PR_PAULI': 0})
        self.assertEqual(load(self.config['round']), template)
    def test_run_index_links_to_git_relative_evidence_without_moving_existing_runs(self):
        checkout = self.home/'controller'
        self.config.update(code_root=str(checkout), run_evidence_root=str(checkout/'search-evidence'))
        q = request(4, 6)
        q['action'] = 'prepare'
        gateway = Gateway(self.config)
        result = gateway.handle(q)
        indexed = pathlib.Path(result['run_directory'])
        self.assertEqual(indexed.parent, self.home/'state/runs')
        self.assertTrue(indexed.is_symlink())
        self.assertEqual(indexed.resolve().relative_to(checkout), pathlib.Path('search-evidence', indexed.name))
        self.assertTrue((indexed/'ROUND.requested.json').is_file())
        self.assertEqual(gateway.handle(q)['run_id'], result['run_id'])
    def test_server2_can_confirm_target_without_grant_then_bare_start(self):
        g=build(self.config);q=request(4,6);q['action']='prepare';q.pop('instruction');prepared=g.handle(q)
        run=pathlib.Path(prepared['run_directory']);self.assertFalse((run/'AUTHORIZATION.json').exists());self.assertFalse((run/'LAUNCH.json').exists())
        started=g.handle(dict(action='start',instruction='探索を開始して'))
        self.assertEqual(started['run_id'],prepared['run_id']);self.assertTrue((run/'AUTHORIZATION.json').exists())
    def test_contract_change_does_not_import_prior_answers(self):
        g=build(self.config);old=g.handle(request(4,6));seed(old['run_directory'],self.config)
        p=load(self.config['problem']);p['accuracy']['epsilon_E_decimal']='2';write(self.config['problem'],p)
        changed=g.handle(request(5,7));self.assertEqual(changed['preflight']['reused']['PR_PAULI'],0)
    def test_missing_prior_directory_is_not_treated_as_unseen(self):
        r=load(self.config['round']);r['executor']['prior_run_dirs']=[str(self.home/'missing-source')];write(self.config['round'],r)
        with self.assertRaisesRegex(RequestError,'10.5'):build(self.config).handle(request(5,7))
    def test_existing_launched_unknown_job_is_not_restarted(self):
        g=build(self.config);old=g.handle(request(4,6));prior=seed(old['run_directory'],self.config)
        job=pathlib.Path(prior['job']);(job/'OUTCOME.json').unlink();write(job/'LAUNCH.json',dict(pid=1))
        with self.assertRaisesRegex(RequestError,'10.5'):g.handle(request(5,7))
    def test_corrupt_same_q3_receipt_holds_before_start(self):
        gateway=build(self.config);old=gateway.handle(request(4,6));prior=seed(old['run_directory'],self.config)
        path=pathlib.Path(prior['job'])/'OUTCOME.json';out=load(path);out['identity']['binding_id']='CONFLICT';write(path,out)
        with self.assertRaisesRegex(RequestError,'10.5'):gateway.handle(request(5,7,1))
        calls=load(pathlib.Path(self.config['state_directory'])/'ARTIFICIAL_CALLS.json');self.assertEqual(calls,[])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=pathlib.Path);a=ap.parse_args();OUTPUT=a.output;OUTPUT.mkdir(parents=True,exist_ok=True)
    log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    write(OUTPUT/'TEST_RESULT.json',dict(passed=result.wasSuccessful(),tests_run=result.testsRun,log=log.getvalue()))
    print(log.getvalue());raise SystemExit(0 if result.wasSuccessful() else 1)
