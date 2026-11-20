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

    def test_technical_ast_resume_requires_ended_owners_and_preserves_stop(self):
        import sqlite3

        started = []
        invocation = '1' * 32
        gateway = Gateway(self.config, preflight=lambda *args: {'status': 'PASS'},
                          launch=lambda run, *args: started.append(run), resume_ast_failure=invocation)
        prepared = gateway.handle(dict(request(4, 6), action='prepare'))
        run = pathlib.Path(prepared['run_directory'])
        launch = dict(invocation_id=invocation, pid=2147483647)
        ended = dict(invocation_id=invocation, returncode=1)
        history = run/'invocations'/invocation
        for p in (run, history):
            write(p/'LAUNCH.json', launch)
            write(p/'EXIT.json', ended)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'distributed/relay/STATUS.json', dict(stopped=True, pid=2147483647))
        write(run/'distributed/pbs/STATE.json', dict(allocations={}))
        stop = dict(reason='RUNTIME_EXCEPTION', error_type='SystemError',
                    message='AST constructor recursion depth mismatch (synthetic)')
        write(run/'KAPPA_STATE.json', dict(stop=stop))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(dict(
            stop, phase='DRIVER', traceback='pf_optimizer_connection.py'))+'\n')
        (run/'FAST_DRIVER.lock').touch()
        with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
            db.execute('CREATE TABLE jobs (state TEXT, attempt_id TEXT, detail TEXT)')
            db.execute("INSERT INTO jobs VALUES ('RUNNING','attempt','{}')")
        resume = dict(action='resume', instruction='Artificial technical restart')
        with self.assertRaisesRegex(RequestError, 'unresolved execution or recovery'):
            gateway.start(run, resume)
        self.assertEqual(started, [])
        with sqlite3.connect(run/'distributed/QUEUE.sqlite3') as db:
            db.execute("UPDATE jobs SET state='UNKNOWN', detail=?",
                       (json.dumps(dict(automatic_retry=False, resource_released=True)),))
        self.assertEqual(gateway.start(run, resume), 'STARTED')
        self.assertEqual(started, [run])
        self.assertEqual(load(run/'stop-history'/('ast-'+invocation+'.json')), stop)
        self.assertEqual(load(run/'EXIT.json'), ended)
        self.assertEqual(load(history/'EXIT.json'), ended)

    def test_resume_restores_archived_input_after_ownership_guard_and_before_preflight(self):
        import gzip
        from trottertracks.rpe_preparation_retention import file_sha256, restore_resume_archives

        started = []
        state = pathlib.Path(self.config['state_directory'])
        archive = state.parent/'archives/preparation-inputs/retained'
        archive.mkdir(parents=True)
        source = '1'*40
        ref = 'refs/rpe-preparation/input-bundles/inputs-' + '2'*40
        payload = b'artificial encoded pack entry'
        blob = archive/'pack-entries.bin'
        blob.write_bytes(payload)
        sha = file_sha256(blob)

        def preflight(run, problem, spec):
            self.assertEqual((run/'distributed/artifacts'/sha[:2]/sha).read_bytes(), payload)
            return {'status':'PASS'}

        gateway = Gateway(self.config, preflight=preflight, launch=lambda run,*args:started.append(run))
        prepared = gateway.handle(dict(request(4,6),action='prepare'))
        run = pathlib.Path(prepared['run_directory'])
        original = run/'jobs/PREPARATION:signal/artificial/REQUEST.json'
        cache = run/'distributed/artifacts'/sha[:2]/sha
        write(original, {'preparation':{'source':{'commit':source},'input_hashes':[],
                                        'input_git_bundles':[{'ref':ref,'sha256':sha}]}})
        write(archive/'ARCHIVE.json', {'requests':[{'request':str(original), 'request_sha256':file_sha256(original),
                                                   'artifact_sha256':sha}]})
        with gzip.open(archive/(sha+'.recipe.json.gz'),'wt') as file:
            json.dump(dict(sha256=sha,ref=ref,source=source,header='',trailer='',entries=[[0,len(payload)]]),file)
        write(archive.with_name('retained.reclaimed.json'),
              {'archive':str(archive),'bundles':{sha:{'run_id':run.name,'request':str(original),'source_path':str(cache)}}})
        resume = dict(action='resume',run_id=run.name,instruction='Artificial archived input acceptance')
        write(run/'LAUNCH.json', {'invocation_id':'artificial-invocation'})
        with self.assertRaisesRegex(RequestError, 'previous launch has no completion receipt'):
            gateway.handle(resume)
        self.assertFalse(cache.exists())
        self.assertEqual(started, [])
        write(run/'EXIT.json', {'invocation_id':'artificial-invocation','returncode':0})
        result = gateway.handle(resume)
        self.assertEqual(result['status'], 'STARTED')
        self.assertEqual(result['preflight']['preparation_archive_recovery']['restored'], [sha])
        self.assertEqual(started, [run])
        self.assertEqual(restore_resume_archives(state,run), {'restored':[], 'already_present':[sha]})

    def test_projected_dispatch_resume_preserves_unowned_queue_and_failed_invocation(self):
        import sqlite3

        started = []
        invocation = '2'*32
        gateway = Gateway(self.config, preflight=lambda *args: {'status': 'PASS'},
                          launch=lambda run, *args: started.append(run), resume_dispatch_failure=invocation)
        result = gateway.handle(dict(request(4, 6), action='prepare'))
        run = pathlib.Path(result['run_directory'])
        launch = dict(invocation_id=invocation, pid=2147483647)
        ended = dict(invocation_id=invocation, returncode=1)
        history = run/'invocations'/invocation
        for p in (run, history):
            write(p/'LAUNCH.json', launch)
            write(p/'EXIT.json', ended)
        write(run/'NORMAL_DRIVER.json', launch)
        write(run/'distributed/relay/STATUS.json', dict(stopped=True, pid=2147483647))
        write(run/'distributed/pbs/STATE.json', dict(allocations={}))
        stop = dict(reason='RUNTIME_EXCEPTION', error_type='KeyError', message=repr('a'*64))
        write(run/'KAPPA_STATE.json', dict(stop=stop))
        (run/'RUNTIME_ERRORS.jsonl').write_text(json.dumps(dict(stop, phase='DRIVER',
            traceback="in prefetch_standby\nsource = configurations[projection['source_configuration_id']]"))+'\n')
        (run/'FAST_DRIVER.lock').touch()
        database = run/'distributed/QUEUE.sqlite3'
        with sqlite3.connect(database) as db:
            db.execute('CREATE TABLE jobs (state TEXT, attempt_id TEXT, detail TEXT, host_id TEXT,'
                       ' assigned_epoch REAL, started_epoch REAL)')
            db.execute("INSERT INTO jobs VALUES ('QUEUED',NULL,'{}','worker',NULL,NULL)")
        resume = dict(action='resume', instruction='Artificial technical restart')
        with self.assertRaisesRegex(RequestError, 'acquired an attempt'):
            gateway.start(run, resume)
        with sqlite3.connect(database) as db:
            db.execute("UPDATE jobs SET host_id=NULL")
        self.assertEqual(gateway.start(run, resume), 'STARTED')
        self.assertEqual(started, [run])
        self.assertEqual(load(history/'EXIT.json'), ended)
        self.assertEqual(load(run/'stop-history'/('dispatch-'+invocation+'.json')), stop)
        with sqlite3.connect(database) as db:
            self.assertEqual(db.execute("SELECT state FROM jobs").fetchone(), ('QUEUED',))

    def test_status_observes_local_q3_body_before_the_worker_state_report(self):
        from trottertracks.rpe_local0_search import local0_progress
        from trottertracks.rpe_q3_store import JobStore

        run=self.home/'telemetry'
        spool=self.home/'worker'
        store=JobStore(run/'distributed'/'QUEUE.sqlite3')
        task=dict(route=dict(kind='artificial'),configuration_id='ARTIFICIAL')
        store.reserve('PR_PAULI','job',task,1,True)
        row,=store.claim('local',1)
        self.assertEqual(row['state'],'ASSIGNED')
        write(run/'distributed'/'LOCAL_WORKER.json',dict(host_id='local',work_dir=str(spool)))
        attempt=spool/'job'/row['attempt_id'];attempt.mkdir(parents=True)
        (attempt/'Q3_BODY_TIMING.jsonl').write_text('{"event":"Q3_BODY_START"}\n')
        assignment=attempt/'ASSIGNMENT.json'
        owner={key:row[key] for key in ('job_id','attempt_id','request_sha256')}
        write(assignment,dict(owner,request_sha256='not-the-assigned-request'))
        before=local0_progress({}, {}, run)
        self.assertEqual(before['stages'][0]['execution_started'],0)
        write(assignment,owner)
        observed=local0_progress({}, {}, run)
        self.assertEqual(observed['stages'][0]['execution_started'],1)
        self.assertEqual(observed['stages'][0]['execution_completed'],0)
        self.assertEqual(observed['candidates'][0]['state'],'EVALUATION_RUNNING')
        self.assertEqual(store.row('job')['state'],'ASSIGNED')

    def test_start_returns_a_small_duplicate_check_summary_and_keeps_full_evidence(self):
        refused=[dict(job=str(i),reason='DIFFERENT_JOB_KEY_OR_ROUTE') for i in range(100)]
        g=Gateway(self.config,preflight=lambda *args:dict(status='PASS',refused=refused),
                  launch=lambda *args:None)
        result=g.handle(request(4,6))
        self.assertEqual(result['preflight']['refused_count'],100)
        self.assertEqual(len(result['preflight']['refused']),16)
        self.assertEqual(load(result['preflight']['report_path'])['refused'],refused)
        self.assertGreaterEqual(result['preflight']['elapsed_seconds'],0)

    def test_fixed_pbs_only_preview_and_prepare_remove_other_compute_paths(self):
        path = pathlib.Path(self.config['round'])
        spec = load(path)
        distributed = spec.setdefault('executor', {}).setdefault('distributed', {})
        distributed.update(local_worker={'host_id':'local'}, relay={'workers':[]},
            required_hosts=['local'], preparation_profiles={'local':{},'pbs-0':{}},
            pbs={'workers':[{'host_id':'pbs-0'}]})
        write(path, spec)
        gateway = Gateway(self.config)
        value = request(4, 6)
        value['scope']['execution_hosts'] = ['pbs-0']
        preview = gateway.handle(dict(value, action='preview'))
        self.assertEqual(preview['applied']['execution_hosts'], ['pbs-0'])
        gateway.handle(dict(value, action='prepare'))
        run = self.home/'state'/'runs'/preview['run_id']
        pinned = load(run/'ROUND.requested.json')['executor']['distributed']
        self.assertEqual(pinned['preparation_profiles'], {'pbs-0':{}})
        self.assertEqual(pinned['required_hosts'], [])
        self.assertNotIn('local_worker', pinned)
        self.assertNotIn('relay', pinned)

    def test_shared_pbs_walltime_is_displayed_and_pinned_at_prepare(self):
        common = pathlib.Path(self.config['round'])
        round_spec = load(common)
        profile = dict(route_profiles=[dict(route_kind='pauli', view_commit='view',
                                           profile=dict(adapter_manifest=dict(commit='adapter')))])
        round_spec['executor'] = dict(distributed=dict(
            pbs=dict(transport_source='worker', workers=[dict(host_id='pbs-0', profile=profile)])))
        write(common, round_spec)
        path = self.home/'PBS_WALLTIME.json'
        policy = dict(schema='RPE_PBS_WALLTIME_POLICY_V1', id='policy-v1',
                      computation_source=self.config['computation_source'], transport_source='worker',
                      q3=[dict(id='pauli', method='PR_PAULI', route_kind='pauli', view_commit='view',
                               adapter_source='adapter', seconds=14400)], preparation=[])
        write(path, policy)
        self.config['pbs_walltime_policy'] = str(path)
        g = Gateway(self.config)
        selected = request(4, 6)
        preview = g.handle(dict(selected, action='preview'))
        prepared = g.handle(dict(selected, action='prepare'))
        self.assertEqual(preview['run_id'], prepared['run_id'])
        self.assertEqual(preview['progress']['effective']['pbs_walltime']['q3'][0]['seconds'], 14400)
        work = pathlib.Path(prepared['run_directory'])
        pinned = load(work/'ROUND.requested.json')['executor']['distributed']['pbs']
        self.assertEqual(pinned['q3_task_seconds'][0]['seconds'], 14400)
        selected['scope']['pbs_walltime_overrides'] = dict(pauli=15000)
        override = g.handle(dict(selected, action='preview'))
        self.assertEqual(override['applied']['pbs_walltime']['q3'][0]['seconds'], 15000)
        status = g.handle(dict(action='status', run_id=prepared['run_id']))
        self.assertEqual(status['applied']['pbs_walltime']['q3'][0]['seconds'], 14400)

    def test_method_subset_keeps_its_settings_and_all_duplicate_history(self):
        selected = dict(method='PR_PAULI', run_directory='/prior/pauli')
        excluded = dict(method='TRUNCATED_PF', run_directory='/prior/pf')
        round_spec = load(self.config['round'])
        round_spec['q3_budget_parents'] = [selected, excluded]
        history = ['/prior/pauli', '/prior/pf']
        round_spec['executor']['prior_run_dirs'] = history
        selected_hold = dict(method='PR_PAULI', job_id='a'*20)
        excluded_hold = dict(method='TRUNCATED_PF', job_id='b'*20)
        round_spec['executor']['prior_q3_holds'] = [selected_hold, excluded_hold]
        round_spec['search_overrides'] = dict(requested_widths={'PR_DF': ['1/5']})
        round_spec['width_families'] = [dict(method='PR_DF', L_D=8)]
        write(self.config['round'], round_spec)

        _, scoped, _ = resolve(self.config, request(4, 6, 'UNLIMITED'))

        self.assertEqual(scoped['q3_budget_parents'], [selected])
        self.assertEqual(scoped['executor']['prior_run_dirs'], history)
        self.assertEqual(scoped['executor']['prior_q3_holds'], [selected_hold])
        self.assertEqual(scoped['width_families'], [])
        from trottertracks.rpe_schedule_search import requested_widths
        self.assertIsNone(requested_widths(load(self.config['problem']), scoped))
        self.config['entry'] = 'schedule'
        explicit = request(4, 6, 'UNLIMITED')
        explicit['scope']['per_method']['PR_PAULI']['delta_pf'] = [dict(kind='exact', value='1/7')]
        problem, scoped, _ = resolve(self.config, explicit)
        self.assertEqual(requested_widths(problem, scoped)['PR_PAULI'][0]['exact'], '1/7')

    def test_explicit_unlimited_crosses_old_cap_and_uses_rule_termination(self):
        gateway=build(self.config)
        previous=gateway.handle(request(4,4,1))
        source=pathlib.Path(previous['run_directory'])
        old=(source/'ROUND.json').read_bytes()
        current=gateway.handle(request(4,4,'UNLIMITED'))
        parent=pathlib.Path(current['run_directory'])
        applied=load(parent/'ROUND.json')
        self.assertIsNone(applied['max_steps_per_method'])
        self.assertEqual(applied['max_new_q3_per_method'],{'PR_PAULI':'UNLIMITED'})
        self.assertNotEqual(current['run_id'],previous['run_id'])
        self.assertEqual(current['preflight']['reused']['PR_PAULI'],1)
        jobs=load(self.home/'state'/'ARTIFICIAL_CALLS.json')
        self.assertGreater(sum(update['run']==str(parent) for update in jobs),1)
        self.assertEqual(load(parent/'SEARCH_TERMINATION.json')['reason'],'SEARCH_RULE_EXHAUSTED')
        self.assertEqual(gateway.handle(dict(action='resume',run_id=current['run_id'],instruction='Artificial replay'))['status'],
                         'REUSED_COMPLETED')
        self.assertEqual(jobs,load(self.home/'state'/'ARTIFICIAL_CALLS.json'))
        self.assertEqual(old,(source/'ROUND.json').read_bytes())
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
    def test_normal_width_preview_enables_refill_only_for_selected_release(self):
        self.config['entry'] = 'schedule'
        scope = request(4, 6, 2)
        scope['action'] = 'preview'
        scope['scope']['per_method']['PR_PAULI']['delta_pf'] = [dict(kind='exact', value='1/5')]
        without = Gateway(self.config).handle(scope)
        self.config['live_ready_refill'] = True
        with_refill = Gateway(self.config).handle(scope)
        _, spec, _ = resolve(self.config, scope)
        self.assertNotIn('live_ready_refill', without['applied'])
        self.assertTrue(with_refill['applied']['live_ready_refill'])
        self.assertTrue(spec['overlap_ready_batch'])
        self.assertNotEqual(without['run_id'], with_refill['run_id'])
        self.config['work_conserving_search'] = True
        work = Gateway(self.config).handle(scope)
        self.assertTrue(work['applied']['work_conserving_search'])
        self.assertNotEqual(work['run_id'], with_refill['run_id'])

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
    def test_unidentified_rows_do_not_create_a_false_ownership_match(self):
        from unittest.mock import patch
        from rpe_local0_acceptance_fixture import context, O
        gateway = Gateway(self.config)
        prior = self.home/'prior'
        attempt = prior/'jobs'/'PREPARATION:signal'/'unrelated'
        write(attempt/'REQUEST.json', dict(configuration_id=None))
        write(attempt/'LAUNCH.json', dict(pid=1))
        problem = load(self.config['problem'])
        spec = load(self.config['round'])
        spec['executor']['prior_run_dirs'] = [str(prior)]
        candidate_context = context(problem)
        candidate_context['candidates'].append(dict(candidate_context['candidates'][0],
                                                   configuration_id=None, ready=False))
        with patch.object(O, 'prepare', return_value=candidate_context):
            report = gateway.normal_preflight(self.home/'next', problem, spec)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['owned_unresolved_prior_jobs'], [])

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
