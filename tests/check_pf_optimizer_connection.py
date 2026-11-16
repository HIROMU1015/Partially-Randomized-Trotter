"""Artificial registry/normal-entry checks. No scientific calls or H4 loading."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import pf_optimizer_connection as PF
from trottertracks import rpe_method_optimizer as O
from trottertracks import pf_candidate_space as P
from trottertracks.saved_rpe_cost_catalog import Catalog


def loc(path):
    return dict(commit='ARTIFICIAL_SOURCE', path=path)


def enc(n, d=1):
    return dict(numerator=n, denominator=d)


class Reader:
    def __init__(self, docs):
        self.docs = docs
        self.root = ROOT

    def commit(self, value):
        return value

    def read(self, source):
        value = deepcopy(self.docs[source['path']])
        for key in source.get('selector', []):
            value = value[key]
        return value

    def document(self, commit, path):
        return deepcopy(self.docs[path])


def fixture(p=10):
    method = 'FULL_PF' if p == 10 else 'TRUNCATED_PF'
    cid = 'ARTIFICIAL_' + str(p)
    cfg = dict(id=cid, spec=dict(L_D=p, formula='2nd', delta_pf=enc(1, 5), h=enc(1, 10),
               binding=dict(rank=10, snapshot='SNAP', hamiltonian='HAM', state='STATE'),
               split={} if p == 10 else dict(original_PF_method='B0I')),
               rounds=[dict(m=m, identity=dict(m=m, p=p), law_binding=dict(p=p, law='deterministic'),
                            q_PF=2 ** m, T=enc(2 ** m, 10)) for m in range(13)])
    ids = {m: 'PHYSICAL_P%d_M%d' % (p, m) for m in range(13)}
    packet = dict(configuration_id=cid, means=[dict(m=m, identity=deepcopy(r['identity']), law_binding=deepcopy(r['law_binding']),
                 physical_candidate=dict(id=ids[m]), status='SIGNAL_AVAILABLE_NOT_RMSE_QUALIFIED')
                 for m, r in enumerate(cfg['rounds'])])
    axes = [dict(m=m, axis=a, physical_id=ids[m], configuration_id=cid, unit_price_RZ=enc((m + 1) * (1 if a == 'X' else 2)),
                 kind='CONDITIONAL_MODEL_REFERENCE', price_evidence=loc('PRICE_ORIGIN')) for m in range(13) for a in PF.AXES]
    q3 = dict(cases=[dict(original_configuration_id=cid, configuration=loc('CONFIG'), signal_packet=loc('SIGNAL'),
                         total_configuration_id='TOTAL_' + cid, binding_id='BIND_' + cid)])
    entry = dict(method=method, p=p, formula='2nd', delta_over_h='2', configuration=loc('CONFIG'), signal_packet=loc('SIGNAL'),
                 prices=loc('PRICES'), q3_inputs=loc('Q3'), total_configuration_id='TOTAL_' + cid,
                 binding_id='BIND_' + cid, price_scope='ARTIFICIAL_CONDITIONAL_ONLY')
    docs = dict(CONFIG=cfg, SIGNAL=packet, PRICES=dict(axes=axes), Q3=q3, INDEX=dict(candidates=[]))
    registry = dict(schema=PF.SCHEMA, configurations={cid: entry})
    problem = dict(methods={method:dict(family='PF', index_method='FULL_PF' if p == 10 else 'B0I_P7',
                   formulas=['2nd'], delta_over_h_range=['1/2', '4'], caps=[4] * 13, p_values=[p])},
                   evidence=dict(pf_candidate_index=loc('INDEX')), generator=dict(commit='ARTIFICIAL', path='x', function='generate'),
                   execution_routes={method:[dict(kind='pf_b_pf6', interpreter=sys.executable, view='ARTIFICIAL', view_commit='ARTIFICIAL',
                                                  configuration_ids='*', templates=[])]},
                   search=dict(decrement_steps=[1], large_step_headroom='9/10', seed_rules=['B_PRICE'], max_steps_per_method=4,
                               max_new_q3_per_method=0), accuracy=dict(epsilon_E_decimal='0.1'), target=dict(energy_ha=-1), rpe=dict(M=12))
    return cid, ids, docs, registry, problem


def generator(caps, prices, budget, rule):
    return dict(shots=list(caps), conditional_cost=O.cost(caps, prices))


class Connection(unittest.TestCase):
    def hydrate(self, p=10, mutate=None, candidates=None):
        cid, ids, docs, registry, problem = fixture(p)
        if mutate:
            mutate(docs, registry)
        candidates = [] if candidates is None else candidates
        with patch.object(PF, 'physical_ids', return_value=ids):
            expected = PF.hydrate(ROOT, Reader(docs), problem, candidates, registry)
        return cid, candidates, expected, registry, problem

    def test_new_and_existing_pf_get_own_prices_and_automatic_seed(self):
        for p in (0, 7, 8, 9, 10):
            cid, cs, expected, registry, problem = self.hydrate(p)
            c = cs[0]
            self.assertTrue(c['ready']);self.assertEqual(c['key']['p'], p)
            self.assertEqual(c['prices'][:2], [Fraction(1), Fraction(2)])
            self.assertEqual(c['price_scope'], 'ARTIFICIAL_CONDITIONAL_ONLY')
            self.assertEqual(expected[cid]['binding_id'], 'BIND_' + cid)
            proposals, _ = O.proposals(cs, {}, Fraction(1, 100), problem['search'], generator)
            self.assertEqual(proposals[0]['origin']['kind'], 'GENERATOR_SEED')
            self.assertEqual(proposals[0]['configuration_id'], cid)
        # Indexed / old PF candidates are not marked generated, but must hydrate.
        cid, ids, docs, reg, problem = fixture()
        existing = dict(method='FULL_PF', family='PF', configuration_id=cid, key=dict(formula='2nd', delta_over_h='2'),
                        caps=[4] * 13, prices=None, signal_source=None, labels=[], saved_points=[],
                        needs=['SIGNAL_13ROUND_PACKET', 'PRICE_26AXIS'])
        _, cs, _, _, _ = self.hydrate(candidates=[existing])
        self.assertEqual(len(cs), 1);self.assertTrue(cs[0]['ready'])

    def test_requested_prefixes_exist_as_missing_before_registry_selection(self):
        _, _, _, _, problem=fixture(7)
        _, _, _, _, full_problem=fixture(10)
        problem['methods'].update(full_problem['methods'])
        problem['methods']['TRUNCATED_PF']['p_values']=[0,7,8,9,10]
        docs=dict(INDEX=dict(candidates=[dict(method=m,formula='2nd',requested_delta_over_h='2')
                                        for m in ('FULL_PF','B0I_P7')]))
        # Canonical configuration generation and saved T1/T2 metadata only.
        out=[]
        with patch.object(P.I,'produce_signals',side_effect=AssertionError('science')):
            PF.enumerate_prefixes(ROOT,Reader(docs),problem,out)
        self.assertEqual(len(out),5)
        self.assertEqual({(c['method'],c['key']['p'])for c in out},
                         {('FULL_PF',10)}|{('TRUNCATED_PF',p)for p in (0,7,8,9)})
        self.assertEqual(len({c['configuration_id']for c in out}),5)
        self.assertTrue(all(not c['ready'] and c['prices'] is None and c['signal_source'] is None for c in out))
        self.assertTrue(all(c['configuration_id']==c['configuration_body']['id']for c in out))
        for c in out:
            self.assertIn('PRICE_26AXIS',c['needs'])
            self.assertEqual(c['priority_evidence']['usage'],'PRIORITY_ONLY_NOT_Q3_ACCEPTANCE_OR_REJECTION')
        full=next(c for c in out if c['method']=='FULL_PF')
        self.assertTrue(full['p10_alias_full_pf'])

    def test_saved_catalog_non_p7_prices_keep_their_own_prefix(self):
        cid, _, docs, _, problem=fixture(8)
        point=dict(configuration_id=cid,method='TRUNCATED_PF',L_D=8,formula='2nd',delta_pf=enc(1,5),h=enc(1,10),
                   label='ARTIFICIAL_P8',point_id='POINT',signal_source=loc('SIGNAL'),N_X=[1]*13,N_Y=[1]*13,
                   conditional_cost=enc(26),price_axes=[dict(m=m,axis=a,unit_price_RZ=enc(1))for m in range(13)for a in PF.AXES])
        out=O.enumerate_candidates(Reader(docs),problem,dict(fragment_count=10),dict(candidates=[point]))
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['key']['p'],8)
        self.assertTrue(out[0]['ready'])
        self.assertEqual(out[0]['prices'],[Fraction(1)]*26)

    def test_partial_does_not_turn_into_ready_or_compute_prices(self):
        def partial(docs, reg):
            e = next(iter(reg['configurations'].values()))
            for k in ('signal_packet', 'prices', 'q3_inputs'):
                e[k] = None
        _, cs, _, _, _ = self.hydrate(8, partial)
        self.assertFalse(cs[0]['ready'])
        self.assertEqual(set(cs[0]['needs']), {'SIGNAL_13ROUND_PACKET', 'PRICE_26AXIS', 'Q3_ROUTE_TEMPLATE_FOR_NEW_CONFIGURATION'})
        self.assertIsNone(cs[0]['prices'])

    def test_prefix_physical_axis_law_and_sources_are_not_interchangeable(self):
        mutations = [
            lambda d, r: d['CONFIG']['spec'].update(L_D=8),
            lambda d, r: d['PRICES']['axes'][0].update(physical_id='OTHER_PREFIX'),
            lambda d, r: d['PRICES']['axes'][1].update(axis='X'),
            lambda d, r: d['SIGNAL']['means'][0].update(law_binding={'other':True}),
            lambda d, r: d['Q3']['cases'][0].update(signal_packet=loc('OTHER_SIGNAL')),
            lambda d, r: d['PRICES']['axes'][0].update(kind='STRUCTURAL_S'),
            lambda d, r: d['PRICES']['axes'][0].update(price_evidence=None),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.hydrate(7, mutate)

    def test_prefix_scope_and_p10_alias_do_not_expand_other_methods(self):
        cid, ids, docs, reg, problem = fixture(8)
        problem['methods']['TRUNCATED_PF']['p_values'] = [7]
        out = []
        PF.hydrate(ROOT, Reader(docs), problem, out, reg)
        self.assertEqual(out, [])
        reg['configurations'][cid]['p'] = 10
        with self.assertRaises(ValueError):
            PF.hydrate(ROOT, Reader(docs), problem, [], reg)

    def test_route_template_uses_existing_pf_worker_and_exact_inputs_selector(self):
        cid, cs, expected, reg, problem = self.hydrate()
        reg['configurations'][cid]['q3_inputs']['selector'] = ['inputs']
        rules = PF.registry_routes(problem['execution_routes'], reg)
        rule = rules['FULL_PF'][0]
        self.assertEqual(rule['kind'], 'pf_b_pf6')
        self.assertEqual(rule['templates'][0]['inputs_selector'], ['inputs'])
        self.assertEqual(rule['configurations'][cid]['binding_id'], expected[cid]['binding_id'])
        self.assertEqual(problem['execution_routes']['FULL_PF'][0]['templates'], [])

    def test_existing_pf_total_view_id_spelling_is_preserved(self):
        def old_pf_case(docs, reg):
            case = docs['Q3']['cases'][0]
            case['total_view_id'] = case.pop('total_configuration_id')
        _, cs, expected, _, _ = self.hydrate(mutate=old_pf_case)
        self.assertTrue(cs[0]['ready'])
        self.assertTrue(expected[cs[0]['configuration_id']]['total_configuration_id'].startswith('TOTAL_'))

    def test_c_catalog_reads_prepared_unit_prices_without_q3_or_total(self):
        cid, ids, docs, reg, problem = fixture()
        cfg=docs['CONFIG'];s=cfg['spec']
        s.update(method='PF',M=12,profile=dict(P=0),law=dict(model='deterministic'))
        s['split']['deterministic_scope']='FULL_H'
        for r in cfg['rounds']:
            r.update(actual_delta_pf=enc(1,5),candidate=dict(id='CANONICAL_M'+str(r['m'])))
        docs['REGISTRY']=reg
        comparison=dict(binding=dict(snapshot='SNAP',hamiltonian='HAM',state='STATE'),h=enc(1,10),M=12,compiler_profile=dict(P=0))
        cat=Catalog(Reader(docs),comparison)
        with patch.object(PF,'physical_ids',return_value=ids):
            cat.add(loc('REGISTRY'))
        result=cat.result();row=result['candidates'][0]
        self.assertEqual(row['status'],'Q3_WAITING')
        self.assertTrue(row['prepared_for_optimizer'])
        self.assertEqual(len(row['price_axes']),26)
        self.assertEqual(row['price_axes'][1]['unit_price_RZ'],enc(2))
        for key in ('q3_status','conditional_cost','N_X','N_Y','allocation'):
            self.assertIsNone(row[key])
        self.assertFalse(row['qualified_for_candidate'])
        self.assertIsNone(result['method_minima']['FULL_PF']['conditional_cost'])

    def test_normal_c_reconcile_integrates_prepared_registry_without_science(self):
        _,ids,docs,registry,_=fixture()
        cfg=docs['CONFIG'];s=cfg['spec']
        s.update(method='PF',M=12,profile=dict(P=0),law=dict(model='deterministic'))
        s['split']['deterministic_scope']='FULL_H'
        for r in cfg['rounds']:
            r.update(actual_delta_pf=enc(1,5),candidate=dict(id='CANONICAL_M'+str(r['m'])))
        path='docs/artificial/C_RESULT_JOIN.json';docs[path]=registry
        source=loc(path)
        comparison=dict(binding=dict(snapshot='SNAP',hamiltonian='HAM',state='STATE'),h=enc(1,10),M=12,compiler_profile=dict(P=0))
        module_spec=importlib.util.spec_from_file_location('pf_normal_c_sync_fixture',ROOT/'scripts/sync_saved_rpe_publications.py')
        sync=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(sync)
        self.assertTrue(sync.schema_supported(registry,path))
        item=dict(source=source,blob='ARTIFICIAL_BLOB',branches=['ARTIFICIAL_D_BRANCH'])
        key=path+'@ARTIFICIAL_BLOB'
        with patch.object(PF,'physical_ids',return_value=ids), \
             patch.object(sync,'git',side_effect=AssertionError('No Git publication/scheduler in reconcile fixture')), \
             patch.object(PF.subprocess,'run',side_effect=AssertionError('No producer or science process permitted')):
            result,records,accepted=sync.reconcile(Reader(docs),dict(comparison=comparison,sources=[]),{key:item},{})
        self.assertEqual(records[key]['status'],'INTEGRATED')
        self.assertEqual(accepted,[source])
        row=result['candidates'][0]
        self.assertEqual(row['status'],'Q3_WAITING')
        self.assertEqual(len(row['price_axes']),26)
        self.assertEqual(row['price_scope'],'ARTIFICIAL_CONDITIONAL_ONLY')
        self.assertIsNone(row['q3_status']);self.assertIsNone(row['conditional_cost'])
        self.assertIsNone(result['method_minima']['FULL_PF']['conditional_cost'])
        self.assertEqual(result['new_Q3'],0)
        self.assertEqual(result['new_compile'],0)
        self.assertEqual(sync.stage_summary(result,list(records.values()))['science_execution'],'NOT_STARTED_BY_SYNC')

    def test_c_sync_metadata_only_optimizer_return_is_reference_only(self):
        spec=importlib.util.spec_from_file_location('pf_c_metadata_sync',ROOT/'scripts/sync_saved_rpe_publications.py')
        sync=importlib.util.module_from_spec(spec);spec.loader.exec_module(sync)
        source=loc('ROUND_RESULT.json');docs={'ROUND_RESULT.json':dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1',methods={'FULL_PF':dict(best=None,new_calls=0)})}
        item=dict(source=source,blob='ARTIFICIAL_BLOB',branches=['ARTIFICIAL'])
        fake=type('EmptyCatalog',(),{'__init__':lambda self,*args:None,'inputs':[],'visited':set(),'result':lambda self:{}})
        with patch.object(sync,'Catalog',fake):
            _,records,accepted=sync.reconcile(Reader(docs),dict(comparison={},sources=[]),{'META':item},{})
        self.assertEqual(records['META']['status'],'REFERENCE_ONLY')
        self.assertFalse(accepted)

    def test_real_canonical_pf_method_dispatches_using_split_metadata(self):
        from trottertracks import pf_evidence_preparation as E
        module_spec=importlib.util.spec_from_file_location('pf_provider_dispatch_fixture',ROOT/'scripts/prepare_pf_candidate.py')
        provider=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(provider)
        calls=[]
        def emitter(route):
            def native_row(physical):
                calls.append((route,deepcopy(physical)))
                return dict(selected_route=route)
            return SimpleNamespace(native_row=native_row)
        # Only the native emitter is replaced. These are the real canonical
        # method='PF' configurations and original split/physical metadata.
        native=SimpleNamespace(ROUTES={m:emitter(m)for m in PF.METHODS})
        with patch.dict(sys.modules,{'run_pf_m0_prices':native}):
            for p in (0,7,8,10):
                method='FULL_PF' if p==10 else 'TRUNCATED_PF'
                cfg=P.configure(ROOT,method,p,'2nd','2')
                self.assertEqual(cfg['spec']['method'],'PF')
                prepared=E.plan(ROOT,cfg,loc('ARTIFICIAL_METADATA_SOURCE'))
                self.assertEqual(prepared['science_calls'],0)
                physical=prepared['pending']['requests'][0]['physical_candidate']
                self.assertEqual(provider.comparison_method(cfg),method)
                self.assertEqual(provider.native_row(cfg,physical)['selected_route'],method)
                self.assertEqual(calls[-1][0],method)

    def test_normal_round_accepts_top_level_completed_registry_without_provider(self):
        cid,ids,docs,registry,problem=fixture()
        reader=Reader(docs)
        facts=dict(fragment_count=10,snapshot_sha256='SNAP',hamiltonian_hash='HAM',state_hash='STATE')
        module_spec=importlib.util.spec_from_file_location('pf_completed_registry_cli_fixture',ROOT/'scripts/run_rpe_method_optimizer.py')
        cli=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);path=root/'completed_registry.json';path.write_text(json.dumps(registry))
            spec=dict(round_id='ARTIFICIAL_COMPLETED_PF_REGISTRY',methods=['FULL_PF'],initial='all',
                      max_new_q3_per_method={'FULL_PF':0},science_approved=False,pf_evidence_registry=str(path))
            round_path=root/'ROUND.json';round_path.write_text(json.dumps(spec))
            args=SimpleNamespace(round=round_path,run_dir=root/'run')
            with patch.object(cli,'ROOT',root),patch.object(cli,'head',return_value='ARTIFICIAL_SOURCE'), \
                 patch.object(O,'hamiltonian_facts',return_value=facts), \
                 patch.object(O,'load_catalog',return_value=(reader,{'comparison':{}},{'candidates':[]})), \
                 patch.object(O,'check_comparison'),patch.object(O,'pinned_function',return_value=generator), \
                 patch.object(PF,'physical_ids',return_value=ids), \
                 patch.object(PF.subprocess,'run',side_effect=AssertionError('No provider/science process allowed')):
                cli.round_mode(args,problem,[])
            result=json.loads((args.run_dir/'ROUND_RESULT.json').read_text())
            self.assertIsNone(result['pf_completion'])
            row=result['methods']['FULL_PF']
            self.assertEqual(len(row['ready_candidates']),1)
            self.assertEqual(row['pending_request']['configuration_id'],cid)
            self.assertEqual(row['pending_request']['origin']['kind'],'GENERATOR_SEED')
            self.assertEqual(row['new_calls'],0)
            self.assertFalse((args.run_dir/'pf_completion').exists())
            self.assertFalse((args.run_dir/'jobs').exists())

    def test_normal_round_runs_explicit_provider_then_same_optimizer_seeds_and_resumes(self):
        cid, ids, docs, reg, problem = fixture()
        reader = Reader(docs)
        facts = dict(fragment_count=10, snapshot_sha256='SNAP', hamiltonian_hash='HAM', state_hash='STATE')
        module_spec = importlib.util.spec_from_file_location('pf_normal_cli_fixture', ROOT / 'scripts/run_rpe_method_optimizer.py')
        cli = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);registry_path = root / 'registry.json';registry_path.write_text(json.dumps(reg))
            provider = root / 'provider.py'
            provider.write_text("import argparse,json,pathlib\n"
                                "p=argparse.ArgumentParser();p.add_argument('--request');p.add_argument('--output');p.add_argument('--mode');a=p.parse_args()\n"
                                "r=json.load(open(a.request));c=pathlib.Path(a.output).parent/'CALLS';c.write_text(str(int(c.read_text())+1) if c.exists() else '1')\n"
                                "json.dump({'status':'ARTIFICIAL_COMPLETE','selected':r['selected'],'science_calls':0},open(a.output,'w'))\n")
            spec = dict(round_id='ARTIFICIAL_PF_NORMAL_ENTRY', methods=['FULL_PF'], initial='all',
                        max_new_q3_per_method={'FULL_PF':0}, science_approved=True,
                        pf_completion=dict(entry=str(provider), interpreter=sys.executable, selected=[dict(configuration_body=docs['CONFIG'])],
                                           registry_file=str(registry_path), output_root=str(root / 'evidence')))
            round_path = root / 'round.json';round_path.write_text(json.dumps(spec))
            args = type('Args', (), dict(round=round_path, run_dir=root / 'run'))()
            with patch.object(cli, 'ROOT', root), patch.object(cli, 'head', return_value='ARTIFICIAL_SOURCE'), \
                 patch.object(O, 'hamiltonian_facts', return_value=facts), \
                 patch.object(O, 'load_catalog', return_value=(reader, {'comparison':{}}, {'candidates':[]})), \
                 patch.object(O, 'check_comparison'), patch.object(O, 'pinned_function', return_value=generator), \
                 patch.object(PF, 'physical_ids', return_value=ids):
                cli.round_mode(args, problem, [])
                first = json.loads((args.run_dir / 'ROUND_RESULT.json').read_text())
                row = first['methods']['FULL_PF']
                self.assertEqual(row['new_calls'], 0)
                self.assertEqual(row['pending_request']['origin']['kind'], 'GENERATOR_SEED')
                self.assertEqual(row['pending_request']['configuration_id'], cid)
                self.assertEqual(len(row['ready_candidates']), 1)
                cli.round_mode(args, problem, [])
            self.assertEqual((args.run_dir / 'pf_completion/CALLS').read_text(), '1')
            self.assertFalse((args.run_dir / 'jobs').exists(), 'No scientific Q3 worker may run in the fixture')
            response = json.loads((args.run_dir / 'pf_completion/RESPONSE.json').read_text())
            self.assertEqual(response['science_calls'], 0)
            spec['pf_completion']['selected'] = []
            with self.assertRaises(ValueError):
                PF.complete_selected(root, spec, root / 'refused')


class PriorJobOwnership(unittest.TestCase):
    """A prior run's live/incomplete job is metadata-only for the new run."""
    route=dict(kind='artificial',interpreter=sys.executable,configuration_ids='*',view_commit='ARTIFICIAL')
    proposal=dict(configuration_id='A',vector=tuple([2]*26),cost=Fraction(1),origin=dict(kind='TEST'))

    def executors(self,base,prior_route=None,current_route=None):
        prior_route=self.route if prior_route is None else prior_route
        current_route=prior_route if current_route is None else current_route
        problem=dict(accuracy=dict(e=1),target=dict(energy_ha=0),rpe=dict(M=12))
        expected=dict(A=dict(total_configuration_id='TOTAL',binding_id='BIND'))
        prior=O.Executor(ROOT,base/'B',dict(M=[prior_route]),expected,problem).for_method('M')
        current=O.Executor(ROOT,base/'D',dict(M=[current_route]),expected,problem,prior_run_dirs=[base/'B']).for_method('M')
        job=prior.job_dir(self.proposal);job.mkdir(parents=True)
        (job/'REQUEST.json').write_text(json.dumps(prior.request(self.proposal,job,prior_route),default=str))
        (job/'LAUNCH.json').write_text(json.dumps(dict(pid=123456789)))
        (job/'PARTIAL.json').write_text(json.dumps(dict(status='OWNED_BY_B')))
        return current,job

    def hashes(self,job):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in job.iterdir()if p.is_file()}

    def test_prior_running_and_stale_jobs_block_prefetch_answer_and_direct_launch_read_only(self):
        for running in (True,False):
            with self.subTest(running=running),tempfile.TemporaryDirectory()as tmp:
                current,prior=self.executors(Path(tmp));before=self.hashes(prior)
                with patch.object(current,'alive',return_value=running), \
                     patch.object(current,'run_adapter',side_effect=AssertionError('Never collect or execute prior-owner job')):
                    self.assertEqual(current.prior_job(self.proposal),prior)
                    self.assertTrue(current.has_job(self.proposal))
                    current.prefetch([self.proposal])
                    answer=current.answer(self.proposal,allow_new=True)
                    self.assertEqual(answer['status'],'IN_PROGRESS' if running else 'STALE_JOB')
                    self.assertEqual(answer['source'],'PRIOR_ROUND_JOB')
                    self.assertEqual(answer['calls'],0)
                    self.assertTrue(answer['owner_review_required'])
                    self.assertEqual(current.launch(self.proposal,self.route),prior)
                self.assertEqual(self.hashes(prior),before)
                self.assertFalse(current.job_dir(self.proposal).exists())
                self.assertEqual(current.launched('M'),0)

    def test_malformed_prior_launch_stays_stale_and_does_not_collect(self):
        with tempfile.TemporaryDirectory()as tmp:
            current,prior=self.executors(Path(tmp))
            (prior/'LAUNCH.json').write_text('{BROKEN_OLD_RECORD')
            before=self.hashes(prior)
            with patch.object(current,'run_adapter',side_effect=AssertionError('No prior collection')):
                self.assertEqual(current.answer(self.proposal,allow_new=True)['status'],'STALE_JOB')
            self.assertEqual(self.hashes(prior),before)
            self.assertFalse(current.job_dir(self.proposal).exists())

    def test_completed_prior_observation_is_ingested_without_mutation(self):
        with tempfile.TemporaryDirectory()as tmp:
            current,prior=self.executors(Path(tmp))
            outcome=dict(status=O.PASS,MSE_lower=enc(1,1000),MSE_upper=enc(1,1000),identity=dict(
                configuration_id='A',allocation=[dict(m=m,N_X=2,N_Y=2)for m in range(13)],
                total_configuration_id='TOTAL',binding_id='BIND',contract_matches=True))
            (prior/'OUTCOME.json').write_text(json.dumps(outcome))
            before=self.hashes(prior)
            with patch.object(current,'run_adapter',side_effect=AssertionError('No prior execution or collection')):
                answer=current.answer(self.proposal,allow_new=True)
            self.assertEqual((answer['status'],answer['source'],answer['calls']),(O.PASS,'PRIOR_ROUND_JOB',0))
            self.assertEqual(self.hashes(prior),before)
            self.assertFalse(current.job_dir(self.proposal).exists())

    def test_unrelated_pf_template_append_preserves_prior_owner_identity(self):
        target=dict(loc('TARGET_TEMPLATE'),inputs_selector=['inputs'])
        unrelated=loc('OTHER_TEMPLATE')
        old=dict(kind='pf_b_pf6',interpreter=sys.executable,configuration_ids='*',view='FIXED_VIEW',view_commit='FIXED_SOURCE',
                 templates=[target])
        new=dict(old,templates=[unrelated,target])
        docs=dict(TARGET_TEMPLATE=dict(inputs=dict(cases=[dict(configuration_id='A')])),
                  OTHER_TEMPLATE=dict(cases=[dict(configuration_id='ANOTHER_CONFIGURATION')]))
        with tempfile.TemporaryDirectory()as tmp:
            current,prior=self.executors(Path(tmp),old,new)
            current.pf_template_reader=Reader(docs);before=self.hashes(prior)
            with patch.object(current,'alive',return_value=True), \
                 patch.object(current,'run_adapter',side_effect=AssertionError('No prior mutation/collection')):
                self.assertTrue(current.same_route(prior,'A'))
                self.assertTrue(current.has_job(self.proposal))
                current.prefetch([self.proposal])
                self.assertEqual(current.answer(self.proposal,allow_new=True)['status'],'IN_PROGRESS')
            self.assertEqual(self.hashes(prior),before)
            self.assertFalse(current.job_dir(self.proposal).exists())
            # Reader uses inputs_selector as the worker does, never the
            # unrelated locator's contents as the selected SOURCE.
            self.assertEqual(current.pf_template_source(new,'A')['path'],'TARGET_TEMPLATE')
            self.assertEqual(current.pf_template_source(new,'A')['selector'],['inputs'])

    def test_changed_or_unresolved_pf_template_view_source_is_held_without_launch(self):
        old=dict(kind='pf_b_pf6',interpreter=sys.executable,configuration_ids='*',view='FIXED_VIEW',view_commit='FIXED_SOURCE',
                 templates=[loc('TARGET_TEMPLATE')])
        variants=[dict(old,templates=[loc('CHANGED_TARGET_TEMPLATE')]),dict(old,view='OTHER_VIEW'),
                  dict(old,view_commit='OTHER_WORKER_SOURCE'),
                  dict(old,templates=[dict(loc('TARGET_TEMPLATE'),commit='CHANGED_TEMPLATE_SOURCE')]),
                  dict(old,templates=[loc('MISSING_TEMPLATE')])]
        docs=dict(TARGET_TEMPLATE=dict(cases=[dict(configuration_id='A')]),
                  CHANGED_TARGET_TEMPLATE=dict(cases=[dict(configuration_id='A')]))
        for new in variants:
            with self.subTest(route=new),tempfile.TemporaryDirectory()as tmp:
                current,prior=self.executors(Path(tmp),old,new)
                current.pf_template_reader=Reader(docs);before=self.hashes(prior)
                with patch.object(current,'alive',return_value=True), \
                     patch.object(current,'run_adapter',side_effect=AssertionError('No execute or collect on changed source')):
                    self.assertFalse(current.same_route(prior,'A'))
                    self.assertTrue(current.has_job(self.proposal))
                    current.prefetch([self.proposal])
                    answer=current.answer(self.proposal,allow_new=True)
                    self.assertEqual(answer['status'],'STALE_JOB')
                    self.assertTrue(answer['owner_review_required'])
                    self.assertEqual(current.launch(self.proposal,new),prior)
                self.assertEqual(self.hashes(prior),before)
                self.assertFalse(current.job_dir(self.proposal).exists())


if __name__ == '__main__':
    unittest.main()
