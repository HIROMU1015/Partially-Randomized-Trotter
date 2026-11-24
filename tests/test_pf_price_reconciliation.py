"""PF price decisions reopen for new evidence while preserving prior attempts."""
import json
from pathlib import Path
import gzip
import hashlib
import sqlite3
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from trottertracks import pf_candidate_selection as selection
from trottertracks import pf_saved_prices as prices


ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PFPriceReconciliationTests(unittest.TestCase):
    def test_terminal_reuses_frozen_model_and_preserves_price_provenance(self):
        from trottertracks.pf_distributed_preparation import terminal_model_prices
        from trottertracks.saved_rpe_cost_catalog import enc

        scope = 'CONDITIONAL_COMPARISON_MODEL_WITH_EXPLICIT_LONG_Q_UNKNOWN'
        kind = 'CONDITIONAL_AFFINE_PRICE_REFERENCE'
        gate = 'MODEL_REFERENCE_LONG_Q_UNCERTAINTY_UNKNOWN'
        model_source = dict(commit='a'*40, path='FROZEN_MODEL.json')
        profile = dict(compiler='fixed')
        spec = dict(M=12, delta_pf='1/2', profile=profile)
        previous = dict(m=12, q_PF=8, T=4, actual_delta_pf='1/2', structure={'formula_id': 'fixed'})
        terminal = dict(previous, m=13, q_PF=16, T=8)
        parent = dict(id='parent', spec=spec, rounds=[previous])
        target = dict(id='target', spec=dict(spec, M=13), rounds=[previous, terminal])
        receipt = dict(source={'commit': 'b'*40, 'path': 'PRICES.json'})
        table = dict(configuration_id='parent', model=model_source, qualified_for_candidate=False,
                     scope=scope, holdout=[], axes=[])
        model = dict(configuration_id='parent', holdout_refit=False, long_q_error_bound=None,
                     holdout_round=6, models={})
        for axis, slope in (('X', 2), ('Y', 3)):
            model['models'][axis] = dict(model='PRICE_LINEAR_FEATURE_MODEL_V1', features=['one', 'q'],
                coefficients=dict(one=enc(1), q=enc(slope)),
                domain=dict(route='TRUNCATED_PF', axis=axis, profile=profile, kind='DIRECT_COMPILED_DETERMINISTIC'))
            table['holdout'].append(dict(m=6, axis=axis, q=2, receipt=receipt))
            table['axes'].append(dict(m=6, axis=axis, price_evidence=receipt, gate='C_IMPLEMENTATION_GUARD_PASS'))
            table['axes'].append(dict(m=12, axis=axis, q=8, actual_delta_pf='1/2', kind=kind, gate=gate,
                                      price_evidence=dict(source=model_source, axis=axis)))

        rows = terminal_model_prices(parent, target, 'TRUNCATED_PF', table, model, 'terminal')
        self.assertEqual([row['unit_price_RZ'] for row in rows], [enc(33), enc(49)])
        for row in rows:
            self.assertEqual(row['price_evidence'], dict(source=model_source, axis=row['axis']))
            self.assertEqual((row['m'], row['physical_id'], row['kind'], row['gate']), (13, 'terminal', kind, gate))
            self.assertTrue(row['unvalidated_extrapolation'])
            self.assertIsNone(row['model_error_bound_RZ'])
        with self.assertRaisesRegex(ValueError, 'model, admitted prefix or holdout mismatch'):
            terminal_model_prices(parent, target, 'FULL_PF', table, model, 'terminal')
        terminal['actual_delta_pf'] = '1/4'
        with self.assertRaisesRegex(ValueError, 'adopted model schedule'):
            terminal_model_prices(parent, target, 'TRUNCATED_PF', table, model, 'terminal')

    def test_selected_provider_dispatches_signal_with_owned_price_and_replays_response(self):
        from trottertracks import pf_distributed_preparation as distributed
        from trottertracks.pf_optimizer_connection import complete_selected

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root/'fixed'
            run = root/'run'
            leaf = run/'pf_completion'
            workspace = leaf/'provider_workspace'
            folder = workspace/'docs/research/distributed_selected_pf/cfg'
            entry = original/'scripts/prepare_pf_candidate.py'
            entry.parent.mkdir(parents=True)
            entry.touch()
            request = dict(mode='complete', selected=[{'configuration_body': {'id': 'cfg'}}],
                           output_root=str(run/'output'), registry_file=str(run/'registry.json'))
            spec = dict(methods=['TRUNCATED_PF'], science_approved=True,
                        pf_completion=dict(request, entry=str(entry)))
            write(leaf/'ORIGINAL_OWNER_REQUEST.json', dict(request, prior_owner_directories=[]))
            owned_axis = dict(m=3, axis='X', action='HELD', findings=[{'running_job': {'pid': 4321}}])
            write(leaf/'ORIGINAL_OWNER_CHECK.json', dict(rows=[dict(
                configuration_id='cfg', status='ORIGINAL_OWNER_PREFLIGHT_PASS',
                deferred_price_rounds={3: [owned_axis]})]))
            write(leaf/'PROVIDER_SOURCE.json', dict(root=str(original), source='artificial'))
            write(leaf/'PLAN_RESPONSE.json', dict(items=[dict(status='PLANNED',
                  configuration_id='cfg', output=str(folder))]))
            write(folder/'PLAN.json', dict(configuration={'id': 'cfg'}, new_signal_means=1,
                  direct_prices=[], selected_axis_compile=True,
                  new_compile_cells=[dict(m=3, axes=['X'], role='CALIBRATION')]))
            write(folder/'preflight.METADATA_RESULT.json', dict(source={'commit': 'artificial'}))
            calls = []

            def invoke(context, command, directory, stage, output, *, science):
                calls.append(stage)
                self.assertEqual(stage, 'PF_SIGNALS')
                result = dict(status='SIGNALS_READY', publication='signal-artificial')
                write(output, result)
                return result

            runtime = SimpleNamespace(call=invoke)
            with patch.object(distributed.subprocess, 'check_output', return_value=b'artificial'), \
                 patch.object(distributed.subprocess, 'run'):
                first = complete_selected(root, spec, run, stage_executor=runtime)
                second = complete_selected(root, spec, run, stage_executor=runtime)
            self.assertEqual(calls, ['PF_SIGNALS'])
            self.assertEqual(first, second)
            row, = first['response']['items']
            self.assertEqual(row['independent_signal']['publication'], 'signal-artificial')
            self.assertEqual(row['audited_independent_prices'], [])

    def test_catalog_follows_committed_axis_receipts_once_and_preserves_unknown(self):
        carrier = {'commit': 'a'*40, 'path': 'catalog.json.gz'}
        receipt = {'commit': 'b'*40, 'path': 'batch/PRICES.json', 'selector': [0]}
        missing = {'commit': 'c'*40, 'path': 'missing/PRICES.json'}
        records = {carrier['commit']+':'+carrier['path']: gzip.compress(json.dumps(
            {'p': {'proof': [{'receipt': receipt}, {'receipt': missing}]}}).encode()),
            receipt['commit']+':'+receipt['path']: b'[{"axis":"X"}]'}
        def blob(command, **kwargs):
            if command[-1] not in records:
                raise subprocess.CalledProcessError(128, command)
            return records[command[-1]]
        rows = [dict(physical_id='p', m=1, axis=axis, action='HELD',
                     findings=[dict(source=carrier, key='p')]) for axis in ('X', 'Y')]
        with patch.object(prices.subprocess, 'check_output', side_effect=blob) as reads:
            located = prices.catalog_receipt_locators(ROOT, rows)
        self.assertEqual(reads.call_count, 3)
        self.assertEqual(located['p', 'X'], ([receipt], [missing]))
        self.assertEqual(located['p', 'X'], located['p', 'Y'])
        rows[0].update(checked_receipts=[receipt], unavailable_receipts=[missing])
        rows[1]['findings'].append({'running_job': {'pid': 234}})
        actions = prices.price_axis_actions({'deferred_price_rounds': {1: rows}})
        self.assertEqual([row['action'] for row in actions],
                         ['NO_NEW_CALCULATION_UNEVALUATED', 'RECONCILE_ORIGINAL_ATTEMPT'])
        self.assertTrue(all(not row['new_calculation'] for row in actions))

    def test_legacy_receipt_index_only_reopens_affected_price_plans(self):
        x = dict(commit='a'*40, path='old/PRICES.json', selector=[0])
        y = dict(commit='b'*40, path='other/PRICES.json', selector=[1])
        settings = {'recovered_price_receipts_by_physical_id': {'one': [x], 'two': [y]}}
        self.assertEqual(prices.price_indexes([], settings, {'one'}), [x])
        self.assertEqual(prices.price_indexes([], settings, {'unknown'}), [])
        self.assertEqual(prices.price_indexes([], settings), [x, y])

    def test_original_calibration_requires_exact_plan_completed_compile_and_all_raw_files(self):
        relative = Path('docs/research/distributed_selected_pf/cfg')
        raw = relative/'calibration/run01'
        plan = dict(configuration={'id': 'cfg', 'spec': {'split': {'original_PF_method': 'B0I'}}},
                    prepared={'pending': {'requests': [{'m': 1, 'T': '0.8'}]}},
                    new_compile_cells=[dict(m=1, axes=['Y'], role='EARLY_DIRECT')])
        result = dict(child_exit_code=0, results=[
            dict(ok=True, cell={'m': 1}, wrappers={'sine': {}})])
        records = {relative/'PLAN.json': plan, raw/'RESULT.json': result,
                   raw/'PRICES.json': [{'m': 1, 'axis': 'Y'}],
                   raw/'SIDECARS_UNBOUND.json': [{}], raw/'IMPLEMENTATION_EVIDENCE.json': {}}
        references = {1: [dict(checked_receipts=[dict(commit='a'*40, path=(raw/'PRICES.json').as_posix())])]}
        def blob(command, **kwargs):
            path = Path(command[-1].split(':', 1)[1])
            if path not in records:
                raise subprocess.CalledProcessError(128, command)
            return json.dumps(records[path]).encode()
        with patch.object(prices.subprocess, 'check_output', side_effect=blob):
            compile_receipt = prices.original_calibration(ROOT, plan, references, relative)
            self.assertEqual(compile_receipt['axes_by_round'], {'1': ['Y']})
            result['child_exit_code'] = 1
            self.assertIsNone(prices.original_calibration(ROOT, plan, references, relative))
            result['child_exit_code'] = 0
            del records[raw/'IMPLEMENTATION_EVIDENCE.json']
            self.assertIsNone(prices.original_calibration(ROOT, plan, references, relative))

    def test_catalog_recovery_requires_c_guard_and_populates_price_pool(self):
        physical = {'id': 'p'}
        profile = {'compiler': 'fixed'}
        request = dict(m=3, axis='X', physical_context=dict(physical_candidate=physical, profile=profile),
            measurement_axis='X', observable='X', outcome_convention='fixed',
            quantity='wrapper_cost', units='RZ', P=0)
        origin = dict(commit='b'*40, path='native/RESULT.json')
        value = dict(request, source=origin, identity={'fixed': True}, law_binding={'fixed': True},
            cohort='DIRECT', mean={'rz_count': 17}, classification='DIRECT_COMPILED_DETERMINISTIC',
            expectation='DETERMINISTIC', price_profile=dict(profile, lowering_id='H46_LONG_TIME_NATIVE_COMPOSE'))
        receipt = dict(commit='a'*40, path='native/PRICES.json', selector=[0])
        side = dict(value, price_source=origin)
        guard = dict(status='IMPLEMENTATION_POLICY_CHECKED_NOT_SCIENTIFIC_QUALIFICATION',
                     policy={'id': 'PF_SELECTED_SAME_PHYSICAL_WRAPPER_V1'},
                     checked=[dict(m=3, axis='X', evidence=dict(receipt, path='native/SIDE.json', selector=[]))])
        documents = {'native/PRICES.json': [value], 'native/GUARD_CHECK.json': guard, 'native/SIDE.json': side}
        def read(command, **kwargs):
            return json.dumps(documents[command[-1].split(':', 1)[1]]).encode()
        native = SimpleNamespace(duplicate_check=lambda _, selected: dict(
            rows=[dict(row, action='HELD', findings=[dict(receipts=[receipt])]) for row in selected]))
        evidence = SimpleNamespace(physical_key=lambda candidate: candidate['id'],
                                   ref=lambda commit, path: dict(commit=commit, path=path),
                                   I=SimpleNamespace(price_requests=lambda *_: [request]))
        plan = dict(configuration={'id': 'cfg'}, new_compile_cells=[dict(m=3, axes=['X'])],
                    prepared=dict(pending=dict(requests=[dict(m=3, physical_candidate=physical)])))
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {
                'run_pf_m0_prices': native, 'trottertracks.pf_evidence_preparation': evidence}), \
                patch.object(prices.subprocess, 'check_output', side_effect=read), \
                patch.object(prices.subprocess, 'run', side_effect=lambda command, **kw:
                             SimpleNamespace(stdout=read(command))):
            extra = {}
            report = Path(tmp)/'DUPLICATES.json'
            prices.unresolved_rounds(ROOT, plan, report, extra, discover=True)
            self.assertEqual(json.loads(report.read_bytes())['discovered_admitted_receipts'], [receipt])
            self.assertEqual(extra['p', 'X'][0]['value']['mean'], {'rz_count': 17})
            side['law_binding'] = {'fixed': False}
            prices.unresolved_rounds(ROOT, plan, report, {}, discover=True)
            self.assertEqual(json.loads(report.read_bytes())['discovered_admitted_receipts'], [])

    def test_axis_recovery_uses_only_clear_unmeasured_wrappers(self):
        original = dict(selected_axis_compile=True, maximum_new_compile=4, new_compile_cells=[
            dict(m=2, axes=['X', 'Y'], role='CALIBRATION'), dict(m=4, axes=['X', 'Y'], role='HOLDOUT')])
        overlap = {'2': [dict(axis='X', action='HELD'), dict(axis='Y', action='NEW_COMPILE_MISSING_CONFIRMED')],
                   '4': [dict(axis='X', action='HELD'), dict(axis='Y', action='HELD')]}
        reduced = prices.independent_plan(original, overlap)
        self.assertEqual(reduced['new_compile_cells'], [dict(m=2, axes=['Y'], role='CALIBRATION')])
        self.assertEqual(reduced['maximum_new_compile'], 1)
        self.assertEqual(original['new_compile_cells'][0]['axes'], ['X', 'Y'])

    def test_original_ledger_ownership_is_scoped_to_bound_stage_and_axes(self):
        from trottertracks.rpe_q3_coordinator import digest

        plan = dict(configuration={'id': 'cfg'}, new_signal_means=0,
                    new_compile_cells=[dict(m=m, axes=['X', 'Y']) for m in range(4)],
                    prepared={'pending': {'requests': [
                        dict(m=m, physical_candidate={'id': str(m)}) for m in range(4)]}},
                    selected_axis_compile=True)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            ledger = base/'distributed/QUEUE.sqlite3'
            ledger.parent.mkdir()
            job_root = base/'jobs/PREPARATION:compile'
            with sqlite3.connect(ledger) as db:
                db.execute('create table jobs (job_id text, request text, request_sha256 text, method text, '
                           'state text, attempt_id text, completion_sha256 text)')
                directories = []
                for m, state in enumerate(('UNKNOWN', 'CANCELLED', 'ACCEPTED'), 1):
                    workspace = base/f'owner-{m}'
                    owner = workspace/'cfg'
                    directories.append(owner)
                    payload = owner/'PF_COMPILE.INPUT.json'
                    write(payload, dict(value=plan, rounds=[m], axes_by_round={str(m): ['X']}))
                    result = owner/'PF_COMPILE.RESULT.json'
                    job = f'prep-{m}'
                    request = {'preparation': dict(
                        provider='PF_COMPILE', source={'commit': 'original'},
                        controller_root=str(workspace), inputs={
                            'workspace/cfg/PF_COMPILE.INPUT.json': hashlib.sha256(payload.read_bytes()).hexdigest()},
                        command=['python', '--input', str(payload), '--output', str(result)])}
                    write(owner/'PF_COMPILE.DISTRIBUTED.json', dict(source='original', request=request, job_id=job))
                    stored = dict(request, request_id=job)
                    sha = digest(stored)
                    attempt = None if state == 'CANCELLED' else 'attempt'
                    db.execute('insert into jobs values (?, ?, ?, ?, ?, ?, ?)',
                               (job, json.dumps(stored), sha, 'PREPARATION:compile', state, attempt, 'received'))
                    if state == 'ACCEPTED':
                        write(job_root/job/'OUTCOME.json',
                              dict(status='PREPARATION_FAILED', request_id=job, request_sha256=sha))
                        write(job_root/job/'PROCESS_RETURN.json', dict(returncode=1, signal=None))
                write(directories[0]/'STATUS.json', {'status': 'COMPLETE'})
                write(directories[0]/'STOP.json', dict(stage='audit', rounds=[0], axes_by_round={'0': ['Y']}))
            original = prices.scoped_original_owners(plan, directories)
            blocked, audits = prices.reconcile_original_prices(base, plan, {}, original)
            self.assertEqual(audits, [])
            self.assertEqual({(r['m'], r['axis']): r['findings'][0]['original_owner_state']
                              for rows in blocked.values() for r in rows},
                             {(0, 'Y'): 'STOP', (1, 'X'): 'UNKNOWN', (3, 'X'): 'STOP'})
            self.assertEqual([c['axes'] for c in prices.independent_plan(plan, blocked)['new_compile_cells']],
                             [['X'], ['Y'], ['X', 'Y'], ['Y']])

    def test_reconciled_compile_audits_only_missing_unowned_axes(self):
        plan = dict(configuration={'id': 'cfg'}, selected_axis_compile=True,
                    new_compile_cells=[dict(m=1, axes=['Y']), dict(m=2, axes=['X', 'Y'])],
                    prepared={'pending': {'requests': [
                        dict(m=m, physical_candidate={'id': str(m)}) for m in (1, 2)]}})
        receipt = dict(raw_commit='a'*40, relative='cfg/calibration/run01', rounds=[1])
        owners = dict(holds=[dict(stage='audit', axes=[(2, 'X')], path='old', state='UNKNOWN')],
                      compiles=[receipt])
        raw_prices = [dict(m=1, axis=axis) for axis in ('X', 'Y')]
        raw_sides = [dict(implementation_evidence={'selector': []}) for _ in raw_prices]
        with patch.object(prices, 'raw_compile_documents', return_value=({}, raw_prices, raw_sides, {})):
            blocked, audits = prices.reconcile_original_prices(ROOT, plan, {}, owners)
        self.assertEqual(audits, [dict(compiled=receipt, axes_by_round={'1': ['Y']})])
        self.assertEqual(prices.independent_plan(plan, blocked)['new_compile_cells'], [dict(m=2, axes=['Y'])])

    def test_exploratory_refresh_only_hydrates_new_pf_entries(self):
        from trottertracks import rpe_method_optimizer as optimizer

        old = {'schema': selection.PF.SCHEMA, 'configurations': {'retained': {'id': 'retained'}}}
        received = dict(old, configurations=dict(old['configurations'], new={'id': 'new'}))
        context = dict(facts={}, expected={'retained': {'binding_id': 'same'}},
                       candidates=[dict(configuration_id='new', ready=False)],
                       pf_evidence_registry=old)
        problem = {'methods': {'FULL_PF': {}}, 'search': {'exploratory_cost': True}}

        def hydrate(root, reader, plan, candidates, delta, facts):
            self.assertEqual(delta['configurations'], {'new': {'id': 'new'}})
            candidates[0]['ready'] = True
            return {'new': {'binding_id': 'new'}}

        with patch.object(selection.PF, 'hydrate', side_effect=hydrate), patch.object(optimizer, 'prepare') as prepare:
            actual = selection._refresh(ROOT, problem, context, ['FULL_PF'], received, False)
        prepare.assert_not_called()
        self.assertTrue(actual['candidates'][0]['ready'])
        self.assertFalse(context['candidates'][0]['ready'])
        self.assertEqual(actual['expected'], {'retained': {'binding_id': 'same'}, 'new': {'binding_id': 'new'}})
        self.assertEqual(actual['pf_evidence_registry'], received)

    def test_preflight_pid_exit_retries_metadata_and_other_failures_hold_one_candidate(self):
        from trottertracks.pf_distributed_preparation import run_preflight
        from trottertracks.pr_candidate_completion import Held, PreparationWait

        command = ['fixture-preflight']
        exited = subprocess.CalledProcessError(1, command, stderr=(
            b'  File "pf_price_stages.py", line 147, in active_price_jobs\n'
            b'ProcessLookupError: [Errno 3] No such process\n'))
        result = subprocess.CompletedProcess(command, 0)
        with patch.object(subprocess, 'run', side_effect=[exited, result]) as call:
            self.assertEqual(run_preflight(command), result)
        self.assertEqual(call.call_count, 2)
        with patch.object(subprocess, 'run', side_effect=exited):
            with self.assertRaises(PreparationWait):
                run_preflight(command)
        rejected = subprocess.CalledProcessError(1, command, stderr=b'Owner proof differs')
        with patch.object(subprocess, 'run', side_effect=rejected) as call:
            with self.assertRaisesRegex(Held, 'Owner proof differs'):
                run_preflight(command)
        self.assertEqual(call.call_count, 1)

    def test_normal_completion_uses_saved_price_and_adopts_each_needed_stage_once(self):
        from trottertracks import pf_distributed_preparation as distributed
        from trottertracks.pf_optimizer_connection import complete_selected
        from trottertracks.rpe_q3_coordinator import digest
        from trottertracks.rpe_q3_store import JobStore

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'run'
            leaf = run/'pf_completion'
            workspace = leaf/'provider_workspace'
            folder = workspace/'docs/research/distributed_selected_pf/cfg'
            entry = root/'fixed/scripts/prepare_pf_candidate.py'
            entry.parent.mkdir(parents=True)
            entry.touch()
            request = dict(mode='complete', exploratory_cost=True, selected_axis_compile=True,
                           selected=[dict(configuration_body={'id': 'cfg'})],
                           output_root=str(run/'output'), registry_file=str(run/'registry.json'))
            spec = dict(methods=['FULL_PF'], science_approved=True, pf_completion=dict(request, entry=str(entry)))
            write(leaf/'ORIGINAL_OWNER_REQUEST.json', dict(request, prior_owner_directories=[]))
            write(leaf/'PROVIDER_SOURCE.json', dict(root=str(entry.parents[1]), source='artificial'))
            write(leaf/'PLAN_RESPONSE.json', {'items': [dict(status='PLAN_ONLY',
                configuration_id='cfg', output=str(folder))]})
            saved = dict(m=1, axis='X', physical_id='one', price_evidence={
                'source': {'commit': 'old'}, 'admission': {'path': 'fixture/GUARD_CHECK.json'}})
            plan = dict(configuration={'id': 'cfg'}, selected_axis_compile=True, direct_prices=[saved],
                        calibration_rounds=[1, 2],
                        new_compile_cells=[dict(m=m, axes=axes, role='CALIBRATION')
                                           for m, axes in ((1, ['Y']), (2, ['X', 'Y']))])
            write(folder/'PLAN.json', plan)
            write(folder/'preflight.METADATA_RESULT.json', {'source': {'commit': 'artificial'}})
            write(leaf/'ORIGINAL_OWNER_CHECK.json', {'rows': [dict(
                configuration_id='cfg', status='ORIGINAL_OWNER_PREFLIGHT_PASS',
                original_owner_directories=[], original_signal={'commit': 'saved-signal'},
                original_audits=[dict(compiled={'raw_commit': 'old'}, axes_by_round={'1': ['Y']})],
                deferred_price_rounds={1: [dict(axis='Y', action='HELD')],
                                       2: [dict(axis='X', action='HELD')]})]})
            store = JobStore(run/'distributed/QUEUE.sqlite3')
            submitted = []
            registered = []

            def fake_worker(context, command, directory, stage, output, *, science, wait):
                self.assertTrue(science)
                self.assertFalse(wait)
                payload = json.loads(Path(command[command.index('--input')+1]).read_bytes())
                req = dict(route={'kind': 'artificial'}, stage=stage, payload=payload)
                job = digest(req)[:24]
                self.assertTrue(store.reserve('PREPARATION:'+stage, job, req, 'UNLIMITED', True))
                assignment, = store.claim('fixture', 1, allowed_jobs=[job])
                identity = ('fixture', job, assignment['attempt_id'])
                store.update(*identity, 'RUNNING')
                submitted.append((stage, payload))
                if stage == 'PF_COMPILE':
                    result = dict(rounds=payload['rounds'], axes_by_round=payload['axes_by_round'],
                                  relative='fixture-raw')
                else:
                    result = dict(prices=[dict(m=int(m), axis=axis) for m, axes in
                                          payload['compiled']['axes_by_round'].items() for axis in axes])
                files = {'RESULT.json': result}
                self.assertFalse(store.complete(*identity, assignment['request_sha256'], files)[1])
                self.assertTrue(store.complete(*identity, assignment['request_sha256'], files)[1])
                self.assertTrue(store.adopt(job))
                self.assertFalse(store.adopt(job))
                write(output, result)
                return result

            def metadata(command, **kwargs):
                if command[0] == 'git':
                    return
                payload = json.loads(Path(command[command.index('--input')+1]).read_bytes())
                output = Path(command[command.index('--output')+1])
                if payload['phase'] == 'exploratory-prices':
                    self.assertEqual([(p['m'], p['axis']) for p in payload['prices']],
                                     [(1, 'X'), (1, 'Y'), (2, 'Y')])
                    write(output, {'prices': {'commit': 'fixture-prices', 'path': 'PRICES'}})
                else:
                    self.assertEqual(payload['phase'], 'register')
                    registered.append(payload['prices'])
                    write(workspace/payload['registry'], {'configurations': {'cfg': {
                        'signal_packet': {'commit': 'saved-signal'}, 'q3_inputs': {'commit': 'fixture-q3'},
                        'prices': payload['prices']}}})
                    write(output, dict(status='READY_FOR_SAME_OPTIMIZER', configuration_id='cfg'))

            recovered = folder/'recovered_prices/old'
            recovered.mkdir(parents=True)
            runtime = SimpleNamespace(call=fake_worker)
            with patch.object(distributed.subprocess, 'check_output', return_value=b'artificial'), \
                 patch.object(distributed.subprocess, 'run', side_effect=metadata), \
                 patch.object(distributed, 'reuse_signal') as reuse_signal, \
                 patch.object(distributed, 'track_collected_prices'), \
                 patch.object(distributed, 'stage_original_audit', return_value=(recovered, {'axes_by_round': {'1': ['Y']}})):
                first = complete_selected(root, spec, run, stage_executor=runtime)
                second = complete_selected(root, spec, run, stage_executor=runtime)
            self.assertEqual(first, second)
            reuse_signal.assert_called_once()
            self.assertEqual(len(registered), 1)
            self.assertEqual([stage for stage, _ in submitted], ['PF_AUDIT', 'PF_COMPILE', 'PF_AUDIT'])
            self.assertEqual(submitted[1][1]['axes_by_round'], {'2': ['Y']})
            with sqlite3.connect(store.path) as db:
                self.assertEqual(db.execute("select event,count(*) from events where event in "
                                            "('RUNNING','ADOPT') group by event order by event").fetchall(),
                                 [('ADOPT', 3), ('RUNNING', 3)])

    def test_terminal_failure_supersedes_old_reservation_message(self):
        reason = "DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:PF_COMPILE:{'job_id': 'job'}"
        evidence = dict(action='PRESERVE_FAILED_ATTEMPT', job_id='job', attempt_id='old')
        records = {'candidate': dict(status='PREPARATION_INCOMPLETE', reason=reason)}
        runtime = SimpleNamespace(hold_evidence=lambda *_: evidence,
                                  received_after_reservation_stop=lambda *_: self.fail('Retried terminal attempt'))
        for _ in range(2):
            selection.reconcile_received_preparation(Path('/unused'), records, runtime)
        row = records['candidate']
        self.assertEqual(row['previous_preparation_hold'], reason)
        self.assertEqual(row['preparation_recovery'], evidence)
        self.assertTrue(row['reason'].startswith('STAGE_FAILED:PF_COMPILE:'))
        self.assertEqual(row['status'], 'PREPARATION_INCOMPLETE')

    def test_received_stage_reopens_the_exact_pf_leaf(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            base = run/'pf_auto/price-recovery'
            records = {
                'finished': dict(status='PREPARATION_INCOMPLETE', price_receipt_recheck=True,
                                 reason='DISTRIBUTED_PREPARATION_OPERATIONAL_HOLD:PF_COMPILE:received'),
                'failed': dict(status='PREPARATION_INCOMPLETE', reason='STAGE_FAILED:PF_AUDIT:NO_AUTOMATIC_RETRY'),
            }
            runtime = SimpleNamespace(received_after_reservation_stop=lambda path, reason: (
                path == base/'candidates/finished/pf_auto/pf_completion/provider_workspace/docs/research/distributed_selected_pf/finished'
                and reason.endswith(':received')))
            selection.reconcile_received_preparation(run, records, runtime)
            self.assertEqual(records['finished']['status'], 'PREPARATION_WAITING')
            self.assertEqual(records['failed']['status'], 'PREPARATION_INCOMPLETE')

    def test_restored_candidates_are_announced_before_price_reconciliation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            entry = {'saved': {'id': 'saved'}}
            write(run/'pf_auto/WAVES.json', {'saved': {'status': 'READY_RETURNED_TO_SAME_OPTIMIZER',
                                                       'configurations': entry}})
            context = dict(pf_evidence_registry={'configurations': {}}, candidates=[])
            spec = dict(methods=['TRUNCATED_PF'], pf_completion={'mode': 'complete'})
            events = []

            def refresh(root, problem, old, methods, registry, artificial):
                return dict(old, pf_evidence_registry=registry)

            def announce(updated, routes):
                self.assertEqual(updated['pf_evidence_registry']['configurations'], entry)
                events.append('READY')

            with patch.object(selection, '_refresh', side_effect=refresh), \
                 patch.object(selection.PF, 'registry_routes', return_value={}), \
                 patch.object(selection, 'reconsider_prices', side_effect=lambda *args, **kwargs: events.append('RECONCILE')), \
                 patch.object(selection, 'select', return_value={'selected': None}):
                selection.prepare_next(run, {}, context, spec, run, {}, object(), announce=announce)
            self.assertEqual(events, ['READY', 'RECONCILE'])

    def test_pf_wave_announces_first_ready_while_other_candidate_prepares(self):
        from copy import deepcopy
        first = dict(configuration_id='first', key={'formula': '2nd', 'p': 6, 'delta_over_h': '1'},
                     method='TRUNCATED_PF', ready=False, needs=['PRICE'])
        second = dict(first, configuration_id='second', key=dict(first['key'], p=7))
        context = dict(pf_evidence_registry={'configurations': {}}, candidates=[first, second],
                       catalog={'candidates': []})
        spec = dict(methods=['TRUNCATED_PF'], pf_completion={'mode': 'complete', 'max_candidates': 2})
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            announced = []
            ready = threading.Event()

            def prepare(*args, selection, **kwargs):
                cid = selection['selected']['configuration_id']
                if cid == 'second':
                    self.assertTrue(ready.wait(timeout=3))
                return {'pf_evidence_registry': {'configurations': {cid: {'id': cid}}}}, {}, {
                    'status': 'READY_RETURNED_TO_SAME_OPTIMIZER'}

            def announce(updated, routes):
                saved = json.loads((run/'pf_auto/WAVES.json').read_bytes())
                announced.append(updated['pf_evidence_registry']['configurations'])
                if list(announced[-1]) == ['first']:
                    self.assertEqual(saved['first']['status'], 'READY_RETURNED_TO_SAME_OPTIMIZER')
                    self.assertEqual(saved['second']['status'], 'PREPARATION_SCHEDULED')
                    ready.set()

            def refresh(root, problem, old, methods, registry, artificial):
                return dict(old, pf_evidence_registry=deepcopy(registry))

            with patch.object(selection, 'prepare', side_effect=prepare), \
                 patch.object(selection, '_refresh', side_effect=refresh), \
                 patch.object(selection.PF, 'registry_routes', return_value={}):
                selection.prepare_next(run, {}, context, spec, run, {}, object(), announce=announce)
            self.assertEqual([list(snapshot) for snapshot in announced], [['first'], ['first', 'second']])

    def test_candidate_refill_starts_an_independent_family_before_the_slow_leaf_returns(self):
        from copy import deepcopy
        candidates = [dict(configuration_id=str(p), key={'formula': '2nd', 'p': p, 'delta_over_h': '1'},
                           method='TRUNCATED_PF', ready=False, needs=['PRICE']) for p in (6, 7, 8)]
        context = dict(pf_evidence_registry={'configurations': {}}, candidates=candidates,
                       catalog={'candidates': []})
        settings = dict(mode='complete', max_candidates=2, refill_candidate_preparation=True)
        spec = dict(methods=['TRUNCATED_PF'], pf_completion=settings)
        slow = threading.Event()
        replenished = threading.Event()

        def prepare(*args, selection, **kwargs):
            cid = selection['selected']['configuration_id']
            if cid == '7':
                slow.set()
                self.assertTrue(replenished.wait(3))
            elif cid == '6':
                self.assertTrue(slow.wait(3))
            else:
                replenished.set()
            return {'pf_evidence_registry': {'configurations': {cid: {'id': cid}}}}, {}, {
                'status': 'READY_RETURNED_TO_SAME_OPTIMIZER'}

        def refresh(root, problem, old, methods, registry, artificial):
            return dict(old, pf_evidence_registry=deepcopy(registry))

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(selection, 'prepare', side_effect=prepare), \
             patch.object(selection, '_refresh', side_effect=refresh), \
             patch.object(selection.PF, 'registry_routes', return_value={}):
            run = Path(tmp)
            _, _, outcome = selection.prepare_next(run, {}, context, spec, run, {}, object(),
                                                    refill_requested=lambda: True)
        self.assertEqual({row['status'] for row in outcome['records'].values()},
                         {'READY_RETURNED_TO_SAME_OPTIMIZER'})
        self.assertEqual(len(outcome['records']), 3)

    def test_completed_subset_requires_original_stage_end_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            leaf = Path(tmp)
            provider = leaf/'provider_workspace/docs/research/distributed_selected_pf/cfg'
            axes = [{'physical_id': 'circuit', 'axis': axis, 'gate': 'C_IMPLEMENTATION_GUARD_PASS'}
                    for axis in ('X', 'Y')]
            write(leaf/'RESPONSE.json', {'items': [{
                'configuration_id': 'cfg', 'status': 'PRICE_EVIDENCE_REQUIRED',
                'audited_independent_prices': axes}]})
            write(provider/'PF_AUDIT.RESULT.json', dict(status='AUDIT_PASS', prices=axes))
            for stage in ('PF_COMPILE', 'PF_AUDIT'):
                write(provider/(stage+'.DISTRIBUTED.json'), dict(job_id=stage))
                write(provider/(stage+'.RETURN.json'), dict(job_id=stage, returncode=0))
            self.assertEqual(prices.completed_partial({}, leaf, 'cfg'), [['circuit', 'X'], ['circuit', 'Y']])
            write(provider/'PF_COMPILE.RETURN.json', dict(job_id='other', returncode=0))
            self.assertIsNone(prices.completed_partial({}, leaf, 'cfg'))
            write(provider/'PF_COMPILE.RETURN.json', dict(job_id='PF_COMPILE', returncode=0))
            write(provider/'STOP.json', dict(reason='mandatory'))
            self.assertIsNone(prices.completed_partial({}, leaf, 'cfg'))

    def test_new_axis_provider_retains_original_owner_paths_and_stable_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            original = run/'original'
            settings = dict(entry=str(original/'scripts/prepare_pf_candidate.py'))
            spec = dict(pf_price_recheck_provider=dict(source='fixed-axis-source'))
            row = dict(status='PRICE_EVIDENCE_REQUIRED', price_reconciliation={})
            for _ in range(2):
                selection.reconsider_prices(ROOT, run, spec, settings, {'cfg': row})
            self.assertEqual(row['price_recheck_generation'], 1)
            self.assertEqual(row['price_recheck_provider_source'], 'fixed-axis-source')
            self.assertIn(str(original/'.pf_evidence_private/cfg'), row['price_prior_owner_directories'])

    def test_rechecks_when_original_owner_publishes_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            original = run/'original'
            row = dict(status='PRIOR_OWNER_HOLD',
                       price_reconciliation=dict(original_owner_directories=[str(original)]))
            for status, generation in [('PRICE_RECHECK_AVAILABLE', 1), ('PRICE_ACTION_UNAVAILABLE', 1)]:
                selection.reconsider_prices(ROOT, run, {}, {}, {'cfg': row})
                self.assertEqual(row['status'], status)
                self.assertEqual(row['price_recheck_generation'], generation)
                row['status'] = 'PRICE_ACTION_UNAVAILABLE'
            write(original/'COMPLETION.json', {'configuration_id': 'cfg'})
            selection.reconsider_prices(ROOT, run, {}, {}, {'cfg': row})
            self.assertEqual(row['status'], 'PRICE_RECHECK_AVAILABLE')
            self.assertEqual(row['price_recheck_generation'], 2)

    def test_incompatible_receipt_does_not_hold_an_otherwise_clear_round(self):
        physical = {'id': 'physical'}
        profile = {'compiler': 'fixed'}
        requests = [dict(m=3, axis=axis, physical_context=dict(physical_candidate=physical, profile=profile),
                         measurement_axis=axis, observable=axis, outcome_convention='fixed',
                         quantity='wrapper_cost', units='RZ', P=0) for axis in ('X', 'Y')]
        hit = dict(source={'commit':'archived'}, value=dict(requests[0],
                   price_profile=dict(compiler='different', lowering_id='H46_LONG_TIME_NATIVE_COMPOSE'),
                   classification='DIRECT_COMPILED_DETERMINISTIC', expectation='DETERMINISTIC'))
        plan = dict(configuration={'id':'cfg'}, new_compile_cells=[{'m':3}],
                    prepared=dict(pending=dict(requests=[dict(m=3, physical_candidate=physical)])))

        def duplicate_check(_, *, selected):
            return dict(rows=[dict(row, action='NEW_COMPILE_MISSING_CONFIRMED', findings=[])
                              for row in selected])

        native = SimpleNamespace(duplicate_check=duplicate_check)
        evidence = SimpleNamespace(physical_key=lambda candidate: candidate['id'],
                                   I=SimpleNamespace(price_requests=lambda *_: requests))
        with tempfile.TemporaryDirectory() as tmp, patch.dict('sys.modules', {
                'run_pf_m0_prices': native, 'trottertracks.pf_evidence_preparation': evidence}):
            out = Path(tmp)/'AUDIT.json'
            self.assertEqual(prices.unresolved_rounds(ROOT, plan, out, {('physical','X'):[hit]}), {})
            self.assertTrue(all(row['action']=='NEW_COMPILE_MISSING_CONFIRMED'
                                for row in json.loads(out.read_bytes())['rows']))
