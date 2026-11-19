"""Exercise mixed deployment assembly using metadata only; never start a worker."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import prepare_rpe_mixed_df_release as D
from trottertracks.rpe_preparation_worker import select_preparation_profile
from trottertracks.rpe_release_deployment import manifest_reference

OLD_SIGNAL, OLD_PRICE, SIGNAL, PRICE, CONTROL, SCIENCE = [c * 40 for c in 'abcdef']
RULES = [('signal', 'DF_SIGNALS', SIGNAL, OLD_SIGNAL),
         ('compile', 'DF_CALIBRATION', PRICE, OLD_PRICE),
         ('compile', 'DF_HOLDOUT', PRICE, OLD_PRICE),
         ('audit', 'DF_MODEL_FREEZE', PRICE, OLD_PRICE),
         ('audit', 'DF_HOLDOUT_COMPARE', PRICE, OLD_PRICE)]


class MixedRelease(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.hosts = ['local', 'server2', 'pbs-0', 'offline']
        self.base_profiles = {}
        layouts, profiles = {}, {}
        for host in self.hosts:
            core = dict(host_id=host, root='/synthetic/' + host + '/q3',
                        interpreter='/synthetic/' + host + '/python',
                        work_dir='/synthetic/' + host + '/work', slots='auto',
                        slot_policy={'mode': 'auto'}, artifact_cache_directory='/synthetic/' + host + '/artifacts',
                        preparation_metadata_cache_directory='/synthetic/' + host + '/metadata',
                        storage_limit={'max_bytes': 123}, route_views={SCIENCE: '/synthetic/q3'})
            core['preparation_profiles'] = [dict(stage=s, provider=p, source_commit=o,
                profile=dict(host_id=host, interpreter=core['interpreter'],
                             preparation_sources={o: '/synthetic/' + host + '/' + o},
                             route_views={SCIENCE: '/synthetic/q3'}, work_dir='/old'))
                for s, p, _, o in RULES]
            self.base_profiles[host] = deepcopy(core)
            profiles[host] = str(self.write(host + '/PROFILE.json', core))
            layouts[host] = str(self.write(host + '/HOST.json', dict(host_id=host,
                component_roots={'optimizer': '/synthetic/' + host + '/releases/old'},
                manifest_path='/synthetic/' + host + '/metadata/old/RELEASE_COMPONENTS.json',
                runtimes={'optimizer': {'python': core['interpreter']}},
                admission={'enabled': host != 'offline'})))
        manifest = dict(schema='RPE_RELEASE_COMPONENTS_V1', release_id='old',
            components={'optimizer': {'source': OLD_SIGNAL, 'hosts': self.hosts},
                        'q3': {'source': SCIENCE, 'hosts': self.hosts}},
            stage_components={'signal': ['q3'], 'compile': [], 'audit': []}, hosts=self.hosts,
            runtime_contracts={h: [] for h in self.hosts})
        self.old_policy = dict(id='old-policy', q3=[{'id': 'old-q3', 'method': 'BASE', 'seconds': 90}],
                              preparation=[{'id': 'old-prep', 'stage': 'signal', 'source': OLD_SIGNAL,
                                            'provider': 'OTHER', 'seconds': 60}])
        self.base_round = dict(science_approved=False, methods=['PR_DF', 'PR_PAULI'],
            q3_budget_parents=['protected-parent'], protected_stops_from=['frozen-stop'],
            initial={'shots': [12, 20]}, width_families=[{'L_D': 4}],
            candidate_completion=dict(df_signal_source=OLD_SIGNAL, df_price_source=OLD_PRICE,
                saved_df_descriptors={'8': {'commit': OLD_SIGNAL, 'path': 'old-descriptor'}},
                resume_preparation_from={'commit': OLD_SIGNAL, 'path': 'old-state'},
                protected_stops_from=['old-stop'], budgets={'signal': 0}),
            executor=dict(prior_run_dirs=['old-prior'], distributed=dict(
                preparation_profiles=profiles, preparation_provider_hosts={'PAULI_COMPILE': ['local']},
                host_admission='/synthetic/ADMISSION.json', auto_allocation={'priority_groups': [self.hosts]},
                preparation={'bundle_validation_sources': {OLD_SIGNAL: '/synthetic/old'}},
                local_worker=deepcopy(self.base_profiles['local']),
                relay={'workers': [dict(host_id='server2', python='/synthetic/server2/transport-python',
                    ssh_alias='synthetic-server2', root='/synthetic/old-optimizer',
                    profile='/synthetic/old-profile', deployment={'entry': '/synthetic/old-entry',
                    'manifest': '/synthetic/old-manifest', 'layout': '/synthetic/old-layout'}),
                    dict(host_id='offline', enabled=False, disabled_reason='retain offline')]},
                pbs={'workers': [{'host_id': 'pbs-0', 'profile': deepcopy(self.base_profiles['pbs-0'])}],
                     'deployment': {'manifest': '/synthetic/existing-manifest'}})))
        self.base = dict(code_source=OLD_SIGNAL, code_root='/synthetic/old-controller',
            computation_source=SCIENCE, python='/synthetic/python',
            round=str(self.write('ROUND.json', self.base_round)),
            absolute_kappa_route={'round': str(self.write('ABSOLUTE.json', self.base_round))},
            release_manifest=str(self.write('RELEASE_COMPONENTS.json', manifest)), host_layouts=layouts,
            pbs_walltime_policy=str(self.write('POLICY.json', self.old_policy)),
            state_directory='/synthetic/state', scaled_shots_default=True,
            preparation_budget_defaults={'signal': 0}, publication={'push': False}, fixed_settings={})
        accepted_pbs = deepcopy(self.base_profiles['pbs-0'])
        accepted_pbs.update(work_dir='/isolated/work', artifact_cache_directory='/isolated/cache')
        for row, (_, _, source, _) in zip(accepted_pbs['preparation_profiles'], RULES):
            row['source_commit'] = source
            row['profile'].update(interpreter='/synthetic/pbs/chemistry-python',
                                  preparation_sources={source: '/synthetic/pbs/tested/' + source})
        accepted_round = dict(candidate_completion=dict(df_signal_source=SIGNAL, df_price_source=PRICE,
            df_price_openblas_coretype='NEHALEM', df_descriptor_replacement={'forbidden': True},
            resume_preparation_from={'path': 'scoped-state'}), executor={'distributed': dict(
                preparation_profiles={'pbs-0': accepted_pbs}, preparation=dict(
                    bundle_validation_sources={SIGNAL: '/synthetic/signal', PRICE: '/synthetic/price'},
                    df_snapshot_directory='/synthetic/saved-inputs', bundle_fixed_source_only=True))})
        self.common = dict(id='df-common', method='PR_DF', route_kind='prdf_d2_common_entry',
                           view_commit=SCIENCE, adapter_source=OLD_SIGNAL, seconds=14400, memory_bytes=8 * 1024**3)
        self.new_policy = dict(q3=[self.common], preparation=[dict(
            id=p.lower(), stage=s, provider=p, source=source,
            seconds=120 if s == 'signal' else 3600 if s == 'compile' else 600)
            for s, p, source, _ in RULES])
        self.accepted = dict(code_source=CONTROL, code_root='/synthetic/new-controller',
            computation_source=SCIENCE, round=str(self.write('accepted/ROUND.json', accepted_round)),
            pbs_walltime_policy=str(self.write('accepted/POLICY.json', self.new_policy)))
        self.base_path = self.write('SERVICE.json', self.base)
        alternate = deepcopy(self.base_round)
        alternate['alternate_sentinel'] = {'shots': [44], 'scientific_scope': 'fixed-alternate'}
        self.write('ABSOLUTE.json', alternate)
        self.accepted_path = self.write('accepted/SERVICE.json', self.accepted)
        self.output = self.root / 'new-release'
        self.original_bytes = {str(p): p.read_bytes() for p in self.root.rglob('*.json')}

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    def prepare(self, **kwargs):
        with patch.object(subprocess, 'Popen', side_effect=AssertionError('No processes or scientific execution')):
            return D.prepare(self.base_path, self.accepted_path, self.output, 'mixed-test', **kwargs)

    def test_preserves_scientific_scope_and_original_inputs(self):
        report = self.prepare()
        service = D.read(self.output / 'SERVICE.json')
        for key in ('state_directory', 'scaled_shots_default', 'preparation_budget_defaults', 'publication'):
            self.assertEqual(service[key], self.base[key])
        for filename in ('ROUND.template.json', 'ROUND.absolute-kappa.json'):
            value = D.read(self.output / filename)
            for key in ('science_approved', 'methods', 'q3_budget_parents', 'protected_stops_from', 'initial', 'width_families'):
                self.assertEqual(value[key], self.base_round[key])
            completion = deepcopy(value['candidate_completion'])
            completion.update(df_signal_source=OLD_SIGNAL, df_price_source=OLD_PRICE)
            self.assertEqual(completion, self.base_round['candidate_completion'])
            dist = value['executor']['distributed']
            old = self.base_round['executor']['distributed']
            for key in ('host_admission', 'auto_allocation', 'preparation_provider_hosts'):
                self.assertEqual(dist[key], old[key])
            self.assertEqual(value['executor']['prior_run_dirs'], ['old-prior'])
        self.assertEqual(D.read(service['absolute_kappa_route']['round'])['alternate_sentinel'],
                         {'shots': [44], 'scientific_scope': 'fixed-alternate'})
        for path, raw in self.original_bytes.items():
            self.assertEqual(Path(path).read_bytes(), raw)
        self.assertEqual((report['scientific_calls'], report['queue_operations'], report['registry_modified']), (0, 0, False))

    def test_all_mixed_profiles_select_new_fixed_sources_and_keep_host_resources(self):
        self.prepare()
        service = D.read(self.output / 'SERVICE.json')
        manifest = D.read(service['release_manifest'])
        round_ = D.read(service['round'])
        dist = round_['executor']['distributed']
        self.assertEqual(dist['required_release'], manifest_reference(manifest))
        self.assertEqual(dist['local_worker'], D.read(dist['preparation_profiles']['local']))
        self.assertEqual(dist['pbs']['workers'][0]['profile'], D.read(dist['preparation_profiles']['pbs-0']))
        pbs_layout = D.read(service['host_layouts']['pbs-0'])
        self.assertEqual(dist['pbs']['deployment']['manifest'], pbs_layout['manifest_path'])
        self.assertEqual(dist['pbs']['deployment']['layout'], pbs_layout['layout_path'])
        self.assertEqual(set(dist['preparation_profiles']), set(self.hosts))
        for filename in ('ROUND.template.json', 'ROUND.absolute-kappa.json'):
            relay = D.read(self.output / filename)['executor']['distributed']['relay']['workers']
            layout = D.read(service['host_layouts']['server2'])
            self.assertEqual(relay[0]['root'], layout['component_roots']['optimizer'])
            self.assertEqual(relay[0]['profile'], layout['profile_path'])
            self.assertEqual(relay[0]['deployment'], dict(
                entry=str(Path(relay[0]['root']) / 'scripts/manage_rpe_release_deployment.py'),
                manifest=layout['manifest_path'], layout=layout['layout_path']))
            self.assertEqual(relay[0]['python'], '/synthetic/server2/transport-python')
            self.assertEqual(relay[0]['ssh_alias'], 'synthetic-server2')
            self.assertEqual(relay[1], self.base_round['executor']['distributed']['relay']['workers'][1])
        for host in self.hosts:
            profile = D.read(dist['preparation_profiles'][host])
            layout = D.read(service['host_layouts'][host])
            self.assertEqual(layout['admission']['enabled'], host != 'offline')
            for stage, provider, source, _ in RULES:
                selected = select_preparation_profile(profile, {'preparation': dict(stage=stage, provider=provider, source={'commit': source})})
                kind = 'signal' if stage == 'signal' else 'price'
                component = 'portable_df_' + kind
                self.assertEqual(selected['preparation_sources'][source], layout['component_roots'][component])
                self.assertEqual(manifest['components'][component]['source'], source)
                self.assertIn(host, manifest['components'][component]['hosts'])
                self.assertIn(component, manifest['stage_components'][stage])
                for key in ('work_dir', 'slots', 'slot_policy', 'artifact_cache_directory', 'preparation_metadata_cache_directory', 'storage_limit'):
                    self.assertEqual(selected[key], self.base_profiles[host][key])
                self.assertEqual(selected['interpreter'], '/synthetic/pbs/chemistry-python' if host == 'pbs-0' else self.base_profiles[host]['interpreter'])

    def test_pbs_estimates_preserve_prior_and_add_exact_accepted_rules(self):
        self.prepare()
        policy = D.read(self.output / 'PBS_WALLTIME_POLICY.json')
        for group in ('q3', 'preparation'):
            self.assertEqual(policy[group][:len(self.old_policy[group])], self.old_policy[group])
            additions = deepcopy(policy[group][len(self.old_policy[group]):])
            for row in additions:
                self.assertTrue(row['id'].startswith('mixed-'))
                row['id'] = row['id'][len('mixed-'):]
            self.assertEqual(additions, self.new_policy[group])

    def test_rejects_existing_output_without_overwriting(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('protected')
        with self.assertRaisesRegex(ValueError, 'new output directory'):
            self.prepare()
        self.assertEqual(marker.read_text(), 'protected')

    def test_rejects_scientific_source_mismatch_before_writing(self):
        self.accepted['computation_source'] = '0' * 40
        self.write('accepted/SERVICE.json', self.accepted)
        with self.assertRaisesRegex(ValueError, 'Scientific Q3 SOURCE differs'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_rejects_alternate_route_with_different_host_profile(self):
        alternate = D.read(self.base['absolute_kappa_route']['round'])
        original = deepcopy(self.base_profiles['server2'])
        original['slots'] = 7
        alternate['executor']['distributed']['preparation_profiles']['server2'] = original
        self.write('ABSOLUTE.json', alternate)
        with self.assertRaisesRegex(ValueError, 'Alternate route has different host profiles'):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.assertEqual(D.read(self.base_path), self.base)

    def test_stages_fixed_source_bundle_and_preserves_prior_bundle_references(self):
        bundle = self.root / 'existing-fixed-source.bundle'
        raw = b'synthetic fixed code bundle\x00\xff\n'
        bundle.write_bytes(raw)
        for host, path in self.base['host_layouts'].items():
            layout = D.read(path)
            layout['bundle_files'] = ['/synthetic/' + host + '/existing.bundle']
            Path(path).write_text(json.dumps(layout))
        self.prepare(source_bundle=bundle)
        self.assertEqual((self.output / 'DF_FIXES.bundle').read_bytes(), raw)
        self.assertEqual(bundle.read_bytes(), raw)
        service = D.read(self.output / 'SERVICE.json')
        for host, path in service['host_layouts'].items():
            layout = D.read(path)
            metadata = self.output if host == 'local' else Path('/synthetic') / host / 'metadata/mixed-test'
            self.assertEqual(layout['bundle_files'], [
                '/synthetic/' + host + '/existing.bundle', str(metadata / 'DF_FIXES.bundle')])


if __name__ == '__main__':
    unittest.main()
