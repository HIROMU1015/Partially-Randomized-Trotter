"""Generic PR-Pauli signal entry on saved LD1/LD2/LD3/LD5 and artificial B2 variants (no new mean).

Saved handoffs and configurations are read from Git. Artificial variants are B ``generate`` outputs
of edited specs; their new rounds are answered by a synthetic stub, never by the provider.
"""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_pauli_signal_entry as RUN  # noqa: E402

E = RUN.E
B2_NEEDS = dict(commit='a4b4d0e1f2110e63ae9a1dc6a89e1699f95dc0eb',
                path='docs/research/rpe_method_optimizer_h4_20261116/run02/NEEDS.json')


def physical(config, m):
    r = config['rounds'][m]; st = r['structure']
    return (r['T'], r['q_PF'], r['actual_delta_pf'], r['R_outer'], st['event_count'], st['allocation'], st['signed_weights'],
            st['formula_id'], r['candidate']['formula'], r['candidate']['K'])


def variant(config, **edit):
    s = deepcopy(config['spec'])
    if 'kappa' in edit:
        k = s['kappa']['value']; f = Fraction(k['numerator'], k['denominator']) * edit['kappa']
        s['kappa']['value'] = dict(numerator=f.numerator, denominator=f.denominator)
    for key in ('delta_pf', 'h'):
        if key in edit:
            f = Fraction(edit[key]); s[key] = dict(numerator=f.numerator, denominator=f.denominator)
    for key in ('finite_K', 'formula'):
        if key in edit:
            s[key] = edit[key]
    return s


class Entry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = E.saved_index(ROOT)
        cls.saved = {}
        for ref in E.SAVED_PACKETS:
            handoff = E.read_ref(ROOT, ref)
            for i, pk in enumerate(handoff['packets']):
                cls.saved[pk['prefix']] = dict(ref=ref, handoff=handoff, index=i, config=pk['configuration'], packet=pk)
        cls.b = E.p1.b_api(ROOT)

    def test_physical_key_ignores_ids_locators_profile_only(self):
        rec = self.saved[3]['packet']['rounds'][12]['mean_record']; key = E.physical_key(rec['physical_context'])[0]
        same = deepcopy(rec['physical_context'])
        same['physical_candidate']['id'] = 'RMSE:other'; same['profile'] = dict(qiskit_version='other')
        same['binding']['compiler'] = 'other'; same['split']['source'] = dict(commit='other')
        same['split']['tail_dictionary']['source'] = dict(commit='other'); same['numerical_recipe']['source'] = dict(commit='x')
        self.assertEqual(E.physical_key(same)[0], key)
        for path, value in [(('physical_candidate', 'R_outer'), 661), (('physical_candidate', 'T'), 204.8),
                            (('physical_candidate', 'delta'), 0.2), (('physical_candidate', 'K'), 4),
                            (('physical_candidate', 'formula'), '4th(m5_best)'), (('structure', 'allocation'), [659]),
                            (('split', 'scalar'), 0.0), (('split', 'lambda_R'), dict(numerator=1, denominator=2)),
                            (('split', 'tail_dictionary', 'dictionary_sha256'), '0'*64), (('binding', 'state'), 'other'),
                            (('law', 'event_model'), 'other'), (('numerical_recipe', 'finite_RTE'), 'other')]:
            changed = deepcopy(rec['physical_context']); target = changed
            for k in path[:-1]:
                target = target[k]
            target[path[-1]] = value
            self.assertNotEqual(E.physical_key(changed)[0], key, path)

    def test_saved_configurations_reach_published_packets(self):
        self.assertEqual(len(self.index), 52)
        for prefix in (1, 2, 3, 5):
            s = self.saved[prefix]
            config, locator = E.load_configuration(ROOT, s['ref'], configuration_id=s['config']['id'])
            self.assertEqual(config, s['config'])
            self.assertEqual(locator['selector'], ['packets', s['index'], 'configuration'])
            with tempfile.TemporaryDirectory() as tmp:
                value, problems = RUN.assemble(Path(tmp)/'out', s['ref'], configuration_id=config['id'])
                summary = (Path(tmp)/'out'/'ROUND_SUMMARY.csv').read_text().splitlines()
            self.assertEqual(problems, [], prefix)
            self.assertEqual(value['status'], 'ALL_13_PAULI_LD%d_MEANS_AVAILABLE_NOT_RPE_QUALIFIED' % prefix)
            self.assertEqual(len(summary), 14)
            pkt = value['packets'][0]
            self.assertEqual(pkt['configuration'], s['config'])
            self.assertEqual(pkt['dictionary_identity'], s['packet']['dictionary_identity'])
            self.assertEqual((pkt['reused_rounds'], pkt['new_rounds'], pkt['evaluated_rounds']), (list(range(13)), [], list(range(13))))
            for mine, published in zip(pkt['rounds'], s['packet']['rounds']):
                self.assertEqual({k: v for k, v in mine.items() if k != 'reused_without_reanalysis'},
                                 {k: v for k, v in published.items() if k != 'reused_without_reanalysis'}, (prefix, mine['m']))
                self.assertIs(mine['reused_without_reanalysis'], True)
            if prefix in (3, 5):
                self.assertEqual(value['configuration_source'], dict(s['ref'], selector=['packets', 0, 'configuration']))
                self.assertEqual(E.read_ref(ROOT, value['configuration_source']), E.read_ref(ROOT, s['handoff']['configuration_source']))
                self.assertEqual(value['B_intake_dictionary_identity'], s['handoff']['B_intake_dictionary_identity'])

    def test_without_own_signals_plan_is_the_split_runs_jobs(self):
        """LD3/LD5 with their own packet hidden: 13 requests whose jobs and inputs are the published runs'."""
        from trottertracks import df_head_pauli_tail_split_signals as F
        for prefix, ref in ((3, E.SAVED_PACKETS[1]), (5, E.SAVED_PACKETS[2])):
            s = self.saved[prefix]
            index = {k: v for k, v in self.index.items() if v['handoff'] != ref}
            pl = E.plan(ROOT, s['config'], index=index)
            self.assertEqual(pl['request_rounds'], list(range(13)))
            self.assertEqual({r['provider_support'] for r in pl['rounds']}, {'SUPPORTED_VALIDATED_FOR_PAULI'})
            for row, saved in zip(pl['rounds'], s['packet']['rounds']):
                self.assertEqual(row['physical_key'], E.physical_key(saved['mean_record']['physical_context'])[0])
            meta = E.L.p0_metadata(ROOT, (2, prefix))
            specific = (E.L.dictionary_item(ROOT, E.S3.P0_LD3['commit'], E.S3.P0_LD3['directory'], 3, meta) if prefix == 3
                        else F.dictionary_item(ROOT, F.SPECS[5], meta))
            generic = E.dictionary_item(ROOT, s['config'])
            self.assertEqual(generic, specific)
            self.assertEqual(pl['rounds'][0]['request']['context']['tail_operators'],
                             dict(E.S3.operators_locator(specific), field='Pauli_nonidentity'))

    def test_artificial_kappa_and_delta_request_only_changed_rounds(self):
        for prefix in (3, 5):
            config = self.saved[prefix]['config']
            for edit in (dict(kappa=Fraction(3, 2)), dict(kappa=Fraction(2, 3)), dict(delta_pf=Fraction(1, 5))):
                v, locator = E.load_configuration(ROOT, variant(config, **edit))
                self.assertIsNone(locator); self.assertNotEqual(v['id'], config['id'])
                changed = [m for m in range(13) if physical(v, m) != physical(config, m)]
                unchanged = [m for m in range(13) if m not in changed]
                pl = E.plan(ROOT, v, index=self.index)
                self.assertTrue(changed and unchanged, (prefix, edit))
                self.assertEqual(pl['request_rounds'], changed, (prefix, edit))
                self.assertEqual(pl['reuse_rounds'], unchanged, (prefix, edit))
                self.assertEqual(pl['rebind_rounds'], unchanged, (prefix, edit))
                self.assertEqual(pl['unsupported_rounds'], [], (prefix, edit))
                for row in pl['rounds']:
                    if row['decision'] == 'REUSE':
                        self.assertEqual(row['saved']['configuration_id'], config['id'])
                        self.assertNotEqual(row['candidate_id'], row['saved']['candidate_id'])
                    else:
                        self.assertTrue(row['changed_from_saved_round']['fields'])

    def test_rebound_and_new_rounds_through_execute_and_assemble(self):
        """LD5 kappa x3/2: rebound carriers for m0-m9, a synthetic stub only for m10-m12, B intake conditions PASS."""
        config = self.saved[5]['config']; saved_rows = self.saved[5]['packet']['rounds']
        spec = variant(config, kappa=Fraction(3, 2)); calls = []
        def factory(jobs):
            def evaluate(m, job):
                calls.append(m); rec = deepcopy(saved_rows[m]['mean_record'])
                rec.update(physical_context=deepcopy(job['physical_context']), candidate=deepcopy(job['physical_context']['physical_candidate']),
                           B_identity=deepcopy(job['B_identity']), B_law_binding=deepcopy(job['B_law_binding']),
                           B_configuration_id=job['B_configuration_id'], B_candidate_id=job['B_candidate_id'], m=m,
                           source=deepcopy(job['physical_context']['numerical_recipe']['source']),
                           signed_weights=job['physical_context']['structure']['signed_weights'],
                           allocation=job['physical_context']['structure']['allocation'],
                           event_count=job['physical_context']['structure']['event_count'], artificial_fixture='SYNTHETIC_STUB_NOT_A_MEAN')
                return rec
            return evaluate
        with tempfile.TemporaryDirectory() as tmp:
            spec_file = Path(tmp)/'spec.json'; spec_file.write_text(json.dumps(spec))
            held = RUN.execute(Path(tmp)/'held', 'SYNTHETIC', str(spec_file), check_source=False, evaluate_factory=factory)
            self.assertEqual((held['status'], held['pending_requests'], held['rebound'], calls),
                             ('STOP_REQUESTS_NOT_APPROVED', [10, 11, 12], list(range(10)), []))
            run = RUN.execute(Path(tmp)/'run', 'SYNTHETIC', str(spec_file), approve=True, check_source=False, evaluate_factory=factory)
            self.assertEqual((run['status'], run['rebound'], calls, run['scope']['new_mean_calls']),
                             (RUN.SUCCESS, list(range(10)), [10, 11, 12], 3))
            value, problems = RUN.assemble(Path(tmp)/'out', str(spec_file), source='SYNTHETIC', raw_commit='SYNTHETIC',
                                           run_dir=str(Path(tmp)/'run') + '/', git=False)
            self.assertEqual(problems, [])
            pkt = value['packets'][0]
            self.assertEqual((pkt['reused_rounds'], pkt['new_rounds'], value['status']),
                             (list(range(10)), [10, 11, 12], 'ALL_13_PAULI_LD5_MEANS_AVAILABLE_NOT_RPE_QUALIFIED'))
            self.assertEqual(pkt['configuration']['id'], run['configuration_id'])
            for row in pkt['rounds']:
                rec = row['mean_record']; m = row['m']
                self.assertEqual(row['candidate_id'], pkt['configuration']['rounds'][m]['candidate']['id'])
                if m < 10:
                    self.assertEqual(rec['signal'], saved_rows[m]['mean_record']['signal'])
                    self.assertEqual(rec['source'], saved_rows[m]['mean_record']['source'])
                    self.assertEqual(rec['reuse']['mean_source'], saved_rows[m]['mean_source'])
                    self.assertEqual(row['mean_source']['scientific_source'], saved_rows[m]['mean_source']['scientific_source'])
                else:
                    self.assertEqual(rec['artificial_fixture'], 'SYNTHETIC_STUB_NOT_A_MEAN')
            broken = deepcopy(value); broken['packets'][0]['rounds'][3]['mean_record']['B_candidate_id'] = 'RMSE:x'
            records = {r['m']: r['mean_record'] for r in broken['packets'][0]['rounds']}
            dictionary = E.read_ref(ROOT, config['spec']['split']['tail_dictionary']['source'])
            self.assertIn('record_job_m3', E.b_intake_check(pkt['configuration'], broken, records, dictionary, self.b))

    def test_provider_support_classes_and_b2_inputs(self):
        config = self.saved[3]['config']
        for edit, support in ((dict(finite_K=4), 'SUPPORTED_NOT_VALIDATED_FOR_PAULI'),
                              (dict(formula='4th(m5_best)'), 'SUPPORTED_NOT_VALIDATED_FOR_PAULI')):
            pl = E.plan(ROOT, E.load_configuration(ROOT, variant(config, **edit))[0], index=self.index)
            self.assertEqual(pl['request_rounds'], list(range(13)), edit)
            self.assertEqual({r['provider_support'] for r in pl['rounds']}, {support}, edit)
        pl = E.plan(ROOT, E.load_configuration(ROOT, variant(config, h=Fraction(1, 20)))[0], index=self.index)
        self.assertEqual(pl['unsupported_rounds'], [0])
        self.assertEqual(pl['rounds'][0]['reasons'], ['NO_SAVED_REFERENCE_AT_THIS_ROUND_T'])
        df = E.L.read(ROOT, E.p0.MENU, E.p0.DIRECTORY+'CONFIGURATION_LD2.json')['spec']
        for key in ('schedule_adapter', 'A_input', 'A_release'):   # as P1 generate_configuration does
            df.pop(key, None)
        df, _ = E.load_configuration(ROOT, df)                      # B canonical DF-tail configuration
        pl = E.plan(ROOT, df, index=self.index)
        self.assertEqual(pl['unsupported_rounds'], list(range(13)))
        self.assertIn('NOT_PAULI_DICTIONARY', pl['configuration_reasons'])
        needs = {c['name']: c for c in E.read_ref(ROOT, B2_NEEDS)['candidates'] if c['method'] == 'PR_PAULI'}
        for name, prefix in (('PR_PAULI:L_D=2:d7dd4f52', 2), ('PR_PAULI:L_D=3:dd5b881d', 3), ('PR_PAULI:L_D=5:4d6e2c9c', 5)):
            self.assertEqual(E.load_configuration(ROOT, needs[name])[0], self.saved[prefix]['config'])
        with self.assertRaisesRegex(ValueError, 'CONFIGURATION_AND_SCHEDULE_REQUIRED_FIRST'):
            E.load_configuration(ROOT, needs['PR_PAULI:L_D=4:NONE'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
