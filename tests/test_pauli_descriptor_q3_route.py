"""Focused negative metadata checks using the published descriptor fixture.

Run the old fixed intake in a subprocess so its module namespace cannot replace
the current controller's modules. All carrier means below are artificial and
the published fixture forbids scientific functions.
"""
from pathlib import Path
import json
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
VIEW = ROOT.parent/'d2-auto-b-intake-33e324b'
PYTHON = Path('/home/AbeHiromu/projects/h6-environments/20261012/native-dependency-v2/venv/bin/python')


def isolated_checks():
    from copy import deepcopy
    sys.path[:0] = [str(VIEW/'src'), str(VIEW/'tests')]
    import check_pauli_ld4_ld6_intake as F
    A, C, T = F.A, F.C, F.T
    repo = F.Repository(A.DESCRIPTOR_ENTRY_SOURCE)
    checks = dict(scientific_calls=0, fixture='ARTIFICIAL_CARRIERS_ONLY', intake_source=F.PUBLIC_B)
    def rejected(name, function):
        try:
            function()
        except ValueError as exc:
            checks[name] = dict(rejected=True, reason=str(exc))
        else:
            raise AssertionError(name+' was accepted')
    try:
        config, handoff, ref = F.fixture(repo, 4, A.DESCRIPTOR_ENTRY_SOURCE)
        with F.metadata_only():
            accepted = A.accept_saved(repo.root, integration=C.interfaces(str(repo.root)).pipeline, handoff_source=ref)
            item = accepted['results'][0]
            packet = handoff['packets'][0]
            store = A.p.SavedArtifacts(repo.root)
            records = {row['m']:F.A._selected(store,row['mean_source']) for row in packet['rounds']}
            dictionary = store.read(packet['dictionary_identity']['source'])
            bad_dictionary = deepcopy(dictionary);bad_dictionary['prefix'] = 5
            rejected('bad_dictionary', lambda:A.bind_packet(config, packet, records, bad_dictionary,
                target=item['signal_packet']['target'], target_source=item['signal_packet']['target_view_source'],
                assembly_source=ref, helpers=A._ld3_metadata_helpers(str(repo.root))))
            api = F.b2_api(adapted=True)
            binding = T.create(repo.root, config, original_source=F.configuration_source(4))
            request = dict(request_id='SYNTHETIC_DESCRIPTOR_INPUT_ONLY', configuration_id=config['id'],
                N_X=[2]*13, N_Y=[3]*13,
                route=dict(contract_template=F.CONTRACT, configurations={config['id']:dict(
                    configuration=F.configuration_source(4), signal_packet=ref, binding_id=binding['id'],
                    total_configuration_id=binding['total_configuration']['id'])}))
            case, inputs = api.common_case(repo.root, request)
            allocation = deepcopy(case['allocation']);allocation['source'] = dict(fixture='ARTIFICIAL_METADATA_ONLY')
            contract = deepcopy(inputs['statistical_contract']);contract['source'] = allocation['source']
            bad_target = deepcopy(case['evaluation_target']);bad_target['original_target_source'] = dict(commit='WRONG',path='bad.json')
            rejected('bad_evaluation_target_contract', lambda:T.K.prepare(config, item['signal_packet'], [allocation],
                evaluation_target=bad_target, statistical_contract=contract, integration=C.interfaces(str(repo.root)).pipeline))
            bad_contract = deepcopy(contract);bad_contract['source'] = None
            rejected('missing_statistical_contract_source', lambda:T.K.prepare(config, item['signal_packet'], [allocation],
                evaluation_target=case['evaluation_target'], statistical_contract=bad_contract,
                integration=C.interfaces(str(repo.root)).pipeline))
            dependency = repo.root/'src/trottertracks/pauli_split_descriptor.py'
            dependency.write_bytes(dependency.read_bytes()+b'\n# artificial incompatible dependency\n')
            bad_ref = repo.save(handoff,'bad_descriptor_source.json')
            rejected('bad_assembly_SOURCE', lambda:A.accept_saved(repo.root,
                integration=C.interfaces(str(repo.root)).pipeline, handoff_source=bad_ref))
            checks['valid_descriptor_metadata'] = 'PASS_WITH_ARTIFICIAL_CARRIERS'
            checks['intake_source'] = '33e324b43690caeaa353ecdb9e3d42f3139198b6'
        print(json.dumps(checks,ensure_ascii=False))
    finally:
        repo.close()


class DescriptorRouteNegatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (VIEW/'tests/check_pauli_ld4_ld6_intake.py').exists() or not PYTHON.exists():
            raise unittest.SkipTest('fixed published fixture/runtime is unavailable')
        output = subprocess.check_output([str(PYTHON),'-B',str(Path(__file__).resolve()),'--isolated-checks'], cwd=ROOT, text=True)
        cls.checks = json.loads(output)

    def test_modified_assembly_dependency_is_rejected(self):
        self.assertTrue(self.checks['bad_assembly_SOURCE']['rejected'])
        self.assertIn('entry code differs',self.checks['bad_assembly_SOURCE']['reason'])

    def test_wrong_dictionary_is_rejected(self):
        self.assertTrue(self.checks['bad_dictionary']['rejected'])
        self.assertIn('dictionary',self.checks['bad_dictionary']['reason'])

    def test_changed_evaluation_and_missing_statistical_contract_are_rejected(self):
        self.assertTrue(self.checks['bad_evaluation_target_contract']['rejected'])
        self.assertTrue(self.checks['missing_statistical_contract_source']['rejected'])
        self.assertEqual(self.checks['scientific_calls'],0)


if __name__ == '__main__':
    if '--isolated-checks' in sys.argv: isolated_checks()
    else: unittest.main(verbosity=2)
