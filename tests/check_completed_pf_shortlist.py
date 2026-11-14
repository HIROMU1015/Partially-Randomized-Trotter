"""Artificial-only regression for the reused-intake metadata overlay."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_pf_metadata', ROOT/'scripts/accept_completed_pf_shortlist.py')
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def fixture():
    source = dict(fixture='ARTIFICIAL_ONLY')
    item = dict(candidate='ARTIFICIAL_FULL', configuration_id='old', total_view_id='total',
        B_intake='PASS', configuration=source, signal_packet=source,
        unified_numerical_SOURCE=source, all_rounds=list(range(13)))
    evidence = dict(configuration_id='old', total_view_configuration_id='total', intake='PASS',
        science_calls=0, mean_values_preserved=True, configuration_source=source, packet_source=source)
    config = dict(id='old', spec=dict(law='deterministic', binding='H', profile='P'), rounds=[])
    packet = dict(configuration_id='old', source=source, target='T', means=[])
    candidate = dict(configuration_id='old', law='deterministic', binding='H', compiler_profile='P')
    rows = []
    for m in range(13):
        r = dict(m=m, identity=dict(m=m), law_binding=dict(m=m, law='deterministic'),
                 candidate=dict(id=str(m)), structure=dict(signed=[1]))
        config['rounds'].append(r)
        packet['means'].append(dict(m=m, identity=r['identity'], law_binding=r['law_binding'],
            source=source, target='T', qualification='UNQUALIFIED', physical_candidate=dict(id='p'+str(m)),
            provider_binding=dict(input_source=source), raw=dict(real=1, imag=0), normalization=dict(B=1)))
        for axis in ('X', 'Y'):
            rows.append(dict(m=m, axis=axis, physical_candidate=dict(id='p'+str(m)),
                original_canonical_id=str(m), structure=dict(canonical=r['structure'])))
    return item, evidence, config, packet, candidate, rows


class MetadataTests(unittest.TestCase):
    def test_whole_packet_preserved_no_qualification_or_science(self):
        args = fixture(); original = deepcopy(args)
        out = M.check(*args)
        self.assertEqual(args, original)
        self.assertEqual(len(out['rounds']), 13)
        self.assertEqual(out['original_qualification'], 'UNQUALIFIED')
        for key in ('MSE', 'N', 'sufficient_N', 'G'):
            self.assertIsNone(out[key])

    def test_source_canonical_law_and_physical_mixing_rejected(self):
        for mutation in ('source', 'canonical', 'law', 'physical', 'coverage'):
            with self.subTest(mutation=mutation):
                args = fixture()
                if mutation == 'source': args[1]['packet_source'] = dict(fixture='FOREIGN')
                elif mutation == 'canonical': args[0]['total_view_id'] = 'FOREIGN'
                elif mutation == 'law': args[4]['law'] = 'FOREIGN'
                elif mutation == 'physical': args[5][0]['physical_candidate'] = dict(id='FOREIGN')
                else: args[3]['means'].pop()
                with self.assertRaises(ValueError): M.check(*args)


if __name__ == '__main__': unittest.main(verbosity=2)
