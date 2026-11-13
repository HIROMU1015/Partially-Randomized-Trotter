"""Fictional prices/N/sidecars only: C's pinned guard on B's direct entries."""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from check_pr_rpe_rmse_controller import spec, data
from trottertracks import rpe_shot_allocation as W
from trottertracks import rpe_price_implementation_guard as R

B = W.B
G = R.guard(ROOT)
D = G.C.published_api(str(ROOT)).D
MOCK = dict(fixture='SYNTHETIC_ONLY')
CODE = dict(commit='ce0517de6e7b58cf4bf4c69f99f998d86099c406', path='src/trottertracks/pf_pr_rpe_pipeline.py')
NEW, OLD = G.NEW_ZZ, G.OLD_ZZ
NEW_CODE = dict(commit='01e068af7e06d9f1f5621a5b3800c30eb32ecb06', path='src/trottertracks/efficient_controlled_zz.py')
OLD_CODE = dict(commit='c31ef424175c02379eafebe8fc6c7548c1d9aabe', path='src/trotterlib/df_rte_qiskit.py')
RTE = ('PR', 'FULL_H_RTE')
LAYOUTS = {0: dict(q=1, R_outer=2), 1: dict(q=1, R_outer=2)}
SKIP = 'm', 'axis'


def base(method='PR', scientific=False):
    s = spec(method, M=1); s['binding'].update(system='H4', rank=10)
    if scientific:  # fictional context so that only the implementation gate can block cost
        s['binding'].update(geometry='ARTIFICIAL_GEOMETRY', basis='ARTIFICIAL_BASIS')
        s['rmse_policy'].update(scientific_status='EXTERNALLY_QUALIFIED', error_contract=dict(
            fixture='ARTIFICIAL_ONLY', state_assumption='ARTIFICIAL', energy_prior_ha=[-4, 4], decoder='ARTIFICIAL'))
    return s


def configuration(method, scientific=False):
    return deepcopy(_configuration(method, scientific))


@lru_cache(maxsize=None)
def _configuration(method, scientific=False):
    """Each comparison method through the existing B/D generators, one common binding."""
    base_ = lambda kind='PR': base(kind, scientific)
    if method == 'PR':
        return B.generate(base_())
    if method == 'FULL_PF':
        return D.generate(D.full_pf_spec(base_(), formula='2nd', delta_pf=1, M=1, rank=10))
    if method == 'FULL_H_RTE':
        s = base_()
        prep = dict(representation='DF', source=dict(D.SOURCES['full_h_prepare'], function='prepare_full_h'),
            one_body_policy='RANDOMIZED', one_body_components=1, deterministic_blocks=0,
            randomized_fragments=list(range(10)), all_fragments=10, lambda_R=3, one_body_lambda_R=1,
            lambda_source=MOCK, normalization_policy='full_h_df_plus_one_body_exact_Z',
            scalar=dict(constant=0, extracted_identity=0, application='ONCE_AT_TOTAL_T', identity_components_randomized=0),
            preparation_hash='MOCK_PREPARATION', binding=deepcopy(s['binding']))
        return D.generate(D.full_h_rte_spec(s, M=1, finite_K=2, kappa=None, preparation=prep), layout_by_round=LAYOUTS)
    s = base_('PF')
    s['split'].update(deterministic_scope='TRUNCATED_H', original_PF_method='B0I', retained_hamiltonian=dict(
        construction='MOCK_B0I_KEEP_ONE_BODY_AND_PREFIX', prefix=s['L_D'],
        target_hamiltonian=s['binding']['hamiltonian'], source=MOCK))
    return B.generate(s)


def physical(config, row):
    """The same physical context a C/D producer would bind (artificial values)."""
    s = config['spec']; method = B.comparison_method(s); structure = deepcopy(row['structure'])
    if method in ('FULL_PF', 'FULL_H_RTE'):
        layouts = LAYOUTS if method == 'FULL_H_RTE' else None
        c = D.physical_schedule(s, layout_by_round=layouts)['rounds'][row['m']]['physical_candidate']
        if method == 'FULL_H_RTE':
            structure.update(allocation=row['mapping']['physical_layout']['allocation'], deterministic_blocks=0)
    else:
        c = dict(system=s['binding']['system'], method='PR' if method == 'PR' else s['split']['original_PF_method'],
            formula=s['formula'], prefix=s['L_D'], T=float(B.a.exact_scalar(row['T'])), q=row['q_PF'],
            delta=float(B.a.exact_scalar(row['actual_delta_pf'])),
            R_outer=row['R_outer'] if method == 'PR' else None, K=s['finite_K'])
        c['id'] = B.p.candidate_id(c)
    return dict(binding=deepcopy(s['binding']), profile=deepcopy(s['profile']), physical_candidate=deepcopy(c),
        structure=structure, split=deepcopy(s['split']), law=deepcopy(s['law']))


def policy(lowering=NEW, ident='ARTIFICIAL_COMMON_NEW_ZZ_SCENARIO'):
    return dict(id=ident, source=MOCK, profile=deepcopy(configuration('PR')['spec']['profile']),
        by_method={m: dict(native_policy='ART_NATIVE', rte_rotation_lowering=lowering if m in RTE else G.NO_RTE,
                           scalar_quantity='ART_TOTAL_' + m) for m in ('PR', 'FULL_H_RTE', 'FULL_PF', 'TRUNCATED_PF')})


def fixture(method='PR', *, lowering=NEW, ident='ARTIFICIAL_COMMON_NEW_ZZ_SCENARIO', kind='SYNTHETIC_ONLY'):
    """B's whole-allocation fixture (G=[1002,216]) plus SOURCE-bound sidecars."""
    config = configuration(method, kind == 'SCIENTIFIC'); s = config['spec']; method = B.comparison_method(s)
    contract = dict(id='MOCK_STATISTICAL_CONTRACT', source=MOCK, claim_level='SYNTHETIC_ONLY')
    allocations = [dict(allocation_id=key, source=dict(fixture='ALLOCATION_'+key), shot_policy='USER_SUPPLIED',
        shots=[dict(m=m, N_X=n, N_Y=n) for m, n in enumerate(vector)]) for key, vector in (('a', (1, 5)), ('b', (8, 1)))]
    evaluations = []
    for allocation in allocations:
        packet, prices, cohorts = data(config)
        packet['evidence_kind'] = kind
        for row, shots in zip(packet['rounds'], allocation['shots']):
            row.update(N_X=shots['N_X'], N_Y=shots['N_Y'])
        packet['rmse_evidence']['statistical_contract'] = deepcopy(contract)
        if method == 'TRUNCATED_PF':
            packet['rmse_evidence'].update(truncation_status='QUALIFIED', covers_truncation_error=True,
                target_hamiltonian=deepcopy(s['binding']['hamiltonian']),
                retained_hamiltonian=deepcopy(s['split']['retained_hamiltonian']))
        evaluations.append(dict(configuration_id=config['id'], allocation=deepcopy(allocation),
            statistical_contract=deepcopy(contract), source=MOCK, n_packet=packet))
    pol = policy(lowering, ident); domain = pol['by_method'][method]
    rte = method in RTE
    extension = dict(lowering_id=domain['rte_rotation_lowering'] if rte else domain['native_policy'], source=MOCK)
    coverage = dict(id='MOCK_SIMULTANEOUS_BOX', source=MOCK, configuration_id=config['id'],
        simultaneous=True, scope='ALL_SELECTED_ROUNDS_AXES')
    bindings = []; proofs = {}
    for price in prices:
        row = config['rounds'][price['identity']['m']]; value = 1 if row['m'] == 0 else 100
        price.update(mean=dict(rz_count=value), quantity=G.QUANTITY, normalization_is_price_multiplier=False)
        facts = dict(domain, method=method, profile=deepcopy(pol['profile']), measurement=B.p.measurement(row, price['axis']),
            native_source=CODE, wrapper_source=CODE, scalar_source=CODE, scalar_application=G.SCALAR_ONCE,
            rte_lowering_source=deepcopy({NEW: NEW_CODE, OLD: OLD_CODE}[lowering]) if rte else None,
            resolved_scalar=dict(numerator=1, denominator=8), expected_scalar=dict(numerator=1, denominator=8))
        selector = str(row['m']) + price['axis']
        side = dict(identity=deepcopy(price['identity']), law_binding=deepcopy(price['law_binding']),
            axis=price['axis'], cohort=price['cohort'], price_source=deepcopy(price['source']), source=MOCK,
            physical_context=physical(config, row), profile_extension=deepcopy(extension), mean_interval=dict(
                quantity='FULL_WRAPPER_MEAN_COMPILED_RZ_INTERVAL', lower=value, upper=value,
                source=MOCK, qualification='QUALIFIED', evidence_kind='SIMULTANEOUS_MEAN_INTERVAL',
                domain=deepcopy(price['identity']['candidate']), coverage=deepcopy(coverage)),
            implementation=facts,
            implementation_evidence=dict(commit='a'*40, path='ARTIFICIAL_FIXED_METADATA.json', selector=[selector]))
        bindings.append(side)
        proofs[selector] = {k: deepcopy(side[k]) for k in ('price_source', 'identity', 'law_binding', 'axis', 'cohort',
                                                            'physical_context', 'profile_extension', 'implementation')}
    options = dict(cohorts=cohorts, statistical_contract=contract, price_extension=extension, price_bindings=bindings,
        interval_contract=coverage, root=ROOT, comparison_policy=pol)
    return dict(config=config, allocations=allocations, evaluations=evaluations, prices=prices,
                options=options, proofs=proofs)


def mutate_fact(v, name, value, index=0):
    """Change the sidecar AND its pinned assertion consistently (a well-formed lie)."""
    side = v['options']['price_bindings'][index]
    side['implementation'][name] = deepcopy(value)
    v['proofs'][side['implementation_evidence']['selector'][0]]['implementation'][name] = deepcopy(value)


def compare(v, **extra):
    options = dict(v['options'], **extra)
    with patch.object(G, '_document', return_value=v['proofs']):
        return W.compare(v['config'], v['allocations'], v['evaluations'], v['prices'], **options)


def direct_cost(v, index=0):
    o = v['options']
    with patch.object(G, '_document', return_value=v['proofs']):
        return R.cost(ROOT, v['config'], v['evaluations'][index]['n_packet'], v['prices'], cohorts=o['cohorts'],
                      price_bindings=o['price_bindings'], comparison_policy=o['comparison_policy'])


def no_arithmetic():
    """B.cost/B.select are the arithmetic entries; reaching either is a bypass."""
    return (patch.object(B, 'cost', side_effect=AssertionError('rejected price reached B.cost')),
            patch.object(B, 'select', side_effect=AssertionError('rejected price reached B.select')))


class Tests(unittest.TestCase):
    METHODS = ('PR', 'FULL_H_RTE', 'FULL_PF', 'TRUNCATED_PF')

    def assertRejected(self, call, pattern=None):
        cost, select = no_arithmetic()
        with cost, select:
            if pattern is None:
                with self.assertRaises(ValueError): call()
            else:
                with self.assertRaisesRegex(ValueError, pattern): call()

    def test_01_legacy_fullH_RZZ_and_new_PR_ZZ_cannot_share_one_new_scenario(self):
        # (a) A legacy generic-RZZ Full-H price inside the new-ZZ scenario.
        v = fixture('FULL_H_RTE')
        for i in range(len(v['prices'])):
            mutate_fact(v, 'rte_rotation_lowering', OLD, i); mutate_fact(v, 'rte_lowering_source', OLD_CODE, i)
        self.assertRejected(lambda: compare(v), 'rte_rotation_lowering')
        self.assertRejected(lambda: direct_cost(v), 'rte_rotation_lowering')
        # (b) Each family checked under its own policy; selection refuses to mix them.
        pr = compare(fixture('PR')); old = compare(fixture('FULL_H_RTE', lowering=OLD, ident='ARTIFICIAL_OLD_RZZ'))
        self.assertIsNotNone(old['point_minimum']['cost'])
        configs = [fixture('PR')['config'], fixture('FULL_H_RTE')['config']]
        self.assertRejected(lambda: R.select(configs, [pr['outer_evaluation'], old['outer_evaluation']],
                                             required_methods=RTE), 'common SOURCE-bound')
        # (c) One policy cannot declare the two lowerings comparable.
        v = fixture('PR'); v['options']['comparison_policy']['by_method']['FULL_H_RTE']['rte_rotation_lowering'] = OLD
        self.assertRejected(lambda: compare(v), 'Mixed RTE lowering')

    def test_02_same_compiler_hash_different_lowering_rejected(self):
        for method in RTE:
            v = fixture(method)
            self.assertEqual(v['options']['comparison_policy']['profile'], v['config']['spec']['profile'])
            mutate_fact(v, 'rte_rotation_lowering', OLD)                       # facts disagree with policy
            self.assertRejected(lambda: compare(v), 'rte_rotation_lowering')
            v = fixture(method); mutate_fact(v, 'rte_lowering_source', OLD_CODE)   # new ID on the old emitter
            self.assertRejected(lambda: compare(v), 'emitter SOURCE')
            v = fixture(method)                                                 # self-consistent extension only
            for side in v['options']['price_bindings']:
                side['profile_extension']['lowering_id'] = OLD
            v['options']['price_extension']['lowering_id'] = OLD
            self.assertRejected(lambda: compare(v))

    def test_03_unknown_contradictory_or_altered_SOURCE_rejected(self):
        changes = (
            lambda v: v['options']['price_bindings'][0].pop('implementation'),
            lambda v: mutate_fact(v, 'rte_rotation_lowering', 'UNKNOWN_LOWERING'),
            lambda v: mutate_fact(v, 'rte_lowering_source', None),
            lambda v: mutate_fact(v, 'native_source', None),
            lambda v: mutate_fact(v, 'wrapper_source', dict(commit='b'*40, path='MISSING.py')),
            lambda v: v['prices'][0].update(source=dict(fixture='FOREIGN_PRODUCER')),
            lambda v: v['proofs']['0X']['implementation'].update(native_policy='ALTERED'),
            lambda v: v['options']['price_bindings'][0]['implementation_evidence'].update(commit='HEAD'),
            lambda v: v['options']['comparison_policy'].pop('source'))
        for change in changes:
            v = fixture(); change(v)
            self.assertRejected(lambda: compare(v))
            self.assertRejected(lambda: direct_cost(v))

    def test_04_scalar_wrapper_axis_and_candidate_identity_rejected(self):
        wrapper = dict(policy()['profile'], wrapper='WITHOUT_MEASUREMENT')
        for name, value in (('scalar_quantity', 'TAIL_ONLY'), ('scalar_application', 'TWICE'),
                            ('resolved_scalar', dict(numerator=3, denominator=8)), ('profile', wrapper),
                            ('measurement', dict(axis='FOREIGN')), ('method', 'FULL_PF')):
            v = fixture(); mutate_fact(v, name, value)
            self.assertRejected(lambda: compare(v))
        v = fixture(); v['prices'][0]['measurement_axis'] = 'sine'
        self.assertRejected(lambda: compare(v))
        for method, key, value in (('PR', 'T', 9.), ('PR', 'id', 'FOREIGN_ID'), ('TRUNCATED_PF', 'method', 'FULL'),
                                   ('FULL_PF', 'q', 99), ('FULL_H_RTE', 'R_outer', 3)):
            v = fixture(method); side = v['options']['price_bindings'][0]
            side['physical_context']['physical_candidate'][key] = value
            v['proofs']['0X']['physical_context']['physical_candidate'][key] = value
            self.assertRejected(lambda: compare(v))
            self.assertRejected(lambda: direct_cost(v))
        v = fixture(); v['options']['price_bindings'][0]['identity'] = deepcopy(v['config']['rounds'][1]['identity'])
        self.assertRejected(lambda: compare(v))

    def test_05_full_and_truncated_PF_use_no_RTE_domain(self):
        for method in ('FULL_PF', 'TRUNCATED_PF'):
            v = fixture(method); side = v['options']['price_bindings'][0]
            self.assertEqual(side['implementation']['rte_rotation_lowering'], G.NO_RTE)
            self.assertIsNone(side['implementation']['rte_lowering_source'])
            self.assertEqual(side['profile_extension']['lowering_id'], 'ART_NATIVE')
            self.assertEqual([B.a.exact_scalar(r['G']) for r in compare(v)['allocations']], [1002, 216])
            mutate_fact(v, 'rte_lowering_source', CODE)
            self.assertRejected(lambda: compare(v), 'no RTE')
            v = fixture(method); v['options']['comparison_policy']['by_method'][method]['rte_rotation_lowering'] = NEW
            self.assertRejected(lambda: compare(v), 'RTE applicability')

    def test_06_same_domain_gives_existing_cost_and_select(self):
        evaluations = []; configs = []
        for method in self.METHODS:
            v = fixture(method); before = deepcopy(v)
            plain = dict(v['options']); plain.pop('root'); plain.pop('comparison_policy')
            legacy = W.compare(v['config'], v['allocations'], v['evaluations'], v['prices'], **plain)
            guarded = compare(v)
            self.assertIsNone(legacy['price_implementation_check'])
            self.assertEqual(guarded['price_implementation_check']['policy'], v['options']['comparison_policy'])
            self.assertEqual(len(guarded['price_implementation_check']['checked']), 4)
            for key in set(legacy) - {'price_implementation_check', 'outer_evaluation'}:
                self.assertEqual(guarded[key], legacy[key], key)
            self.assertEqual({k: x for k, x in guarded['outer_evaluation'].items() if k != 'price_implementation_check'},
                             {k: x for k, x in legacy['outer_evaluation'].items() if k != 'price_implementation_check'})
            self.assertEqual([B.a.exact_scalar(r['G']) for r in guarded['allocations']], [1002, 216])
            for i, ev in enumerate(v['evaluations']):
                expected = B.cost(v['config'], ev['n_packet'], v['prices'], cohorts=v['options']['cohorts'])
                got = direct_cost(v, i)
                self.assertEqual({k: x for k, x in got.items() if k != 'price_implementation_check'}, expected)
            self.assertEqual(v['config'], before['config']); self.assertEqual(v['prices'], before['prices'])
            evaluations.append(guarded['outer_evaluation']); configs.append(v['config'])
        self.assertEqual(R.select(configs, evaluations), B.select(configs, evaluations))
        self.assertTrue(R.select(configs, evaluations)['fair_method_comparison_complete'])

    def test_07_rejections_never_reach_B_arithmetic_and_unknowns_stay_unknown(self):
        v = fixture(kind='SCIENTIFIC'); options = dict(v['options']); options.pop('comparison_policy')
        cost, select = no_arithmetic()
        with cost, select:
            out = W.compare(v['config'], v['allocations'], v['evaluations'], v['prices'], **options)
        self.assertIsNone(out['point_minimum']['cost']); self.assertIsNone(out['outer_evaluation'])
        self.assertTrue(all(r['G'] is None and r['G_status'] == 'UNKNOWN' for r in out['allocations']))
        self.assertTrue(all(r['evaluation']['reasons'] == ['MISSING_PRICE_COMPARISON_IMPLEMENTATION_POLICY']
                            for r in out['allocations']))
        self.assertEqual(out['pending_allocation_ids'], ['a', 'b'])
        science = compare(v)                                   # the policy is the only missing gate
        self.assertEqual([B.a.exact_scalar(r['G']) for r in science['allocations']], [1002, 216])
        self.assertEqual(science['N_evidence_kind'], 'SCIENTIFIC')
        v = fixture(); v['prices'].pop()                       # missing price: UNKNOWN, never 0
        with patch.object(B, 'cost', side_effect=AssertionError('missing price reached B.cost')):
            out = compare(v)
        self.assertIsNone(out['point_minimum']['cost'])
        self.assertTrue(all(r['G'] is None for r in out['allocations']))
        self.assertEqual(len(out['price_implementation_check']['checked']), 3)
        v = fixture(); v['options']['root'] = None
        self.assertRejected(lambda: compare(v), 'root')
        pf = fixture('FULL_PF')
        unchecked = W.compare(pf['config'], pf['allocations'], pf['evaluations'], pf['prices'],
            **{k: x for k, x in pf['options'].items() if k not in ('root', 'comparison_policy')})
        science = deepcopy(unchecked['outer_evaluation']); science['N_evidence_kind'] = 'SCIENTIFIC'
        self.assertRejected(lambda: R.select([pf['config']], [science]), 'requires implementation')
        checked = compare(fixture())['outer_evaluation']
        self.assertRejected(lambda: R.select([configuration('PR'), pf['config']],
                                             [checked, unchecked['outer_evaluation']]), 'common SOURCE-bound')

    def test_08_guard_is_Cs_pinned_bytes_and_routes_agree(self):
        name = R.GUARD['commit'] + ':' + R.GUARD['path']
        code = compile(R.blob(ROOT, R.GUARD['commit'], R.GUARD['path']), name, 'exec')
        for function in ('check', 'cost', 'evidence', '_rte_source'):
            pinned = next(c for c in code.co_consts if getattr(c, 'co_name', None) == function)
            loaded = getattr(G, function).__wrapped__ if hasattr(getattr(G, function), '__wrapped__') else getattr(G, function)
            self.assertEqual((loaded.__code__.co_filename, loaded.__code__.co_code), (name, pinned.co_code))
        self.assertIs(G.C.published_api(str(ROOT)).B, B)
        self.assertIs(R.guard(ROOT), R.guard(str(ROOT)))
        with self.assertRaises(ValueError): G.C.published_api('/elsewhere')
        # W.compare, R.cost and C's own G.cost(controller=B) agree on accept and reject.
        for method in self.METHODS:
            v = fixture(method); o = v['options']
            with patch.object(G, '_document', return_value=v['proofs']):
                c_cost = G.cost(str(ROOT), B, v['config'], v['evaluations'][1]['n_packet'], v['prices'],
                    cohorts=o['cohorts'], price_bindings=o['price_bindings'], comparison_policy=o['comparison_policy'])
            b_cost = direct_cost(v, 1)
            self.assertEqual(c_cost['cost'], b_cost['cost']); self.assertEqual(c_cost['cost'], compare(v)['allocations'][1]['G'])
            self.assertEqual(b_cost['price_implementation_check'], compare(v)['price_implementation_check'])
            mutate_fact(v, 'scalar_quantity', 'TAIL_ONLY')
            messages = []
            for call in (lambda: compare(v), lambda: direct_cost(v), lambda: G.cost(str(ROOT), B, v['config'],
                    v['evaluations'][0]['n_packet'], v['prices'], cohorts=o['cohorts'], price_bindings=o['price_bindings'],
                    comparison_policy=o['comparison_policy'])):
                with patch.object(G, '_document', return_value=v['proofs']), patch.object(B, 'cost', side_effect=AssertionError):
                    with self.assertRaises(ValueError) as caught: call()
                messages.append(str(caught.exception))
            self.assertEqual(len(set(messages)), 1, messages)


if __name__ == '__main__':
    unittest.main(verbosity=2)
