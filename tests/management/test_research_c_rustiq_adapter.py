"""Standard-library typed/mocked tests ONLY: no QuantumCircuit or science imports."""
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import math
import hashlib
import importlib.metadata  # keep module lifetime stable across synthetic sys.modules patches
import json
import sys
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.pr_dictionary_validation import rustiq_adapter as adapter
from trottertracks.pr_dictionary_validation.fixture import EvolutionContext, digest
from trottertracks.pr_dictionary_validation.prototype import MixedAtom


def atom(name, word=('Z', 'I', 'I', 'I'), *, kind='FRAME_Z', negative=False, frame='F0'):
    basis = ('frame:'+digest(frame) if kind == 'FRAME_Z' else
             'local:'+''.join(ch if ch in ('X', 'Y') else 'Z' for ch in word)
             if kind == 'PHYSICAL_PAULI' else None)
    return MixedAtom(name, kind, Fraction(-1 if negative else 1, 7), ('artificial',),
                     word, basis, frame if kind == 'FRAME_Z' else None)


class Event(SimpleNamespace):
    def to_dict(self):
        return dict(ids=self.selected_component_ids, angle=self.unsigned_rotation_angle,
                    phase=[self.phase.real, self.phase.imag])


def event(child, *products, angle=0.125, phase=1):
    applications = tuple(SimpleNamespace(component_id=a.atom_id, role=role,
        coefficient_sign=1 if a.coefficient > 0 else -1,
        is_identity=a.kind == 'IDENTITY', basis_id=a.basis_key)
        for a, role in [*((a, 'product') for a in products), (child, 'rotation')])
    return Event(application_sequence=applications,
        selected_component_ids=tuple(a.component_id for a in applications),
        unsigned_rotation_angle=angle, phase=complex(phase))


def provenance(route='R1', **updates):
    value = adapter.route_identity(route, input_identity='artificial-input',
                                   source_identity='artificial-source')
    value.update(updates)
    return value


def plan(events, atoms, route='R1'):
    return adapter.plan_sequence(events, atoms, 4, provenance=provenance(route))


class Recorder:
    """Protocol token recorder, deliberately NOT a quantum circuit simulator."""
    def __init__(self, qubits):
        self.qubits = qubits
        self.calls = []

    def compose(self, token, **kwargs):
        self.calls.append(('compose', token, kwargs))

    def p(self, *args):
        self.calls.append(('p', args))

    def cx(self, *args):
        self.calls.append(('cx', args))

    def cz(self, *args):
        self.calls.append(('cz', args))

    def rz(self, *args):
        self.calls.append(('rz', args))


class AdapterTests(unittest.TestCase):
    def tearDown(self):
        for name in ('numpy', 'qiskit', 'mpmath', 'trotterlib'):
            self.assertNotIn(name, sys.modules)

    def test_fixed_exact_phase_options(self):
        adapter.Options().validate()
        for key, value in [('preserve_order', False), ('upto_phase', True),
                           ('upto_clifford', True), ('resynth_clifford_method', 1),
                           ('preserve_order', 1), ('resynth_clifford_method', False)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                replace(adapter.Options(), **{key: value}).validate()

    def test_sparse_word_indices_and_already_controlled_angle_pair(self):
        terms = adapter.controlled_terms(('I', 'X', 'I', 'Y'), -0.375, ancilla=4)
        self.assertEqual(terms, (('XY', (1, 3), -0.375), ('XYZ', (1, 3, 4), 0.375)))
        # Nearby angle is retained at its original binary64 value, never snapped.
        angle = math.nextafter(math.pi/2, 0)
        self.assertEqual(adapter.controlled_terms(('Z',), angle, ancilla=1)[0][2].hex(), angle.hex())

    def test_invalid_binding_identity_and_nonfinite_rejected(self):
        for word, value, anc in [(('I',), 0.1, 1), (('Z',), float('nan'), 1),
                                 (('X',), 0.1, 0), (('A',), 0.1, 1),
                                 (('Z',), 1, 1)]:
            with self.subTest(word=word, value=value, anc=anc), self.assertRaises(ValueError):
                adapter.controlled_terms(word, value, ancilla=anc)

    def test_duplicate_occurrences_and_negative_angles_are_retained(self):
        a = atom('a', negative=True)
        p = plan([event(a, angle=-0.125)]*2, [a])
        self.assertEqual([o.ordinal for o in p.occurrences], [0, 1])
        self.assertEqual([o.atom_id for o in p.occurrences], ['a', 'a'])
        self.assertFalse(p.runs[0].singleton)
        self.assertEqual(adapter.region_network(p.runs[0].regions[0], ancilla=4),
                         (('Z', (0,), -0.125), ('ZZ', (0, 4), 0.125))*2)

    def test_physical_basis_is_preserved_and_only_central_z_enters_network(self):
        a = atom('xy', ('I', 'X', 'I', 'Y'), kind='PHYSICAL_PAULI')
        p = plan([event(a)], [a])
        o = p.occurrences[0]
        self.assertEqual(o.word, ('I', 'X', 'I', 'Y'))
        self.assertEqual(o.support, (1, 3))
        self.assertEqual(adapter.region_network(p.runs[0].regions[0], ancilla=4),
                         (('ZZ', (1, 3), 0.125), ('ZZZ', (1, 3, 4), -0.125)))
        self.assertEqual(p.system_qubits, (0, 1, 2, 3))

    def test_noncommuting_physical_words_keep_occurrence_order_and_basis_fence(self):
        z = atom('z', ('I', 'I', 'I', 'Z'), kind='PHYSICAL_PAULI')
        x = atom('x', ('I', 'I', 'X', 'X'), kind='PHYSICAL_PAULI')
        p = plan([event(z), event(x), event(z)], [z, x])
        self.assertEqual([o.atom_id for o in p.occurrences], ['z', 'x', 'z'])
        self.assertEqual([o.run_id for o in p.occurrences], [0, 1, 2])
        self.assertEqual(len(p.runs), 3)

    def test_products_fence_regions_without_changing_whole_run_singleton(self):
        a = atom('a')
        p = plan([event(a, a, a), event(a)], [a])
        run = p.runs[0]
        self.assertFalse(run.singleton)
        self.assertEqual([r.eligible for r in run.regions], [False, False, True])
        self.assertEqual([len(r.occurrences) for r in run.regions], [1, 1, 2])
        with self.assertRaises(ValueError):
            adapter.region_network(run.regions[0], ancilla=4)

    def test_identity_splits_run_and_keeps_phase_records_in_legacy_order(self):
        a = atom('negative', negative=True)
        identity = atom('I', ('I',)*4, kind='IDENTITY')
        events = [event(identity, identity, a, angle=0.2, phase=-1), event(a, angle=-0.125)]
        p = plan(events, [a, identity])
        self.assertEqual([o.run_id for o in p.occurrences], [None, 0, None, 1])
        self.assertEqual([r.singleton for r in p.runs], [True, True])
        self.assertEqual([r.kind for r in p.phases],
                         ['taylor', 'product_sign', 'taylor', 'identity_rotation'])
        self.assertEqual(p.phase, math.pi+math.pi-0.2)
        # Product identity has no network; rotation identity retains its phase.
        self.assertEqual(len(p.runs), 2)

    def test_all_identity_has_no_rustiq_regions(self):
        a = atom('I', ('I',)*4, kind='IDENTITY', negative=True)
        p = plan([event(a, a, a, angle=-0.125, phase=-1)], [a])
        self.assertEqual(p.runs, ())
        self.assertEqual(p.phase, 3*math.pi+0.125)

    def test_frame_boundary_and_same_frame_sharing(self):
        a, b = atom('a'), atom('b', ('I', 'Z', 'I', 'I'))
        c = atom('c', frame='F1')
        p = plan([event(a), event(b), event(c)], [a, b, c])
        self.assertEqual([o.run_id for o in p.occurrences], [0, 0, 1])
        self.assertEqual([r.singleton for r in p.runs], [False, True])
        self.assertEqual(p.occurrences[0].basis_hash, a.basis_key[6:])
        crossing = adapter.Region('invalid', (p.occurrences[0], p.occurrences[2]), True)
        with self.assertRaises(ValueError):
            adapter.region_network(crossing, ancilla=4)

    def test_stale_plan_is_rejected_before_any_build(self):
        a = atom('a')
        p = replace(plan([event(a)], [a]), phase=0.5)
        with patch.object(adapter, '_new_circuit', side_effect=AssertionError('No build')), \
                self.assertRaises(ValueError):
            adapter.emit_sequence(p, [a], SimpleNamespace(modes=4))

    def test_stale_order_sign_identity_basis_and_role_rejected(self):
        a = atom('a')
        for field, value in [('selected_component_ids', ('wrong',)), ('phase', 1j),
                             ('unsigned_rotation_angle', float('inf'))]:
            e = event(a)
            setattr(e, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan([e], [a])
        for field, value in [('coefficient_sign', -1), ('is_identity', True),
                             ('basis_id', 'stale'), ('role', 'invented')]:
            e = event(a)
            setattr(e.application_sequence[0], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                plan([e], [a])

    def test_bad_atom_records_and_duplicate_definitions_rejected(self):
        a = atom('a')
        # Corrupted external records bypass MixedAtom's own constructor checks.
        for bad in [SimpleNamespace(**dict(vars(a), word=('X', 'I', 'I', 'I'), support=(0,))),
                    SimpleNamespace(**dict(vars(a), coefficient=Fraction(0), support=a.support)),
                    replace(a, basis_key='frame:bad'),
                    atom('wrong', ('Z',), kind='PHYSICAL_PAULI')]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                plan([event(bad)], [bad])
        with self.assertRaises(ValueError):
            plan([event(a)], [a, a])

    def test_cache_context_separates_routes_sources_inputs_options(self):
        identities = [adapter.route_identity(r, input_identity='i', source_identity='s')
                      for r in adapter.ROUTES]
        identities.extend([adapter.route_identity('R1', input_identity='i2', source_identity='s'),
                           adapter.route_identity('R1', input_identity='i', source_identity='s2')])
        self.assertEqual(len({digest(x) for x in identities}), len(adapter.ROUTES)+2)
        p = provenance()
        p['options']['upto_phase'] = True
        with self.assertRaises(ValueError):
            adapter.plan_sequence([], [], 4, provenance=p)
        result = SimpleNamespace(input_identity='i', source_identity='s', fingerprint='e',
            context=EvolutionContext('1/5', 2, 2, 2), construction_policy=adapter.ADAPTER+'_R1')
        identity = adapter.compile_identity(result, route='R1')
        self.assertEqual(identity['evolution'], 'e')
        self.assertEqual(identity['options']['resynth_clifford_method'], 0)
        with self.assertRaises(ValueError):
            adapter.compile_identity(result, route='B1')
        # This is context separation, NOT execution of QPY/compile_cache_key.

    def test_public_Q13_dispatch_is_checked_with_a_synthetic_module_only(self):
        synthesis = Mock(return_value='synthetic-output-token')
        module = SimpleNamespace(synth_pauli_network_rustiq=synthesis)
        network = (('XY', (1, 3), -0.125), ('XYZ', (1, 3, 4), 0.125))
        with patch.dict(sys.modules, {'qiskit.synthesis.evolution': module}), \
                patch('importlib.metadata.version', return_value='1.3.0'):
            self.assertEqual(adapter._q13(5, network, adapter.Options()), 'synthetic-output-token')
        synthesis.assert_called_once_with(5,
            [('XY', [1, 3], -0.125), ('XYZ', [1, 3, 4], 0.125)],
            optimize_count=True, preserve_order=True, upto_clifford=False,
            upto_phase=False, resynth_clifford_method=0)
        with patch('importlib.metadata.version', return_value='different'), \
                self.assertRaises(ValueError):
            adapter._q13(5, network, adapter.Options())

    def test_default_route_delegates_to_unmodified_baseline_mock(self):
        sentinel = object()
        with patch.object(adapter.legacy, 'build_evolution', return_value=sentinel) as old, \
                patch.object(adapter, '_q13', side_effect=AssertionError('Forbidden synthesis')):
            self.assertIs(adapter.build_evolution('s', 'c', 'm', 'a', 'p',
                input_identity='i', source_identity='source'), sentinel)
            old.assert_called_once_with('s', 'c', 'm', 'a', 'p',
                input_identity='i', source_identity='source', optimized=True)

    def test_emission_protocol_mock_has_one_control_and_actual_basis_inverse(self):
        a = atom('a')
        p = plan([event(a, a, a), event(a)], [a])
        basis = Mock()
        basis.inverse.return_value = 'actual-basis-inverse-token'
        provider = SimpleNamespace(modes=4, basis=Mock(return_value=basis),
                                   reference_support=lambda a, singleton: a.support)
        recorder = Recorder(5)
        trace = []
        with patch.object(adapter, '_new_circuit', return_value=recorder), \
                patch.object(adapter, '_q13', return_value='already-controlled-token') as q13:
            self.assertIs(adapter.emit_sequence(p, [a], provider, trace=trace), recorder)
        provider.basis.assert_called_once_with(a, singleton=False)
        basis.inverse.assert_called_once_with()
        q13.assert_called_once_with(5, adapter.region_network(p.runs[0].regions[-1], ancilla=4),
                                    adapter.Options())
        self.assertEqual(recorder.calls[0], ('compose', 'actual-basis-inverse-token',
                                          dict(qubits=(0, 1, 2, 3), inplace=True)))
        self.assertIn(('compose', 'already-controlled-token', dict(inplace=True)), recorder.calls)
        self.assertEqual(sum(c[0] == 'cz' for c in recorder.calls), 2)  # products only
        self.assertEqual(len(trace), 1)

    def test_B1_same_partition_uses_only_legacy_central_emitter(self):
        a = atom('a', negative=True)
        p = plan([event(a, angle=-0.125)]*2, [a], route='B1')
        provider = SimpleNamespace(modes=4, basis=Mock(return_value=Mock()),
                                   reference_support=lambda a, singleton: a.support)
        with patch.object(adapter, '_new_circuit', return_value=Recorder(5)), \
                patch.object(adapter, '_q13', side_effect=AssertionError('Forbidden synthesis')), \
                patch.object(adapter.legacy, 'append_central') as central:
            adapter.emit_sequence(p, [a], provider)
        self.assertEqual(central.call_count, 2)
        self.assertTrue(all(call.args[1:] == ((0,), 4, -0.125) for call in central.call_args_list))

    def test_mock_evolution_keeps_D_fusion_scalar_result_and_route_metadata(self):
        a = atom('a')
        e = event(a)
        context = EvolutionContext('1/5', 2, 2, 2)
        spec = SimpleNamespace(modes=4, scalar='1/37', deterministic=('1/20',)*4,
                               validate=Mock())
        provider = SimpleNamespace(modes=4)
        trace = []
        with patch.object(adapter, '_new_circuit', side_effect=Recorder), \
                patch.object(adapter, 'emit_sequence', return_value='sequence-token') as emit, \
                patch.object(adapter.legacy, 'append_deterministic') as deterministic:
            result = adapter.build_evolution(spec, context, ((e, e), (e, e)), [a], provider,
                input_identity='i', source_identity='s', route='R1', trace=trace)
        self.assertEqual([c.args[2] for c in deterministic.call_args_list], [0.05, 0.1, 0.05])
        self.assertEqual(emit.call_count, 2)
        self.assertEqual(result.circuit.calls[-1], ('p', (-float(Fraction(1, 37))*0.2, 4)))
        self.assertEqual(result.ordered_events, ((e.to_dict(), e.to_dict()),)*2)
        self.assertEqual(result.construction_policy, adapter.ADAPTER+'_R1')
        self.assertEqual(trace[0]['scalar_phase']['kind'], 'scalar')

    def test_draft_contract_has_exact_cases_region_budget_and_no_science_authority(self):
        path = ROOT/'docs/research/research_c_rustiq_limited_contract_20261024.json'
        c = json.loads(path.read_bytes())
        self.assertFalse(c['scientific_execution_authorized'])
        self.assertEqual([case['id'] for case in c['cases']],
                         ['SS_motif_2', 'SS_motif_3', 'SS_motif_4', 'PP_motif_0'])
        self.assertEqual(sum(len(x['target_regions']) for x in c['cases']), 10)
        counts = c['call_counts']
        self.assertEqual(counts['full_evolution_build_requests'], 12)
        self.assertEqual(counts['Rustiq_synthesis_calls'], 20)
        self.assertEqual(counts['operator_materializations'], sum(counts['operator_by_kind'].values()))
        self.assertEqual(counts['reference_bundles_90_128'], sum(counts['reference_by_kind'].values()))
        self.assertEqual(counts['transpile_calls'], 24)
        self.assertEqual(counts['retry'], 0)
        self.assertTrue(all(c['resources'][name] is None for name in
                            ('cpu_cap', 'memory_cap', 'worker_cap', 'gpu_cap', 'wall_time_cap')))
        self.assertFalse(c['tolerances']['phase_fitting'])

    def test_contract_SOURCE_hashes_bind_current_reuse_and_adapter_without_imports(self):
        path = ROOT/'docs/research/research_c_rustiq_limited_contract_20261024.json'
        c = json.loads(path.read_bytes())
        for name, expected in c['source_files'].items():
            with self.subTest(path=name):
                frozen = subprocess.check_output(['git','show',
                    '5038427e034c5e2cca6674998d28715cd93ec551:'+name], cwd=ROOT)
                self.assertEqual(hashlib.sha256(frozen).hexdigest(), expected)

    def test_B2_maximal_adjacent_groups_keep_P_Q_P_and_atom_aliases(self):
        a, alias = atom('a'), atom('alias', negative=True)
        q = atom('q', ('I','Z','I','I'))
        p = plan([event(a),event(alias,angle=-0.25),event(q),event(a)], [a,alias,q], 'B2')
        groups = adapter.fusion_groups(p, p.runs[0].regions[0], case_id='artificial', macro_index=0)
        self.assertEqual([g.signed_angle for g in groups], [-0.125,0.125,0.125])
        self.assertEqual([[m['ordinal'] for m in g.ledger['members']] for g in groups], [[0,1],[2],[3]])
        self.assertEqual([m['atom_id'] for m in groups[0].ledger['members']], ['a','alias'])
        self.assertEqual([m['coefficient_sign'] for m in groups[0].ledger['members']], [1,-1])
        self.assertEqual(len(p.occurrences),4)

    def test_B2_fsum_keeps_distinct_Taylor_angles_without_representative_multiplier(self):
        a = atom('a')
        events = [event(a,angle=x) for x in (1e16,1.0,-1e16)]
        for e, n in zip(events, (2,0,0)):
            e.taylor_order = n
        p = plan(events,[a],'B2')
        g, = adapter.fusion_groups(p,p.runs[0].regions[0],case_id='artificial',macro_index=0)
        self.assertEqual(g.signed_angle,1.0)
        self.assertEqual([e.taylor_order for e in events],[2,0,0])
        self.assertEqual([m['angle_hex'] for m in g.ledger['members']], [x.hex() for x in (1e16,1.0,-1e16)])

    def test_B2_singleton_negative_zero_and_zero_sum_are_not_deleted(self):
        a = atom('a')
        for angles in [(-0.0,), (0.125,-0.125), (math.pi,math.pi)]:
            p = plan([event(a,angle=x) for x in angles],[a],'B2')
            g, = adapter.fusion_groups(p,p.runs[0].regions[0],case_id='artificial',macro_index=0)
            wanted = angles[0] if len(angles)==1 else math.fsum(angles)
            self.assertEqual(g.signed_angle.hex(),wanted.hex())
        for bad in (float('nan'),float('inf'),1):
            with self.assertRaises(ValueError):
                adapter._signed_sum([SimpleNamespace(signed_angle=bad)])
        with self.assertRaisesRegex(ValueError,'overflow'):
            adapter._signed_sum([SimpleNamespace(signed_angle=1e308)]*2)

    def test_B2_original_word_and_frame_are_in_the_group_key(self):
        a = atom('a')
        p = plan([event(a),event(a)],[a],'B2')
        for field,value in [('word',('X','I','I','I')),('frame_id','other')]:
            second = replace(p.occurrences[1],**{field:value})
            occurrences = (p.occurrences[0],second)
            region = replace(p.runs[0].regions[0],occurrences=occurrences)
            changed = replace(p,occurrences=occurrences,runs=(replace(p.runs[0],regions=(region,)),))
            data = adapter.asdict(changed); data.pop('fingerprint')
            changed = replace(changed,fingerprint=digest(data))
            self.assertEqual(len(adapter.fusion_groups(changed,region,case_id='artificial',macro_index=0)),2)

    def test_B2_product_identity_basis_macro_fences_and_phase_are_preserved(self):
        a, other = atom('a',negative=True),atom('other',frame='F1')
        identity = atom('identity',('I',)*4,kind='IDENTITY')
        p = plan([event(a,a,phase=-1),event(a),event(identity,angle=0.25),event(other),event(a)],
                 [a,identity,other],'B2')
        groups = [g for run in p.runs for r in run.regions if r.eligible
                  for g in adapter.fusion_groups(p,r,case_id='artificial',macro_index=0)]
        self.assertEqual([len(g.ledger['members']) for g in groups],[2,1,1])
        self.assertEqual(p.phase,math.pi+math.pi-0.25)
        region = p.runs[0].regions[-1]
        other_macro = adapter.fusion_groups(p,region,case_id='artificial',macro_index=1)
        self.assertNotEqual(digest(groups[0].ledger),digest(other_macro[0].ledger))
        with self.assertRaises(ValueError):
            adapter.fusion_groups(p,p.runs[0].regions[0],case_id='artificial',macro_index=0)
        crossing = adapter.Region('crossing',(p.occurrences[1],p.occurrences[-1]),True)
        with self.assertRaises(ValueError):
            adapter.fusion_groups(p,crossing,case_id='artificial',macro_index=0)

    def test_B2_emits_one_existing_controlled_primitive_keeps_original_full_basis(self):
        a = atom('a')
        p = plan([event(a),event(a,angle=-0.125)],[a],'B2')
        provider = SimpleNamespace(modes=4,basis=Mock(return_value=Mock()),
                                   reference_support=lambda a,singleton:a.support)
        trace=[]
        with patch.object(adapter,'_new_circuit',return_value=Recorder(5)), \
                patch.object(adapter,'_q13',side_effect=AssertionError('No Rustiq')), \
                patch.object(adapter.legacy,'append_central') as central:
            adapter.emit_sequence(p,[a],provider,trace=trace,case_id='artificial',macro_index=0)
        provider.basis.assert_called_once_with(a,singleton=False)
        self.assertEqual(central.call_count,1)
        self.assertEqual(central.call_args.args[1:],((0,),4,0.0))
        self.assertEqual(trace[0]['kind'],'fusion')
        self.assertEqual(trace[0]['ledger'][0]['original_plan'],p.fingerprint)

    def test_B2_route_fingerprint_cache_and_scoped_original_order(self):
        a = atom('a'); e = event(a)
        context = EvolutionContext('1/5',2,2,2)
        spec = SimpleNamespace(modes=4,scalar='1/37',deterministic=('1/20',)*4,validate=Mock())
        provider = SimpleNamespace(modes=4,basis=Mock(return_value=Mock()),
                                   reference_support=lambda a,singleton:a.support)
        with patch.object(adapter,'_new_circuit',side_effect=Recorder), \
                patch.object(adapter.legacy,'append_deterministic'), \
                patch.object(adapter.legacy,'append_central'), \
                patch.object(adapter,'_q13',side_effect=AssertionError('No Rustiq')):
            result = adapter.build_evolution(spec,context,((e,e),(e,e)),[a],provider,
                input_identity='i',source_identity='s',route='B2',case_id='artificial')
        self.assertEqual(result.ordered_events,((e.to_dict(),e.to_dict()),)*2)
        identity=adapter.compile_identity(result,route='B2')
        self.assertEqual(identity['fusion_policy'],adapter.FUSION_POLICY)
        self.assertEqual(identity['evolution'],result.fingerprint)
        self.assertNotEqual(plan([e],[a],'B1').fingerprint,plan([e],[a],'B2').fingerprint)


if __name__ == '__main__':
    unittest.main()
