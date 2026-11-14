"""Metadata-only LD1 Q3 -> saved structural means / paired path receipts.

No probability, signal, price averaging, residual subtraction, fit or cost call.
The output is an evidence index, never an expected-price provider record.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from trottertracks import price_implementation_guard as G

ROOT = Path(__file__).resolve().parents[1]
BASE = 'docs/research/pr_df_ld1_q3_price_admission_20261115'
B = 'a82b0f0ec7f0f11d9981ddb2343fcd5378a87503'
D = '39ddd59edf54a3a901e5302207b2765ddbd03fbb'
B_DIR = 'docs/research/pr_df_fullh_q3_fixed_results_20261115/'
D_DIR = 'docs/research/'
REFS = {
    'join': dict(commit=B, path=B_DIR+'C_RESULT_JOIN.json'),
    'inputs': dict(commit=B, path=B_DIR+'INPUTS.json'),
    'B_result': dict(commit=B, path=B_DIR+'run01/PR_DF_LD1/RESULT.json'),
    'alignment': dict(commit='0b4b2d941cd669aaa45231e1236bc2d3158a9d12', path='docs/research/h4_five_method_candidate_alignment_20261115/ALIGNMENT.json'),
    'prior': dict(commit='855573064b2c17cfe20eb94a01ece1a554141b0b', path='docs/research/truncated_p7_admission_pr_df_evidence_20261114/PR_DF_COVERAGE.json'),
    'ES': dict(commit='9d95d3ea544e1fa1d0e4710c1ca24d8c5776469f', path='docs/research/h4_structural_schedule_results_20261113/run01/COVERAGE.json'),
    'closure': dict(commit='a637ed1b813dc3952339cb05f36beb79168ace23', path='docs/research/h4_pr_physical_closure_20261113/run02/RESULT.json'),
    'preparation': dict(commit='1cf87201ebc6c9e0296fe647354981b6f7f8b0c1', path='docs/research/h4_df10_price_v2_independent_q_20261110/completion01/PROVENANCE.json'),
    'P1': dict(commit=D, path=D_DIR+'d_p1_efficient_czz_20261113/run02/RESULT.json'),
    'P2': dict(commit=D, path=D_DIR+'d_p2_direct_pilot_20261113/run01/RESULT.json'),
    'P1_plan': dict(commit=D, path=D_DIR+'d_p1_efficient_czz_20261113/PLAN.json'),
    'P2_plan': dict(commit=D, path=D_DIR+'d_p2_direct_pilot_20261113/PLAN.json'),
}
AXES = {'X': (0, 'cosine'), 'Y': (1, 'sine')}


def at(ref, *selector):
    return dict(ref, selector=list(selector))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def equal(a, b, reason):
    require(a == b, reason)


def law_projection(law):
    """Only configuration/canonical carriers change; law contents do not."""
    return {k: deepcopy(v) for k, v in law.items() if k not in ('configuration_id', 'candidate_id')}


def path_role(phase, acquisition_phase):
    replay = phase == 'P2' and acquisition_phase == 'replay'
    return dict(replay=replay, counts_as_nonreplay_path=not replay,
                new_science_acquisition=False, statistical_IID_guarantee=None)


def coverage_summary(rows):
    return dict(requested_axes=len(rows), structural_connected_axes=sum(
        r['structural']['status']=='CONNECTED_SAVED_STRUCTURAL_EXPECTATION' for r in rows),
        accepted_original_path_axes=sum(p['status']=='ACCEPTED_FINITE_PATH_ONLY' for r in rows for p in r['finite_paths']),
        accepted_replay_reference_axes=sum(p['status']=='ACCEPTED_REPLAY_REFERENCE_ONLY' for r in rows for p in r['finite_paths']),
        held_path_axes=sum(p['status']=='HELD' for r in rows for p in r['finite_paths']),
        qualified_compiled_expectation_axes=0)


def structural_binding(row, physical, structural, original_law, profile, binding):
    equal(structural['candidate_id'], physical['id'], 'ES physical ID')
    equal(structural['m'], row['m'], 'ES round')
    equal(structural['axis'], AXES[row['axis']][1], 'ES axis')
    equal(structural['q_PF'], physical['q'], 'ES q')
    equal(structural['R_outer'], physical['R_outer'], 'ES R')
    equal(structural['quantity'], 'FULL_WRAPPER_MEAN_STRUCTURAL_RZ_COUNT', 'ES quantity')
    equal(structural['expectation'], 'IDEAL_FINITE_K_SOURCE_LAW', 'ES law meaning')
    equal(structural['status'], 'K0_PARITY_PASS_K1_NUMERIC_ENCLOSURE', 'ES saved status')
    equal(law_projection(structural['law_binding']), law_projection(original_law), 'ES generating law/grid/kappa')
    identity = structural['identity']
    equal(identity['structure'], row['structure']['physical'], 'ES signed allocation/scalar policy')
    for key in ('system', 'hamiltonian', 'state', 'snapshot', 'rank', 'rank_policy', 'geometry', 'basis'):
        equal(identity['binding'][key], binding[key], 'ES input '+key)
    equal(identity['profile'], profile, 'ES wrapper/compiler/P0 profile')
    c = identity['candidate']
    for key in ('q', 'R_outer', 'K', 'prefix', 'formula'):
        equal(c[key], physical[key], 'ES candidate '+key)
    for key in ('T', 'delta'):
        numerator, denominator = physical[key].as_integer_ratio()
        equal(c[key], dict(numerator=numerator, denominator=denominator), 'ES exact binary64 '+key)
    equal(c['source_candidate_id'], physical['id'], 'ES original physical source')


def path_binding(row, public, original, receipt, axis, cell, candidate, source, compiler):
    """Compare saved objects only, retaining paired S/C and separate X/Y."""
    a, name = AXES[axis]
    equal(original['candidate_id'], candidate['id'], 'Receipt physical ID')
    equal(original['cell_id'], cell['id'], 'Receipt cell')
    equal(original['cohort'], cell['cohort'], 'Receipt cohort')
    equal(original['index'], public['index'], 'Receipt path index')
    require(original['conditional'] is False, 'Forced/conditional path is not normal cohort')
    equal(original['source']['execution_source'], source, 'Receipt execution SOURCE')
    for key in ('C', 'S', 'event_digest', 'compiler', 'fingerprints'):
        equal(original[key], public[key], 'Public/original '+key)
    equal(public['compiler'], compiler, 'Compiler profile')
    equal(receipt['compiler'], compiler, 'Axis receipt compiler')
    equal(receipt['fingerprint'], public['fingerprints'][a], 'Axis fingerprint')
    equal(receipt['metrics'], public['compiled'][name], 'Axis compiled receipt metrics')
    equal(receipt['metrics']['rz_count'], public['C'][a], 'Axis RZ quantity')
    require(type(public['C'][a]) is int and public['C'][a] >= 0, 'RZ price domain')
    equal(public['seeds'], cell['seeds'][public['index']], 'Frozen seed selector')
    require(bool(public['event_digest']), 'Missing event digest')
    equal(len(public['laws']), candidate['q'], 'Outer/piece law count')
    for law in public['laws']:
        equal(law['steps'], candidate['R_outer'], 'Law R')
        equal(law['orders'], [0, 2], 'Finite K2 support')
    # This is a copy of paired observations, never a subtraction or cohort mean.
    return dict(C=public['C'][a], S=public['S'][a], fingerprint=public['fingerprints'][a],
                event_digest=public['event_digest'], axis=axis, native_axis=name,
                full_wrapper_statevector_validation=original['full_wrapper_statevector_validation'])


def source_checks(root):
    """Static checks of the already-executed SOURCE routes; load no backend."""
    C = G.C
    backend = C.blob(root, '1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7', 'src/trottertracks/r_price_phase1_science.py').decode()
    carrier = C.blob(root, '155529e20a8fac39e00786c40b57830b71203c4f', 'scripts/h46_long_time_native.py').decode()
    fast = C.blob(root, 'ea0706ab05313423f336614912758a93e9e8285c', 'src/trottertracks/compiled_price_fast.py').decode()
    for text, snippets in ((backend, ('scalar=ham.constant+prep.extracted_identity_coefficient',
            "tuple(prep.deterministic_fragment_indices)==tuple(range(prep.ld))", "prep.preparation_hash==expected_prep",
            "ctx.h['compose'](c,ctx.prep,requests,ctx.scalar", 'include_measurement=True', 'SAVED_INPUT_BYTES')),
            (carrier, ("qc.p(-scalar*row['T'],n)", 'scalar=0.,controlled_tails=[tail]', "control_policy='symmetric_directional'")),
            (fast, ('ctx.Accounting, ctx.blocks, c, ctx.scalar, events', "ctx.h['compose'](c, ctx.prep, requests, ctx.scalar", 'include_measurement=True'))):
        for snippet in snippets:
            require(snippet in text, 'Executed SOURCE route changed: '+snippet)
    # Reuse C's existing SOURCE-aware new-ZZ emitter guard, without price arithmetic.
    G._rte_source(str(root), G.NEW_ZZ, 'ea0706ab05313423f336614912758a93e9e8285c', 'src/trottertracks/efficient_controlled_zz.py')
    node = next(n for n in ast.parse(carrier).body if isinstance(n, ast.FunctionDef) and n.name == 'compose')
    scalar_calls = [n for n in node.body if isinstance(n, ast.If) and "scalar * row['T']" in ast.unparse(n.test)]
    require(len(scalar_calls) == 1, 'Scalar must occur once outside step loop')
    return dict(status='STATIC_EXECUTED_SOURCE_ROUTE_CHECKED', scalar='ham.constant + prep.extracted_identity_coefficient',
                scalar_application='ONCE_AFTER_ALL_OUTER_STEPS', fragment_order='saved_snapshot_order',
                lowering=G.NEW_ZZ, emitter_guard='EXISTING_G._rte_source',
                complete_G_check='NOT_INVOKED_NO_MEAN_PRICE_RECORD_OR_COST', scientific_qualification=False,
                routes=[dict(commit='1a87e5331b52a49732b7db9d49d28f1fe6d9c9a7', path='src/trottertracks/r_price_phase1_science.py', functions=['Backend.context','Backend.price_requests','structural']),
                    dict(commit='155529e20a8fac39e00786c40b57830b71203c4f', path='scripts/h46_long_time_native.py', functions=['compose','sample_requests','trial_seed']),
                    dict(commit='ea0706ab05313423f336614912758a93e9e8285c', path='src/trottertracks/compiled_price_fast.py', functions=['structural_pair','make_backend_class']),
                    dict(commit='c31ef424175c02379eafebe8fc6c7548c1d9aabe', path='src/trottertracks/resource_applicability/ax2b_native_df_v5.py', functions=['build_native_hadamard_wrapper']),
                    dict(commit='ea0706ab05313423f336614912758a93e9e8285c', path='src/trottertracks/efficient_controlled_zz.py'),
                    dict(commit='fefd548ecd56f60d72031b3d385f6f5e6ee95d06', path='src/trottertracks/pr_rpe_signal_provider.py', functions=['evaluate']),
                    dict(commit='19b95d8b3c88063d2dbae4e5bb47455b52b00903', path='src/trottertracks/s0a_structure_context.py', functions=['context_body'])])


def build(root, archive):
    root = Path(root)
    # B saved exact risk numerators can exceed Python's default input digit limit.
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    doc = {k: G.evidence(root, ref) for k, ref in REFS.items()}
    case = next(c for c in doc['join']['candidates'] if c['label'] == 'PR_DF_LD1')
    inp = next(c for c in doc['inputs']['cases'] if c['label'] == 'PR_DF_LD1')
    config = G.evidence(root, inp['configuration_source'])
    binding = doc['B_result']['record']['deterministic_input']['canonical_binding']
    equal(case['model_status'], 'MODEL_BOUND_MEETS_ADOPTED_THRESHOLD', 'Saved Q3 status')
    equal(binding['id'], case['canonical_binding_id'], 'B canonical bridge')
    require(binding['physical_circuit_changed'] is False, 'Decoder bridge changed physical circuit')
    equal(case['original_configuration_id'], config['id'], 'Original configuration')
    checks = source_checks(root)
    closure = doc['closure']
    require(closure['preparation']['component_count'] == 324, 'Saved preparation support')
    equal(next(v['preparation_hash'] for v in doc['preparation']['trial_bindings'] if v['method']=='PR'),
          closure['preparation']['hash'], 'D executed context/ES saved preparation')
    source_facts = dict(checks, scalar_saved=closure['scalar'], preparation_hash=closure['preparation']['hash'],
                        input_binding=inp['physical_binding'], profile=config['spec']['profile'],
                        scalar_source=at(REFS['closure'], 'scalar'))
    align = {(r['m'], r['axis']): (i, r) for i, r in enumerate(doc['alignment']['rows']) if r['candidate'] == 'PR-DF LD1'}
    es = {(v['m'], v['axis']): (i, v) for i, v in enumerate(doc['ES']['values'])}
    rows = []
    for j, b in enumerate(case['rows']):
        i, r = align[b['m'], b['axis']]
        physical = r['physical_candidate']; m = b['m']; axis = b['axis']
        equal((r['configuration_id'], r['physical_id'], r['original_canonical_id']),
              (case['original_configuration_id'], b['physical_id'], b['original_candidate_id']), 'Q3/ALIGNMENT identity')
        equal(config['rounds'][m]['physical_candidate'], physical, 'Physical schedule')
        mapping = binding['mapping'][m]
        equal((mapping['original_candidate_id'], mapping['total_candidate_id']),
              (b['original_candidate_id'], b['total_candidate_id']), 'Total decoder candidate')
        law = mapping['original_law_binding']
        equal(law, config['rounds'][m]['law_binding'], 'Original law')
        out = dict(m=m, axis=axis, configuration_id=case['original_configuration_id'],
            total_configuration_id=case['total_configuration_id'], physical_candidate=deepcopy(physical),
            physical_id=physical['id'], original_canonical_id=b['original_candidate_id'], total_candidate_id=b['total_candidate_id'],
            alignment=at(REFS['alignment'], 'rows', i), Q3=at(REFS['join'], 'candidates', 0, 'rows', j),
            allocation_id=case['allocation']['allocation_id'], N_fixed_Q3=b['N'], sufficient_N=None,
            Q3_status=case['model_status'], Q3_qualification='CONDITIONAL_FIXED_ALLOCATION_MODEL_THRESHOLD_ONLY',
            original_law=at(REFS['B_result'], *b['original_law_selector']), total_law=at(REFS['B_result'], *b['total_law_selector']),
            structural=None, finite_paths=[], compiled_expectation=dict(status='UNQUALIFIED_NOT_OBTAINED', E_C=None,
            law_distance=None, rare_contribution=None, total_CI=None, shared_covariance=None), G=None)
        try:
            ei, value = es[m, AXES[axis][1]]
            structural_binding(r, physical, value, law, config['spec']['profile'], inp['physical_binding'])
            out['structural'] = dict(status='CONNECTED_SAVED_STRUCTURAL_EXPECTATION', source=at(REFS['ES'], 'values', ei),
                quantity=value['quantity'], expectation=value['expectation'], law_source=at(REFS['ES'], 'values', ei, 'law_binding'),
                mean_selector=['values', ei, 'mean_S'], enclosure_selector=['values', ei, 'numerical_enclosure'],
                kernel_source=value['kernel_source'], original_configuration_id=doc['ES']['configuration_id'], compiled_price=False)
        except (KeyError, ValueError) as error:
            out['structural'] = dict(status='HELD', reason=str(error), compiled_price=False)
        for phase in ('P1', 'P2'):
            plan = doc[phase+'_plan']
            for k, public in enumerate(doc[phase]['rows']):
                cell = next(c for c in plan['cells'] if c['id'] == public['cell_id'])
                if cell['m'] != m:
                    continue
                replay = phase == 'P2' and public['phase'] == 'replay'
                if replay:
                    cell = next(c for c in doc['P1_plan']['cells'] if c['id'] == public['cell_id'])
                pc = next(c for c in plan['menus']['H4']['candidates'] if c['m'] == m)
                entry = dict(phase=phase, acquisition_phase=public.get('phase', 'original'), index=public['index'],
                    cohort=cell['cohort'], role=cell['role'], **path_role(phase, public.get('phase', 'original')),
                    source=at(REFS[phase], 'rows', k), status='HELD', pair=None, reason=None)
                try:
                    equal(pc['candidate'], physical, 'D/current physical candidate')
                    equal(pc['structure'], r['structure']['physical'], 'D/current signed structure')
                    for key in ('hamiltonian','state','snapshot','rank_policy','geometry','basis'):
                        equal(plan['menus']['H4']['binding'][key], inp['physical_binding'][key], 'D input '+key)
                    equal(plan['lowering']['id'], G.NEW_ZZ, 'D lowering')
                    equal(doc[phase]['runtime']['qiskit'], config['spec']['profile']['qiskit_version'], 'Actual Qiskit runtime')
                    equal(plan['compiler']['basis'], config['spec']['profile']['basis_gates'], 'Compiler basis')
                    equal(plan['compiler']['optimization_level'], config['spec']['profile']['optimization_level'], 'Compiler optimization')
                    equal(plan['compiler']['seed_transpiler'], config['spec']['profile']['seed_transpiler'], 'Compiler seed')
                    if phase == 'P1':
                        equal(public['status'], 'PASS', 'P1 saved path status')
                        equal(public['parity']['failures'], [], 'P1 tail parity')
                        require(all(o['equal'] for o in public['parity']['operator']), 'P1 local operator parity')
                    directory = Path(public['private_result'] if phase == 'P1' else public['private_directory'])
                    original = archive(directory/'result.json')
                    receipt = archive(directory/(AXES[axis][1]+'.json'))
                    pair = path_binding(r, public, original['value'], receipt['value'], axis, cell, physical,
                        doc[phase]['source'], config['spec']['profile']['compiler'])
                    entry.update(status='ACCEPTED_REPLAY_REFERENCE_ONLY' if replay else 'ACCEPTED_FINITE_PATH_ONLY',
                        pair=pair, original_receipt=original['locator'], axis_receipt=receipt['locator'],
                        scientific_source=doc[phase]['source'], original_price_configuration_id=plan['configuration_id'],
                        original_price_candidate_id=pc['V2_candidate_id'], sampler=plan['sampling_law'],
                        seeds=at(REFS[phase], 'rows', k, 'seeds'), laws=at(REFS[phase], 'rows', k, 'laws'),
                        price_quantity='FULL_MEASURED_CONTROLLED_WRAPPER_COMPILED_RZ_COUNT_SINGLE_PATH',
                        expectation='FINITE_DEVELOPMENT_COHORT_ONLY', implementation_facts='SOURCE_FACTS.json')
                    if replay:
                        equal(public['replay_checks'], {key: True for key in ('event_digest','seeds','C','S','fingerprints','compiled','compiler')}, 'P1 replay gates')
                except (KeyError, ValueError, OSError) as error:
                    entry.update(status='HELD', pair=None, reason=str(error))
                out['finite_paths'].append(entry)
        rows.append(out)
    require(len(rows) == 26 and {(r['m'], r['axis']) for r in rows} == {(m, a) for m in range(13) for a in AXES}, 'LD1 26-axis scope')
    return dict(schema='PR_DF_LD1_Q3_SAVED_PRICE_EVIDENCE_INDEX_V1', sources=REFS, rows=rows, coverage_summary=coverage_summary(rows),
        source_facts=source_facts, observation_role='METADATA_ADMISSION_NO_SCIENTIFIC_REAGGREGATION',
        qualified_E_C=None, sufficient_N=None, G=None, winner=None, new_science_calls=0, mandatory_stop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    def archive(path):
        raw = path.read_bytes(); value = json.loads(raw); sha = hashlib.sha256(raw).hexdigest()
        dest = args.out/'receipts'/(sha+'.json'); dest.parent.mkdir(exist_ok=True)
        if dest.exists(): equal(dest.read_bytes(), raw, 'Receipt copy differs')
        else: dest.write_bytes(raw)
        return dict(value=value, locator=dict(original_path=str(path), sha256=sha, archived_path=str(dest.relative_to(ROOT))))
    data = build(ROOT, archive)
    facts = data.pop('source_facts')
    (args.out/'SOURCE_FACTS.json').write_text(json.dumps(facts,indent=2)+'\n')
    (args.out/'ADMISSION.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(dict(axes=len(data['rows']), structural_connected=sum(r['structural']['status']=='CONNECTED_SAVED_STRUCTURAL_EXPECTATION' for r in data['rows']), held_path_axes=sum(p['status']=='HELD' for r in data['rows'] for p in r['finite_paths']), new_science_calls=0)))


if __name__ == '__main__':
    main()
