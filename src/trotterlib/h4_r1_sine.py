"""Three saved sine programs, six state actions, no new ideal-target action."""
from fractions import Fraction as F
import struct
from . import h4_r0_empirical as R
from . import h4_r1_order2 as O

C, P, ROOT = R.C, R.P, R.ROOT
BASE = ROOT / 'artifacts/non_df_h4_central_r1_sine/2026-10-24'
KEYS = ('PF0_sine', 'PR0_sine', 'PR1_sine')
CASES = ('PF', 'PR0', 'PR1')
ACTION_LIMITS = {key + '_' + role: 1 for key in CASES for role in ('input', 'output')}
LIMITS = dict(preflight=1, **{key + '_diagnostic': 1 for key in CASES}, **ACTION_LIMITS)
OLD0 = R.BASE
OLD1 = O.BASE


def validate_sine(source, controlled, cosine, sine, sidecar):
    checks = R.validate_controlled(source, controlled, cosine, sidecar)
    a = R.api()
    cos, sig = a.signature(cosine), a.signature(sine)
    n = source.num_qubits
    sdg = dict(name='sdg', params=[], qubits=[n], clbits=[])
    C.require(sig['global_phase'] == cos['global_phase'] and
              sig['instructions'] == [*cos['instructions'][:-2], sdg, *cos['instructions'][-2:]], 'SINE_SDG_BEFORE_FINAL_H')
    checks.update(axis='sine', convention='+Im using Sdg before final H',
                  new_wrapper_build=0, body_replay=0, new_compile=0)
    return checks


def saved_target(key, cresult, nu, receipts):
    if key == 'PF':
        # Reuse the exact saved-interval conversion; select imag, never recompute C.
        imag = cresult['implemented_law']['PF']['corrected_encoded_mean']['imag']
        proxy = dict(implemented_law=dict(PF=dict(corrected_encoded_mean=dict(real=imag))), state_norm=cresult['state_norm'])
        value, record = R.saved_PF_target(proxy, nu)
        record.update(source='saved C PF corrected_encoded_mean.imag / exact raw ray norm', axis='sine', new_C_target_state_actions=0)
        return value, record
    receipt = receipts[key]
    C.require(not receipt['PR_law_mean_used'] and not receipt['normalization_B_applied_to_target'], 'INDIVIDUAL_C_CACHE_ONLY')
    C.require(receipt['complex_signal']['real'] == receipt['signal'] and receipt['numerical_error_upper'] is None, 'CACHE_POINT_POLICY')
    value = F(receipt['complex_signal']['imag'])
    return value, dict(source='saved selected ' + key + ' complex C-target.imag', axis='sine',
        selected_trajectory_id=receipt['selected_trajectory_id'], representative_exact=str(value),
        representative_display=float(value), numerical_error_upper=None, new_C_target_state_actions=0,
        PR_law_mean_used=False, compiled_signal_error_enclosed=False)


def check_sine_metadata(key, target, inputmeta, metrics):
    C.require(inputmeta['axis'] == metrics['axis'] == 'sine' and inputmeta['trajectory_id'] == target['trajectory_id'], 'SINE_METADATA')
    if target['SOURCE'] == P.AA:
        case = 'PF' if key == 'PF' else 'PR0000'
        C.require(key in ('PF', 'PR0') and inputmeta['case'] == metrics['case'] == case, 'A_AXES_CASE')
        descriptor = metrics['target']
        C.require(all(descriptor[k] == target[k] for k in ('trajectory_id', 'trajectory_record_hash', 'event_record_hash')), 'A_AXES_NESTED_TRAJECTORY')
    else:
        C.require(target['SOURCE'] == P.AG and key == 'PR1' and metrics['trajectory_id'] == target['trajectory_id'], 'A_GATE_FLAT_TRAJECTORY')


def preflight():
    a = R.api()
    C.require(a.qiskit.__version__ == '1.3.0', 'FIXED_QISKIT')
    inpmanifest = C.read(BASE / 'inputs.json')
    targets, cosines, files = inpmanifest['targets'], inpmanifest['cosine_targets'], inpmanifest['files']
    C.require(tuple(t['key'] for t in targets) == KEYS, 'THREE_FIXED_SINE_TARGETS')
    model = C.read(ROOT / P.MODEL)
    ah = R.extract('src/trotterlib/h4_ball_evidence.py', 'array_hash', dict(struct=struct, hashlib=R.hashlib, json=R.json))
    C.require(ah(model['state'], vector=True) == model['state_hash'], 'RAW_STATE_BITS')
    C.require(C.digest({k: v for k, v in model.items() if k != 'binding_hash'}) == model['binding_hash'], 'C_MODEL_BINDING')
    C.require(model['n_qubits'] == 8 and model['qubit_order'] == 'qiskit_little_endian', 'STATE_LAYOUT')
    verify_join = R.extract('src/trotterlib/rte_event_audit.py', 'verify_join', dict(digest=C.digest))
    records, circuits = [], {}
    for key, target, ct in zip(CASES, targets, cosines):
        expected_method = 'PF' if key == 'PF' else 'PR'
        expected_index = 1 if key == 'PR1' else 0
        identity = P.TARGETS[2 if key == 'PR1' else 0 if key == 'PF' else 1]
        C.require((target['method'], target['index'], target['trajectory_id']) == identity, 'FIXED_TRAJECTORY')
        C.require(target['axis'] == 'sine' and ct['axis'] == 'cosine' and target['trajectory_id'] == ct['trajectory_id'], 'AXIS_TRAJECTORY_JOIN')
        C.require((target['method'], target['index']) == (expected_method, expected_index), 'FIXED_CASE')
        join = C.read(ROOT / target['price_join'])
        verify_join(join, expected_case='pauli_jw_' + expected_method)
        C.require(all(join[k] == target[k] == ct[k] for k in ('trajectory_id', 'trajectory_record_hash', 'event_record_hash')), 'EVENT_HASH_JOIN')
        if key != 'PF': O.check_event_semantics(model, join)
        if key == 'PR1':
            C.require([e['taylor_order'] for e in join['events']] == [0] * 30 + [2, 0] and target['aggregate_source_phase'] == O.SAVED_PHASE, 'FIXED_PR1_PHASE_ORDER2')
        loaded = {}
        roles = dict(source_body=target['source_body'], controlled_body=target['controlled_body'],
                     cosine=ct['input_qpy'], input=target['input_qpy'], output=target['compiled_qpy'])
        for role, path in roles.items():
            C.require(C.sha(ROOT / path) == files[path], 'ACTUAL_QPY_SHA:' + path)
            with (ROOT / path).open('rb') as f: values = a.qpy.load(f)
            C.require(len(values) == 1, 'ONE_SAVED_QPY')
            loaded[role] = values[0]
        folder = (ROOT / target['input_qpy']).parent
        cosfolder = (ROOT / ct['input_qpy']).parent
        inputmeta, metrics = C.read(folder / 'input.json'), C.read(folder / 'metrics.json')
        cosmeta = C.read(cosfolder / 'input.json')
        P.check_saved_tags(inputmeta, metrics, P.AG if key == 'PR1' else P.AA)
        P.check_saved_tags(cosmeta, C.read(cosfolder / 'metrics.json'), P.AG if key == 'PR1' else P.D2)
        check_sine_metadata(key, target, inputmeta, metrics)
        expected, _ = a.recipe(model, expected_method, join)
        C.require([a.token(x, loaded['source_body']) for x in loaded['source_body'].data] == expected, 'SOURCE_BODY_RECIPE')
        checks = validate_sine(loaded['source_body'], loaded['controlled_body'], loaded['cosine'], loaded['input'], cosmeta['sidecar'])
        C.require(checks['aggregate_phase'] == target['aggregate_source_phase'], 'EXACT_SAVED_AGGREGATE_PHASE')
        if key != 'PR1':
            binding = C.read(folder / 'source_binding.json')
            C.require(binding['body_signature_sha256'] == C.digest(a.signature(loaded['controlled_body'])) and binding['direct_saved_body'], 'A_AXES_SAVED_BODY_BINDING')
        for role, prefix in [('input', 'input'), ('output', 'output')]:
            sig = R.validate_program(loaded[role], compiled=role == 'output', n=8)
            fp = a.fingerprint(loaded[role])
            C.require(fp == target[prefix + '_fingerprint'], 'ACTUAL_FINGERPRINT')
            C.require(dict(loaded[role].count_ops()) == metrics['posttranspile_histogram' if role == 'output' else 'pretranspile_histogram'], 'ACTUAL_HISTOGRAM')
            checks[role] = dict(sha256=C.sha(ROOT / roles[role]), fingerprint=fp, global_phase=sig['global_phase'], instructions=len(loaded[role].data))
        C.require(a.fingerprint(loaded['cosine']) == ct['input_fingerprint'], 'SAVED_COSINE_STRUCTURE_ONLY')
        records.append(dict(key=key, trajectory_id=target['trajectory_id'], state_hash=model['state_hash'], status='PASS_STRUCTURAL_SOURCE_BINDING', **checks))
        circuits[key] = dict(input=loaded['input'], output=loaded['output'])
    state0, state1 = C.read(OLD0 / 'run01/state_identity.json'), C.read(OLD1 / 'run01/state_identity.json')
    for state in (state0, state1):
        C.require(state['state_hash'] == model['state_hash'] and state['nu_exact'] == state0['nu_exact'] and not state['raw_state_changed'], 'SAME_CACHED_RAW_RAY_NORM')
    seals = [C.read(old / 'seal.json') for old in (OLD0, OLD1)]
    C.require(seals[0]['python_sha256'] == seals[1]['python_sha256'] and seals[0]['bound_manifests']['runtime.json'] == seals[1]['bound_manifests']['runtime.json'], 'CACHE_RUNTIME_MATCH')
    for seal in seals:
        C.require(seal['code_files']['src/trotterlib/h4_r0_empirical.py'] == C.sha(ROOT / 'src/trotterlib/h4_r0_empirical.py'), 'C_CACHE_NUMERIC_SOURCE_UNCHANGED')
    for key, old in [('PR0', OLD0), ('PR1', OLD1)]:
        cache = C.read(old / ('run01/' + key + '_C_target.json'))
        t = targets[1 if key == 'PR0' else 2]
        C.require(cache['selected_trajectory_id'] == t['trajectory_id'] and not cache['PR_law_mean_used'], 'CACHE_TRAJECTORY')
    return targets, model, circuits, dict(status='PASS', QPY_decodes=15, state_actions=0, records=records,
        cache_runtime_match=True, state_hash=model['state_hash'], new_C_target_state_actions=0,
        historical_hash='FAIL_CAUSE_UNKNOWN_UNCHANGED')


def run(ns, contract, budget):
    budget.take('preflight')
    targets, model, circuits, binding = preflight()
    C.save(ns / 'preflight.json', binding)
    state = C.read(OLD0 / 'run01/state_identity.json')
    nu = F(state['nu_exact'])
    C.save(ns / 'state_identity.json', dict(state, norm_source=str((OLD0 / 'run01/state_identity.json').relative_to(ROOT))))
    raw = R.api().np.array([complex(x, y) for x, y in model['state']], dtype=R.api().np.complex128)
    receipts = {k: C.read(old / ('run01/' + k + '_C_target.json')) for k, old in [('PR0', OLD0), ('PR1', OLD1)]}
    cresult = C.read(ROOT / P.CRESULT)
    guards = C.read(P.BASE / 'guard_inputs_quoted.json')['axes']
    results = []
    for key, t in zip(CASES, targets):
        budget.take(key + '_diagnostic')
        mc, record = saved_target(key, cresult, nu, receipts)
        C.save(ns / (key + '_C_target_reused.json'), record)
        signals = []
        for role in ('input', 'output'):
            call = key + '_' + role
            budget.take(call)
            C.save(ns / (call + '_claim.json'), dict(call=call, UTC=C.utc(), reserved_before_call=True))
            out, receipt = R.propagate(circuits[key][role], raw, nu)
            receipt.update(call=call, UTC=C.utc(), binary64=True, output_state_pairs=[[float(z.real), float(z.imag)] for z in out])
            C.save(ns / (call + '.json'), receipt)
            signals.append(receipt['signal'])
        gr = next(g for g in guards if (g['case'], g['epsilon_sig'], g['axis']) == ('pauli_jw_' + t['method'], '0.0005', 'sine'))
        C.require(F(t['B_saved']) == F(gr['scale']['B_saved_exact']), 'B_SAVED_ONCE')
        result = R.compare_signals(mc, *signals, t['B_saved'], gr['rho_guard_exact'])
        result.update(key=key, method=t['method'], index=t['index'], axis='sine', trajectory_id=t['trajectory_id'], scope='FIXED_PF' if key == 'PF' else 'SELECTED_' + key + '_ONLY')
        C.save(ns / (key + '_result.json'), result)
        results.append(result)
    C.save(ns / 'results.json', results)
    cosine = [*C.read(OLD0 / 'run01/results.json'), *C.read(OLD1 / 'run01/results.json')]
    C.save(ns / 'axes_comparison.json', dict(operation='copy old cosine rows and new sine; no cosine reevaluation', cosine=cosine, sine=results, law_wide_inference=False))
    used = C.read(ns / 'call_ledger.json')['used']
    C.save(ns / 'result_summary.json', dict(status='R1_SINE_EMPIRICAL_COMPLETE_MANDATORY_STOP', logical_diagnostics=3,
        successful_state_actions=sum((ns / (k + '.json')).is_file() for k in ACTION_LIMITS), total_state_action_attempts=sum(used[k] for k in ACTION_LIMITS),
        binary64_only=True, new_C_target_state_actions=0, cosine_reevaluation=0, forbidden_calls=contract['forbidden_calls'],
        historical_hash='FAIL_CAUSE_UNKNOWN_UNCHANGED', original_native='UNKNOWN_UNCHANGED', original_N_G_null=True,
        C_L='COMBINATION_BLOCKED', old_compiler='MISSING_EVIDENCE', population_cost_tail='UNKNOWN',
        rho_upper=None, numerical_error_upper=None, PR_law_rho=None, LOCAL_BUDGET_MET=False, formal_certificate_accepted=False, mandatory_STOP=True))
