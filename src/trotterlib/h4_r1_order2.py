"""One saved PR index1 cosine: reuse R0 numerics without changing old sources."""
from fractions import Fraction as F
import struct

from . import h4_r0_empirical as R

C, P, ROOT = R.C, R.P, R.ROOT
BASE = ROOT / 'artifacts/non_df_h4_central_r1_order2/2026-10-24'
R0BASE = R.BASE
KEYS = ('PR1_cosine',)
ACTION_LIMITS = dict(PR1_input=1, PR1_output=1, PR1_C_target=1)
LIMITS = dict(preflight=1, PR1_diagnostic=1, **ACTION_LIMITS)
TID = '596897f2d5ec2fb9b28a2b73c16423192d1494fd490fc58ca52cfc818baae0aa'
THASH = '0d16d201c87fdbb5f3e219dffb7dca7d48a69799290453e9ebb110a267db0e4b'
EHASH = '9cea6ce6d3659094b6b0d0a6da8b935cf32860c97a2e01a75a0761eded9f1779'
SAVED_PHASE = 0.26518225073344787


def check_event_semantics(model, join):
    """Check saved descriptors before any matvec, no RNG or angle generation."""
    case = model['cases']['PR']
    for i, e in enumerate(join['events']):
        order = e['taylor_order']
        C.require(order in (0, 2, 4), 'EVENT_ORDER')
        C.require((e['event_position'], e['outer_step_index'], e['inner_RTE_position']) == (i, i // 4, i % 4), 'EVENT_POSITION')
        ids = [*e['product_component_ids'], e['rotation_component_id']]
        C.require(e['selected_component_ids'] == ids and len(ids) == order + 1, 'SELECTED_PRODUCT_ROTATION_ORDER')
        C.require(e['phase'] == dict(real=float((-1) ** (order // 2)), imag=0.), 'ORDER_PHASE')
        C.require(F(e['rotation_angle']) == F(case['law']['angles'][(0, 2, 4).index(order)]), 'SAVED_ANGLE')
        apps = e['application_sequence']
        C.require(len(apps) == len(ids), 'APPLICATION_COUNT')
        for k, (cid, app) in enumerate(zip(ids, apps)):
            comp = case['components'][int(cid)]
            C.require(app['component_id'] == cid and app['application_index'] == k, 'APPLICATION_SEQUENCE')
            C.require(app['role'] == ('rotation' if k == order else 'product'), 'PRODUCT_THEN_ROTATION')
            C.require(app['coefficient_sign'] == comp['sign'] and app['coefficient_abs'] == comp['coefficient_abs'], 'C_SIGN_COMPONENT')


def preflight():
    a = R.api()
    C.require(a.qiskit.__version__ == '1.3.0', 'FIXED_QISKIT')
    target = C.read(BASE / 'inputs.json')['target']
    C.require(target['key'] == KEYS[0] and target['SOURCE'] == P.AG, 'FIXED_PR1_COSINE')
    C.require((target['trajectory_id'], target['trajectory_record_hash'], target['event_record_hash']) == (TID, THASH, EHASH), 'FIXED_IDENTITY')
    model = C.read(ROOT / P.MODEL)
    array_hash = R.extract('src/trotterlib/h4_ball_evidence.py', 'array_hash', dict(struct=struct, hashlib=R.hashlib, json=R.json))
    C.require(array_hash(model['state'], vector=True) == target['state_hash'] == model['state_hash'], 'RAW_STATE_HASH')
    C.require(C.digest({k: v for k, v in model.items() if k != 'binding_hash'}) == model['binding_hash'], 'C_MODEL_BINDING')
    C.require(model['n_qubits'] == 8 and model['qubit_order'] == 'qiskit_little_endian', 'STATE_LAYOUT')
    join = C.read(ROOT / target['price_join'])
    verify_join = R.extract('src/trotterlib/rte_event_audit.py', 'verify_join', dict(digest=C.digest))
    verify_join(join, expected_case='pauli_jw_PR')
    C.require((join['trajectory_id'], join['trajectory_record_hash'], join['event_record_hash']) == (TID, THASH, EHASH), 'JOIN_IDENTITY')
    C.require(join['trajectory_index'] == 1 and len(join['events']) == 32, 'FIXED_INDEX_COUNT')
    check_event_semantics(model, join)
    C.require([e['taylor_order'] for e in join['events']] == [0] * 30 + [2, 0], 'ONE_FIXED_ORDER2')
    event = join['events'][30]
    C.require(event['product_component_ids'] == ['43', '4'] and event['rotation_component_id'] == '72', 'FIXED_ORDER2_COMPONENTS')
    C.require([app['coefficient_sign'] for app in event['application_sequence']] == [1, -1, -1], 'FIXED_ORDER2_SIGNS')
    C.require(event['rotation_angle'] == 0.029672025283184623, 'FIXED_ORDER2_ANGLE')
    paths = C.read(BASE / 'inputs.json')['files']
    loaded = {}
    for role in ('source_body', 'controlled_body', 'input_qpy', 'compiled_qpy'):
        path = ROOT / target[role]
        C.require(C.sha(path) == paths[target[role]], 'ACTUAL_QPY_SHA:' + role)
        with path.open('rb') as f:
            items = a.qpy.load(f)
        C.require(len(items) == 1, 'SINGLE_QPY')
        loaded[role] = items[0]
    folder = (ROOT / target['input_qpy']).parent
    inp, metrics = C.read(folder / 'input.json'), C.read(folder / 'metrics.json')
    P.check_saved_tags(inp, metrics, P.AG)
    C.require(inp['axis'] == metrics['axis'] == 'cosine' and inp['trajectory_id'] == metrics['trajectory_id'] == TID, 'QPY_SCOPE')
    expected, phase_ledger = a.recipe(model, 'PR', join)
    C.require([a.token(x, loaded['source_body']) for x in loaded['source_body'].data] == expected, 'SOURCE_BODY_RECIPE')
    checks = R.validate_controlled(loaded['source_body'], loaded['controlled_body'], loaded['input_qpy'], inp['sidecar'])
    C.require(checks['aggregate_phase'] == target['aggregate_source_phase'] == SAVED_PHASE, 'PR1_PHASE_NO_PR0_SUBSTITUTION')
    parent = (ROOT / target['source_body']).parent
    body = C.read(parent / 'body_source_check.json')
    C.require(body['numeric_tokens_match'] and body['sidecar']['aggregate_phase'] == SAVED_PHASE, 'A_GATE_BODY_BINDING')
    C.require(C.read(parent / 'controlled_mapping.json') == inp['sidecar'], 'SAVED_CONTROLLED_MAPPING')
    updates = body['sidecar']['phase_updates']
    sequence = [(u['owner'], u['step'], u.get('order')) for u in updates]
    expected_updates = []
    for step in range(8):
        expected_updates.extend(('event', step, e['taylor_order']) for e in join['events'][step * 4:step * 4 + 4])
        expected_updates.append(('constant', step, None))
    C.require(sequence == expected_updates and updates[-1]['after'] == SAVED_PHASE, 'ORDERED_PHASE_HISTORY')
    for role, prefix in [('input_qpy', 'input'), ('compiled_qpy', 'output')]:
        circuit = loaded[role]
        signature = R.validate_program(circuit, compiled=role == 'compiled_qpy', n=8)
        fp = a.fingerprint(circuit)
        C.require(fp == target[prefix + '_fingerprint'], 'QPY_FINGERPRINT')
        C.require(dict(circuit.count_ops()) == metrics['posttranspile_histogram' if role == 'compiled_qpy' else 'pretranspile_histogram'], 'QPY_HISTOGRAM')
        checks[role] = dict(sha256=C.sha(ROOT / target[role]), fingerprint=fp,
                            instructions=len(circuit.data), global_phase=signature['global_phase'],
                            terminal_measurement='wire8 -> clbit0')
    proof = dict(event_position=30, step=7, inner=2, product_ids=['43', '4'], rotation_id='72',
                 signs=[1, -1, -1], labels=[model['cases']['PR']['components'][int(k)]['label'] for k in ['43', '4', '72']],
                 rotation_angle=event['rotation_angle'], order_phase=event['phase'],
                 application='P43 then signed P4 then signed rotation72; event_unitary = phase * rotation @ P4 @ P43',
                 negative_product='signed operand in C; original append adds pi before order phase',
                 constant_phase='after reverse retained blocks on each outer step',
                 saved_phase_updates=updates, ordered_SOURCE_phase_rules=phase_ledger)
    binding = dict(status='PASS', key=KEYS[0], trajectory_id=TID, trajectory_record_hash=THASH, event_record_hash=EHASH,
                   QPY_decodes=4, state_actions=0, state_hash=model['state_hash'], order2=proof,
                   old_PR_historical_hash='FAIL_CAUSE_UNKNOWN_UNCHANGED', **checks)
    return target, model, join, loaded, binding


def run(ns, contract, budget):
    budget.take('preflight')
    target, model, join, circuits, binding = preflight()
    C.save(ns / 'preflight.json', binding)
    a = R.api()
    raw = a.np.array([complex(x, y) for x, y in model['state']], dtype=a.np.complex128)
    saved_norm = C.read(R0BASE / 'run01/state_identity.json')
    C.require(saved_norm['state_hash'] == model['state_hash'] and not saved_norm['raw_state_changed'], 'R0_SAME_RAW_RAY')
    nu = F(saved_norm['nu_exact'])
    C.save(ns / 'state_identity.json', dict(saved_norm, norm_source=str((R0BASE / 'run01/state_identity.json').relative_to(ROOT))))
    budget.take('PR1_diagnostic')

    def action(name, function):
        budget.take(name)
        C.save(ns / (name + '_claim.json'), dict(call=name, UTC=C.utc(), reserved_before_call=True))
        out, receipt = function()
        receipt.update(call=name, UTC=C.utc(), binary64=True,
                       output_state_pairs=[[float(z.real), float(z.imag)] for z in out])
        C.save(ns / (name + '.json'), receipt)
        return receipt

    before = action('PR1_input', lambda: R.propagate(circuits['input_qpy'], raw, nu))
    after = action('PR1_output', lambda: R.propagate(circuits['compiled_qpy'], raw, nu))
    ideal = action('PR1_C_target', lambda: R.selected_C_action(model, join, raw, nu))
    guard = next(g for g in C.read(P.BASE / 'guard_inputs_quoted.json')['axes']
                 if (g['case'], g['epsilon_sig'], g['axis']) == ('pauli_jw_PR', '0.0005', 'cosine'))
    C.require(F(target['B_saved']) == F(guard['scale']['B_saved_exact']), 'SAVED_B_ONLY')
    result = R.compare_signals(ideal['signal'], before['signal'], after['signal'], target['B_saved'], guard['rho_guard_exact'])
    result.update(key='PR1', method='PR', index=1, axis='cosine', trajectory_id=TID,
                  scope='SELECTED_PR_INDEX1_ORDER2_ONLY')
    C.save(ns / 'PR1_result.json', result)
    C.save(ns / 'results.json', [result])
    old_results = C.read(R0BASE / 'run01/results.json')
    C.save(ns / 'R0_R1_comparison.json', dict(operation='copy saved R0 values and new PR1; no R0 reevaluation',
        rows=[*old_results, result], causal_order2_effect_established=False, law_wide_inference=False))
    used = C.read(ns / 'call_ledger.json')['used']
    C.save(ns / 'result_summary.json', dict(status='R1_ORDER2_EMPIRICAL_COMPLETE_MANDATORY_STOP', logical_diagnostics=1,
        successful_state_actions=sum((ns / (k + '.json')).is_file() for k in ACTION_LIMITS),
        total_state_action_attempts=sum(used[k] for k in ACTION_LIMITS), binary64_only=True,
        forbidden_calls=contract['forbidden_calls'], rho_upper=None, numerical_error_upper=None,
        PR_law_rho=None, LOCAL_BUDGET_MET=False, population_cost_tail='UNKNOWN',
        historical_hash='FAIL_CAUSE_UNKNOWN_UNCHANGED', original_native='UNKNOWN_UNCHANGED', original_N_G_null=True,
        C_L='COMBINATION_BLOCKED', old_compiler='MISSING_EVIDENCE', formal_certificate_accepted=False, mandatory_STOP=True))
