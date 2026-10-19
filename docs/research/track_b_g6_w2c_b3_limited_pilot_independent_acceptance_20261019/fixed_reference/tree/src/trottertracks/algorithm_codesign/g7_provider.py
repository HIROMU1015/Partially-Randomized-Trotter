"""Conditional exact controlled-Q provider IR. No provider search or execution."""
def controlled_event_ir(event, sigma=1):
    if sigma not in (-1, 1) or event['phase_i_power'] not in (0, 2):
        raise ValueError('signed time and even-parent phase required')
    c, helper = 'outer', 'helper'
    ir = []
    if event['phase_i_power'] == 2:
        ir.append({'op': 'Z', 'wire': c})
    for i in reversed(event['word']):
        ir.append({'op': 'CQ', 'control': c, 'label': i})
    ir += [{'op': 'H', 'wire': helper}, {'op': 'CQ', 'control': helper, 'label': event['child']},
           {'op': 'H', 'wire': helper},
           {'op': 'RZ', 'wire': helper, 'ratio': str(event['ratio']), 'sign': sigma},
           {'op': 'CX', 'control': c, 'target': helper},
           {'op': 'RZ', 'wire': helper, 'ratio': str(event['ratio']), 'sign': -sigma},
           {'op': 'CX', 'control': c, 'target': helper},
           {'op': 'H', 'wire': helper},
           {'op': 'CQ_actual_adjoint', 'control': helper, 'label': event['child']},
           {'op': 'H', 'wire': helper}]
    return ir


def provider_contract():
    return {'primary': 'exact unitary controlled Hermitian involution provider',
            'provider_controlled_Q_errors': '0 by explicit conditional query-model assumption',
            'physical_T_CX_1Q': 'unspecified nonnegative per-label symbols; never set to zero as a total-native claim',
            'provider_actual_adjoint_required': True,
            'outer_control_qubits': 1, 'additional_eigenvalue_helper_qubits': 1,
            'workspace_beyond_system': 2,
            'rotation_realization': 'H CQ H, two-Rz/two-CX controlled-Rz, actual-adjoint uncompute',
            'rotation_provider_calls': 2, 'rotation_helper_H': 4,
            'strict_rotation_error_bound': '2*Rz_epsilon under the exact-provider assumption',
            'shared_word_simplification': 'adjacent identical Q cancellation for every arm; phase retained',
            'excluded': 'no Pauli-specific reduction, provider basis search, free V_i, or whole-circuit optimizer'}


def bind_synthesized_provider_ir(event, cache, sigma=1):
    """Literal acquired Rz sequence or its actual adjoint; CQ remains a provider.

    This is a conditional native description, not an instantiated system circuit.
    The inverse retains the W scalar and reverses the full matrix-product string.
    """
    output = []
    inverse = {'H': 'H', 'T': 't', 't': 'T', 'S': 'Sdag',
               'X': 'X', 'W': 'Wdag'}
    for gate in controlled_event_ir(event, sigma):
        if gate['op'] != 'RZ':
            output.append(gate)
            continue
        row = cache[gate['ratio']]
        sequence = row['sequence']
        tokens = list(sequence) if gate['sign'] == 1 else [inverse[g] for g in reversed(sequence)]
        output.append({'op': 'SYNTHESIZED_RZ', 'wire': gate['wire'],
            'matrix_product_tokens': tokens, 'matrix_product_convention': 'same as strict sequence guard',
            'sequence_sha256': row['sequence_sha256'], 'actual_adjoint': gate['sign'] == -1,
            'global_phase_tokens_retained': True,
            'strict_operator_error_upper': row['strict_operator_error_upper']})
    return output
