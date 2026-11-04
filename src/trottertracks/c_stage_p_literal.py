"""Finite-word C->R->D certificates; no all-law or confidence promotion."""
from fractions import Fraction as F
from . import c_per_word_population_range as literal
from . import c_pr_aware_df_fixed_certificate as fixed


def events_from_saved(arm, ids, metadata, probabilities, normalization):
    """Typed K0 emission records, preserving saved angles; no law generation."""
    from trotterlib.rte import RTEEvent, RTEEventApplication
    atoms = {a.atom_id: a for a in arm.plan.atoms}
    rows = metadata['atoms']
    p = {row['atom_id']: float(F(prob)) for row, prob in zip(rows, probabilities, strict=True)}
    result = []
    for key in ids:
        a = atoms[key]
        app = RTEEventApplication(0, 'rotation', key, abs(float(a.coefficient)),
            1 if a.coefficient > 0 else -1, False, a.frame_id, a.basis_key, tuple(a.support), ())
        result.append(RTEEvent(0, key, (), (key,), (a.frame_id,), p[key],
            p[key]*normalization, 1, metadata['angle'], normalization, a.basis_key, (), (app,)))
    return tuple(result)


def compiled_raw(compiled, raw, *, parallel=True):
    value = fixed.compare(compiled, raw, parallel=parallel)
    return dict(value, left_id=literal._digest(compiled), right_id=literal._digest(raw))


def bind_population(reference, population):
    b = reference['binding']
    if (b['target_id'] != population['target_id'] or population['word_length'] != len(b['word'])
            or any(population['event_angle_bits'].get(key) != angle for key, angle in b['word'])
            or F(population['d_range_upper_exact']) != 1+F(population['D_norm_upper_exact'])):
        raise ValueError('PER_WORD_STAGE_R_TARGET_ANGLE_RANGE')
    return dict(target_id=b['target_id'], population_a=population['D_norm_upper_exact'],
                population_M=population['d_range_upper_exact'], range_clipping_applied=False,
                finite_word_only=True, all_q_compiler_totality_qualified=False,
                calibration_ready=False)


def triangle(raw, compiled, reference, cr, rd):
    if (rd['raw_id'] != literal._digest(raw) or rd['target_id'] != reference['binding']['target_id']
            or cr['left_id'] != literal._digest(compiled) or cr['right_id'] != literal._digest(raw)
            or rd['word_id'] != reference['binding']['word_id']
            or cr['dimension'] != rd['dimension'] or cr['phase_fit'] or rd['phase_fit']
            or cr['columns_evaluated'] != rd['dimension'] or rd['columns_evaluated'] != rd['dimension']):
        raise ValueError('SAME_RAW_FULL_COLUMNS_AND_PHASE')
    return dict(status='VALIDATED_PER_WORD_COMPILED_LITERAL_OPERATOR', target=literal.TARGET,
                binding=reference['binding'], kind='rigorous_enclosure',
                upper_exact=str(F(cr['upper_exact'])+F(rd['upper_exact'])),
                lower_exact=str(max(F(), F(rd['lower_exact'])-F(cr['upper_exact']))),
                compiled_raw=cr, raw_literal=rd, raw_id=literal._digest(raw),
                compiled_id=literal._digest(compiled), phase_fit=False, rounding_enclosed=True,
                all_word_uniform=False, Head_bridge_to_B2_included=False,
                provenance='two disjoint legs with same raw record;full controlled action',
                actual_CA_achieved=None, Stage_C_ready=False)


def shell_certificate(raw, compiled, label):
    """Only proven ancilla-only gates are reduced; all relative phases remain."""
    if label not in ('head', 'X', 'Y'):
        raise ValueError('REGISTERED_SHELL')
    R = fixed.record(raw, reduce_ancilla=True, strip_terminal_measurement=True)
    C = fixed.record(compiled, reduce_ancilla=True, strip_terminal_measurement=True)
    ideal = fixed.identity(1)
    if label == 'Y': ideal['gates'].append(dict(name='sdg', qubits=[0], params_hex=[]))
    ideal['gates'].append(dict(name='h', qubits=[0], params_hex=[]))
    return dict(label=label, compiled_ideal=fixed.compare(C, ideal, parallel=False),
                raw_ideal=fixed.compare(R, ideal, parallel=False),
                compiled_raw=fixed.compare(C, R, parallel=False),
                accounting='use compiled_ideal once;other pairs are diagnostics',
                scope='ancilla-only exact embedded operator;terminal measurement removed')


def finite_signal(per_word, population, shells):
    """One word/one occurrence only. Both map norms enter the raw signal bound.

    If ||C-D||<=e, ||D||<=a, shells errors eH/eQ, then composite action
    error <=eQ+e+a*eH and observable error <=(1+a)*that. No cap2 on D.
    This does not reuse the expectation-calibration formula or assume IID
    representativeness of the selected word.
    """
    if set(shells) != {'head', 'X', 'Y'}:
        raise ValueError('ALL_THREE_SHELLS')
    if any(x is None for x in shells.values()):
        return dict(status='UNRESOLVED_SHELL', X=None, Y=None, Stage_C_ready=False)
    if per_word['binding']['target_id'] != population['target_id']:
        raise ValueError('FINITE_SIGNAL_TARGET')
    from flint import ctx
    from . import c_pr_aware_df_zsxx_enclosure as z
    with z._BALL_LOCK, ctx.workprec(192):
        a = z._real(F(population['D_norm_upper_exact']))
        e = z._real(F(per_word['upper_exact']))
        values = {axis: literal.upper_record((1+a)*(z._real(F(shells[axis]))+e+a*z._real(F(shells['head']))))
                  for axis in ('X', 'Y')}
        complex_upper = (literal.upper_record(sum((z._real(F(v['upper_exact']))**2 for v in values.values()), z._real(F())).sqrt())
                         if all(v['upper_exact'] is not None for v in values.values()) else None)
    return dict(status='FINITE_WORD_ONE_OCCURRENCE_ONLY', axes=values, occurrences=1,
                complex_signal=complex_upper,
                statistical_confidence_computed=False, all_law_synthesis_qualified=False,
                actual_CA_achieved=None, Stage_C_ready=False)
