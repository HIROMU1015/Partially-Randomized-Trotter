"""Future V1 numerical adapter. Never import science before a validated permit.

Use the pinned H6 signal paths, frozen full preparation and unchanged shot rule.
No new molecular preparation, RTE law, trajectory or full wrapper is needed by
the signal stage. This file has NOT been exercised on real H6 by this change.
"""
from copy import deepcopy
from fractions import Fraction
import importlib
import math
from pathlib import Path
from types import SimpleNamespace


def evaluate(config, permit, contract, writer):
    permit.require('signal')
    m = importlib.import_module('trottertracks.resource_applicability.h6_matched_port_v1')
    original = importlib.import_module('trottertracks.resource_applicability.h6_matched_contract_v1')
    codec = importlib.import_module('trottertracks.resource_applicability.h6_roundoff_preparation_v1')
    schedules = importlib.import_module('trottertracks.resource_applicability.ax2b_h6_contract')
    overlay = importlib.import_module('h6_native_b0i_bridge_v1')
    np = m.np
    permit.take('snapshot_load')
    paths = config['input_mapping']
    ham, sector, full, state, metadata, _ = m.load_h6_snapshot(Path(paths['h6_input_snapshot.npz']),
        permit.json_input('snapshot_receipt.json'), permit.json_input('df_receipt.json'))
    if (ham.n_qubits, len(ham.lambdas), sector.dimension) != (12, 19, 400):
        raise ValueError('FIXED_H6_LAYOUT')
    permit.take('frozen_preparation_decode')
    preparations, archive = codec.read_preparations(config['archive']['path'], config['archive']['sha256'])
    if archive['binding'] != config['archive']['binding']:
        raise ValueError('ARCHIVE_BINDING_CHANGED')
    whole = preparations['faithful:p19']
    if (whole.hamiltonian_hash != config['binding']['hamiltonian'] or
            whole.deterministic_fragment_indices != tuple(range(19))):
        raise ValueError('FULL_PREPARATION_IDENTITY')
    cs = [c for c in contract.registrations()['additional'] if c['method'] == 'B0']
    p = original.plan()
    p.update(cells=cs, epsilons=list(contract.EPSILONS), matrix_free=dict(p['matrix_free'], num_threads=1))
    p['caps_proposed'].update(primitive=20000, compile=0, trajectory=0, occurrence=0, control_probe=0)
    times = {(i, t) for c in cs for i, t in
             schedules.primitive_time_schedule(c, T=.8)['unique_primitive_times']}
    p['coverage']['primitive_actions'] = 3*len(times)
    if p['coverage']['primitive_actions'] > 20000:
        raise ValueError('PRIMITIVE_VALIDATION_BUDGET')
    port = m.MatchedPort(Path(config['provider_root']), dict(kind='H6_TECHNICAL', plan=p), writer, permit)
    port.ham, port.sector = ham, sector
    port.qindices, port.qstate = m.checked_basis_bridge(ham, sector, full, state, 1e-12)
    port.state = state.copy(); port.saved_state = state.copy()
    port.spectra = {}; port.native_actions = {}; port.tails = {}; port.oracle_tails = {}
    port.truncated_references = {}
    for c in cs:
        # A new declared PREFIX VIEW of existing blocks, never a regenerated
        # native preparation or a false reuse of the full preparation hash.
        prefix = c['prefix']
        port.preps[c['id']] = SimpleNamespace(deterministic_blocks=whole.deterministic_blocks[:prefix+1],
            preparation_hash=contract.digest(dict(scope='V1_FROZEN_FULL_PREFIX_VIEW', prefix=prefix,
                full_preparation_hash=whole.preparation_hash, archive_sha256=config['archive']['sha256'])),
            constant_coefficient=whole.constant_coefficient, extracted_identity_coefficient=0.,
            exact_rte_lambda_r=0., randomized_block_indices=())
    permit.take('full_reference_pairs')
    matrix = np.asarray(m.occupation_df_matrix(ham.constant, ham.one_body, ham.lambdas,
        ham.g_matrices, sector.basis_indices), dtype=complex)
    operator, _ = m.df_linear_operator(ham, sector, **p['matrix_free'])
    errors = []
    for j in range(sector.dimension):
        permit.take('reference_matvec')
        errors.append(float(np.linalg.norm(operator@np.eye(1, sector.dimension, j, dtype=complex)[0]-matrix[:, j])))
    error = max(errors)
    m.finite_scale_guard(log_B=0., intermediate_norm=float(np.linalg.norm(matrix)), absolute_discrepancy=error)
    port.reference, pair = m.reference_pair(matrix, state, .8)
    port.reference_error = error + pair['signal_difference']
    writer.write('reference.json', dict(reference=m.complex_record(port.reference), pair=pair,
        occupation_matrix_vs_saved_DF_action_error=error, metadata=metadata, certified=False))
    del matrix, operator
    permit.guard(); port.primitives()
    # Existing corrected/native and independent forward oracle, with exact
    # truncated references. The source port writes the ordinary signal rows.
    native_cell_signal, original_oracle = port.cell_signal, port.oracle_signal
    def counted_signal(raw, **kwargs):
        permit.take('new_B0_signals')
        return native_cell_signal(raw, **kwargs)
    def counted_oracle(raw):
        if raw['prefix'] not in port.truncated_references:
            permit.take('exact_truncated_reference_pairs')
        return original_oracle(raw)
    port.cell_signal, port.oracle_signal = counted_signal, counted_oracle
    port.correctness()
    saved = permit.read_output('signal_summary.json')['signals']
    overlays = []
    scalars = {}
    for record in saved:
        prefix = record['cell']['prefix']
        if prefix not in scalars:
            permit.take('tail_scalar')
            exact = sum((overlay.exact_identity(G, lam)
                for G, lam in zip(ham.g_matrices[prefix:], ham.lambdas[prefix:])), Fraction())
            C = float(exact)
            scalars[prefix] = C, overlay.upper_float(abs(Fraction(C)-exact))
        C, C_error = scalars[prefix]
        effective = deepcopy(record)
        effective['error_components'] = overlay.conservative_saved_error_components(effective)
        rounding = max(1e-12, 64*np.finfo(float).eps*(1+abs(.8*C)))
        phase_error = overlay.upper_float(abs(Fraction(-.8*C)+Fraction(.8)*Fraction(C)))
        permit.take('analytic_B0I_overlays')
        out = overlay.saved_signal_overlay(effective, C=C, C_error=C_error,
            rotation_roundoff=rounding, phase_roundoff=phase_error)
        out['cell'] = deepcopy(record['cell'])
        out['cell'].update(method='B0I', id=record['cell']['id'].replace('B0_', 'B0I_', 1))
        out['error_decomposition'] = m.differences(
            {k: complex(v['real'], v['imag']) for k, v in out['signals'].items()}, port.reference, 'B0')
        out['scalar_receipt'] = dict(C=C, scalar_error_upper=C_error, full_Fock_identity=True,
                                     phase_roundoff_upper=phase_error, certified=False)
        out['precision_rows'] = m.precision_rows(out, p)
        writer.write(out['cell']['id']+'_signal.json', out); overlays.append(out)
    diagnostics = []
    # This split uses the unextracted tail and original scalar. Adding/removing
    # an identity from both pieces gives the same exact-tail evolution.
    for prefix in (3, 4):
        view = SimpleNamespace(deterministic_blocks=whole.deterministic_blocks[:prefix+1],
                               preparation_hash='V1_OUTER_DIAGNOSTIC_'+str(prefix))
        actual = port.actions(view)
        independent = port.actions(view, oracle=True)
        tail = np.asarray(m.occupation_df_matrix(0., np.zeros_like(ham.one_body),
            ham.lambdas[prefix:], ham.g_matrices[prefix:], sector.basis_indices), dtype=complex)
        for q in contract.QS:
            permit.take('outer_S2_diagnostics'); permit.guard()
            vectors = []
            tail_step = m.expm(-1j*(.8/q)*tail)
            for actions in (actual, independent):
                v = state.copy()
                for _ in range(q):
                    for action in actions: v = action(v, .8/q/2)
                    v = tail_step@v
                    for action in reversed(actions): v = action(v, .8/q/2)
                    v *= np.exp(-1j*whole.constant_coefficient*.8/q)
                vectors.append(v)
            discrepancy = float(np.linalg.norm(vectors[0]-vectors[1]))
            m.finite_scale_guard(log_B=0., intermediate_norm=max(float(np.linalg.norm(v)) for v in vectors),
                                 absolute_discrepancy=discrepancy)
            signal = complex(np.vdot(state, vectors[0]))
            diagnostics.append(dict(prefix=prefix, q=q, T=.8, signal=m.complex_record(signal),
                outer_PF_signed_bias=m.complex_record(signal-port.reference),
                independent_path_discrepancy=discrepancy, formal_candidate=False,
                normalization_evaluated=False, shots_evaluated=False, cost_evaluated=False,
                allowance_certified=False))
    writer.write('outer_S2_diagnostics.json', dict(rows=diagnostics))
    writer.write('V1_signal_complete.json', dict(status='COMPLETE_MANDATORY_STOP',
        signals=saved+overlays, diagnostic_count=len(diagnostics), call_counts=permit.used,
        primitive_calls=port.calls.used, conservative_bias_certificate=None,
        new_native_full_wrapper=0, new_compile=0, new_sampling=0))
