"""Approved T2: Truncated PF (B0I) H_eff cells for saved H4 DF10, p=7,8,9.

Scope (user approval of SCIENCE_PLAN @ c3ae31f3 after T1 @ 5e6f9fd2): at most
42 unique (prefix, formula, actual-width) cells; per cell one existing PF step,
A's unchanged H_eff evaluator, overlap-matched truncation / PF|ret / total
shifts, and the first-use + m12-time direct/spectral checks. The p7/S2/h cell
must reproduce the 13 saved B0I raw signals (367dc98d) within 1e-9, else STOP.
No T2b, extra width/prefix, price, compile, new 13-round signal, N, MSE or G.
SOURCE/input/cross-check failure -> global STOP. Per-cell numerical, branch or
correspondence failure -> that cell UNKNOWN. No retry, window change or rescue.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]

CALLER = 'scripts/run_h4_truncated_t2.py'
T1_CALLER = ('fd984d8836ec5c82311048347f709353a4d179de', 'scripts/run_h4_truncated_t1.py')
T1_RESULT = '5e6f9fd214631f7ce92fad0e9c0d273dfb479533'
T1_DIR = 'docs/research/truncated_pf_t1_results_20261114/run01/'
PLAN_META = dict(commit='c3ae31f353d27a9f720f51fa0b089acd3d57e211',
                 path='docs/research/truncated_pf_optimization_plan_20261114/PLAN_METADATA.json')
A_FOLLOWUP_PLAN = dict(commit='d48d8732ee7ef521aab5d67a0945439364f5a2f8',
                       path='docs/research/h4_full_pf_followup_plan_20261114/PLAN.json')
A_REFERENCE_PLAN = dict(commit='9af1ac8ed8adcaef059f439ce690b22b7a320de0',
                        path='docs/research/h4_full_pf_heff_preparation_20261113/PLAN.json')
SAVED_B0I = dict(commit='367dc98d832ecb4d769f2a3da3d174bc697331a5',
                 directory='docs/research/truncated_fullh_signal_results_20261114/run01/B0I_P7_S2_D01/')
PF_KERNEL = ('dd24dd329949c255d5e37b90ff8993fa8df7159b', 'src/trottertracks/pf_power_stability.py')
PREFIXES = (7, 8, 9)
H = Fraction(.1)
LADDER = (('2nd', (H, 2*H, 16*H/7, 8*H/3)), ('4th', (4*H, 16*H/3, 32*H/5)), ('4th(m5_best)', (8*H, 16*H)))
CROSS_CHECK_CELL = (7, '2nd', H)
COMPLETE = 'T2_HEFF_CELLS_COMPLETE_NOT_RPE_QUALIFIED'


def load_module(path, commit, name):
    import subprocess
    code = subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)
    mod = types.ModuleType(name); mod.__file__ = str(ROOT/path)
    exec(compile(code, commit+':'+path, 'exec'), mod.__dict__)
    return mod


T1 = load_module(T1_CALLER[1], T1_CALLER[0], 'frozen_run_h4_truncated_t1')
E = load_module(T1.EVALUATOR['path'], T1.EVALUATOR['commit'], 'frozen_pf_effective_hamiltonian')


def enc(v):
    return dict(numerator=v.numerator, denominator=v.denominator)


def exact(v):
    return Fraction(v['numerator'], v['denominator']) if isinstance(v, dict) else Fraction(v)


def plan_cells(api, P, base_spec, target, binding, physics_ref):
    """Regenerate the fixed p7..9 schedule with the published generators; check IDs."""
    meta = T1.read(PLAN_META); expected = {(r['prefix'], r['formula'], r['requested_delta_over_h']): r
                                           for r in meta['configurations_table'] if r['prefix'] in PREFIXES}
    cells = {}
    for p in PREFIXES:
        for formula, widths in LADDER:
            for width in widths:
                spec = api.D.full_pf_spec(base_spec, formula=formula, delta_pf=enc(width), M=12, rank=10)
                spec['L_D'] = p
                spec['split'].update(L_D=p, deterministic_scope='TRUNCATED_H', original_PF_method='B0I')
                spec['split']['retained_hamiltonian'].update(construction='B0I', prefix=p)
                config = P.configure(ROOT, spec, api=api)
                row = expected[(p, formula, str(width/H))]
                if config['id'] != row['configuration_id']:
                    raise ValueError('Regenerated configuration ID differs from fixed PLAN_METADATA')
                refs = [dict(m=r['m'], T=deepcopy(r['T']), binding=deepcopy(binding),
                             source=dict(**physics_ref, selector=['rounds', r['m'], 'diagnostics', 'reference_pair']))
                        for r in config['rounds']]
                plan = P.prepare(ROOT, config, target=target, references=refs, api=api)
                for r, req, fixed in zip(config['rounds'], plan['requests'], row['rounds']):
                    c = req['physical_candidate']; actual = exact(r['actual_delta_pf'])
                    if c['id'] != fixed['physical_candidate_id'] or r['q_PF'] != fixed['q_PF']:
                        raise ValueError('Regenerated physical round differs from fixed PLAN_METADATA')
                    key = (p, formula, actual)
                    cell = cells.setdefault(key, dict(prefix=p, formula=formula, actual_delta_pf=enc(actual),
                        actual_delta_over_h=str(actual/H), signed_weights=deepcopy(r['structure']['signed_weights']),
                        uses=[], first_check=None, m12_check=None))
                    if cell['signed_weights'] != r['structure']['signed_weights']:
                        raise ValueError('Signed weights differ within one cell')
                    use = dict(configuration_id=config['id'], requested_delta_over_h=str(width/H), m=r['m'],
                               q_PF=r['q_PF'], physical_candidate=deepcopy(c))
                    cell['uses'].append(use)
                    if cell['first_check'] is None or r['m'] < cell['first_check']['m']:
                        cell['first_check'] = use
                    if r['m'] == 12:
                        cell['m12_check'] = use
    for (p, formula, actual), cell in cells.items():
        if cell['m12_check'] is None:  # check-only T_12 power of an early-only cell
            q = Fraction(4096)*H/actual
            if q.denominator != 1:
                raise ValueError('Non-integer m12 check power')
            c = dict(system='H4', method='B0I', prefix=p, formula=formula, T=float(4096*H), q=int(q),
                     delta=float(4096*H)/int(q), R_outer=None, K=None)
            c['id'] = api.B.p.candidate_id(c)
            cell['m12_check'] = dict(configuration_id=None, m=12, q_PF=int(q), physical_candidate=c,
                                     check_only_not_in_any_fixed_schedule=True)
    if len(cells) != 42 or sorted({k[0] for k in cells}) != list(PREFIXES):
        raise ValueError('Fixed finite cell count differs')
    return [cells[k] for k in sorted(cells, key=lambda k: (k[0], k[1], k[2]))]


def branch_for(cell, follow_plan):
    match = [c for c in follow_plan['step_cells'] if c['formula'] == cell['formula']
             and exact(c['actual_delta_pf']) == exact(cell['actual_delta_pf'])]
    if len(match) != 1 or match[0]['branch'].get('mode') not in ('PRINCIPAL', 'ENERGY_WINDOW'):
        raise ValueError('No unique adopted A branch rule for cell')
    i = follow_plan['step_cells'].index(match[0])
    branch = {k: deepcopy(v) for k, v in match[0]['branch'].items() if k != 'status'}
    branch['source'] = dict(A_FOLLOWUP_PLAN, selector=['step_cells', i, 'branch'])
    return branch


def match_groups(basis_a, indices_a, basis_b, groups_b):
    import numpy as np
    ov = abs(basis_a[:, indices_a].conj().T@basis_b)**2
    trace = [float(np.sum(ov[:, g['indices']])) for g in groups_b]
    top = max(trace); best = [i for i, v in enumerate(trace) if v == top]
    second = sorted(trace, reverse=True)[1] if len(trace) > 1 else None
    return best, top, second, trace


def classify(cell_result, *, lam_R, eps_trot, eps_E):
    t, pf, tot = cell_result['delta_E_trunc_ha'], cell_result['delta_E_PF_ret_ha'], cell_result['delta_E_total_ha']
    L1, L2, L3 = lam_R+abs(pf), abs(t)+abs(pf), abs(tot)
    within = dict(L1=L1 <= eps_trot, L2=L2 <= eps_trot, L3=L3 <= eps_trot)
    return dict(L1_weyl_plus_abs_pf_ha=L1, L2_abs_sum_ha=L2, L3_signed_total_ha=L3, within_eps_trot=within,
                cancellation_dependent=within['L3'] and not within['L2'],
                label='CANCELLATION_DEPENDENT' if within['L3'] and not within['L2'] else
                      'WITHIN_L2_ROBUST' if within['L2'] else 'EXCEEDS_EPS_TROT_L3',
                signs=dict(trunc=math.copysign(1, t) if t else 0, pf=math.copysign(1, pf) if pf else 0),
                q2_diagnostic_total_over_eps_E=tot/eps_E, q2_or_N_not_qualified=True)


def diagnose_cell(cell, *, ctx, kernels, full_h, cache, full_basis, full_ground, retained, c_tail, lam_R,
                  delta_trunc, reference, operator, tolerances, branch, policy, budgets, save):
    """One cell -> result dict; numerical/branch/correspondence failures -> UNKNOWN."""
    import numpy as np
    delta = float(exact(cell['actual_delta_pf']))
    out = dict(prefix=cell['prefix'], formula=cell['formula'], actual_delta_pf=cell['actual_delta_pf'],
               actual_delta_over_h=cell['actual_delta_over_h'], status='UNKNOWN', reason=None,
               branch=deepcopy(branch), c_tail_ha=c_tail, delta_E_trunc_ha=delta_trunc,
               delta_E_PF_ret_ha=None, delta_E_total_ha=None, rpe_bias_ha=None, sufficient_N=None, G=None)
    try:
        step = ctx.pf_step(cell['first_check']['physical_candidate'], cell['signed_weights'])
        unitary = np.exp(-1j*delta*(ctx.constant+c_tail))*step
        save('OPERATORS.npz', dict(PF_step_without_global_scalars=step, scalars_included_unitary=unitary))
        op = deepcopy(operator); op['identity'] = dict(op['identity'], prefix=cell['prefix'], formula=cell['formula'],
            actual_delta_pf=cell['actual_delta_pf'], signed_weights=cell['signed_weights'],
            first_check=cell['first_check']['physical_candidate'])
        op['retained_hamiltonian'] = dict(op['retained_hamiltonian'], prefix=cell['prefix'], c_tail_ha=c_tail)
        diagnostic = E.evaluate(unitary, delta, state=ctx.state, operator=op, reference=reference, branch=branch,
            tolerances=tolerances, reference_hamiltonian=full_h, reference_eigensystem=cache)
        arrays = {}
        if diagnostic['status'] == 'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE':
            sp = diagnostic['spectrum']
            vecs = np.asarray(sp['eigenvectors']['real'])+1j*np.asarray(sp['eigenvectors']['imag'])
            arrays = dict(H_eff=np.asarray(diagnostic['H_eff']['real'])+1j*np.asarray(diagnostic['H_eff']['imag']),
                energies_ha=np.asarray(sp['energies_ha']), eigenvectors=vecs, state_weights=np.asarray(sp['state_weights']),
                unitary_eigenvalues=np.asarray(sp['unitary_eigenvalues']['real'])+1j*np.asarray(sp['unitary_eigenvalues']['imag']))
        compact = deepcopy(diagnostic)
        for key in ('H_eff',):
            compact[key] = None if compact.get(key) is None else 'SAVED_IN_SPECTRUM_NPZ'
        if compact.get('spectrum'):
            compact['spectrum'] = {k: v for k, v in compact['spectrum'].items() if k not in ('eigenvectors', 'unitary_eigenvalues')}
        if compact.get('reference_comparison'):
            compact['reference_comparison'] = {k: v for k, v in compact['reference_comparison'].items()
                                               if k not in ('eigenvectors', 'squared_eigenbasis_overlaps')}
        save('EIGENSYSTEM_SUMMARY.json', compact)
        out['evaluator'] = dict(status=diagnostic['status'], reason=diagnostic.get('reason'),
                                branch_status=diagnostic.get('branch_status'), diagnostics=diagnostic['diagnostics'])
        if diagnostic['status'] != 'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE':
            raise ArithmeticError('Evaluator gate: '+str(diagnostic.get('reason')))
        sp = diagnostic['spectrum']; groups = sp['energy_groups']; energies = arrays['energies_ha']; vecs = arrays['eigenvectors']
        best_full, top_full, second_full, trace_full = match_groups(full_basis, full_ground['indices'], vecs, groups)
        r_vals, r_vecs, r_index = retained
        best_ret, top_ret, second_ret, trace_ret = match_groups(r_vecs, [r_index], vecs, groups)
        arrays.update(trace_overlap_original_ground=np.asarray(trace_full), trace_overlap_retained_ground=np.asarray(trace_ret))
        save('SPECTRUM.npz', arrays)
        g = groups[best_full[0]]; e_eff = g['minimum_energy']; e_full = float(cache['eigenvalues_ha'][full_ground['indices'][0]])
        e_ret = float(r_vals[r_index])
        out.update(matched_effective_group_index=best_full, matched_to_retained_index=best_ret,
            max_trace_overlap_original=top_full, second_trace_overlap_original=second_full,
            max_trace_overlap_retained=top_ret, second_trace_overlap_retained=second_ret,
            matched_group_multiplicity=g['multiplicity'], matched_group_is_minimum=best_full == [0],
            saved_state_weight_on_matched_group=g['weight'],
            mixture_phase_bound_rad=math.asin((1-g['weight'])/g['weight']) if g['weight'] > .5 else None,
            effective_minus_original_minimum_ha=diagnostic['spectral_shift_diagnostics']['minimum_branched_energy_minus_reference_matrix_minimum_ha'],
            delta_E_PF_ret_ha=e_eff-e_ret, delta_E_total_ha=e_eff-e_full, retained_minus_original_ha=e_ret-e_full)
        out['decomposition_identity_error_ha'] = abs(out['delta_E_total_ha']-(delta_trunc+out['delta_E_PF_ret_ha']))
        out['trunc_reuse_difference_ha'] = abs(out['retained_minus_original_ha']-delta_trunc)
        records = []; z = lambda v: dict(real=float(v.real), imag=float(v.imag))
        for label in ('first_check', 'm12_check'):
            row = cell[label]; c = row['physical_candidate']
            power, metrics = kernels.power_diagnostics(step, c['q'], ctx.state)
            direct = complex(np.vdot(ctx.state, power@ctx.state))*np.exp(-1j*c['T']*(ctx.constant+c_tail))
            spectral = complex(np.sum(arrays['state_weights']*np.exp(-1j*c['T']*energies)))
            gap = abs(direct-spectral); finite = all(math.isfinite(v) for v in (direct.real, direct.imag, spectral.real, spectral.imag, gap))
            rec = dict(check=label, m=row['m'], q_PF=c['q'], T=c['T'], physical_candidate=deepcopy(c),
                       check_only_not_in_any_fixed_schedule=row.get('check_only_not_in_any_fixed_schedule', False),
                       direct_PF_mean=z(direct), effective_spectrum_mean=z(spectral), absolute_difference=gap,
                       numerical=metrics, gate_pass=finite and metrics['gate_pass'] and gap <= policy['signal_difference_absolute'],
                       scalars_once_at_total_T_in_direct_route=True)
            records.append(rec); save('round_%s.json' % label, rec)
        out['two_time_checks'] = [dict(check=r['check'], m=r['m'], q_PF=r['q_PF'], absolute_difference=r['absolute_difference'],
                                       gate_pass=r['gate_pass']) for r in records]
        if not all(r['gate_pass'] for r in records):
            raise ArithmeticError('Two-time direct/spectral or power gate')
        if out['decomposition_identity_error_ha'] > 1e-12 or out['trunc_reuse_difference_ha'] > 1e-12:
            raise ArithmeticError('Decomposition identity / T1 reuse gate')
        ok = (len(best_full) == 1 and best_full == best_ret and top_full > .5 and top_ret > .5
              and g['multiplicity'] == 1 and best_full == [0])
        out['classification'] = classify(out, lam_R=lam_R, **budgets)
        out['status'] = 'HEFF_CELL_AVAILABLE' if ok else 'UNKNOWN'
        out['reason'] = None if ok else 'STATE_CORRESPONDENCE_AMBIGUOUS_OR_LEVEL_CROSSING'
        out['_spectrum'] = (arrays['energies_ha'], arrays['state_weights'])
    except (ArithmeticError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        out.update(status='UNKNOWN', reason=str(error))
    return out


def execute(out, source):
    import numpy as np
    from trottertracks import baseline_signal_numerics as N
    from trottertracks import rpe_optimization_pipeline as I
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh output directory required; no retry/resume')
    out.mkdir(parents=True)
    def write(directory, name, value):
        directory.mkdir(parents=True, exist_ok=True)
        if name.endswith('.npz'):
            with (directory/name).open('xb') as f:
                np.savez(f, **value)
        else:
            with (directory/name).open('x') as f:
                f.write(json.dumps(E._finite_payload(value), indent=1, ensure_ascii=False, allow_nan=False,
                                   default=lambda o: o.item() if hasattr(o, 'item') else str(o))+'\n')
    started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter(); current = None
    result = dict(status='STARTED', source=dict(commit=source, caller=CALLER), cells=[], errors=[],
                  counts=dict(context_loads=0, PF_steps=0, evaluator_calls=0, direct_means=0, spectral_means=0,
                              saved_signal_comparisons=0, new_reference_actions=0, original_H_eigh=0, retained_H_eigh=0,
                              prices=0, compiles=0, N_MSE_G=0),
                  automatic_retry=False, automatic_extension=False, mandatory_stop=True,
                  rpe_bias_ha=None, sufficient_N=None, G=None, winner=None)
    try:
        fixed = {CALLER: source, T1_CALLER[1]: T1_CALLER[0],
                 'src/trottertracks/baseline_signal_numerics.py': T1.NUMERICS[0]}
        fixed.update({p: T1.CORE for p in T1.EXTRACTOR_DEPS})
        for path, commit in fixed.items():
            if (ROOT/path).read_bytes() != T1.blob(commit, path):
                raise ValueError('Scientific SOURCE differs: '+path)
        manifest = T1.read(T1.MANIFEST); binding = manifest['shared']['binding']; target = manifest['shared']['target']
        policy = T1.read(T1.POLICY); follow = T1.read(A_FOLLOWUP_PLAN); refplan = T1.read(A_REFERENCE_PLAN)
        cache_rc = T1.read(T1.CACHE); anchor = T1.read(dict(commit=T1.CACHE['commit'], path=T1.CACHE['path']))
        if anchor['reference']['hamiltonian'] != binding['hamiltonian']:
            raise ValueError('Cached eigensystem binding differs')
        budget_eps = dict(eps_trot=T1.EPS_TROT, eps_E=T1.EPS_E)
        api = I.providers(str(ROOT)); P = I.C
        base = T1.read(dict(commit='a9abf8de75bf3432fcd714e868e5c14c8884b8f8',
                            path='docs/research/pr_rpe_first_signal_menu_20261113/CONFIGURATION_LD1.json'))
        physics_ref = dict(commit='bba150834ca30bdb186fbfd58c55a8d51177b855',
                           path='docs/research/h4_pr_physical_error_contract_20261113/PHYSICAL_INPUTS.json')
        bundle_target = T1.read(dict(commit='91349e703e0d09ef702ccd108745dfaf1acc29f7',
            path='docs/research/pr_rpe_first_signal_results_20261113/B_MEANS_LD1.json'))['target']
        cells = plan_cells(api, P, base['spec'], bundle_target, deepcopy(base['spec']['binding']), physics_ref)
        branches = [branch_for(c, follow) for c in cells]
        t1 = {p: json.loads(T1.blob(T1_RESULT, T1_DIR+'prefix_%d.json' % p)) for p in PREFIXES}
        t1_npz = np.load(io.BytesIO(T1.blob(T1_RESULT, T1_DIR+'RETAINED_EIGENSYSTEMS.npz')))
        saved_rounds = [json.loads(T1.blob(SAVED_B0I['commit'], SAVED_B0I['directory']+'round_%d.json' % m)) for m in range(13)]
        tolerances = dict(unitarity=policy['unitarity_fro'], reconstruction=policy['reconstruction_fro'],
            generator_hermiticity_ha=policy['generator_hermiticity_and_reference_eigen_residual_ha'],
            energy_cluster_ha=policy['energy_cluster_ha'], phase_guard_rad=policy['phase_guard_rad'], source=T1.POLICY)
        reference = dict(energy=target['reference_solver_energy_ha'], definition=refplan['reference']['declared_energy_definition'],
            hamiltonian=binding['hamiltonian'], source=deepcopy(refplan['reference']['declared_energy_source']),
            ground_state_certified=False, carrier=dict(T1.MANIFEST, selector=['shared', 'target', 'reference_solver_energy_ha']))
        operator = dict(kind='TRUNCATED_PF', identity=dict(binding=deepcopy(binding)), source=dict(commit=source, caller=CALLER),
            target_hamiltonian=binding['hamiltonian'], retained_hamiltonian=dict(construction='B0I',
            source=dict(commit=T1_RESULT, path=T1_DIR)), scalar_policy='INCLUDED_ONCE_AT_STEP_TIME')
        write(out, 'SCOPE.json', dict(approval='USER_T2_PRIMARY_42_CELLS_AND_SAVED_B0I_CROSS_CHECK', plan=T1.PLAN,
            plan_metadata=PLAN_META, scientific_sources=fixed, evaluator=T1.EVALUATOR, t1_result=T1_RESULT,
            branch_rules=A_FOLLOWUP_PLAN, saved_signal=SAVED_B0I, tolerances=tolerances, policy=policy,
            reference=reference, budgets=budget_eps, cells=[dict(c, branch=b) for c, b in zip(cells, branches)],
            excluded=['p5/p6 diagnostic cells', 'p0-p6', 'T2b', 'width extension', 'T3+']))
        restored = N.load_shared_h4(ROOT, binding); result['counts']['context_loads'] = 1
        saved, ctx_binding, metadata = restored; ctx = N.Context(ROOT, *restored); kernels = N.kernels(str(ROOT))
        if N.EXPM != PF_KERNEL[0]:
            raise ValueError('PF kernel SOURCE differs')
        full_h = np.asarray(saved.full, dtype=np.complex128)
        full_evals, full_basis, full_diag = T1.check_full_eigensystem(full_h, cache_rc, policy)
        groups = T1.groups_function(); full_groups = groups(full_evals, abs(full_basis.conj().T@ctx.state)**2, policy['energy_cluster_ha'])
        cache = dict(source=T1.CACHE, hamiltonian=binding['hamiltonian'], eigenvalues_ha=cache_rc['eigenvalues_ha'],
                     eigenvectors=cache_rc['eigenvectors'])
        retained = {}
        for p in PREFIXES:
            row = t1[p]
            if row['status'] != 'TRUNCATION_SHIFT_AVAILABLE' or row['matched_group_index'] != 0:
                raise ValueError('T1 prefix not available for T2: '+str(p))
            c_tail = row['tail_coefficients']['used_c_tail']
            if ctx.tail_scalar(p) != c_tail:
                raise ValueError('Tail scalar differs from T1 at p='+str(p))
            vals, vecs = t1_npz['p%d_eigenvalues' % p], t1_npz['p%d_eigenvectors' % p]
            hr = (ctx.constant+c_tail)*np.eye(36)+sum(ctx.matrices[:p+1], np.zeros_like(full_h))
            res = float(np.linalg.norm(hr@vecs-vecs*vals, 'fro')); gram = float(np.linalg.norm(vecs.conj().T@vecs-np.eye(36), 'fro'))
            if res > policy['generator_hermiticity_and_reference_eigen_residual_ha'] or gram > policy['reconstruction_fro']:
                raise ValueError('T1 retained eigensystem does not match rebuilt retained H')
            retained[p] = dict(system=(vals, vecs, 0), c_tail=c_tail, lam=row['lambda_R'], trunc=row['delta_E_trunc_B0I_ha'],
                               residual=res, gram=gram)
        write(out, 'INPUT_PROVENANCE.json', dict(binding=ctx_binding, inputs=metadata['source'], dimension=len(ctx.state),
            constant_ha=ctx.constant, original_eigensystem=dict(source=T1.CACHE, **full_diag),
            retained_eigensystems=dict(source=dict(commit=T1_RESULT, path=T1_DIR+'RETAINED_EIGENSYSTEMS.npz'),
                checks={p: dict(residual_fro=v['residual'], gram_defect_fro=v['gram'], c_tail_ha=v['c_tail']) for p, v in retained.items()})))
        order = sorted(range(len(cells)), key=lambda i: (cells[i]['prefix'], cells[i]['formula'], exact(cells[i]['actual_delta_pf']))
                       != (CROSS_CHECK_CELL[0], CROSS_CHECK_CELL[1], CROSS_CHECK_CELL[2]))
        for n, i in enumerate(order):
            cell = cells[i]; p = cell['prefix']; current = (p, cell['formula'], cell['actual_delta_over_h'])
            directory = out/('cell_p%d_%s_%s' % (p, cell['formula'].replace('(', '_').replace(')', ''), cell['actual_delta_over_h'].replace('/', 'o')))
            write(directory, 'CELL_SCOPE.json', dict(cell, branch=branches[i]))
            r = retained[p]
            summary = diagnose_cell(cell, ctx=ctx, kernels=kernels, full_h=full_h, cache=cache, full_basis=full_basis,
                full_ground=full_groups[0], retained=r['system'], c_tail=r['c_tail'], lam_R=r['lam'], delta_trunc=r['trunc'],
                reference=reference, operator=operator, tolerances=tolerances, branch=branches[i], policy=policy,
                budgets=budget_eps, save=lambda name, value, d=directory: write(d, name, value))
            spectrum = summary.pop('_spectrum', None)
            summary['path'] = directory.name
            if n == 0:  # mandatory implementation cross-check against saved B0I direct signals
                if (p, cell['formula'], exact(cell['actual_delta_pf'])) != CROSS_CHECK_CELL or spectrum is None:
                    raise ValueError('Cross-check cell unavailable; STOP')
                energies, weights = spectrum; rows = []
                for m, saved_round in enumerate(saved_rounds):
                    c = saved_round['physical_candidate']; use = [u for u in cell['uses'] if u['m'] == m
                        and u['requested_delta_over_h'] == '1']
                    if (len(use) != 1 or c['id'] != use[0]['physical_candidate']['id'] or saved_round['status'] != 'NUMERICAL_MEAN_AVAILABLE'
                            or saved_round['numerical'].get('gates_passed') is not True):
                        raise ValueError('Saved B0I round identity/status differs at m='+str(m))
                    raw = complex(saved_round['signal']['raw']['real'], saved_round['signal']['raw']['imag'])
                    spectral = complex(np.sum(weights*np.exp(-1j*c['T']*energies)))
                    rows.append(dict(m=m, T=c['T'], q=c['q'], physical_candidate_id=c['id'],
                        saved_raw=dict(real=raw.real, imag=raw.imag), spectral=dict(real=spectral.real, imag=spectral.imag),
                        absolute_difference=abs(raw-spectral)))
                gate = max(v['absolute_difference'] for v in rows) <= policy['signal_difference_absolute']
                result['counts']['saved_signal_comparisons'] = len(rows)
                check = dict(cell=summary['path'], saved_source=SAVED_B0I, rows=rows, tolerance=policy['signal_difference_absolute'],
                             maximum_absolute_difference=max(v['absolute_difference'] for v in rows), gate_pass=gate)
                write(out, 'SAVED_B0I_CROSS_CHECK.json', check); result['saved_signal_cross_check'] = {k: check[k] for k in
                    ('maximum_absolute_difference', 'tolerance', 'gate_pass')}
                if not gate:
                    write(directory, 'result.json', summary); result['cells'].append(summary)
                    raise ValueError('Saved B0I 13-round cross-check failed; STOP')
            write(directory, 'result.json', summary); result['cells'].append(summary)
            print('DONE', current, summary['status'], flush=True)
        files = list(metadata['source']['inputs'].values())+[metadata['source']['context']]
        result['input_hashes_after'] = [dict(path=f['path'], expected=f['sha256'],
                                             observed=hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()) for f in files]
        if any(v['expected'] != v['observed'] for v in result['input_hashes_after']):
            raise ValueError('Scientific input changed during run')
        result['status'] = COMPLETE
    except Exception as error:
        result.update(status='STOP_PARTIAL_NO_RETRY', failed_cell=current, errors=[dict(type=type(error).__name__, message=str(error))])
    result['counts'].update(PF_steps=len(list(out.glob('cell_*/OPERATORS.npz'))),
        evaluator_calls=len(list(out.glob('cell_*/EIGENSYSTEM_SUMMARY.json'))),
        direct_means=len(list(out.glob('cell_*/round_*.json'))), spectral_means=len(list(out.glob('cell_*/round_*.json'))))
    import numpy, scipy
    result.update(started_UTC=started, finished_UTC=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter()-tick,
                  python=sys.version.split()[0], executable=sys.executable, numpy=numpy.__version__, scipy=scipy.__version__)
    write(out, 'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True); parser.add_argument('--source', required=True)
    a = parser.parse_args(); value = execute(a.out, a.source)
    print(json.dumps(dict(status=value['status'], counts=value['counts'], errors=value['errors'],
                          cross_check=value.get('saved_signal_cross_check')), indent=1), flush=True)
    sys.exit(0 if value['status'] == COMPLETE else 2)
