"""Artificial toy checks of the Truncated PF error decomposition. No H4 input.

Reuses UNCHANGED frozen code: A's pf_effective_hamiltonian.evaluate (blob
878000d8d) and the PF constructor expm_step (blob dd24dd329, via C kernels).
The decomposition helper below is a test oracle for the proposed connection,
not an adopted implementation. Toy matrices are random Hermitian 6x6.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from scipy.linalg import eigh, expm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks import baseline_signal_numerics as N  # noqa: E402

EVALUATOR = ('878000d8d5d956b1d2f5513986ae40d0612e0bdd', 'src/trottertracks/pf_effective_hamiltonian.py')
SRC = dict(scope='ARTIFICIAL_TOY_ONLY', path='docs/research/truncated_pf_optimization_plan_20261114/TOY_CHECKS.py')
TOL = dict(unitarity=1e-9, reconstruction=1e-9, generator_hermiticity_ha=1e-9, phase_guard_rad=1e-8,
           energy_cluster_ha=1e-8, source=SRC)
Y4 = (lambda a: (a, 1-2*a, a))(1/(2-2**(1/3)))  # standard Yoshida4 signed weights


def evaluator():
    code = subprocess.check_output(['git', 'show', ':'.join(EVALUATOR)], cwd=ROOT)
    mod = types.ModuleType('frozen_pf_effective_hamiltonian')
    exec(compile(code, ':'.join(EVALUATOR), 'exec'), mod.__dict__)
    return mod


E = evaluator(); K = N.kernels(str(ROOT))


def toy(seed, dim=6, scales=(1., .6, .2, .05, .01), constant=-1.):
    rng = np.random.default_rng(seed); mats = []
    for s in scales:
        a = rng.normal(size=(dim, dim)) + 1j*rng.normal(size=(dim, dim)); h = (a+a.conj().T)/2
        mats.append(s*h/np.linalg.norm(h, 2))
    return mats, constant


def step(mats, constant, p, delta, weights=(1.,), b0i=True):
    tail = sum(mats[p+1:], np.zeros_like(mats[0])); c_tail = float(np.trace(tail).real/len(tail)) if b0i else 0.
    u = np.eye(len(mats[0]), dtype=complex)
    for w in weights:  # signed pieces at time delta*w, same order as Context.pf_step
        u = K.expm_step(mats[:p+1], (1.,), delta*w, {}) @ u
    return np.exp(-1j*delta*(constant+c_tail))*u, c_tail


def full_h(mats, constant):
    return constant*np.eye(len(mats[0])) + sum(mats)


def retained_h(mats, constant, p, c_tail):
    return (constant+c_tail)*np.eye(len(mats[0])) + sum(mats[:p+1])


def evaluate(u, delta, psi, full, kind='TRUNCATED_PF', branch=None):
    w, _ = eigh(full)
    op = dict(kind=kind, identity=dict(toy=True), source=SRC, target_hamiltonian='TOY_FULL',
              retained_hamiltonian='TOY_FULL' if kind == 'FULL_PF' else dict(method='B0I', source=SRC),
              scalar_policy='INCLUDED_ONCE_AT_STEP_TIME')
    ref = dict(energy=float(w[0]), definition='TOY_FULL_MINIMUM', hamiltonian='TOY_FULL', source=SRC)
    return E.evaluate(u, delta, state=psi, operator=op, reference=ref, tolerances=TOL,
                      branch=branch or dict(mode='PRINCIPAL', source=SRC), reference_hamiltonian=full)


def matched(evecs_a, ga, evecs_b, groups_b):
    """Index of b-group with largest trace overlap with a's group ga."""
    ov = abs(evecs_a[:, ga].conj().T @ evecs_b)**2
    scores = [float(np.sum(ov[:, g])) for g in groups_b]
    return int(np.argmax(scores)), max(scores)


def decompose(mats, constant, p, delta, weights=(1.,), branch=None):
    full = full_h(mats, constant); wf, vf = eigh(full); psi = vf[:, 0]
    u, c_tail = step(mats, constant, p, delta, weights)
    r = evaluate(u, delta, psi, full, branch=branch)
    assert r['status'] == 'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE', r.get('reason')
    hr = retained_h(mats, constant, p, c_tail); wr, vr = eigh(hr)
    groups_r = [[i] for i in range(len(wr))]
    gr, ov_r = matched(vf, [0], vr, groups_r)                 # retained group <-> full ground
    sp = r['spectrum']; ve = np.array(sp['eigenvectors']['real'])+1j*np.array(sp['eigenvectors']['imag'])
    groups_e = [g['indices'] for g in sp['energy_groups']]
    ge, ov_e = matched(vr, groups_r[gr], ve, groups_e)        # effective group <-> retained group
    e_eff = sp['energy_groups'][ge]['minimum_energy']
    trunc = float(wr[gr]-wf[0]); pf = float(e_eff-wr[gr]); total_eval = r['spectral_shift_diagnostics'][
        'minimum_branched_energy_minus_reference_matrix_minimum_ha']
    return dict(c_tail=c_tail, trunc=trunc, pf_on_retained=pf, total_matched=float(e_eff-wf[0]),
                total_evaluator_minimum=total_eval, retained_overlap=ov_r, effective_overlap=ov_e,
                ground_weight=sp['energy_groups'][ge]['weight'], evaluator=r, psi=psi, full=full, u=u)


def run():
    out = dict(source=SRC, evaluator_source=dict(commit=EVALUATOR[0], path=EVALUATOR[1]),
               pf_kernel_source=N.EXPM, H4_inputs_read=False, checks={})
    mats, c = toy(11)
    # 1. decomposition identity and evaluator total for several prefixes/formulas
    rows = []
    for p in range(len(mats)-1):
        for weights, label in (((1.,), 'S2'), (Y4, 'Y4')):
            d = decompose(mats, c, p, .2, weights)
            rows.append(dict(prefix=p, formula=label, trunc=d['trunc'], pf_on_retained=d['pf_on_retained'],
                             total=d['total_matched'], evaluator_total=d['total_evaluator_minimum'],
                             identity_error=abs(d['total_matched']-d['trunc']-d['pf_on_retained']),
                             evaluator_vs_matched=abs(d['total_matched']-d['total_evaluator_minimum']),
                             retained_overlap=d['retained_overlap'], effective_overlap=d['effective_overlap'],
                             ground_weight=d['ground_weight']))
    out['decomposition'] = rows
    out['checks']['decomposition_identity'] = all(r['identity_error'] < 1e-12 for r in rows)
    out['checks']['evaluator_total_equals_matched_total'] = all(r['evaluator_vs_matched'] < 1e-12 for r in rows)
    # 2. Weyl bound on the truncation shift
    weyl = []
    for p in range(len(mats)-1):
        tail = sum(mats[p+1:]); ct = float(np.trace(tail).real/len(tail))
        nonid = tail-ct*np.eye(len(tail)); lam = sum(np.linalg.norm(m-np.trace(m).real/len(m)*np.eye(len(m)), 2) for m in mats[p+1:])
        d = decompose(mats, c, p, .2)
        weyl.append(dict(prefix=p, trunc=d['trunc'], norm_nonidentity_tail=float(np.linalg.norm(nonid, 2)), lambda_proxy=float(lam)))
    out['weyl'] = weyl
    out['checks']['weyl_bound'] = all(abs(w['trunc']) <= w['norm_nonidentity_tail']+1e-12 <= w['lambda_proxy']+2e-12 for w in weyl)
    # 3. B0 vs B0I: exact constant shift by -c_tail (principal clearance holds here)
    p = 1; full = full_h(mats, c); psi = eigh(full)[1][:, 0]
    u0, _ = step(mats, c, p, .2, b0i=False); u1, ct = step(mats, c, p, .2)
    r0 = evaluate(u0, .2, psi, full); r1 = evaluate(u1, .2, psi, full)
    s0 = r0['spectral_shift_diagnostics']['minimum_branched_energy_minus_reference_matrix_minimum_ha']
    s1 = r1['spectral_shift_diagnostics']['minimum_branched_energy_minus_reference_matrix_minimum_ha']
    out['b0_vs_b0i'] = dict(prefix=p, c_tail=ct, shift_B0=s0, shift_B0I=s1, difference_minus_c_tail=s0-(s1-ct))
    out['checks']['b0_equals_b0i_minus_c_tail'] = abs(s0-(s1-ct)) < 1e-12
    # 4. log branch: a large scalar*delta wraps principal energies; a target-free window fixes it
    big = 2.9/1.0; mats_b = [m.copy() for m in mats]; cb = c+big+2.0  # spectrum*delta crosses +pi
    full_b = full_h(mats_b, cb); wb, vb = eigh(full_b); psi_b = vb[:, 0]
    ub, _ = step(mats_b, cb, 2, 1.0)
    rp = evaluate(ub, 1.0, psi_b, full_b)
    hb = retained_h(mats_b, cb, 2, float(np.trace(sum(mats_b[3:])).real/6))
    gersh = [float(min(np.real(np.diag(hb))-np.sum(abs(hb), 1)+abs(np.diag(hb)))),
             float(max(np.real(np.diag(hb))+np.sum(abs(hb), 1)-abs(np.diag(hb))))]
    window = [gersh[0]-.5, gersh[0]-.5+5.5]  # chosen from a Gershgorin bound of the RETAINED H, not the target
    rw = evaluate(ub, 1.0, psi_b, full_b, branch=dict(mode='ENERGY_WINDOW', purpose='EFFECTIVE_SPECTRUM_WINDOW',
                                                     window=window, source=SRC))
    def total(r):
        return None if r['status'] != 'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE' else \
            r['spectral_shift_diagnostics']['minimum_branched_energy_minus_reference_matrix_minimum_ha']
    out['branch'] = dict(delta=1.0, principal_status=rp['status'], principal_total=total(rp),
                         window=window, window_status=rw['status'], window_total=total(rw),
                         alias_period_ha=2*math.pi/1.0)
    out['checks']['principal_branch_wraps_window_recovers'] = (
        rw['status'] == 'EFFECTIVE_HAMILTONIAN_DIAGNOSTIC_AVAILABLE' and abs(total(rw)) < 1 and
        (total(rp) is None or abs(total(rp)-total(rw)) > 1))
    # 5. sign cancellation and oracle-dependent selection (search toy seeds for opposite signs)
    cancel = None
    for seed in range(40):
        m2, c2 = toy(seed, scales=(1., .6, .2, .08, .03))
        d = decompose(m2, c2, 2, .3)
        if d['trunc']*d['pf_on_retained'] < 0 and abs(d['pf_on_retained']) > .3*abs(d['trunc']):
            cancel = (seed, m2, c2); break
    assert cancel is not None
    seed, m2, c2 = cancel; ladder = [.05, .1, .15, .2, .25, .3, .35, .4, .5, .6]; grid = []
    for delta in ladder:
        d = decompose(m2, c2, 2, delta)
        lam = sum(np.linalg.norm(m-np.trace(m).real/6*np.eye(6), 2) for m in m2[3:])
        grid.append(dict(delta=delta, trunc=d['trunc'], pf=d['pf_on_retained'], total=d['total_matched'],
                         L1_weyl_plus_pf=float(lam+abs(d['pf_on_retained'])),
                         L2_abs_sum=abs(d['trunc'])+abs(d['pf_on_retained']), L3_signed=abs(d['total_matched'])))
    budget = 1.2*min(abs(g['trunc']) for g in grid)+1e-12
    pick = lambda key: max([g['delta'] for g in grid if g[key] <= budget], default=None)
    sweet = min(grid, key=lambda g: abs(g['total']))  # near-exact cancellation point of this toy
    perturbed = []
    for scale in (.9, 1.1):  # same width, discarded tail rescaled: relative sensitivity of the signed total
        m3 = [m.copy() for m in m2]; m3[3:] = [scale*m for m in m3[3:]]
        perturbed.append(dict(tail_scale=scale, signed_total=decompose(m3, c2, 2, sweet['delta'])['total_matched']))
    out['cancellation'] = dict(seed=seed, prefix=2, budget=budget, grid=grid,
        coarsest_L1=pick('L1_weyl_plus_pf'), coarsest_L2=pick('L2_abs_sum'), coarsest_L3=pick('L3_signed'),
        sweet_spot=dict(delta=sweet['delta'], signed_total=sweet['total'], trunc=sweet['trunc']),
        sweet_spot_total_after_tail_rescaling=perturbed)
    # Reported diagnostic only (not a pass/fail check): relative change of the signed total
    out['cancellation']['sweet_spot_relative_change'] = [
        abs(v['signed_total']-sweet['total'])/abs(sweet['total']) for v in perturbed]
    out['checks']['L3_le_L2_le_L1'] = all(g['L3_signed'] <= g['L2_abs_sum']+1e-15 <= g['L1_weyl_plus_pf']+2e-15 for g in grid)
    out['checks']['cancellation_widens_signed_selection'] = (out['cancellation']['coarsest_L3'] or 0) > (out['cancellation']['coarsest_L2'] or 0)
    # 6. early-round actual widths: the m-specific H_eff, not the late one, reproduces the round signal
    d1 = decompose(mats, c, 3, .1); d2 = decompose(mats, c, 3, .2)
    direct = complex(np.vdot(d1['psi'], d2['u'] @ d1['psi']))  # T=.2 with one actual step of .2
    def spectral(d, T):
        sp = d['evaluator']['spectrum']
        return complex(sum(w*np.exp(-1j*e*T) for e, w in zip(sp['energies_ha'], sp['state_weights'])))
    out['early_width'] = dict(T=.2, direct_one_step_02=[direct.real, direct.imag],
        error_using_cell_02=abs(direct-spectral(d2, .2)), error_using_cell_01=abs(direct-spectral(d1, .2)))
    out['checks']['round_specific_actual_width_required'] = (out['early_width']['error_using_cell_02'] < 1e-12
                                                           and out['early_width']['error_using_cell_01'] > 1e-9)
    # 7. state overlap: phase deviation from the matched effective ground bounded by arcsin((1-w)/w)
    worst = 0.; d = decompose(mats, c, 1, .2); w = d['ground_weight']; e = None
    sp = d['evaluator']['spectrum']; ge = int(np.argmax([g['weight'] for g in sp['energy_groups']]))
    e = sp['energy_groups'][ge]['minimum_energy']
    for T in np.linspace(.2, 400, 2001):
        z = spectral(d, T)*np.exp(1j*e*T); worst = max(worst, abs(math.atan2(z.imag, z.real)))
    out['overlap'] = dict(prefix=1, ground_group_weight=w, max_phase_deviation=worst,
                          bound=math.asin((1-w)/w) if w > .5 else None)
    out['checks']['mixture_phase_bound'] = w > .5 and worst <= math.asin((1-w)/w)+1e-12
    out['all_checks_pass'] = all(out['checks'].values())
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out'); args = parser.parse_args()
    result = run()
    text = json.dumps(result, indent=1, allow_nan=False)+'\n'
    if args.out:
        Path(args.out).write_text(text)
    print(json.dumps(dict(checks=result['checks'], cancellation={k: result['cancellation'][k] for k in
          ('seed', 'budget', 'coarsest_L1', 'coarsest_L2', 'coarsest_L3', 'sweet_spot', 'sweet_spot_total_after_tail_rescaling')},
          branch=result['branch'], overlap=result['overlap'], b0=result['b0_vs_b0i'], early=result['early_width']), indent=1))
    if not result['all_checks_pass']:
        raise SystemExit('TOY CHECK FAILED')
