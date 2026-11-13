"""Approved T1: retained-H truncation shifts for saved H4 DF10, p=0..9 (B0I).

Scope (user approval of SCIENCE_PLAN @ c3ae31f3): one context load, tail
identity/lambda_R via the existing extractor for all p (saved six compared),
ten 36x36 retained-H eigensystems, overlap-matched shifts against A's SOURCE-
bound original-H eigensystem cache. No PF step, H_eff, signal, price, N or G.
Input/SOURCE mismatch -> STOP. Per-prefix numerical/correspondence failure ->
that prefix UNKNOWN. No retry, threshold change or extra prefix.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

CALLER = 'scripts/run_h4_truncated_t1.py'
PLAN = dict(commit='c3ae31f353d27a9f720f51fa0b089acd3d57e211',
            path='docs/research/truncated_pf_optimization_plan_20261114/SCIENCE_PLAN.md')
MANIFEST = dict(commit='c5466946ab0ef9b50511210ca1e58fb7147bb0f0',
                path='docs/research/baseline_finite_signal_plan_20261113/BASELINE_FINITE_MANIFEST.json')
CACHE = dict(commit='69ee0ff8898960d5f9959af8d34d89f223caec87',
             path='docs/research/h4_full_pf_heff_results_20261113/run01/EIGENSYSTEM.json', selector=['reference_comparison'])
POLICY = dict(commit='9af1ac8ed8adcaef059f439ce690b22b7a320de0',
              path='docs/research/h4_full_pf_heff_preparation_20261113/PLAN.json', selector=['numerical_policy'])
EVALUATOR = dict(commit='878000d8d5d956b1d2f5513986ae40d0612e0bdd', path='src/trottertracks/pf_effective_hamiltonian.py')
NUMERICS = ('5cc7c07e112ce97205948465c94441cba0042416', '86f37d25ffd5eee3daf7f25ce001b090b1f2cf33')
CORE = 'e417ffc9f52b13c18f5493193ce510d858664259'  # extractor and its trotterlib dependencies
EXTRACTOR_DEPS = ('src/trotterlib/df_trotter/decompose.py', 'src/trotterlib/df_rte_tail.py')
PREFIXES = tuple(range(10))
SAVED_COMPARE_REL = 1e-12  # saved vs re-extracted tail coefficients (same extractor => expected exact)
EPS_E = 0.00159360010199040
EPS_TROT = 0.00079680005099520
COMPLETE = 'T1_TRUNCATION_SHIFTS_COMPLETE_NOT_RPE_QUALIFIED'


def blob(commit, path):
    return subprocess.check_output(['git', 'show', commit+':'+path], cwd=ROOT)


def read(ref):
    value = json.loads(blob(ref['commit'], ref['path']))
    for key in ref.get('selector', ()):
        value = value[key]
    return value


def groups_function():
    """Reuse the frozen evaluator's energy-group definition (no copy of its logic)."""
    mod = types.ModuleType('frozen_pf_effective_hamiltonian')
    exec(compile(blob(EVALUATOR['commit'], EVALUATOR['path']), EVALUATOR['commit']+':'+EVALUATOR['path'], 'exec'), mod.__dict__)
    return mod._groups


def check_full_eigensystem(full_h, cache, policy):
    import numpy as np
    evals = np.asarray(cache['eigenvalues_ha'], dtype=float)
    basis = np.asarray(cache['eigenvectors']['real'])+1j*np.asarray(cache['eigenvectors']['imag'])
    eye = np.eye(len(full_h))
    diag = dict(full_H_hermiticity_defect_fro=float(np.linalg.norm(full_h-full_h.conj().T, 'fro')),
                cache_eigen_residual_fro=float(np.linalg.norm(full_h@basis-basis*evals, 'fro')),
                cache_gram_defect_fro=float(np.linalg.norm(basis.conj().T@basis-eye, 'fro')),
                cache_sorted=bool(np.all(np.diff(evals) >= 0)))
    if (evals.shape != (len(full_h),) or basis.shape != full_h.shape or not diag['cache_sorted']
            or max(diag['full_H_hermiticity_defect_fro'], diag['cache_eigen_residual_fro'])
            > policy['generator_hermiticity_and_reference_eigen_residual_ha']
            or diag['cache_gram_defect_fro'] > policy['reconstruction_fro']):
        raise ValueError('Original-H eigensystem cache does not match the restored H (input/SOURCE STOP)')
    return evals, basis, diag


def prefix_record(p, *, matrices, constant, state, full_evals, full_basis, full_ground, c_tail, lam_R, groups, policy):
    """One retained Hamiltonian. Returns a record; numerical failures -> UNKNOWN."""
    import numpy as np
    from scipy.linalg import eigh
    dim = len(state); eye = np.eye(dim, dtype=complex)
    record = dict(prefix=p, retained_blocks='one-body + fragments 0..'+str(p-1) if p else 'one-body only',
                  construction='B0I', c_tail_ha=c_tail, lambda_R=lam_R, status='UNKNOWN', reason=None,
                  delta_E_trunc_B0I_ha=None, delta_E_trunc_B0_ha=None)
    try:
        retained = (constant+c_tail)*eye+sum(matrices[:p+1], np.zeros_like(eye))
        tail_nonid = sum(matrices[p+1:], np.zeros_like(eye))-c_tail*eye
        diag = dict(retained_hermiticity_defect_fro=float(np.linalg.norm(retained-retained.conj().T, 'fro')))
        evals, vecs = eigh(retained, check_finite=True)
        diag.update(retained_eigen_residual_fro=float(np.linalg.norm(retained@vecs-vecs*evals, 'fro')),
                    retained_gram_defect_fro=float(np.linalg.norm(vecs.conj().T@vecs-eye, 'fro')))
        weights = abs(vecs.conj().T@state)**2
        diag['state_weight_sum_error'] = abs(float(np.sum(weights))-1.)
        record['numerical'] = diag
        if (max(diag['retained_hermiticity_defect_fro'], diag['retained_eigen_residual_fro'])
                > policy['generator_hermiticity_and_reference_eigen_residual_ha']
                or max(diag['retained_gram_defect_fro'], diag['state_weight_sum_error']) > policy['reconstruction_fro']):
            raise ArithmeticError('Retained eigensystem numerical gate')
        rgroups = groups(evals, weights, policy['energy_cluster_ha'])
        overlaps = abs(full_basis[:, full_ground['indices']].conj().T@vecs)**2
        trace = [float(np.sum(overlaps[:, g['indices']])) for g in rgroups]
        best = int(np.argmax(trace)); ordered = sorted(trace, reverse=True)
        matched = rgroups[best]
        tail_norm = float(np.linalg.norm(tail_nonid, 2))
        shift = matched['minimum_energy']-float(full_evals[full_ground['indices'][0]])
        record.update(
            retained_lowest_groups=[dict(index=i, energy_ha=g['minimum_energy'], multiplicity=g['multiplicity'],
                                         state_weight=g['weight'], trace_overlap_with_original_ground=trace[i])
                                    for i, g in enumerate(rgroups[:4])],
            matched_group_index=best, matched_group_is_minimum=best == 0,
            max_trace_overlap=trace[best], second_trace_overlap=ordered[1] if len(ordered) > 1 else None,
            matched_group_energy_ha=matched['minimum_energy'], matched_group_multiplicity=matched['multiplicity'],
            saved_state_weight_on_matched_group=matched['weight'],
            retained_gap_to_next_group_ha=(rgroups[best+1]['minimum_energy']-matched['maximum_energy']
                                           if best+1 < len(rgroups) else None),
            delta_E_trunc_B0I_ha=shift, delta_E_trunc_B0_ha=shift-c_tail,
            weyl=dict(sector_norm_tail_minus_c_tail_ha=tail_norm,
                      abs_shift_le_sector_norm=abs(shift) <= tail_norm+policy['generator_hermiticity_and_reference_eigen_residual_ha'],
                      sector_norm_le_lambda_R=tail_norm <= lam_R*(1+1e-12)+1e-15),
            screens=dict(abs_shift_over_eps_trot=abs(shift)/EPS_TROT, abs_shift_over_eps_E=abs(shift)/EPS_E,
                         within_eps_trot=abs(shift) <= EPS_TROT, within_eps_E=abs(shift) < EPS_E),
            eigenvalues_ha=evals.tolist(), state_weights=weights.tolist())
        if not record['weyl']['abs_shift_le_sector_norm'] or not record['weyl']['sector_norm_le_lambda_R']:
            raise ArithmeticError('Weyl/lambda_R consistency gate')
        if trace[best] <= .5 or best != 0:
            record.update(status='UNKNOWN', reason='STATE_CORRESPONDENCE_AMBIGUOUS_OR_LEVEL_CROSSING')
        else:
            record['status'] = 'TRUNCATION_SHIFT_AVAILABLE'
        record['_eigensystem'] = (evals, vecs)
    except (ArithmeticError, ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        record.update(status='UNKNOWN', reason=str(error))
    return record


def tail_coefficients(ham, saved_prefixes, extractor):
    """Extractor for all p; saved six must agree (else input/SOURCE STOP)."""
    rows = {}
    for p in PREFIXES:
        identity, magnitude = extractor(ham, p)
        row = dict(prefix=p, extracted_c_tail=identity, extracted_lambda_R=magnitude,
                   saved=None, used_c_tail=identity, used_lambda_R=magnitude, c_tail_source='EXISTING_EXTRACTOR')
        saved = saved_prefixes.get(str(p))
        if saved is not None:
            rel = lambda a, b: abs(a-b)/max(abs(b), 1e-300)
            row.update(saved=dict(scalar=saved['scalar'], lambda_R=saved['lambda_R']),
                       scalar_relative_difference=rel(identity, saved['scalar']),
                       lambda_relative_difference=rel(magnitude, saved['lambda_R']),
                       exact_match=identity == saved['scalar'] and magnitude == saved['lambda_R'],
                       used_c_tail=saved['scalar'], used_lambda_R=saved['lambda_R'],
                       c_tail_source='SAVED_CONTEXT_VALUE_AS_USED_BY_B0I_SIGNALS')
            if max(row['scalar_relative_difference'], row['lambda_relative_difference']) > SAVED_COMPARE_REL:
                raise ValueError('Saved tail coefficient differs from existing extractor at p='+str(p))
        rows[p] = row
    return rows


def execute(out, source):
    import numpy as np
    from trottertracks import baseline_signal_numerics as N
    out = Path(out)
    if out.exists():
        raise ValueError('Fresh output directory required; no retry/resume')
    out.mkdir(parents=True)
    def write(name, value):
        with (out/name).open('x') as f:
            json.dump(value, f, indent=1, allow_nan=False); f.write('\n')
    started = datetime.now(timezone.utc).isoformat(); tick = time.perf_counter()
    result = dict(status='STARTED', source=dict(commit=source, caller=CALLER), plan=PLAN,
                  counts=dict(context_loads=0, extractor_calls=0, retained_eigensystems=0,
                              PF_steps=0, H_eff=0, signals=0, prices=0, N_MC_G=0),
                  errors=[], automatic_retry=False, mandatory_stop=True)
    try:
        fixed = {CALLER: source, 'src/trottertracks/baseline_signal_numerics.py': NUMERICS[0]}
        fixed.update({path: CORE for path in EXTRACTOR_DEPS})
        for path, commit in fixed.items():
            if (ROOT/path).read_bytes() != blob(commit, path):
                raise ValueError('Scientific SOURCE differs: '+path)
        if blob(NUMERICS[0], 'src/trottertracks/baseline_signal_numerics.py') != blob(NUMERICS[1], 'src/trottertracks/baseline_signal_numerics.py'):
            raise ValueError('C numerics aliases differ')
        manifest = read(MANIFEST); binding = manifest['shared']['binding']
        cache = read(CACHE); policy = read(POLICY)
        anchor = read(dict(commit=CACHE['commit'], path=CACHE['path']))
        if anchor['reference']['hamiltonian'] != binding['hamiltonian']:
            raise ValueError('Cached eigensystem Hamiltonian binding differs')
        groups = groups_function()
        restored = N.load_shared_h4(ROOT, binding); result['counts']['context_loads'] = 1
        saved, ctx_binding, metadata = restored
        ctx = N.Context(ROOT, *restored)
        full_h = np.asarray(saved.full, dtype=np.complex128)
        rebuilt = ctx.constant*np.eye(len(ctx.state))+sum(ctx.matrices, np.zeros_like(full_h))
        assembly = float(np.linalg.norm(full_h-rebuilt, 'fro'))
        if assembly > policy['generator_hermiticity_and_reference_eigen_residual_ha']:
            raise ValueError('Restored full H differs from constant + matrices')
        full_evals, full_basis, full_diag = check_full_eigensystem(full_h, cache, policy)
        full_weights = abs(full_basis.conj().T@ctx.state)**2
        full_groups = groups(full_evals, full_weights, policy['energy_cluster_ha'])
        target = manifest['shared']['target']
        write('INPUT_PROVENANCE.json', dict(binding=ctx_binding, inputs=metadata['source'],
            context_prefix_keys=sorted(metadata['prefixes'], key=int), sector_dimension=len(ctx.state),
            matrices=len(ctx.matrices), constant_ha=ctx.constant, full_H_assembly_difference_fro=assembly,
            original_eigensystem=dict(source=CACHE, **full_diag, minimum_ha=float(full_evals[0]),
                lowest_group=dict(indices=full_groups[0]['indices'], multiplicity=full_groups[0]['multiplicity'],
                                  saved_state_weight=full_groups[0]['weight']),
                second_group_energy_ha=full_groups[1]['minimum_energy']),
            declared_solver_proxy_ha=target.get('reference_solver_energy_ha'),
            numerical_policy=dict(source=POLICY, **{k: policy[k] for k in (
                'generator_hermiticity_and_reference_eigen_residual_ha', 'reconstruction_fro', 'energy_cluster_ha')}),
            scientific_sources=fixed, evaluator_groups=EVALUATOR, manifest=MANIFEST))
        extractor = N.kernels(str(ROOT)).matrix_only_tail_coefficients
        coefficients = tail_coefficients(ctx.ham, metadata['prefixes'], extractor)
        result['counts']['extractor_calls'] = len(coefficients)
        records = []; eig = {}
        for p in PREFIXES:
            row = coefficients[p]
            rec = prefix_record(p, matrices=ctx.matrices, constant=ctx.constant, state=ctx.state,
                full_evals=full_evals, full_basis=full_basis, full_ground=full_groups[0],
                c_tail=row['used_c_tail'], lam_R=row['used_lambda_R'], groups=groups, policy=policy)
            result['counts']['retained_eigensystems'] += 1
            system = rec.pop('_eigensystem', None)
            if system is not None:
                eig['p%d_eigenvalues' % p], eig['p%d_eigenvectors' % p] = system
            rec['tail_coefficients'] = row
            rec['declared_solver_proxy_minus_matrix_minimum_ha'] = (target['reference_solver_energy_ha']-float(full_evals[0])
                                                                    if target.get('reference_solver_energy_ha') is not None else None)
            records.append(rec); write('prefix_%d.json' % p, rec)
        with (out/'RETAINED_EIGENSYSTEMS.npz').open('xb') as f:
            np.savez(f, **eig)
        result.update(status=COMPLETE, prefixes=[dict(prefix=r['prefix'], status=r['status'], reason=r['reason'],
            delta_E_trunc_B0I_ha=r['delta_E_trunc_B0I_ha'], delta_E_trunc_B0_ha=r['delta_E_trunc_B0_ha'],
            max_trace_overlap=r.get('max_trace_overlap'), matched_group_is_minimum=r.get('matched_group_is_minimum'),
            saved_state_weight_on_matched_group=r.get('saved_state_weight_on_matched_group'),
            c_tail_ha=r['c_tail_ha'], lambda_R=r['lambda_R'],
            sector_norm_tail_minus_c_tail_ha=(r.get('weyl') or {}).get('sector_norm_tail_minus_c_tail_ha'))
            for r in records], rpe_bias_ha=None, sufficient_N=None, G=None,
            interpretation='SPECTRAL_TRUNCATION_DIAGNOSTIC_NOT_RPE_RMSE_OR_FEASIBILITY')
    except Exception as error:
        result.update(status='STOP_NO_RETRY', errors=[dict(type=type(error).__name__, message=str(error))])
    result.update(started_UTC=started, finished_UTC=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.perf_counter()-tick, python=sys.version.split()[0], executable=sys.executable,
                  numpy=__import__('numpy').__version__, scipy=__import__('scipy').__version__,
                  platform=platform.platform())
    write('RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True); parser.add_argument('--source', required=True)
    args = parser.parse_args()
    r = execute(args.out, args.source)
    print(json.dumps({k: r[k] for k in ('status', 'counts', 'errors')}, indent=1))
