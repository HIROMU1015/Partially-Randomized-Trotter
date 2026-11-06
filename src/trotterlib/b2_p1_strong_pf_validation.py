"""Fixed eight-arm P1 signal pilot: no circuits, shots, prices or state solve.

The adapter interprets StrongBuilder's existing grouped diagonal/relative-basis
semantics using exterior-power matrices. Independent CAR block exponentials use
the existing Stage3 reference. Only one base-delta operator is powered per arm.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np

from . import b2_p1_signal_validation as p
from . import b2_p1_branch_validation as saved
from .b2_rpe_connection import strong_baseline_schedule
from .df_rte_tail import diagonal_pauli_matrix
from .o1_m1_resource_bridge import radius_phase_diagnostics
from .rpe_resource_accounting import RPERoundSpecification
from trottertracks import c1_h4_strong_baseline as strong
from trottertracks.c1_stage3_reference import _ordered_exponentials

ROOT = Path(__file__).resolve().parents[2]
ARMS = tuple((formula, s) for formula in ('Full_S2', 'Yoshida4') for s in (1, 2, 4, 8))
SHOT_COMMIT = '0d5ca357ccfd8fcc850bef800c5bcadfde2a153e'
SHOT_PATH = 'artifacts/track_a_b2_p1_shots/2026-11-07/run01/result.json'
BRANCH_COMMIT = 'd424bb8bec1aaf1224a33513a06e64801749a69f'
BRANCH_PATH = 'artifacts/track_a_b2_p1_branch/2026-11-07/run01/result.json'


def checked_references():
    """Reuse identity/provenance checks, never rerun B2 signals or decoder."""
    combined, cm = saved.fixed_json(saved.ALL_COMMIT, saved.ALL_PATH)
    new, nm = saved.fixed_json(saved.ALL_COMMIT, saved.NEW_PATH)
    end, em = saved.fixed_json(saved.END_COMMIT, saved.END_PATH)
    selected, identity, conditions = saved.validate_saved_bundle(combined, new, end)
    branch, bm = saved.fixed_json(BRANCH_COMMIT, BRANCH_PATH)
    shot, sm = saved.fixed_json(SHOT_COMMIT, SHOT_PATH)
    p.require(branch['status'] == 'P1_CONDITIONAL_BRANCH_ENERGY_PASS' and
        branch['scientific_context'] == identity and branch['fixed_conditions'] == conditions and
        branch['saved_inputs'] == [cm, nm, em], 'SAVED_BRANCH_CONTEXT')
    p.require(shot['status'] == 'ALL18_CONDITIONAL_SUFFICIENT_SHOTS' and
        shot['scientific_context'] == identity and shot['fixed_conditions'] == conditions and
        shot['saved_inputs'] == [cm, nm, em, bm], 'SAVED_B2_SHOT_CONTEXT')
    return combined['rounds'], selected, identity, conditions, [cm, nm, em, bm, sm]


def diagonal_values(primitives, modes):
    """Interpret existing RZ/RZZ/global phase specs, without creating gates."""
    phase = np.full(1 << modes, primitives.global_phase, dtype=float)
    for q, angle in primitives.rz:
        phase -= .5*angle*np.diag(diagonal_pauli_matrix(modes, (q,))).real
    for q, r, angle in primitives.rzz:
        phase -= .5*angle*np.diag(diagonal_pauli_matrix(modes, (q, r))).real
    return np.exp(1j*phase)


def sector_lift(orbital, ref, indices):
    """Restriction only after checking both sector leakage directions."""
    full = ref.fock_lift(np.asarray(orbital, complex))
    outside = np.setdiff1d(np.arange(len(full)), indices)
    leak = max(float(np.max(np.abs(full[np.ix_(outside, indices)]), initial=0)),
               float(np.max(np.abs(full[np.ix_(indices, outside)]), initial=0)))
    p.require(leak <= p.MATCH, 'GAUSSIAN_SECTOR_NOT_INVARIANT')
    return full[np.ix_(indices, indices)], leak


def native_mean_step(model, schedule, ref, indices, *, scalar_time=p.DELTA,
                     basis_cache=None, relative_cache=None):
    """Ideal semantics of StrongBuilder.sequence(relative_fusion=True).

    No Gaussian emitter, circuit object or compiler is invoked. Vacuum=1
    exterior lifts retain the analytic relative phase of the ideal emitter;
    compiled/finite-precision gate implementations remain unassessed.
    """
    basis_cache = {} if basis_cache is None else basis_cache
    relative_cache = {} if relative_cache is None else relative_cache
    leaks = []
    def basis(i):
        if i not in basis_cache:
            basis_cache[i] = sector_lift(model.blocks[i].rotation, ref, indices)
        value, leak = basis_cache[i]
        leaks.append(leak)
        return value
    groups = strong.native.runs(model, schedule, optimized=True)
    result = basis(groups[0][0][0]).conj().T
    for k, group in enumerate(groups):
        i = group[0][0]
        diagonal = diagonal_values(strong.native.combined_diagonal(model, group), model.modes)[indices]
        result = diagonal[:, None]*result
        if k+1 < len(groups):
            j = groups[k+1][0][0]
            if (i, j) not in relative_cache:
                orbital = np.asarray(model.blocks[j].rotation).conj().T @ np.asarray(model.blocks[i].rotation)
                relative_cache[i, j] = sector_lift(orbital, ref, indices)
            relative, leak = relative_cache[i, j]
            leaks.append(leak)
            result = relative @ result
        else:
            result = basis(i) @ result
    return np.exp(-1j*model.constant*float(Fraction(scalar_time)))*result, max(leaks, default=0.)


def qualify(m, z_phys, z_df, z_pf, z_other):
    """P1 budgets unchanged; deterministic RTE disk, normalization error=0."""
    comparison = p.signal_difference(z_pf, z_other)
    impl = comparison['phase_rad']
    model = p.phase_distance(z_df, z_phys)
    pf = p.phase_distance(z_pf, z_df)
    finite = all(math.isfinite(v) for z in (z_phys, z_df, z_pf, z_other) for v in (z.real, z.imag))
    numerical = finite and all(comparison[k] is not None and comparison[k] <= p.MATCH
        for k in ('complex_absolute', 'radius_absolute', 'phase_rad'))
    reasons = [] if numerical else ['independent_numerical_match']
    rho = abs(z_pf)-p.U
    bounds = dict(reference_radius_lower_bound=rho, corrected_signal_error_bound=0.,
        pf_phase_bound=None if pf is None else pf+p.U,
        model_phase_bound=None if model is None else model+p.U,
        implementation_phase_bound=None if impl is None else impl+p.U)
    if rho <= 0 or rho > 1 or any(v is None for v in bounds.values()):
        reasons.append('nonpositive_or_invalid_radius_phase')
        diagnostic = dict(feasible=False, rho_observed_lower_bound=rho, attenuation=1., finite_rte_phase_bound=0.)
    else:
        diagnostic = radius_phase_diagnostics(specification=RPERoundSpecification(m, p.DELTA),
            allocation=p.ALLOCATION, beta_rpe=.4, config=None, **bounds)
        reasons.extend(diagnostic['reasons'])
        if bounds['model_phase_bound'] > p.CAPS['model']:
            reasons.append('model_phase_cap')
        if bounds['implementation_phase_bound'] > p.CAPS['implementation']:
            reasons.append('implementation_phase_cap')
    # Keep unused RTE cap .04 in the reservation: never reallocate it.
    total = None if model is None or impl is None else (
        p.ALLOCATION.beta_total+bounds['model_phase_bound']+
        bounds['implementation_phase_bound']+p.CAPS['reserve'])
    if total is None or total > .4:
        reasons.append('P1_total_with_reserve')
    return dict(bounds=bounds, independent_difference=comparison,
        independent_numerical_match=numerical, PF_phase_observed=pf,
        model_phase_observed=model, implementation_phase_observed=impl,
        finite_RTE_complex_disk=0., finite_RTE_phase_error=0., normalization=1., attenuation=1.,
        raw_radius_lower=diagnostic['rho_observed_lower_bound'],
        phase_total_with_reserved_statistical_and_unused_RTE=total,
        RTE_budget_reserved_unreallocated=.04, feasible=not reasons, reasons=reasons,
        status='CONDITIONAL_PASS' if not reasons else 'STOP_ARM_INELIGIBLE',
        guarantee_status='not_certified', shot_accounting_called=False)


def evaluate_arm(formula, substeps, evaluate, save):
    """An arm stops at its first failure; no evaluation of later rounds."""
    rows = []
    for m in range(18):
        row = evaluate(m)
        rows.append(row)
        save(rows)
        if not row['qualification']['feasible']:
            return rows, 'STOP_ARM_INELIGIBLE'
    return rows, 'P1_ALL18_CONDITIONAL_SIGNAL_PASS'


def write_result(output, result):
    (output/'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def run(private_dir, output):
    started = time.perf_counter()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    source = p.git('rev-parse', 'HEAD').decode().strip()
    paths = ['src/trotterlib/b2_p1_strong_pf_validation.py', 'scripts/run_b2_p1_strong_pf_validation.py',
        'src/trotterlib/b2_rpe_connection.py', 'src/trotterlib/b2_p1_signal_validation.py',
        'src/trotterlib/b2_p1_branch_validation.py', 'src/trotterlib/o1_m1_resource_bridge.py',
        'src/trotterlib/rpe_resource_accounting.py', 'src/trotterlib/product_formula.py',
        'src/trotterlib/df_trotter/ops.py', 'src/trottertracks/c1_h4_strong_baseline.py',
        'src/trottertracks/c1_stage2_native.py', 'src/trottertracks/c1_stage3_reference.py',
        'scripts/run_o1_m1_h4_operator_correctness.py']
    for path in paths:
        p.require((ROOT/path).read_bytes() == p.git('show', source+':'+path), 'COMMITTED_SOURCE_REQUIRED')
    result = dict(schema_version='b2_p1_strong_pf_eight_arms_v1', execution_SOURCE=source,
        SOURCE_SHA256={path: hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in paths},
        started_UTC=datetime.now(timezone.utc).isoformat(), arms=[], mandatory_stop=True,
        guarantee_status='not_certified', no_global_phase_alignment=True, no_spin_projection=True,
        implementation_scope='Ideal relative-Gaussian/diagonal semantics vs independent CAR expm; no emitted/compiled long circuit certificate',
        not_assessed=['Strong-PF branch reconstruction', 'Strong-PF shots', 'Prices and total RPE cost',
            'Real-device 95% success', 'Rigorous floating-point enclosure', 'All-support/long compiled circuits',
            'GPT detailed scientific review'])
    try:
        references, provenance, identity, conditions, saved_inputs = checked_references()
        result.update(saved_inputs=saved_inputs, scientific_context=identity,
            P1_conditions=conditions, arm_set=[dict(formula=f, substeps=s) for f, s in ARMS])
        inputs = p.load_private_inputs(private_dir)
        p.require(all(inputs['public'][k] == v for k, v in identity.items()), 'SHARED_PRIVATE_PUBLIC_CONTEXT')
        result['input_identity'] = inputs['public']
        ham = inputs['ham']; indices = inputs['indices']; ix = np.ix_(indices, indices)
        outside = np.setdiff1d(np.arange(len(inputs['state'])), indices)
        full_generators = [inputs['one_body'], *inputs['fragments']]
        generator_leaks = [max(float(np.max(np.abs(g[np.ix_(outside, indices)]), initial=0)),
            float(np.max(np.abs(g[np.ix_(indices, outside)]), initial=0))) for g in full_generators]
        p.require(max(generator_leaks) <= p.MATCH, 'SHARED_GENERATORS_SECTOR_INVARIANCE')
        result['generator_sector_leakage_max_abs'] = generator_leaks
        generators = tuple(g[ix] for g in full_generators)
        model, reports = strong.prepare_xdf_basis(ham.one_body, ham.lambdas, ham.g_matrices,
            ham.constant, source_identity=source, input_identity=identity['df_hamiltonian'], basis_policy=strong.SPIN)
        p.require(len(model.blocks) == 13 and [b.block_id for b in model.blocks] ==
            ['one_body']+['F'+str(i) for i in range(12)], 'ALL_SIGNED_FACTORS_ORDER')
        result['spin_basis_reports'] = reports
        result['shared_input_status'] = 'PASS'
        basis_cache = {}; relative_cache = {}; exponential_cache = {}
        psi = inputs['sector_state']
        for formula, substeps in ARMS:
            arm = dict(arm_id=formula+'_s'+str(substeps), formula=formula, substeps=substeps,
                execution_SOURCE=source, scientific_context=identity, rounds=[], status='RUNNING',
                state_preparation_included=False, finite_RTE=False, normalization=1., attenuation=1.)
            result['arms'].append(arm)
            schedule = strong_baseline_schedule(len(model.blocks), substeps=substeps,
                formula=formula, delta_time=p.DELTA)
            arm['base_delta_schedule'] = [dict(block_index=i, time_fraction=str(t), time=float(t)) for i, t in schedule]
            step, leak = native_mean_step(model, schedule, inputs['ref'], indices,
                basis_cache=basis_cache, relative_cache=relative_cache)
            other_step = np.exp(-1j*ham.constant*p.DELTA)*_ordered_exponentials(
                generators, schedule, exponential_cache)
            arm['base_step_independent_max_abs'] = float(np.max(np.abs(step-other_step)))
            arm['native_sector_leakage_max_abs'] = leak
            arm['base_step_unitarity_max_abs'] = float(np.max(np.abs(step.conj().T@step-np.eye(len(psi)))))
            def evaluate(m):
                q = 1 << m
                ref = references[m]
                z_phys = saved.complex_signal(ref['signals']['phys'])
                z_df = saved.complex_signal(ref['signals']['DF'])
                z_pf = complex(np.vdot(psi, np.linalg.matrix_power(step, q)@psi))
                z_other = complex(np.vdot(psi, np.linalg.matrix_power(other_step, q)@psi))
                qualification = qualify(m, z_phys, z_df, z_pf, z_other)
                if arm['base_step_independent_max_abs'] > p.MATCH:
                    qualification['reasons'].append('independent_base_step_match')
                    qualification.update(feasible=False, status='STOP_ARM_INELIGIBLE')
                row = dict(m=m, Q=q, T_Ha_inverse=q*p.DELTA,
                    signals={k:p.signal_record(v) for k,v in (('phys',z_phys),('DF',z_df),('PF',z_pf))},
                    independent_PF=p.signal_record(z_other),
                    adjacent_errors=dict(model=p.signal_difference(z_df,z_phys), PF=p.signal_difference(z_pf,z_df)),
                    reference_provenance=provenance[m], qualification=qualification,
                    status=qualification['status'])
                # Provenance carries a raw B2 complex scalar internally; publish
                # only its source identifiers, not unused B2 signal fields.
                row['reference_provenance'] = {k:v for k,v in provenance[m].items() if k.startswith('signal_')}
                print(arm['arm_id']+' m='+str(m)+' '+row['status']+
                    ' PF_phase='+str(qualification['bounds']['pf_phase_bound']), flush=True)
                return row
            def save(rows):
                arm['rounds'] = rows
                write_result(output, result)
            rows, arm['status'] = evaluate_arm(formula, substeps, evaluate, save)
            arm['first_ineligible_round'] = rows[-1]['m'] if arm['status'].startswith('STOP') else None
            arm['first_ineligible_reasons'] = rows[-1]['qualification']['reasons'] if arm['status'].startswith('STOP') else []
            write_result(output, result)
        result['qualified_arms'] = [a['arm_id'] for a in result['arms'] if a['status']=='P1_ALL18_CONDITIONAL_SIGNAL_PASS']
        result['status'] = 'FIXED_EIGHT_ARM_SCREEN_COMPLETE'
    except Exception as error:
        result.update(status='STOP_SHARED_INPUT_SOURCE_OR_EXECUTION', error_type=type(error).__name__, error=str(error))
        write_result(output, result)
        raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        result['environment'] = dict(hostname=platform.node(), python=platform.python_version(),
            numpy=np.__version__, scipy=importlib.metadata.version('scipy'),
            CPU_affinity_count=len(os.sched_getaffinity(0)), load_average=os.getloadavg(),
            private_input_arrays_published=False, arbitrary_resource_caps=False)
        write_result(output, result)
    fields = ['arm_id', 'formula', 'substeps', 'm', 'Q', 'T_Ha_inverse',
        'z_phys_real', 'z_phys_imag', 'z_phys_radius', 'z_phys_phase',
        'z_DF_real', 'z_DF_imag', 'z_DF_radius', 'z_DF_phase',
        'z_PF_real', 'z_PF_imag', 'z_PF_radius', 'z_PF_phase',
        'model_complex_error', 'model_phase_bound', 'PF_complex_error', 'PF_phase_bound',
        'implementation_phase_bound', 'PF_radius_lower', 'raw_radius_lower',
        'independent_complex_difference', 'independent_radius_difference', 'independent_phase_difference',
        'RTE_disk', 'normalization', 'attenuation', 'status', 'reasons']
    with (output/'rounds.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n'); writer.writeheader()
        for arm in result['arms']:
            for row in arm['rounds']:
                qual = row['qualification']; bounds = qual['bounds']; dif = qual['independent_difference']
                flat = {k:arm[k] for k in ('arm_id','formula','substeps')}
                flat.update({k:row[k] for k in ('m','Q','T_Ha_inverse','status')})
                for label, sig in row['signals'].items():
                    for short, name in (('real','real'),('imag','imag'),('radius','radius'),('phase','phase_rad')):
                        flat['z_'+label+'_'+short] = sig[name]
                flat.update(model_complex_error=row['adjacent_errors']['model']['complex_absolute'],
                    model_phase_bound=bounds['model_phase_bound'], PF_complex_error=row['adjacent_errors']['PF']['complex_absolute'],
                    PF_phase_bound=bounds['pf_phase_bound'], implementation_phase_bound=bounds['implementation_phase_bound'],
                    PF_radius_lower=bounds['reference_radius_lower_bound'], raw_radius_lower=qual['raw_radius_lower'],
                    independent_complex_difference=dif['complex_absolute'], independent_radius_difference=dif['radius_absolute'],
                    independent_phase_difference=dif['phase_rad'], RTE_disk=0, normalization=1, attenuation=1,
                    reasons=';'.join(qual['reasons']))
                writer.writerow(flat)
    return result
