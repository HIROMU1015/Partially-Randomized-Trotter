"""Fixed three PASS strong-PF arms: saved means -> existing decoder/shots.

No private arrays, new PF signals/operators, circuits, prices or STOP-arm replay.
The original-H energy record is read only after every arm's decoder has finished.
"""
from dataclasses import asdict
from datetime import datetime, timezone
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

from . import b2_p1_branch_validation as branch
from . import b2_p1_shot_accounting as shots
from .o1_m1_resource_bridge import radius_phase_diagnostics
from .pauli_signal_adapter import digest
from .rpe_resource_accounting import RPERoundSpecification

ROOT = Path(__file__).resolve().parents[2]
SIGNAL_COMMIT = '2583ce4957ed44e914ce6571e3e676830439dfa7'
SIGNAL_SOURCE = 'a7940b58c6cb939d77e9f8e44d214b3787b7e101'
SIGNAL_PATH = 'artifacts/track_a_b2_p1_strong_pf/2026-11-07/run01/result.json'
P1_COMMIT = '7bc952772e451f27c1654dab6803192f2b204c9f'
P1_PATH = 'docs/research/track_a_b2_signal_conditions_20261106/REPORT.md'
B2_SHOT_COMMIT = '0d5ca357ccfd8fcc850bef800c5bcadfde2a153e'
B2_SHOT_PATH = 'artifacts/track_a_b2_p1_shots/2026-11-07/run01/result.json'
ARMS = (('Full_S2_s1', 'Full_S2', 1), ('Full_S2_s2', 'Full_S2', 2),
        ('Yoshida4_s1', 'Yoshida4', 1))
PASS = 'P1_CONDITIONAL_BRANCH_ENERGY_AND_SHOTS_PASS'


def expected_conditions():
    return dict(target='H_phys', DF_rank=12, L_D=3, r=4, K=2, delta=branch.DELTA,
        energy_precision_Ha=branch.EPSILON, success_probability_target=.95,
        beta_RPE=branch.BETA, energy_prior_Ha=list(branch.PRIOR), energy_shift_Ha=0,
        phase_convention='exp(-itH); phi=-delta E', caps_rad=shots.CAPS,
        allocation=asdict(shots.ALLOCATION), numerical_allowance=branch.U,
        independent_match_tolerance=1e-9)


def checked_source_hashes(record):
    for path, expected in record['SOURCE_SHA256'].items():
        actual = hashlib.sha256(branch.git('show', record['execution_SOURCE']+':'+path)).hexdigest()
        branch.require(actual == expected, 'SAVED_EXECUTION_SOURCE_HASH:'+path)


def validate_shared(record):
    """Shared metadata only: never inspect/recompute any of the five STOP arms."""
    branch.require(record['schema_version'] == 'b2_p1_strong_pf_eight_arms_v1' and
        record['status'] == 'FIXED_EIGHT_ARM_SCREEN_COMPLETE' and
        record['execution_SOURCE'] == SIGNAL_SOURCE and record['shared_input_status'] == 'PASS',
        'SHARED_STRONG_PF_STATUS_SOURCE')
    branch.require(record['P1_conditions'] == expected_conditions(), 'FIXED_P1_CONDITIONS')
    identity = record['scientific_context']
    branch.require(identity['original_hamiltonian'] ==
        'H_phys:S0_array_hash:bab58c17220230196aff8a7c7d1e1264c36e101f0fe6bf367eacba4538ae10b2' and
        identity['original_state'] ==
        'S0_array_hash:1a2df11b16743e68fb27a3f4b39e78e145897cb49a8cc931b080e44e7400c1db' and
        identity['df_hamiltonian'] ==
        'de7a549238e3a21f15a84018bef28440c345b31030282c01cf874f3d1d212424' and
        identity['sector'] ==
        'Nalpha=2,Nbeta=2;OF_basis_S0_hash:665f72f65a4300043a32f3aabc76b7deafbd2563ad3947228dd937bc6d521dd7' and
        identity['spin_orbital_order'] == '2p=alpha,2p+1=beta;Qiskit full index=sum(n_q*2**q)' and
        identity['state_role'] == 'original_H_ground_state' and
        identity['state_preparation_included'] is False and identity['DF_state_used'] is False,
        'SHARED_ORIGINAL_H_STATE_DF_SECTOR_ORDER')
    branch.require(record['input_identity']['status'] == 'PASS' and
        all(record['input_identity'][k] == v for k, v in identity.items()), 'SHARED_INPUT_IDENTITY')
    branch.require(record['qualified_arms'] == [a[0] for a in ARMS], 'FIXED_THREE_PASS_ARMS')
    selected = []
    for name, formula, substeps in ARMS:
        matches = [a for a in record['arms'] if a['arm_id'] == name]
        branch.require(len(matches) == 1, 'SHARED_ARM_INVENTORY')
        arm = matches[0]
        branch.require(arm['execution_SOURCE'] == SIGNAL_SOURCE and arm['scientific_context'] == identity,
                       'SHARED_ARM_SOURCE_IDENTITY')
        branch.require(arm['formula'] == formula and arm['substeps'] == substeps,
                       'FIXED_FORMULA_SUBSTEPS')
        selected.append(arm)
    return selected, identity, record['P1_conditions']


def validate_arm(arm):
    """Only supplied PF scalars/bounds are evaluated; no PF kernel is called."""
    branch.require(arm['status'] == 'P1_ALL18_CONDITIONAL_SIGNAL_PASS', 'ARM_NOT_ALL18_SIGNAL_PASS')
    branch.require(arm['state_preparation_included'] is False and arm['finite_RTE'] is False and
                   arm['normalization'] == arm['attenuation'] == 1., 'DETERMINISTIC_NO_PREP_SCOPE')
    rows = arm['rounds']
    branch.require([r['m'] for r in rows] == list(range(18)), 'ARM_ALL18_ROUND_COMPLETENESS')
    selected = []
    for m, row in enumerate(rows):
        branch.require(type(row['m']) is int and type(row['Q']) is int and
            row['Q'] == 1 << m and row['T_Ha_inverse'] == (1 << m)*branch.DELTA, 'ARM_ROUND_Q_TIME')
        qualification = row['qualification']
        branch.require(row['status'] == qualification['status'] == 'CONDITIONAL_PASS' and
            qualification['feasible'] is True and not qualification['reasons'] and
            qualification['independent_numerical_match'] is True, 'SAVED_ROUND_SIGNAL_QUALIFICATION')
        z = branch.complex_signal(row['signals']['PF'])
        saved_radius = branch.finite(row['signals']['PF']['radius'])
        saved_phase = branch.finite(row['signals']['PF']['phase_rad'])
        branch.require(abs(abs(z)-saved_radius) <= 1e-9 and
            branch._phase_distance(float(np.angle(z)), saved_phase) <= 1e-9, 'SAVED_PF_RADIUS_PHASE')
        bounds = qualification['bounds']
        rho = abs(z)-branch.U
        branch.require(0 < rho <= 1 and rho == bounds['reference_radius_lower_bound'],
                       'PF_RADIUS_MINUS_FIXED_ALLOWANCE')
        branch.require(bounds['corrected_signal_error_bound'] == 0. and
            qualification['finite_RTE_complex_disk'] == qualification['finite_RTE_phase_error'] == 0. and
            qualification['normalization'] == qualification['attenuation'] == 1. and
            qualification['RTE_budget_reserved_unreallocated'] == shots.CAPS['RTE'], 'FIXED_A1_DISK0_RTE_RESERVE')
        for key in ('complex_absolute', 'radius_absolute', 'phase_rad'):
            value = branch.finite(qualification['independent_difference'][key])
            branch.require(0 <= value <= 1e-9, 'SAVED_INDEPENDENT_NUMERICAL_MATCH')
        diagnostic = radius_phase_diagnostics(specification=RPERoundSpecification(m, branch.DELTA),
            allocation=shots.ALLOCATION, beta_rpe=branch.BETA, config=None, **bounds)
        branch.require(diagnostic['feasible'] and
            diagnostic['rho_observed_lower_bound'] == qualification['raw_radius_lower'] and
            bounds['model_phase_bound'] <= shots.CAPS['model'] and
            bounds['implementation_phase_bound'] <= shots.CAPS['implementation'], 'P1_RADIUS_PHASE_CONDITIONS')
        total = shots.ALLOCATION.beta_total+bounds['model_phase_bound']+bounds['implementation_phase_bound']+shots.CAPS['reserve']
        branch.require(total == qualification['phase_total_with_reserved_statistical_and_unused_RTE'] and
                       total <= branch.BETA, 'P1_TOTAL_RESERVE_UNCHANGED')
        selected.append(dict(row=row, raw=z, raw_phase=float(np.angle(z)), bounds=dict(bounds,
            radius_provenance='saved strong-PF signal on original-H state; abs(z_PF)-P1 u_radius; conditional numerical allowance')))
    return selected


def decode_arms(arms):
    """Every decoder completes or stops before any energy-reference callback."""
    results = []
    for arm in arms:
        item = dict(arm_id=arm['arm_id'], formula=arm['formula'], substeps=arm['substeps'])
        try:
            selected = validate_arm(arm)
            item['selected'] = selected
            decoded = branch.decode_raw_means([r['raw'] for r in selected], [r['row']['Q'] for r in selected])
            item['reconstruction'] = decoded
            branch.require(decoded['status'] == 'RAW_MEAN_DECODE_PASS', 'ARM_BRANCH_INELIGIBLE')
            item['status'] = 'DECODED_AWAITING_POSTHOC_REFERENCE'
        except (ValueError, KeyError, TypeError) as error:
            item.update(status='STOP_ARM_BRANCH_OR_SIGNAL', stop_reason=str(error))
        results.append(item)
    return results


def account_arm(item, *, context_hash, arm_hash):
    """Translate strong-PF saved fields to the existing B2 shot adapter."""
    rows = []
    for evidence, decoded in zip(item['selected'], item['reconstruction']['rounds'], strict=True):
        original = evidence['row']; q = original['qualification']
        proxy = dict(m=original['m'], Q=original['Q'], T_Ha_inverse=original['T_Ha_inverse'],
            normalization=1., attenuation=1., qualification=dict(
                rho_observed_lower_bound=q['raw_radius_lower'], finite_rte_phase_bound=0.,
                phase_total_with_reserved_statistical=q['phase_total_with_reserved_statistical_and_unused_RTE']))
        account = shots.account_saved_round(proxy, evidence['bounds'], None,
                                          context_hash=context_hash, arm_hash=arm_hash)
        account.update(arm_id=item['arm_id'], signal_result_commit=SIGNAL_COMMIT,
            signal_result_path=SIGNAL_PATH, signal_execution_SOURCE=SIGNAL_SOURCE,
            complex_PF_signal=original['signals']['PF'], complex_raw_signal=original['signals']['PF'],
            raw_phase_rad=evidence['raw_phase'], branch_index=decoded['selected_branch'],
            branch=decoded, energy_estimate_Ha=decoded['energy_estimate_Ha'],
            energy_absolute_error_Ha=decoded['energy_absolute_error_Ha'],
            fixed_phase_caps_rad=dict(shots.CAPS), fixed_allocation=asdict(shots.ALLOCATION),
            RTE_budget_reserved_unreallocated=shots.CAPS['RTE'])
        rows.append(account)
    return rows, shots.aggregate_shots(rows)


def complete_arms(items, load_reference, *, context_hash):
    """Known energy enters posthoc correctness only, after all decoding."""
    for item in items:
        branch.require(item['status'].startswith('STOP_') or
            item['reconstruction']['all_decoding_complete'], 'ALL_DECODERS_BEFORE_ENERGY_REFERENCE')
    if not any(i['status'] == 'DECODED_AWAITING_POSTHOC_REFERENCE' for i in items):
        return None
    reference, metadata = load_reference()
    energy = reference['validation']['energy_Ha']
    for item in items:
        if item['status'] != 'DECODED_AWAITING_POSTHOC_REFERENCE':
            continue
        try:
            before = [r['selected_branch'] for r in item['reconstruction']['rounds']]
            correctness = branch.posthoc_correctness(item['reconstruction'],
                [r['raw_phase'] for r in item['selected']], energy)
            item['correctness'] = correctness
            branch.require(before == [r['selected_branch'] for r in item['reconstruction']['rounds']],
                           'POSTHOC_MUST_NOT_REPAIR_BRANCHES')
            branch.require(correctness['status'] == 'P1_CONDITIONAL_BRANCH_ENERGY_PASS', 'ARM_ENERGY_INELIGIBLE')
            rows, totals = account_arm(item, context_hash=context_hash,
                arm_hash=digest(dict(formula=item['formula'], substeps=item['substeps'],
                    signal_SOURCE=SIGNAL_SOURCE, deterministic=True, P1=expected_conditions())))
            correctness['shots_prices_total_cost'] = 'Conditional sufficient shots below; prices/total circuit cost NOT_EVALUATED'
            item.update(rounds=rows, totals=totals, status=PASS)
        except (ValueError, KeyError, TypeError, OverflowError) as error:
            item.update(status='STOP_ARM_ENERGY_OR_SHOTS', stop_reason=str(error))
    return metadata


def write_arm(output, item):
    """Reuse the existing exact CSV/JSON shot readback verification."""
    output.mkdir()
    public = {k:v for k,v in item.items() if k != 'selected'}
    fields = ['arm_id', 'm', 'Q', 'T_Ha_inverse', 'z_PF_real', 'z_PF_imag', 'z_PF_radius',
        'raw_phase_rad', 'branch_index', 'energy_estimate_Ha', 'energy_absolute_error_Ha',
        'PF_radius_lower', 'attenuation', 'corrected_error_disk', 'N_X', 'N_Y', 'N_round',
        'beta_stat', 'alpha_X', 'alpha_Y', 'alpha_each_exact', 'phase_caps_rad',
        'P1_total_with_reserve', 'RTE_budget_reserved_unreallocated', 'status', 'guarantee_status']
    with (output/'rounds.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
        writer.writeheader()
        for row in public.get('rounds', []):
            writer.writerow(dict(row, z_PF_real=row['complex_PF_signal']['real'],
                z_PF_imag=row['complex_PF_signal']['imag'], z_PF_radius=row['complex_PF_signal']['radius'],
                phase_caps_rad=json.dumps(row['fixed_phase_caps_rad'], sort_keys=True, separators=(',', ':'))))
    path = output/'result.json'
    path.write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if item['status'] == PASS:
        public['serialization_check'] = shots.verify_serialized_totals(output, item['totals'])
        path.write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    return {k:v for k,v in public.items() if k not in ('rounds', 'reconstruction')}


def check_b2_context(identity, conditions, strong):
    """Metadata-only comparisons to the fixed B2 branch/shot results."""
    b, bm = branch.fixed_json(shots.BRANCH_COMMIT, shots.BRANCH_PATH)
    s, sm = branch.fixed_json(B2_SHOT_COMMIT, B2_SHOT_PATH)
    branch.require(b['status'] == 'P1_CONDITIONAL_BRANCH_ENERGY_PASS' and
        s['status'] == 'ALL18_CONDITIONAL_SUFFICIENT_SHOTS', 'FIXED_B2_REFERENCE_STATUS')
    branch.require(all(r['scientific_context'] == identity and r['fixed_conditions'] == conditions
                       for r in (b, s)), 'SHARED_B2_STRONG_P1_CONTEXT')
    branch.require(strong['saved_inputs'][-2:] == [bm, sm] and s['saved_inputs'][-1] == bm,
                   'SHARED_B2_REFERENCE_PROVENANCE')
    for record in (b, s):
        checked_source_hashes(record)
    return [bm, sm], dict(total_shots=s['totals']['N_all_rounds'],
        role='fixed B2 quotation only; never substituted into strong-PF counts',
        price_or_total_circuit_cost_comparison='NOT_EVALUATED')


def run(output):
    started = time.perf_counter(); output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = branch.git('rev-parse', 'HEAD').decode().strip()
    paths = ['src/trotterlib/b2_p1_strong_pf_branch_shots.py',
        'scripts/run_b2_p1_strong_pf_branch_shots.py', 'src/trotterlib/b2_p1_branch_validation.py',
        'src/trotterlib/b2_p1_shot_accounting.py', 'src/trotterlib/rpe_four_round_phase_validation.py',
        'src/trotterlib/o1_m1_resource_bridge.py', 'src/trotterlib/rpe_resource_accounting.py',
        'src/trotterlib/finite_rte_signal_validation.py']
    result = dict(schema_version='b2_p1_strong_pf_saved_branch_shots_v1', execution_SOURCE=source,
        started_UTC=datetime.now(timezone.utc).isoformat(), arms=[], mandatory_stop=True,
        guarantee_status='not_certified', reference_energy_used_in_decoder=False,
        not_certified=['Real-device 95% success', 'Rigorous numerical/implementation bounds',
            'All-support/long compiled-circuit correctness', 'Circuit prices/total RPE cost'],
        assumptions=['Fixed P1 engineering allowances and saved original-H-state signal qualifications',
            'Independent bounded +/-1 Hadamard outcomes on each axis; ideal noiseless model',
            'Existing strict beta<pi/3 branch condition and adopted initial energy prior'],
        forbidden_work=dict(private_input_reads=0, new_PF_signal_computation=0, stopped_arm_reanalysis=0,
            SCF_DF_state_solve=0, step_precision_changes=0, sampling=0, circuit_build_compile=0,
            price_MC=0, total_RPE_cost=0, optimization=0, scientific_review=0))
    error_to_raise = None
    try:
        for path in paths:
            branch.require((ROOT/path).read_bytes() == branch.git('show', source+':'+path), 'COMMITTED_SOURCE_REQUIRED')
        result['SOURCE_SHA256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
        strong, signal_metadata = branch.fixed_json(SIGNAL_COMMIT, SIGNAL_PATH)
        checked_source_hashes(strong)
        selected, identity, conditions = validate_shared(strong)
        p1_bytes = branch.git('show', P1_COMMIT+':'+P1_PATH)
        p1_metadata = dict(commit=P1_COMMIT, path=P1_PATH,
                           blob=branch.git('rev-parse', P1_COMMIT+':'+P1_PATH).decode().strip(),
                           sha256=hashlib.sha256(p1_bytes).hexdigest())
        result.update(saved_inputs=[signal_metadata, p1_metadata], scientific_context=identity,
            fixed_conditions=conditions, shared_input_status='PASS', selected_arms=[a[0] for a in ARMS])
        items = decode_arms(selected)
        # Explicit boundary: no original energy reference is opened until this
        # function has finished every selected arm's full decoder (or its STOP).
        result['all_arm_decoders_finished_before_energy_reference'] = True
        context_hash = digest(dict(identity=identity, P1=conditions, execution_SOURCE=source))
        references, b2_quotation = check_b2_context(identity, conditions, strong)
        result['saved_inputs'].extend(references)
        result['B2_reference_quotation'] = b2_quotation
        reference_metadata = complete_arms(items,
            lambda: branch.fixed_json(branch.STATE_COMMIT, branch.STATE_RESULT), context_hash=context_hash)
        result['posthoc_reference_input'] = reference_metadata
        for item in items:
            item.update(execution_SOURCE=source, signal_result_commit=SIGNAL_COMMIT,
                signal_execution_SOURCE=SIGNAL_SOURCE, guarantee_status='not_certified')
            summary = write_arm(output/item['arm_id'], item)
            summary['result_path'] = item['arm_id']+'/result.json'
            summary['rounds_CSV_path'] = item['arm_id']+'/rounds.csv'
            result['arms'].append(summary)
            print(item['arm_id']+' '+item['status']+
                  ('; shots='+str(item['totals']['N_all_rounds']) if item['status'] == PASS else ''), flush=True)
        result['status'] = ('THREE_ARMS_CONDITIONAL_BRANCH_SHOTS_PASS' if all(i['status'] == PASS for i in items)
                            else 'SELECTED_ARMS_COMPLETE_WITH_STOPS')
    except Exception as error:
        result.update(status='STOP_SHARED_INPUT_SOURCE_OR_EXECUTION', stop_reason=str(error))
        error_to_raise = error
    result['elapsed_seconds'] = time.perf_counter()-started
    result['environment'] = dict(host=platform.node(), python=platform.python_version(),
        numpy=np.__version__, scipy=importlib.metadata.version('scipy'),
        CPU_affinity_count=len(os.sched_getaffinity(0)), load_average=os.getloadavg(),
        BLAS_env={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        gpu_used=False, arbitrary_resource_caps=False)
    (output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if error_to_raise is not None:
        raise error_to_raise
    return result
