"""Saved five HP arms -> unchanged branch/shot pipeline; reuse old three rows.

No step construction, signal simulation, private inputs, measurement or price.
HP arithmetic below checks scalar conversion, never selects an HP phase branch.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import time

import mpmath

from . import b2_p1_strong_pf_branch_shots as old
from . import b2_p1_strong_pf_h4_diagnosis as hp
from .pauli_signal_adapter import digest
from .rpe_resource_accounting import RPERoundSpecification

ROOT = Path(__file__).resolve().parents[2]
SIGNAL_COMMIT = '3e316e071fe671aa61b31bc89f83bf4380d0e92f'
SIGNAL_SOURCE = 'd401d82ab9e77ab43c6eec3483b3086633dcea6d'
PRECISION_SOURCE = '75cb38e686ad8b82d00d787531d767e816e1331d'
SIGNAL_PATH = 'artifacts/track_a_b2_strong_pf_hp_requalification/2026-11-07/run01/result.json'
OLD_COMMIT = '55fa5f5eb121ee01ae963da7ab6e5bbe9a507173'
OLD_SOURCE = '909171eb3e4f1f4b0ec9e4234624b7000ecd8215'
OLD_ROOT = 'artifacts/track_a_b2_p1_strong_pf_branch_shots/2026-11-07/run01'
ARMS = (('Full_S2_s4','Full_S2',4), ('Full_S2_s8','Full_S2',8),
        ('Yoshida4_s2','Yoshida4',2), ('Yoshida4_s4','Yoshida4',4), ('Yoshida4_s8','Yoshida4',8))


def context():
    ctx = mpmath.mp.clone(); ctx.dps = 120
    return ctx


def number(ctx, value):
    x = ctx.mpf(value)
    old.branch.require(ctx.isfinite(x), 'FINITE_HP_SCALAR')
    return x


def validate_shared(record, original, original_metadata):
    """Reuse old input/P1 validation without decoding/reaccounting old arms."""
    _, identity, conditions = old.validate_shared(original)
    old.branch.require(record['schema_version'] == 'b2_strong_pf_fixed_five_arm_HP_requalification_v1' and
        record['status'] == 'FIXED_FIVE_ARM_HP_REQUALIFICATION_COMPLETE' and
        record['execution_SOURCE'] == SIGNAL_SOURCE and record['precision_SOURCE'] == PRECISION_SOURCE and
        record['precision_dps'] == [80,120] and record['evaluated_points'] == 90 and
        record['historical_results_overwritten'] is False, 'FIXED_HP_SOURCE_RESULT')
    old.branch.require(record['scientific_context'] == identity and record['P1_conditions'] == conditions and
        record['original_result'] == original_metadata and
        record['historical_qualified_arms'] == original['qualified_arms'], 'SHARED_HP_ORIGINAL_P1_CONTEXT')
    old.branch.require(record['input_gate']['status'] == record['input_identity']['status'] == 'PASS' and
        all(record['input_identity'][k] == v for k,v in identity.items()) and
        record['input_identity']['file_SHA256'] == original['input_identity']['file_SHA256'], 'SHARED_HP_INPUT_IDENTITY')
    old.branch.require(record['arm_set'] == [dict(formula=f,substeps=s) for _,f,s in ARMS] and
        record['new_qualified_arms'] == [a[0] for a in ARMS], 'FIXED_HP_FIVE_ARMS')
    old.branch.require(record['reused_SOURCE_SHA256'] == original['SOURCE_SHA256'], 'REUSED_SIGNAL_SOURCE')
    return identity, conditions


def scalar_row(ctx, row):
    """Use the original HP->binary64 rule and retain every saved HP field."""
    q = row['qualification']; match = ctx.mpf(float(1e-9))
    old.branch.require(row['status'] == 'CONDITIONAL_PASS' and q['feasible'] and
        all(q['HP_tests'].values()) and row['normalization'] == row['attenuation'] == 1. and
        row['finite_RTE_disk'] == 0., 'SAVED_HP_QUALIFICATION')
    for label in ('N','C'):
        p = row['precision_check'][label]
        old.branch.require(p['passed'] is True and
            0 <= number(ctx,p['operator_max_abs_80_vs_120']) <= match and
            all(0 <= number(ctx,p['signal_difference'][k]) <= match
                for k in ('complex_absolute','radius_absolute','phase_rad')), 'SAVED_80_120_STABILITY')
    errors = {}
    converted = {}
    for label, field in (('PF','PF'),('CAR','independent_PF')):
        sig = row['signals'][field]
        z = ctx.mpc(number(ctx,sig['real']),number(ctx,sig['imag']))
        old.branch.require(abs(z) > 0, 'NONZERO_HP_SIGNAL')
        z64 = complex(float(z.real),float(z.imag))
        # Exactly the conversion used by qualify_hp; no normalization or fitting.
        comparison = hp.hp_difference(ctx,z,ctx.mpc(z64.real,z64.imag))
        saved_error = number(ctx,q['conversion_error'][label])
        old.branch.require(0 <= saved_error <= match and
            number(ctx,comparison['complex_absolute']) <= match and
            abs(number(ctx,comparison['complex_absolute'])-saved_error) <= match and
            abs(abs(z)-number(ctx,sig['radius'])) <= match and
            abs(ctx.arg(ctx.exp(ctx.j*(ctx.arg(z)-number(ctx,sig['phase_rad']))))) <= match,
            'HP_SCALAR_CONVERSION_OR_SIGNAL_RECORD')
        converted[label] = z64
        errors[label] = {k:v for k,v in comparison.items() if k != 'delta'}
    old.branch.require(all(0 <= number(ctx,q['HP_independent_difference'][k]) <= match
        for k in ('complex_absolute','radius_absolute','phase_rad')), 'SAVED_HP_INDEPENDENT_MATCH')
    phase_error = abs(ctx.arg(ctx.exp(ctx.j*(number(ctx,row['signals']['PF']['phase_rad'])-
        ctx.mpf(float(hp.old.p.signal_record(converted['PF'])['phase_rad']))))))
    hp_radius = number(ctx,q['HP_bounds']['PF_radius_lower'])
    old.branch.require(0 < hp_radius <= 1 and
        abs(hp_radius-number(ctx,row['signals']['PF']['radius'])+ctx.mpf(float(old.branch.U))) <= match and
        abs(hp_radius-ctx.mpf(q['bounds']['reference_radius_lower_bound'])) <= match, 'HP_SAVED_RADIUS_CONVERSION')
    mapped = copy.deepcopy(row)
    mapped['signals']['PF'] = hp.old.p.signal_record(converted['PF'])
    audit = dict(HP_signal=row['signals']['PF'], HP_PF_radius_lower=q['HP_bounds']['PF_radius_lower'],
        binary64_signal=mapped['signals']['PF'], saved_conversion_error=q['conversion_error'],
        checked_conversion_difference=errors, phase_conversion_error_rad=hp.hp_text(ctx,phase_error),
        unchanged_precision_dps=[80,120], tolerance_SOURCE='P1 independent_match_tolerance=1e-9',
        status='HP_SCALAR_CONVERSION_PASS')
    return mapped, audit


def branch_conversion_check(ctx, item, audits):
    """Call the same decoder on rounded saved HP phases; no HP branch search."""
    decoded = item['reconstruction']
    phase_input = [float(a['HP_signal']['phase_rad']) for a in audits]
    direct = old.branch.reconstruct_rpe_phase(phase_input,[r['q_m'] for r in decoded['rounds']])
    old.branch.require([r['selected_branch'] for r in direct['rounds']] ==
        [r['selected_branch'] for r in decoded['rounds']], 'HP_PHASE_TO_SCALAR_BRANCH_CHANGE')
    phase_errors = [number(ctx,a['phase_conversion_error_rad']) for a in audits]
    for index, row in enumerate(decoded['rounds']):
        if index:
            transported = 2*(phase_errors[index]/row['q_m']+
                phase_errors[index-1]/decoded['rounds'][index-1]['q_m'])
            old.branch.require(ctx.mpf(row['best_runner_up_distance_gap']) >
                transported+ctx.mpf(old.branch.BRANCH_AMBIGUITY_ATOL), 'BRANCH_CONVERSION_MARGIN_UNSTABLE')
        else:
            old.branch.require(ctx.mpf(decoded['initial_alias']['interval_endpoint_phase_margin_after_beta_rad']) >
                phase_errors[0], 'INITIAL_ALIAS_CONVERSION_UNSTABLE')
        # Lift only the ALREADY selected branch. This never consults an energy
        # oracle and does not implement a second nearest-branch decoder.
        phase = number(ctx,audits[index]['HP_signal']['phase_rad'])
        base = ctx.arg(ctx.exp(ctx.j*(phase+2*ctx.pi*row['selected_branch'])/row['q_m']))
        energy = -base/ctx.mpf(float(old.branch.DELTA))
        difference = abs(energy-ctx.mpf(row['energy_estimate_Ha']))
        old.branch.require(abs(base-ctx.mpf(row['base_phase_estimate'])) <=
            ctx.mpf(float(old.branch.U))/row['q_m'], 'FIXED_BRANCH_ENERGY_CONVERSION_UNSTABLE')
        audits[index].update(phase_only_decoder_branch_match=True,
            fixed_selected_branch_HP_energy_Ha=hp.hp_text(ctx,energy),
            HP_vs_binary64_energy_difference_Ha=hp.hp_text(ctx,difference),
            branch_conversion_status='PASS')
    return audits


def decode_one(arm):
    """One independent saved arm. Worker never reads private/old results."""
    item = dict(arm_id=arm['arm_id'],formula=arm['formula'],substeps=arm['substeps'])
    active_row = None
    try:
        ctx = context()
        expected = next(a for a in ARMS if a[0] == arm['arm_id'])
        old.branch.require((arm['formula'],arm['substeps']) == expected[1:], 'HP_ARM_FORMULA_SOURCE')
        old.branch.require(arm['status'] == 'P1_ALL18_HP_CONDITIONAL_SIGNAL_PASS' and
            [r['m'] for r in arm['rounds']] == list(range(18)), 'HP_ARM_ALL18_SIGNAL_PASS')
        rows, audits = [], []
        for row in arm['rounds']:
            active_row = row
            mapped, audit = scalar_row(ctx,row); rows.append(mapped); audits.append(audit)
        adapted = dict(arm,rounds=rows,status='P1_ALL18_CONDITIONAL_SIGNAL_PASS',
            state_preparation_included=False,finite_RTE=False,normalization=1.,attenuation=1.)
        item = old.decode_arms([adapted])[0]
        if item['status'] != 'DECODED_AWAITING_POSTHOC_REFERENCE':
            return item
        item['HP_conversion'] = branch_conversion_check(ctx,item,audits)
    except (ValueError,KeyError,TypeError,OverflowError) as error:
        item.update(status='STOP_HP_SIGNAL_OR_CONVERSION',stop_reason=str(error))
        if active_row is not None:
            item['failure_saved_round_evidence'] = active_row
    return item


def shot_conversion_check(ctx, row, audit, *, context_hash, arm_hash):
    """Enclose HP/saved scalar radii by floats; call unchanged shot API twice.

    These are conversion diagnostics, not new scientific conditions or a new
    shot law. Counts used in the result come from the nominal saved-radius call.
    """
    radius64 = row['PF_radius_lower']; rounded_hp = float(number(ctx,audit['HP_PF_radius_lower']))
    lower = math.nextafter(min(radius64,rounded_hp),-math.inf)
    upper = math.nextafter(max(radius64,rounded_hp),math.inf)
    counts = []
    for rho in (lower,upper):
        bounds = dict(row['phase_bounds'],reference_radius_lower_bound=rho)
        result = old.shots.account_radius_round(context_hash=context_hash,arm_hash=arm_hash,
            specification=RPERoundSpecification(row['m'],old.branch.DELTA), allocation=old.shots.ALLOCATION,
            beta_rpe=old.branch.BETA,config=None,cost=None,cost_metric='rz_count',backend_hash='no_backend',**bounds)
        old.branch.require(result['feasible'] and result['shots'] ==
            dict(cosine=row['N_X'],sine=row['N_Y']),
            'HP_RADIUS_SHOT_INTEGER_UNSTABLE: m='+str(row['m'])+' rho='+repr(rho)+
            ' nominal='+repr((row['N_X'],row['N_Y']))+' endpoint='+repr(result.get('shots')))
        counts.append(result['shots'])
    return dict(radius_float_enclosure=[lower,upper],endpoint_shots=counts,
                nominal_shots=dict(cosine=row['N_X'],sine=row['N_Y']),status='SAME_EXISTING_API_CEIL')


def complete_one(item, energy_reference, *, context_hash):
    if item['status'] != 'DECODED_AWAITING_POSTHOC_REFERENCE':
        return item
    try:
        ctx = context()
        before = [r['selected_branch'] for r in item['reconstruction']['rounds']]
        correctness = old.branch.posthoc_correctness(item['reconstruction'],
            [r['raw_phase'] for r in item['selected']],energy_reference)
        item['correctness'] = correctness
        old.branch.require(correctness['status'] == 'P1_CONDITIONAL_BRANCH_ENERGY_PASS' and before ==
            [r['selected_branch'] for r in item['reconstruction']['rounds']], 'POSTHOC_ENERGY_OR_BRANCH')
        energy_hp = number(ctx,item['HP_conversion'][-1]['fixed_selected_branch_HP_energy_Ha'])
        error_hp = abs(energy_hp-ctx.mpf(float(energy_reference)))
        delta_e = abs(energy_hp-ctx.mpf(item['reconstruction']['final_energy_estimate_Ha']))
        eps = ctx.mpf(float(old.branch.EPSILON))
        old.branch.require(bool(error_hp<=eps) == correctness['final_within_energy_target'] and
            abs(error_hp-eps) > delta_e, 'HP_ENERGY_TARGET_DECISION_UNSTABLE')
        correctness.update(HP_fixed_branch_final_error_Ha=hp.hp_text(ctx,error_hp),
            HP_scalar_energy_target_decision_match=True,
            shots_prices_total_cost='Conditional sufficient shots below; prices/total circuit cost NOT_EVALUATED')
        arm_hash = digest(dict(formula=item['formula'],substeps=item['substeps'],signal_SOURCE=SIGNAL_SOURCE,
                               precision_SOURCE=PRECISION_SOURCE,P1=old.expected_conditions()))
        rows, totals = old.account_arm(item,context_hash=context_hash,arm_hash=arm_hash)
        for row,audit in zip(rows,item['HP_conversion'],strict=True):
            audit['shot_integer_conversion'] = shot_conversion_check(ctx,row,audit,
                context_hash=context_hash,arm_hash=arm_hash)
            row.update(signal_result_commit=SIGNAL_COMMIT,signal_result_path=SIGNAL_PATH,
                signal_execution_SOURCE=SIGNAL_SOURCE,precision_SOURCE=PRECISION_SOURCE,
                HP_conversion=audit,complex_PF_signal_HP=audit['HP_signal'])
        item.update(rounds=rows,totals=totals,status=old.PASS)
    except (ValueError,KeyError,TypeError,OverflowError) as error:
        item.update(status='STOP_HP_ENERGY_SHOTS_OR_CONVERSION',stop_reason=str(error))
    return item


def load_old_comparison(identity,conditions):
    """Read and copy old results, never call decoder/shot API on old arms."""
    summary, metadata = old.branch.fixed_json(OLD_COMMIT,OLD_ROOT+'/result.json')
    old.branch.require(summary['status'] == 'THREE_ARMS_CONDITIONAL_BRANCH_SHOTS_PASS' and
        summary['execution_SOURCE'] == OLD_SOURCE and summary['scientific_context'] == identity and
        summary['fixed_conditions'] == conditions,'SHARED_OLD_HP_COMPARISON_CONTEXT')
    old.checked_source_hashes(summary)
    rows = []; inputs = [metadata]
    for arm in summary['arms']:
        old.branch.require(arm['arm_id'] in [a[0] for a in old.ARMS] and arm['status'] == old.PASS,
                           'REUSED_OLD_ARM_STATUS')
        data, provenance = old.branch.fixed_json(OLD_COMMIT,OLD_ROOT+'/'+arm['result_path'])
        old.branch.require(data['totals'] == arm['totals'] and data['execution_SOURCE'] == OLD_SOURCE and
            data['signal_result_commit'] == old.SIGNAL_COMMIT and
            data['signal_execution_SOURCE'] == old.SIGNAL_SOURCE,'REUSED_OLD_ARM_PROVENANCE')
        rows.append(comparison_row(data,provenance,reused=True)); inputs.append(provenance)
    old.branch.require([r['arm_id'] for r in rows] == [a[0] for a in old.ARMS], 'REUSED_OLD_ARM_INVENTORY')
    return rows,inputs,summary['saved_inputs'][-2:]


def comparison_row(item, provenance, *, reused):
    passed = item['status'] == old.PASS
    last = item['rounds'][-1] if passed else {}
    totals = item['totals'] if passed else {}
    return dict(arm_id=item['arm_id'],formula=item['formula'],substeps=item['substeps'],
        signal_SOURCE=old.SIGNAL_SOURCE if reused else SIGNAL_SOURCE,
        precision_SOURCE=None if reused else PRECISION_SOURCE,
        signal_result_commit=old.SIGNAL_COMMIT if reused else SIGNAL_COMMIT,
        branch_shot_execution_SOURCE=OLD_SOURCE if reused else item['execution_SOURCE'],
        result_provenance=provenance,reused_without_new_calculation=reused,
        signal_qualification='ALL18_CONDITIONAL_PASS',branch_status='UNIQUE_PASS' if passed else item['status'],
        final_energy_estimate_Ha=last.get('energy_estimate_Ha'),
        final_energy_absolute_error_Ha=item.get('correctness',{}).get('final_energy_absolute_error_Ha'),
        N_X_total=totals.get('N_X_total'),N_Y_total=totals.get('N_Y_total'),N_all_rounds=totals.get('N_all_rounds'),
        guarantee_status='not_certified',price=None,total_RPE_cost=None)


def run(output):
    output = Path(output); output.mkdir(parents=True,exist_ok=False); started = time.perf_counter()
    source = old.branch.git('rev-parse','HEAD').decode().strip()
    paths = ['src/trotterlib/b2_p1_hp_branch_shots.py','scripts/run_b2_p1_hp_branch_shots.py',
        'src/trotterlib/b2_p1_strong_pf_branch_shots.py','src/trotterlib/b2_p1_branch_validation.py',
        'src/trotterlib/b2_p1_shot_accounting.py','src/trotterlib/rpe_four_round_phase_validation.py',
        'src/trotterlib/o1_m1_resource_bridge.py','src/trotterlib/finite_rte_signal_validation.py',
        'src/trotterlib/b2_p1_strong_pf_h4_diagnosis.py']
    result = dict(schema_version='b2_p1_saved_HP_five_arm_branch_shots_v1',execution_SOURCE=source,
        started_UTC=datetime.now(timezone.utc).isoformat(),arms=[],mandatory_stop=True,guarantee_status='not_certified',
        reference_energy_used_in_decoder=False,historical_STOP_results_overwritten=False,
        forbidden_work=dict(private_input_reads=0,PF_step_or_signal_simulation=0,old_three_arm_recalculation=0,
            new_HP_requalification=0,measurement_sampling=0,circuit_build_compile=0,prices=0,total_RPE_cost=0,
            parameter_optimization=0,scientific_review=0))
    error_to_raise = None
    try:
        for path in paths:
            old.branch.require((ROOT/path).read_bytes() == old.branch.git('show',source+':'+path),'COMMITTED_SOURCE')
        result['SOURCE_SHA256'] = {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
        saved, sm = old.branch.fixed_json(SIGNAL_COMMIT,SIGNAL_PATH)
        original, om = old.branch.fixed_json(old.SIGNAL_COMMIT,old.SIGNAL_PATH)
        old.checked_source_hashes(saved); old.checked_source_hashes(original)
        identity,conditions = validate_shared(saved,original,om)
        result.update(saved_inputs=[sm,om],scientific_context=identity,fixed_conditions=conditions,
                      signal_SOURCE=SIGNAL_SOURCE,precision_SOURCE=PRECISION_SOURCE)
        arms = []
        for name,formula,substeps in ARMS:
            matches = [a for a in saved['arms'] if a['arm_id'] == name]
            arms.append(matches[0] if len(matches) == 1 else dict(arm_id=name,formula=formula,substeps=substeps,
                rounds=[],status='STOP_MISSING_OR_DUPLICATE_HP_ARM'))
        workers = min(len(arms),len(os.sched_getaffinity(0)))
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
            items = list(executor.map(decode_one,arms))
        result['decoder_workers'] = workers
        result['all_five_decoders_finished_before_energy_reference'] = True
        # Old saved summaries contain known energies: open them only now.
        old_rows, old_inputs, b2_refs = load_old_comparison(identity,conditions)
        result['saved_inputs'].extend(old_inputs)
        old.branch.require(saved['saved_references'][-2:] == b2_refs,
            'SHARED_B2_REFERENCE_PROVENANCE')
        reference, rm = old.branch.fixed_json(old.branch.STATE_COMMIT,old.branch.STATE_RESULT)
        result['posthoc_reference_input'] = rm
        energy = reference['validation']['energy_Ha']
        context_hash = digest(dict(identity=identity,P1=conditions,execution_SOURCE=source))
        for item in items:
            complete_one(item,energy,context_hash=context_hash)
            item.update(execution_SOURCE=source,signal_result_commit=SIGNAL_COMMIT,
                signal_execution_SOURCE=SIGNAL_SOURCE,precision_SOURCE=PRECISION_SOURCE,guarantee_status='not_certified')
            if item['status'] == old.PASS:
                item.pop('HP_conversion',None)  # Details retained once in each round.
            summary = old.write_arm(output/item['arm_id'],item)
            summary['result_path'] = item['arm_id']+'/result.json'
            summary['rounds_CSV_path'] = item['arm_id']+'/rounds.csv'
            # Per-round conversion details are in arm JSON, not repeated in root summary.
            summary.pop('HP_conversion',None)
            result['arms'].append(summary)
            new_row = comparison_row(item,dict(path=summary['result_path'],execution_SOURCE=source),reused=False)
            old_rows.append(new_row)
            print(item['arm_id']+' '+item['status']+
                ('; shots='+str(item['totals']['N_all_rounds']) if item['status'] == old.PASS else ''),flush=True)
        result['comparison'] = sorted(old_rows,key=lambda r:(r['formula']!='Full_S2',r['substeps']))
        with (output/'comparison.csv').open('x',newline='') as stream:
            fields = list(result['comparison'][0])
            writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for row in result['comparison']:
                writer.writerow(dict(row,result_provenance=json.dumps(row['result_provenance'],sort_keys=True)))
        result['status'] = ('FIVE_HP_ARMS_CONDITIONAL_BRANCH_SHOTS_PASS' if all(i['status'] == old.PASS for i in items)
                            else 'FIVE_HP_ARMS_COMPLETE_WITH_STOPS')
    except Exception as error:
        result.update(status='STOP_SHARED_SOURCE_INPUT_OR_EXECUTION',stop_reason=str(error)); error_to_raise = error
    result['elapsed_seconds'] = time.perf_counter()-started
    result['environment'] = dict(host=platform.node(),python=platform.python_version(),numpy=importlib.metadata.version('numpy'),
        mpmath=mpmath.__version__,CPU_affinity=len(os.sched_getaffinity(0)),load_average=os.getloadavg(),
        arbitrary_CPU_RAM_BLAS_caps=False,shared_settings_changed=False)
    (output/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if error_to_raise is not None: raise error_to_raise
    return result
