"""Saved-scalar descriptive mechanism analysis; passive import/preflight.

Crossed shots/prices are accounting counterfactuals, never new physical arms.
No fitter, shot solver, sampler, operator, circuit, or certificate is called.
"""
import math
from pathlib import Path
import resource
import subprocess
import time

from . import c1_stage1_execution as io
from . import c1_stage3_accuracy as accuracy
from .c1_stage3_integrated import Calls

VERSION = 'saved_PR_shots_price_descriptive_decomposition_v1'
SCOPE = 'C_PR_AWARE_DF_SAVED_MECHANISM_M1_V1'
BASE = '0075f660b1e3eaa3ca93347e6249d3c8c5ffe987'
FRESH = 'artifacts/research_c_pr_aware_df_mechanism/2026-10-28/M1-run01'
ARMS = ('baseline', 'proposed')
LANES = ('ideal_exact_native', 'assumed_raw_signal_budget')
METRICS = accuracy.METRICS
COST_SCOPE = 'serial_native_kernel_composition_without_state_preparation'
LIMITS = dict(saved_bundle_load=1, identity_gate=1, scenario_attachment=2,
              counterfactual_corner=8, descriptive_decomposition=2,
              tradeoff_table=2, calibration_uncertainty=2,
              reconciliation=1, result_summary=1)
MAX_CALLS = sum(LIMITS.values())
CONDITIONS = dict(molecule='linear H4', adjacent_distance_A=1., basis='STO-3G',
    Nalpha=2, Nbeta=2, sector_dimension=36, DF='normal number CDF', rank=8,
    original_frame_order=list(range(8)), L_D=2, PF='partial ordered symmetric S2',
    delta=.1, s=4, h_exact='1/40', r=16, K=0, rounds=list(range(13)),
    CA_Ha=.00159, DF_reserve_Ha=.00019875, beta_total='pi/6', beta_stat='pi/12',
    failure_per_candidate=.01, failure_per_axis='.01/26',
    state='own_DF_ground_state_oracle', state_preparation_cost_excluded=True,
    native_policy='saved B2 run02 complete kernels;head-internal relative fusion;equal-tail-basis sharing',
    compiler='saved P0 Qiskit1.3.0 opt1 seed17 rz,sx,x,cx MPFR160 declared phase',
    coherent_occurrences='fresh independent trajectory per occurrence in the assumed cost model',
    depth_semantics='scheduled_serial_sum', paired_calibration_cohorts=False)
RECONCILIATION = dict(rel_tol=1e-12, abs_tol=1e-6,
    meaning='scalar accounting closure only;not a physical/norm/phase allowance')


def preflight(root, contract_path):
    """Only contract, SOURCE and input bytes; saved numeric packets stay closed."""
    root = Path(root).resolve()
    c = io.read(contract_path)
    if (c['scope'] != SCOPE or c['implementation'] != VERSION or c['base_commit'] != BASE
            or c['status'] != 'UNAPPROVED' or c['executable'] is not True
            or c['conditions'] != CONDITIONS or c['lanes'] != list(LANES)
            or c['call_limits'] != LIMITS or c['maximum_logical_units'] != MAX_CALLS
            or c['fresh_output'] != FRESH or c['retry'] or c['resume']
            or c['reconciliation'] != RECONCILIATION):
        raise ValueError('M1_FIXED_SCOPE')
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('M1_SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'], c['saved_inputs']):
        for name, expected in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != expected:
                raise ValueError('SOURCE_OR_INPUT: '+name)
    if not set(c['records'].values()) <= set(c['saved_inputs']):
        raise ValueError('UNPINNED_RECORD')
    if (root/FRESH).exists():
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_RESUME')
    return c


def load_bundle(root, c):
    return {key: io.read(Path(root)/name) for key, name in c['records'].items()}


def identity_gate(b, c):
    """Saved provenance/verdicts; no performance aggregation here."""
    if (b['C1_SOURCE']['execution_commit'] != 'b1c735da394d4b0a444cbc618ead9781ef4d9134'
            or b['C1_result']['status'] != 'C1_SAVED_SCALAR_SCENARIOS_COMPLETE'
            or b['C1_STOP']['scientific_units'] != 164 or b['C1_STOP']['failed']
            or b['C1_STOP']['retry'] or b['C1_STOP']['resume']
            or not b['C1_STOP']['mandatory_stop']):
        raise ValueError('C1_SOURCE_OR_STOP')
    prior = b['C1_contract']
    for key in ('A_result', 'B1_result', 'B2_result'):
        path = c['records'][key]
        if prior['records'][key] != path or prior['saved_inputs'][path] != c['saved_inputs'][path]:
            raise ValueError('C1_SAVED_RECORD_BINDING')
    if (b['A_result']['status'] != 'STAGE_A_PROXY_AND_DF_PASS'
            or b['B1_result']['status'] != 'B1_MECHANISM_EVALUATION_COMPLETE'
            or b['B2_result']['status'] != 'B2_FINITE_BIT_NATIVE_CALIBRATION_COMPLETE'
            or set(b['C1_result']['assumed_scenarios']) != set(LANES)):
        raise ValueError('SAVED_STAGE_OR_LANES')
    for arm in ARMS:
        export = prior['final_exports'][arm]
        target = arm+':'+prior['export_hashes'][export]
        if b['B2_result']['arms'][arm]['samples'] != 32:
            raise ValueError('REGISTERED_INDEPENDENT_COHORTS')
        costs = b[arm+'_costs']
        if len(costs) != 26 or {(x['m'], x['axis']) for x in costs} != {(m, a) for m in range(13) for a in ('X', 'Y')}:
            raise ValueError('COST_ROUND_COVERAGE')
        for x in costs:
            if (x['target_id'] != target or x['scope'] != COST_SCOPE
                    or x['kernel_occurrences'] != 4*(1 << x['m'])
                    or x['depth_semantics'] != 'scheduled_serial_sum'
                    or not x['long_round_coverage'] or x['classical_cost_samples'] != 32):
                raise ValueError('SAVED_SERIAL_POLICY_OR_IDENTITY')
        for lane in LANES:
            x = b['C1_result']['assumed_scenarios'][lane][arm]
            if (x['arm'] != arm or x['lane'] != lane or x['target_id'] != target
                    or [r['m'] for r in x['rounds']] != list(range(13))
                    or x['qualification']['kind'] != 'ASSUMED_SCENARIO'
                    or x['qualification']['all_law_bound_qualified'] is not False):
                raise ValueError('SCENARIO_ARM_OR_QUALIFICATION')
            if any(r['target_id'] != target for r in x['rounds']):
                raise ValueError('SAVED_PLAN_TARGET_IDENTITY')
    if any(b['C1_result'][k] is not None for k in
           ('CA_achieved', 'shots_guaranteed', 'total_resources_guaranteed', 'cost_support')):
        raise ValueError('UNAUTHORIZED_QUALIFICATION_PROMOTION')
    return dict(status='SAVED_BINDINGS_PASS', paired_cohorts=False,
                numerical_CA_reassessment=False, new_shot_formula_calls=0)


def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0:
        raise ValueError('FINITE_NONNEGATIVE_SAVED_SCALAR_REQUIRED')
    return x


def attach(b, lane):
    arms = b['C1_result']['assumed_scenarios'][lane]
    if any(not arms[a]['conditional_energy_assessment'].get('same_accuracy') for a in ARMS):
        return dict(status='UNRESOLVED_SAVED_ACCURACY', lane=lane, arms=arms)
    for a in ARMS:
        for x in arms[a]['rounds']:
            if x['status'] != 'CONDITIONAL_ROUND_PASS' or type(x['shots']) is not int or x['shots'] < 1:
                raise ValueError('SAVED_SHOT_PLAN_INVALID_NO_RECOMPUTE')
    return dict(status='SAVED_CONDITIONAL_SCENARIO_ATTACHED', lane=lane, arms=arms,
                assumptions={a: arms[a]['qualification'] for a in ARMS})


def corner(b, packet, shots_arm, price_arm):
    if packet['status'] != 'SAVED_CONDITIONAL_SCENARIO_ATTACHED':
        return dict(status='UNRESOLVED_SAVED_ACCURACY', totals=None, rounds=None,
                    shots_arm=shots_arm, price_arm=price_arm)
    lookup = {(x['m'], x['axis']): x for x in b[price_arm+'_costs']}
    plans = packet['arms'][shots_arm]['rounds']
    rows = []
    for p in plans:
        weighted = {k: math.fsum(p['shots']*number(lookup[p['m'], a]['mean'][k]) for a in ('X', 'Y')) for k in METRICS}
        if any(not math.isfinite(v) for v in weighted.values()):
            raise ValueError('NONFINITE_TOTAL_NO_CLIP')
        rows.append(dict(m=p['m'], shots_per_axis=p['shots'], weighted_cost=weighted))
    totals = {k: math.fsum(x['weighted_cost'][k] for x in rows) for k in METRICS}
    # Own-arm corners can reuse the existing physical-identity-aware aggregator.
    # Crossed corners retain both identities; they must not pretend to be a new
    # physical candidate by rewriting target_id to bypass that old guard.
    if shots_arm == price_arm:
        own = packet['arms'][shots_arm]
        reused = accuracy.aggregate_totals(own['conditional_energy_assessment'], plans,
                                           b[price_arm+'_costs'], cost_scope=COST_SCOPE)
        check_close(totals, reused['totals'])
    return dict(status='DESCRIPTIVE_COUNTERFACTUAL_ONLY', lane=packet['lane'],
        shots_arm=shots_arm, price_arm=price_arm,
        shots_target_id=packet['arms'][shots_arm]['target_id'],
        price_target_id=packet['arms'][price_arm]['target_id'],
        physically_realized_hybrid=False, causal_effect=False, totals=totals, rounds=rows,
        cost_support=None, guaranteed_advantage=None)


def check_close(a, b):
    if b is None or any(not math.isclose(a[k], b[k], rel_tol=RECONCILIATION['rel_tol'],
                                       abs_tol=RECONCILIATION['abs_tol']) for k in METRICS):
        raise ValueError('SAVED_ACCOUNTING_CLOSURE_FAIL')


def differences(v):
    bb, bp, pb, pp = (v[x] for x in ('bb', 'bp', 'pb', 'pp'))
    return dict(net_saving=bb-pp, price_first=bb-bp, shots_after_price=bp-pp,
        shots_first=bb-pb, price_after_shots=pb-pp,
        interaction=pp-bp-pb+bb,
        symmetric_price=.5*((bb-bp)+(pb-pp)),
        symmetric_shots=.5*((bb-pb)+(bp-pp)))


def decompose(corners):
    if any(x['totals'] is None for x in corners.values()):
        return dict(status='UNRESOLVED_SAVED_ACCURACY', attribution=None)
    rows = []
    for m in range(13):
        row = {k: differences({ab: x['rounds'][m]['weighted_cost'][k] for ab, x in corners.items()}) for k in METRICS}
        rows.append(dict(m=m, contribution=row))
    totals = {k: differences({ab: x['totals'][k] for ab, x in corners.items()}) for k in METRICS}
    for k, d in totals.items():
        if not math.isclose(d['symmetric_price']+d['symmetric_shots'], d['net_saving'],
                            rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError('DECOMPOSITION_CLOSURE_FAIL')
    groups = {name: {k: {term: math.fsum(rows[m]['contribution'][k][term] for m in indices)
                               for term in totals[k]} for k in METRICS}
              for name, indices in [('m0_to_9', range(10)), ('m10_to_12', range(10, 13)), ('m12', (12,))]}
    cumulative = [dict(through_m=m, net_saving={k: math.fsum(x['contribution'][k]['net_saving'] for x in rows[:m+1]) for k in METRICS}) for m in range(13)]
    rz = totals['rz_count']
    dominant = ('shots' if rz['symmetric_shots'] > rz['symmetric_price'] else
                'price' if rz['symmetric_price'] > rz['symmetric_shots'] else 'equal')
    return dict(status='DESCRIPTIVE_POINT_DECOMPOSITION', totals=totals, rounds=rows,
        fixed_round_groups=groups, cumulative=cumulative,
        dominant_positive_net_RZ_component=dominant if rz['net_saving'] > 0 else None,
        causal_attribution=False, population_mean_advantage=None, guaranteed_advantage=None)


def uncertainty(b, packet):
    if packet['status'] != 'SAVED_CONDITIONAL_SCENARIO_ATTACHED':
        return dict(status='UNRESOLVED_SAVED_ACCURACY', conditional_SE=None)
    W = {a: sum(2*p['shots']*4*(1 << p['m']) for p in packet['arms'][a]['rounds']) for a in ARMS}
    out = {}
    for k in METRICS:
        eb, ep = (number(b['B2_result']['arms'][a]['sample_mean_standard_error'][k]) for a in ARMS)
        wb, wp = W['baseline'], W['proposed']
        out[k] = dict(net_saving_SE=math.hypot(wb*eb, wp*ep),
            symmetric_price_SE=.5*(wb+wp)*math.hypot(eb, ep),
            symmetric_shots_SE=.5*abs(wb-wp)*math.hypot(eb, ep),
            price_shots_covariance=.25*(wb+wp)*(wb-wp)*(eb*eb-ep*ep))
    return dict(status='DESCRIPTIVE_CONDITIONAL_ON_SAVED_WEIGHTS', W=W, conditional_SE=out,
        samples_per_arm=32, independent_arm_cohorts=True,
        shared_calibration_across_all_rounds=True, independent_round_calibrations=1,
        confidence_interval=None, cost_support=None,
        note='No sqrt(13) gain or 32x shots;price and shots components are correlated;no PF/fit/shot-plan uncertainty included')


def tradeoff(b, arm):
    ref = b['B1_result']['arms'][arm]
    return dict(arm=arm, saved_DF_bound_Ha=ref['saved_DF_bound_Ha'],
        saved_sector=b['A_result']['sector'][arm],
        saved_fit_diagnostic=b['A_result']['fit_diagnostics'][arm],
        rounds=[dict(m=x['m'], saved_reference=x['reference'],
            saved_shots={lane: b['C1_result']['assumed_scenarios'][lane][arm]['rounds'][x['m']]['shots'] for lane in LANES}) for x in ref['rounds']],
        meaning='saved diagnostics copied,not new PF/RTE/DF/shot-formula evaluation',
        attenuation_is_not_raw_signal_modulus=True, DF_Ha_not_added_to_signal_error=True)


def reconcile(b, packets, corners):
    for lane in LANES:
        if packets[lane]['status'] != 'SAVED_CONDITIONAL_SCENARIO_ATTACHED':
            continue
        for arm, key in [('baseline', 'bb'), ('proposed', 'pp')]:
            check_close(corners[lane][key]['totals'], packets[lane]['arms'][arm]['resources']['totals'])
            u = packets[lane]['arms'][arm]['calibration_uncertainty']
            w = sum(2*p['shots']*4*(1 << p['m']) for p in packets[lane]['arms'][arm]['rounds'])
            if w != u['kernel_reuse_weight']:
                raise ValueError('SAVED_REUSE_WEIGHT_MISMATCH')
    return dict(status='SAVED_ENDPOINTS_AND_WEIGHTS_RECONCILED', new_shot_solver_calls=0)


def execute(root, c, output, call):
    out = Path(output)
    b = call('saved_bundle_load', 'saved scalar JSON only', lambda: load_bundle(root, c))
    io.save(out/'identity.json', call('identity_gate', 'C1/B1/B2 bindings', lambda: identity_gate(b, c)))
    packets, corners, analyses = {}, {}, {}
    for lane in LANES:
        packet = call('scenario_attachment', lane, lambda lane=lane: attach(b, lane))
        packets[lane] = packet
        values = {}
        for sa in ARMS:
            for pa in ARMS:
                key = sa[0]+pa[0]
                values[key] = call('counterfactual_corner', lane+':'+key,
                    lambda sa=sa, pa=pa: corner(b, packet, sa, pa))
                io.save(out/(lane+'_'+key+'.json'), values[key])
        corners[lane] = values
        d = call('descriptive_decomposition', lane, lambda: decompose(values))
        u = call('calibration_uncertainty', lane, lambda: uncertainty(b, packet))
        analyses[lane] = dict(decomposition=d, calibration_uncertainty=u)
        io.save(out/(lane+'_analysis.json'), analyses[lane])
    for arm in ARMS:
        io.save(out/(arm+'_tradeoff.json'), call('tradeoff_table', arm, lambda arm=arm: tradeoff(b, arm)))
    io.save(out/'reconciliation.json', call('reconciliation', 'saved self corners and W', lambda: reconcile(b, packets, corners)))
    def summary():
        return dict(status='M1_DESCRIPTIVE_SAVED_ANALYSIS_COMPLETE', scope=SCOPE, analyses=analyses,
            evidence_lane=b['C1_result']['evidence_lane'],
            CA_achieved=None, shots_guaranteed=None, whole_law_bound=None, cost_support=None,
            population_mean_advantage=None, guaranteed_resource_superiority=None,
            causal_attribution=False, strong_deterministic_baseline_ranking=None,
            new_fit=0, new_sampling=0, new_shot_formula_calls=0, build_compile=0,
            mandatory_stop=True)
    result = call('result_summary', 'scope and null qualifications', summary)
    io.save(out/'result.json', result)
    return result


def run(root, contract_path):
    root = Path(root).resolve()
    c = preflight(root, contract_path)
    out = root/FRESH
    out.mkdir(parents=True, exist_ok=False)
    call = Calls(out, LIMITS, MAX_CALLS)
    start = time.monotonic()
    try:
        io.save(out/'contract.json', c)
        io.save(out/'SOURCE.json', dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()))
        return execute(root, c, out, call)
    except BaseException as error:
        call.stop.set()
        io.save(out/'failure.json', dict(error=str(error), first_failure=call.first_failure))
        raise
    finally:
        io.save(out/'STOP.json', dict(failed=call.stop.is_set(), scientific_units=sum(call.counts.values()),
            call_counts=dict(call.counts), first_failure=call.first_failure, retry=0, resume=0, mandatory_stop=True))
        io.save(out/'resources.json', dict(wall_seconds=time.monotonic()-start,
            parent_peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
