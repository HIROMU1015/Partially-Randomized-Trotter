"""Metadata-only PF domain alignment and a callback-only one-shot harness.

This module never imports a scientific provider, loads NPZ, builds circuits,
evaluates signals/shots, or ranks molecular costs. Saved labels are transcribed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

FORMULAS = ('2nd', '4th', '4th(m5_best)')
PIECES = dict(zip(FORMULAS, (1, 3, 11)))
RANKS = {'H4': 10, 'H6': 19}
QS = (1, 2, 4, 8)
EPS = ('0.05', '0.01', '0.005', '0.001', '0.0005')
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'
NATIVE = 'M5_TOL_ONLY_DF_EXPLICIT_GAUSSIAN_DIRECTIONAL_SCALAR_ONCE_V1'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def normalized(system, row, *, rank=None):
    rank = RANKS[system] if rank is None else rank
    mode, prefix, q, formula = row['method'], int(row['prefix']), int(row['q']), row['formula']
    require(formula in FORMULAS and q in (1, 2, 4, 8, 16), 'REGISTERED_FORMULA_Q')
    mode = 'FULL' if mode == 'B1' else mode
    require(mode in ('FULL', 'B0', 'B0I'), 'DETERMINISTIC_ONLY')
    require((mode == 'FULL' and prefix == rank) or
            (mode != 'FULL' and 0 <= prefix < rank), 'NO_FULL_RANK_TRUNCATED_ALIAS')
    return dict(system=system, method=mode, prefix=prefix, q=q, formula=formula)


def key(row):
    return tuple(row[k] for k in ('system', 'method', 'prefix', 'q', 'formula'))


def identifier(row):
    return '_'.join(map(str, key(row)))


def common_domain(registrations):
    """Union *current input series* prefixes across formulas, then a common grid.

    FULL has rank as its prefix and is registered once, rather than once per cut.
    Existing q16 rows remain in the inventory, outside this new common q<=8 grid.
    """
    out = []
    for system in RANKS:
        prefixes = sorted({r['prefix'] for r in registrations if r['system'] == system
                           and r.get('input_comparable') is True and r['method'] != 'FULL'})
        require(prefixes and all(1 <= p < RANKS[system] for p in prefixes), 'CURRENT_PREFIX_UNION')
        for formula in FORMULAS:
            for q in QS:
                out.append(normalized(system, dict(method='FULL', prefix=RANKS[system], q=q, formula=formula)))
                for p in prefixes:
                    for mode in ('B0', 'B0I'):
                        out.append(normalized(system, dict(method=mode, prefix=p, q=q, formula=formula)))
    require(len({key(r) for r in out}) == len(out), 'UNIQUE_COMMON_DOMAIN')
    return out


def status_summary(rows):
    """A saved-label join only; no headroom or scientific arithmetic."""
    if not rows:
        return 'ELIGIBILITY_UNCHECKED'
    values = list(rows.values())
    if 'ELIGIBLE' in values:
        return 'SAVED_ELIGIBLE_AT_SOME_ENDPOINT_POLICY_RECHECK_REQUIRED'
    if all(x == 'INELIGIBLE' for x in values):
        return 'SAVED_INELIGIBLE_ALL_ENDPOINTS'
    return 'SAVED_ELIGIBILITY_UNRESOLVED'


def signal_tasks(rows):
    """Reuse deterministic B0 -> B0I scalar overlay, without equal-cost aliasing."""
    lookup = {key(r): r for r in rows}
    tasks = []
    for row in rows:
        if row['signal_status'] != 'MISSING_SIGNAL':
            continue
        require(row['method'] != 'FULL', 'UNEXPECTED_MISSING_FULL_REFERENCE_SIGNAL')
        base = dict(row, method='B0')
        base_row = lookup[key(base)]
        task = {k: row[k] for k in ('system', 'method', 'prefix', 'q', 'formula')}
        task.update(id=identifier(task), kind='B0_SIGNAL' if row['method'] == 'B0' else 'B0I_ANALYTIC_OVERLAY',
                    depends=[] if row['method'] == 'B0' or base_row['signal_status'] != 'MISSING_SIGNAL'
                    else [identifier(base)])
        tasks.append(task)
    return tasks


def compile_pool(rows):
    """Predeclare a maximum pool, never compile ineligible/unknown candidates.

    Existing labels use a smaller or equal allowance floor than the proposed
    common policy. An all-ineligible row cannot become eligible by increasing u.
    A legacy measured row remains a bridge task until identity evidence passes.
    """
    return [dict(candidate={k: r[k] for k in ('system', 'method', 'prefix', 'q', 'formula')},
                 id=identifier(r), reason='MISSING_PAIRED_NATIVE_COST' if not r['cost_evidence']
                 else 'SAVED_COST_NATIVE_PROFILE_BRIDGE_REQUIRED',
                 execute_only_if='BOTH_AXES_ELIGIBLE_SAME_EPS_AND_NO_ACCEPTED_IDENTITY_MATCHED_PAIR')
            for r in rows if not r['same_native_cost_pair'] and
            r['eligibility_status'] != 'SAVED_INELIGIBLE_ALL_ENDPOINTS']


def compile_caps(tasks):
    n = len(tasks)
    return dict(native_evolution=n,
                native_signed_S2_pieces=sum(PIECES[t['candidate']['formula']]*t['candidate']['q'] for t in tasks),
                native_low_actions=n, native_high_actions=n, independent_occupation_actions=n,
                wrapper_build=2*n, wrapper_compile=2*n, uncompiled_wrapper_actions=2*n,
                compiled_wrapper_actions=2*n, saved_context_restores_max=n,
                preparation_reconstructions_max=n, sampling=0, RTE_events=0,
                new_signal=0, new_reference=0, SCF=0, DF=0, state_solver=0,
                fit=0, shot_evaluation=0, cost_ranking=0)


def plan(rows):
    tasks = signal_tasks(rows)
    pool = compile_pool(rows)
    systems = {}
    for system in RANKS:
        selected = [r for r in rows if r['system'] == system]
        st = [r for r in tasks if r['system'] == system]
        base = [r for r in st if r['kind'] == 'B0_SIGNAL']
        ct = [r for r in pool if r['candidate']['system'] == system]
        systems[system] = dict(rank=RANKS[system],
            prefixes=sorted({r['prefix'] for r in selected if r['method'] != 'FULL'}),
            common_candidates=len(selected), by_formula={f:sum(r['formula'] == f for r in selected) for f in FORMULAS},
            existing_signal_or_fixed_endpoint_candidates=sum(r['signal_status'] != 'MISSING_SIGNAL' for r in selected),
            existing_raw_signal_candidates=sum(r['signal_status'] == 'SAVED_RAW_SIGNAL' for r in selected),
            saved_same_native_paired_cost_candidates=sum(r['same_native_cost_pair'] for r in selected),
            saved_any_direct_paired_cost_candidates=sum(r['any_direct_cost_pair'] for r in selected),
            missing_signal_candidates=len(st), base_signal_calls=len(base), analytic_overlay_calls=len(st)-len(base),
            independent_forward_signal_calls=len(base),
            new_signal_signed_S2_pieces=sum(PIECES[r['formula']]*r['q'] for r in base),
            signal_preparation_prefixes=sorted({r['prefix'] for r in base}),
            future_paired_shot_evaluations_max=5*len(selected),
            compile_candidate_pool_max=len(ct), compile_axes_max=2*len(ct), compile_caps=compile_caps(ct))
        target = systems[system]
        bridge = [r for r in selected if identifier(r) in {t['id'] for t in ct} and r['any_direct_cost_pair']]
        target['compile_pool_existing_direct_bridge_candidates'] = len(bridge)
        target['compile_pool_without_any_direct_cost_pair'] = len(ct)-len(bridge)
        target['signal_context_caps'] = dict(saved_snapshot_restores=1, saved_reference_reads=1,
            full_preparation=1, prefix_preparations=len(target['signal_preparation_prefixes']),
            occupation_oracle_construction=1, new_full_H_reference=0, tail_spectrum=0,
            SCF=0, DF=0, state_solver=0, sampling=0, circuit_build=0, compile=0)
        # A spectral signal and an independent forward occupation action for
        # each B0 task; B0I is a recorded scalar overlay, not a cost alias.
        target['signal_kernel_caps'] = dict(base_signal=len(base), independent_forward_signal=len(base),
            analytic_B0I_overlay=len(st)-len(base), signed_S2_piece_steps=target['new_signal_signed_S2_pieces'],
            spectral_block_actions=sum(2*(r['prefix']+1)*PIECES[r['formula']]*r['q'] for r in base),
            independent_forward_block_actions=sum(2*(r['prefix']+1)*PIECES[r['formula']]*r['q'] for r in base))
    return dict(schema='PF_COMMON_DOMAIN_PREPARATION_V1', execution_authorized=False,
        status='PREPARED_REQUIRES_SEPARATE_EXPLICIT_SCIENTIFIC_APPROVAL',
        systems=systems, T=.8, P=0, q=list(QS), delta=[.8/q for q in QS], epsilons=list(EPS),
        formula_order=list(FORMULAS), compiler_hash=COMPILER, native_policy=NATIVE,
        common_candidates=len(rows), signal_tasks=tasks, compile_candidate_pool=pool,
        selection_rule='Retain every registration; acquire only paired eligible unresolved costs after signal/identity gates; no cost-based selection',
        additional_new_signal_candidates=len(tasks), additional_base_signal_calls=sum(x['base_signal_calls'] for x in systems.values()),
        additional_overlay_calls=sum(x['analytic_overlay_calls'] for x in systems.values()),
        additional_independent_forward_calls=sum(x['independent_forward_signal_calls'] for x in systems.values()),
        compile_candidates_max=len(pool), compile_axes_max=2*len(pool),
        shot_pair_calls_max=5*len(rows), numerical_accounting_or_ranking_authorized=False,
        new_reference=0, SCF=0, DF=0, state_solver=0, PR_sampling=0, top_up=0,
        one_shot=True, automatic_retry=False, resume=False, next_stage_authorized=False,
        grant_valid=False, grant_id=None, absolute_launch_command=None, mandatory_STOP=True)


def comparison_policy():
    return dict(status='PROPOSED_FOR_SEPARATE_EXECUTION_APPROVAL',
        scope='FIXED_T_0.8_COMPLEX_SIGNAL_EXPECTED_RZ_NOT_FULL_RPE_OR_ENERGY',
        primary_bias='USUAL_SIGNED', epsilons=list(EPS), alpha_axis=.025, allowance_factor=1,
        common_allowance=dict(floor=1e-9, safety_factor=10., roundoff_multiplier=64.,
            log_B_margin_floor=1e-12, maximum_u_over_headroom=.01,
            rule='Use frozen h46_m5_signal_contract.eligibility observation formula for all formulas; also max with saved u. No old allowance may be reduced.',
            reference_absolute_error=None, rigorous=False,
            missing_norm_closure_or_stage_metadata='STOP;do not infer observations or regenerate signals outside the frozen budget'),
        shots=dict(source='scripts/d0_frozen_shot_rule_v1.py',
            rule='same binary64 primitive rounding and Decimal integer crosscheck as h246_saved_cost_evaluation.shot_record',
            reuse='only exact same signal/epsilon/bias/u/log_B_margin/alpha/integer-rule binding; otherwise new qualification call within cap',
            missing_old_raw='Recover immutable original metadata first; unresolved candidates prevent claiming a complete common-domain minimum'),
        cost=dict(quantity='N_cos*compiled_RZ_cos+N_sin*compiled_RZ_sin', axes_paired=True,
            native_policy=NATIVE, compiler_hash=COMPILER, state_preparation_P=0,
            reuse='candidate/formula/scalar/control/preparation/provider/runtime identity bridge; same settings hash alone insufficient',
            missing_runtime_closure='STOP reuse; separate new native acquisition only if eligible and inside sealed pool',
            models_allowed_in_primary=False, ineligible_or_unknown_filled_with_zero=False),
        within_formula=dict(PR='saved fixed exploration rows per formula; preserve all original inputs/cohorts/n and SOURCE',
            PF='minimum over the complete common grid restricted to that same formula',
            confirmation='old confirmations keep their original comparison; no new PF comparator is independently confirmed'),
        all_formula=dict(PF='minimum over all three formulas on the complete same grid',
            PR='report fixed PR candidates and saved exploratory choices distinctly; no resampling or new candidate search'),
        completeness='Any unresolved eligible/eligibility candidate => whole relevant minimum UNRESOLVED; never silently drop it',
        ties='exact total RZ, then canonical candidate id',
        q16='H4 existing S2/S4 q16 retained in inventory, excluded from primary q<=8 common grid; no claim of best among every historically available PF',
        noncancellation='secondary missing-component inventory only; no new decomposition, reference or alternate ranking in this proposed primary budget',
        PR_SOURCE_or_results_modified=False, production=False, detailed_scientific_review=False,
        ranking_executed=False, execution_authorized=False, mandatory_STOP=True)


def choose_workers(*, pending, available_cpu_slots, available_memory_bytes,
                   worker_memory_bytes, free_disk_bytes, worker_disk_bytes):
    require(type(pending) is int and pending >= 0, 'PENDING_TASK_COUNT')
    require(available_cpu_slots >= 0 and available_memory_bytes >= 0 and free_disk_bytes >= 0
            and worker_memory_bytes > 0 and worker_disk_bytes > 0, 'MEASURED_RESOURCES')
    return max(0, min(pending, int(available_cpu_slots),
                      int(available_memory_bytes//worker_memory_bytes),
                      int(free_disk_bytes//worker_disk_bytes)))


def paired_eligibility(axis_rows):
    require(set(axis_rows) == set(EPS) and all(set(v) == {'real', 'imag'} for v in axis_rows.values()),
            'EXACT_FIVE_ENDPOINT_TWO_AXIS_LABELS')
    require(all(x in ('ELIGIBLE', 'INELIGIBLE', 'UNDETERMINED')
                for v in axis_rows.values() for x in v.values()), 'KNOWN_ELIGIBILITY_LABELS')
    usable = [e for e, axes in axis_rows.items() if all(v == 'ELIGIBLE' for v in axes.values())]
    unresolved = [e for e, axes in axis_rows.items()
                  if 'INELIGIBLE' not in axes.values() and 'UNDETERMINED' in axes.values()]
    return usable, unresolved


def identity_equal(expected, observed):
    """Value equality, never a profile-name or same-rank shortcut."""
    required = ('hamiltonian', 'state', 'scientific_arrays', 'preparation', 'native_policy',
                'formula_coefficients', 'scalar_control_policy', 'compiler', 'runtime_closure')
    return all(isinstance(expected.get(k), str) and bool(expected[k]) and
               expected[k] == observed.get(k) for k in required)


def select_compile_tasks(pool, qualifications, accepted_identity_pairs):
    require(set(qualifications) == {t['id'] for t in pool}, 'EXACT_PREREGISTERED_QUALIFICATION_DOMAIN')
    require(set(accepted_identity_pairs) <= set(qualifications), 'NO_UNREGISTERED_REUSE')
    require(isinstance(accepted_identity_pairs, dict), 'IDENTITY_RECEIPTS_REQUIRED_NOT_ID_FLAGS')
    for evidence in accepted_identity_pairs.values():
        require(identity_equal(evidence.get('expected', {}), evidence.get('observed', {}))
                and evidence.get('receipt_sha256s')
                and all(isinstance(v, str) and len(v) == 64 and all(x in '0123456789abcdef' for x in v)
                        for v in evidence['receipt_sha256s']), 'FULL_IDENTITY_RECEIPT_VALUES')
    selected, unresolved = [], []
    for task in pool:
        eligible, unknown = paired_eligibility(qualifications[task['id']])
        if unknown:
            unresolved.append(task['id'])
        if eligible and task['id'] not in accepted_identity_pairs:
            selected.append(task)
    return selected, unresolved


class OneShot:
    """Callback-only technical harness; the CLI supplies no molecular backend.

    Approval and fresh, content-bound identity checks precede output/callbacks.
    Marker reservation is exclusive and remains consumed after callback failure.
    """
    def __init__(self, contract, approval, fresh, root):
        require(approval.get('execution_authorized') is True and approval.get('plan_digest') == digest(contract),
                'SEPARATE_EXPLICIT_APPROVAL_REQUIRED')
        require(contract.get('SOURCE') and approval.get('SOURCE') == contract['SOURCE']
                and approval.get('grant_id') and approval.get('stage') == contract['stage'], 'SOURCE_STAGE_GRANT')
        require(contract.get('retry') is False and contract.get('resume') is False, 'NO_RETRY_RESUME')
        for flag in ('source_bytes', 'input_equivalence', 'preparation', 'compiler_runtime', 'resources'):
            require(fresh.get(flag) is True, 'FRESH_GATE:'+flag)
        require(fresh.get('plan_digest') == digest(contract), 'FRESH_PLAN_BINDING')
        observed = fresh.get('monotonic', -float('inf'))
        require(isinstance(observed, (int, float)) and 0 <= time.monotonic()-observed <= 30,
                'FRESH_RECEIPT_EXPIRED')
        self.root = Path(root)
        self.root.mkdir(exist_ok=False)
        self.caps, self.used = dict(contract['caps']), {}
        self._write('CONSUMED.json', dict(grant=approval['grant_id'], nonrefundable=True, retry=False))

    def _write(self, name, value):
        with (self.root/name).open('x') as f:
            json.dump(value, f, sort_keys=True, allow_nan=False)
            f.write('\n')

    def call(self, kind, callback):
        require(not (self.root/'STOP.json').exists(), 'STAGE_STOPPED')
        require(kind in self.caps and self.used.get(kind, 0) < self.caps[kind], 'CALL_BUDGET:'+kind)
        index = self.used.get(kind, 0)
        self._write(f'ledger_{kind}_{index}.json', dict(kind=kind, index=index, entered_nonrefundable=True))
        self.used[kind] = index+1
        try:
            return callback()
        except BaseException as e:
            self._write('STOP.json', dict(reason=type(e).__name__, used=self.used, retry=False))
            raise
