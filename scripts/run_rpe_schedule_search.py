"""Wide PR kappa / split / requested PF width / shot search, resumable.

With search_overrides.requested_widths={PR_PAULI:[...], PR_DF:[...]}, widths
are absolute exact requested values. The existing method mapper supplies each
round's q and actual width. Without that declaration the historical Pauli-only
round entry and saved round05 behavior remain unchanged.

One cycle
  1. plan: rpe_schedule_search.plan_wide over every (L_D, width) family -- schedules (exact kappa intervals), not kappa values;
     saved / evaluated schedules are reused, so no schedule is evaluated twice;
  2. inputs: a family with a prepared base goes through A's preparation entry (configuration), D2 PriceRules MODEL and the
     C/A common calibration-request entry (round04 driver); a schedule whose proven price floor (one shot per round and
     axis) is not below the verified best is pruned without science; a family without a prepared base (new L_D) goes to
     D2's completion API (``Completion.complete`` with the selected L_D, width and absolute kappa); a saved READY record of
     the same design is reused, never re-requested;
  3. signals (A entry, missing rounds only) -> B intake -> route registry (round04 driver);
  4. V2 batch (coarse-first FINE class, HYBRID slots) on all ready schedules; completed Q3 jobs are reused by job key;
  5. the verified-all best (every identity-checked completed job, adopted or not) and the rule-adopted best are both kept.
Stops when the plan proposes nothing, nothing is pending and V2 has no non-small candidate (bases per family in the state).

    python run_rpe_schedule_search.py --problem P --round R --run-dir D [--batch 64] [--plan-only]
``--plan-only`` writes the first decision (proposals, configurations, prices, price floors, completion requests) and stops
before any signal, calibration or Q3 (metadata only).
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction
from functools import wraps
import hashlib
import json
import subprocess
from pathlib import Path
import sys
import time

# The central resume launcher can explicitly keep the original run's source
# checkout for imports, Git locators and providers when selecting a new planner.
ROOT = Path(globals().get('FIXED_RUN_CODE_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402
from trottertracks import rpe_schedule_search as W  # noqa: E402
import run_rpe_kappa_neighbourhood as K  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def read_loc(loc):
    x = json.loads(subprocess.check_output(['git', 'show', loc['commit'] + ':' + loc['path']], cwd=ROOT))
    for k in loc.get('selector', []):
        x = x[k]
    return x


def apply_worker_trial(spec, trial, *, run_dir=None):
    """Install the selected execution policy in memory after fixed-input checks."""
    distributed = spec['executor']['distributed']
    if 'ready_queue_waves' in trial:
        distributed['ready_queue_waves'] = trial['ready_queue_waves']
    if trial.get('pbs_hold_node') is True and distributed.get('pbs'):
        distributed['pbs_hold_node'] = True
    for module_name, field in (('rpe_live_q3_readiness', 'live_readiness_sha256'),
                               ('pf_saved_prices', 'pf_saved_prices_sha256'),
                               ('pf_distributed_preparation', 'pf_distributed_sha256'),
                               ('rpe_scaled_shots', 'scaled_shots_sha256'),
                               ('pf_candidate_selection', 'pf_selection_sha256'),
                               ('rpe_method_optimizer', 'method_optimizer_sha256')):
        if field not in trial:
            continue
        # A resumed run imports other modules from its fixed base release.
        import importlib.util
        path = Path(__file__).resolve().parents[1]/'src/trottertracks'/(module_name+'.py')
        if hashlib.sha256(path.read_bytes()).hexdigest() != trial[field]:
            raise ValueError('Selected scheduler helper differs: '+module_name)
        name = 'trottertracks.'+module_name
        helper = importlib.util.spec_from_file_location(name, path)
        # The executor and CLI may already hold the base optimizer module.
        # Update that module before starting selectors so both use the handoff.
        module = O if module_name == 'rpe_method_optimizer' else importlib.util.module_from_spec(helper)
        helper.loader.exec_module(module)
        sys.modules[name] = module
        # The optimizer also imports from the package directly.
        setattr(sys.modules['trottertracks'], module_name, module)
    if trial.get('recover_saved_price_rounds'):
        for file, key in (('run_pf_preparation_stage.py', 'pf_stage_sha256'),
                          ('check_pf_preparation_owner.py', 'pf_owner_sha256')):
            path = Path(__file__).resolve().with_name(file)
            if hashlib.sha256(path.read_bytes()).hexdigest() != trial[key]:
                raise ValueError('Selected PF metadata entry differs: '+file)
        if isinstance(spec.get('pf_completion'), dict):
            spec['pf_completion']['recover_saved_price_rounds'] = True
    for host, workers in trial.get('calibration_workers', {}).items():
        profile = (distributed['local_worker'] if host == 'local' else
                   json.loads(Path(distributed['preparation_profiles'][host]).read_text()))
        for rule in profile['preparation_profiles']:
            if rule['provider'] == 'DF_CALIBRATION' and rule['profile'].get('task_cpu_count'):
                rule['profile']['task_cpu_count'] = workers
        if host == 'local':
            profile['slots'] = 2
            profile['slot_policy'] = dict(mode='fixed', initial_slots=2)
        distributed['preparation_profiles'][host] = profile
    if 'server2_profile' in trial:
        for worker in distributed['relay']['workers']:
            if worker['host_id'] == 'server2':
                worker['profile'] = trial['server2_profile']
    if 'server1_worker' in trial:
        worker = trial['server1_worker']
        workers = distributed['relay']['workers']
        previous = [entry for entry in workers if entry['host_id'] == 'server1']
        if (worker['host_id'] != 'server1' or len(previous) != 1
                or previous[0].get('enabled') is not False):
            raise ValueError('Expected one disabled server1 worker in the fixed relay configuration')
        workers[workers.index(previous[0])] = deepcopy(worker)
        distributed['preparation_profiles']['server1'] = trial['server1_preparation_profile']
        # The ordinary Start copies this mutable file into the run. The
        # release's admission template is covered by fixed_settings.
        admission = Path(distributed['host_admission']).resolve()
        if run_dir is None or admission != Path(run_dir).resolve() / 'HOST_ADMISSION.json':
            raise ValueError('Expected the run-owned HOST_ADMISSION.json')
        policy = O.load_json(admission)
        if policy['hosts']['server1'].get('reason') == 'USER_REPORTED_STOPPED':
            from trottertracks.rpe_q3_coordinator import atomic_json
            policy['hosts']['server1'] = dict(accepting=True, optional=True, reason='SERVER1_QG1_VALIDATED')
            atomic_json(admission, policy)
    if 'local_memory_reserve_bytes' in trial:
        distributed['local_calibration_pair'] = dict(
            memory_reserve_bytes=trial['local_memory_reserve_bytes'],
            pf_preparation=trial.get('local_pf_preparation_pair', False))
    if 'preparation_host_preference' in trial:
        distributed['preparation_host_preference'] = trial['preparation_host_preference']
    if 'pf_compile_hosts' in trial:
        hosts = trial['pf_compile_hosts']
        configured = distributed['preparation_profiles']
        if not hosts or not set(hosts) <= configured.keys():
            raise ValueError('PF compile placement requires configured preparation profiles')
        placement = distributed.setdefault('preparation_provider_hosts', {})
        legacy = trial.get('pf_legacy_compile_hosts')
        if legacy is not None:
            if not set(legacy) <= set(hosts):
                raise ValueError('Legacy PF compile hosts must belong to the selected placement')
            distributed['pf_legacy_compile_hosts'] = list(legacy)
            helper = Path(__file__).resolve().parents[1]/'src/trottertracks/rpe_preparation_paths.py'
            if hashlib.sha256(helper.read_bytes()).hexdigest() != trial['pf_native_paths_sha256']:
                raise ValueError('Selected PF native path resolver differs')
        placement.update(PF_COMPILE=list(hosts), PF_HOLDOUT_COMPILE=list(hosts))
    if trial.get('pbs_storage_launcher'):
        distributed['pbs_gateway_storage_launcher'] = trial['pbs_storage_launcher']
    if 'pbs_collection_workers' in trial:
        distributed['pbs_collection_workers'] = trial['pbs_collection_workers']
    if trial.get('pbs_dispatch_gateway'):
        distributed['pbs_dispatch_gateway'] = trial['pbs_dispatch_gateway']
    if trial.get('controller_queue_first'):
        distributed['controller_queue_first'] = True
        pf = spec.get('pf_completion') or {}
        if pf.get('mode') == 'complete' and not pf.get('selected'):
            # The approved preparation concurrency can feed distinct PF
            # families as well as the PR lanes. Keep explicit PF limits.
            pf.setdefault('max_candidates', spec.get('candidate_completion', {}).get('parallel', 1))
    if trial.get('pbs_only'):
        distributed['execution_hosts'] = [worker['host_id'] for worker in distributed['pbs']['workers']]
    if trial.get('pbs_preparation_resources'):
        from trottertracks.rpe_pbs import apply_preparation_resources
        apply_preparation_resources(distributed, trial['pbs_preparation_resources'])


@contextmanager
def serialize_legacy_ast():
    """Keep parallel controller metadata reads safe on the pinned CPython."""
    if sys.implementation.name != 'cpython' or not (3, 11, 0) <= sys.version_info[:3] < (3, 11, 8):
        yield
        return
    import ast
    from threading import RLock

    # CPython gh-106905: these releases share AST-conversion recursion
    # state. Serialize just AST parsing, leaving all worker jobs parallel.
    parse = ast.parse
    lock = RLock()

    @wraps(parse)
    def parse_serially(*args, **kwargs):
        with lock:
            return parse(*args, **kwargs)

    ast.parse = parse_serially
    try:
        yield
    finally:
        ast.parse = parse


class WideDriver(K.Driver):
    def __init__(self, a):
        existing = (a.run_dir.resolve() / 'KAPPA_STATE.json').exists()
        super().__init__(a)
        if not existing and self.spec.get('continues'):
            previous = ROOT / self.spec['continues']['run_dir'] / 'KAPPA_STATE.json'
            carried = O.load_json(previous)
            for key in ('points', 'family_constants', 'saved_packets', 'splits'):
                if key in carried:self.state[key] = deepcopy(carried[key])
            self.state['continuation'] = dict(directory=str(previous.parent),
                previous_round=carried.get('round_id'), previous_stop=carried.get('stop'))
        # These requests were never executed; actual scientific HELD/STOPs stay held.
        for p in self.state['points']:
            if p['status'] == 'HELD' and p.get('reason') == 'COMPLETION_REQUIRED_NOT_RUN_IN_THIS_INVOCATION':
                p.update(status='COMPLETION', unrun_previous_reason=p.pop('reason'))
        self.wide = self.spec['search_overrides']['schedule_wide']
        if 'seeded' not in self.state:
            self.seed()

    # ------------------------------------------------------------ state
    def seed(self):
        """Points from the ledger (every configuration with verified saved Q3, incl. saved multiplier configurations) and the
        carried kappa state; family constants of prepared and new splits (exact, from canonical configurations)."""
        led = O.load_json(ROOT / self.spec['ledger_summary'])
        have = {(p['L_D'], Fraction(p['kappa'])) for p in self.state['points']}
        for cid, d in led['by_configuration'].items():
            if d.get('kappa') is None:
                continue
            k = O.frac(d['kappa']);ld = d['L_D']
            if (ld, k) not in have:
                self.state['points'].append(dict(L_D=ld, kappa=str(k), status='EVALUATED', configuration_id=cid, requested_cycle=0,
                                                 request_reason='SAVED_EVALUATED_SCHEDULE'));have.add((ld, k))
        for p in self.state['points']:
            d = led['by_configuration'].get(p.get('alias_of') or p.get('configuration_id') or '')
            if d:
                p['verified_all_best'] = d.get('verified_all');p['rule_adopted_best'] = d.get('rule_adopted')
        fc = self.state.setdefault('family_constants', {})
        for ld, loc in (self.spec.get('new_families') or {}).items():
            cfg = read_loc(loc['configuration']);cfg = cfg.get('configuration', cfg)
            rows = KS.constants(cfg)
            if KS.R_vector(rows, KS._frac(cfg['spec']['kappa']['value'])) != [r['R_outer'] for r in cfg['rounds']]:
                raise ValueError('MAPPER_DOES_NOT_REPRODUCE_NEW_SPLIT_' + str(ld))
            fc.setdefault(str(ld), KS.encode_rows(rows))
            self.state.setdefault('new_split_configurations', {})[str(ld)] = dict(loc['configuration'], configuration_id=cfg['id'])
        for p in self.state['points']:  # a new split is not 'waiting for P0' any more: D2's completion prepares it when selected
            if str(p['L_D']) in (self.spec.get('new_families') or {}) and p['status'] == 'HELD' and str(p.get('reason', '')).startswith('SPLIT_PREPARATION_REQUIRED'):
                p.update(status='COMPLETION', reopened=dict(previous=p.pop('reason'), by='D2_COMPLETION_API'))
        self.state['seeded'] = dict(UTC=K.now(), ledger=self.spec['ledger_summary'], new_families=sorted(self.spec.get('new_families') or {}))
        self.save()

    def families(self):
        return {int(k): W.Family(int(k), rows) for k, rows in self.state.get('family_constants', {}).items()}

    def wide_points(self):
        fams = self.families();out = []
        for p in self.state['points']:
            if p['L_D'] not in fams:
                continue
            best = [x for x in (p.get('best'), p.get('verified_all_best'), p.get('run_completed_best')) if x is not None]
            best = min(O.frac(x) for x in best) if best else None
            st = p['status']
            status = ('FEASIBLE' if best is not None else
                      'NO_FEASIBLE_FOUND' if st in ('EVALUATED', 'ALIAS') and not p.get('promising') else
                      'UNAVAILABLE' if st in ('HELD', 'PRUNED') else 'IN_PROGRESS' if st in ('EVALUATED', 'ALIAS') else 'PENDING')
            out.append(dict(family=p['L_D'], kappa=Fraction(p['kappa']), status=status, best=best, min_lower_over_tau=p.get('min_lower_over_tau')))
        return fams, out

    # ------------------------------------------------------------ plan
    def plan(self):
        fams, pts = self.wide_points()
        floors = sorted((Fraction(p['kappa']), p['L_D'], O.frac(p['price_floor'])) for p in self.state['points'] if p.get('price_floor') is not None)

        def price_floor(ld, R):  # prices grow with R: the floor of a quoted smaller-or-equal schedule bounds this one from below
            f = fams[ld];lb = [fl for k, l, fl in floors if l == ld and all(a <= b for a, b in zip(f.R(k), R))]
            return max(lb) if lb else None
        proposals, decisions, report = W.plan_wide(fams, pts, self.wide, price_floor)
        for q in proposals:
            if self.point(q['family'], q['kappa']) is None:
                new_split = str(q['family']) in (self.state.get('new_split_configurations') or {}) and not any(
                    p['L_D'] == q['family'] and p.get('configuration_id') for p in self.state['points'])
                self.state['points'].append(dict(L_D=q['family'], kappa=str(q['kappa']), status='COMPLETION' if new_split else 'PENDING',
                                                 requested_cycle=self.state['cycle'], request_reason=q['reason'],
                                                 request_evidence={k: v for k, v in q.items() if k not in ('family', 'kappa')}))
        self.state['history'].append(dict(cycle=self.state['cycle'], UTC=K.now(), rule=W.RULE, decisions=decisions, report=report,
                                          proposals=[dict(L_D=q['family'], kappa=str(q['kappa']), reason=q['reason']) for q in proposals]))
        return proposals, report

    # ------------------------------------------------------------ inputs
    def prune(self, ctx):
        """Proven price floor: an allocation of one shot per round and axis is the cheapest of the schedule."""
        fams, pts = self.wide_points();best = min((p['best'] for p in pts if p['status'] == 'FEASIBLE'), default=None)
        by = {c['configuration_id']: c for c in ctx['candidates']}
        for p in self.state['points']:
            c = by.get(p.get('configuration_id'))
            if c is None or c.get('prices') is None:
                continue
            p['price_floor'] = sum(c['prices'])
            if p['status'] == 'PREPARED' and best is not None and p['price_floor'] >= best:
                p.update(status='PRUNED', reason='PROVEN_COST_FLOOR:%s>=%s' % (O.show(p['price_floor']), O.show(best)))

    def completion_requests(self, on_wait=None):
        """New splits: D2 completion (READY -> search, HELD -> recorded).  --plan-only records the request only."""
        todo = [p for p in self.state['points'] if p['status'] == 'COMPLETION']
        if not todo:
            return
        cc = self.spec.get('candidate_completion');imported = {}
        if cc and cc.get('resume_preparation_from'):
            for r in read_loc(cc['resume_preparation_from'])['records'].values():
                if r['status'] == 'READY_FOR_SEARCH' and r.get('design') and r.get('method') == 'PR_PAULI':  # never a PR-DF record
                    d = r['design'];imported[(d['L_D'], O.frac(d['kappa']), O.frac(d['delta_pf']))] = r
        run = []
        for p in todo:
            design = dict(L_D=p['L_D'], formula='2nd', delta_over_h='1', kappa=str(p['kappa']), K=2)
            p['completion_request'] = dict(method='PR_PAULI', selection=dict(rule=W.RULE, reason=p.get('request_reason'), cycle=p['requested_cycle']),
                                           design=design)
            saved = next((r for (ld, k, dl), r in imported.items() if ld == p['L_D'] and k == Fraction(p['kappa'])
                          and dl == Fraction(0.1)), None)  # same split, absolute kappa and width (delta_PF = h = binary64 0.1)
            if saved is not None:
                p.update(status='REGISTERED', configuration_id=saved['configuration_id'], completion=dict(status='REUSED_SAVED_READY_RECORD',
                                                                                                         record=saved['key']));continue
            if self.a.plan_only or not cc or self.spec.get('science_approved') is not True:
                p.update(status='HELD', reason='COMPLETION_REQUIRED_NOT_RUN_IN_THIS_INVOCATION');continue
            run.append(p)
        if not run:
            return
        comp = self.completion_api(dict(cc, max_attempted_per_method=0))
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max(1, min(len(run), int(cc.get('parallel', 4))))) as pool:  # independent splits / schedules
            futures = [pool.submit(comp.complete, p['completion_request']) for p in run]
            self.log(event='COMPLETION_WAIT_START', requests=len(run))
            if on_wait is not None:on_wait()
            for p, future in zip(run, futures):
                rec = future.result()
                if rec['status'] == 'READY_FOR_SEARCH':
                    p.update(status='REGISTERED', configuration_id=rec['configuration_id'], completion=dict(status=rec['status'], key=rec['key']))
                else:  # HELD: the concrete reason is kept and the other candidates go on
                    p.update(status='HELD', reason='COMPLETION:%s:%s' % (rec['status'], str(rec.get('reason'))[:300]),
                             completion=dict(status=rec['status'], key=rec['key']))
            self.log(event='COMPLETION_WAIT_END', requests=len(run))

    def completion_api(self, settings):
        from trottertracks.pr_candidate_completion import Completion
        return Completion(self.run / 'completion', settings)

    def write_points(self):
        """A's preparation entry gets only schedules of splits with a prepared base (new splits go through D2 completion)."""
        rows = [dict(L_D=p['L_D'], kappa=p['kappa']) for p in self.state['points']
                if p['status'] not in ('HELD', 'PRUNED', 'COMPLETION') and not p.get('completion')]
        self.points_file.parent.mkdir(parents=True, exist_ok=True);self.points_file.write_text(json.dumps(rows, indent=1) + '\n')

    def absorb(self, res):
        """Rule-adopted bests (V2) plus the cheapest feasible completed job of this run per configuration, adopted or not
        (provisional for planning; the ledger re-verifies every job with the executor's identity checks before reporting)."""
        super().absorb(res)
        done = {}
        for q in (self.run / 'jobs' / 'PR_PAULI').glob('*/REQUEST.json') if (self.run / 'jobs' / 'PR_PAULI').exists() else []:
            job = q.parent
            if not (job / 'OUTCOME.json').exists():
                continue
            out, _ = O.job_outcome(job);r = json.loads(q.read_text())
            if out.get('status') == O.PASS:
                c = O.frac(r['cost']);cid = r['configuration_id']
                done[cid] = min(done.get(cid, c), c)
        for p in self.state['points']:
            cid = p.get('alias_of') if p['status'] == 'ALIAS' else p.get('configuration_id')
            if cid in done:
                p['run_completed_best'] = done[cid]

    # ------------------------------------------------------------ loop
    def batch(self, ready_only=False):
        spec = json.loads(json.dumps(self.spec))
        if ready_only:
            # Completion owns its state/registry while working. This interim batch
            # uses only the already published points and route registry snapshot.
            spec.pop('candidate_completion', None)
        if spec.get('candidate_completion'):
            spec['candidate_completion']['max_attempted_per_method'] = 0  # install READY records only; B2 chooses what to complete
        self.spec, keep = spec, self.spec
        try:
            return super().batch()
        finally:
            self.spec = keep

    def loop(self):
        previous_stop = self.state.get('stop', {}).get('reason')
        transient = {'USER_REQUESTED_STOP', 'PLAN_ONLY_FIRST_DECISION_WRITTEN', 'SCIENCE_NOT_APPROVED_FIRST_DECISION_WRITTEN',
                     'PLAN_ONLY_NOTHING_TO_PREPARE', 'MAX_CYCLES'}
        if previous_stop and previous_stop not in transient:
            self.log(event='PRESERVED_STOP', reason=previous_stop);return
        if previous_stop:self.state.pop('stop')
        while self.state['cycle'] < self.a.max_cycles:
            if (self.run/'STOP_REQUESTED.json').exists():
                self.state['stop']=dict(reason='USER_REQUESTED_STOP',UTC=K.now());self.save();return
            self.state['cycle'] += 1;self.save()
            proposals, report = self.plan();self.save()
            interim = []
            def ready_work():
                if self.spec.get('overlap_ready_batch') and any(p['status'] in ('REGISTERED', 'EVALUATED', 'ALIAS') for p in self.state['points']):
                    self.log(event='READY_BATCH_DURING_COMPLETION_START')
                    code, res = self.batch(ready_only=True)
                    if res is None:
                        interim.append(dict(stop_reason='INVOCATION_FAILED', returncode=code))
                        self.log(event='READY_BATCH_DURING_COMPLETION_END', returncode=code, stop='INVOCATION_FAILED')
                        return
                    interim.append(res)
                    self.absorb(res);self.save()
                    self.log(event='READY_BATCH_DURING_COMPLETION_END', stop=res['stop_reason'], new_calls=res['new_calls'])
            self.completion_requests(on_wait=ready_work);self.save()
            if interim and interim[-1]['stop_reason'] == 'INVOCATION_FAILED':
                self.state['stop'] = dict(reason='V2_INVOCATION_FAILED', returncode=interim[-1]['returncode'], UTC=K.now())
                self.save();return
            fatal = {'UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER', 'HOST_ELIGIBILITY_HOLD',
                     'Q3_BUDGET_EXHAUSTED', 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET'}
            if interim and interim[-1]['stop_reason'] in fatal:
                self.state['stop'] = dict(reason='V2_' + interim[-1]['stop_reason'], UTC=K.now())
                self.save();return
            stop_here = self.a.plan_only or self.spec.get('science_approved') is not True
            if any(p['status'] in ('PENDING', 'PREPARED') for p in self.state['points']):
                ctx = self.prepare();self.prune(ctx);self.save()
                if stop_here:
                    self.state['stop'] = dict(reason='PLAN_ONLY_FIRST_DECISION_WRITTEN' if self.a.plan_only else
                                              'SCIENCE_NOT_APPROVED_FIRST_DECISION_WRITTEN', UTC=K.now());self.save();return
                self.signals();self.save()
            elif stop_here:
                self.state['stop'] = dict(reason='PLAN_ONLY_NOTHING_TO_PREPARE', UTC=K.now());self.save();return
            code, res = self.batch()
            if res is None:
                self.state['stop'] = dict(reason='V2_INVOCATION_FAILED', returncode=code, UTC=K.now());self.save();return
            self.absorb(res);self.save()
            pending = [p for p in self.state['points'] if p['status'] in ('PENDING', 'PREPARED', 'REGISTERED', 'COMPLETION')]
            promising = sum(p.get('promising') or 0 for p in self.state['points'] if p['status'] == 'EVALUATED')
            self.log(event='CYCLE_END', v2_stop=res['stop_reason'], proposals=len(proposals), pending=len(pending), promising=promising,
                     plan_stop=report['stop'])
            if res['stop_reason'] in ('Q3_BUDGET_EXHAUSTED', 'NEW_Q3_REQUIRED_OUTSIDE_APPROVED_BUDGET'):
                self.state['stop'] = dict(reason=res['stop_reason'], UTC=K.now());self.save();return
            if res['stop_reason'] in ('UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER', 'HOST_ELIGIBILITY_HOLD'):
                self.state['stop'] = dict(reason='V2_' + res['stop_reason'], UTC=K.now());self.save();return
            if not proposals and not pending and res['stop_reason'] in ('ONLY_SMALL_OR_DEFERRED_LEFT', 'NO_CANDIDATE_LEFT_IN_RANGE'):
                self.state['stop'] = dict(reason='SCHEDULE_WIDE_RULE_EXHAUSTED', UTC=K.now(), v2_stop=res['stop_reason'],
                                          families=report['families']);self.save();return
        self.state['stop'] = dict(reason='MAX_CYCLES', UTC=K.now());self.save()


class WidthDriver:
    """Declared widths for both PR methods, reusing common completion and V2.

    The original round05 driver above remains the default without an explicit
    requested_widths declaration.  This path never invents q/R from a width:
    family constants come from the method's existing canonical generator.
    """
    work_conserving_search = False

    def __init__(self, a):
        self.a = a; self.run = a.run_dir.resolve(); self.run.mkdir(parents=True, exist_ok=True)
        self.spec = O.load_json(a.round)
        self.problem = K.CLI.price_overrides(K.CLI.scoped(O.load_json(a.problem), self.spec.get('method_overrides')), self.spec)
        self.rule_stop = self.spec.get('search_to_rule_stop',False)
        self.widths = W.requested_widths(self.problem, self.spec)
        if self.widths is None:
            if not self.rule_stop or any(m.startswith('PR_') for m in self.spec['methods']):
                raise ValueError('Explicit requested_widths required')
            self.widths={}
        self.scaled_shots = (self.spec.get('search_overrides', {}).get('rule', self.problem['search'].get('rule'))
                             == 'RPE_SCALED_SHOTS_V1')
        self.methods = list(self.spec['methods']) if self.rule_stop or self.scaled_shots else list(self.widths)
        self.live_ready_refill = self.spec.get('live_ready_refill') is True
        if self.live_ready_refill and (
            self.spec.get('overlap_ready_batch') is not True
            or self.spec.get('executor', {}).get('scheduler') != 'concurrent'
            or self.spec.get('search_overrides', {}).get('rule', self.problem['search'].get('rule'))
                not in ('METHOD_MINIMUM_PRIORITY_V2', 'RPE_SCALED_SHOTS_V1')
        ):
            raise ValueError('live_ready_refill requires overlapping preparation and a concurrent selector')
        self.work_conserving_search = self.spec.get('search_overrides', {}).get('work_conserving_search') is True
        if self.work_conserving_search and not self.live_ready_refill:
            raise ValueError('work_conserving_search requires live_ready_refill')
        for method,widths in self.widths.items():
            self.problem['methods'][method]['requested_delta_pf_values']=[O.enc(KS._frac(w['raw']))for w in widths]
        self.state_file = self.run/'KAPPA_STATE.json'
        self.state = O.load_json(self.state_file) if self.state_file.exists() else dict(
            schema='RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1', round_id=self.spec['round_id'],
            requested_widths=self.widths, cycle=0, points=[], families={}, history=[])
        if self.state.get('schema') != 'RPE_REQUESTED_WIDTH_SCHEDULE_STATE_V1':
            raise ValueError('Preserve the old single-width run; use an independent multi-width directory')
        if self.state['round_id'] != self.spec['round_id'] or self.state['requested_widths'] != self.widths:
            raise ValueError('Run belongs to different requested widths or round')
        self.wide = self.spec.get('search_overrides', {}).get('schedule_wide', {})
        self.completion = None
        self.check_fixed_inputs()
        trial = globals().get('FIXED_RUN_WORKER_TRIAL')
        if trial is not None and not a.plan_only:
            apply_worker_trial(self.spec, trial, run_dir=self.run)
        if not self.state['families']:
            started=time.monotonic();self.log(event='SEED_START')
            self.seed();self.log(event='SEED_END',wall_seconds=time.monotonic()-started,families=len(self.state['families']))

    def log(self, **row):
        with (self.run/'DRIVER_PROGRESS.jsonl').open('a') as f:
            f.write(json.dumps(dict(UTC=K.now(),cycle=self.state['cycle'],**row),default=str)+'\n')

    def save(self):
        tmp=self.state_file.with_suffix('.tmp');tmp.write_text(json.dumps(O.jsonable(self.state),indent=2)+'\n')
        tmp.replace(self.state_file)

    def check_fixed_inputs(self):
        """Keep cached families tied to the same physical/evaluation inputs."""
        completion={k:v for k,v in(self.spec.get('candidate_completion')or{}).items()
                    if k not in('price_evidence_results','plan_only')}
        current=dict(schema='RPE_REQUESTED_WIDTH_FIXED_INPUTS_V1',problem=self.problem,
            requested_methods=self.methods,requested_widths=self.widths,
            round_scope={k:self.spec.get(k)for k in('method_overrides','search_overrides','width_families','new_families','kappa_families')},
            completion=completion)
        preparation=(self.spec.get('executor',{}).get('distributed')or{}).get('preparation')
        if preparation:
            current['preparation_budgets']={stage:preparation.get('budgets',{}).get(stage,0)
                for stage in ('signal','compile','audit')}
        if self.live_ready_refill:
            current['round_scope']['live_ready_refill'] = True
        pf_candidates=(self.spec.get('pf_completion') or self.problem.get('pf_completion') or {}).get('max_candidates',1)
        if pf_candidates!=1:
            current['round_scope']['pf_parallel_candidates']=pf_candidates
        path=self.run/'FIXED_INPUTS.json'
        if path.exists():
            if O.load_json(path)!=current:
                raise ValueError('FIXED_INPUTS_CHANGED: use a separate run for another physical/evaluation/search scope')
            return
        # The first technical multi-width prototype predates this small input
        # record.  Its metadata-only state can be adopted after source checks;
        # an executed state without its original inputs cannot be inferred safe.
        if self.state['families']:
            if (self.state.get('stop',{}).get('reason')!='REQUESTED_WIDTH_METADATA_ONLY'or
                    any(p['status']in('REGISTERED','EVALUATED')for p in self.state['points'])or
                    (self.run/'jobs').exists()):
                raise ValueError('FIXED_INPUTS_MISSING_FOR_EXECUTED_STATE: preserve this run and use a new one')
            self.verify_metadata_inputs()
        path.write_text(json.dumps(current,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

    def verify_metadata_inputs(self):
        reader,_=self.catalog();facts=getattr(self,'catalog_facts',{})
        expected_binding={k:facts.get(v)for k,v in(('hamiltonian','hamiltonian_hash'),('state','state_hash'),('snapshot','snapshot_sha256'))}
        for key,f in self.state['families'].items():
            method=f['method'];setting=self.problem['methods'].get(method)
            if setting is None or method not in self.methods:raise ValueError('METADATA_SCOPE_METHOD_CHANGED:'+key)
            if setting.get('L_D_values')and f['L_D']not in setting['L_D_values']:raise ValueError('METADATA_SCOPE_SPLIT_CHANGED:'+key)
            formulas=(setting.get('generation')or{}).get('formulas',['2nd'])
            if f['formula']not in formulas:raise ValueError('METADATA_SCOPE_FORMULA_CHANGED:'+key)
            if KS._frac(f['requested_delta_pf'])not in{KS._frac(v['raw'])for v in self.widths[method]}:
                raise ValueError('METADATA_SCOPE_WIDTH_CHANGED:'+key)
            if not f.get('configuration'):continue
            cfg=f['configuration'];s=cfg['spec']
            if s['M']!=self.problem['rpe']['M']:raise ValueError('METADATA_SCOPE_M_CHANGED:'+key)
            if self.problem['rpe'].get('h')is not None and KS._frac(s['h'])!=KS._frac(self.problem['rpe']['h']):
                raise ValueError('METADATA_SCOPE_H_CHANGED:'+key)
            for field,value in expected_binding.items():
                if value is not None and s.get('binding',{}).get(field)!=value:raise ValueError('METADATA_SCOPE_BINDING_CHANGED:'+key)
            source=f.get('base_source')or f.get('template_source')
            if source:
                base=reader.read(source);base=base.get('configuration',base)['spec']
                if KS._frac(s['h'])!=KS._frac(base['h'])or s['M']!=base['M']:
                    raise ValueError('METADATA_TIME_GRID_DIFFERS_FROM_FIXED_SOURCE:'+key)
            if f.get('rows')and KS.encode_rows(KS.constants(cfg))!=f['rows']:
                raise ValueError('METADATA_CONSTANTS_DIFFER_FROM_CONFIGURATION:'+key)

    def catalog(self):
        reader, inputs, catalog = self.saved_catalog(self.problem)
        self.catalog_facts=O.hamiltonian_facts(self.problem)
        O.check_comparison(self.problem,self.catalog_facts,inputs['comparison'])
        return reader,catalog

    def saved_catalog(self, problem):
        source=problem['evidence']['saved_catalog_inputs']
        if not hasattr(self,'_saved_catalog') or self._saved_catalog[0]!=source:
            self._saved_catalog=(deepcopy(source),O.load_catalog(ROOT,problem))
        return self._saved_catalog[1]

    def generate_families(self, method, items):
        return O.run_generator(ROOT,self.problem['methods'][method]['generation'],items)['items']

    def seed(self):
        reader,catalog=self.catalog();sources={}
        for point in catalog['candidates']:
            if point['method'] in self.methods and point.get('configuration_source'):
                sources.setdefault((point['method'],point['L_D']),point['configuration_source'])
        for ld,value in self.spec.get('new_families',{}).items():
            sources.setdefault(('PR_PAULI',int(ld)),value['configuration'])
        for value in self.spec.get('width_families',[]):
            sources[(value['method'],int(value['L_D']))]=value['configuration']
        for method, widths in self.widths.items():
            setting=self.problem['methods'][method]
            setting['requested_delta_pf_values']=[O.enc(KS._frac(w['raw'])) for w in widths]
            allowed=setting.get('L_D_values') or self.spec.get('kappa_families')
            if allowed is None and setting.get('L_D_range')and getattr(self,'catalog_facts',None):
                bound=setting['L_D_range']
                allowed=list(range(bound['min'],self.catalog_facts['fragment_count']+bound['max_offset_from_fragment_count']+1))
            bases=[(ld,loc)for (name,ld),loc in sources.items()if name==method and (allowed is None or ld in allowed)]
            families={}
            items=[]
            for ld,loc in bases:
                base=reader.read(loc);base=base.get('configuration',base)
                for formula in (setting.get('generation')or{}).get('formulas',['2nd']):
                    for width in widths:
                        key=W.width_family_key(method,ld,formula,width['raw'],base['spec']['finite_K'])
                        family=dict(method=method,L_D=ld,formula=formula,K=base['spec']['finite_K'],
                            requested_delta_pf=width['raw'],delta_pf=O.enc(KS._frac(width['raw'])),
                            base_source=loc,status='MAPPER_PENDING')
                        self.state['families'][key]=family
                        ratio=KS._frac(width['raw'])/KS._frac(base['spec']['h'])
                        items.append(dict(item_id=key,base=loc,changes=dict(formula=formula,delta_over_h=str(ratio))))
                        families[key]=(family,base['spec'])
            if items:
                started=time.monotonic()
                self.log(event='SEED_MAPPER_START',method=method,families=len(items))
                results=self.generate_families(method,items)
                by_id={r['item_id']:r for r in results}
                if len(by_id)!=len(results) or by_id.keys()!=families.keys():
                    raise ValueError('Width mapper results differ from the requested families')
                self.log(event='SEED_MAPPER_END',method=method,families=len(items),wall_seconds=time.monotonic()-started)
                for key,(family,base) in families.items():
                    made=by_id[key]
                    try:
                        if made['status']!='GENERATED':
                            raise ValueError(made.get('reason')or made['status'])
                        cfg=made['configuration'];s=cfg['spec']
                        if KS._frac(s['delta_pf'])!=KS._frac(family['requested_delta_pf']):
                            raise ValueError('MAPPER_DOES_NOT_PRESERVE_EXACT_REQUESTED_WIDTH')
                        if KS._frac(s['h'])!=KS._frac(base['h']) or s['M']!=base['M']:
                            raise ValueError('MAPPER_CHANGED_FIXED_RPE_TIME_GRID')
                        rows=KS.constants(cfg); k=KS._frac(s['kappa']['value'])
                        if KS.R_vector(rows,k)!=[v['R_outer']for v in sorted(cfg['rounds'],key=lambda r:r['m'])]:
                            raise ValueError('SCHEDULE_CONSTANTS_DO_NOT_REPRODUCE_CANONICAL_MAPPER')
                        family.update(status='READY_METADATA',rows=KS.encode_rows(rows),configuration=cfg)
                    except (ValueError,KeyError,TypeError) as exc:
                        family.update(status='UNSUPPORTED_MAPPER',reason=str(exc))
                    self.log(event='SEED_FAMILY_END',family=key,status=family['status'],reason=family.get('reason'))
            # Bootstrap a missing split through D2's ordinary preparation entry.
            # A template supplies only the fixed time grid / binding and a
            # metadata kappa.  The target split supplies its own descriptor and
            # lambda before W chooses any scientific schedule.
            templates=[loc for (name,_),loc in sources.items()if name==method]
            for ld in allowed or []:
                if not any(v['method']==method and v['L_D']==ld for v in self.state['families'].values()):
                    for formula in(setting.get('generation')or{}).get('formulas',['2nd']):
                        for width in widths:
                            key=W.width_family_key(method,ld,formula,width['raw'])
                            f=dict(method=method,L_D=ld,formula=formula,K=2,
                                requested_delta_pf=width['raw'],delta_pf=O.enc(KS._frac(width['raw'])),
                                status='NEEDS_FAMILY_CONFIGURATION_SOURCE',
                                reason='No same-method saved template: fixed H/state/snapshot/h/M and metadata kappa are required')
                            self.state['families'][key]=f
                            if templates:
                                source=templates[0];template=reader.read(source);template=template.get('configuration',template)
                                f.update(status='FAMILY_PREPARATION_PENDING',template_source=source,template=template,
                                    preparation_request=dict(method=method,template_source=source,
                                        selection=dict(stage='FAMILY_CONSTANTS_METADATA_ONLY',family=key),
                                        design=dict(L_D=ld,formula=formula,K=template['spec']['finite_K'],delta_pf=width['raw'],
                                            kappa=str(KS._frac(template['spec']['kappa']['value'])))))
        # Only saved evaluations with this exact method/formula/requested width
        # seed a family; kappa boundaries and Q3 never cross widths implicitly.
        for p in catalog['candidates']:
            if p['method']not in self.methods or p.get('kappa')is None or p.get('delta_pf')is None:continue
            family=W.width_family_key(p['method'],p['L_D'],p['formula'],p['delta_pf'],p.get('finite_K')or 2)
            if self.state['families'].get(family,{}).get('status')!='READY_METADATA':continue
            k=KS._frac((p['kappa'].get('value')if isinstance(p['kappa'],dict)and'value'in p['kappa']else p['kappa']))
            if not W.kappa_allowed(self.wide_for(p['method']), k):
                continue
            same=next((v for v in self.state['points']if v['family']==family and Fraction(v['kappa'])==k),None)
            if same is None:
                same=dict(family=family,method=p['method'],L_D=p['L_D'],kappa=str(k),status='EVALUATED',
                    configuration_id=p['configuration_id'],best=None,saved_point_ids=[])
                self.state['points'].append(same)
            same['saved_point_ids'].append(p['point_id'])
            if p.get('status')=='FEASIBLE' and p.get('conditional_cost')is not None:
                cost=O.frac(p['conditional_cost'])
                if same['best']is None or cost<O.frac(same['best']):same['best']=O.enc(cost)
        if (self.spec.get('candidate_completion') or {}).get('resume_preparation_from'):
            self.seed_ready_continuation()
        self.state['seeded']=dict(UTC=K.now(),mapper='METHOD_SPECIFIC_EXISTING_GENERATOR',
            old_single_width_state_untouched=True)
        self.save()

    def seed_ready_continuation(self):
        comp = self.completion_api()
        for key, rec in comp.state['records'].items():
            if rec.get('status') != 'READY_FOR_SEARCH' or rec.get('method') not in self.methods:
                continue
            cfg = read_loc(rec['configuration'])
            method, spec = rec['method'], cfg['spec']
            kappa = KS._frac(spec['kappa']['value'])
            family = W.width_family_key(method, spec['L_D'], spec['formula'], spec['delta_pf'], spec['finite_K'])
            if not W.kappa_allowed(self.wide_for(method), kappa) or self.state['families'].get(family, {}).get('status') != 'READY_METADATA':
                continue
            point = dict(family=family, method=method, L_D=spec['L_D'], kappa=str(kappa),
                         status='REGISTERED', configuration_id=cfg['id'],
                         completion=dict(status='READY_FOR_SEARCH', key=key))
            if any(p['family'] == family and Fraction(p['kappa']) == kappa for p in self.state['points']):
                continue
            try:
                self.check_completed_configuration(point, rec)
                comp.verify_continuation_inputs(rec)
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.log(event='CONTINUATION_INPUT_HOLD', family=family, key=key, reason=str(error))
                continue
            self.state['points'].append(point)
            self.log(event='CONTINUATION_INPUT_READY', family=family, key=key, configuration_id=cfg['id'])

    def bootstrap_families(self):
        pending=[f for f in self.state['families'].values()if f['status']=='FAMILY_PREPARATION_PENDING']
        if not pending:return
        comp=self.completion_api();approved=self.spec.get('science_approved')is True and not self.a.plan_only
        for f in pending:
            request=f['preparation_request'];started=time.monotonic()
            self.log(event='FAMILY_BOOTSTRAP_START',family=request['selection']['family'])
            if hasattr(comp,'prepare_family'):result=comp.prepare_family(request,allow_science=approved)
            else:result=comp.plan(request)
            f['preparation']=result;config=result.get('configuration')
            self.log(event='FAMILY_BOOTSTRAP_END',family=request['selection']['family'],status=result['status'],
                reason=result.get('reason'),wall_seconds=time.monotonic()-started)
            if config and 'commit'in config and 'path'in config:config=read_loc(config)
            if not config:
                f['reason']=result.get('reason',result['status'])
                if result['status'] == 'UNSUPPORTED_MAPPER':
                    f['status'] = 'UNSUPPORTED_MAPPER'
                continue
            config=config.get('configuration',config);s=config['spec'];template=f['template']['spec']
            try:
                if (KS._frac(s['delta_pf'])!=KS._frac(f['requested_delta_pf'])or
                        KS._frac(s['h'])!=KS._frac(template['h'])or s['M']!=template['M']or
                        s['formula']!=f['formula']or s['finite_K']!=f['K']or s['split']['L_D']!=f['L_D']):
                    raise ValueError('FAMILY_PREPARATION_CHANGED_REQUESTED_DESIGN_OR_TIME_GRID')
                fields=('hamiltonian','state','snapshot','compiler','control')
                if any(s.get('binding',{}).get(k)!=template.get('binding',{}).get(k)for k in fields):
                    raise ValueError('FAMILY_PREPARATION_CHANGED_FIXED_BINDING')
                rows=KS.constants(config)
                if KS.R_vector(rows,KS._frac(s['kappa']['value']))!=[r['R_outer']for r in sorted(config['rounds'],key=lambda r:r['m'])]:
                    raise ValueError('FAMILY_PREPARATION_MAPPER_MISMATCH')
                f.update(status='READY_METADATA',configuration=config,rows=KS.encode_rows(rows),
                         source=result.get('configuration_source'),metadata_kappa_not_a_scientific_candidate=True)
                f.pop('reason',None)  # the fulfilled preparation need is no longer a blocking reason
            except(ValueError,KeyError,TypeError)as exc:
                f.update(status='HELD_FAMILY_MISMATCH',reason=str(exc))

    def families(self):
        return {key:W.Family(key,rows['rows'])for key,rows in self.state['families'].items()
                if rows['status']=='READY_METADATA'}

    def wide_for(self,method):
        overrides=getattr(self,'spec',{}).get('search_overrides',{}).get('schedule_wide_by_method',{})
        return overrides.get(method,self.wide)

    def point(self, family, kappa):
        return next((p for p in self.state['points']if p['family']==family and Fraction(p['kappa'])==Fraction(kappa)),None)

    def model_floor(self, families):
        """Use the same active price model to reproduce a saved feasible cost."""
        if getattr(self,'spec',{}).get('search_overrides',{}).get('common_pauli_price_floor') is not True:
            return None,None
        configs={key:self.state['families'][key]['configuration'] for key in families
                 if all(r['J']==1 for r in families[key].rows)}
        if not configs or not any(p.get('best_MSE_upper') is not None and p.get('best') is not None
                                  for p in self.state['points'] if p['family'] in configs):
            return None,None
        lookup=getattr(self.completion_api(),'common_pauli_model_terms',None)
        if lookup is None:
            return None,None
        try:
            terms=lookup(configs)
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
            return None,None

        def prices(key,R):
            return [one+slope*(r-1) for (one,slope),r in zip(terms[key],(r for r in R for _ in O.AXES))]

        feasible=[]
        tau=O.tau(self.problem)
        for p in self.state['points']:
            key=p['family']
            if (key not in terms or p.get('best_MSE_upper') is None or p.get('best') is None
                    or not 0<=O.frac(p['best_MSE_upper'])<=tau):
                continue
            nx,ny=p.get('best_N_X'),p.get('best_N_Y')
            if nx is None or ny is None or len(nx)!=13 or len(ny)!=13:
                continue
            vector=O.interleave(nx,ny)
            if any(type(n)is not int or n<1 for n in vector):
                continue
            own=prices(key,families[key].R(Fraction(p['kappa'])))
            cost=O.cost(vector,own)
            if cost==O.frac(p['best']):
                feasible.append(cost)
        if not feasible:
            return None,None
        return lambda key,R:sum(prices(key,R)) if key in terms else None,min(feasible)

    def plan(self):
        families=self.families();points=[]
        for p in self.state['points']:
            costs=[O.frac(v)for v in(p.get('best'),p.get('confirmed_all_best'))if v is not None]
            best=min(costs)if costs else None
            status=('FEASIBLE'if best is not None else 'NO_FEASIBLE_FOUND'if p['status']=='EVALUATED'
                    else 'UNAVAILABLE'if p['status']in('HELD','PRUNED')else'PENDING')
            points.append(dict(family=p['family'],kappa=Fraction(p['kappa']),status=status,
                best=best,min_lower_over_tau=p.get('min_lower_over_tau')))
        # The existing rule chooses schedules within each method. An incumbent
        # from another method must not close this method's refinement window.
        proposals=[];decisions=[];reports={};family_reports={}
        methods=list(dict.fromkeys(self.state['families'][key]['method']for key in families))
        for method in methods:
            scoped={key:f for key,f in families.items()if self.state['families'][key]['method']==method}
            scaled = getattr(self, 'scaled_shots', False)
            floor,verified_best=self.model_floor(scoped) if method=='PR_PAULI' and not scaled else (None,None)
            selection = self.wide_for(method)
            if scaled:
                selection = dict(selection, stop_by_observed_trend=False)
            grid = W.kappa_values(selection)
            if (grid is not None and not self.a.plan_only and self.spec.get('science_approved') is True
                    and self.work_conserving_search and self.live_ready_refill
                    and (self.spec.get('executor', {}).get('distributed') or {}).get('preparation')):
                # An explicit grid does not need a result to select its next
                # point. Keep the independent preparation lanes supplied even
                # when an earlier point (or its split) is still running.
                # complete_points still limits concurrent work and owns reuse.
                selection = dict(selection, per_decision=len(scoped) * len(grid))
            chosen,why,summary=W.plan_wide(scoped,[p for p in points if p['family']in scoped],selection,
                                          floor,price_floor_best=verified_best)
            proposals.extend(chosen);decisions.extend(why);reports[method]=summary;family_reports.update(summary['families'])
        report=dict(per_method=reports,global_best_by_method={m:r['global_best']for m,r in reports.items()},
            global_best=next(iter(reports.values()))['global_best']if len(reports)==1 else None,families=family_reports,
            scope='INDEPENDENT_METHOD_MINIMA',stop=None if proposals else
                'WAIT_FOR_PENDING'if any(r['stop']=='WAIT_FOR_PENDING'for r in reports.values())else'RULE_EXHAUSTED')
        for proposal in proposals:
            family=proposal['family']
            if self.point(family,proposal['kappa'])is not None:continue
            f=self.state['families'][family]
            request=dict(method=f['method'],selection=dict(rule=W.RULE,family=family,reason=proposal['reason']),
                design=dict(L_D=f['L_D'],formula=f['formula'],K=f['K'],delta_pf=f['requested_delta_pf'],
                            kappa=str(proposal['kappa'])))
            self.state['points'].append(dict(family=family,method=f['method'],L_D=f['L_D'],
                requested_delta_pf=f['requested_delta_pf'],kappa=str(proposal['kappa']),
                status='COMPLETION',completion_request=request,request_reason=proposal['reason']))
        self.state['history'].append(dict(cycle=self.state['cycle'],decisions=decisions,report=report,
            proposals=[dict(family=p['family'],kappa=str(p['kappa']),reason=p['reason'])for p in proposals]))
        return proposals,report

    def completion_api(self):
        if self.completion is None:
            from trottertracks.pr_candidate_completion import Completion
            settings=dict(self.spec.get('candidate_completion')or{})
            settings.setdefault('price_registry_file',self.problem.get('price_rules',{}).get('registry_file'))
            settings['plan_only']=self.a.plan_only or self.spec.get('science_approved')is not True
            self.completion=Completion(self.run/'completion',settings)
        return self.completion

    def complete_points(self, stop_requested=None, announce_ready=None, stop_on_error=None,
                        wait_for_preparation=None, *, continuous=False):
        comp=self.completion_api()
        if hasattr(comp,'ingest_price_results'):comp.ingest_price_results()
        if hasattr(comp,'reconcile_committed_df_holdout'):
            for point in self.state['points']:
                completion = point.get('completion') or {}
                record = comp.state['records'].get(completion.get('key'))
                if point['status'] == 'HELD' and record and comp.reconcile_committed_df_holdout(record):
                    point.update(status='WAITING_INPUTS',reason=record['reason'])
        # INIT may expose a whole far-kappa ladder at once. Distributed
        # preparation refills bounded lanes as candidates finish; other callers
        # keep a single per-decision wave.
        pending=[p for p in self.state['points']if p['status']in('COMPLETION','WAITING_INPUTS','PRICE_ACTION_UNAVAILABLE')]
        total_pending = len(pending)
        priced_rechecks = (
            getattr(comp, 'priced_pauli_recheck_keys', set)() if continuous else set())
        blocked = comp.price_deferred_keys() if hasattr(comp, 'price_deferred_keys') else set()
        if blocked:
            from trottertracks.pr_candidate_completion import key
            pending = [p for p in pending if key(p['completion_request']) not in blocked]
        cycle=self.state['cycle'];limit=W.settings(self.wide)['per_decision']
        if continuous:
            pending = [p for p in pending if p.get('input_dispatch_cycle') != cycle]
        wave=self.state.get('preparation_wave')
        if wave is None or wave['cycle']!=cycle:
            wave=self.state['preparation_wave']=dict(cycle=cycle,dispatched={})
        selected=[];counts={}
        review_order=self.spec.get('search_overrides',{}).get('review_search_order') is True
        for method in self.methods:
            queues={}
            for p in pending:
                if p['method']==method:queues.setdefault(p['family'],[]).append(p)
            costs={};loads={};dear=set()
            incumbents=[O.frac(p['best']) for p in self.state['points']
                        if p.get('method')==method and p.get('best') is not None]
            incumbent=min(incumbents) if review_order and incumbents else None
            for family in queues:
                members=[p for p in self.state['points']if p.get('family')==family]
                saved=[O.frac(v)for p in members for v in(p.get('best'),p.get('confirmed_all_best'))if v is not None]
                costs[family]=min(saved)if saved else None
                loads[family]=sum(p.get('input_dispatches',0)for p in members)
                if incumbent is not None:
                    anchors=[Fraction(p['kappa']) for p in members if p.get('own_price_floor') is not None
                             and O.frac(p['own_price_floor'])>=incumbent]
                    if anchors:
                        dear.update(id(p) for p in queues[family] if Fraction(p['kappa'])>=min(anchors))
                queues[family].sort(key=lambda p:(id(p) in dear,p.get('input_dispatches',0)))
            known=sorted((f for f in queues if costs[f]is not None),key=lambda f:costs[f])
            unknown=[f for f in queues if costs[f]is None]
            order=[]
            for i in range(max(len(known),len(unknown))):
                if i<len(known):order.append(known[i])
                if i<len(unknown):order.append(unknown[i])
            # Fairness is only a dispatch order, not a price estimate or veto.
            order.sort(key=lambda f:loads[f])
            if self.work_conserving_search:
                # Diversify the next READY supply across requested widths, while
                # keeping the existing family and kappa order within each width.
                by_width = {}
                for family in order:
                    delta = KS._frac(self.state['families'][family]['requested_delta_pf'])
                    by_width.setdefault(delta, []).append(family)
                from itertools import zip_longest
                order = [family for level in zip_longest(*by_width.values()) for family in level if family is not None]
            available = sum(map(len, queues.values())) if continuous else max(0, limit-wave['dispatched'].get(method,0))
            chosen=[]
            while len(chosen)<available and any(queues.values()):
                ready=[family for family in order if queues[family] and id(queues[family][0]) not in dear]
                for family in ready or order:
                    if queues[family]and len(chosen)<available:chosen.append(queues[family].pop(0))
            selected.extend(chosen);counts[method]=len(chosen)
            if not continuous:
                wave['dispatched'][method]=wave['dispatched'].get(method,0)+len(chosen)
        if priced_rechecks:
            from trottertracks.pr_candidate_completion import key
            def priced_recheck(point):
                recorded = (point.get('completion') or {}).get('key')
                return (recorded or key(point['completion_request'])) in priced_rechecks

            priced_points = {id(point) for point in selected if priced_recheck(point)}
            selected.sort(key=lambda point: id(point) not in priced_points)
        else:
            priced_points = set()
        reason = ('CONTINUOUS_PER_METHOD_WIDTH_INTERLEAVED_FAMILY_ROUND_ROBIN' if continuous else
                  'PER_DECISION_PER_METHOD_WIDTH_INTERLEAVED_FAMILY_ROUND_ROBIN' if self.work_conserving_search else
                  'EXISTING_PER_DECISION_PER_METHOD_FAMILY_ROUND_ROBIN')
        self.log(event='PREPARATION_WAVE_START',dispatch_reason=reason,per_method_limit=limit,
            selected_by_method=counts,pending_before=total_pending,deferred=total_pending-len(selected),
            waiting_unchanged_price=total_pending-len(pending),
            selected=[dict(method=p['method'],family=p['family'],kappa=p['kappa'])for p in selected])
        def prepare_points(points):
            for p in points:
                if ((wait_for_preparation is not None and not wait_for_preparation())
                    or (stop_requested is not None and stop_requested())):
                    self.log(event='PREPARATION_YIELD_AFTER_FATAL');break
                p['input_dispatches']=p.get('input_dispatches',0)+1
                if continuous:
                    p['input_dispatch_cycle'] = cycle
                started=time.monotonic();self.log(event='CANDIDATE_INPUTS_START',family=p['family'],kappa=p['kappa'])
                if self.a.plan_only or self.spec.get('science_approved')is not True:
                    record=comp.plan(p['completion_request'])
                else:record=comp.complete(p['completion_request'])
                self.log(event='CANDIDATE_INPUTS_END',family=p['family'],kappa=p['kappa'],status=record['status'],
                    reason=record.get('reason'),wall_seconds=time.monotonic()-started)
                p['completion']=dict(status=record['status'],key=record.get('key'),reason=record.get('reason'),
                                     signal_plan=record.get('signal_plan'),price_plan=record.get('price_plan',record.get('price_quote')),
                                     calibration_requests=record.get('calibration_requests'),readiness=record.get('readiness'),
                                     rounds=record.get('rounds'))
                if record.get('conflicting_signal_rounds'):
                    p['completion']['conflicting_signal_rounds']=record['conflicting_signal_rounds']
                if record.get('price_actions'):
                    p['completion']['price_actions']=record['price_actions']
                if record['status']=='READY_FOR_SEARCH':
                    try:self.check_completed_configuration(p,record)
                    except (ValueError,KeyError,TypeError) as exc:
                        p.update(status='HELD',reason='COMPLETED_CONFIGURATION_MISMATCH:'+str(exc));continue
                    p.update(status='REGISTERED',configuration_id=record['configuration_id'])
                    if announce_ready is not None:
                        announce_ready(p['method'], record)
                elif record['status']=='SELECTED'and str(record.get('reason','')).startswith('ORCHESTRATOR_YIELD_BEFORE_STAGE:'):
                    p.update(status='COMPLETION',reason=record['reason'])
                    self.log(event='PREPARATION_YIELD_AFTER_FATAL',family=p['family'],stage=record.get('interrupted_before_stage'))
                    break
                elif record['status']=='PRICE_ACTION_UNAVAILABLE':
                    p.update(status='PRICE_ACTION_UNAVAILABLE',reason=record['reason'])
                elif (record.get('conflicting_signal_rounds') or record['status'].startswith('STOP')
                      or str(record.get('reason','')).startswith(('PREPARATION_REQUEST_MISMATCH:',
                                                                 'PREPARATION_RESULT_MISMATCH:'))
                      or 'NUMERICAL_GATE' in str(record.get('reason','')) or
                      getattr(self,'rule_stop',False) and record.get('scientific_failure')):
                    p.update(status='HELD',reason=record.get('reason',record['status']))
                else:
                    p.update(status='WAITING_INPUTS',reason=record.get('reason',record['status']))
        if announce_ready is None:
            prepare_points(selected)
        elif selected:
            # Keep one split in order to reuse its descriptor and signals.
            # Pauli also needs the distributed provider's isolated attempts.
            from collections import Counter, deque
            from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
            from contextlib import ExitStack
            parallel_methods={'PR_DF'}
            if getattr(comp,'parallel_pauli_preparation',lambda:False)():
                parallel_methods.add('PR_PAULI')
            groups = {}
            for point in selected:
                method = point['method']
                split = self.state['families'][point['family']]['L_D'] if method in parallel_methods else None
                if id(point) in priced_points and method in parallel_methods:
                    # Exact configurations already have their saved descriptor
                    # and share no signal assignment.
                    split = (split, id(point))
                groups.setdefault(method, {}).setdefault(split, []).append(point)
            parallel = int((self.spec.get('candidate_completion') or {}).get('parallel', 1))
            lane_limit = parallel if continuous else min(limit, parallel)
            lanes = {m: max(1, min(lane_limit, len(splits)) if m in parallel_methods else 1)
                     for m, splits in groups.items()}
            with ExitStack() as stack:
                pools = {method: stack.enter_context(ThreadPoolExecutor(max_workers=workers))
                         for method, workers in lanes.items()}
                tasks = []
                try:
                    if continuous:
                        # A preparation may be queued while Q3 owns every slot.
                        # Only these bounded, independent candidates can submit
                        # stages. The common executor prioritizes Q3 on dispatch.
                        by_point = {id(point): (method, split) for method, splits in groups.items()
                                    for split, points in splits.items() for point in points}
                        backlog = deque(selected)
                        active = {}
                        max_active = sum(lanes.values())
                        while backlog or active:
                            if stop_requested is not None and stop_requested():
                                backlog.clear()
                            busy = set(active.values())
                            used = Counter(method for method, _ in busy)
                            priced_waiting = any(id(point) in priced_points for point in backlog)
                            priced_active = any(isinstance(split, tuple) for _, split in busy)
                            for _ in range(len(backlog)):
                                if len(active) >= max_active:
                                    break
                                point = backlog.popleft()
                                method, split = by_point[id(point)]
                                if method == 'PR_PAULI' and id(point) not in priced_points and (priced_waiting or priced_active):
                                    backlog.append(point)
                                    continue
                                if used[method] >= lanes[method] or (method, split) in busy:
                                    backlog.append(point)
                                    continue
                                active[pools[method].submit(prepare_points, [point])] = (method, split)
                                busy.add((method, split))
                                used[method] += 1
                                wave['dispatched'][method] = wave['dispatched'].get(method, 0) + 1
                            if active:
                                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                                for task in finished:
                                    del active[task]
                                    task.result()
                    else:
                        for method, splits in groups.items():
                            tasks.extend(pools[method].submit(prepare_points, points) for points in splits.values())
                        for task in as_completed(tasks):
                            task.result()
                except BaseException:
                    if stop_on_error is not None:
                        stop_on_error()
                    raise
        self.log(event='PREPARATION_WAVE_END',dispatch_reason=reason,selected_by_method=counts,
            remaining_pending=sum(p['status']in('COMPLETION','WAITING_INPUTS')for p in self.state['points']))

    def check_completed_configuration(self, point, record):
        """Metadata check against the mapper that set this family's kappa intervals."""
        config=record['configuration']
        if 'commit'in config and 'path'in config:config=read_loc(config)
        config=config.get('configuration',config);s=config['spec'];f=self.state['families'][point['family']]
        base=f['configuration']['spec'];rows=KS.constants(config)
        if KS.encode_rows(rows)!=f['rows']:raise ValueError('DIFFERENT_LAMBDA_TIME_GRID_OR_PF_SCHEDULE')
        if ([r['R_outer']for r in sorted(config['rounds'],key=lambda r:r['m'])]
                !=KS.R_vector(rows,Fraction(point['kappa']))):raise ValueError('DIFFERENT_R_SCHEDULE')
        if (KS._frac(s['delta_pf'])!=KS._frac(f['requested_delta_pf'])or
                KS._frac(s['h'])!=KS._frac(base['h'])or s['M']!=base['M']or
                s['formula']!=f['formula']or s['finite_K']!=f['K']or
                s['split']['L_D']!=f['L_D']):raise ValueError('DIFFERENT_REQUESTED_DESIGN')
        for section in('binding','law','profile'):
            if s.get(section)!=base.get(section):raise ValueError('DIFFERENT_'+section.upper())
        for field in('scalar','one_body_policy','tail_dictionary','lambda_source'):
            if s['split'].get(field)!=base['split'].get(field):raise ValueError('DIFFERENT_SPLIT_'+field.upper())

    def context(self, problem):
        return O.prepare(ROOT,problem,methods=self.methods,generate_configurations=False,
                         preparation_dir=self.run/'preparation',saved_catalog=self.saved_catalog(problem))

    def confirm_completed(self, context, executor, result):
        """Existing ledger checks after finish, including unadopted speculation.

        No Q3/collect/price fit is run. Rule-adopted best stays separate; only
        source/identity-checked outcomes with consistent exact bounds and own
        request prices enter the confirmed minimum, regardless of gain size.
        """
        import hashlib
        tau=context['tau'];methods={}
        if getattr(self, 'scaled_shots', False):
            # Prior/free-vector results cannot update a multiplier-grid minimum.
            for method, r in result.items():
                r.setdefault('rule_adopted_best', r['best'])
                r['confirmed_all_best'] = r['best']
                methods[method] = dict(rule_adopted_best=r['rule_adopted_best'], confirmed_all_best=r['best'],
                                       scaled_shots=r['scaled_shots'])
            self.state['method_minima'] = methods
            K.CLI.write(self.run/'CONFIRMED_MINIMA.json', K.CLI.slim(dict(
                schema='RPE_WIDTH_CONFIRMED_MINIMA_V1', tau=tau, methods=methods,
                scope='RPE_SCALED_SHOTS_V1_CONDITIONAL_GRID_ONLY', new_scientific_calls=0)))
            return
        for method,r in result.items():
            candidates=[c for c in context['candidates']if c['method']==method];ready={c['configuration_id']:c for c in candidates if c['ready']}
            view=executor.for_method(method)
            view.prior_run_dirs=list(dict.fromkeys([*view.prior_run_dirs,self.run]))
            known,refused=view.prior_records(candidates)
            adopted={(t['configuration_id'],tuple(t['vector']))for t in r.get('trace',[])}
            r['rule_adopted_best']=deepcopy(r.get('best'));confirmed=deepcopy(r.get('best'));rows=[];by_config={}
            for b in r.get('by_configuration')or[]:
                b['rule_adopted_best']=deepcopy(b.get('best'));b['confirmed_all_best']=deepcopy(b.get('best'))
                by_config[b['configuration_id']]=b
            for (cid,vector),saved in known.items():
                job=Path(saved['job']);job=job if job.is_absolute()else view.root/job
                request=O.load_json(job/'REQUEST.json');c=ready[cid];price=O.cost(vector,c['prices'])
                requested=O.frac(request['cost'])if request.get('cost')is not None else None
                lo,up=saved.get('MSE_lower'),saved.get('MSE_upper');status=saved['status']
                consistent=((status==O.PASS and up is not None and up<=tau)or
                    (status==O.FAIL and lo is not None and lo>tau)or
                    (status=='INCONCLUSIVE'and lo is not None and up is not None and lo<=tau<up))
                outcome,path=O.job_outcome(job)
                row=dict(configuration_id=cid,N_X=O.split(vector)[0],N_Y=O.split(vector)[1],status=status,
                    cost=price,request_cost=requested,price_match=requested==price,status_consistent=consistent,
                    identity='EXECUTOR_PRIOR_RECORD_CHECK_PASSED',verified=requested==price and consistent,
                    adopted_by_rule=(cid,vector)in adopted,job=str(job),outcome_file=str(path),
                    outcome_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),artificial=outcome.get('artificial')is True)
                rows.append(row)
                if not row['verified']or status!=O.PASS:continue
                best=dict(candidate=c['name'],configuration_id=cid,configuration=c['key'],N_X=row['N_X'],N_Y=row['N_Y'],
                    status=status,MSE_lower=lo,MSE_upper=up,cost=price,cost_display=O.show(price),evidence='VERIFIED_COMPLETED_JOB',
                    job=str(job),outcome_source=dict(path=str(path),sha256=row['outcome_sha256']),adopted_by_rule=row['adopted_by_rule'])
                if confirmed is None or price<O.frac(confirmed['cost']):confirmed=best
                b=by_config.get(cid)
                if b is not None and(b['confirmed_all_best']is None or price<O.frac(b['confirmed_all_best']['cost'])):
                    b['confirmed_all_best']=dict(vector=list(vector),cost=price,cost_display=O.show(price),status=status,
                        MSE_lower=lo,MSE_upper=up,source='VERIFIED_COMPLETED_JOB',job=str(job),outcome_source=best['outcome_source'],
                        adopted_by_rule=row['adopted_by_rule'])
            r['confirmed_all_best']=confirmed
            r['completed_ledger']=dict(jobs_checked=len(rows),jobs_verified=sum(row['verified']for row in rows),
                refused=refused,rows=rows,scope='EXACT_TAU_AND_OWN_PRICE_CHECK_AFTER_EXECUTOR_FINISH')
            methods[method]=dict(rule_adopted_best=r['rule_adopted_best'],confirmed_all_best=confirmed,
                rule_adopted_min=None if r['best']is None else r['best']['cost'],
                verified_all_min=None if confirmed is None else confirmed['cost'],
                jobs_checked=len(rows),jobs_verified=r['completed_ledger']['jobs_verified'],jobs_refused=len(refused))
        self.state['method_minima']=methods
        K.CLI.write(self.run/'CONFIRMED_MINIMA.json',K.CLI.slim(dict(schema='RPE_WIDTH_CONFIRMED_MINIMA_V1',tau=tau,
            methods=methods,scope='PER_METHOD_EVALUATED_SET_CONDITIONAL_PRICE_NOT_FORMAL_G',new_scientific_calls=0)))

    def live_ready_source(self, executor, completion, problem, stop_preparation, *, continuous=False):
        """Publish validated registrations to each selector's private view."""
        from contextlib import nullcontext
        from types import SimpleNamespace
        announced = {method: set() for method in self.methods}
        observed = {method: set() for method in self.methods}
        from trottertracks.rpe_preparation_round import PF_METHODS, _merge_pf, snapshot_context
        pf_snapshot = None
        pf_revision = 0
        pf_observed = {method: 0 for method in self.methods}
        state = executor.state
        condition = state.cv

        def wait_for_preparation():
            # The distributed queue gives demand Q3 first access at dispatch.
            # A direct preparation still waits before acquiring a slot itself.
            with condition:
                while not (stop_preparation.is_set() or state.failure or executor._user_stop_requested()):
                    if continuous:
                        refill = getattr(executor, 'preparation_needs_refill', None)
                        if refill is None or refill():
                            return True
                    elif executor.has_spare_slot(allow_queued_speculation=self.work_conserving_search):
                        return True
                    condition.wait(.5)  # also observes STOP written outside the executor
                return False

        def announce(method, record):
            with condition:
                executor.for_method(method).log(event='CANDIDATE_READY', configuration=record['configuration_id'],
                                                completion_key=record['key'])
                announced[method].add(record['key'])
                condition.notify_all()

        def announce_pf(context, routes):
            nonlocal pf_snapshot, pf_revision
            with condition:
                pf_snapshot = (context, routes)
                pf_revision += 1
                condition.notify_all()

        def has_new(method):
            with condition:
                return not stop_preparation.is_set() and (
                    bool(announced[method] - observed[method]) or
                    (method in PF_METHODS and pf_revision > pf_observed[method]))

        def subscriber(method, context, candidates, view):
            seen = set()
            pf_seen = 0
            delivered = {c['configuration_id'] for c in candidates if c['ready']}

            def pending():
                with condition:
                    return not stop_preparation.is_set() and (
                        bool(announced[method] - seen) or
                        (method in PF_METHODS and pf_revision > pf_seen))

            def take():
                nonlocal pf_seen
                with condition:
                    keys = announced[method] - seen
                    seen.update(keys)
                    observed[method].update(keys)
                    pf = pf_snapshot if method in PF_METHODS and pf_revision > pf_seen else None
                    if pf is not None:
                        pf_seen = pf_revision
                        pf_observed[method] = pf_revision
                if (not keys and pf is None) or stop_preparation.is_set():
                    return [], None
                scratch = dict(context, candidates=deepcopy(candidates), expected=dict(view.expected))
                scoped = dict(problem, execution_routes=view.routes)
                if pf is not None:
                    scratch = snapshot_context(scratch)
                    scratch, routes = _merge_pf(scratch, pf[0], dict(view.routes), pf[1], [method])
                    registered = {c['configuration_id'] for c in scratch['candidates'] if c['method'] == method}
                elif hasattr(completion, 'install_snapshot'):
                    routes, records = completion.install_snapshot(scratch, scoped, keys)
                else:
                    with getattr(completion, 'lock', nullcontext()):
                        records = deepcopy([completion.state['records'][key] for key in keys])
                    routes = completion.install(scratch, scoped, records)
                if pf is None:
                    registered = {r.get('installed_configuration_id', r.get('configuration_id')) for r in records}
                O.scope_absolute_kappa(scratch, problem)
                O.apply_published(scratch['candidates'], routes)
                # The installer resolves aliases. Keep existing routes and
                # bindings, including those of jobs already running.
                trial = view.for_method(method)
                trial.routes, trial.expected = routes, scratch['expected']
                added = []
                new_routes = []
                expected = dict(view.expected)
                for candidate in scratch['candidates']:
                    cid = candidate['configuration_id']
                    if (candidate['method'] != method or not candidate['ready']
                        or not cid or cid not in registered or cid in delivered):
                        continue
                    route = trial.route(cid)
                    if route is None or trial.refresh_execution_hold(cid):
                        continue
                    candidate['execution_needs'] = []
                    candidate.pop('execution_hold', None)
                    added.append(candidate)
                    delivered.add(cid)
                    new_routes.append(dict(route, configuration_ids=[cid]))
                    expected[cid] = trial.expected_for(cid, route)
                view.routes = dict(view.routes, **{method: new_routes + view.routes.get(method, [])})
                view.expected = expected
                if added:
                    self.log(event='LIVE_READY_INSTALLED', method=method,
                             configurations=[c['configuration_id'] for c in added])
                return added, scratch.get('saved') if pf is not None else None
            return pending, take
        return announce, wait_for_preparation, SimpleNamespace(subscribe=subscriber, has_new=has_new,
                                                              announce_pf=announce_pf)

    def batch(self, prepare_during=False):
        # Same V2/Executor and risk contract as the old ordinary round entry.
        # Completion selects no new designs here: the width planner already did.
        problem=deepcopy(self.problem)
        for method, widths in self.widths.items():
            problem['methods'][method]['requested_delta_pf_values']=[O.enc(KS._frac(w['raw']))for w in widths]
        context=self.context(problem);comp=self.completion_api()
        routes=K.CLI.merged_routes(problem,self.spec)
        pf_settings=self.spec.get('pf_completion') or self.problem.get('pf_completion') or {}
        pf_enabled=getattr(self,'rule_stop',False) and ('TRUNCATED_PF' in self.methods or
            'FULL_PF' in self.methods and bool(pf_settings))
        if pf_enabled:
            from trottertracks import pf_optimizer_connection as PF
            routes=PF.registry_routes(routes,context['pf_evidence_registry'])
        def install_ready(current_routes, snapshot=False):
            accepted={p.get('completion',{}).get('key')for p in self.state['points']if p['status']in('REGISTERED','EVALUATED')}
            if snapshot and hasattr(comp,'install_snapshot'):
                current_routes,records=comp.install_snapshot(context,dict(problem,execution_routes=current_routes),accepted)
            else:
                records=[r for r in comp.state['records'].values()if r.get('key')in accepted]
                current_routes=comp.install(context,dict(problem,execution_routes=current_routes),records)
            installed={r['key']:r.get('installed_configuration_id',r.get('configuration_id'))for r in records}
            for p in self.state['points']:
                canonical=installed.get(p.get('completion',{}).get('key'))
                if canonical:p['evaluation_configuration_id']=canonical
            return current_routes
        routes=install_ready(routes)
        O.apply_published(context['candidates'],routes)
        options=self.spec.get('executor',{})
        launched={m:sum((job/'LAUNCH.json').exists()or(job/'STARTING.json').exists()
            for job in(self.run/'jobs'/m).glob('*'))for m in self.methods}
        budgets={m:launched[m]+self.a.batch for m in self.methods}
        for method,budget in budgets.items():
            maximum=self.spec.get('max_new_q3_per_method',{}).get(method)
            if maximum=='UNLIMITED' and getattr(self,'rule_stop',False):budgets[method]='UNLIMITED'
            elif maximum not in(None,'UNLIMITED'):budgets[method]=min(budget,maximum)
        search=dict(problem['search'],**self.spec.get('search_overrides',{}))
        if getattr(self,'rule_stop',False):
            search['max_steps_per_method']=self.spec.get('max_steps_per_method')
            search['work_slice_steps']=W.settings(self.wide)['per_decision']
        # Existing V2 switch: kappa schedules are chosen by this outer width
        # planner, not by a second inner multiplier-neighbour chooser.
        search.setdefault('kappa_neighbourhood',dict(outer_planner=W.RULE))
        # The width planner still chooses designs. Continuation may refresh and
        # install its existing waiting records, without a second design chooser.
        active_spec=deepcopy(self.spec);active_spec['max_new_q3_per_method']=budgets
        active_spec['executor']=dict(options,prior_run_dirs=list(options.get('prior_run_dirs',[]))+[str(self.run)])
        if self.live_ready_refill and active_spec['executor'].get('distributed'):
            if self.work_conserving_search and (active_spec['executor']['distributed'].get('ready_queue_waves', 2) > 1):
                search['prefetch_independent_references'] = True
                discrete = all(W.kappa_values(self.wide_for(method)) is not None for method in self.widths)
                search['ready_work_at_slice'] = 'refill' if discrete else 'drain'
            active_spec['executor']['distributed'] = dict(active_spec['executor']['distributed'], live_ready_refill=True,
                work_conserving_search=self.work_conserving_search)
        active_spec['candidate_completion']=dict(self.spec.get('candidate_completion')or{},max_attempted_per_method=0)
        executor,optimize=K.CLI.make_executor(ROOT,self.run,routes,context['expected'],problem,active_spec)
        manager = getattr(executor, 'pbs_manager', None)
        if manager is not None and manager.hold_node:
            manager.retain_between_batches = True
            self.run_pbs_manager = manager
        preparation_pool=None
        try:
            runtime=K.CLI.preparation_runtime(executor,self.run,active_spec)
            if runtime is not None:comp.bind_stage_executor(runtime)
            if any(hold.get('replay_exact') is True for hold in getattr(executor, 'prior_q3_holds', ())):
                prepare_during = executor.replay_prior_q3() and prepare_during
            interim=None
            fatal={'UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD'}
            if prepare_during:
                # Snapshot/install happens before starting preparation. While
                # it owns Completion state, Q3 only reads this frozen context.
                from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
                import inspect,threading
                stop_preparation=threading.Event()

                def stop_preparing():
                    with executor.state.cv:
                        stop_preparation.set()
                        executor.state.cv.notify_all()

                preparation_pool=ThreadPoolExecutor(max_workers=2 if pf_enabled else 1)
                executor.state.failure_callback=lambda error:stop_preparing()
                comp.stop_requested=stop_preparation.is_set
                supports_stop='stop_requested'in inspect.signature(self.complete_points).parameters
                prepare_options = {'stop_requested': stop_preparation.is_set} if supports_stop else {}
                search_options = {}
                if self.live_ready_refill:
                    continuous = self.work_conserving_search and runtime is not None
                    announce, wait_for_preparation, subscriber = self.live_ready_source(
                        executor, comp, problem, stop_preparation, continuous=continuous)
                    prepare_options.update(announce_ready=announce, stop_on_error=stop_preparing,
                                           wait_for_preparation=wait_for_preparation, continuous=continuous)
                    search_options['ready_source'] = subscriber
                def prepare():
                    if runtime is not None and not stop_preparation.is_set():
                        self.bootstrap_families()
                    if not stop_preparation.is_set():
                        self.complete_points(**prepare_options)
                pf_future=None
                self.pf_more=False
                if pf_enabled:
                    from trottertracks import pf_candidate_selection
                    from trottertracks.rpe_preparation_round import snapshot_context,_merge_pf
                    trial = globals().get('FIXED_RUN_WORKER_TRIAL') or {}
                    if trial.get('recover_saved_price_rounds') and not active_spec.get('pf_completion'):
                        active_spec = dict(active_spec, pf_completion=dict(problem['pf_completion'],
                                                                           recover_saved_price_rounds=True))
                    args = (ROOT, problem, snapshot_context(context), active_spec, self.run, deepcopy(routes), runtime)
                    def prepare_pf():
                        stopped = lambda: stop_preparation.is_set() or executor._user_stop_requested()
                        if (self.live_ready_refill and continuous and search.get('ready_work_at_slice') == 'refill'):
                            return pf_candidate_selection.prepare_available(*args, stop_requested=stopped,
                                                                           announce=subscriber.announce_pf,
                                                                           wait_for_work=wait_for_preparation)
                        result = pf_candidate_selection.prepare_next(*args, stop_requested=stopped)
                        if self.live_ready_refill:
                            subscriber.announce_pf(*result[:2])
                        return result
                    pf_future=preparation_pool.submit(prepare_pf)

                def install_pf():
                    nonlocal context, routes, pf_future
                    pf_context,pf_routes,pf=pf_future.result()
                    context,routes=_merge_pf(context,pf_context,routes,pf_routes,self.methods)
                    executor.routes=routes;executor.expected=context['expected']
                    O.apply_published(context['candidates'],routes)
                    self.pf_more=pf['status']=='PREPARATION_WAVE_PROCESSED'
                    K.CLI.write(self.run/'PF_PREPARATION_STATUS.json',pf)
                    pf_future=None
                def ready_keys():
                    return {p['completion']['key'] for p in self.state['points']
                            if p['status'] in ('REGISTERED', 'EVALUATED') and p.get('completion', {}).get('key')}

                installed=ready_keys()
                pending=preparation_pool.submit(prepare)
                can_search=any(c['ready']for c in context['candidates'])
                while True:
                    if pf_future is not None and pf_future.done():
                        install_pf()
                        can_search=any(c['ready']for c in context['candidates'])
                        self.log(event='PF_READY_INSTALLED_DURING_PREPARATION',pr_pending=not pending.done())
                    if can_search:
                        with K.CLI.round_phase(self.run,'WIDTH_READY_DURING_PREPARATION',on_error=lambda error:stop_preparing()):
                            interim=optimize(ROOT,dict(problem,search=search),context=context,methods=self.methods,
                                initial=self.spec.get('initial','all'),max_new=budgets,executor=executor,science_approved=True,
                                **search_options)
                        if any(r['stop_reason']in fatal for r in interim.values()):
                            stop_preparing();pending.result();break
                    if pending.done():
                        pending.result()
                        if pf_future is None: break
                    ready=ready_keys()
                    changed=ready-installed
                    # Extend an internal batch only inside the same adopted
                    # overall per-method budget. Finite budgets stay hard caps.
                    replenished=False;remaining_ready_work=False
                    if interim:
                        for method,r in interim.items():
                            spent=executor.launched(method);maximum=self.spec.get('max_new_q3_per_method',{}).get(method)
                            can_continue = (r['stop_reason']=='STEP_LIMIT' or
                                            (r['stop_reason']=='WORK_SLICE_YIELDED' and r['new_calls']>0))
                            if can_continue and (budgets[method]=='UNLIMITED' or spent<budgets[method]):
                                remaining_ready_work=True
                            if isinstance(budgets[method],int) and spent>=budgets[method]and(changed or r['stop_reason']in('Q3_BUDGET_EXHAUSTED','STEP_LIMIT')):
                                target=spent+self.a.batch
                                if maximum not in(None,'UNLIMITED'):target=min(target,maximum)
                                if target>budgets[method]:budgets[method]=target;replenished=True
                    if changed:
                        routes=install_ready(routes,snapshot=True);executor.routes=routes;executor.expected=context['expected']
                        O.apply_published(context['candidates'],routes);installed=ready
                        self.log(event='LATE_READY_INSTALLED_DURING_PREPARATION',keys=sorted(changed))
                    pf_ready = self.live_ready_refill and any(subscriber.has_new(m) for m in self.methods)
                    can_search=bool(changed or replenished or remaining_ready_work or pf_ready)
                    if not can_search:
                        producers=[f for f in (pending,pf_future) if f is not None and not f.done()]
                        wait(producers,timeout=1,return_when=FIRST_COMPLETED)
                # No other thread now changes Completion or its registry.
                routes=install_ready(routes);executor.routes=routes;executor.expected=context['expected']
                if pf_future is not None:
                    install_pf()
                O.apply_published(context['candidates'],routes)
            if interim and any(r['stop_reason']in fatal for r in interim.values()):result=interim
            else:
                with K.CLI.round_phase(self.run,'WIDTH_SEARCH_AND_Q3'):
                    result=optimize(ROOT,dict(problem,search=search),context=context,methods=self.methods,
                        initial=self.spec.get('initial','all'),max_new=budgets,executor=executor,
                        science_approved=self.spec.get('science_approved')is True)
            actionable = any(req['kind'] != 'HORIZON_INPUTS_MISSING'
                             for r in result.values() for req in r.get('input_requests', []))
            if not any(r['stop_reason'] in fatal for r in result.values()) and actionable:
                from trottertracks.pr_candidate_completion import continue_selected_preparation
                with K.CLI.round_phase(self.run,'WIDTH_PREPARATION_CONTINUATION'):
                    result=continue_selected_preparation(ROOT,dict(problem,search=search),context,
                        active_spec,self.run,comp,executor,result,optimizer=optimize)
        except BaseException as error:
            if preparation_pool is not None:stop_preparing()
            O.record_runtime_error(self.run,error,'WIDTH_BATCH')
            if hasattr(executor,'_stop_after_failure'):executor._stop_after_failure(error)
            self.state['stop']=dict(reason='RUNTIME_EXCEPTION',error_type=type(error).__name__,message=str(error),
                error_log='RUNTIME_ERRORS.jsonl',drain='WAIT_FOR_OWNED_RUNNING_STAGES_NO_NEW_PREPARATION',UTC=K.now())
            self.log(event='BATCH_EXCEPTION',error_type=type(error).__name__,message=str(error))
            raise
        finally:
            original_error=sys.exc_info()[1];cleanup_error=None
            actions=[] if preparation_pool is None else [('PREPARATION_DRAIN',lambda:preparation_pool.shutdown(wait=True))]
            actions.append(('EXECUTOR_DRAIN',executor.finish))
            for phase,close in actions:
                try:close()
                except BaseException as error:
                    O.record_runtime_error(self.run,error,phase)
                    cleanup_error=cleanup_error or error
            if original_error is not None:
                try:self.save()  # the preparation producer has now drained
                except BaseException as error:O.record_runtime_error(self.run,error,'FAILED_STATE_SAVE')
            if original_error is None and cleanup_error is not None:raise cleanup_error
        self.confirm_completed(context,executor,result)
        launched_now={m:executor.launched(m)for m in self.methods}
        self.state['q3_launched']=launched_now
        with(self.run/'INVOCATIONS.jsonl').open('a')as f:
            f.write(json.dumps(O.jsonable(dict(time=time.time(),requested_widths=self.widths,budget=budgets,
                launched=launched_now,
                stop={m:r['stop_reason']for m,r in result.items()},new_calls={m:r['new_calls']for m,r in result.items()})))+'\n')
        K.CLI.write(self.run/'ROUND_RESULT.json',K.CLI.slim(dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1',
            round=self.spec,methods=result,optimizer_source=K.CLI.head())))
        K.CLI.table(result,self.run/'PER_METHOD.csv')
        if str(self.run).startswith(str(ROOT/'docs/research')):
            K.CLI.write(self.run/'NEEDS.json',dict(candidates=K.CLI.candidate_rows(context)))
            from publish_h4_search_run import prepare as publish_search
            publish_search(ROOT,self.run,self.run.parent/'SEARCH_PUBLICATION.json',run_id=self.spec['round_id'])
        priced={c['configuration_id']:c['prices'] for c in context['candidates']
                if c.get('ready') and c.get('prices')}
        for p in self.state['points']:
            if p['status']not in('REGISTERED','EVALUATED'):continue
            prices=priced.get(p.get('evaluation_configuration_id',p.get('configuration_id')),[])
            if (search.get('review_search_order') is True and len(prices)==26
                    and all(type(v) in (int,Fraction) and v>=0 for v in prices)):
                p['own_price_floor']=sum(prices)
            rows=result.get(p['method'],{}).get('by_configuration',[])
            cid = p.get('evaluation_configuration_id', p.get('configuration_id'))
            if getattr(self, 'scaled_shots', False):
                matches = [v for v in rows if cid and cid in (v['configuration_id'], v.get('source_configuration_id'))]
                b = min(matches, key=lambda v: (v['best'] is None,
                    v['best']['cost'] if v['best'] else 0, v['M']), default=None)
                if b:
                    p['best_M'] = b['M'] if b['best'] else None
                    b = dict(b, promising=sum(v['promising'] for v in matches))
            else:
                b = next((v for v in rows if v['configuration_id'] == cid), None)
            if b:
                adopted=b.get('best');best=b.get('confirmed_all_best')or adopted
                p.update(status='EVALUATED',best=None if best is None else best['cost'],
                    rule_adopted_best=None if adopted is None else adopted['cost'],
                    confirmed_all_best=None if best is None else best['cost'],
                    best_N_X=None if best is None else O.split(best['vector'])[0],
                    best_N_Y=None if best is None else O.split(best['vector'])[1],
                    best_MSE_upper=None if best is None else best.get('MSE_upper'),
                    promising=b.get('promising'))
        return result

    def loop(self):
        previous=self.state.get('stop',{}).get('reason')
        transient={'USER_REQUESTED_STOP','REQUESTED_WIDTH_METADATA_ONLY','WAITING_FOR_INPUTS','SCOPE_HELD_INPUTS',
                   'SCOPE_HELD_Q3','MAX_CYCLES','BOUNDED_SCHEDULE_PLAN_ONLY','DISCRETE_SCHEDULE_PLAN_ONLY'}
        trial = globals().get('FIXED_RUN_WORKER_TRIAL') or {}
        recovery = trial.get('recovered_failure')
        if recovery is None and trial.get('resumed_ast_failure'):
            recovery = dict(kind='ast', invocation=trial['resumed_ast_failure'])
        if previous == 'RUNTIME_EXCEPTION' and recovery and not self.a.plan_only:
            kind, invocation = recovery['kind'], recovery['invocation']
            preserved = O.load_json(self.run/'stop-history'/(kind+'-'+invocation+'.json'))
            if preserved != self.state['stop']:
                raise ValueError('Reconciled controller failure differs from its saved stop')
            self.state.pop('stop')
            self.log(event='RESUMED_AFTER_'+kind.upper()+'_FAILURE', invocation_recovered=invocation)
            previous = None
        if previous and previous not in transient:
            self.log(event='PRESERVED_STOP',reason=previous);return
        if previous:self.state.pop('stop')
        bounded={method:W.kappa_bounds(self.wide_for(method)) for method in self.widths} if self.a.plan_only else {}
        grids={method:W.kappa_values(self.wide_for(method)) for method in self.widths} if self.a.plan_only else {}
        if self.a.plan_only and (any(bounded.values()) or any(grids.values())):
            # Native physical schedule selection only. Missing family/input
            # preparation remains explicit; no Completion, Executor or worker.
            self.state['cycle'] += 1
            proposals, report = self.plan()
            self.state['scope_plan'] = dict(rule=W.RULE,
                kappa_bounds=next(([str(v) for v in b] for b in bounded.values() if b),None),
                kappa_bounds_by_method={m:[str(v) for v in b] if b else None for m,b in bounded.items()},
                kappa_values_by_method={m:[str(v) for v in grid] if grid else None for m,grid in grids.items()},
                bounds_inclusive=[True, True],
                requested_L_D={m:self.problem['methods'][m].get('L_D_values') for m in self.methods},
                family_status={k:dict(L_D=v['L_D'],status=v['status'],reason=v.get('reason'))
                               for k,v in self.state['families'].items()},
                preparation_pending=[k for k,v in self.state['families'].items() if v['status']!='READY_METADATA'],
                scientific_calls=0, queue_operations=0, dry_check_calls=0)
            self.state['stop'] = dict(reason='DISCRETE_SCHEDULE_PLAN_ONLY' if any(grids.values())
                                     else 'BOUNDED_SCHEDULE_PLAN_ONLY',UTC=K.now())
            self.save()
            return
        end=None if self.a.max_cycles is None else self.state['cycle']+self.a.max_cycles
        auto_recheck = end is None and self.work_conserving_search and bool(
            self.spec.get('executor', {}).get('distributed'))
        recheck_delay = 15
        held_q3, unresolved = False, []
        while end is None or self.state['cycle']<end:
            if (self.run/'STOP_REQUESTED.json').exists():
                self.state.pop('recovery_wait', None)
                self.state['stop']=dict(reason='USER_REQUESTED_STOP',UTC=K.now());self.save();return
            self.state['cycle']+=1;started=time.monotonic();self.log(event='PLAN_START')
            self.state.pop('recovery_wait', None)
            distributed_prep=bool((self.spec.get('executor',{}).get('distributed')or{}).get('preparation'))
            overlap=((self.spec.get('overlap_ready_batch')is True or distributed_prep) and not self.a.plan_only and
                     self.spec.get('science_approved')is True)
            if not (distributed_prep and overlap):self.bootstrap_families()
            ready_families=sum(f['status']=='READY_METADATA' for f in self.state['families'].values())
            proposals,report=self.plan();self.save()
            self.log(event='PLAN_END',wall_seconds=time.monotonic()-started,proposals=len(proposals))
            started=time.monotonic();self.log(event='INPUTS_START')
            if not overlap:self.complete_points()
            self.save();self.log(event='INPUTS_END',wall_seconds=time.monotonic()-started,deferred_to_batch=overlap)
            if self.a.plan_only or self.spec.get('science_approved')is not True:
                self.state['stop']=dict(reason='REQUESTED_WIDTH_METADATA_ONLY',UTC=K.now());self.save();return
            started=time.monotonic();self.log(event='V2_BATCH_START')
            result=self.batch(prepare_during=True)if overlap else self.batch()
            self.save();self.log(event='V2_BATCH_END',wall_seconds=time.monotonic()-started,
                methods={m:dict(stop=r['stop_reason'],new_calls=r['new_calls'])for m,r in result.items()})
            failed={m:r['stop_reason']for m,r in result.items()if r['stop_reason']in
                    ('UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER','HOST_ELIGIBILITY_HOLD')}
            if failed:
                self.state['stop']=dict(reason='V2_RESULT_REQUIRES_REVIEW',methods=failed,UTC=K.now());self.save();return
            caps=self.spec.get('max_new_q3_per_method',{})
            launched=self.state.get('q3_launched',{})
            budget_stops = {m for m, r in result.items() if r['stop_reason'] == 'Q3_BUDGET_EXHAUSTED'}
            budget_remaining = {m for m in budget_stops
                                if type(caps.get(m)) is not int or launched.get(m, 0) < caps[m]}
            unresolved = next(iter(result.values()), {}).get('executor_metrics', {}).get('unresolved_q3_attempts', [])
            held_q3 = bool(unresolved) or any(r.get('prior_q3_holds') or
                any(h['status']=='UNKNOWN' for h in r.get('execution_holds', [])) for r in result.values())
            if (any(launched.values()) and
                    all(type(caps.get(m)) is int and launched[m] >= caps[m] for m in launched)):
                self.state['stop']=dict(reason='SCOPE_HELD_Q3' if held_q3 else 'Q3_BUDGET_EXHAUSTED',
                    UTC=K.now(),launched=launched,unresolved_q3_attempts=unresolved)
                self.save();return
            scaled = getattr(self, 'scaled_shots', False)
            # The next cycle raises the batch cap, bounded by the same round budget.
            if scaled:
                next_batch = [m for m in sorted(budget_remaining)
                              if type(batch_cap := result[m]['budget']['max_new_q3']) is int
                              and 0 < batch_cap <= launched.get(m, 0)]
                if next_batch:
                    self.log(event='Q3_BATCH_ADVANCE', methods=next_batch)
                    continue
            waiting_inputs=(any(p['status']=='WAITING_INPUTS'for p in self.state['points'])or
                     any(f['status']in('FAMILY_PREPARATION_PENDING','NEEDS_FAMILY_CONFIGURATION_SOURCE')for f in self.state['families'].values())or
                     any(r.get('input_requests') for r in result.values()))
            waiting = waiting_inputs or any(r.get('prior_q3_holds') for r in result.values())
            pending=any(p['status']in('COMPLETION','REGISTERED')or(p['status']=='EVALUATED'and p.get('promising'))for p in self.state['points'])
            if getattr(self,'rule_stop',False):
                if any(r['stop_reason']=='STEP_LIMIT' for r in result.values()):
                    self.state['stop']=dict(reason='SCOPE_HELD_Q3' if held_q3 else 'EXPLICIT_STEP_LIMIT',
                        UTC=K.now(),unresolved_q3_attempts=unresolved);self.save();return
                if getattr(self,'pf_more',False) or any(r['stop_reason']=='WORK_SLICE_YIELDED' for r in result.values()):
                    continue
                pending=any(p['status'] in ('COMPLETION','REGISTERED') for p in self.state['points'])
            pf_status_path = self.run/'PF_PREPARATION_STATUS.json'
            pf_status = O.load_json(pf_status_path) if pf_status_path.is_file() else {}
            pf_records = list(pf_status.get('records', {}).values()) or [pf_status]
            pf_held = any(row.get('status') in ('PREPARATION_INCOMPLETE', 'PREPARATION_WAITING',
                                               'PRIOR_OWNER_HOLD', 'PRICE_EVIDENCE_REQUIRED',
                                               'PRICE_ACTION_UNAVAILABLE', 'PRICE_RECHECK_AVAILABLE',
                                               'PRICE_OWNER_WAITING')
                          for row in pf_records)
            if (not proposals and not pending and ready_families ==
                    sum(f['status']=='READY_METADATA' for f in self.state['families'].values())):
                blocked=waiting or any(f['status']!='READY_METADATA' for f in self.state['families'].values()) or any(
                    p['status'] in ('HELD','PRICE_ACTION_UNAVAILABLE') for p in self.state['points']) or pf_held or any(
                    r.get('not_ready_candidates') for r in result.values())
                reason=('SCOPE_HELD_INPUTS' if blocked else 'REQUESTED_WIDTH_RULE_EXHAUSTED') if getattr(self,'rule_stop',False) else (
                    'WAITING_FOR_INPUTS'if waiting else'REQUESTED_WIDTH_RULE_EXHAUSTED')
                if held_q3 or (scaled and not blocked and (budget_remaining or
                        any(r['stop_reason'] == 'UNRESOLVED_SCALED_SCOPE' for r in result.values()))):
                    reason = 'SCOPE_HELD_Q3'
                elif scaled and not blocked and budget_stops:
                    reason = 'Q3_BUDGET_EXHAUSTED'
                waits = []
                if waiting_inputs or any(row.get('status') == 'PREPARATION_WAITING' for row in pf_records):
                    waits.append('PREPARATION_OR_INPUT')
                if any(row.get('status') in ('PRICE_RECHECK_AVAILABLE', 'PRICE_OWNER_WAITING')
                       for row in pf_records):
                    waits.append('PF_PRICE_RECHECK')
                if unresolved or any(h.get('status') == 'IN_PROGRESS' for r in result.values()
                                     for h in r.get('held', ())):
                    waits.append('OWNED_ATTEMPT')
                if any(h.get('reason') == 'NO_MATCHING_ELIGIBLE_HOST' or h.get('status') == 'HOST_ELIGIBILITY_HOLD'
                       for r in result.values() for h in r.get('held', ())):
                    waits.append('ELIGIBLE_HOST')
                if auto_recheck and waits and reason in ('WAITING_FOR_INPUTS', 'SCOPE_HELD_INPUTS', 'SCOPE_HELD_Q3'):
                    self.wait_for_recovery(waits, recheck_delay)
                    recheck_delay = min(recheck_delay * 2, 60)
                    continue
                self.state['stop']=dict(reason=reason,
                    UTC=K.now(),methods={m:r['stop_reason']for m,r in result.items()},
                    unresolved_q3_attempts=unresolved);self.save();return
            recheck_delay = 15
        self.state['stop']=dict(reason='SCOPE_HELD_Q3' if held_q3 else 'MAX_CYCLES',
            UTC=K.now(),unresolved_q3_attempts=unresolved);self.save()

    def wait_for_recovery(self, reasons, seconds):
        self.state['recovery_wait'] = dict(reasons=reasons, recheck_epoch=time.time()+seconds)
        self.save()
        self.log(event='WAITING_FOR_RECOVERY', **self.state['recovery_wait'])
        deadline = time.monotonic() + seconds
        stop = self.run/'STOP_REQUESTED.json'
        while not stop.exists() and (remaining := deadline - time.monotonic()) > 0:
            time.sleep(min(1, remaining))


@serialize_legacy_ast()
def main(argv=None, *, driver_factory=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True);ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max-cycles', type=int);ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--prepare-readiness', action='store_true', help='plan-only: delta saved history, chosen requests, server2 dry-check and immutable remote delivery')
    ap.add_argument('--preparation-cache', type=Path, help='local disposable cache for --prepare-readiness')
    ap.add_argument('--host-readiness', type=Path, help='saved host READINESS descriptors to join during --prepare-readiness')
    a = ap.parse_args(argv);a.problem = a.problem.resolve()
    if a.prepare_readiness:
        from trottertracks.rpe_plan_prepare import prepare
        return prepare(a, code_root=ROOT, cli=K.CLI)
    problem=O.load_json(a.problem);spec=O.load_json(a.round)
    if a.max_cycles is None and not spec.get('search_to_rule_stop'):a.max_cycles=200
    declared=W.requested_widths(problem,spec)
    if driver_factory is None and declared is None and not spec.get('search_to_rule_stop') and any(m in('FULL_PF','TRUNCATED_PF')for m in spec.get('methods',[])):
        # PF selection and shot/Q3 remain owned by the existing method CLI.
        # This is the same public entry, not another PF driver/optimizer.
        from types import SimpleNamespace
        a.run_dir.mkdir(parents=True,exist_ok=True)
        K.CLI.round_mode(a,problem,[])
        result=O.load_json(a.run_dir/'ROUND_RESULT.json')
        return SimpleNamespace(run=a.run_dir,state=dict(stop=dict(reason='METHOD_ROUND_COMPLETE',
            methods={m:r['stop_reason']for m,r in result['methods'].items()})))
    factory=driver_factory or (WidthDriver if declared is not None or spec.get('search_to_rule_stop') else WideDriver)
    d = factory(a);d.log(event='DRIVER_START', plan_only=a.plan_only)
    try:d.loop()
    except KeyboardInterrupt:
        # Executor cleanup drains owned work. Scientific UNKNOWN/STOP evidence
        # remains in job artifacts; only this operational stop is resumable.
        d.state['stop']=dict(reason='USER_REQUESTED_STOP',UTC=K.now())
        d.save();d.log(event='USER_REQUESTED_STOP')
    except BaseException as error:
        O.record_runtime_error(a.run_dir,error,'DRIVER')
        d.state['stop']=dict(reason='RUNTIME_EXCEPTION',error_type=type(error).__name__,message=str(error),
            error_log='RUNTIME_ERRORS.jsonl',UTC=K.now())
        try:d.save();d.log(event='DRIVER_ERROR',stop=d.state['stop'])
        except BaseException as save_error:O.record_runtime_error(a.run_dir,save_error,'FAILED_DRIVER_STATE_SAVE')
        raise
    finally:
        manager = getattr(d, 'run_pbs_manager', None)
        if manager is not None and manager.detached:
            manager.close()
    if (a.run_dir/'STOP_REQUESTED.json').exists() and d.state.get('stop',{}).get('reason') not in (
            'V2_RESULT_REQUIRES_REVIEW','RUNTIME_EXCEPTION'):
        d.state['stop']=dict(reason='USER_REQUESTED_STOP',UTC=K.now());d.save()
    d.log(event='DRIVER_END', stop=d.state.get('stop'))
    termination_reasons = ('REQUESTED_WIDTH_RULE_EXHAUSTED', 'EXPLICIT_STEP_LIMIT', 'Q3_BUDGET_EXHAUSTED')
    if spec.get('search_to_rule_stop') and d.state.get('stop', {}).get('reason') in termination_reasons:
        K.CLI.write(a.run_dir/'SEARCH_TERMINATION.json', d.state['stop'])
    print(json.dumps(O.jsonable(d.state.get('stop')), default=str)[:2000])
    return d


if __name__ == '__main__':
    main()
