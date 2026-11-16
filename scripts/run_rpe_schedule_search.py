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
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402
from trottertracks import rpe_schedule_search as W  # noqa: E402
import run_rpe_kappa_neighbourhood as K  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def read_loc(loc):
    import subprocess
    x = json.loads(subprocess.check_output(['git', 'show', loc['commit'] + ':' + loc['path']], cwd=ROOT))
    for k in loc.get('selector', []):
        x = x[k]
    return x


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
        transient = {'PLAN_ONLY_FIRST_DECISION_WRITTEN', 'SCIENCE_NOT_APPROVED_FIRST_DECISION_WRITTEN',
                     'PLAN_ONLY_NOTHING_TO_PREPARE', 'MAX_CYCLES'}
        if previous_stop and previous_stop not in transient:
            self.log(event='PRESERVED_STOP', reason=previous_stop);return
        if previous_stop:self.state.pop('stop')
        while self.state['cycle'] < self.a.max_cycles:
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
            fatal = {'UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER',
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
            if res['stop_reason'] in ('UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER'):
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
    def __init__(self, a):
        self.a = a; self.run = a.run_dir.resolve(); self.run.mkdir(parents=True, exist_ok=True)
        self.spec = O.load_json(a.round)
        self.problem = K.CLI.price_overrides(K.CLI.scoped(O.load_json(a.problem), self.spec.get('method_overrides')), self.spec)
        self.widths = W.requested_widths(self.problem, self.spec)
        if self.widths is None: raise ValueError('Explicit requested_widths required')
        self.methods = list(self.widths)
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
        reader, inputs, catalog = O.load_catalog(ROOT,self.problem)
        self.catalog_facts=O.hamiltonian_facts(self.problem)
        O.check_comparison(self.problem,self.catalog_facts,inputs['comparison'])
        return reader,catalog

    def generate(self, method, base, changes):
        g=self.problem['methods'][method]['generation']
        result=O.run_generator(ROOT,g,[dict(item_id='WIDTH_FAMILY',base=base,changes=changes)])
        return result['items'][0]

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
            for ld,loc in bases:
                base=reader.read(loc);base=base.get('configuration',base)
                for formula in (setting.get('generation')or{}).get('formulas',['2nd']):
                    for width in widths:
                        key=W.width_family_key(method,ld,formula,width['raw'],base['spec']['finite_K'])
                        family=dict(method=method,L_D=ld,formula=formula,K=base['spec']['finite_K'],
                            requested_delta_pf=width['raw'],delta_pf=O.enc(KS._frac(width['raw'])),
                            base_source=loc,status='MAPPER_PENDING')
                        self.state['families'][key]=family
                        started=time.monotonic();self.log(event='SEED_FAMILY_START',family=key)
                        try:
                            ratio=KS._frac(width['raw'])/KS._frac(base['spec']['h'])
                            made=self.generate(method,loc,dict(formula=formula,delta_over_h=str(ratio)))
                            if made['status']!='GENERATED':
                                raise ValueError(made.get('reason')or made['status'])
                            cfg=made['configuration'];s=cfg['spec']
                            if KS._frac(s['delta_pf'])!=KS._frac(width['raw']):
                                raise ValueError('MAPPER_DOES_NOT_PRESERVE_EXACT_REQUESTED_WIDTH')
                            if KS._frac(s['h'])!=KS._frac(base['spec']['h']) or s['M']!=base['spec']['M']:
                                raise ValueError('MAPPER_CHANGED_FIXED_RPE_TIME_GRID')
                            rows=KS.constants(cfg); k=KS._frac(s['kappa']['value'])
                            if KS.R_vector(rows,k)!=[v['R_outer']for v in sorted(cfg['rounds'],key=lambda r:r['m'])]:
                                raise ValueError('SCHEDULE_CONSTANTS_DO_NOT_REPRODUCE_CANONICAL_MAPPER')
                            family.update(status='READY_METADATA',rows=KS.encode_rows(rows),configuration=cfg)
                        except (ValueError,KeyError,TypeError) as exc:
                            family.update(status='UNSUPPORTED_MAPPER',reason=str(exc))
                        self.log(event='SEED_FAMILY_END',family=key,status=family['status'],
                            reason=family.get('reason'),wall_seconds=time.monotonic()-started)
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
            same=next((v for v in self.state['points']if v['family']==family and Fraction(v['kappa'])==k),None)
            if same is None:
                same=dict(family=family,method=p['method'],L_D=p['L_D'],kappa=str(k),status='EVALUATED',
                    configuration_id=p['configuration_id'],best=None,saved_point_ids=[])
                self.state['points'].append(same)
            same['saved_point_ids'].append(p['point_id'])
            if p.get('status')=='FEASIBLE' and p.get('conditional_cost')is not None:
                cost=O.frac(p['conditional_cost'])
                if same['best']is None or cost<O.frac(same['best']):same['best']=O.enc(cost)
        self.state['seeded']=dict(UTC=K.now(),mapper='METHOD_SPECIFIC_EXISTING_GENERATOR',
            old_single_width_state_untouched=True)
        self.save()

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
                f['reason']=result.get('reason',result['status']);continue
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

    def point(self, family, kappa):
        return next((p for p in self.state['points']if p['family']==family and Fraction(p['kappa'])==Fraction(kappa)),None)

    def plan(self):
        families=self.families();points=[]
        for p in self.state['points']:
            status=('FEASIBLE'if p.get('best')is not None else 'NO_FEASIBLE_FOUND'if p['status']=='EVALUATED'
                    else 'UNAVAILABLE'if p['status']in('HELD','PRUNED')else'PENDING')
            points.append(dict(family=p['family'],kappa=Fraction(p['kappa']),status=status,
                best=None if p.get('best')is None else O.frac(p['best']),min_lower_over_tau=p.get('min_lower_over_tau')))
        proposals,decisions,report=W.plan_wide(families,points,self.wide)
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

    def complete_points(self):
        comp=self.completion_api()
        if hasattr(comp,'ingest_price_results'):comp.ingest_price_results()
        for p in self.state['points']:
            if p['status']not in('COMPLETION','WAITING_INPUTS'):continue
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
            if record['status']=='READY_FOR_SEARCH':
                try:self.check_completed_configuration(p,record)
                except (ValueError,KeyError,TypeError) as exc:
                    p.update(status='HELD',reason='COMPLETED_CONFIGURATION_MISMATCH:'+str(exc));continue
                p.update(status='REGISTERED',configuration_id=record['configuration_id'])
            elif record['status'].startswith('STOP') or 'NUMERICAL_GATE' in str(record.get('reason','')):
                p.update(status='HELD',reason=record.get('reason',record['status']))
            else:
                p.update(status='WAITING_INPUTS',reason=record.get('reason',record['status']))

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
                         preparation_dir=self.run/'preparation')

    def batch(self, prepare_during=False):
        # Same V2/Executor and risk contract as the old ordinary round entry.
        # Completion selects no new designs here: the width planner already did.
        problem=deepcopy(self.problem)
        for method, widths in self.widths.items():
            problem['methods'][method]['requested_delta_pf_values']=[O.enc(KS._frac(w['raw']))for w in widths]
        context=self.context(problem);comp=self.completion_api()
        routes=K.CLI.merged_routes(problem,self.spec)
        def install_ready(current_routes):
            accepted={p.get('completion',{}).get('key')for p in self.state['points']if p['status']in('REGISTERED','EVALUATED')}
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
            if maximum not in(None,'UNLIMITED'):budgets[method]=min(budget,maximum)
        search=dict(problem['search'],**self.spec.get('search_overrides',{}))
        # Existing V2 switch: kappa schedules are chosen by this outer width
        # planner, not by a second inner multiplier-neighbour chooser.
        search.setdefault('kappa_neighbourhood',dict(outer_planner=W.RULE))
        # The width planner still chooses designs. Continuation may refresh and
        # install its existing waiting records, without a second design chooser.
        active_spec=deepcopy(self.spec);active_spec['max_new_q3_per_method']=budgets
        active_spec['executor']=dict(options,prior_run_dirs=list(options.get('prior_run_dirs',[]))+[str(self.run)])
        active_spec['candidate_completion']=dict(self.spec.get('candidate_completion')or{},max_attempted_per_method=0)
        executor,optimize=K.CLI.make_executor(ROOT,self.run,routes,context['expected'],problem,active_spec)
        preparation_pool=None
        try:
            interim=None
            if prepare_during:
                # Snapshot/install happens before starting preparation. While
                # it owns Completion state, Q3 only reads this frozen context.
                from concurrent.futures import ThreadPoolExecutor
                preparation_pool=ThreadPoolExecutor(max_workers=1)
                pending=preparation_pool.submit(self.complete_points)
                if any(c['ready']for c in context['candidates']):
                    with K.CLI.round_phase(self.run,'WIDTH_READY_DURING_PREPARATION'):
                        interim=optimize(ROOT,dict(problem,search=search),context=context,methods=self.methods,
                            initial=self.spec.get('initial','all'),max_new=budgets,executor=executor,science_approved=True)
                pending.result()
                # No other thread now changes Completion or its registry.
                routes=install_ready(routes);executor.routes=routes;executor.expected=context['expected']
                O.apply_published(context['candidates'],routes)
            fatal={'UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER'}
            if interim and any(r['stop_reason']in fatal for r in interim.values()):result=interim
            else:
                with K.CLI.round_phase(self.run,'WIDTH_SEARCH_AND_Q3'):
                    result=optimize(ROOT,dict(problem,search=search),context=context,methods=self.methods,
                        initial=self.spec.get('initial','all'),max_new=budgets,executor=executor,
                        science_approved=self.spec.get('science_approved')is True)
            if not any(r['stop_reason']in fatal for r in result.values())and any(r.get('input_requests')for r in result.values()):
                from trottertracks.pr_candidate_completion import continue_selected_preparation
                with K.CLI.round_phase(self.run,'WIDTH_PREPARATION_CONTINUATION'):
                    result=continue_selected_preparation(ROOT,dict(problem,search=search),context,
                        active_spec,self.run,comp,executor,result,optimizer=optimize)
        finally:
            if preparation_pool is not None:preparation_pool.shutdown(wait=True)
            executor.finish()
        with(self.run/'INVOCATIONS.jsonl').open('a')as f:
            f.write(json.dumps(O.jsonable(dict(time=time.time(),requested_widths=self.widths,budget=budgets,
                launched={m:executor.launched(m)for m in self.methods},
                stop={m:r['stop_reason']for m,r in result.items()},new_calls={m:r['new_calls']for m,r in result.items()})))+'\n')
        K.CLI.write(self.run/'ROUND_RESULT.json',K.CLI.slim(dict(schema='RPE_OPTIMIZER_ROUND_RESULT_V1',
            round=self.spec,methods=result,optimizer_source=K.CLI.head())))
        K.CLI.table(result,self.run/'PER_METHOD.csv')
        for p in self.state['points']:
            if p['status']not in('REGISTERED','EVALUATED'):continue
            rows=result.get(p['method'],{}).get('by_configuration',[])
            b=next((v for v in rows if v['configuration_id']==p.get('evaluation_configuration_id',p.get('configuration_id'))),None)
            if b:
                best=b.get('best');p.update(status='EVALUATED',best=None if best is None else best['cost'],
                    best_N_X=None if best is None else O.split(best['vector'])[0],
                    best_N_Y=None if best is None else O.split(best['vector'])[1],
                    best_MSE_upper=None if best is None else best.get('MSE_upper'),
                    promising=b.get('promising'))
        return result

    def loop(self):
        previous=self.state.get('stop',{}).get('reason')
        transient={'REQUESTED_WIDTH_METADATA_ONLY','WAITING_FOR_INPUTS','MAX_CYCLES'}
        if previous and previous not in transient:
            self.log(event='PRESERVED_STOP',reason=previous);return
        if previous:self.state.pop('stop')
        end=self.state['cycle']+self.a.max_cycles
        while self.state['cycle']<end:
            self.state['cycle']+=1;started=time.monotonic();self.log(event='PLAN_START')
            self.bootstrap_families();proposals,report=self.plan();self.save()
            self.log(event='PLAN_END',wall_seconds=time.monotonic()-started,proposals=len(proposals))
            started=time.monotonic();self.log(event='INPUTS_START')
            overlap=(self.spec.get('overlap_ready_batch')is True and not self.a.plan_only and
                     self.spec.get('science_approved')is True)
            if not overlap:self.complete_points()
            self.save();self.log(event='INPUTS_END',wall_seconds=time.monotonic()-started,deferred_to_batch=overlap)
            if self.a.plan_only or self.spec.get('science_approved')is not True:
                self.state['stop']=dict(reason='REQUESTED_WIDTH_METADATA_ONLY',UTC=K.now());self.save();return
            started=time.monotonic();self.log(event='V2_BATCH_START')
            result=self.batch(prepare_during=True)if overlap else self.batch()
            self.save();self.log(event='V2_BATCH_END',wall_seconds=time.monotonic()-started,
                methods={m:dict(stop=r['stop_reason'],new_calls=r['new_calls'])for m,r in result.items()})
            failed={m:r['stop_reason']for m,r in result.items()if r['stop_reason']in
                    ('UNKNOWN_RESULT','STALE_JOB_NEEDS_REVIEW','NO_ROUTE_ADAPTER')}
            if failed:
                self.state['stop']=dict(reason='V2_RESULT_REQUIRES_REVIEW',methods=failed,UTC=K.now());self.save();return
            waiting=(any(p['status']=='WAITING_INPUTS'for p in self.state['points'])or
                     any(f['status']in('FAMILY_PREPARATION_PENDING','NEEDS_FAMILY_CONFIGURATION_SOURCE')for f in self.state['families'].values()))
            pending=any(p['status']=='REGISTERED'or(p['status']=='EVALUATED'and p.get('promising'))for p in self.state['points'])
            if not proposals and not pending:
                self.state['stop']=dict(reason='WAITING_FOR_INPUTS'if waiting else'REQUESTED_WIDTH_RULE_EXHAUSTED',
                    UTC=K.now(),methods={m:r['stop_reason']for m,r in result.items()});self.save();return
        self.state['stop']=dict(reason='MAX_CYCLES',UTC=K.now());self.save()


def main(argv=None, *, driver_factory=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True);ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max-cycles', type=int, default=200);ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--prepare-readiness', action='store_true', help='plan-only: delta saved history, chosen requests, server2 dry-check and immutable remote delivery')
    ap.add_argument('--preparation-cache', type=Path, help='local disposable cache for --prepare-readiness')
    a = ap.parse_args(argv);a.problem = a.problem.resolve()
    if a.prepare_readiness:
        from trottertracks.rpe_plan_prepare import prepare
        return prepare(a, code_root=ROOT, cli=K.CLI)
    problem=O.load_json(a.problem);spec=O.load_json(a.round)
    declared=W.requested_widths(problem,spec)
    if driver_factory is None and declared is None and any(m in('FULL_PF','TRUNCATED_PF')for m in spec.get('methods',[])):
        # PF selection and shot/Q3 remain owned by the existing method CLI.
        # This is the same public entry, not another PF driver/optimizer.
        from types import SimpleNamespace
        a.run_dir.mkdir(parents=True,exist_ok=True)
        K.CLI.round_mode(a,problem,[])
        result=O.load_json(a.run_dir/'ROUND_RESULT.json')
        return SimpleNamespace(run=a.run_dir,state=dict(stop=dict(reason='METHOD_ROUND_COMPLETE',
            methods={m:r['stop_reason']for m,r in result['methods'].items()})))
    factory=driver_factory or (WidthDriver if declared is not None else WideDriver)
    d = factory(a);d.log(event='DRIVER_START', plan_only=a.plan_only);d.loop();d.log(event='DRIVER_END', stop=d.state.get('stop'))
    print(json.dumps(O.jsonable(d.state.get('stop')), default=str)[:2000])
    return d


if __name__ == '__main__':
    main()
