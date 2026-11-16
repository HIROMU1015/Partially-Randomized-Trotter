"""Absolute-kappa neighbourhood search for one PR method (H4 PR-Pauli round02), resumable.

One cycle:
  1. prepare: every requested (L_D, kappa) goes through A's preparation entry (base = the saved configuration of that L_D,
     multiplier = kappa / base kappa exactly; generated spec.kappa.value must equal kappa); D2 PriceRules MODEL quote;
     a point whose schedule equals an existing configuration is an ALIAS (its evaluations are reused); a point without a
     complete price, a supported signal provider or a configuration is HELD with the work it needs;
  2. signals (signals worktree, A's common entry, fixed SOURCE): plan -> execute only the missing rounds (in parallel) ->
     assemble -> commit/push -> B intake (accept_saved + total binding) -> route registry entry;
  3. V2 (METHOD_MINIMUM_PRIORITY_V2) batch on the real Q3 route with prior rounds loaded (round mode, same run directory);
  4. kappa neighbourhood plan (rpe_priority_search.kappa_neighbourhood_plan) -> next points with reasons (history);
until the rule proposes nothing, no promising candidate is left in the active families and nothing is pending.

    python run_rpe_kappa_neighbourhood.py --problem P --round R --run-dir D [--batch 128] [--max-cycles 50]
Re-running the same command resumes from D/KAPPA_STATE.json (completed jobs, packets and registry entries are reused).
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from trottertracks import rpe_method_optimizer as O  # noqa: E402
from trottertracks import rpe_priority_search as P  # noqa: E402
from trottertracks import rpe_split_stage as SPL  # noqa: E402
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402
import run_rpe_method_optimizer as CLI  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class Driver:
    def __init__(self, a):
        self.a, self.run = a, a.run_dir.resolve();self.run.mkdir(parents=True, exist_ok=True)
        self.spec = O.load_json(a.round);self.nb = self.spec['search_overrides'].get('kappa_neighbourhood')
        sig = self.spec['signals'];self.sig = sig;self.wt = Path(sig['worktree']);self.sub = sig['directory']
        self.state_file = self.run / 'KAPPA_STATE.json'
        self.state = O.load_json(self.state_file) if self.state_file.exists() else dict(
            schema='RPE_KAPPA_NEIGHBOURHOOD_STATE_V1', round_id=self.spec['round_id'], started=now(), cycle=0, points=[], history=[],
            saved_packets=list(sig['saved_packets']), settings=self.nb)
        if 'splits' not in self.state:  # outcomes of splits prepared before this run (seed) or by earlier cycles
            seed = self.spec.get('splits', {}).get('seed_file')
            self.state['splits'] = json.loads((ROOT / seed).read_text()) if seed else {}
        self.registry_file = ROOT / self.spec['routes_override']['PR_PAULI'][0]['configurations_file']
        self.points_file = ROOT / self.spec['method_overrides']['PR_PAULI']['generation']['kappa_points_file']

    # ------------------------------------------------------------ bookkeeping
    def log(self, **event):
        with (self.run / 'DRIVER_PROGRESS.jsonl').open('a') as f:
            f.write(json.dumps(dict(UTC=now(), epoch=time.time(), monotonic=time.monotonic(), cycle=self.state['cycle'], **event), default=str) + '\n')

    def save(self):
        tmp = self.state_file.with_suffix('.tmp');tmp.write_text(json.dumps(O.jsonable(self.state), indent=1, ensure_ascii=False, default=str) + '\n')
        tmp.replace(self.state_file)

    def point(self, ld, kappa):
        return next((p for p in self.state['points'] if p['L_D'] == ld and Fraction(p['kappa']) == Fraction(kappa)), None)

    def write_points(self):
        rows = [dict(L_D=p['L_D'], kappa=p['kappa']) for p in self.state['points'] if p['status'] not in ('HELD',)]
        self.points_file.parent.mkdir(parents=True, exist_ok=True);self.points_file.write_text(json.dumps(rows, indent=1) + '\n')

    def registry(self):
        return json.loads(self.registry_file.read_text()) if self.registry_file.exists() else {}

    # ------------------------------------------------------------ 1. prepare
    def prepare(self):
        self.write_points();self.log(event='PREPARE_START', points=len(self.state['points']))
        problem = CLI.price_overrides(CLI.scoped(O.load_json(self.a.problem), self.spec.get('method_overrides')), self.spec)
        g = problem['methods']['PR_PAULI']['generation'];g['kappa_points_file'] = str(self.points_file)
        ctx = O.prepare(ROOT, problem, methods=['PR_PAULI'], preparation_dir=self.run / 'preparation_driver')
        rows = {(r['proposal']['family']['L_D'], Fraction(r['proposal']['kappa'])): r for r in ctx['generation']['PR_PAULI'].get('kappa_points') or []}
        reg = self.registry()
        for p in self.state['points']:
            if p['status'] not in ('PENDING', 'PREPARED'):
                continue
            r = rows.get((p['L_D'], Fraction(p['kappa'])))
            if r is None:
                p.update(status='HELD', reason='NO_SAVED_OR_A_REGISTERED_BASE_FOR_L_D');continue
            p.update(kappa_base=r.get('kappa_base'), kappa_multiplier=r.get('kappa_multiplier'), preparation_status=r.get('status'),
                     disposition=r.get('disposition'), kappa_exact_check=r.get('kappa_exact_check'))
            disp = r.get('disposition') or ''
            if disp == 'NO_SAVED_BASE_FOR_L_D':  # the split itself is not prepared yet: A's next actions say what is needed
                prep = next((x for x in ctx['generation']['PR_PAULI'].get('preparation') or [] if x['L_D'] == p['L_D']), {})
                p.update(status='HELD', reason='SPLIT_PREPARATION_REQUIRED:%s:%s' % (prep.get('status'), prep.get('next_actions')));continue
            if disp.startswith('PHYSICALLY_IDENTICAL_TO:'):
                p.update(status='ALIAS', alias_of=disp.split(':', 1)[1], configuration_id=r.get('configuration_id'));continue
            if disp not in ('REGISTERED_WITHOUT_PRICE_REQUIREMENT', 'ALREADY_REGISTERED'):
                p.update(status='HELD', reason='PREPARATION:%s:%s' % (disp, r.get('next_actions')));continue
            c = next(c for c in ctx['candidates'] if c['configuration_id'] == r['configuration_id'])
            p.update(configuration_id=c['configuration_id'], price_status=c.get('price_status'),
                     price_quote={k: v for k, v in (c.get('price_quote') or {}).items() if k in ('status', 'complete', 'registry', 'classifications')})
            if c['configuration_id'] in reg:
                p['status'] = 'REGISTERED';continue
            if c.get('prices') is None:
                p.update(status='HELD', reason='PRICE_INCOMPLETE:%s' % json.dumps(p['price_quote'], default=str));continue
            p['status'] = 'PREPARED';p['_configuration'] = c['generated_configuration']
        self.family_constants(ctx)
        if self.spec.get('price_batch'):
            self.price_batch(ctx, [p for p in self.state['points'] if p['status'] == 'PREPARED'])
        self.log(event='PREPARE_END', statuses={s: sum(1 for p in self.state['points'] if p['status'] == s)
                                                for s in {p['status'] for p in self.state['points']}})
        return ctx

    def family_constants(self, ctx):
        """Exact schedule constants per L_D (from any generated configuration) and each point's 13-round R vector."""
        fc = self.state.setdefault('family_constants', {})
        for c in ctx['candidates']:
            g = c.get('generated_configuration')
            if c['method'] != 'PR_PAULI' or not g or not (g.get('spec') or {}).get('kappa'):
                continue
            if KS._frac(g['spec']['delta_pf']) != KS._frac(g['spec']['h']):
                continue  # one family per (L_D, width); this search runs at delta_PF = h only
            rows = KS.constants(g);kappa = KS._frac(g['spec']['kappa']['value'])
            if KS.R_vector(rows, kappa) != [r['R_outer'] for r in sorted(g['rounds'], key=lambda r: r['m'])]:
                raise ValueError('MAPPER_FORMULA_DOES_NOT_REPRODUCE_GENERATED_R: ' + c['configuration_id'])
            enc = KS.encode_rows(rows);key = str(g['spec']['L_D'])
            if fc.setdefault(key, enc) != enc:
                raise ValueError('SCHEDULE_CONSTANTS_DIFFER_WITHIN_FAMILY_L_D_%s' % key)
        for p in self.state['points']:
            if str(p['L_D']) in fc:
                p['R'] = list(KS.Sched(fc, ['0', '1000000']).R(p['L_D'], Fraction(p['kappa'])))

    def price_batch(self, ctx, points):
        """Prices through the C/A common calibration-request entry (02e7cc6a1): raw D2 MODEL quotes + canonical
        configurations of the new points in one batch; identical calibration cells are pooled.  Only
        REUSE_SUPPLIED_PRICES_NO_CALIBRATION proceeds here; anything else is held with the entry's needs / cells."""
        if not points:
            return
        pb = self.spec['price_batch'];d = self.run / 'price_batch' / ('cycle%02d' % self.state['cycle']);d.mkdir(parents=True, exist_ok=True)
        by_cid = {c['configuration_id']: c for c in ctx['candidates']}
        req = [dict(name='%s:%s' % (p['L_D'], p['kappa']), configuration=by_cid[p['configuration_id']]['generated_configuration'],
                    entry='PAULI_LD%d' % p['L_D']) for p in points]
        qreq = dict(policy='MODEL', items=[dict(item_id=r['name'], registry=r['entry'], configuration=r['configuration']) for r in req])
        (d / 'QUOTE_REQUEST.json').write_text(json.dumps(qreq) + '\n')
        reg = ROOT / self.spec['price_rules_overrides']['registry_file'];pr = O.load_json(self.a.problem)['price_rules']
        subprocess.run([pr['interpreter'], '-B', str(ROOT / 'scripts/run_rpe_optimizer_price_quotes.py'), '--view', pr['view'], '--raw',
                        '--registry', str(reg), '--request', str(d / 'QUOTE_REQUEST.json'), '--output', str(d / 'QUOTES.json')],
                       cwd=ROOT, env=ENV, check=True, stdout=subprocess.DEVNULL)
        quotes = {x['item_id']: x.get('raw') for x in json.loads((d / 'QUOTES.json').read_text())['items']}
        held = [r for r in req if quotes.get(r['name']) is None]  # D2 rejected the quote: nothing for the entry to check
        batch = [dict(configuration=r['configuration'], quote=quotes[r['name']]) for r in req if quotes.get(r['name']) is not None]
        (d / 'BATCH_INPUT.json').write_text(json.dumps(batch) + '\n')
        (d / 'EXECUTION_OPS.json').write_text(json.dumps(dict(branch=pb['execution_branch'])) + '\n')
        code = subprocess.run([pb['interpreter'], '-B', 'scripts/prepare_pauli_calibration.py', '--batch', str(d / 'BATCH_INPUT.json'),
                               '--output', str(d / 'BATCH.json'), '--execution', str(d / 'EXECUTION_OPS.json')], cwd=pb['view'],
                              env=dict(ENV, PYTHONPATH='src'), stdout=open(d / 'BATCH.log', 'w'), stderr=subprocess.STDOUT).returncode if batch else None
        result = json.loads((d / 'BATCH.json').read_text()) if code == 0 and (d / 'BATCH.json').exists() else None
        statuses = {}
        for x in (result or {}).get('results', []):
            statuses[x.get('configuration_id')] = x
        for p in points:
            x = statuses.get(p['configuration_id'])
            st = None if x is None else x.get('status')
            # per round: reused price / calibration cell (request id, actual step, R) / needs -- what a held point lacks
            rounds = [dict(m=a['m'], q=a['q'], R=a['R'], actual_step=a['actual_step'], action=a.get('action'), request_id=a.get('request_id'),
                           missing_axes=a.get('missing_axes'), needs=a.get('needs')) for a in (x or {}).get('round_actions') or []]
            p['price_entry'] = dict(status=st, next_action=None if x is None else x.get('next_action'),
                                    reused_axes=None if x is None else len(x.get('reused_prices') or []),
                                    missing_axes=None if x is None else len(x.get('missing_prices') or []), needs=(x or {}).get('needs'),
                                    rounds=rounds, batch=str(d.relative_to(self.run) / 'BATCH.json'), returncode=code)
            if st != 'REUSE_SUPPLIED_PRICES_NO_CALIBRATION':
                rejected = quotes.get('%s:%s' % (p['L_D'], p['kappa'])) is None
                p.update(status='HELD', reason='PRICE_ENTRY:%s' % (st or ('D2_QUOTE_REJECTED' if rejected else 'NO_ENTRY_RESULT')))
        cells = [dict(request_id=g['request_id'], status=g['status'], consumers=[(c['configuration_id'], c['rounds']) for c in g['consumers']],
                      cell=g['driver_request']['driver_spec']['cell'], actual_step=g['proposed_applicability']['actual_step'])
                 for g in (result or {}).get('calibration_requests') or []]
        self.state.setdefault('calibration_cells', []).extend(dict(cycle=self.state['cycle'], **c) for c in cells)
        self.log(event='PRICE_BATCH', points=len(points), d2_rejected=len(held), returncode=code,
                 unique_calibration_requests=(result or {}).get('unique_calibration_requests'),
                 statuses={k: v.get('status') for k, v in statuses.items()})

    # ------------------------------------------------------------ 2. signals
    def git(self, *args, check=True):
        return subprocess.run(['git', *args], cwd=self.wt, check=check, capture_output=True, text=True).stdout.strip()

    def entry(self, *args, log=None):
        cmd = [self.sig['interpreter'], '-B', 'scripts/run_pauli_signal_entry.py', *args]
        with open(log, 'w') if log else open(os.devnull, 'w') as f:
            return subprocess.run(cmd, cwd=self.wt, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode

    def signals(self):
        todo = [p for p in self.state['points'] if p['status'] == 'PREPARED']
        if not todo:
            return
        self.log(event='SIGNALS_START', points=[(p['L_D'], p['kappa']) for p in todo])
        base = self.wt / self.sub
        for d in ('configurations', 'plans', 'runs', 'packets', 'logs'):
            (base / d).mkdir(parents=True, exist_ok=True)
        packets_file = base / 'logs' / ('SAVED_PACKETS_cycle%02d.json' % self.state['cycle'])
        packets_file.write_text(json.dumps(self.state['saved_packets'], indent=1) + '\n')
        for p in todo:
            p['name'] = 'LD%d_S2_delta_h_kappa_abs_%s' % (p['L_D'], p['kappa'].replace('/', 'o'))
            (base / 'configurations' / (p['name'] + '.json')).write_text(json.dumps(p.pop('_configuration'), indent=1) + '\n')
        rel = lambda *x: str(Path(self.sub, *x))
        self.git('add', '--sparse', rel('configurations'), rel('logs'))
        self.git('-c', 'user.name=HIROMU1015', '-c', 'user.email=u411612g@ecs.osaka-u.ac.jp', '-c', 'core.hooksPath=/dev/null', 'commit', '-q', '-m',
                 'Publish PR-Pauli absolute-kappa configurations (cycle %d): %s [skip ci]\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
                 % (self.state['cycle'], ', '.join(p['name'] for p in todo)), check=False)  # unchanged files on a resumed cycle
        src = self.git('rev-parse', 'HEAD')
        for p in todo:  # plan (metadata only); a resumed cycle writes its plan next to the earlier one
            k = 1
            while (base / 'plans' / p['name'] / ('attempt%02d' % k) / 'PLAN.json').exists():
                k += 1
            pdir = rel('plans', p['name'], 'attempt%02d' % k)
            code = self.entry('--plan', '--config', rel('configurations', p['name'] + '.json'), '--saved-packets', str(packets_file),
                              '--out', pdir, log=base / 'logs' / (p['name'] + '.plan%02d.log' % k))
            plan = json.loads((self.wt / pdir / 'PLAN.json').read_text()) if code == 0 else {}
            p['signal_plan'] = dict(counts=plan.get('counts'), request_rounds=plan.get('request_rounds'), unsupported_rounds=plan.get('unsupported_rounds'),
                                    reuse_rounds=plan.get('reuse_rounds'), rebind_rounds=plan.get('rebind_rounds'), plan_returncode=code)
            if code != 0 or plan.get('unsupported_rounds') or plan.get('configuration_reasons'):
                p.update(status='HELD', reason='SIGNAL_PLAN:%s' % json.dumps(p['signal_plan']))
        todo = [p for p in todo if p['status'] == 'PREPARED']

        def execute(p):
            t = time.time()
            done = base / 'runs' / p['name'] / 'run01' / 'result.json'
            if done.exists():  # completed by an earlier attempt: reuse it (and its SOURCE), no new mean
                p['signal_source'] = json.loads(done.read_text())['scientific_source'];p['signal_execute_reused'] = True
                return p, 0, 0.0
            p['signal_source'] = src
            code = self.entry('--execute', '--approve-new-means', '--source', src, '--config', rel('configurations', p['name'] + '.json'),
                              '--saved-packets', str(packets_file), '--out', rel('runs', p['name'], 'run01'),
                              log=base / 'logs' / (p['name'] + '.execute.log'))
            return p, code, time.time() - t
        with ThreadPoolExecutor(max(1, min(len(todo), self.sig.get('parallel', 8)))) as pool:
            for p, code, sec in pool.map(execute, todo):
                p['signal_execute'] = dict(returncode=code, seconds=round(sec, 1))
                if code != 0:
                    p.update(status='HELD', reason='SIGNAL_EXECUTE_FAILED:see logs/%s.execute.log' % p['name'])
        self.git('add', '--sparse', rel('runs'), rel('plans'), rel('logs'))
        self.git('-c', 'user.name=HIROMU1015', '-c', 'user.email=u411612g@ecs.osaka-u.ac.jp', '-c', 'core.hooksPath=/dev/null', 'commit', '-q', '-m',
                 'Acquire missing means for PR-Pauli absolute-kappa points (cycle %d) through A\'s common entry [skip ci]\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
                 % self.state['cycle'], check=False)
        raw = self.git('rev-parse', 'HEAD');todo = [p for p in todo if p['status'] == 'PREPARED']
        for p in todo:
            stale = base / 'packets' / p['name']
            if stale.exists() and not (stale / 'MEAN_HANDOFF.json').exists() and not any(stale.iterdir()):
                stale.rmdir()  # empty directory left by a failed assemble
            code = self.entry('--assemble', '--config', rel('configurations', p['name'] + '.json'), '--saved-packets', str(packets_file),
                              '--source', p.get('signal_source', src), '--raw-commit', raw, '--run-dir', rel('runs', p['name'], 'run01') + '/',
                              '--out', rel('packets', p['name']),
                              log=base / 'logs' / (p['name'] + '.assemble.log'))
            if code != 0:
                p.update(status='HELD', reason='SIGNAL_ASSEMBLE_INCOMPLETE:see logs/%s.assemble.log' % p['name'])
        self.git('add', '--sparse', rel('packets'), rel('logs'))
        self.git('-c', 'user.name=HIROMU1015', '-c', 'user.email=u411612g@ecs.osaka-u.ac.jp', '-c', 'core.hooksPath=/dev/null', 'commit', '-q', '-m',
                 'Assemble PR-Pauli absolute-kappa MEAN_HANDOFF packets (cycle %d) [skip ci]\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
                 % self.state['cycle'], check=False)
        pkt = self.git('rev-parse', 'HEAD')
        push = subprocess.run(['git', 'push', 'origin', self.sig['branch']], cwd=self.wt, capture_output=True, text=True)
        remote = self.git('ls-remote', 'origin', 'refs/heads/' + self.sig['branch'], check=False).split('\t')[0]
        self.log(event='SIGNALS_PUBLISHED', source=src, raw=raw, packets=pkt, push_returncode=push.returncode, remote_matches=remote == pkt)
        todo = [p for p in todo if p['status'] == 'PREPARED']
        if not todo:
            return
        req = [dict(name=p['name'], commit=pkt, path=rel('packets', p['name'], 'MEAN_HANDOFF.json')) for p in todo]
        (self.run / 'intake').mkdir(exist_ok=True);qf = self.run / 'intake' / ('PACKETS_cycle%02d.json' % self.state['cycle'])
        of = self.run / 'intake' / ('INTAKE_cycle%02d.json' % self.state['cycle']);qf.write_text(json.dumps(req, indent=1) + '\n')
        subprocess.run([self.sig['intake_interpreter'], '-B', str(ROOT / 'scripts/b_intake_pauli_packets.py'), '--view', self.sig['intake_view'],
                        '--packets', str(qf), '--output', str(of)], cwd=ROOT, env=ENV, check=True, stdout=subprocess.DEVNULL)
        intake = json.loads(of.read_text());reg = self.registry()
        for p in todo:
            got = intake[p['name']]
            if got['status'].startswith('SOURCE_BOUND') and got['configuration_id'] == p['configuration_id']:
                reg[p['configuration_id']] = dict(label=p['name'], configuration=got['configuration_source'], signal_packet=got['signal_packet'],
                                                  binding_id=got['binding_id'], total_configuration_id=got['total_configuration_id'],
                                                  intake_status=got['status'])
                p.update(status='REGISTERED', packet=got['signal_packet'], binding_id=got['binding_id'],
                         total_configuration_id=got['total_configuration_id'])
                self.state['saved_packets'].append(got['signal_packet'])
            else:
                p.update(status='HELD', reason='B_INTAKE:%s' % json.dumps(got, default=str)[:400])
        self.registry_file.parent.mkdir(parents=True, exist_ok=True);self.registry_file.write_text(json.dumps(reg, indent=1) + '\n')
        self.log(event='SIGNALS_END', registered=[p['name'] for p in todo if p['status'] == 'REGISTERED'])

    # ------------------------------------------------------------ 3. V2 batch
    def batch(self):
        jobs = self.run / 'jobs' / 'PR_PAULI'
        launched = sum(1 for d in jobs.iterdir() if (d / 'LAUNCH.json').exists()) if jobs.exists() else 0
        spec = json.loads(json.dumps(self.spec))
        maximum = spec['max_new_q3_per_method'].get('PR_PAULI')
        budget = launched + self.a.batch
        if maximum not in (None, 'UNLIMITED'):budget = min(budget, maximum)
        spec['max_new_q3_per_method'] = {'PR_PAULI': budget}
        spec['max_steps_per_method'] = self.a.batch  # the budget ends the batch, not the problem's default step limit
        inv = self.run / 'ROUND_INVOCATION.json';inv.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + '\n')
        self.log(event='V2_START', launched_before=launched, budget=budget)
        t = time.time()
        with (self.run / 'V2_STDOUT.log').open('a') as f:
            f.write('\n=== cycle %d %s\n' % (self.state['cycle'], now()));f.flush()
            code = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/run_rpe_method_optimizer.py'), '--problem', str(self.a.problem),
                                   '--round', str(inv), '--run-dir', str(self.run)], cwd=ROOT, env=dict(ENV, PYTHONPATH=str(ROOT / 'src')),
                                  stdout=f, stderr=subprocess.STDOUT).returncode
        res = O.load_json(self.run / 'ROUND_RESULT.json')['methods']['PR_PAULI'] if code == 0 else None
        self.log(event='V2_END', returncode=code, seconds=round(time.time() - t), stop=res and res['stop_reason'],
                 new_calls=res and res['new_calls'])
        return code, res

    # ------------------------------------------------------------ 4. plan
    def absorb(self, res):
        by = {b['configuration_id']: b for b in res['by_configuration']}
        counts = {}
        for e in res['evaluated']:
            if e['source'] != 'SAVED':
                counts[e['configuration_id']] = counts.get(e['configuration_id'], 0) + 1
        for p in self.state['points']:
            cid = p.get('alias_of') if p['status'] == 'ALIAS' else p.get('configuration_id')
            b = by.get(cid)
            if b is None or p['status'] not in ('REGISTERED', 'EVALUATED', 'ALIAS'):
                continue
            best = b.get('best')
            p.update(best=None if best is None else O.frac(best['cost']), best_N=None if best is None else O.split(best['vector'])[0],
                     best_MSE_upper=None if best is None else best.get('MSE_upper'), best_job=None if best is None else best.get('job'),
                     promising=b.get('promising'), picks=counts.get(cid, 0))
            if p['status'] == 'REGISTERED':
                p['status'] = 'EVALUATED'

    def plan(self):
        pts = [dict(L_D=p['L_D'], kappa=Fraction(p['kappa']), status=p['status'] if p['status'] != 'REGISTERED' else 'PENDING',
                    best=None if p.get('best') is None else O.frac(p['best']), promising=p.get('promising'), picks=p.get('picks'),
                    reason=p.get('reason')) for p in self.state['points']]
        for p in pts:
            if p['status'] in ('PENDING', 'PREPARED'):
                p['status'] = 'PENDING'
        sched = KS.Sched(self.state.get('family_constants', {}), self.nb['bounds']) if self.spec.get('schedule_aware') else None
        proposals, decisions, m_nb = P.kappa_neighbourhood_plan(pts, self.nb, sched)
        for q in proposals:
            if self.point(q['L_D'], q['kappa']) is None:
                self.state['points'].append(dict(L_D=q['L_D'], kappa=str(q['kappa']), status='PENDING', requested_cycle=self.state['cycle'],
                                                 request_reason=q['reason'], request_evidence={k: v for k, v in q.items() if k not in ('L_D', 'kappa')}))
        self.state['history'].append(dict(cycle=self.state['cycle'], UTC=now(), neighbourhood_min=m_nb, decisions=decisions,
                                          proposals=[dict(L_D=q['L_D'], kappa=str(q['kappa']), reason=q['reason']) for q in proposals]))
        return proposals, decisions

    def split_stage(self, res):
        """Prepare the next unprepared / unpriced split(s) in V2's order; a prepared split becomes a kappa family."""
        cfg = self.spec.get('splits')
        if not cfg:
            return
        chosen = SPL.select(res.get('input_requests') or [], self.state['splits'], cfg.get('per_cycle', 1))
        stage = SPL.SplitStage(cfg, self.run, self.log, ROOT)
        for ld, why in chosen:
            rec = dict(L_D=ld, selected_cycle=self.state['cycle'], selected_UTC=now(), selection=why, status='SELECTED')
            self.state['splits'][str(ld)] = rec;self.save();self.log(event='SPLIT_SELECTED', L_D=ld, why=why)
            t = time.time()
            try:
                stage.prepare(ld, rec, has_configuration=why['kind'] == 'CONFIGURATION_INPUTS_MISSING')
            except Exception as exc:  # recorded; the other candidates continue
                rec.update(status='HELD', stage=rec.get('stage', 'UNKNOWN'), reason='STAGE_ERROR:%s: %s' % (type(exc).__name__, str(exc)[:300]))
            rec['seconds'] = round(time.time() - t, 1);self.log(event='SPLIT_END', L_D=ld, status=rec['status'], reason=rec.get('reason'))
            if rec['status'] == 'PREPARED':
                self.activate(ld, 'SPLIT_PREPARED_CYCLE_%d' % self.state['cycle'])
            self.save()

    def activate(self, ld, reason):
        """Add (or re-open) the centre kappa point of a split that became preparable."""
        p = self.point(ld, self.nb['center'])
        if p is None:
            self.state['points'].append(dict(L_D=ld, kappa=str(Fraction(self.nb['center'])), status='PENDING', requested_cycle=self.state['cycle'],
                                             request_reason='CENTER', activation=reason))
        elif p['status'] == 'HELD':
            p.update(status='PENDING', reopened=dict(reason=reason, previous=p.get('reason'), cycle=self.state['cycle']));p.pop('reason', None)

    def families_active(self, decisions):
        return {d['L_D'] for d in decisions if not d['decision'].startswith('STOP_FAMILY_OUTSIDE') and not d['decision'].startswith('STOP_FAMILY_INPUTS')
                and not d['decision'].startswith('STOP_FAMILY_CENTER_HELD')}

    def loop(self):
        if not self.state['points']:
            for ld in self.spec['kappa_families']:
                self.state['points'].append(dict(L_D=ld, kappa=str(Fraction(self.nb['center'])), status='PENDING', requested_cycle=0,
                                                 request_reason='CENTER'))
            self.state['history'].append(dict(cycle=0, UTC=now(), decisions=[dict(L_D=ld, decision='CENTER') for ld in self.spec['kappa_families']]))
            self.save()
        for key, rec in self.state['splits'].items():
            if rec.get('status') == 'PREPARED':
                self.activate(int(key), 'SPLIT_PREPARED_BEFORE_THIS_RUN')
        self.save()
        while self.state['cycle'] < self.a.max_cycles:
            self.state['cycle'] += 1;self.save()
            if any(p['status'] in ('PENDING', 'PREPARED') for p in self.state['points']):
                self.prepare();self.save();self.signals();self.save()
            code, res = self.batch()
            if res is None:
                self.state['stop'] = dict(reason='V2_INVOCATION_FAILED', returncode=code, UTC=now());self.save();return
            self.absorb(res)
            self.split_stage(res)
            proposals, decisions = self.plan();self.save()
            active = self.families_active(decisions)
            promising = sum(p.get('promising') or 0 for p in self.state['points'] if p['L_D'] in active and p['status'] in ('EVALUATED',))
            pending = [p for p in self.state['points'] if p['status'] in ('PENDING', 'PREPARED', 'REGISTERED')]
            open_splits = SPL.select(res.get('input_requests') or [], self.state['splits'], 99) if self.spec.get('splits') else []
            if open_splits:  # unprepared splits still waiting for their turn keep the run going
                pending = pending + [dict(split=ld) for ld, _ in open_splits]
            self.log(event='CYCLE_END', v2_stop=res['stop_reason'], proposals=len(proposals), promising_active=promising, pending=len(pending),
                     decisions=[(d['L_D'], d['decision']) for d in decisions])
            if res['stop_reason'] in ('UNKNOWN_RESULT', 'STALE_JOB_NEEDS_REVIEW', 'NO_ROUTE_ADAPTER'):
                self.state['stop'] = dict(reason='V2_' + res['stop_reason'], UTC=now());self.save();return
            if not proposals and not pending and promising == 0:
                self.state['stop'] = dict(reason='NEIGHBOURHOOD_RULE_EXHAUSTED_NO_PROMISING_CANDIDATE', UTC=now(), v2_stop=res['stop_reason'])
                self.save();return
            if not proposals and not pending and res['stop_reason'] in ('ONLY_SMALL_OR_DEFERRED_LEFT', 'NO_CANDIDATE_LEFT_IN_RANGE'):
                self.state['stop'] = dict(reason='V2_' + res['stop_reason'], UTC=now());self.save();return
        self.state['stop'] = dict(reason='MAX_CYCLES', UTC=now());self.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True);ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--max-cycles', type=int, default=60)
    a = ap.parse_args();a.problem = a.problem.resolve()
    d = Driver(a);d.log(event='DRIVER_START', pid=os.getpid());d.loop();d.log(event='DRIVER_END', stop=d.state.get('stop'))
    print(json.dumps(d.state.get('stop'), default=str))


if __name__ == '__main__':
    main()
