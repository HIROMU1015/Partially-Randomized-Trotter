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
import run_rpe_method_optimizer as CLI  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


class Driver:
    def __init__(self, a):
        self.a, self.run = a, a.run_dir.resolve();self.run.mkdir(parents=True, exist_ok=True)
        self.spec = O.load_json(a.round);self.nb = self.spec['search_overrides']['kappa_neighbourhood']
        sig = self.spec['signals'];self.sig = sig;self.wt = Path(sig['worktree']);self.sub = sig['directory']
        self.state_file = self.run / 'KAPPA_STATE.json'
        self.state = O.load_json(self.state_file) if self.state_file.exists() else dict(
            schema='RPE_KAPPA_NEIGHBOURHOOD_STATE_V1', round_id=self.spec['round_id'], started=now(), cycle=0, points=[], history=[],
            saved_packets=list(sig['saved_packets']), settings=self.nb)
        self.registry_file = ROOT / self.spec['routes_override']['PR_PAULI'][0]['configurations_file']
        self.points_file = ROOT / self.spec['method_overrides']['PR_PAULI']['generation']['kappa_points_file']

    # ------------------------------------------------------------ bookkeeping
    def log(self, **event):
        with (self.run / 'DRIVER_PROGRESS.jsonl').open('a') as f:
            f.write(json.dumps(dict(UTC=now(), cycle=self.state['cycle'], **event), default=str) + '\n')

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
        problem = CLI.scoped(O.load_json(self.a.problem), self.spec.get('method_overrides'))
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
        self.log(event='PREPARE_END', statuses={s: sum(1 for p in self.state['points'] if p['status'] == s)
                                                for s in {p['status'] for p in self.state['points']}})
        return ctx

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
        spec = json.loads(json.dumps(self.spec));spec['max_new_q3_per_method'] = {'PR_PAULI': launched + self.a.batch}
        inv = self.run / 'ROUND_INVOCATION.json';inv.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + '\n')
        self.log(event='V2_START', launched_before=launched, budget=launched + self.a.batch)
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
        proposals, decisions, m_nb = P.kappa_neighbourhood_plan(pts, self.nb)
        for q in proposals:
            if self.point(q['L_D'], q['kappa']) is None:
                self.state['points'].append(dict(L_D=q['L_D'], kappa=str(q['kappa']), status='PENDING', requested_cycle=self.state['cycle'],
                                                 request_reason=q['reason'], request_evidence={k: v for k, v in q.items() if k not in ('L_D', 'kappa')}))
        self.state['history'].append(dict(cycle=self.state['cycle'], UTC=now(), neighbourhood_min=m_nb, decisions=decisions,
                                          proposals=[dict(L_D=q['L_D'], kappa=str(q['kappa']), reason=q['reason']) for q in proposals]))
        return proposals, decisions

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
        while self.state['cycle'] < self.a.max_cycles:
            self.state['cycle'] += 1;self.save()
            if any(p['status'] in ('PENDING', 'PREPARED') for p in self.state['points']):
                self.prepare();self.save();self.signals();self.save()
            code, res = self.batch()
            if res is None:
                self.state['stop'] = dict(reason='V2_INVOCATION_FAILED', returncode=code, UTC=now());self.save();return
            self.absorb(res)
            proposals, decisions = self.plan();self.save()
            active = self.families_active(decisions)
            promising = sum(p.get('promising') or 0 for p in self.state['points'] if p['L_D'] in active and p['status'] in ('EVALUATED',))
            pending = [p for p in self.state['points'] if p['status'] in ('PENDING', 'PREPARED', 'REGISTERED')]
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
