"""Split preparation stage of the kappa-neighbourhood driver: choose an unprepared PR-Pauli split and prepare it.

Selection: the V2 input requests (NEW_L_D = no configuration, CONFIGURATION_INPUTS_MISSING = configuration without
price / signals / route) in V2's own order (nearest L_D to the method's leading L_D first); splits without a price get a
turn like any other; ``per_cycle`` splits are prepared per driver cycle and every choice is recorded with its reason.

Preparation of one split, each step through the fixed existing entry, stopping at the first step that cannot pass:
  P0     A's scripts/prepare_pauli_candidate.py --prepare-missing (SOURCE 6e4b6bd6a code, B2 checkout/branch; the entry
         commits the P0 products itself); a numerical/gate STOP is recorded and the split is not retried
  SIGNAL the signal entry the Q3 route accepts (93a5ef3b9, B intake bd42dd1e9) must support the split's configuration
  PRICE  D2 PriceRules MODEL (registry: D2 c1f95e080 plus B2-registered calibrated entries); if incomplete:
         A's scripts/prepare_pauli_calibration.py (7d6a8c385) -> CALIBRATION_REQUEST_READY_NOT_EXECUTED -> existing driver
         scripts/run_pauli_split_cal.py --metadata/--input-preflight/--freeze (commit+push)/--execute -> raw commit ->
         request-driven S0-r1 audit -> registry entry PAULI_LD<p> (PAULI_AUDIT) -> quote again
A prepared split becomes a kappa family (centre point) and goes through the driver's normal signals / Q3 path.
"""
import json
import os
from pathlib import Path
import subprocess

IDENTITY = dict(GIT_AUTHOR_NAME='HIROMU1015', GIT_AUTHOR_EMAIL='u411612g@ecs.osaka-u.ac.jp',
                GIT_COMMITTER_NAME='HIROMU1015', GIT_COMMITTER_EMAIL='u411612g@ecs.osaka-u.ac.jp')
GITC = ['-c', 'user.name=HIROMU1015', '-c', 'user.email=u411612g@ecs.osaka-u.ac.jp', '-c', 'core.hooksPath=/dev/null']
TRAILER = '\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'


def select(requests, splits, per_cycle):
    """Next splits to prepare, in V2's input-request order, with the recorded reason."""
    out = []
    for r in requests:
        if r['kind'] not in ('NEW_L_D', 'CONFIGURATION_INPUTS_MISSING'):
            continue
        ld = (r.get('configuration') or {}).get('L_D')
        if ld is None or str(ld) in splits or any(x[0] == ld for x in out):
            continue
        out.append((ld, dict(kind=r['kind'], needs=r.get('needs'), distance_to_leading_L_D=r.get('distance_to_leading_L_D'),
                             rule='V2 input-request order (nearest L_D to the leading L_D first); unprepared and unpriced splits alike')))
        if len(out) >= per_cycle:
            break
    return out


class SplitStage:
    def __init__(self, cfg, run_dir, log, root):
        self.c, self.run, self.log, self.root = cfg, Path(run_dir), log, Path(root)

    def sh(self, cmd, cwd, log_file=None, env=None):
        e = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', **(env or {}))
        if log_file:
            with open(log_file, 'w') as f:
                return subprocess.run(cmd, cwd=str(cwd), env=e, stdout=f, stderr=subprocess.STDOUT).returncode
        return subprocess.run(cmd, cwd=str(cwd), env=e, capture_output=True, text=True)

    def git(self, cwd, *args, check=True):
        return subprocess.run(['git', *args], cwd=str(cwd), check=check, capture_output=True, text=True).stdout.strip()

    def out(self, ld):
        d = self.run / 'splits' / ('LD%d' % ld);d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ P0 / P1
    def p0(self, ld, rec):
        ck = Path(self.c['p0_checkout']);d = self.out(ld)
        if (ck / self.c['p0_directory'] / ('LD%d' % ld)).exists():
            rec.update(stage='P0', status='HELD', reason='P0_OUTPUT_EXISTS_NO_RETRY: ' + str(ck / self.c['p0_directory'] / ('LD%d' % ld)))
            return None
        req = dict(kind='pauli_native', view=str(ck), view_commit=self.c['prep_source'],
                   items=[dict(item_id='PR_PAULI:L_D=%d:kappa=%s' % (ld, self.c['centre']), configuration=dict(L_D=ld),
                               changes=dict(formula='2nd', delta_over_h='1', kappa_multiplier=self.c['centre_multiplier'], K=2))])
        (d / 'REQUEST_P0.json').write_text(json.dumps(req, indent=1) + '\n')
        source = self.git(ck, 'rev-parse', 'HEAD')
        code = self.sh([self.c['prep_interpreter'], '-B', 'scripts/prepare_pauli_candidate.py', '--request', str(d / 'REQUEST_P0.json'),
                        '--output', str(d / 'PREPARATION.json'), '--cache', str(d / 'SPLITS.json'), '--prepare-missing', '--source', source,
                        '--p0-directory', str(ck / self.c['p0_directory'])], ck, d / 'P0.log', env=IDENTITY)
        item = json.loads((d / 'PREPARATION.json').read_text())['items'][0] if (d / 'PREPARATION.json').exists() else {}
        rec['p0'] = dict(source=source, returncode=code, status=item.get('status'), carrier=self.git(ck, 'rev-parse', 'HEAD'))
        if item.get('status') != 'GENERATED':
            produced = ck / self.c['p0_directory'] / ('LD%d' % ld)
            if produced.exists():  # keep the stopped run's partial products as evidence (no retry)
                self.git(ck, 'add', '--sparse', str(produced));self.git(ck, *GITC, 'commit', '-q', '-m',
                         'Record the stopped PR-Pauli LD%d P0 run (%s; no retry) [skip ci]%s' % (ld, item.get('status'), TRAILER), check=False)
                rec['p0']['evidence'] = self.git(ck, 'rev-parse', 'HEAD')
            self.git(ck, 'push', 'origin', self.c['p0_branch'], check=False)
            rec.update(stage='P0', status='HELD', reason='P0:%s:%s' % (item.get('status'), json.dumps(item.get('split_preparation') or
                                                                                                       item.get('missing_inputs'))[:300]))
            return None
        self.git(ck, 'push', 'origin', self.c['p0_branch'], check=False)
        return d / 'PREPARATION.json', d / 'PREPARATION.0.CONFIGURATION.json'

    def metadata_item(self, ld, rec):
        """A split that already has a dictionary/configuration: A's entry in metadata mode for the centre kappa."""
        ck = Path(self.c['p0_checkout']);d = self.out(ld)
        req = dict(kind='pauli_native', view=str(ck), view_commit=self.c['prep_source'],
                   items=[dict(item_id='PR_PAULI:L_D=%d:kappa=%s' % (ld, self.c['centre']), configuration=dict(L_D=ld),
                               changes=dict(formula='2nd', delta_over_h='1', kappa_multiplier=self.c['centre_multiplier'], K=2))])
        (d / 'REQUEST_META.json').write_text(json.dumps(req, indent=1) + '\n')
        self.sh([self.c['prep_interpreter'], '-B', 'scripts/prepare_pauli_candidate.py', '--request', str(d / 'REQUEST_META.json'),
                 '--output', str(d / 'PREPARATION_META.json'), '--cache', str(d / 'SPLITS.json')], ck, d / 'META.log')
        item = json.loads((d / 'PREPARATION_META.json').read_text())['items'][0] if (d / 'PREPARATION_META.json').exists() else {}
        if item.get('status') != 'GENERATED':
            rec.update(stage='CONFIGURATION', status='HELD', reason='A_METADATA:%s:%s' % (item.get('status'), item.get('next_actions')))
            return None
        return d / 'PREPARATION_META.json', d / 'PREPARATION_META.0.CONFIGURATION.json'

    # ------------------------------------------------------------ signal entry support
    def signal_support(self, ld, config_file, rec):
        res = self.sh([self.c['signal_interpreter'], '-B', 'scripts/run_pauli_signal_entry.py', '--plan', '--config', str(config_file)],
                      self.c['signal_worktree'])
        try:
            plan = json.loads(res.stdout)
        except ValueError:
            plan = dict(error=res.stdout[-300:] + res.stderr[-300:])
        rec['signal_plan'] = {k: plan.get(k) for k in ('counts', 'unsupported_rounds', 'configuration_reasons', 'error')}
        if plan.get('unsupported_rounds') or plan.get('configuration_reasons') or plan.get('error'):
            rec.update(stage='SIGNAL', status='HELD',
                       reason='SIGNAL_ENTRY_%s_UNSUPPORTED:%s; the Q3 route (B intake %s) accepts packets of that entry only' % (
                           self.c['signal_entry_source'][:9], plan.get('configuration_reasons') or plan.get('error'), self.c['intake_source'][:9]))
            return False
        return True

    # ------------------------------------------------------------ price (quote, then calibration)
    def quote(self, item_file, d, registry):
        script = ('import json,sys;from pathlib import Path;sys.set_int_max_str_digits(0);V=Path(sys.argv[1]);sys.path.insert(0,str(V/"src"))\n'
                  'from trottertracks.pr_price_rules import PriceRules,GitReader\n'
                  'reg=json.loads(Path(sys.argv[2]).read_text());it=json.loads(Path(sys.argv[3]).read_text())["items"][0]\n'
                  'r=it["price_handoff"]["request"];q=PriceRules(GitReader(V),reg).quote(dict(configuration=r["configuration"],'
                  'implementation=r["implementation"],policy="MODEL"))\n'
                  'Path(sys.argv[4]).write_text(json.dumps(q,indent=1,default=str)+"\\n");print(q["complete"])')
        res = self.sh([self.c['driver_interpreter'], '-B', '-c', script, self.c['d2_view'], str(registry), str(item_file), str(d / 'D2_QUOTE_MODEL.json')],
                      self.root)
        return res.returncode == 0 and res.stdout.strip().endswith('True')

    def price(self, ld, item_file, rec):
        d = self.out(ld);reg = self.root / self.c['registry_file']
        if self.quote(item_file, d, reg):
            rec['price'] = dict(status='COMPLETE_MODEL_26_AXES', registry=str(reg));return True
        ck = Path(self.c['calibration_checkout']);rel = 'docs/research/b2_pauli_calibration_20261116/LD%d' % ld;w = ck / rel
        w.mkdir(parents=True, exist_ok=True)
        for f in (item_file, d / 'D2_QUOTE_MODEL.json'):
            (w / Path(f).name).write_bytes(Path(f).read_bytes())
        (w / 'EXECUTION_OPS.json').write_text(json.dumps(dict(branch=self.c['calibration_branch'],
                                                             private=str(Path(self.c['calibration_private_root']) / ('LD%d' % ld) / 'CELL'))) + '\n')
        self.sh([self.c['prep_interpreter'], '-B', 'scripts/prepare_pauli_calibration.py', '--preparation', str(w / Path(item_file).name),
                 '--price-result', str(w / 'D2_QUOTE_MODEL.json'), '--output', str(w / 'CAL_PREPARATION.json'), '--execution',
                 str(w / 'EXECUTION_OPS.json')], ck, w / 'CAL_PREPARATION.log')
        cal = json.loads((w / 'CAL_PREPARATION.json').read_text()) if (w / 'CAL_PREPARATION.json').exists() else {}
        rec['calibration'] = dict(request_status=cal.get('status'), needs=cal.get('needs'))
        if cal.get('status') != 'CALIBRATION_REQUEST_READY_NOT_EXECUTED':
            rec.update(stage='PRICE', status='HELD', reason='CALIBRATION_REQUEST:%s:%s' % (cal.get('status'), cal.get('needs')));return False
        reqf = rel + '/CAL_PREPARATION.CALIBRATION_REQUEST.json';spec = json.loads((ck / reqf).read_text())['driver_spec'];base = spec['base']
        py = self.c['driver_interpreter'];drv = [py, '-B', 'scripts/run_pauli_split_cal.py', '--request', reqf]
        steps = [('metadata', drv + ['--metadata']), ('preflight', drv + ['--input-preflight', str(Path(self.c['calibration_private_root']) /
                 ('LD%d_preflight' % ld))]), ('freeze', drv + ['--freeze'])]
        for name, cmd in steps:
            if self.sh(cmd, ck, w / ('DRIVER_%s.log' % name.upper())) != 0:
                rec.update(stage='PRICE', status='HELD', reason='CALIBRATION_DRIVER_%s_FAILED' % name.upper());return False
        self.git(ck, 'add', '--sparse', rel, base + '/BINDING.json')
        self.git(ck, *GITC, 'commit', '-q', '-m', 'Freeze the PR-Pauli LD%d calibration binding (%s) [skip ci]%s' % (ld, spec['cell']['cell_id'], TRAILER))
        self.git(ck, 'push', 'origin', self.c['calibration_branch']);binding = self.git(ck, 'rev-parse', 'HEAD')
        if self.sh(drv + ['--execute', binding], ck, w / 'DRIVER_EXECUTE.log') != 0:
            rec.update(stage='PRICE', status='HELD', reason='CALIBRATION_EXECUTE_STOP (see %s/DRIVER_EXECUTE.log)' % rel);return False
        self.git(ck, 'add', '--sparse', base + '/run01', rel)
        self.git(ck, *GITC, 'commit', '-q', '-m', 'Acquire the PR-Pauli LD%d calibration cell %s [skip ci]%s' % (ld, spec['cell']['cell_id'], TRAILER))
        self.git(ck, 'push', 'origin', self.c['calibration_branch']);raw = self.git(ck, 'rev-parse', 'HEAD')
        cfg_ref = '%s:%s/%s' % (binding, rel, Path(item_file).name.replace('.json', '.0.CONFIGURATION.json'))
        audit = [py, '-B', 'scripts/pauli_split_cal_audit_request.py', '--request', reqf, '--config', cfg_ref]
        for name, extra in (('build', ['--build', raw]), ('run', ['--run', '--workers', '2']), ('analyze', ['--analyze'])):
            if self.sh(audit + extra, ck, w / ('AUDIT_%s.log' % name.upper())) != 0:
                rec.update(stage='PRICE', status='HELD', reason='CALIBRATION_AUDIT_%s_FAILED' % name.upper());return False
        price = json.loads((ck / base / 'PRICE_26AXIS.json').read_text())
        if price.get('status') != 'PRICE_RULE_SUPPORTED_MODEL_REFERENCE':
            rec.update(stage='PRICE', status='HELD', reason='AUDIT_RULE_NOT_SUPPORTED:%s' % price.get('status'))
            self.git(ck, 'add', '--sparse', base, rel);self.git(ck, *GITC, 'commit', '-q', '-m',
                     'Record the unsupported PR-Pauli LD%d calibration audit [skip ci]%s' % (ld, TRAILER), check=False)
            self.git(ck, 'push', 'origin', self.c['calibration_branch'], check=False);return False
        self.git(ck, 'add', '--sparse', base, rel)
        self.git(ck, 'rm', '-q', '--cached', '-r', '--ignore-unmatch', base + '/audit_run01/work')
        self.git(ck, *GITC, 'commit', '-q', '-m', 'Audit the PR-Pauli LD%d calibration cell (S0-r1, supported) [skip ci]%s' % (ld, TRAILER))
        self.git(ck, 'push', 'origin', self.c['calibration_branch']);pc = self.git(ck, 'rev-parse', 'HEAD')
        registry = json.loads(reg.read_text());name = 'PAULI_LD%d' % ld
        entry = next((e for e in registry['entries'] if e['name'] == name), None)
        template = next(e for e in registry['entries'] if e['name'] == 'PAULI_LD5')
        if entry is None:
            entry = dict(name=name, implementation=json.loads(Path(item_file).read_text())['items'][0]['price_handoff']['request']['implementation']
                         or template['implementation'], direct_sources=[]);registry['entries'].append(entry)
        R = spec['cell']['R']
        entry.update(configuration_source=dict(commit=binding, path=cfg_ref.split(':', 1)[1]),
                     price_source=dict(commit=pc, path=base + '/PRICE_26AXIS.json', selector=[]),
                     model_source=dict(commit=pc, path=base + '/PRICE_26AXIS.json', selector=[]), mode='PAULI_AUDIT',
                     calibration_domain=dict(q=[8, 8], R=[R, R]), calibration_R=[R],
                     caveats=['A S0-r1 decomposition supports this candidate (B2 request-driven audit); all round prices are model applications'],
                     calibration_evidence=dict(commit=pc, path=base + '/PRICE_26AXIS.json', selector=['calibration_cell', 'rows']),
                     registration=dict(by='B2 kappa-neighbourhood driver', raw=raw, binding=binding))
        reg.write_text(json.dumps(registry, indent=1) + '\n')
        if not self.quote(item_file, d, reg):
            rec.update(stage='PRICE', status='HELD', reason='REGISTERED_BUT_QUOTE_INCOMPLETE');return False
        rec['price'] = dict(status='CALIBRATED_AND_REGISTERED', registry=str(reg), price_commit=pc, raw=raw, binding=binding)
        return True

    # ------------------------------------------------------------ one split
    def prepare(self, ld, rec, has_configuration=False):
        """has_configuration: the split already has a dictionary/configuration (CONFIGURATION_INPUTS_MISSING)."""
        files = self.metadata_item(ld, rec) if has_configuration else self.p0(ld, rec)
        if files is None:
            return rec
        item_file, config_file = files
        if not self.signal_support(ld, config_file, rec):
            return rec
        if not self.price(ld, item_file, rec):
            return rec
        rec.update(stage='READY', status='PREPARED', reason='configuration, signal entry support and 26-axis MODEL price')
        return rec
