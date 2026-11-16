"""Requested widths Delta_PF != h for PR-Pauli: schedule, price reuse and calibration cells (metadata only).

For every (L_D, requested width) at kappa itself = 2:
  1. A's preparation entry (pinned view of the problem) maps the width with B2's mapper: per round q_m, T_m and the actual
     width T_m/q_m.  A width keeps its JSON representation: the string "1/5" is the exact rational, the number 0.2 is its
     binary64 value -- two different widths, never rounded into one another;
  2. the exact kappa intervals of that family's 13-round schedule around 2 (first different schedule below / above);
  3. D2's PriceRules MODEL quote (raw PR_PRICE_APPLICABILITY_V1, B2-extended registry) per configuration;
  4. one batch through the C/A common calibration-request entry (02e7cc6a1): per round reused price or calibration cell,
     identical cells across configurations pooled, missing SOURCE recorded.
No signal, path, compile, fit, quote model change or Q3 is run (new science for Delta_PF != h is not approved).

    python prepare_pauli_width_requests.py --problem P --round R --bases ROUTE_REGISTRY.json --out DIR
"""
import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from trottertracks import rpe_candidate_preparation as R  # noqa: E402
from trottertracks import rpe_kappa_schedule as KS  # noqa: E402

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
WIDTHS = [dict(label='exact_1o5', delta_pf='1/5'), dict(label='binary64_0p2', delta_pf=0.2)]


def frac_text(v):
    v = KS._frac(v)
    return dict(exact=str(v), binary64_hex=float(v).hex(), is_binary64=Fraction(float(v)) == v)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--problem', type=Path, required=True);ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--bases', type=Path, required=True, help='route registry holding the kappa=2, Delta=h configuration per L_D')
    ap.add_argument('--l-d', type=int, nargs='+', default=[1, 2, 3, 5]);ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args();out = a.out.resolve();out.mkdir(parents=True, exist_ok=True)
    problem = json.loads(a.problem.read_text());spec = json.loads(a.round.read_text())
    prep = problem['methods']['PR_PAULI']['generation']['preparation'];nb = spec['search_overrides']['kappa_neighbourhood']
    reg = json.loads(a.bases.read_text())
    bases = {}
    for cid, e in reg.items():
        ld = int(e['label'].split('_')[0][2:])
        if e['label'].endswith('_kappa_abs_2') and ld in a.l_d:
            bases[ld] = dict(source=e['configuration'], configuration_id=cid)
    items, rows = [], []
    for ld in a.l_d:
        for w in WIDTHS:
            fam = dict(L_D=ld, formula='2nd', delta_over_h=None, delta_pf=w['delta_pf'])
            items.append(R.kappa_item('WIDTH', fam, bases[ld], Fraction(1), 'kappa=2', K=prep.get('K', 2)))
            rows.append(dict(L_D=ld, width=w, item_id=items[-1]['item_id'], base=bases[ld]))
    response, call = R.run_preparation(ROOT, prep, items, out / 'preparation', 'width_requests')
    by_id = {x['item_id']: x for x in response.get('items', [])}
    families, quote_items = {}, []
    for r in rows:
        x = by_id.get(r['item_id'], {});cfg = x.get('configuration')
        r.update(status=x.get('status', response.get('status')), configuration_id=x.get('configuration_id'))
        if not cfg:
            r['needs'] = x.get('next_actions') or x.get('missing_inputs') or response.get('log_tail');continue
        if KS._frac(cfg['spec']['kappa']['value']) != 2:
            raise ValueError('KAPPA_NOT_PRESERVED:' + r['item_id'])
        s = cfg['spec'];rows_c = KS.constants(cfg)
        if KS.R_vector(rows_c, 2) != [x['R_outer'] for x in cfg['rounds']]:
            raise ValueError('MAPPER_FORMULA_DOES_NOT_REPRODUCE_GENERATED_R:' + r['item_id'])
        r['requested_width'] = dict(json_value=r['width']['delta_pf'], json_type=type(r['width']['delta_pf']).__name__, **frac_text(s['delta_pf']))
        r['schedule'] = [dict(m=x['m'], q=x['q_PF'], T=frac_text(x['T']), actual_width=frac_text(x['actual_delta_pf']), R=x['R_outer'])
                         for x in cfg['rounds']]
        fam = '%d:%s' % (r['L_D'], r['width']['label']);families[fam] = rows_c
        sch = KS.Sched({r['L_D']: KS.encode_rows(rows_c)}, nb['bounds'])
        r['kappa_intervals'] = {d: sch.describe(r['L_D'], Fraction(2), d) for d in ('down', 'up')}
        (out / 'configurations').mkdir(exist_ok=True)
        f = out / 'configurations' / ('LD%d_%s_kappa_2.json' % (r['L_D'], r['width']['label']));f.write_text(json.dumps(cfg, indent=1) + '\n')
        r['configuration_file'] = str(f.relative_to(out))
        quote_items.append(dict(item_id=r['item_id'], registry='PAULI_LD%d' % r['L_D'], configuration=cfg))
    # D2 raw MODEL quotes
    pr = problem['price_rules'];qreq = out / 'QUOTE_REQUEST.json';qout = out / 'QUOTES.json'
    qreq.write_text(json.dumps(dict(policy='MODEL', items=quote_items)) + '\n')
    regfile = ROOT / spec['price_rules_overrides']['registry_file']
    subprocess.run([pr['interpreter'], '-B', str(ROOT / 'scripts/run_rpe_optimizer_price_quotes.py'), '--view', pr['view'], '--raw',
                    '--registry', str(regfile), '--request', str(qreq), '--output', str(qout)], cwd=ROOT, env=ENV, check=True,
                   stdout=subprocess.DEVNULL)
    quotes = {x['item_id']: x for x in json.loads(qout.read_text())['items']}
    batch, order = [], []
    for r in rows:
        q = quotes.get(r['item_id'])
        if q is None:
            continue
        r['d2_quote'] = dict(status=q['status'], complete=q.get('complete'), reason=q.get('reason'),
                             classifications={c: sum(1 for x in q.get('rows') or [] if x['classification'] == c)
                                              for c in {x['classification'] for x in q.get('rows') or []}})
        if q.get('raw') is not None:
            cfg = json.loads((out / r['configuration_file']).read_text())
            batch.append(dict(configuration=cfg, quote=q['raw']));order.append(r)
    pb = spec['price_batch'];(out / 'BATCH_INPUT.json').write_text(json.dumps(batch) + '\n')
    (out / 'EXECUTION_OPS.json').write_text(json.dumps(dict(branch=pb['execution_branch'])) + '\n')
    code = subprocess.run([pb['interpreter'], '-B', 'scripts/prepare_pauli_calibration.py', '--batch', str(out / 'BATCH_INPUT.json'),
                           '--output', str(out / 'BATCH.json'), '--execution', str(out / 'EXECUTION_OPS.json')], cwd=pb['view'],
                          env=dict(ENV, PYTHONPATH='src'), stdout=open(out / 'BATCH.log', 'w'), stderr=subprocess.STDOUT).returncode
    result = json.loads((out / 'BATCH.json').read_text()) if code == 0 else {}
    by_cid = {x.get('configuration_id'): x for x in result.get('results', [])}
    for r in order:
        x = by_cid.get(r['configuration_id'], {})
        r['common_entry'] = dict(status=x.get('status'), next_action=x.get('next_action'), needs=x.get('needs'),
                                 reused_axes=len(x.get('reused_prices') or []), missing_axes=len(x.get('missing_prices') or []),
                                 rounds=[dict(m=y['m'], q=y['q'], R=y['R'], actual_step=frac_text(y['actual_step']), action=y.get('action'),
                                              request_id=y.get('request_id'), missing_axes=y.get('missing_axes'), needs=y.get('needs'))
                                         for y in x.get('round_actions') or []])
    cells = [dict(request_id=g['request_id'], status=g['status'], next_action=g['next_action'], cell=g['driver_request']['driver_spec']['cell'],
                  actual_step=frac_text(g['proposed_applicability']['actual_step']), covered_rounds=g['covered_rounds'],
                  consumers=[dict(configuration_id=c['configuration_id'], rounds=c['rounds']) for c in g['consumers']],
                  base=g['driver_request']['driver_spec']['base'], saved_calibration_source=g.get('saved_calibration_source'))
             for g in result.get('calibration_requests') or []]
    summary = dict(schema='B2_PAULI_WIDTH_REQUESTS_V1', scope='metadata only: A mapper + D2 MODEL quote + C/A common calibration-request '
                   'entry; no signal/path/compile/fit/Q3 (Delta_PF != h science not approved)',
                   preparation=dict(view_commit=prep['view_commit'], call=str(call.relative_to(out))),
                   common_entry=dict(view=pb['view'], view_commit=pb['view_commit'], returncode=code,
                                     unique_calibration_requests=result.get('unique_calibration_requests'),
                                     maximum_new_paths=result.get('maximum_new_paths'), maximum_new_compiles=result.get('maximum_new_compiles'),
                                     execution_authorized=result.get('execution_authorized')),
                   rows=rows, calibration_cells=cells)
    (out / 'WIDTH_REQUESTS.json').write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=str) + '\n')
    print(json.dumps(dict(rows=[(r['L_D'], r['width']['label'], r.get('status'), (r.get('common_entry') or {}).get('status')) for r in rows],
                          unique_cells=len(cells)), default=str))


if __name__ == '__main__':
    main()
