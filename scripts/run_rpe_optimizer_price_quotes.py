"""Quote PR unit prices with D2's PriceRules (read-only view at its implementation SOURCE): no fit, no compile.

Request: {"policy": "MODEL", "items": [{"item_id", "registry": <entry name>, "configuration": <full config or null>}]}
The implementation contract is always the registered entry's candidate-specific contract; the configuration is
the registered one (null) or a generated one.  The policy is passed explicitly and never defaulted to AUTO.

    python run_rpe_optimizer_price_quotes.py --view <view> --request REQ.json --output OUT.json
"""
import argparse
import json
from pathlib import Path
import sys

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--view', type=Path, required=True);p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--registry', type=Path, help='registry JSON to use instead of the view\'s own (e.g. a B2 extension of D2\'s '
                                                 'registry with newly calibrated entries; D2\'s PriceRules code is unchanged)')
    a = p.parse_args();req = json.loads(a.request.read_text())
    if req.get('policy') not in ('MODEL', 'AUTO', 'DIRECT'):
        raise SystemExit('An explicit price policy is required')
    sys.path.insert(0, str(a.view / 'src'))
    from trottertracks.pr_price_rules import PriceRules, GitReader
    rules = PriceRules(GitReader(a.view), json.loads(a.registry.read_text())) if a.registry else PriceRules.from_repo(a.view)
    registry = {e['name']: rules.template(e['name']) for e in rules.entries}
    out = dict(policy=req['policy'], registry={k: dict(configuration_id=v['configuration']['id']) for k, v in registry.items()}, items=[])
    for item in req['items']:
        t = registry[item['registry']]
        cfg = item.get('configuration') or t['configuration']
        try:
            q = rules.quote(dict(configuration=cfg, implementation=t['implementation'], policy=req['policy']))
            rows = [dict(m=r['m'], axis=r['axis'], q=r['q'], R=r['R'], unit_price_RZ=r['unit_price_RZ'], classification=r['classification'],
                         rule=r.get('registered_rule'), extrapolation=(r.get('model_alternative') or {}).get('extrapolation'),
                         event_step_changed=(r.get('model_alternative') or {}).get('event_step_changed_from_calibration'),
                         needs=r.get('needs')) for r in q['rows']]
            out['items'].append(dict(item_id=item['item_id'], registry=item['registry'], configuration_id=cfg['id'], status='QUOTED',
                                     complete=q['complete'], policy=q['policy'], rows=rows))
        except Exception as exc:
            out['items'].append(dict(item_id=item['item_id'], registry=item['registry'], status='QUOTE_REJECTED',
                                     reason=type(exc).__name__ + ': ' + str(exc), complete=False))
    a.output.write_text(json.dumps(out, ensure_ascii=False) + '\n')
    print(json.dumps({i['item_id']: (i['status'], i['complete']) for i in out['items']}))


if __name__ == '__main__':
    main()
