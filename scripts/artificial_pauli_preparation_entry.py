"""ARTIFICIAL stand-in for A's PR-Pauli preparation entry (connection tests only; no H4 data, no science).

Same CLI and response fields as A's entry.  Behaviour per item comes from request['artificial_responses'] keyed by
'L_D=<n>' or 'L_D=<n>:kappa=<k>': {"status", "configuration_id", "complete", "next_actions", "missing_inputs",
"request_rounds", "price_registry"}; '*fail*' at top level -> exit code 3 without output.
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--request', type=Path, required=True);ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cache', type=Path)
    a = ap.parse_args();req = json.loads(a.request.read_text());spec = req.get('artificial_responses', {})
    if spec.get('*fail*'):
        raise SystemExit(3)
    items = []
    for it in req['items']:
        ld = (it.get('configuration') or {}).get('L_D') or it.get('artificial_L_D') or int(it['item_id'].split('L_D=')[1].split(':')[0])
        kappa = (it.get('changes') or {}).get('kappa_multiplier', '1')
        s = spec.get('L_D=%s:kappa=%s' % (ld, kappa)) or spec.get('L_D=%s' % ld) or dict(status='NEEDS_SPLIT_PREPARATION', complete=False,
                                                                                            next_actions=['DERIVE_DESCRIPTOR'])
        complete = s.get('complete', s.get('status') == 'GENERATED')
        cid = s.get('configuration_id')
        row = dict(item_id=it['item_id'], request=it, status=s['status'], L_D=ld, requested_changes=it.get('changes'),
                   configuration_id=cid if complete else None, configuration_source=s.get('configuration_source'),
                   configuration=dict(schema='ARTIFICIAL', id=cid, L_D=ld) if complete else None,
                   readiness=dict(configuration_complete=complete, signal_13round_complete=complete and not s.get('request_rounds'),
                                  price_26axis_complete=False if complete else None),
                   next_actions=s.get('next_actions', ['ASSEMBLE_REUSED_SIGNAL_PACKET', 'PROCESS_D2_PRICE_HANDOFF'] if complete else []),
                   missing_inputs=s.get('missing_inputs', []), preparation_status='ARTIFICIAL',
                   signal_plan=dict(counts={'REQUEST': len(s.get('request_rounds', [])), 'REUSE': 13 - len(s.get('request_rounds', []))},
                                    request_rounds=s.get('request_rounds', []), unsupported_rounds=[]) if complete else None,
                   price_handoff=dict(implementation_status='ARTIFICIAL', needs=['ARTIFICIAL_PRICE_CONTRACT'], next_action='QUOTE_MODEL',
                                      quote_executed=False, evidence=dict(registry_entry=s.get('price_registry'))) if complete else None,
                   scientific_request=None if complete else dict(prefix=ld, artificial=True))
        items.append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(schema='A_PAULI_CANDIDATE_PREPARATION_V1', kind=req.get('kind'), items=items, artificial=True,
                                        signal_price_Q3_evaluation_performed=False), indent=1) + '\n')


if __name__ == '__main__':
    main()
