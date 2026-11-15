"""Evidence for the PR-Pauli signal entry: saved LD1/LD2/LD3/LD5 and artificial B2 variants (no new mean).

1. Saved configurations: --assemble from the saved signals only; rows compared with the published
   packets; optionally B's own bind_packet (--b-root/--b-python, see check_pauli_signal_entry_b_bind.py).
2. Artificial variants of LD3/LD5 (B generate of an edited spec): kappa x3/2, kappa x2/3,
   Delta_PF=1/5, Delta_PF=binary64 0.2, K=4, formula 4th(m5_best), h=1/20. The plan's REQUEST rounds
   are compared with the rounds whose canonical physical row changed.
3. kappa x3/2 and Delta_PF=1/5: execute with a synthetic stub for the REQUEST rounds (records marked
   SYNTHETIC_STUB_NOT_A_MEAN; no provider call), assemble, B intake conditions, optional B bind_packet
   with records taken from the rows.
Writes VALIDATION.json and PLANS.json in --out (no carrier, no packet is published from step 3).
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'), str(ROOT/'tests')]
import run_pauli_signal_entry as RUN  # noqa: E402
import check_pauli_signal_entry as T  # noqa: E402

E = RUN.E
EDITS = (('kappa_x3/2', dict(kappa=Fraction(3, 2))), ('kappa_x2/3', dict(kappa=Fraction(2, 3))),
         ('delta_pf_1/5', dict(delta_pf=Fraction(1, 5))), ('delta_pf_binary64_0.2', dict(delta_pf=Fraction(0.2))),
         ('K_4', dict(finite_K=4)), ('formula_4th(m5_best)', dict(formula='4th(m5_best)')), ('h_1/20', dict(h=Fraction(1, 20))))


def b_bind(args, handoff, rows):
    if not args.b_root:
        return 'NOT_RUN'
    cmd = [args.b_python, str(ROOT/'scripts'/'check_pauli_signal_entry_b_bind.py'), '--b-root', args.b_root, '--handoff', str(handoff)]
    done = subprocess.run(cmd + (['--records-from-rows'] if rows else []), capture_output=True, text=True)
    if done.returncode:
        return dict(status='FAIL', stderr=done.stderr[-1500:])
    value = json.loads(done.stdout)
    return dict(status=value['status'], packets=[{k: p[k] for k in ('prefix', 'configuration_id', 'means', 'binding_status')}
                                                 for p in value['packets']], records_from_rows=rows)


def stub_factory(rows, calls):
    def factory(jobs):
        def evaluate(m, job):
            calls.append(m); rec = deepcopy(rows[m]['mean_record']); st = job['physical_context']['structure']
            rec.update(physical_context=deepcopy(job['physical_context']), candidate=deepcopy(job['physical_context']['physical_candidate']),
                       B_identity=deepcopy(job['B_identity']), B_law_binding=deepcopy(job['B_law_binding']),
                       B_configuration_id=job['B_configuration_id'], B_candidate_id=job['B_candidate_id'], m=m,
                       source=deepcopy(job['physical_context']['numerical_recipe']['source']), signed_weights=st['signed_weights'],
                       allocation=st['allocation'], event_count=st['event_count'], artificial_fixture='SYNTHETIC_STUB_NOT_A_MEAN')
            return rec
        return evaluate
    return factory


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', required=True); parser.add_argument('--b-root'); parser.add_argument('--b-python', default=sys.executable)
    args = parser.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    index = E.saved_index(ROOT); saved = {}
    for ref in E.SAVED_PACKETS:
        for i, pk in enumerate(E.read_ref(ROOT, ref)['packets']):
            saved[pk['prefix']] = dict(ref=ref, index=i, packet=pk)
    report = dict(head=E.git_head(ROOT), saved_index_records=len(index), saved_packets=list(E.SAVED_PACKETS), saved=[], variants=[],
                  stub_runs=[], new_mean_calls=0, compile=0, Q3=0)
    plans = {}
    work = Path(tempfile.mkdtemp(prefix='entry_validation_'))
    try:
        for prefix in (1, 2, 3, 5):
            s = saved[prefix]; config = s['packet']['configuration']
            value, problems = RUN.assemble(work/('LD%d' % prefix), s['ref'], configuration_id=config['id'])
            pkt = value['packets'][0]
            same = all({k: v for k, v in a.items() if k != 'reused_without_reanalysis'}
                       == {k: v for k, v in b.items() if k != 'reused_without_reanalysis'}
                       for a, b in zip(pkt['rounds'], s['packet']['rounds']))
            report['saved'].append(dict(
                prefix=prefix, configuration_id=config['id'], input=dict(s['ref'], configuration_id=config['id']),
                status=value['status'], reused_rounds=pkt['reused_rounds'], new_rounds=pkt['new_rounds'],
                rows_equal_published_except_reuse_flag=same, configuration_equal=pkt['configuration'] == config,
                mean_sources=sorted({json.dumps({k: r['mean_source'][k] for k in ('commit', 'scientific_source')}) for r in pkt['rounds']}),
                b_intake_conditions=problems or 'PASS', b_bind_packet=b_bind(args, work/('LD%d' % prefix)/'MEAN_HANDOFF.json', False),
                handoff_sha256=hashlib.sha256((work/('LD%d' % prefix)/'MEAN_HANDOFF.json').read_bytes()).hexdigest()))
        for prefix in (3, 5):
            config = saved[prefix]['packet']['configuration']
            for name, edit in EDITS:
                v, _ = E.load_configuration(ROOT, T.variant(config, **edit))
                pl = E.plan(ROOT, v, index=index)
                changed = [m for m in range(13) if T.physical(v, m) != T.physical(config, m)]
                case = 'LD%d %s' % (prefix, name); plans[case] = pl
                report['variants'].append(dict(
                    case=case, configuration_id=v['id'], changed_rounds=changed, request_rounds=pl['request_rounds'],
                    rebind_rounds=pl['rebind_rounds'], unsupported_rounds=pl['unsupported_rounds'],
                    requests_equal_changed=pl['request_rounds'] == changed,
                    unsupported_reasons={r['m']: r['reasons'] for r in pl['rounds'] if r['decision'] == 'UNSUPPORTED'},
                    provider_support=sorted({r['provider_support'] for r in pl['rounds']}),
                    R_outer_saved=[r['R_outer'] for r in config['rounds']], R_outer_variant=[r['R_outer'] for r in v['rounds']],
                    q_PF_variant=[r['q_PF'] for r in v['rounds']],
                    first_request_changed_fields=next((r['changed_from_saved_round']['fields'] for r in pl['rounds']
                                                      if r['decision'] != 'REUSE' and 'changed_from_saved_round' in r), None)))
            for name, edit in EDITS[:1] + EDITS[2:3]:
                calls = []; d = work/('stub_LD%d_%s' % (prefix, name.replace('/', '_')))
                d.mkdir(); spec = d/'spec.json'; spec.write_text(json.dumps(T.variant(config, **edit)))
                run = RUN.execute(d/'run', 'SYNTHETIC', str(spec), approve=True, check_source=False,
                                  evaluate_factory=stub_factory(saved[prefix]['packet']['rounds'], calls))
                value, problems = RUN.assemble(d/'out', str(spec), source='SYNTHETIC', raw_commit='SYNTHETIC',
                                               run_dir=str(d/'run')+'/', git=False)
                pkt = value['packets'][0]
                report['stub_runs'].append(dict(
                    case='LD%d %s' % (prefix, name), configuration_id=run['configuration_id'], run_status=run['status'],
                    rebound_carriers=run['rebound'], stub_calls=calls, provider_calls=0, status=value['status'],
                    reused_rounds=pkt['reused_rounds'], new_rounds=pkt['new_rounds'],
                    reused_signals_equal_saved=all(pkt['rounds'][m]['mean_record']['signal'] == saved[prefix]['packet']['rounds'][m]['mean_record']['signal']
                                                   for m in pkt['reused_rounds']),
                    b_intake_conditions=problems or 'PASS', b_bind_packet=b_bind(args, d/'out'/'MEAN_HANDOFF.json', True)))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    (out/'VALIDATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (out/'PLANS.json').write_text(json.dumps(plans, ensure_ascii=False, indent=1) + '\n')
    ok = (all(x['rows_equal_published_except_reuse_flag'] and x['b_intake_conditions'] == 'PASS' for x in report['saved'])
          and all(x['requests_equal_changed'] for x in report['variants'] if x['unsupported_rounds'] == [])
          and all(x['b_intake_conditions'] == 'PASS' for x in report['stub_runs']))
    print(json.dumps(dict(ok=ok, saved=[(x['prefix'], x['status'], x['b_bind_packet'] if isinstance(x['b_bind_packet'], str)
                                         else x['b_bind_packet']['status']) for x in report['saved']],
                          variants=[(x['case'], x['request_rounds'], x['unsupported_rounds'], x['requests_equal_changed']) for x in report['variants']],
                          stub=[(x['case'], x['stub_calls'], x['b_bind_packet'] if isinstance(x['b_bind_packet'], str)
                                 else x['b_bind_packet']['status']) for x in report['stub_runs']]), indent=1))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
