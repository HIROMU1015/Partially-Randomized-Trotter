"""PR-Pauli signal entry for any B canonical PR-Pauli configuration (no split-specific runner).

    --plan      per-round REUSE / REQUEST / UNSUPPORTED and the provider support class
                (metadata only; no array, no mean).
    --execute   fresh run directory: CONFIGURATION.json, PLAN.json, re-bound carriers of REUSE
                rounds whose identity differs (metadata only), then the REQUEST rounds through the
                fixed provider only with --approve-new-means (no retry/resume). --dry-context builds
                the LD2 control, the context and the saved references and stops before any mean.
    --assemble  MEAN_HANDOFF (LD1-LD5 layout) + ROUND_SUMMARY from saved and committed carriers,
                checked with B's bind_packet conditions. All-identical reuse needs no run directory.

--config: a local JSON file or COMMIT:PATH holding a B canonical configuration, a MEAN_HANDOFF,
one packet, a B2 NEEDS candidate (configuration_id + signal_source) or a B spec for B generate.
No sampling, build/compile, price, N, Q3 or shot choice.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import pauli_signal_entry as E  # noqa: E402
import run_df_head_pauli_tail_ld3_signals as R3  # noqa: E402

S3, L, p0, p1 = E.S3, E.L, E.p0, E.p1
CALLER = 'scripts/run_pauli_signal_entry.py'
LD3_SIGNALS = '8c8ce542e9fc25ead709fe6b0eb71b87ba7434ca'
SUCCESS = 'ENTRY_CARRIERS_COMPLETE_NOT_RPE_QUALIFIED'


class Stop(Exception):
    """A planned stop; result['status'] already says why."""


def code_sources(source):
    """Own files at SOURCE; LD3 stage-2/stage-1 modules and P1 kernel files byte-equal to their commits."""
    records = []
    for commit, path in [(source, E.MODULE), (source, CALLER), (source, 'src/trottertracks/pauli_split_descriptor.py'),
                         (LD3_SIGNALS, S3.MODULE), (LD3_SIGNALS, R3.CALLER),
                         (S3.P1_KERNEL, p1.MODULE), (S3.P1_KERNEL, p1.provider.MODULE),
                         (S3.P1_KERNEL, 'src/trottertracks/df_head_pauli_tail_p0.py'),
                         *[(S3.P1_KERNEL, path) for path in E.KERNEL_FILES[1:]], (S3.STAGE1_CODE, L.MODULE)]:
        raw = p0.blob(ROOT, commit, path)
        if (ROOT/path).read_bytes() != raw:
            raise ValueError('USED_SCIENTIFIC_SOURCE_CHANGED: '+path)
        records.append(dict(commit=commit, path=path, sha256=hashlib.sha256(raw).hexdigest()))
    return records


def saved_packets(path):
    if not path:
        return E.SAVED_PACKETS
    with open(path) as f:
        return tuple(json.load(f))


def plan_only(config_ref, configuration_id=None, packets=E.SAVED_PACKETS):
    config, locator = E.load_configuration(ROOT, config_ref, configuration_id=configuration_id)
    return config, locator, E.plan(ROOT, config, packets=packets, configuration_source=locator)


def dump(path, value, mode='x'):
    with Path(path).open(mode) as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def execute(out, source, config_ref, *, configuration_id=None, packets=E.SAVED_PACKETS, approve=False, dry_context=False,
            evaluate_factory=None, check_source=True):
    """One fresh run. evaluate_factory(jobs) -> evaluate(m, job) replaces the provider in synthetic tests."""
    out = Path(out)
    if out.exists():
        raise ValueError('FRESH_RUN_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    def save(name, value):
        dump(out/name, value, 'w' if name == 'CHECKPOINT.json' else 'x')
    start = time.monotonic(); normalizers = {}
    result = dict(status='STARTED', stage='PAULI_SIGNAL_ENTRY', scientific_source=source, numerical_provider_source=S3.P1_KERNEL,
                  common_operator_source=S3.COMMON, B_source=p1.B, started_UTC=datetime.now(timezone.utc).isoformat(),
                  rebound=[], records=[], approved_new_means=bool(approve), dry_context=bool(dry_context),
                  scope=dict(new_mean_calls=0, new_complex_outputs=0, new_unique_normalizers=0, rebound_records=0,
                             new_references=0, operator_dictionary_recomputations=0, eigensolves=0, path_sampling=0,
                             build_compile=0, price=0, N_MSE_C_G=0, Q3=0, shot_allocation=0),
                  sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None, automatic_retry=False, mandatory_stop=True)
    save('EXECUTION.json', result)
    try:
        if check_source:
            R3.log('START sources', start)
            if E.git_head(ROOT) != source:
                raise ValueError('FIXED_CURRENT_SOURCE_REQUIRED')
            result['code_sources'] = code_sources(source)
        config, locator = E.load_configuration(ROOT, config_ref, configuration_id=configuration_id)
        result.update(configuration_id=config['id'], configuration_source=locator)
        save('CONFIGURATION.json', config)
        index = E.saved_index(ROOT, packets)
        pl = E.plan(ROOT, config, packets=packets, index=index, configuration_source=locator, source=source)
        save('PLAN.json', pl)
        R3.log('PLAN %s request=%s rebind=%s unsupported=%s' % (pl['counts'], pl['request_rounds'], pl['rebind_rounds'],
                                                                pl['unsupported_rounds']), start)
        if pl['unsupported_rounds']:
            result.update(status='STOP_UNSUPPORTED_ROUNDS', error=dict(rounds=pl['unsupported_rounds']))
            raise Stop
        E.write_carriers(ROOT, config, pl, index, save, result)
        result['scope']['rebound_records'] = len(result['rebound'])
        requested = pl['request_rounds']
        if requested and not (approve or dry_context):
            result.update(status='STOP_REQUESTS_NOT_APPROVED', pending_requests=requested)
            raise Stop
        b = p1.b_api(ROOT)
        jobs = {m: p1.job(config, m, E.numerical(source), b) for m in requested}
        if requested and evaluate_factory is None:
            saved = L.p0_metadata(ROOT, (2, config['spec']['L_D']))
            item = E.dictionary_item(ROOT, config)
            R3.log('START LD2 context-builder control', start)
            result['context_control'] = R3.ld2_context_control(ROOT, saved)
            if result['context_control']['equal'] is not True:
                raise ValueError('CONTEXT_BUILDER_DIFFERS_FROM_P1_LOAD_CONTEXTS')
            R3.log('START LD%d context' % config['spec']['L_D'], start)
            ctx = S3.load_context(ROOT, saved, item, normalizers, budget=E.normalizer_budget(jobs))
            if ctx.tails[config['spec']['L_D']]['dictionary_identity'] != config['spec']['split']['tail_dictionary']:
                raise ValueError('CONTEXT_DICTIONARY_IDENTITY_DIFFERS')
            refs = {m: p1.provider.saved_h4_reference(ROOT, job) for m, job in jobs.items()}
            if dry_context:
                result.update(status='DRY_CONTEXT_READY_NO_MEAN', dry_references={str(m): r['source'] for m, r in refs.items()})
                raise Stop
            def evaluate(m, job):
                R3.log('START m%d (q=%d, R=%d)' % (m, job['q_PF'], job['physical_context']['physical_candidate']['R_outer']), start)
                result['scope']['new_mean_calls'] += 1
                value = p1.provider.evaluate(ctx, job, refs[m], root=ROOT)
                if value.get('signal') is not None:
                    result['scope']['new_complex_outputs'] += 3
                R3.log(value['status'] + ' m%d' % m, start)
                return value
        elif requested:
            inner = evaluate_factory(jobs)
            def evaluate(m, job):
                result['scope']['new_mean_calls'] += 1
                return inner(m, job)
        if requested and not E.run_requests(config, pl, jobs, evaluate, save, result):
            raise Stop
        result['status'] = SUCCESS
    except Stop:
        pass
    except Exception as error:
        result['status'] = 'STOP_INPUT_SOURCE_OR_RUNTIME_FAILURE'
        result['error'] = dict(type=type(error).__name__, message=str(error))
    result['scope']['new_unique_normalizers'] = len(normalizers)
    result['normalizer_cache'] = [dict(tau_hex=k[0], K=k[1], b=v if math.isfinite(v) else None,
                                       nonfinite=None if math.isfinite(v) else repr(v)) for k, v in normalizers.items()]
    result['completed_UTC'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.monotonic()-start
    save('result.json', result)
    R3.log('END '+result['status'], start)
    return result


def assemble(out, config_ref, *, configuration_id=None, packets=E.SAVED_PACKETS, source=None, raw_commit=None, run_dir=None,
             git=True):
    """Metadata only. run_dir is the repository path of the committed --execute directory (with '/')."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if (out/'MEAN_HANDOFF.json').exists() or (out/'ROUND_SUMMARY.csv').exists():
        raise ValueError('FRESH_PACKET_REQUIRED')
    config, locator = E.load_configuration(ROOT, config_ref, configuration_id=configuration_id)
    index = E.saved_index(ROOT, packets)
    pl = E.plan(ROOT, config, packets=packets, index=index, configuration_source=locator, source=source or 'PLANNED')
    raw_result_source = None; carriers = lambda m: None
    if any(r.get('reuse') != 'IDENTICAL_IDENTITY' for r in pl['rounds']):
        if not (source and raw_commit and run_dir):
            raise ValueError('RUN_DIRECTORY_REQUIRED: --source --raw-commit --run-dir')
        if git and (R3.head() != raw_commit or not R3.is_ancestor(source, raw_commit)):
            raise ValueError('ASSEMBLE_RUNS_AT_RAW_CARRIER_DESCENDING_FROM_SOURCE')
        if git:
            for path in (E.MODULE, CALLER):
                if (ROOT/path).read_bytes() != p0.blob(ROOT, source, path):
                    raise ValueError('USED_SOURCE_CHANGED: '+path)
        read = (lambda path: L.read(ROOT, raw_commit, path)) if git else (lambda path: json.loads((ROOT/path).read_text()))
        raw = read(run_dir+'result.json'); saved_plan = read(run_dir+'PLAN.json')
        if (raw['scientific_source'] != source or raw['status'] != SUCCESS
                or raw['configuration_id'] != config['id'] or read(run_dir+'CONFIGURATION.json') != config
                or [(r['m'], r['decision'], r.get('reuse'), r['physical_key']) for r in saved_plan['rounds']]
                != [(r['m'], r['decision'], r.get('reuse'), r['physical_key']) for r in pl['rounds']]):
            raise ValueError('RAW_RUN_DIFFERS_FROM_THIS_PLAN')
        new = {e['m']: e for e in raw['records']}
        def carriers(m):
            item = read(run_dir+E.carrier_name(config['spec']['L_D'], m))
            if m in new and item != new[m]:
                raise ValueError('RAW_CARRIER_RECORD_DIFFERS: m=%d' % m)
            return item
        raw_result_source = dict(commit=raw_commit, path=run_dir+'result.json')
        if locator is None:
            locator = dict(commit=raw_commit, path=run_dir+'CONFIGURATION.json')
    rows = E.assemble_rows(ROOT, config, pl, carriers=carriers, raw_commit=raw_commit, directory=run_dir)
    value = E.packet(config, rows, raw_result_source=raw_result_source, configuration_source=locator)
    b = p1.b_api(ROOT)
    dictionary = L.read(ROOT, config['spec']['split']['tail_dictionary']['source']['commit'],
                        config['spec']['split']['tail_dictionary']['source']['path'])
    problems = E.b_intake_check(config, value, {m: r[0] for m, r in rows.items()}, dictionary, b)
    if value['packets'][0]['signal_coverage_complete'] and problems:
        raise ValueError('B_INTAKE_CONDITIONS_FAIL: ' + ','.join(problems))
    summary = []; decision = {r['m']: r.get('reuse') or r['decision'] for r in pl['rounds']}
    for m in sorted(rows):
        rec, src, reused = rows[m]
        row = S3.summary_row(rec, carrier=src['commit'], path=src['path'], science_source=src['scientific_source'])
        row.update(label='LD%d_m%d' % (config['spec']['L_D'], m), provenance=dict(
            IDENTICAL_IDENTITY='REUSED_SAVED_RECORD', REBOUND_SAME_PHYSICAL_KEY='REUSED_SAME_PHYSICAL_KEY_REBOUND',
            REQUEST='NEW_ENTRY_MEAN')[decision[m]])
        summary.append(row)
    dump(out/'MEAN_HANDOFF.json', value)
    with (out/'ROUND_SUMMARY.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n'); w.writeheader(); w.writerows(summary)
    return value, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('--plan', '--execute', '--assemble'):
        modes.add_argument(mode, action='store_true')
    parser.add_argument('--config', required=True); parser.add_argument('--configuration-id')
    parser.add_argument('--saved-packets', help='JSON list of MEAN_HANDOFF locators (default: LD1/LD2, LD3, LD5)')
    parser.add_argument('--tail-descriptor', help='A-prepared descriptor envelope for a split absent from the old saved context')
    parser.add_argument('--source'); parser.add_argument('--raw-commit'); parser.add_argument('--run-dir'); parser.add_argument('--out')
    parser.add_argument('--approve-new-means', action='store_true'); parser.add_argument('--dry-context', action='store_true')
    args = parser.parse_args(); packets = saved_packets(args.saved_packets)
    if args.execute and (not args.source or not args.out):
        parser.error('--execute requires --source and --out')
    if args.assemble and not args.out:
        parser.error('--assemble requires --out')
    if args.tail_descriptor:
        from trottertracks.pauli_split_descriptor import stage1_metadata
        with open(args.tail_descriptor) as f:
            descriptor = json.load(f)
        config, _ = E.load_configuration(ROOT, args.config, configuration_id=args.configuration_id)
        if descriptor['prefix'] != config['spec']['L_D'] or descriptor['binding'] != config['spec']['binding']:
            raise ValueError('PREPARED_DESCRIPTOR_H_OR_SPLIT_DIFFERS')
        # A descriptor envelope locates the committed P0 value, not an arbitrary replacement.
        if E.read_ref(ROOT, descriptor['source']) != descriptor['descriptor']:
            raise ValueError('PREPARED_DESCRIPTOR_SOURCE_DIFFERS')
        with stage1_metadata(ROOT, descriptor['prefix'], descriptor['descriptor'], descriptor['source']):
            return dispatch(args, packets)
    return dispatch(args, packets)


def dispatch(args, packets):
    """Keep the metadata context active through the actual plan/execute/assemble caller."""
    if args.plan:
        _, _, value = plan_only(args.config, args.configuration_id, packets)
        if args.out:
            Path(args.out).mkdir(parents=True, exist_ok=True); dump(Path(args.out)/'PLAN.json', value)
        print(json.dumps(dict({k: value[k] for k in ('configuration_id', 'prefix', 'counts', 'reuse_rounds', 'rebind_rounds',
                                                     'request_rounds', 'unsupported_rounds', 'configuration_reasons')},
                              support={r['m']: r['provider_support'] for r in value['rounds']}), indent=2))
    elif args.execute:
        value = execute(args.out, args.source, args.config, configuration_id=args.configuration_id, packets=packets,
                        approve=args.approve_new_means, dry_context=args.dry_context)
        print(json.dumps(dict(status=value['status'], rebound=value['rebound'], scope=value['scope'],
                              pending=value.get('pending_requests'), error=value.get('error')), indent=2), flush=True)
        sys.exit(0 if value['status'] in (SUCCESS, 'DRY_CONTEXT_READY_NO_MEAN') else 2)
    else:
        value, problems = assemble(args.out, args.config, configuration_id=args.configuration_id, packets=packets,
                                   source=args.source, raw_commit=args.raw_commit, run_dir=args.run_dir)
        pkt = value['packets'][0]
        print(json.dumps(dict(status=value['status'], reused=pkt['reused_rounds'], new=pkt['new_rounds'],
                              failed=pkt['failed_rounds'], b_intake_conditions=problems or 'PASS'), indent=2))
        sys.exit(0 if pkt['signal_coverage_complete'] else 2)


if __name__ == '__main__':
    main()
