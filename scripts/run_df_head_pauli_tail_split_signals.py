"""PR-Pauli stage 2 for a saved split (LD5): 13 round means of the fixed configuration, then the packet.

Reuses the LD3 stage-2 runner (8c8ce542e) helpers unchanged: git/log helpers and the LD2
context-builder control against P1 ``load_contexts``. --execute: one fresh run of at most 13
sector36 means through the fixed P1 provider (no retry/resume). --assemble: metadata-only
MEAN_HANDOFF and ROUND_SUMMARY from the committed raw carrier. No sampling, build/compile,
price, N, Q3 or shot allocation.
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
from trottertracks import df_head_pauli_tail_split_signals as F
import run_df_head_pauli_tail_ld3_signals as R3

S3, L, p0, p1 = F.S3, F.L, F.p0, F.p1
CALLER = 'scripts/run_df_head_pauli_tail_split_signals.py'
LD3_SIGNALS = '8c8ce542e9fc25ead709fe6b0eb71b87ba7434ca'
HELPER = dict(commit=LD3_SIGNALS, path=S3.MODULE, function='b_intake_dictionary_identity')


def sources(source, spec):
    """Own files at SOURCE; LD3 stage-2/stage-1 modules, P1 kernel files byte-equal to their commits."""
    records = []
    for commit, path in [(source, F.MODULE), (source, CALLER), (LD3_SIGNALS, S3.MODULE), (LD3_SIGNALS, R3.CALLER),
            (S3.P1_KERNEL, p1.MODULE), (S3.P1_KERNEL, p1.provider.MODULE),
            (S3.P1_KERNEL, 'src/trottertracks/df_head_pauli_tail_p0.py'),
            (S3.P1_KERNEL, 'src/trottertracks/pf_power_stability.py'),
            (S3.P1_KERNEL, 'src/trottertracks/h4_four_method_screen_mean.py'),
            (S3.P1_KERNEL, 'src/trotterlib/rte.py'), (S3.STAGE1_CODE, L.MODULE)]:
        raw = p0.blob(ROOT, commit, path)
        if (ROOT/path).read_bytes() != raw:
            raise ValueError('USED_SCIENTIFIC_SOURCE_CHANGED: '+path)
        records.append(dict(commit=commit, path=path, sha256=hashlib.sha256(raw).hexdigest()))
    return records


def numerical(source):
    return dict(commit=source, path=p1.provider.MODULE, adapter=F.MODULE, kernel_source=S3.P1_KERNEL,
                recipe=p1.provider.RECIPE)


def execute(out, source, spec):
    out = Path(out)
    if out.exists():
        raise ValueError('FRESH_RUN_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    def save(name, value):
        with (out/name).open('w' if name == 'CHECKPOINT.json' else 'x') as f:
            f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    start = time.monotonic(); normalizers = {}; prefix = spec['prefix']
    result = dict(status='STARTED', stage='LD%d_SIGNALS' % prefix, scientific_source=source,
        numerical_provider_source=S3.P1_KERNEL, configuration_source=spec['config'], configuration_id=spec['config_id'],
        P0_source=spec['p0'], common_operator_source=S3.COMMON, B_source=p1.B,
        started_UTC=datetime.now(timezone.utc).isoformat(), records=[],
        scope=dict(new_mean_calls=0, new_complex_outputs=0, new_unique_normalizers=0, reused_means=0,
            new_full256_means=0, new_references=0, operator_dictionary_recomputations=0, eigensolves=0,
            path_sampling=0, build_compile=0, price=0, N_MSE_C_G=0, Q3=0, shot_allocation=0),
        sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None, automatic_retry=False, mandatory_stop=True)
    save('EXECUTION.json', result)
    try:
        R3.log('START sources', start)
        if R3.head() != source:
            raise ValueError('FIXED_CURRENT_SOURCE_REQUIRED')
        result['code_sources'] = sources(source, spec)
        R3.log('START fixed metadata', start)
        saved, item, config, b = F.load_fixed(ROOT, spec)
        jobs = {m: p1.job(config, m, numerical(source), b) for m in F.ROUNDS}
        save('PLAN.json', dict(configuration_source=spec['config'], configuration_id=config['id'],
            dictionary_identity=config['spec']['split']['tail_dictionary'], selected_rounds=list(F.ROUNDS),
            jobs=[dict(m=m, candidate_id=j['B_candidate_id'], T=j['physical_context']['physical_candidate']['T'],
                qPF=j['q_PF'], qRPE=j['q_RPE'], R_outer=j['physical_context']['physical_candidate']['R_outer'],
                structure=j['physical_context']['structure']) for m, j in jobs.items()],
            new_mean_limit=len(F.ROUNDS), unique_normalizer_limit=len(F.ROUNDS), source=source))
        R3.log('START LD2 context-builder control', start)
        result['context_control'] = R3.ld2_context_control(ROOT, saved)
        if result['context_control']['equal'] is not True:
            raise ValueError('CONTEXT_BUILDER_DIFFERS_FROM_P1_LOAD_CONTEXTS')
        R3.log('START LD%d context' % prefix, start)
        ctx = S3.load_context(ROOT, saved, item, normalizers, budget=len(F.ROUNDS))
        if ctx.tails[prefix]['dictionary_identity'] != config['spec']['split']['tail_dictionary']:
            raise ValueError('CONTEXT_DICTIONARY_IDENTITY_DIFFERS')
        def evaluate(m, job):
            R3.log('START LD%d m%d (q=%d, R=%d)' % (prefix, m, job['q_PF'], job['physical_context']['physical_candidate']['R_outer']), start)
            ref = p1.provider.saved_h4_reference(ROOT, job)
            result['scope']['new_mean_calls'] += 1
            value = p1.provider.evaluate(ctx, job, ref, root=ROOT)
            if value.get('signal') is not None:
                result['scope']['new_complex_outputs'] += 3
            R3.log(value['status']+' LD%d m%d' % (prefix, m), start)
            return value
        F.run(jobs, evaluate, save, result, spec)
        for s in [*saved['inputs'].values(), saved['context_source']]:
            if p0.sha(s['path']) != s['sha256']:
                raise ValueError('USED_INPUT_CHANGED')
        import numpy, scipy
        result['runtime'] = dict(python=sys.version, executable=sys.executable, numpy=numpy.__version__, scipy=scipy.__version__)
    except Exception as error:
        result['status'] = 'STOP_INPUT_SOURCE_OR_RUNTIME_FAILURE'
        result['error'] = dict(type=type(error).__name__, message=str(error))
    attempted = {r['m'] for r in result['records']}
    result['not_started'] = [m for m in F.ROUNDS if m not in attempted]
    result['scope']['new_unique_normalizers'] = len(normalizers)
    result['normalizer_cache'] = [dict(tau_hex=k[0], K=k[1], b=v if math.isfinite(v) else None,
        nonfinite=None if math.isfinite(v) else repr(v)) for k, v in normalizers.items()]
    result['completed_UTC'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.monotonic()-start
    save('result.json', result)
    R3.log('END '+result['status'], start)
    return result


def assemble(out, source, raw_commit, spec):
    """Metadata only: join committed raw records into the MEAN_HANDOFF packet."""
    out = Path(out); prefix = spec['prefix']; folder = spec['directory']
    if (out/'MEAN_HANDOFF.json').exists() or (out/'ROUND_SUMMARY.csv').exists():
        raise ValueError('FRESH_PACKET_REQUIRED')
    current = R3.head()
    if current != raw_commit or not R3.is_ancestor(source, current):
        raise ValueError('ASSEMBLE_RUNS_AT_RAW_CARRIER_DESCENDING_FROM_SOURCE')
    for path in (F.MODULE, CALLER):
        if (ROOT/path).read_bytes() != p0.blob(ROOT, source, path):
            raise ValueError('USED_SOURCE_CHANGED: '+path)
    saved, item, config, b = F.load_fixed(ROOT, spec)
    raw = L.read(ROOT, raw_commit, folder+'run01/result.json')
    if raw['scientific_source'] != source:
        raise ValueError('RAW_SOURCE_DIFFERS')
    jobs = {m: p1.job(config, m, numerical(source), b) for m in F.ROUNDS}
    records = {}
    for entry in raw['records']:
        m = entry['m']; carried = L.read(ROOT, raw_commit, folder+'run01/LD%d_m%d.json' % (prefix, m))
        if carried != entry or m in records:
            raise ValueError('RAW_CARRIER_RECORD_DIFFERS_OR_DUPLICATE: m='+str(m))
        records[m] = entry['record']
    value = F.packet(config, records, jobs, raw_commit=raw_commit, science_source=source, spec=spec, helper_source=HELPER)
    value['raw_status'] = raw['status']
    rows = [F.summary_row(records[m], carrier=raw_commit, path=folder+'run01/LD%d_m%d.json' % (prefix, m),
                          science_source=source, prefix=prefix) for m in sorted(records)]
    with (out/'MEAN_HANDOFF.json').open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (out/'ROUND_SUMMARY.csv').open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight', action='store_true'); modes.add_argument('--execute', action='store_true')
    modes.add_argument('--assemble', action='store_true')
    parser.add_argument('--prefix', type=int, required=True)
    parser.add_argument('--source'); parser.add_argument('--raw-commit'); parser.add_argument('--out')
    args = parser.parse_args()
    if args.prefix not in F.SPECS or args.prefix == 3:
        parser.error('a saved new split with a stage-1 spec is required (LD3 signals are published)')
    spec = F.SPECS[args.prefix]
    if args.preflight:
        saved, item, config, b = F.load_fixed(ROOT, spec)
        print(json.dumps(dict(status='CANONICAL_METADATA_ONLY_NO_MEAN_OR_ARRAY_LOAD', configuration_id=config['id'],
            candidate_ids=[r['candidate']['id'] for r in config['rounds']], R_outer=[r['R_outer'] for r in config['rounds']],
            dictionary_sha256=item['dictionary_sha256'], lambda_P=item['payload']['lambda_P'],
            scalar=config['spec']['split']['scalar'], new_mean_calls=0), indent=2))
    elif args.execute:
        if not args.source or not args.out:
            parser.error('--execute requires --source and --out')
        value = execute(args.out, args.source, spec)
        print(json.dumps(dict(status=value['status'], scope=value['scope'], error=value.get('error'),
            elapsed_seconds=value['elapsed_seconds']), indent=2), flush=True)
        sys.exit(0 if value['status'] == F.success(spec) else 2)
    else:
        if not args.source or not args.raw_commit or not args.out:
            parser.error('--assemble requires --source --raw-commit --out')
        value = assemble(args.out, args.source, args.raw_commit, spec)
        print(json.dumps(dict(status=value['status'], evaluated=value['packets'][0]['evaluated_rounds'],
            failed=value['packets'][0]['failed_rounds']), indent=2))
        sys.exit(0 if value['status'] == F.success(spec) else 2)
