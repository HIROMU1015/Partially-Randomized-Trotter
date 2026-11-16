"""PR-Pauli split stage 1 for a new prefix (LD5): P0 dictionary, then configuration and schedule.

Reuses the LD3 stage-1 extension unchanged (module df_head_pauli_tail_ld3 at 2e1949a0b and the
LD3 runner's writer/log/git helpers): the fixed P0 diagnose with only its prefix loop changed,
the PREPARE_MENU original-spec path and the fixed P1 generate_configuration. Saved LD3 results
(dictionary 50640b74c, configuration dd5b881d) are reused and reproduced as controls, not replaced.

--p0: prefixes (2, P); LD2 is the bitwise control against the saved 4220f3006 products.
--configure: metadata only, after the P0 carrier commit: original DF spec for P, Pauli
configuration and 13-round schedule; LD1/LD2/LD3 reproduction controls; LD3-vs-P table and
shot-weighted event totals for the saved VP and LD3 FULL_PF__B_PRICE allocations.
No mean, sampling, build/compile, QPY, price, Q3 or shot search. Fresh outputs; no retry.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import df_head_pauli_tail_ld3 as A
import run_df_head_pauli_tail_ld3 as R3
import run_df_head_pauli_tail_p0 as fixed_p0_runner

CALLER = 'scripts/run_df_head_pauli_tail_split_stage1.py'
STAGE1_CODE = '2e1949a0b66dd8fe086ac7cd3104eb89b4553e38'
CONTROL = 2
LD3 = dict(prefix=3, p0_commit='50640b74c65c36d381d6058824745f6e38ab3c2d',
           p0_directory='docs/research/df_head_pauli_tail_ld3_stage1_20261116/p0_run01/',
           config=dict(commit='421258d400a86fffd7c4955d5cf1bd24ac5729cb',
                       path='docs/research/df_head_pauli_tail_ld3_stage1_20261116/config_run01/CONFIGURATION_LD3.json'),
           original=dict(commit='421258d400a86fffd7c4955d5cf1bd24ac5729cb',
                         path='docs/research/df_head_pauli_tail_ld3_stage1_20261116/config_run01/ORIGINAL_DF_CONFIGURATION_LD3.json'),
           id='dd5b881d308f6fc48d3c91d08cec22c5fbbac296f60242e9bfb433c3e0321847')
ALLOCATIONS = dict(
    VP=dict(N=[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 13, 9],
            source=dict(commit='2de33a4f4', path='docs/research/pauli_ld3_vp_q3_results_20261116/C_RESULT_JOIN.json',
                        allocation_id='PAULI_LD3_USER_VP_MATCH_LD2_01')),
    FULL_PF__B_PRICE=dict(N=[48, 45, 42, 39, 36, 33, 29, 26, 23, 19, 16, 8, 2],
            source=dict(commit='fb4d78781', path='docs/research/pauli_ld3_pf_budget_q3_results_20261116/INPUTS.json',
                        selector=['cases', 3, 'allocation'], allocation_id='PAULI_LD3_FULL_PF__B_PRICE_VP_CAP_01')))


def directory(prefix):
    return 'docs/research/df_head_pauli_tail_ld%d_stage1_20261116/' % prefix


def success(prefix):
    return 'P0_LD%d_DICTIONARY_COMPLETE_NOT_P1_OR_RPE_QUALIFIED' % prefix


def own_sources(source):
    out = []
    for commit, path in ((STAGE1_CODE, A.MODULE), (STAGE1_CODE, R3.CALLER), (source, CALLER)):
        raw = A.p0.blob(ROOT, commit, path)
        if (ROOT/path).read_bytes() != raw:
            raise ValueError('USED_SOURCE_CHANGED: '+path)
        out.append(dict(commit=commit, path=path, sha256=hashlib.sha256(raw).hexdigest()))
    return out+A.fixed_modules(ROOT)


def dictionary_item(root, commit, folder, prefix, meta, status):
    """A.dictionary_item with this run's P0 status label (same checks, same item layout)."""
    result = A.read(root, commit, folder+'result.json')
    if (result['status'] != status or not result['checks'] or not all(c['PASS'] for c in result['checks'])
            or result['binding'] != meta['binding'] or result['control']['PASS'] is not True):
        raise ValueError('P0_EVIDENCE_NOT_AVAILABLE_OR_DIFFERENT')
    row = next(r for r in result['prefixes'] if r['prefix'] == prefix)
    path = folder+'LD'+str(prefix)+'_DICTIONARY.json'
    obj = A.read(root, commit, path)
    if (A.canonical_sha(obj) != row['dictionary_sha256'] or obj['binding'] != meta['binding']
            or obj['prefix'] != prefix or 'GATES_PASS' not in row['status']
            or obj['saved_DF_scalar'] != meta['saved_tail_descriptors'][str(prefix)]['scalar']):
        raise ValueError('P0_DICTIONARY_BINDING_OR_SCALAR')
    return dict(payload=obj, dictionary_sha256=row['dictionary_sha256'], source=dict(commit=commit, path=path),
        P0_metrics=row['identity_metrics'], P0_result_source=dict(commit=commit, path=folder+'result.json'))


def execute_p0(out, source, prefix):
    out = Path(out)
    if out.exists():
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    start = time.monotonic(); held = {}; prefixes = (CONTROL, prefix)
    keep = lambda name: not (name.startswith('LD'+str(CONTROL)+'_') or name == 'COMMON_OPERATORS.npz')
    save = R3.writer(out, held, keep)
    result = dict(status='STARTED', stage='P0_LD%d' % prefix, fixed_P0_source=A.P0_CODE, stage1_extension=STAGE1_CODE,
        scientific_source=source, started_UTC=datetime.now(timezone.utc).isoformat(),
        prefixes_run=list(prefixes), new_prefix=prefix, control_prefix=CONTROL, prefixes=[], checks=[],
        scope=dict(saved_input_loads=0, full_Fock_extensions=0, Pauli_dictionaries=0, mean_evaluations=0,
            reference_actions=0, eigensolves=0, finite_normalizers=0, sampling=0, circuit_build_compile=0,
            QPY_reads=0, price_Q3_shot_allocations=0),
        sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None,
        scientific_dictionary_adopted=False, automatic_retry=False, mandatory_stop=True)
    save('EXECUTION.json', result)
    try:
        R3.log('START sources', start)
        if R3.head() != source:
            raise ValueError('SOURCE_MUST_BE_FIXED_CURRENT_COMMIT')
        result['code_sources'] = own_sources(source)+fixed_p0_runner.sources(source)
        meta = A.p0_metadata(ROOT, prefixes)
        save('FIXED_INPUTS.json', meta)
        result['binding'] = meta['binding']
        R3.log('START saved input load', start)
        ham, sector, full_state, state, observed = A.p0.load_saved(ROOT, meta)
        result['scope']['saved_input_loads'] = 1
        result['saved_loader_observed'] = observed
        R3.log('START P0 diagnose prefixes %s' % (prefixes,), start)
        A.diagnose(ROOT, ham, sector, full_state, state, meta, result, save, prefixes=prefixes)
        result['scope']['Pauli_dictionaries'] = len(result['prefixes'])
        R3.log('START LD2/common bitwise control', start)
        result['control'] = A.saved_control(ROOT, result, held, CONTROL)
        save('CHECKPOINT.json', result)
        if not result['control']['PASS']:
            raise ValueError('LD2_OR_COMMON_PRODUCTS_DIFFER_FROM_SAVED_P0')
        for spec in [*meta['inputs'].values(), meta['context_source']]:
            if A.p0.sha(spec['path']) != spec['sha256']:
                raise ValueError('USED_INPUT_BYTES_CHANGED_DURING_P0')
        import numpy, scipy, openfermion
        from openfermion.config import EQ_TOLERANCE
        result['runtime'] = dict(python=sys.version, executable=sys.executable, numpy=numpy.__version__,
            scipy=scipy.__version__, openfermion=openfermion.__version__)
        result['coefficient_policy'] = dict(adapter_extra_cutoff=0., collector_imaginary_tolerance=1e-12,
            openfermion_EQ_TOLERANCE=EQ_TOLERANCE, openfermion_internal_simplification_preserved=True,
            rounded_imaginary_coefficients_recorded=True, rigorous_operator_certificate=False)
        result['status'] = success(prefix)
    except Exception as error:
        result['status'] = 'STOP_P0_LD%d_INPUT_OR_NUMERICAL_FAILURE' % prefix
        result['error'] = dict(type=type(error).__name__, message=str(error))
        for name, value in held.items():
            if not (out/('CONTROL_'+name)).exists():
                R3.writer(out)('CONTROL_'+name, value)
    result['completed_UTC'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.monotonic()-start
    save('result.json', result)
    R3.log('END '+result['status'], start)
    return result


def compare_table(old, new, a, b):
    """Per-round T, q, per-step R and whole-round q*R for two configurations; R=1 floor marked."""
    rows = []
    for x, y in zip(A.schedule(old), A.schedule(new), strict=True):
        if (x['m'], x['T'], x['q']) != (y['m'], y['T'], y['q']):
            raise ValueError('ROUND_GRID_DIFFERS')
        row = dict(m=x['m'], T_exact=str(x['T']), T=float(x['T']), q=x['q'])
        for label, z in ((a, x), (b, y)):
            row.update({'R_'+label: z['R'], 'events_'+label: z['events'],
                        'paper_total_target_'+label: z['paper_total_target'],
                        'R_floor_1_'+label: z['R'] == 1,
                        'events_minus_target_'+label: z['events']-z['paper_total_target'],
                        'physical_ID_'+label: z['physical_ID']})
        row['events_ratio_%s_over_%s' % (b, a)] = y['events']/x['events']
        row['events_reduction_%s_minus_%s' % (a, b)] = x['events']-y['events']
        rows.append(row)
    return rows


def weighted(rows, label):
    out = {}
    for name, alloc in ALLOCATIONS.items():
        n = alloc['N']
        out[name] = dict(N_per_axis=n, axes=['X', 'Y'], shots=2*sum(n), source=alloc['source'],
                         events=sum(2*n[r['m']]*r['events_'+label] for r in rows))
    return out


def execute_configure(out, source, p0_commit, prefix):
    out = Path(out)
    if out.exists():
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    start = time.monotonic(); save = R3.writer(out)
    p0_dir = directory(prefix)+'p0_run01/'; tag = 'LD%d' % prefix
    result = dict(status='STARTED', stage='CONFIGURE_'+tag, code_source=source,
        P0_source=dict(commit=p0_commit, path=p0_dir+'result.json'), fixed_P1_source=A.P1_CODE, B_source=A.p1.B,
        started_UTC=datetime.now(timezone.utc).isoformat(), checks={},
        scope=dict(metadata_only=True, mean_evaluations=0, sampling=0, circuit_build_compile=0, QPY_reads=0,
                   price_predictions=0, Q3=0, shot_allocations=0),
        sufficient_N=None, C=None, G=None, winner=None, mandatory_stop=True)
    save('EXECUTION.json', result)
    try:
        R3.log('START sources', start)
        current = R3.head()
        if current != p0_commit or not R3.is_ancestor(source, current):
            raise ValueError('CONFIGURE_RUNS_AT_P0_CARRIER_DESCENDING_FROM_SOURCE')
        result['code_sources'] = own_sources(source)
        meta = A.p0_metadata(ROOT, (1, 2, 3, prefix))
        p0_result = A.read(ROOT, p0_commit, p0_dir+'result.json')
        if p0_result['status'] != success(prefix) or p0_result['scientific_source'] != source or not p0_result['control']['PASS']:
            raise ValueError('P0_NOT_COMPLETE_OR_DIFFERENT_SOURCE')
        items = {p: A.dictionary_item(ROOT, A.SAVED_P0['commit'], A.SAVED_P0['directory'], p, meta) for p in (1, 2)}
        items[3] = A.dictionary_item(ROOT, LD3['p0_commit'], LD3['p0_directory'], 3, meta)
        items[prefix] = dictionary_item(ROOT, p0_commit, p0_dir, prefix, meta, success(prefix))
        R3.log('START original DF configurations (B 437ba4ac path)', start)
        originals = A.original_df_configurations(ROOT, (1, 2, 3, prefix))
        menu = {p: A.read(ROOT, A.p0.MENU, A.p0.DIRECTORY+'CONFIGURATION_LD'+str(p)+'.json') for p in (1, 2)}
        result['checks']['original_DF_LD1_LD2_equal_MENU_a9abf8de'] = all(originals[p] == menu[p] for p in (1, 2))
        result['checks']['original_DF_LD3_equal_saved_421258d40'] = originals[3] == A.read(ROOT, **LD3['original'])
        R3.log('START Pauli configurations (B ce0517de)', start)
        b = A.p1.b_api(ROOT)
        pauli = {p: A.generate_configuration(ROOT, b, originals[p], items[p]) for p in (1, 2, 3, prefix)}
        saved = A.read(ROOT, A.SAVED_PAULI['commit'], A.SAVED_PAULI['path'])['common_kappa_configurations']
        ld3 = A.read(ROOT, **LD3['config'])
        result['checks'].update(
            Pauli_LD1_LD2_equal_saved_P1_PLAN=all(pauli[p] == saved[str(p)] for p in (1, 2)),
            Pauli_LD3_equal_saved_dd5b881d=pauli[3] == ld3 and ld3['id'] == LD3['id'])
        new = pauli[prefix]; b.validate(new); s = new['spec']
        result['checks'].update({
            'validated_by_B': True,
            'id_differs_from_original_DF': new['id'] != originals[prefix]['id'],
            'L_D_split_candidate_prefix': (s['L_D'] == prefix and s['split']['L_D'] == prefix
                                           and all(r['candidate']['prefix'] == prefix for r in new['rounds'])),
            'lambda_is_dictionary_lambda_P': A.exact(s['split']['lambda_R']) == Fraction(items[prefix]['payload']['lambda_P']),
            'scalar_is_saved_DF_sR': s['split']['scalar'] == meta['saved_tail_descriptors'][str(prefix)]['scalar'],
            'same_exact_kappa_as_LD3': s['kappa'] == ld3['spec']['kappa'],
            'same_kappa_as_original_exact': A.exact(s['kappa']['value']) == Fraction(meta['original_exact_kappa']),
            'same_mapper_as_LD3': s['event_mapping'] == ld3['spec']['event_mapping'],
            'spec_differs_from_LD3_only_in': R3.spec_difference(ld3, new),
            '13_rounds': len(new['rounds']) == 13})
        if (not all(v for k, v in result['checks'].items() if k != 'spec_differs_from_LD3_only_in')
                or not set(result['checks']['spec_differs_from_LD3_only_in']) <= {'L_D', 'split', 'law'}):
            raise ValueError('CONFIGURATION_CHECK: '+json.dumps(result['checks']))
        rows = compare_table(ld3, new, 'LD3', tag)
        result['summary'] = dict(
            lambda_P=items[prefix]['payload']['lambda_P'], lambda_P_LD3=items[3]['payload']['lambda_P'],
            lambda_P_ratio_squared_over_LD3=(items[prefix]['payload']['lambda_P']/items[3]['payload']['lambda_P'])**2,
            saved_DF_lambda_R=meta['saved_tail_descriptors'][str(prefix)]['lambda_R'],
            saved_DF_lambda_R_source=meta['tail_descriptor_sources'][str(prefix)],
            pauli_identity=items[prefix]['payload']['pauli_identity'], saved_DF_scalar=items[prefix]['payload']['saved_DF_scalar'],
            nonidentity_terms=len(items[prefix]['payload']['terms']), dictionary_sha256=items[prefix]['dictionary_sha256'],
            dictionary_source=items[prefix]['source'], configuration_id=new['id'],
            original_DF_configuration_id=originals[prefix]['id'], configuration_id_LD3=ld3['id'],
            events_total_LD3=sum(r['events_LD3'] for r in rows), events_total=sum(r['events_'+tag] for r in rows),
            R_floor_rounds=[r['m'] for r in rows if r['R_floor_1_'+tag]],
            R_floor_rounds_LD3=[r['m'] for r in rows if r['R_floor_1_LD3']],
            shot_weighted_events=dict(LD3=weighted(rows, 'LD3'), **{tag: weighted(rows, tag)}))
        save('ORIGINAL_DF_CONFIGURATION_%s.json' % tag, originals[prefix])
        save('CONFIGURATION_%s.json' % tag, new)
        save('SCHEDULE_LD3_%s.json' % tag, dict(kind='PR_PAULI_LD3_%s_COMMON_KAPPA_SCHEDULE' % tag,
            LD3_configuration=dict(id=ld3['id'], source=LD3['config']),
            new_configuration=dict(id=new['id'], source=dict(path=directory(prefix)+'config_run01/CONFIGURATION_%s.json' % tag)),
            definitions=dict(R='B R_outer: integer tail RTE events per S2 step (step time T/q = 0.1)',
                events='q_PF*R_outer: tail RTE events of the whole round = B candidate event_count',
                R_floor_1='R = 1: the minimum of one tail event per step (paper target <= q)',
                events_minus_target='integerized whole-round events minus the paper target (rounding-up excess)',
                shot_weighted_events='sum over m of (N_X + N_Y) * events; an event count, not a price'),
            rows=rows, shot_weighted_events=result['summary']['shot_weighted_events']))
        with (out/('SCHEDULE_LD3_%s.csv' % tag)).open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
        result['status'] = '%s_CONFIGURATION_AND_SCHEDULE_COMPLETE_NOT_SIGNAL_OR_PRICE' % tag
    except Exception as error:
        result['status'] = 'STOP_%s_CONFIGURATION_FAILURE' % tag
        result['error'] = dict(type=type(error).__name__, message=str(error))
    result['completed_UTC'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.monotonic()-start
    save('result.json', result)
    R3.log('END '+result['status'], start)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true'); mode.add_argument('--p0', action='store_true')
    mode.add_argument('--configure', action='store_true')
    parser.add_argument('--prefix', type=int, required=True)
    parser.add_argument('--source'); parser.add_argument('--p0-commit'); parser.add_argument('--out')
    args = parser.parse_args()
    if args.prefix in (1, 2, 3) or not 3 < args.prefix < 10:
        parser.error('a new prefix 4..9 is required; LD1/LD2/LD3 are saved')
    if args.preflight:
        m = A.p0_metadata(ROOT, (1, 2, 3, args.prefix))
        print(json.dumps(dict(status='METADATA_PREFLIGHT_ONLY_NO_ARRAY_LOAD', prefix=args.prefix,
            descriptors=m['saved_tail_descriptors'], descriptor_sources=m['tail_descriptor_sources'],
            fixed_modules=A.fixed_modules(ROOT)), indent=2))
    else:
        if not args.source or not args.out or (args.configure and not args.p0_commit):
            parser.error('--p0 needs --source --out; --configure also needs --p0-commit')
        value = (execute_p0(args.out, args.source, args.prefix) if args.p0
                 else execute_configure(args.out, args.source, args.p0_commit, args.prefix))
        print(json.dumps(dict(status=value['status'], error=value.get('error'),
            elapsed_seconds=value['elapsed_seconds']), indent=2), flush=True)
        ok = success(args.prefix) if args.p0 else 'LD%d_CONFIGURATION_AND_SCHEDULE_COMPLETE_NOT_SIGNAL_OR_PRICE' % args.prefix
        sys.exit(0 if value['status'] == ok else 2)
