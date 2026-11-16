"""PR-Pauli LD3 stage 1 only: P0 dictionary, then B configuration and schedule.

--p0: fixed P0 leaf for prefixes (2, 3); LD2 is a bitwise control against the
saved 4220f3006 products and is not persisted again.
--configure: metadata only, after the P0 products are committed: original DF
LD3 spec, Pauli LD3 configuration and the 13-round schedule, with LD1/LD2
reproduction controls. No mean, sampling, build/compile, QPY, price, Q3 or
shot allocation. Fresh output directories; no retry or resume.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import df_head_pauli_tail_ld3 as A
import run_df_head_pauli_tail_p0 as fixed_p0_runner

CALLER = 'scripts/run_df_head_pauli_tail_ld3.py'
PREFIXES = (2, 3)
CONTROL = 2
NEW = 3
DIRECTORY = 'docs/research/df_head_pauli_tail_ld3_stage1_20261116/'
CONFIG_SUCCESS = 'LD3_CONFIGURATION_AND_SCHEDULE_COMPLETE_NOT_SIGNAL_OR_PRICE'
LD2_PAULI_SOURCE = dict(A.SAVED_PAULI, selector=['common_kappa_configurations', '2'])


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()


def is_ancestor(source, current):
    return subprocess.run(['git', 'merge-base', '--is-ancestor', source, current], cwd=ROOT).returncode == 0


def own_sources(source):
    out = []
    for path in (A.MODULE, CALLER):
        raw = A.p0.blob(ROOT, source, path)
        if (ROOT/path).read_bytes() != raw:
            raise ValueError('USED_SOURCE_CHANGED: '+path)
        out.append(dict(commit=source, path=path, sha256=hashlib.sha256(raw).hexdigest()))
    return out+A.fixed_modules(ROOT)


def log(message, start):
    print('[%7.2fs] %s' % (time.monotonic()-start, message), flush=True)


def writer(out, held=None, persist_names=None):
    def save(name, value):
        if held is not None and not persist_names(name):
            held[name] = value
            return
        if name.endswith('.npz'):
            import numpy as np
            with (out/name).open('xb') as f:
                np.savez_compressed(f, **value)
        else:
            with (out/name).open('w' if name == 'CHECKPOINT.json' else 'x') as f:
                f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    return save


def execute_p0(out, source):
    out = Path(out)
    if out.exists():
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    start = time.monotonic(); held = {}
    # Control (LD2) and common products are compared with 4220f3006, not re-saved.
    keep = lambda name: not (name.startswith('LD'+str(CONTROL)+'_') or name == 'COMMON_OPERATORS.npz')
    save = writer(out, held, keep)
    result = dict(status='STARTED', stage='P0_LD3', plan_source='a50dce8f1cbb38b2914bb2fb4ff79bab35e6e8f6',
        fixed_P0_source=A.P0_CODE, scientific_source=source, started_UTC=datetime.now(timezone.utc).isoformat(),
        prefixes_run=list(PREFIXES), new_prefix=NEW, control_prefix=CONTROL, prefixes=[], checks=[],
        scope=dict(saved_input_loads=0, full_Fock_extensions=0, Pauli_dictionaries=0, mean_evaluations=0,
            reference_actions=0, eigensolves=0, finite_normalizers=0, sampling=0, circuit_build_compile=0,
            QPY_reads=0, price_Q3_shot_allocations=0),
        sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None,
        scientific_dictionary_adopted=False, automatic_retry=False, mandatory_stop=True)
    save('EXECUTION.json', result)
    try:
        log('START sources', start)
        if head() != source:
            raise ValueError('SOURCE_MUST_BE_FIXED_CURRENT_COMMIT')
        result['code_sources'] = own_sources(source)+fixed_p0_runner.sources(source)
        meta = A.p0_metadata(ROOT, PREFIXES)
        save('FIXED_INPUTS.json', meta)
        result['binding'] = meta['binding']
        log('START saved input load', start)
        ham, sector, full_state, state, observed = A.p0.load_saved(ROOT, meta)
        result['scope']['saved_input_loads'] = 1
        result['saved_loader_observed'] = observed
        log('START P0 diagnose prefixes %s' % (PREFIXES,), start)
        A.diagnose(ROOT, ham, sector, full_state, state, meta, result, save, prefixes=PREFIXES)
        result['scope']['Pauli_dictionaries'] = len(result['prefixes'])
        log('START LD2/common bitwise control', start)
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
        result['coefficient_policy'] = dict(adapter_extra_cutoff=0.,
            collector_imaginary_tolerance=1e-12, openfermion_EQ_TOLERANCE=EQ_TOLERANCE,
            openfermion_internal_simplification_preserved=True,
            rounded_imaginary_coefficients_recorded=True, rigorous_operator_certificate=False)
        result['status'] = A.LD3_P0_SUCCESS
    except Exception as error:
        result['status'] = 'STOP_P0_LD3_INPUT_OR_NUMERICAL_FAILURE'
        result['error'] = dict(type=type(error).__name__, message=str(error))
        # Preserve held control products for diagnosis; never repaired or retried.
        for name, value in held.items():
            if not (out/('CONTROL_'+name)).exists():
                writer(out)('CONTROL_'+name, value)
    result['completed_UTC'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.monotonic()-start
    save('result.json', result)
    log('END '+result['status'], start)
    return result


def spec_difference(left, right):
    return sorted(k for k in set(left['spec']) | set(right['spec']) if left['spec'].get(k) != right['spec'].get(k))


def table(ld2, ld3):
    rows = []
    for a, b in zip(A.schedule(ld2), A.schedule(ld3), strict=True):
        if (a['m'], a['T'], a['q']) != (b['m'], b['T'], b['q']):
            raise ValueError('LD2_LD3_ROUND_GRID_DIFFERS')
        rows.append(dict(m=a['m'], T_exact=str(a['T']), T=float(a['T']), q=a['q'],
            R_LD2=a['R'], R_LD3=b['R'], R_ratio_LD3_over_LD2=b['R']/a['R'],
            events_LD2=a['events'], events_LD3=b['events'], events_ratio_LD3_over_LD2=b['events']/a['events'],
            events_reduction_LD2_minus_LD3=a['events']-b['events'],
            paper_total_target_LD2=a['paper_total_target'], paper_total_target_LD3=b['paper_total_target'],
            physical_ID_LD2=a['physical_ID'], physical_ID_LD3=b['physical_ID']))
    return rows


def execute_configure(out, source, p0_commit):
    out = Path(out)
    if out.exists():
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    start = time.monotonic(); save = writer(out)
    p0_dir = DIRECTORY+'p0_run01/'
    result = dict(status='STARTED', stage='CONFIGURE_LD3', plan_source='a50dce8f1cbb38b2914bb2fb4ff79bab35e6e8f6',
        code_source=source, P0_LD3_source=dict(commit=p0_commit, path=p0_dir+'result.json'),
        fixed_P1_source=A.P1_CODE, B_source=A.p1.B, started_UTC=datetime.now(timezone.utc).isoformat(),
        checks={}, scope=dict(metadata_only=True, mean_evaluations=0, sampling=0, circuit_build_compile=0,
            QPY_reads=0, price_predictions=0, Q3=0, shot_allocations=0),
        sufficient_N=None, C=None, G=None, winner=None, mandatory_stop=True)
    save('EXECUTION.json', result)
    try:
        log('START sources', start)
        current = head()
        if current != p0_commit or not is_ancestor(source, current):
            raise ValueError('CONFIGURE_RUNS_AT_P0_CARRIER_DESCENDING_FROM_SOURCE')
        result['code_sources'] = own_sources(source)
        meta = A.p0_metadata(ROOT, (1, 2, 3))
        p0_result = A.read(ROOT, p0_commit, p0_dir+'result.json')
        if (p0_result['status'] != A.LD3_P0_SUCCESS or p0_result['scientific_source'] != source
                or not p0_result['control']['PASS']):
            raise ValueError('P0_LD3_NOT_COMPLETE_OR_DIFFERENT_SOURCE')
        log('START dictionary items', start)
        items = {p: A.dictionary_item(ROOT, A.SAVED_P0['commit'], A.SAVED_P0['directory'], p, meta) for p in (1, 2)}
        items[3] = A.dictionary_item(ROOT, p0_commit, p0_dir, 3, meta)
        log('START original DF configurations (B 437ba4ac path)', start)
        originals = A.original_df_configurations(ROOT, (1, 2, 3))
        menu = {p: A.read(ROOT, A.p0.MENU, A.p0.DIRECTORY+'CONFIGURATION_LD'+str(p)+'.json') for p in (1, 2)}
        result['checks']['original_DF_LD1_LD2_equal_MENU_a9abf8de'] = all(originals[p] == menu[p] for p in (1, 2))
        log('START Pauli configurations (B ce0517de)', start)
        b = A.p1.b_api(ROOT)
        pauli = {p: A.generate_configuration(ROOT, b, originals[p], items[p]) for p in (1, 2, 3)}
        saved = A.read(ROOT, A.SAVED_PAULI['commit'], A.SAVED_PAULI['path'])['common_kappa_configurations']
        result['checks']['Pauli_LD1_LD2_equal_saved_P1_PLAN'] = all(pauli[p] == saved[str(p)] for p in (1, 2))
        result['checks']['Pauli_LD1_LD2_saved_IDs'] = all(pauli[p]['id'] == A.SAVED_PAULI_IDS[p] for p in (1, 2))
        ld2, ld3 = saved['2'], pauli[3]
        b.validate(ld3)
        s = ld3['spec']
        result['checks'].update(
            LD3_validated_by_B=True,
            LD3_id_differs_from_original_DF=ld3['id'] != originals[3]['id'],
            LD3_L_D_split_candidate_prefix=(s['L_D'] == 3 and s['split']['L_D'] == 3
                and all(r['candidate']['prefix'] == 3 for r in ld3['rounds'])),
            LD3_lambda_is_dictionary_lambda_P=A.exact(s['split']['lambda_R']) == Fraction(items[3]['payload']['lambda_P']),
            LD3_scalar_is_saved_DF_sR=s['split']['scalar'] == meta['saved_tail_descriptors']['3']['scalar'],
            LD3_same_exact_kappa_as_LD2=s['kappa'] == ld2['spec']['kappa'],
            LD3_same_kappa_as_original_exact=(A.exact(s['kappa']['value'])
                == Fraction(meta['original_exact_kappa'])),
            LD3_same_mapper_as_LD2=s['event_mapping'] == ld2['spec']['event_mapping'],
            LD3_spec_differs_from_LD2_only_in=spec_difference(ld2, ld3),
            LD3_13_rounds=len(ld3['rounds']) == 13)
        allowed = {'L_D', 'split', 'law'}
        if (not all(v for k, v in result['checks'].items() if k != 'LD3_spec_differs_from_LD2_only_in')
                or not set(result['checks']['LD3_spec_differs_from_LD2_only_in']) <= allowed):
            raise ValueError('LD3_CONFIGURATION_CHECK: '+json.dumps(result['checks']))
        rows = table(ld2, ld3)
        total2 = sum(r['events_LD2'] for r in rows); total3 = sum(r['events_LD3'] for r in rows)
        late = [r for r in rows if r['m'] >= 10]
        result['summary'] = dict(
            lambda_P_LD3=items[3]['payload']['lambda_P'], lambda_P_LD2=items[2]['payload']['lambda_P'],
            lambda_P_ratio_squared=(items[3]['payload']['lambda_P']/items[2]['payload']['lambda_P'])**2,
            saved_DF_lambda_R_LD3=meta['saved_tail_descriptors']['3']['lambda_R'],
            saved_DF_lambda_R_source=meta['tail_descriptor_sources']['3'],
            pauli_identity_LD3=items[3]['payload']['pauli_identity'],
            saved_DF_scalar_LD3=items[3]['payload']['saved_DF_scalar'],
            nonidentity_terms_LD3=len(items[3]['payload']['terms']),
            dictionary_sha256_LD3=items[3]['dictionary_sha256'], dictionary_source_LD3=items[3]['source'],
            configuration_id_LD3=ld3['id'], original_DF_configuration_id_LD3=originals[3]['id'],
            configuration_id_LD2=ld2['id'],
            events_total_LD2=total2, events_total_LD3=total3, events_total_ratio=total3/total2,
            events_m10_m12_LD2=sum(r['events_LD2'] for r in late), events_m10_m12_LD3=sum(r['events_LD3'] for r in late))
        save('ORIGINAL_DF_CONFIGURATION_LD3.json', originals[3])
        save('CONFIGURATION_LD3.json', ld3)
        save('SCHEDULE_LD2_LD3.json', dict(kind='PR_PAULI_LD2_LD3_COMMON_KAPPA_SCHEDULE',
            LD2_configuration=dict(id=ld2['id'], source=LD2_PAULI_SOURCE),
            LD3_configuration=dict(id=ld3['id'], source=dict(path=DIRECTORY+'config_run01/CONFIGURATION_LD3.json')),
            definitions=dict(R='B R_outer: integer tail RTE events per S2 step (step time T/q = 0.1)',
                events='q_PF*R_outer: tail RTE events of the whole round (q_PF steps) = B candidate event_count',
                paper_total_target='B paper_event_mapper total before uniform per-step integerization'),
            rows=rows, totals=dict(LD2=total2, LD3=total3)))
        with (out/'SCHEDULE_LD2_LD3.csv').open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)
        result['status'] = CONFIG_SUCCESS
    except Exception as error:
        result['status'] = 'STOP_LD3_CONFIGURATION_FAILURE'
        result['error'] = dict(type=type(error).__name__, message=str(error))
    result['completed_UTC'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.monotonic()-start
    save('result.json', result)
    log('END '+result['status'], start)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--p0', action='store_true')
    mode.add_argument('--configure', action='store_true')
    parser.add_argument('--source'); parser.add_argument('--p0-commit'); parser.add_argument('--out')
    args = parser.parse_args()
    if args.preflight:
        m = A.p0_metadata(ROOT, (1, 2, 3))
        print(json.dumps(dict(status='METADATA_PREFLIGHT_ONLY_NO_ARRAY_LOAD', binding_snapshot=m['binding']['snapshot'],
            prefixes_run=list(PREFIXES), descriptors=m['saved_tail_descriptors'],
            original_DF_configuration_ids=m['original_DF_configuration_ids'],
            fixed_modules=A.fixed_modules(ROOT)), indent=2))
    else:
        if not args.source or not args.out or (args.configure and not args.p0_commit):
            parser.error('--p0 needs --source --out; --configure also needs --p0-commit')
        value = execute_p0(args.out, args.source) if args.p0 else execute_configure(args.out, args.source, args.p0_commit)
        print(json.dumps(dict(status=value['status'], error=value.get('error'),
            elapsed_seconds=value['elapsed_seconds']), indent=2), flush=True)
        ok = A.LD3_P0_SUCCESS if args.p0 else CONFIG_SUCCESS
        sys.exit(0 if value['status'] == ok else 2)
