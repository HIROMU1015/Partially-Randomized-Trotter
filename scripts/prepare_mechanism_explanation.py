#!/usr/bin/env python3
"""Copy saved explanation fields and identities only; no scientific arithmetic."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path

SCHEMA = 'TRACK_A_MECHANISM_EXPLANATION_V2'
AXES = ('cosine', 'sine')
CHANNEL = dict(cosine='real', sine='imag')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_float=str)


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def records(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get('signals'), list):
        return data['signals']
    if isinstance(data, dict) and 'cell' in data:
        return [data]
    raise ValueError('UNSUPPORTED_SIGNAL_CONTAINER')


def identity(row, method):
    return {k: row[k] for k in ('contrast_id', 'system', 'H4_stage', 'epsilon',
            'factor', 'bias_mode', 'PR_role', 'PR_cohort', 'PR_id', 'PF_id',
            'binding')} | dict(method=method,
                candidate_id=row[method + '_id'], axes=list(AXES))


def build(previous_binding, core_results, lineage_inventory, source_snapshots, output):
    output.mkdir(mode=0o700)
    store = output / 'inputs'
    store.mkdir()
    inputs = []

    def snapshot(path, alias, expected=None):
        path = Path(path)
        digest = sha(path)
        assert expected is None or digest == expected, ('INPUT_SHA', alias)
        target = store / digest
        if not target.exists():
            target.write_bytes(path.read_bytes())
        inputs.append(dict(alias=alias, path=str(target), sha256=digest,
                           origin=str(path)))
        return read(target), target

    binding, _ = snapshot(previous_binding, 'PREVIOUS_BINDING')
    ix = {i['alias']: i for i in binding['input_files']}
    cache = {}

    def prior(alias):
        if alias not in cache:
            x = ix[alias]
            cache[alias] = snapshot(x['path'], alias, x['sha256'])[0]
        return cache[alias]

    payload, _ = snapshot(binding['payload'], 'PREVIOUS_PAYLOAD', binding['payload_sha256'])
    core, core_snapshot = snapshot(core_results, 'FIXED_CORE_RESULTS')
    assert [r['contrast_id'] for r in core] == binding['target_ids']
    assert len(core) == 32 and all(r['core_status'] == 'CORE_COMPLETE' for r in core)
    lineage, _ = snapshot(lineage_inventory, 'LINEAGE_INVENTORY')
    h6 = next(x for x in lineage if x['system'] == 'H6' and x['role'] == 'input_4')
    h6_rows, _ = snapshot(h6['snapshot'], 'H6_SAVED_V1_ACCOUNTING', h6['sha256'])
    commit = '8601b921342fc1cc83780a830e1f65bf64fe9e93'
    root = source_snapshots / commit / 'artifacts/resource_applicability'
    h4_rows, _ = snapshot(root/'track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/candidate_accounting_v1.json', 'H4_D0_ACCOUNTING')
    h4_contract, _ = snapshot(root/'track_a_d0_saved_value_preflight/2026-10-14/d0_reanalysis_contract_v1.json', 'H4_D0_CONTRACT')
    h8_rows, _ = snapshot(root/'track_a_h8_d0_saved_value_execution_v1/2026-10-15/candidate_accounting_v1.json', 'H8_D0_ACCOUNTING')
    h8_contract, _ = snapshot(root/'track_a_h8_d0_saved_value_execution_v1/2026-10-15/contract_v1.json', 'H8_D0_CONTRACT')
    h4_signal_receipt = next(x for x in h4_contract['inputs'] if x['alias'] == 'h4_signals.json')
    assert h4_signal_receipt['sha256'] == ix['H4/old_signals']['sha256']
    h4_scope = h4_contract['scopes']['H4_NATIVE_FIXED16']
    h8_science = h8_contract['binding']
    rows, joins = [], []
    special = {'A18_OLD_PR', 'A58_OLD_PR', 'A82_OLD_PR', 'A98_OLD_PR'}
    aliases = dict(NEW_A_SAVED_SIGNAL='A_signal',
        H4_ORIGINAL_D_SIGNALS='D_SIGNALS', H6_V1_additional='V1_additional',
        H6_V1_old_signals='V1_old_signals', H6_V1_b0i_signals='V1_b0i_signals',
        H8_FIXED_PHASE_A_SIGNALS='PHASE_A_SIGNALS')
    for old in payload['comparisons']:
        row = {k: copy.deepcopy(old[k]) for k in ('artificial','contrast_id',
            'system','H4_stage','epsilon','factor','bias_mode','PR_role','PR_id',
            'PF_id','binding','cohorts_pooled')}
        row['PR_cohort'] = old['PR']['cohort']
        row['sides'] = {}
        for method in ('PR', 'PF'):
            previous = old[method]
            desc = previous.get('explanation')
            signal = None
            if desc is not None:
                label = desc['source_signal']
                alias = old['system'] + '/' + aliases.get(label, label)
                signal = records(prior(alias))[desc['source_signal_record']]
                assert signal['cell']['id'] == previous['candidate_id']
                assert signal['cell'] == desc['candidate_cell']
            ac = None
            ac_ref = None
            if signal is not None:
                ac = signal.get('precision_rows', {}).get(old['epsilon'], {}).get('1.0')
                ac_ref = dict(kind='ORIGINAL_SIGNAL_PRECISION_ROW', alias=alias,
                    signal_record=desc['source_signal_record'], epsilon=old['epsilon'], factor='1.0')
            if method == 'PR' and row['contrast_id'] in special:
                cid = old['PR_id']; eps = old['epsilon']; system = old['system']
                if system == 'H4':
                    candidates = h4_rows['H4_NATIVE_FIXED16']
                    ref = 'H4_D0_ACCOUNTING/H4_NATIVE_FIXED16'
                    for key, saved_key in [('hamiltonian','hamiltonian_sha256'), ('state','state_sha256'), ('snapshot','snapshot_sha256')]:
                        assert old['binding'][key] == h4_scope[saved_key]
                elif system == 'H6':
                    candidates = [r for r in h6_rows['rows'] if r.get('sample_scope') == 'H6_CONFIRMATION_UNION8' and r.get('bias_rule') == 'signed_signal']
                    ref = 'H6_SAVED_V1_ACCOUNTING/H6_CONFIRMATION_UNION8'
                else:
                    candidates = h8_rows['H8_CONFIRMATION_UNION6']
                    ref = 'H8_D0_ACCOUNTING/H8_CONFIRMATION_UNION6'
                    assert old['binding']['snapshot'] == h8_science['input']
                    assert old['binding']['compiler'] == h8_science['compiler']
                selected = [(i,r) for i,r in enumerate(candidates) if r['cell_id'] == cid and str(r['epsilon']) == eps and str(r['factor']) in ('1','1.0')]
                assert len(selected) == 1
                record_index, saved = selected[0]
                if signal is not None:
                    assert signal['cell'] == saved['cell']
                ac = saved['accounting']
                ac_ref = dict(kind='SAVED_USUAL_SIGNED_CONFIRMATION_ACCOUNTING', reference=ref, record=record_index,
                    candidate_id=cid, epsilon=eps, factor='1', old_cohort=old['PR']['cohort'])
            if ac is not None:
                assert {a: ac['axes'][ch]['shots'] for a,ch in CHANNEL.items()} == previous['shots'], ('SAVED_SHOT_COPY_BINDING',row['contrast_id'],method)
            norm = None
            signed = None
            if signal is not None:
                norm = dict(law='FINITE_RTE' if method == 'PR' else 'DETERMINISTIC_PF',
                    step_b=signal.get('b'), step_count=signal['cell']['R'],
                    log_B_nominal=signal.get('log_B'), log_B_margin=signal.get('allowance',{}).get('log_B_margin'),
                    factor='1', upper_route='EXP_OF_LOG_UPPER', normalization_upper_saved=None)
                needed_norm = ('step_b','step_count','log_B_nominal','log_B_margin') if method == 'PR' else ('log_B_nominal','log_B_margin')
                if any(norm[k] is None for k in needed_norm):
                    norm = None
                signed = {a:signal['error_decomposition']['total_signed'].get(ch) for a,ch in CHANNEL.items()}
            side = dict(identity=identity(row, method), eligible=previous['eligible'],
                normalization=norm, signed_bias=signed,
                allowance=None if ac is None else {a:ac['axes'][ch]['allowance'] for a,ch in CHANNEL.items()},
                headroom=None if ac is None else {a:ac['axes'][ch]['headroom'] for a,ch in CHANNEL.items()},
                alpha_axis=None if ac is None else ac['alpha_axis'],
                log_shot_bound=None if ac is None else {a:ac['axes'][ch]['log_shot_bound'] for a,ch in CHANNEL.items()},
                integer_status=None if ac is None else {a:ac['axes'][ch]['integer_status'] for a,ch in CHANNEL.items()},
                shots=copy.deepcopy(previous['shots']))
            row['sides'][method] = side
            missing = [k for k in ('normalization','signed_bias','allowance','headroom','alpha_axis','log_shot_bound','integer_status') if side[k] is None]
            joins.append(dict(contrast_id=row['contrast_id'], method=method,
                side_id=row['contrast_id']+'.'+method, candidate_id=previous['candidate_id'],
                status='INPUT_FIELDS_PRESENT_NOT_NUMERICALLY_VALIDATED' if not missing else 'INPUT_NOT_RECOVERED',
                missing_fields=missing, source_signal=None if signal is None else ac_ref.get('alias', desc['source_signal']),
                accounting_reference=ac_ref, saved_shots_match=True))
        rows.append(row)
    data = dict(schema=SCHEMA,kind='SAVED_REAL_VALUES',target_ids=binding['target_ids'],comparisons=rows)
    payload_path = output/'explanation_payload_UNEXECUTED.json'
    save(payload_path,data)
    inputs.append(dict(alias='EXPLANATION_PAYLOAD',path=str(payload_path),sha256=sha(payload_path)))
    result = dict(schema=SCHEMA,kind='SAVED_REAL_VALUES',status='READY_FOR_USER_AUTHORIZATION',
        target_ids=binding['target_ids'],method_side_ids=[x['side_id'] for x in joins],
        core_results=str(core_snapshot),core_results_sha256=sha(core_snapshot),
        payload=str(payload_path),payload_sha256=sha(payload_path),input_files=inputs,
        joins=joins,allowed_missing_side_ids=[x['side_id'] for x in joins if x['missing_fields']],
        protected_paths=[str(previous_binding.parent.parent),str(core_results.parent)],
        scientific_numerical_reanalysis_executed=0,new_statistics=0)
    save(output/'explanation_input_binding_DRAFT.json',result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ('previous-binding','core-results','lineage-inventory','source-snapshots','output'):
        parser.add_argument('--'+option,type=Path,required=True)
    args=parser.parse_args()
    result=build(args.previous_binding,args.core_results,args.lineage_inventory,args.source_snapshots,args.output)
    print(json.dumps(dict(contrasts=len(result['target_ids']),method_sides=len(result['method_side_ids']),
        missing_side_ids=result['allowed_missing_side_ids'],scientific_reanalysis=0)))
