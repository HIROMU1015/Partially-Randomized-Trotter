"""D3 preparation: saved key inventory and static literal-word certificate only.

Never constructs a provider, matrix, circuit, event, budget or new resource map.
Never enumerates pairs, calls a synthesis backend or launches a science worker.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.algorithm_codesign.rte_development.boundary import gate, inverse

BASE = '4cf2827c14fec96613a60bb1427287b30766d5da'
D1 = 'artifacts/track_b_rte_development_d1_result/2026-10-13/v1'
KEYS = 'artifacts/track_b_d2_key_mechanisms_preparation/2026-10-13/v1/rows'
OUT = 'artifacts/track_b_d3_preparation/2026-10-13/v1'
NEW_EPSILON = '1/10000000'


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def key_spec(ratio, epsilon, options, tool_sha):
    ratio, epsilon = F(ratio), F(epsilon)
    if ratio <= 0 or epsilon <= 0 or options.get('up_to_phase') is not False:
        raise ValueError('positive rational and strict phase required')
    return dict(angle_kind='atan', ratio=str(ratio), scale='1', epsilon=str(epsilon),
                options=options, tool_identity_sha256=tool_sha,
                relative_controlled_phase='strict; signed inverse from actual adjoint')


def key_hash(spec):
    # Match the exact whitespace and ordering of frozen D1 evidence.dumps.
    return sha(json.dumps(spec, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def frozen_read(root, relative, identities):
    data = (root / relative).read_bytes()
    original = subprocess.check_output(['git', '-C', str(root), 'show', BASE + ':' + relative])
    if data != original:
        raise ValueError('saved/source identity changed: ' + relative)
    identities[relative] = dict(bytes=len(data), sha256=sha(data), base_commit=BASE)
    return json.loads(data)


def no_rotation_cancellation_certificate(irs):
    """Set intersection of prefixes and inverse-reversed suffixes, not pairs.

    A cancellation of length n implies equality of those two n-token words.
    If an R-containing word cannot be in both sets, no pair can cancel an R.
    No probabilities, costs or maximum cancellation depths are calculated.
    """
    if not irs:
        raise ValueError('empty saved domain')
    prefixes, inverse_suffixes = set(), set()
    token_count = 0
    for raw in irs:
        ir = [gate(g) for g in raw]
        if not ir:
            raise ValueError('empty event IR')
        token_count += len(ir)
        for seq, target in ((ir, prefixes), ([inverse(g) for g in reversed(ir)], inverse_suffixes)):
            word = []
            contains_rotation = False
            for g in seq:
                word.append(g)
                contains_rotation |= g[0] == 'R'
                if contains_rotation:
                    target.add(tuple(word))
    common = prefixes & inverse_suffixes
    return dict(proof='prefix / inverse-reversed-suffix equality necessary for literal cancellation',
                saved_events=len(irs), saved_tokens=token_count,
                distinct_R_containing_prefixes=len(prefixes),
                distinct_R_containing_inverse_suffixes=len(inverse_suffixes),
                R_containing_intersections=len(common),
                no_R_can_be_cancelled=not common,
                event_pairs_evaluated=0, probabilities_or_costs_used=False,
                new_IR_generated=0,
                scope='all concatenations within this saved row; same literal inverse policy only')


def inventory_from_rows(rows, old_inventory, contract):
    old = {x['ratio']: x for x in old_inventory['keys']}
    if len(old) != len(old_inventory['keys']):
        raise ValueError('duplicate old key')
    used = {}
    for row in rows:
        for ratio in row['ratios']:
            if ratio not in old:
                raise ValueError('missing fixed old key')
            used.setdefault(ratio, []).append(row['row'])
    entries = []
    for ratio in sorted(used, key=F):
        old_spec = key_spec(ratio, contract['primitive_error'], contract['synthesizer_options'],
                            contract['tool_identity']['sha256'])
        if key_hash(old_spec) != old[ratio]['key']:
            raise ValueError('old inventory spec identity mismatch')
        spec = key_spec(ratio, NEW_EPSILON, contract['synthesizer_options'],
                        contract['tool_identity']['sha256'])
        entries.append(dict(ratio=ratio, rows=used[ratio], old_key=old[ratio]['key'],
                            new_key=key_hash(spec), new_spec=spec,
                            planned_acquisition='NEW_ONCE_AFTER_SEPARATE_AUTHORIZATION',
                            new_price=None, synthesis_calls_now=0))
    return dict(keys=entries, used_unique_keys=len(entries), rows=rows,
                old_cache_entries=len(old), unused_old_cache_ratios=sorted(set(old)-set(used), key=F),
                all_six_arms_same_precision_and_acquisition_rule=True,
                freeze_before_new_price=True, registered_new_synthesis=0,
                no_old_looser_sequence_substitution=True)


def prepare(root):
    root = Path(root)
    identities = {}
    c = frozen_read(root, 'artifacts/track_b_rte_development_source/2026-10-13/v1/contract_v1.json', identities)
    old_inventory = frozen_read(root, D1 + '/inventory.json', identities)
    frozen_read(root, c['tool_identity']['path'], identities)
    rows, certificates = [], []
    for index in range(12):
        relative = KEYS + f'/row_{index:02d}.json'
        row = frozen_read(root, relative, identities)
        rows.append(dict(row=index, steps=row['steps'], arm=row['arm'],
                         ratios=sorted(row['key_multiplicity_conditional'], key=F),
                         source=relative, primary=row['arm'] in c['primary_pair']))
        if index >= 6:
            relative = D1 + f'/events/row_{index:02d}.json'
            bindings = frozen_read(root, relative, identities)
            cert = no_rotation_cancellation_certificate([b['native_ir'] for b in bindings])
            cert.update(row=index, arm=row['arm'], source=relative,
                        source_sha256=identities[relative]['sha256'])
            certificates.append(cert)
    inventory = inventory_from_rows(rows, old_inventory, c)
    workers = frozen_read(root, D1 + '/resource_map.json', identities)['worker_receipts']
    performance = dict(old_new_synthesis_keys=len(workers),
                       old_worker_CPU_seconds_min=min(x['CPU_seconds'] for x in workers),
                       old_worker_CPU_seconds_max=max(x['CPU_seconds'] for x in workers),
                       old_worker_CPU_seconds_sum=sum(x['CPU_seconds'] for x in workers),
                       old_worker_peak_RSS_KiB_max=max(x['peak_RSS_KiB'] for x in workers),
                       timing_scope='saved D1 process receipts; not pure online sampler cost',
                       selected_workers=2, numeric_threads_per_worker=1,
                       matrix_validation_not_in_D3=True, pair_enumeration_not_in_D3=True,
                       speedup_claim=False, new_precision_runtime_unmeasured=True)
    return inventory, dict(rows=certificates,
                           all_rows_no_R_cancellation=all(x['no_R_can_be_cancelled'] for x in certificates),
                           registered_pairs_evaluated=0, new_price_evaluations=0,
                           D2_original_depth_cost_attribution_remains_unacquired=True), identities, performance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    output = args.root / OUT
    names = ['key_inventory_plan_v1.json', 'static_boundary_price_certificate_v1.json',
             'saved_inputs_identity_v1.json', 'performance_plan_inputs_v1.json']
    if any((output/name).exists() for name in names):
        raise FileExistsError('preparation output exists; no overwrite')
    inventory, certificate, identities, performance = prepare(args.root)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in zip(names, (inventory, certificate, dict(base_commit=BASE, files=identities), performance)):
        (output/name).write_bytes(encoded(value))
    print(json.dumps(dict(status='D3_STATIC_PLAN_INPUTS_PREPARED',
                          synthesis_keys=inventory['used_unique_keys'],
                          R_cancellation_excluded=certificate['all_rows_no_R_cancellation'],
                          new_science=0, mandatory_STOP=True)))


if __name__ == '__main__':
    main()
