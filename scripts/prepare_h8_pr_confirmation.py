"""Byte-only H8 n48 seal and seed registry; creates invalid grants only.

Allocates deterministic seed identifiers without drawing any Taylor/component
events. No native module, scientific array, preparation or accounting execution.
"""
from copy import deepcopy
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import h8_pr_confirmation as h
c = h.c


def row(path, role):
    path = Path(path)
    c.require(path.is_file() and not path.is_symlink(), 'REGULAR_INPUT')
    return dict(role=role, path=str(path), bytes=path.stat().st_size, sha256=c.sha(path))


def prepare(accounting_spec, destination, source):
    destination = c.home_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    design = c.read(h.ROOT / h.DESIGN)
    contents = dict(design)
    digest = contents.pop('design_digest')
    c.require(c.digest(contents) == digest, 'FIXED_PUBLISHED_DESIGN')
    accounting = c.read(accounting_spec)
    by_role = {r['role']: r for r in accounting['inputs']}
    c.verify_byte_rows(accounting['inputs'])
    origin = c.read(by_role['origin_A_spec']['path'])
    c.require(origin['scope'] == 'H8_A_COST' and origin['binding'] == design['binding'], 'ORIGINAL_H8_BINDING')
    c.verify_byte_rows(origin['inputs'])
    payload_path = by_role['A_payload']['path']
    payload = c.read(payload_path)
    exclusions_path = payload['new_seed_registry']['old_seed_exclusions']
    excluded = set(c.read(exclusions_path)['values'])
    for task in payload['new_seed_registry']['tasks']:
        excluded.update(h.hierarchy(task['seed'], task['cell']['q']))
    c.require(len(payload['new_seed_registry']['tasks']) == 128, 'ALL_ORIGINAL_A_128_EXCLUDED')
    targets = design['targets']
    inputs = [deepcopy(r) for r in origin['inputs']]
    for item in (row(accounting_spec, 'new_accounting_spec'), row(payload_path, 'original_A_payload'),
                 row(by_role['D_result']['path'], 'saved_D_result'),
                 row(exclusions_path, 'old_master_and_derived_seed_exclusions'),
                 row(h.ROOT / h.DESIGN, 'published_selection_design')):
        item['role'] = 'confirmation:' + item['role']
        inputs.append(item)
    seed_plan = dict(SOURCE=source, design_digest=digest, targets=targets,
        immutable_inputs=inputs, cohort=h.COHORT, excluded_digest=c.digest(sorted(excluded)),
        source_hashes={name: c.sha(h.ROOT / name) for name in h.FILES})
    seed_plan_digest = c.digest(seed_plan)
    c.save(destination / 'seed_plan.json', seed_plan)
    registry = dict(schema='H8_INDEPENDENT_SEED_REGISTRY_V1', SOURCE=source,
        seed_plan_digest=seed_plan_digest, targets=targets, excluded=sorted(excluded),
        tasks=h.derive_registry(targets, source, seed_plan_digest, excluded),
        old_cohorts_not_reused=True, redraw_on_collision=False, actual_RTE_event_sampling=0)
    registry_path = destination / 'seed_registry.json'
    c.save(registry_path, registry)
    inputs += [row(destination / 'seed_plan.json', 'confirmation:seed_plan'),
               row(registry_path, 'confirmation:seed_registry')]
    d_rows = c.read(by_role['D_result']['path'])['rows']
    signals = {r['cell']['id']: r for r in c.read(payload['A_signal'])['signals']}
    specs = []
    # Reuse only the laws whose exact hashes were frozen before new sampling.
    old_representatives = c.read(origin['output'] + '/first_control/parallel_receipt.json')
    law_rows = []
    for target in targets:
        handle = next(w for w in old_representatives['workers']
                      if w['task_id'] == target['cell']['id'] + '_0')
        path = Path(handle['directory']) / 'trajectory.json'
        law_rows.append(row(path, 'confirmation:law:' + target['scope']))
    inputs += law_rows
    for target in targets:
        scope = target['scope']
        directory = destination / scope
        directory.mkdir()
        record = signals[target['cell']['id']]
        c.require(c.digest(record) == target['A_signal_record_sha256'], 'SAVED_SIGNAL_IDENTITY')
        eps = target['primary_epsilon']
        shots = {}
        pf_rows = {}
        for mode, field, key in [('primary', 'precision_rows', 'primary_accounting_sha256'),
                ('secondary', 'empirical_noncancellation_precision_rows', 'secondary_accounting_sha256')]:
            accounting_row = record[field][str(eps)]['1.0']
            c.require(c.digest(accounting_row) == target[key], 'FROZEN_ACCOUNTING_ROW')
            shots[mode] = {a: accounting_row['axes'][ch]['shots'] for a, ch in c.core.CHANNEL.items()}
            pf_row = next(r for r in d_rows if r['cell_id'] == target['fixed_PF_id']
                and r['epsilon'] == eps and r['factor'] == 1.
                and r['bias_mode'] == ('USUAL_SIGNED' if mode == 'primary' else 'EMPIRICAL_NONCANCELLATION'))
            if pf_row['G_exact'] is not None:
                c.require(pf_row['eligible'] is True and all(v['measured'] is True
                    and int(v['RZ']) == target['PF_unit_RZ_per_axis']
                    and v['compiler_hash'] == design['binding']['compiler']
                    for v in pf_row['axis_costs'].values()), 'FIXED_PF_DIRECT_PRICE_ONLY')
            pf_rows[mode] = pf_row
        c.require(pf_rows['primary']['G_exact'] == target['fixed_PF_G_exact'], 'PRIMARY_PF_FIXED_COST')
        law_path = next(r['path'] for r in law_rows if r['role'].endswith(scope))
        law = c.read(law_path)['original_RTE_distribution']
        c.require(c.digest(law) == target['original_law_sha256'], 'FIXED_ORIGINAL_LAW')
        calls = dict(trajectory=h.N, occurrence=h.N * target['cell']['q'],
            RTE_event=h.N * target['cell']['R'], native_evolution=h.N,
            wrapper_compile=2 * h.N, control_probe=4, independent_event_oracle=1)
        spec = dict(schema='H8_PR_CONFIRMATION_TECHNICAL_SEAL_V1', scope=scope, SOURCE=source,
            design_digest=digest, origin_result_commit=design['origin_result_commit'],
            binding=design['binding'], target=target, n=h.N, cohort=h.COHORT,
            shots=shots, secondary_PF_G_exact=pf_rows['secondary']['G_exact'],
            secondary_PF_saved_status=pf_rows['secondary']['cost_status'],
            fixed_PF_receipts=pf_rows, law=law, payload=payload_path, registry=str(registry_path),
            source_hashes=seed_plan['source_hashes'], inputs=inputs,
            runtime_receipt=origin['runtime_receipt'], runtime_receipt_sha256=origin['runtime_receipt_sha256'],
            provider_closure=origin['provider_closure'], python=origin['python'],
            python_sha256=origin['python_sha256'], profile_reference=accounting['profile_reference'],
            output=str(directory / 'output'), lock=str(directory / 'ONE_SHOT.json'), calls=calls,
            output_estimate_bytes=512 * 2**20, inode_estimate=10000,
            resource_caps=None, approved=False, runtime_authorization=False, allowed_cpus=[],
            technical_sealed=True, retry=False, next_stage_authorized=False, absolute_launch_command=None)
        spec['technical_digest'] = h.technical_digest(spec)
        path = directory / 'technical_seal.json'
        c.save(path, spec)
        h.verify(spec)
        c.save(directory / 'grant_INVALID.json', dict(scope=scope, SOURCE=source,
            technical_digest=spec['technical_digest'], approved=False, approved_by_user=False,
            runtime_authorization=False, one_shot=True, retry=False, next_stage_authorized=False,
            allowed_cpus=[], absolute_launch_command=None))
        specs.append(dict(scope=scope, path=str(path), technical_digest=spec['technical_digest']))
    c.save(destination / 'preparation_receipt.json', dict(status='PASS_TECHNICAL_SEALS_ONLY',
        SOURCE=source, design_digest=digest, specs=specs, new_master_seeds=96,
        old_and_A_excluded_seed_count=len(excluded), new_scientific_calls=0, valid_grants_created=0))
    return specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accounting-spec', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--source', required=True)
    args = parser.parse_args()
    prepare(args.accounting_spec, args.destination, args.source)
    print('PASS_TECHNICAL_ONLY_NO_SCIENTIFIC_CALLS_NO_VALID_GRANTS')


if __name__ == '__main__':
    main()
