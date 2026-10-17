"""Saved JSON projections -> C records, invoked only after a new C grant.

No shot calculation, model fit, ranking or scientific import occurs here.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/trotterlib'))
import h8_phase_c_accounting as c


def normalize(raw, registry):
    c.b2_gate(raw['B2'])
    c.require(raw['Phase_A']['status'] == 'COMPLETE_MANDATORY_STOP', 'PHASE_A_NOT_COMPLETE')
    cells = {x['id']: x for x in registry['old'] + registry['additional']}
    confirmed = set(registry['confirmation_ids'])
    rows = []
    old = raw['D0']['H8_EXPLORATION92']
    full_pf = {}
    for row in old:
        if row['cell_id'] not in cells:
            continue
        rows.append(dict(cell_id=row['cell_id'], epsilon=row['epsilon'], factor=row['factor'],
                         accounting=row['accounting']))
        if cells[row['cell_id']]['method'] == 'B1' and row['status'] == 'DERIVED_SAVED_SCALAR':
            for axis, value in row['metrics']['RZ']['axis_means'].items():
                c.require(type(value) in (int, float) and int(value) == value and value > 0,
                          'LEGACY_FULL_PF_N1_INTEGER_MEASUREMENT')
                key = row['cell_id'], axis
                c.require(key not in full_pf or full_pf[key]['RZ'] == int(value), 'LEGACY_AXIS_CHANGED_ACROSS_PRECISIONS')
                full_pf[key] = dict(cell_id=key[0], axis=axis, RZ=int(value),
                    compiler_hash=c.COMPILER, provenance_verified=True,
                    kind='LEGACY_FULL_PF_DECLARED_LIMITS', origin='FROZEN_H8_D0_SAVED_MEASURED_N1',
                    profile_sha256=None, legacy_profile_policy='KEEP_ORIGINAL_OR_UNKNOWN_NEVER_BACKFILL_CURRENT',
                    current_compiler_output_bit_identity_verified=False)
    for row in raw['B0I_accounting']:
        c.require(row['cell_id'] in cells and cells[row['cell_id']]['method'] == 'B0I', 'B0I_SAVED_DOMAIN')
        rows.append(dict(cell_id=row['cell_id'], epsilon=row['epsilon'], factor=row['factor'],
                         accounting=row['accounting']))
    phase_a = c.unique(raw['Phase_A']['signals'], lambda x: x['cell']['id'])
    c.require(set(phase_a) == {x['id'] for x in registry['additional']}, 'EXACT_PHASE_A_152')
    for cid, signal in phase_a.items():
        cell = cells[cid]
        c.require(all(signal['cell'][k] == cell[k] for k in ('id', 'method', 'prefix', 'q', 'formula')),
                  'PHASE_A_CELL_CHANGED')
        for e in c.EPSILONS:
            for f in c.FACTORS:
                rows.append(dict(cell_id=cid, epsilon=e, factor=f,
                                 accounting=signal['precision_rows'][str(e)][str(f)]))
    for row in raw['D0']['H8_CONFIRMATION_UNION6']:
        if row['cell_id'] not in confirmed:
            continue
        c.require(row['method'] == 'B2' and row['sample_phase'] == 'confirmation' and
                  row['sample_count'] == 32, 'NO_EXPLORATION_POOLING')
        rows.append(dict(cell_id=row['cell_id'], epsilon=row['epsilon'], factor=row['factor'],
                         accounting=row['accounting'], sample_phase='confirmation', sample_count=32,
                         PR_metric=row['metrics'].get('RZ')))
    anchors = raw['B1_anchors']['rows']
    expected = {(m, p, q, a) for m in ('B0', 'B0I') for p in (14, 21)
                for q in ((1, 8) if m == 'B0' else (1, 2, 4, 8)) for a in c.AXES}
    c.require(len(anchors) == 24 and {(r['method'], r['prefix'], r['q'], r['axis']) for r in anchors} == expected,
              'EXACT_B1_24_MEASURED_ANCHORS')
    measured = list(full_pf.values())
    for row in anchors:
        c.require(row['runtime_profile_sha256'] == c.PROFILE and row['compiler_hash'] == c.COMPILER and
                  row['control'] == 'symmetric_directional' and row['formula'] == '2nd' and
                  row['origin_SOURCE'] == ('db559a518214fd27fa96e2f1451911df05c4f4a8' if row['method'] == 'B0'
                                           else '54ed98c5fa5b7190f77255271f4e3bec8db15630') and
                  row['evidence'] and all(x is True for x in row['evidence'].values()), 'B1_ANCHOR_LINEAGE')
        cid = f"{row['method']}_p{row['prefix']}_q{row['q']}_2nd"
        measured.append(dict(cell_id=cid, axis=row['axis'], RZ=row['saved_RZ'],
            compiler_hash=row['compiler_hash'], profile_sha256=row['runtime_profile_sha256'],
            provenance_verified=True, kind='B1_REGISTERED_ANCHOR', origin_SOURCE=row['origin_SOURCE'],
            original_B0_acquisition_status='STOP' if row['method'] == 'B0' else None))
    return dict(B2=raw['B2'], model=raw['model'], B1_result_commit=c.B1_RESULT,
                B1_terminal=raw['B1_terminal'], accounting=rows, measured_axes=measured)
