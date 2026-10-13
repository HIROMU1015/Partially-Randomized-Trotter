#!/usr/bin/env python3
"""Metadata-only binding for the user-authorized H8 eight-case preflight."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from trottertracks.resource_applicability.h8_signal_preflight_contract_v1 import (
    AUTHORIZATION, input_identity, plan, profile, resources, observation,
    registrations, source_paths, verify_binding)
from trottertracks.resource_applicability.ax2a_preparation import digest
from trottertracks.resource_applicability.ax2b_h4_contract_v5 import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope', choices=('SIGNAL', 'HEAVY', 'REMAINING'), required=True)
    p.add_argument('--source', required=True); p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--signals', type=Path); p.add_argument('--heavy', type=Path)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if len(a.source) != 40:
        raise ValueError('H8_SOURCE_SHA')
    subprocess.run(['git', 'merge-base', '--is-ancestor', a.source, 'HEAD'], cwd=ROOT, check=True)
    out = a.output.resolve()
    if out.exists() or (out.parent/(out.name+'.one_shot_lock')).exists():
        raise ValueError('H8_EXCLUSIVE_OUTPUT')
    s = None if a.signals is None else json.loads(a.signals.read_text())
    m = dict(schema='h8_eight_case_bound_scope_v1', scope=a.scope, kind='H8_TECHNICAL_PREFLIGHT',
        source_commit=a.source, source_hashes={n: file_hash(ROOT/n) for n in source_paths(ROOT)},
        plan=plan(), profile=profile(), input_identity=input_identity(a.input), output=str(out),
        registrations=registrations(a.source, a.scope, s),
        proposed_resources=resources(a.scope, a.workers), observation_policy=observation(a.scope, a.workers),
        base_manifest={'plan': {'caps_proposed': {'output_bytes': 2*2**30, 'log_bytes': 2*2**20}}},
        H8_full_campaign_authorized=False, full_comparison_contract_adopted=False,
        cumulative_carry=dict(H6_known_compile=1716, H6_reserved_compile=20,
            H6_known_trajectory=842, H6_reserved_trajectory=10,
            H4_supplement_compile=532, H4_supplement_trajectory=256,
            H8_integral_build=1, H8_df_decomposition=1, H8_successful_solver_matvec=61,
            H8_failed_solver_matvec_actual=None, H8_failed_solver_reservation=10000,
            new_preflight_compile_upper=20, new_preflight_trajectory_upper=4,
            new_preflight_occurrence_upper=15, refund=0, reset=False,
            production_H4_lineage='unchanged separate lineage', unknown_history_actual=None))
    if a.scope != 'SIGNAL':
        if a.signals is None:
            raise ValueError('H8_SIGNAL_REQUIRED')
        m['signal_binding'] = dict(path=str(a.signals.resolve()), sha256=file_hash(a.signals))
    if a.scope == 'REMAINING':
        if a.heavy is None:
            raise ValueError('H8_HEAVY_REQUIRED')
        m['heavy_binding'] = dict(path=str(a.heavy.resolve()), sha256=file_hash(a.heavy))
    g = dict(schema='h8_eight_case_user_authorization_v1', approved_by_user=True,
        authorization_basis=AUTHORIZATION, manifest_sha256=digest(m), source_commit=a.source,
        output=str(out), one_shot=True, retry=False, automatic_next_full_campaign=False,
        H8_full_campaign_authorized=False, mandatory_stop=True)
    verify_binding(ROOT, m, g)
    a.bundle.mkdir(parents=True, exist_ok=False)
    for name, data in (('manifest.json', m), ('grant.json', g)):
        with (a.bundle/name).open('x') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    (a.bundle/'SHA256SUMS').write_text('\n'.join(file_hash(a.bundle/n)+'  '+n for n in ('manifest.json', 'grant.json'))+'\n')
    print(json.dumps(dict(source=a.source, scope=a.scope, registrations=len(m['registrations']),
                          bundle=str(a.bundle), output=str(out), science_executed=False)))


if __name__ == '__main__':
    main()
