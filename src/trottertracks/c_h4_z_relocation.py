"""Fixed-J mass diagnostic only; imports/preflight never decode saved arrays.

No Hamiltonian, optimizer, circuit, PF/RTE, shots or price implementation.
User permission is external to this module; no approval-proof artifacts.
"""
import csv
import itertools
from pathlib import Path

from . import c1_stage1_execution as io

ATOL = 1e-12
RTOL = 1e-12
SYSTEM = dict(molecule='linear H4', distance_angstrom=1., basis='STO-3G',
              rank=8, norb=4, ld=2, nalpha=2, nbeta=2, frame_order='original')
VERSION = 'H4_fixed_J_linear_Z_relocation_mass_only_v1'


def close(a, b):
    return abs(a-b) <= ATOL+RTOL*max(abs(a), abs(b))


def diagnose(J, *, ld):
    import numpy as np
    from . import c_near_objective as near
    a = np.asarray(J)
    if (a.dtype != np.float64 or a.ndim != 3 or a.shape[1] != a.shape[2]
            or type(ld) is not int or not 0 <= ld < len(a)
            or not np.isfinite(a).all() or not np.array_equal(a, a.transpose(0, 2, 1))):
        raise ValueError('FINITE_BINARY64_SIGNED_SYMMETRIC_J_AND_ORIGINAL_LD')
    spec = near.Objective(len(a), a.shape[1], ld, 1., 1., 0.)
    old = near.mass_diagnostics(a, spec)
    mapped = near.pr.coefficients(a[ld:], np)
    O = float(np.sum(np.triu(abs(a[ld:]), 1)))
    D = float(np.sum(abs(np.diagonal(a[ld:], axis1=1, axis2=2))))
    Z = float(np.sum(abs(mapped[:, :2*a.shape[1]])))
    ZZ = float(np.sum(abs(mapped[:, 2*a.shape[1]:])))
    M, E, gap = (old[k] for k in ('exact_tail_mass', 'exact_entry_mass', 'row_cancellation_gap'))
    residuals = dict(M_minus_MZZ_minus_Z=M-ZZ-Z, MZZ_minus_O_quarterD=ZZ-O-D/4,
        E_minus_3O_5quarterD=E-3*O-5*D/4, gap_minus_E_minus_M=gap-(E-M))
    checks = dict(mapped_mass=close(M, Z+ZZ), ZZ_formula=close(ZZ, O+D/4),
        entry_formula=close(E, 3*O+5*D/4), cancellation=close(gap, E-M),
        gap_nonnegative=gap >= -ATOL-RTOL*max(E, M),
        bound_lower=ZZ+ATOL+RTOL*max(E/5, ZZ) >= E/5,
        bound_upper=ZZ <= E/3+ATOL+RTOL*max(E/3, ZZ))
    if not all(checks.values()):
        raise ValueError('RELOCATION_IDENTITY_OR_BOUND_FAILURE')
    return dict(O=O, D_diag=D, Z=Z, M=M, E=E, E_minus_M=E-M, M_ZZ=ZZ,
        Z_over_M=Z/M if M > 0 else None, zero_mass=M == 0.,
        identity_residuals=residuals, checks=checks,
        total_RZ=None, PF_error=None, shots=None, W=None)


def compare_pair(pr, entry, *, pairing):
    if (pr['arm'] != 'PR' or entry['arm'] != 'ENTRY'
            or any(pr[k] != entry[k] for k in ('start','budget'))):
        raise ValueError('PAIR_CONDITIONS_OR_ARMS')
    fair = pairing == 'frozen_fair_policy_different_strength_explicit'
    if (fair and (pr['start'],pr['budget'],pr['strength'],entry['strength']) != ('C0',64,.1,.03)
            or not fair and pr['strength'] != entry['strength']):
        raise ValueError('PAIR_STRENGTH_BINDING')
    a, b = pr['M_ZZ'], entry['M_ZZ']
    tol = ATOL+RTOL*max(abs(a), abs(b))
    label = 'NUMERICALLY_SAME' if abs(a-b) <= tol else 'PR_LOWER' if a < b else 'ENTRY_LOWER'
    return dict(pairing=pairing, PR_id=pr['candidate_id'], ENTRY_id=entry['candidate_id'],
        difference=a-b, numerical_allowance=tol, classification=label,
        ratio=a/b if b > tol else None,
        ratio_status='DEFINED' if b > tol else 'UNRESOLVED_SMALL_OR_ZERO_DENOMINATOR',
        total_RZ_ranking=None, novelty=None)


def checkpoint_key(row):
    return row['start'], row['strength'], row['budget'], row['arm']


def validate_inventory(inventory):
    if inventory['system'] != SYSTEM:
        raise ValueError('FIXED_H4_SYSTEM')
    entries = inventory['checkpoints']
    wanted = set(itertools.product(('C0','P1','P2'), (.03,.1,.3), (16,64), ('PR','ENTRY')))
    keys = [checkpoint_key(x) for x in entries]
    if len(keys) != 36 or set(keys) != wanted or len(set(keys)) != 36:
        raise ValueError('EXACTLY_36_FIXED_CHECKPOINTS_NO_SELECTION')
    if len({x['candidate_id'] for x in entries}) != 36:
        raise ValueError('UNIQUE_CHECKPOINT_CANDIDATE_IDS')
    for x in entries+inventory['references']:
        if x['J_identity']['shape'] != [8,4,4] or x['J_identity']['dtype'] != '<f8':
            raise ValueError('SAVED_J_METADATA_SCHEMA')
    for x in entries:
        expected_series = 'picklefix' if (x['start'],x['arm'],x['strength']) == ('C0','ENTRY',.3) else 'original'
        if x['series'] != expected_series or set(x['saved_masses']) != {'M','E','E_minus_M'}:
            raise ValueError('EXACT_ORIGINAL17_PICKLEFIX1_BINDING')
    if len(inventory['references']) > 1:
        raise ValueError('AT_MOST_ONE_NEUTRAL_C0_REFERENCE')


def preflight(root, contract_path, *, source_path=None, require_fresh=True):
    """Bytes and JSON only: hash NPZ containers without opening/decoding J/U."""
    root = Path(root).resolve()
    c = io.read(root/contract_path)
    if (c['scope'] != VERSION or c['system'] != SYSTEM or c['retry'] != 0
            or c['resume'] != 0 or c['mandatory_stop'] is not True
            or c['numerics'] != dict(atol_Ha=ATOL, rtol=RTOL)
            or c['maximum'] != dict(checkpoints=36, neutral_references=1, J_reads=37)
            or c['comparisons'] != dict(matched_pairs=18, frozen_common='C0/.1/64',
                frozen_fair='C0/PR.1/ENTRY.03/64')):
        raise ValueError('FIXED_DIAGNOSTIC_CONTRACT')
    inventory = io.read(root/c['inventory'])
    if io.sha(root/c['inventory']) != c['inventory_sha256']:
        raise ValueError('INVENTORY_IDENTITY')
    validate_inventory(inventory)
    source_file = root/(source_path or c['SOURCE'])
    if io.sha(source_file) != c['SOURCE_sha256']:
        raise ValueError('DIAGNOSTIC_SOURCE_MANIFEST')
    source = io.read(source_file)
    for name, expected in source['files'].items():
        if io.sha(root/name) != expected:
            raise ValueError('DIAGNOSTIC_SOURCE_HASH:'+name)
    # Use inherited manifests as byte provenance, not their old approval/grant
    # machinery: this is an independent saved-J diagnostic, not an old resume.
    for item in inventory['protected']:
        p = Path(item['path'])
        if io.sha(p) != item['sha256']:
            raise ValueError('PROTECTED_FILE_IDENTITY:'+str(p))
    for name, expected in inventory['inherited_scientific_SOURCE_files'].items():
        # Historic closure is provenance. Only reused SOURCE files are a live
        # dependency; unrelated historical docs/worktree changes are observed,
        # never treated as a reason to stop this independent mass diagnostic.
        if name in source['files'] and io.sha(Path(inventory['scientific_archive_root'])/name) != expected:
            raise ValueError('INHERITED_SOURCE_HASH:'+name)
    for row in inventory['checkpoints']+inventory['references']:
        if io.sha(row['path']) != row['file_sha256']:
            raise ValueError('SAVED_CHECKPOINT_FILE_HASH:'+row['candidate_id'])
        report_path = Path(inventory['scientific_archive_root'])/row['report_path']
        if io.sha(report_path) != row['report_sha256']:
            raise ValueError('SAVED_CHECKPOINT_REPORT_HASH')
        report = io.read(report_path)
        if row['arm'] != 'REFERENCE':
            point = report['checkpoints'][str(row['budget'])]
            if (any(report[k] != row[k] for k in ('start','arm','strength'))
                    or report['source_sha256'] != row['original_SOURCE_sha256']
                    or point['array_file_sha256'] != row['file_sha256']
                    or point['array_identity']['J'] != row['J_identity']):
                raise ValueError('SAVED_REPORT_CANDIDATE_BINDING')
        else:
            neutral = report['starts']['C0']['identity']
            if (row['start'] != 'C0' or row['budget'] != 0 or row['strength'] is not None
                    or neutral['cold_sha256'] != row['file_sha256']
                    or neutral['array_identity']['J'] != row['J_identity']):
                raise ValueError('NEUTRAL_C0_REFERENCE_BINDING')
    if require_fresh and (root/c['fresh_output']).exists():
        raise FileExistsError('FRESH_OUTPUT_REQUIRED_RETRY_RESUME_ZERO')
    return dict(status='PASSIVE_PREFLIGHT_PASS', scope=VERSION, checkpoints=36,
        references=len(inventory['references']), J_arrays_decoded=0,
        new_scientific_analysis=False, contract=c, inventory=inventory,
        SOURCE_sha256=io.sha(source_file), contract_sha256=io.sha(root/contract_path))


def _write_tsv(path, rows):
    keys = [k for k,v in rows[0].items() if not isinstance(v, (dict,list))]
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, keys, delimiter='\t', extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def execute(root, contract_path):
    """Call only after separate user execution permission; never called in prep."""
    import numpy as np
    from .c_pr_aware_df_stage_b_runner import load
    root = Path(root).resolve()
    verified = preflight(root, contract_path)
    c, inventory = verified['contract'], verified['inventory']
    out = root/c['fresh_output'];out.mkdir(parents=True, exist_ok=False)
    rows, refs = [], []
    failure = None
    try:
        for group, target in (('checkpoints', rows), ('references', refs)):
            for item in inventory[group]:
                # Existing whitelist loader accesses only J, never U/h/g.
                J = load(item['path'], ('J',))['J']
                if io.array_record(J) != item['J_identity']:
                    raise ValueError('SAVED_J_ARRAY_IDENTITY:'+item['candidate_id'])
                row = dict(candidate_id=item['candidate_id'], start=item['start'],
                    arm=item['arm'], strength=item['strength'], budget=item['budget'],
                    original_SOURCE_sha256=item['original_SOURCE_sha256'],
                    candidate_file_sha256=item['file_sha256'],
                    J_sha256=item['J_identity']['sha256'], **diagnose(J, ld=2))
                for key, value in item['saved_masses'].items():
                    if not close(row[key], value):
                        raise ValueError('SAVED_MASS_MISMATCH:'+item['candidate_id']+':'+key)
                row['saved_mass_residuals'] = {k:row[k]-v for k,v in item['saved_masses'].items()}
                target.append(row)
        by_key = {checkpoint_key(r):r for r in rows}
        pairs = [compare_pair(by_key[start,s,b,'PR'], by_key[start,s,b,'ENTRY'],
            pairing=f'matched/{start}/{s}/{b}') for start,s,b in
            itertools.product(('C0','P1','P2'),(.03,.1,.3),(16,64))]
        primary = dict(common=compare_pair(by_key['C0',.1,64,'PR'],by_key['C0',.1,64,'ENTRY'],
            pairing='frozen_common_policy_same_strength'),
            fair=compare_pair(by_key['C0',.1,64,'PR'],by_key['C0',.03,64,'ENTRY'],
                pairing='frozen_fair_policy_different_strength_explicit'))
        preflight(root, contract_path, require_fresh=False)
        _write_tsv(out/'checkpoints.tsv', rows)
        _write_tsv(out/'matched_pairs.tsv', pairs)
        io.save(out/'result.json', dict(status='MASS_DIAGNOSTIC_ONLY', rows=rows,
            references=refs, matched_pairs=pairs, frozen_primary=primary,
            SOURCE_sha256=verified['SOURCE_sha256'],contract_sha256=verified['contract_sha256'],
            PF_RTE=None, shots=None, W=None, total_RZ=None, novelty=None))
    except BaseException as exc:
        failure = repr(exc)
        io.save(out/'failure.json', dict(error=failure, classification='UNRESOLVED_INPUT_OR_NUMERICS',
            completed_rows=rows, references=refs))
        raise
    finally:
        io.save(out/'STOP.json', dict(failed=failure is not None, reason=failure or 'REGISTERED_DIAGNOSTIC_COMPLETE',
            checkpoint_rows=len(rows), reference_rows=len(refs), retry=0, resume=0,
            mandatory_stop=True, downstream_started=False))
    return out
