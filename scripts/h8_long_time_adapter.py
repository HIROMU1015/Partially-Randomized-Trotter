"""H8 long-time interfaces. Import/inspect is metadata-only; execution is opt-in.

The --execute switch prevents accidental dispatch; it is not user permission.
Scientific execution requires separate permission for the fixed SOURCE/scope.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

from h46_m5_contract import core

ROOT = Path(__file__).resolve().parents[1]
COMPILER = '1239304c14c65cde2e66ac783646abae19455ad8af6ce0cafd157557eeabb33b'


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    """Exclusive new evidence; never silently replace an earlier stage/trial."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def candidate(method, formula, prefix, q, R_outer=None, K=None, *, matched_q_diagnostic=False):
    row = dict(system='H8', method=method, formula=formula, prefix=prefix, q=q,
               T=1.6, R_outer=R_outer, K=K)
    if matched_q_diagnostic:
        row['diagnostic'] = 'MATCHED_Q_ONLY'
    validate_candidate(row)
    return dict(row, id=digest(row)[:20])


def validate_candidate(row):
    require(row['system'] == 'H8' and row['T'] == 1.6, 'H8_NEW_TIME_SCOPE')
    require(type(row['q']) is int and row['q'] in (1, 2, 4, 8, 16, 32), 'Q_SCOPE')
    pr = row['method'] == 'PR'
    diagnostic = row.get('diagnostic')
    require(diagnostic is None or (pr and diagnostic == 'MATCHED_Q_ONLY'
            and row['formula'] == '4th(m5_best)' and row['q'] in (8, 16, 32)
            and row['R_outer'] == 32 and row['K'] == 4), 'MATCHED_Q_DIAGNOSTIC_SCOPE')
    if pr:
        require(row['prefix'] in (7, 8) and row['K'] in (2, 4), 'PR_PREFIX_K')
        qs, rs = ((8, 16, 32), (4, 32)) if row['formula'] == '2nd' else ((1, 2, 4), (16, 32))
        require(row['formula'] in ('2nd', '4th(m5_best)') and (row['q'] in qs or diagnostic)
                and row['R_outer'] in rs, 'PR_PROPOSED_DOMAIN')
    else:
        require(row['method'] in ('FULL', 'B0', 'B0I') and row['formula'] in core.FORMULAS,
                'PF_METHOD_FORMULA')
        require(row['prefix'] in ((28,) if row['method'] == 'FULL' else (15, 16)), 'PF_PREFIX')
        require(row.get('K') is None and row.get('R_outer') is None, 'PF_NO_RTE')
    if 'id' in row:
        require(row['id'] == digest({k: v for k, v in row.items() if k != 'id'})[:20], 'CANDIDATE_ID')
    return row


def schedule(row):
    validate_candidate(row)
    return core.schedule(row['formula'], row['q'],
                         'ABS_TIME_PER_OUTER' if row['method'] == 'PR' else 'PER_PIECE',
                         row['R_outer'] if row['method'] == 'PR' else 1, T=row['T'])


def split_blocks(blocks, prefix, rank=28):
    require(len(blocks) == rank + 1 and 0 <= prefix <= rank, 'FULL_BLOCK_LAYOUT')
    require(tuple(b.original_fragment_index for b in blocks[1:]) == tuple(range(rank)),
            'SAVED_FRAGMENT_ORDER')
    return tuple(blocks[:prefix + 1]), tuple(blocks[prefix + 1:])


def effective_scalar(method, original, tail):
    require(method in ('PR', 'FULL', 'B0', 'B0I') and math.isfinite(original + tail), 'SCALAR')
    return original + (tail if method in ('PR', 'B0I') else 0.)


def check_reference(ref, binding, T=1.6):
    require(ref['T'] == T and ref['kind'] == 'NEW_FULL_H_SPARSE_REFERENCE', 'NEW_TIME_REFERENCE_REQUIRED')
    require(ref['binding'] == binding, 'REFERENCE_INPUT_IDENTITY')
    require(ref['certified'] is False and ref['phase_alignment'] is False, 'REFERENCE_POLICY')
    z = complex(ref['signal']['real'], ref['signal']['imag'])
    require(math.isfinite(z.real) and math.isfinite(z.imag), 'REFERENCE_FINITE')
    return z


def check_files(files):
    for row in files:
        p = Path(row['path'])
        require(p.is_file() and p.stat().st_size == row['bytes'] and sha(p) == row['sha256'],
                'INPUT_SOURCE_BYTES:' + row['role'])


def preflight(config):
    """No arrays decoded; no scientific dependency imported."""
    require(config['schema'] == 'H8_LONG_TIME_PRIVATE_V1', 'CONFIG_SCHEMA')
    require(config['binding']['compiler'] == COMPILER, 'COMPILER_BINDING')
    check_files(config['files'])
    for name, expected in config['source_files'].items():
        require(sha(ROOT / name) == expected, 'ADAPTER_REUSE_SOURCE:' + name)
    return dict(status='PASS_METADATA_ONLY', input_files=len(config['files']),
                source_files=len(config['source_files']), scientific_calls=0)


def finite_piece(vector, apply_tail, dt, K, normalization, *, raw, horner=False):
    """P_(K+1)(-i dt H_tail); K denotes the even event cutoff.

    Independent forward/Horner recurrences share only the supplied action.
    Works on artificial scalars/arrays and on the future matrix-free H8 vector.
    """
    require(K in (2, 4) and normalization >= 1 and math.isfinite(normalization), 'FINITE_LAW')
    if horner:
        out = vector / math.factorial(K + 1)
        for degree in range(K, -1, -1):
            out = vector / math.factorial(degree) + (-1j * dt) * apply_tail(out)
    else:
        power = vector.copy() if hasattr(vector, 'copy') else vector
        out = power
        for degree in range(1, K + 2):
            power = (-1j * dt / degree) * apply_tail(power)
            out = out + power
    return out / normalization if raw else out


def paired_G(rows, endpoint, cohort):
    """Exact fixed-shot weighted paired samples, with descriptive sample SE."""
    require(endpoint['status'] == 'ELIGIBLE' and rows, 'ELIGIBLE_MEASURED_PRICE_REQUIRED')
    nc, ns = endpoint['N_cos'], endpoint['N_sin']
    require(all(type(n) is int and n >= 0 for n in (nc, ns)), 'FIXED_INTEGER_SHOTS')
    seen = set()
    weighted = []
    for row in rows:
        require(row['status'] == 'PASS' and row['cohort'] == cohort and not row['conditional'], 'COHORT_OR_CONDITIONAL_MIX')
        require(row['candidate_id'] == endpoint['candidate']['id'], 'PRICE_CANDIDATE')
        require(row['index'] not in seen, 'DUPLICATE_TRAJECTORY')
        require(row['endpoint_digest'] == digest(endpoint), 'FIXED_N_ENDPOINT')
        seen.add(row['index'])
        a, b = row['prices']['cosine']['rz_count'], row['prices']['sine']['rz_count']
        require(all(type(x) is int and x > 0 for x in (a, b)), 'MEASURED_POSITIVE_RZ')
        weighted.append(nc * a + ns * b)
    mean = Fraction(sum(weighted), len(weighted))
    variance = None if len(rows) < 2 else sum((Fraction(x) - mean)**2 for x in weighted) / (len(rows) - 1)
    return dict(n=len(rows), cohort=cohort, fixed_N_cos=nc, fixed_N_sin=ns,
                mean_G_exact=str(mean), sample_variance_G_exact=None if variance is None else str(variance),
                sample_SE_G=None if variance is None else math.sqrt(float(variance / len(rows))),
                population_tail_bound=None, population_guarantee=False, pooled=False)


def confirmation_n(exploratory_variance_G, absolute_SE_target):
    """Planning estimate only, never a stopping rule or automatic top-up."""
    require(exploratory_variance_G >= 0 and math.isfinite(exploratory_variance_G)
            and absolute_SE_target > 0 and math.isfinite(absolute_SE_target), 'PRECISION_DESIGN')
    return max(2, math.ceil(exploratory_variance_G / absolute_SE_target**2))


def confirmation_comparison(summaries, ids, expected_direction, absolute_SE_target):
    """Descriptive confirmation only; no population/tail guarantee is inferred."""
    require(expected_direction in ('PR_LOWER', 'PF_LOWER') and absolute_SE_target > 0
            and math.isfinite(absolute_SE_target), 'FIXED_CONFIRMATION_DESIGN')
    pr, pf = summaries[ids['PR']], summaries[ids['PF']]
    require(pr['cohort'] == pf['cohort'] == 'CONFIRMATION', 'CONFIRMATION_ONLY')
    delta = Fraction(pr['mean_G_exact']) - Fraction(pf['mean_G_exact'])
    direction = 'PR_LOWER' if delta < 0 else 'PF_LOWER' if delta > 0 else 'TIE'
    se = pr['sample_SE_G']
    return dict(PR_minus_PF_exact=str(delta), observed_point_direction=direction,
                matches_fixed_direction=direction == expected_direction,
                absolute_SE_G_target=absolute_SE_target, observed_sample_SE_G=se,
                descriptive_precision_met=se is not None and se <= absolute_SE_target,
                population_guarantee=False, automatic_top_up=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--stage', choices=('inspect', 'A', 'B', 'C'), default='inspect')
    p.add_argument('--scope', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    config = read(args.config)
    metadata = preflight(config)
    if args.stage == 'inspect':
        print(json.dumps(metadata))
        return
    require(args.execute and args.scope is not None and args.output is not None,
            'EXPLICIT_STAGE_SCOPE_OUTPUT_REQUIRED; --execute is not user permission')
    from h8_long_time_runtime import run_stage
    run_stage(args.stage, config, read(args.scope), args.output)


if __name__ == '__main__':
    main()
