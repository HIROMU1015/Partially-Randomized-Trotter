"""Default: byte/literal checks only. --run is future, separately permitted science.

No approval proof system. --run is an explicit execution switch, not permission.
This SOURCE is intentionally unable to certify a physical/implemented experiment.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction as Q
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE_FILES = ('adapter.py', 'run_limited_validation.py', 'fixed_inputs.json',
                'PROOF.md', 'LIMITED_EXECUTION_PLAN.md')
STATUS = dict(RPE_tolerance_certified=False, scientific_status='UNDETERMINED',
              U_virtual_IID_upper=None, U_exec_upper=None, mandatory_STOP=True)


def literal_json(raw):
    return json.loads(raw, parse_float=str, parse_int=str)


def metadata():
    """No real numeric adapter call, sum, radius, angle, or energy comparison."""
    manifest = json.loads((HERE/'fixed_inputs.json').read_text())
    raw = {}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
    for key, record in manifest['files'].items():
        content = (ROOT/record['path']).read_bytes()
        frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                         record['commit']+':'+record['path']], env=env)
        if (content != frozen or hashlib.sha256(content).hexdigest() != record['sha256']
                or len(content) != record['bytes']):
            raise ValueError('UNKNOWN/STOP: fixed input identity: '+key)
        raw[key] = content
    prepared = json.loads(raw['FIXED_ROUNDS'])
    result = json.loads(raw['RESULT'])
    m17 = json.loads(raw['M17_RESULT'])
    rows = prepared['rounds']
    axis_rows = list(csv.DictReader(io.StringIO(raw['AXES'].decode()), delimiter='\t'))
    if (len(rows) != 18 or [x['m'] for x in rows] != list(range(18))
            or len(axis_rows) != 36 or result['SOURCE']['commit'] != manifest['enclosure_SOURCE']
            or result['categorical_local_conditions'] != 'CONDITIONAL_IDEAL_CATEGORICAL_LOCAL_CONDITIONS_CONFIRMED'
            or result['execution_status'] != 'COMPLETED_MANDATORY_STOP'
            or any(result[k] != v for k, v in STATUS.items())):
        raise ValueError('UNKNOWN/STOP: input scope or prior status differs')
    originals = {x['m']: x for g in result['groups'] for x in g['rounds']}
    originals[17] = {'axes': m17['enclosures'][0]['axes']}
    if set(originals) != set(range(18)):
        raise ValueError('UNKNOWN/STOP: prior result inventory differs')
    for row in rows:
        m = row['m']
        if row['ld'] != 3 or row['q'] != 1 << m or row['K'] != 2 or row['delta'] != '0.02':
            raise ValueError('UNKNOWN/STOP: unsupported scope')
        for axis, coordinate in (('cosine', 'real'), ('sine', 'imag')):
            matches = [x for x in axis_rows if x['m'] == str(m) and x['axis'] == axis]
            if len(matches) != 1:
                raise ValueError('UNKNOWN/STOP: axis inventory differs')
            saved, original = matches[0], originals[m]['axes'][axis]
            if (any(saved[k] != row['axes'][axis][k] for k in ('N', 'alpha', 'epsilon'))
                    or saved['saved_mu0'] != row['mu0'][coordinate]
                    or any(saved[k] != str(row[k]) for k in ('q', 'r', 'K'))
                    or original['categorical_decision'] != 'SUFFICIENT_CONDITION_CONFIRMED'
                    or saved['categorical_decision'] != original['categorical_decision']
                    or saved['interval_lower'] != original['interval']['lower']
                    or saved['interval_upper'] != original['interval']['upper']
                    or saved['U_cat_upper'] != original['U_cat_upper']):
                raise ValueError('UNKNOWN/STOP: fixed axis literal mismatch')
    grid, wp01 = literal_json(raw['GRID']), literal_json(raw['WP01'])
    constants = manifest['constants']
    if (grid['hamiltonian']['ground_energy'] != constants['energy_reference_binary64_decimal']
            or grid['request']['delta_time'] != constants['delta_binary64_decimal']
            or any(wp01['configuration'][key] != constants[key] for key in
                   ('alpha_total', 'beta_rpe', 'target_energy_precision_ha'))):
        raise ValueError('UNKNOWN/STOP: reference/budget literal mismatch')
    return manifest, rows


def describe():
    manifest, rows = metadata()
    return dict(STATUS, preparation_status='PREPARED_NOT_SCIENTIFICALLY_EXECUTED',
                fixed_rounds=len(rows), reused_local_axes=36, new_H4_numeric_calls=0,
                precision_bits=manifest['precision_bits'],
                implementation_decoder='UNKNOWN: uniform Float64 envelopes unavailable',
                physical_energy='UNKNOWN: rigorous reference/state bridge unavailable')


def evaluate(rows, constants):
    """Pure arithmetic entry, also usable with artificial fixtures only."""
    import adapter as a
    # Match the old local theorem: binary64 center; exact saved decimal eps/alpha.
    energy = Q.from_float(float(constants['energy_reference_binary64_decimal']))
    delta = Q.from_float(float(constants['delta_binary64_decimal']))
    phi = -energy*delta  # no intermediate binary64 multiplication
    geometries, bounds = [], []
    for row in rows:
        g = a.round_geometry(float(row['mu0']['real']), float(row['mu0']['imag']),
                             row['axes']['cosine']['epsilon'], row['axes']['sine']['epsilon'], phi, row['q'])
        geometries.append(dict(m=row['m'], q=row['q'], **g))
        bounds.append(None if g['phase_upper'] is None else a.phase_upper(g))
    alphas = [row['axes'][axis]['alpha'] for row in rows for axis in ('cosine', 'sine')]
    probability = a.union_bound(alphas, constants['alpha_total'])
    binary64_probability = a.union_bound([float(x) for x in alphas], float(constants['alpha_total']))
    branch, energy_result = None, None
    phase_budgets = [b is not None and b < a.real(float(constants['beta_rpe'])) for b in bounds]
    if all(b is not None for b in bounds):
        branch = a.branch_conditions(bounds, [row['q'] for row in rows])
        if branch['ideal_decision'] == 'SUFFICIENT':
            energy_result = a.energy_conditions(phi, branch['ideal_final_phase_upper']['upper'], delta,
                float(constants['target_energy_precision_ha']), reference_energy_error=None)
    mathematical = (all(phase_budgets) and branch is not None and branch['ideal_decision'] == 'SUFFICIENT'
        and probability['saved_budget_decision'] == 'SUFFICIENT'
        and energy_result is not None and energy_result['literal_target_decision'] == 'SUFFICIENT')
    return dict(STATUS, execution_status='COMPLETED_MANDATORY_STOP',
        mathematical_literal_reference_decision='CONDITIONAL_IDEAL_DECODER_SUFFICIENT' if mathematical else 'UNKNOWN/STOP',
        implementation_decoder_decision='UNKNOWN/STOP', physical_ground_energy_decision='UNKNOWN/STOP',
        reference_phase_exact=str(phi), delta_exact=str(delta), rounds=geometries,
        saved_phase_budget_strict=phase_budgets, branch=branch, energy=energy_result,
        probability_saved_decimal=probability, probability_binary64_diagnostic=binary64_probability,
        reference_energy_error=None, uniform_Float64_envelopes=None,
        prohibited_calls=dict(mean_enclosure=0, H_DF_state=0, sampling_RNG=0, build_compile=0, binomial=0, cost=0))


def frozen_source():
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
    sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], env=env).decode().strip()
    hashes = {}
    for name in SOURCE_FILES:
        path = HERE/name
        raw = path.read_bytes()
        committed = subprocess.check_output(['git', '-C', str(ROOT), 'show', sha+':'+path.relative_to(ROOT).as_posix()], env=env)
        if raw != committed:
            raise ValueError('UNKNOWN/STOP: publish fixed SOURCE before science')
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return dict(commit=sha, files_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Separately permitted fixed scientific reanalysis ONLY')
    parser.add_argument('--output', type=Path, help='New result directory; never overwritten')
    args = parser.parse_args()
    if not args.run:
        if args.output is not None:
            parser.error('--output requires --run')
        print(json.dumps(describe(), indent=2))
        return
    if args.output is None or args.output.exists():
        parser.error('--run requires a new --output directory')
    source = frozen_source()
    manifest, rows = metadata()
    import flint
    import numpy as np
    if (flint.__version__ != '0.9.0' or flint.__FLINT_VERSION__ != '3.6.0'
            or np.__version__ != '1.26.4' or sys.version_info[:2] != (3, 11)
            or sys.float_info.radix != 2 or sys.float_info.mant_dig != 53):
        raise ValueError('UNKNOWN/STOP: fixed numerical runtime differs')
    # Tiny scalar geometry workload: no matrix construction, q-sized arrays or
    # oversubscribing process pool. No CPU/RAM/worker or wall-time cap is imposed.
    resources = {'available_CPUs': len(os.sched_getaffinity(0)),
                 'backend': 'CPU Arb scalar enclosures; no dense matrix workload',
                 'MemAvailable': next((s.strip() for s in Path('/proc/meminfo').read_text().splitlines()
                                       if s.startswith('MemAvailable:')), None)}
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        with flint.ctx.workprec(manifest['precision_bits']):
            result = evaluate(rows, manifest['constants'])
        result.update(SOURCE=source, inputs=manifest['files'], resources=resources,
                      elapsed_seconds=time.monotonic()-start, precision_bits=manifest['precision_bits'])
        (args.output/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    except Exception as exc:
        (args.output/'STOP.json').write_text(json.dumps(dict(STATUS, execution_status='UNKNOWN/STOP',
            reason=str(exc), SOURCE=source), indent=2)+'\n')
        raise
    print(json.dumps(dict(STATUS, execution_status=result['execution_status'],
        mathematical_literal_reference_decision=result['mathematical_literal_reference_decision']), indent=2))


if __name__ == '__main__':
    main()
