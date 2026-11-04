"""Default: fixed bytes/headers/literals only. A/C science needs separate permission.

--run is an execution switch, not an approval certificate. B has no H4 execution
entry because a tractable uniform distance/primitive proof is still unavailable.
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
STATUS = dict(RPE_tolerance_certified=False, scientific_status='UNDETERMINED',
              U_virtual_IID_upper=None, U_exec_upper=None, mandatory_STOP=True)
SOURCE_FILES = ('common.py', 'alpha_adapter.py', 'reference_adapter.py', 'decoder_finite.py',
                'run_three_gaps.py', 'fixed_inputs.json', 'PROOF_AND_SCOPE.md', 'EXECUTION_PLAN.md')


def metadata():
    """No matrix/state decode, exponentials, sums, eigenvalue or decoder calls."""
    manifest = json.loads((HERE/'fixed_inputs.json').read_text())
    raw = {}
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
    for key, item in manifest['files'].items():
        value = (ROOT/item['path']).read_bytes()
        frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', item['commit']+':'+item['path']], env=env)
        if value != frozen or hashlib.sha256(value).hexdigest() != item['sha256']:
            raise ValueError('UNKNOWN/STOP: fixed input/SOURCE differs: '+key)
        raw[key] = value
    axes = list(csv.DictReader(io.StringIO(raw['AXES'].decode()), delimiter='\t'))
    rows = json.loads(raw['ROUNDS'])['rounds']
    if len(axes) != 36 or len(rows) != 18:
        raise ValueError('UNKNOWN/STOP: incomplete saved inventory')
    for row in rows:
        for axis, coordinate in (('cosine', 'real'), ('sine', 'imag')):
            found = [a for a in axes if a['m'] == str(row['m']) and a['axis'] == axis]
            if len(found) != 1:
                raise ValueError('UNKNOWN/STOP: missing/duplicate axis')
            a = found[0]
            if (a['categorical_decision'] != 'SUFFICIENT_CONDITION_CONFIRMED'
                    or any(a[k] != row['axes'][axis][k] for k in ('N', 'epsilon', 'alpha'))
                    or a['saved_mu0'] != row['mu0'][coordinate]
                    or any(a[k] != str(row[k]) for k in ('q', 'r', 'K'))):
                raise ValueError('UNKNOWN/STOP: axis literal differs')
    prior = json.loads(raw['D_RESULT'])
    if (prior['SOURCE']['commit'] != manifest['D_scientific_SOURCE']
            or prior['mathematical_literal_reference_decision'] != 'UNKNOWN/STOP'
            or prior['branch']['ideal_decision'] != 'SUFFICIENT'
            or prior['energy']['literal_target_decision'] != 'SUFFICIENT'
            or any(prior[k] != value for k, value in STATUS.items())):
        raise ValueError('UNKNOWN/STOP: prior D result/scope differs')
    return manifest, raw, axes, prior


def describe():
    manifest, _, axes, _ = metadata()
    return dict(STATUS, status='TECHNICAL_PREPARATION_ONLY', fixed_axes=len(axes),
                new_H4_scientific_calls=0, A='SOURCE_PREPARED_NOT_EXECUTED',
                B='UNKNOWN_NO_H4_EXECUTION_ENTRY', C='SOURCE_PREPARED_NOT_EXECUTED',
                precision_bits=manifest['precision_bits'])


def alpha_task(axes):
    import alpha_adapter as a
    results = [dict(m=x['m'], axis=x['axis'], N=x['N'], epsilon=x['epsilon'],
                    U_cat_upper=x['U_cat_upper'], saved_alpha_unchanged=x['alpha'],
                    **a.failure_upper(int(x['N']), x['epsilon'], x['U_cat_upper'])) for x in axes]
    return dict(a.aggregate(results, target='0.05'), axes=results,
                scope='ideal IID bounded shots; same coordinate events and ideal D decoder')


def reference_task(manifest, raw, prior):
    import numpy as np
    import reference_adapter as a
    from common import D, real, upper, encode
    with np.load(io.BytesIO(raw['H']), allow_pickle=False) as h, np.load(io.BytesIO(raw['STATE']), allow_pickle=False) as s:
        if (str(h['hamiltonian_hash'][()]) != manifest['hamiltonian_hash']
                or str(s['hamiltonian_hash'][()]) != manifest['hamiltonian_hash']
                or str(s['fingerprint'][()]) != manifest['state_fingerprint']
                or int(s['n_qubits'][()]) != 8 or int(s['n_electrons'][()]) != 4
                or h['one_body'].shape != (8, 8) or h['g_matrices'].shape != (12, 8, 8)
                or h['lambdas'].shape != (12,) or s['state'].shape != (70,)
                or h['constant'].dtype != np.dtype('float64') or h['one_body'].dtype != np.dtype('complex128')
                or h['g_matrices'].dtype != np.dtype('complex128') or h['lambdas'].dtype != np.dtype('float64')
                or s['state'].dtype != np.dtype('complex128') or s['basis_indices'].dtype != np.dtype('int64')):
            D.stop('snapshot/state/sector identity differs')
        sector = a.complete_number_sector(8, 4, s['basis_indices'].tolist())
        matrix, reuse = a.represented_df(h['constant'][()], h['one_body'], h['lambdas'], h['g_matrices'], sector)
        e_ref = float(manifest['constants']['energy_reference_binary64_decimal'])
        result = a.ground_bracket(matrix, s['state'].copy(), e_ref, manifest['reference_lower_offset'])
    result.update(target='minimum in complete 4-electron sector of exact polynomial in saved DF binary64 coefficients',
                  reused_excitation_functions=reuse, original_eigensolver_calls=0,
                  real_molecular_energy_decision='UNKNOWN/STOP', implementation_energy_upper=None)
    if result['eta_ref_upper'] is not None:
        eta = real(result['eta_ref_upper']['upper'])
        combined = upper(real(prior['energy']['literal_energy_upper']['upper'])+eta)
        target = real(float(manifest['constants']['target_energy_precision_ha']))
        alias = real(prior['energy']['boundary_margin']['lower']) > real(prior['delta_exact'])*eta
        result.update(ideal_decoder_saved_DF_energy_upper=encode(combined),
                      ideal_saved_DF_target_decision='SUFFICIENT' if combined < target and alias else 'UNKNOWN/STOP',
                      reference_sector_eigenvalue_alias_decision='SUFFICIENT' if alias else 'UNKNOWN/STOP')
    return result


def frozen_source():
    sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip()
    files = {}
    for name in SOURCE_FILES:
        path = HERE/name
        value = path.read_bytes()
        frozen = subprocess.check_output(['git', '-C', str(ROOT), 'show', sha+':'+path.relative_to(ROOT).as_posix()])
        if frozen != value:
            raise ValueError('UNKNOWN/STOP: fixed SOURCE must be published before science')
        files[name] = hashlib.sha256(value).hexdigest()
    return dict(commit=sha, files_sha256=files)


def execute(tasks, output):
    if output.exists() or not tasks or len(set(tasks)) != len(tasks) or any(t not in ('A', 'C') for t in tasks):
        raise ValueError('UNKNOWN/STOP: fresh output and independently selected A/C tasks required')
    source = frozen_source()
    manifest, raw, axes, prior = metadata()
    import flint
    import numpy as np
    if (sys.version_info[:2] != (3, 11) or sys.float_info.mant_dig != 53
            or np.__version__ != '1.26.4' or flint.__version__ != '0.9.0'
            or flint.__FLINT_VERSION__ != '3.6.0'):
        raise ValueError('UNKNOWN/STOP: fixed numerical runtime differs')
    output.mkdir(parents=True, exist_ok=False)
    result = dict(STATUS, status='COMPLETED_MANDATORY_STOP', SOURCE=source, inputs=manifest['files'],
                  selected_tasks=tasks, A=None, B=dict(decision='UNKNOWN/STOP', scientific_calls=0), C=None,
                  resources={'available_CPUs':len(os.sched_getaffinity(0)),
                             'MemAvailable':next(x.strip() for x in Path('/proc/meminfo').read_text().splitlines()
                                                 if x.startswith('MemAvailable:'))})
    started = time.monotonic()
    for task in tasks:
        try:
            with flint.ctx.workprec(manifest['precision_bits'][task]):
                value = alpha_task(axes) if task == 'A' else reference_task(manifest, raw, prior)
        except Exception as exc:
            # Independent tasks: one unresolved proof does not silently rerun it
            # or prevent the other separately permitted task from being evaluated.
            value = dict(decision='UNKNOWN/STOP', reason=str(exc))
        result[task] = value
        (output/(task+'.json')).write_text(json.dumps(value, indent=2)+'\n')
    result['elapsed_seconds'] = time.monotonic()-started
    (output/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--tasks', choices=('A', 'C'), nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not args.run:
        if args.tasks is not None or args.output is not None:
            parser.error('--tasks/--output require separately permitted --run')
        print(json.dumps(describe(), indent=2))
    else:
        if args.tasks is None or args.output is None:
            parser.error('--run requires --tasks and a fresh --output')
        result = execute(args.tasks, args.output)
        print(json.dumps(dict(STATUS, status=result['status'],
                              decisions={t:result[t]['decision'] for t in args.tasks}), indent=2))


if __name__ == '__main__':
    main()
