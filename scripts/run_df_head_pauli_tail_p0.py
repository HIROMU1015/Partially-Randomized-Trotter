"""Fixed LD1/LD2 P0 only. A fresh run has no retry, mean, N, price or circuit leaf."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import df_head_pauli_tail_p0 as p0

MODULE = 'src/trottertracks/df_head_pauli_tail_p0.py'
CALLER = 'scripts/run_df_head_pauli_tail_p0.py'
SUCCESS = 'P0_OPERATOR_DICTIONARY_COMPLETE_NOT_P1_OR_RPE_QUALIFIED'
REUSED = ['src/trottertracks/h4_pr_m1_rmse.py',
    'src/trottertracks/resource_applicability/ax2b_independent_reference.py',
    'src/trotterlib/pauli_signal_adapter.py', 'src/trotterlib/rte.py',
    'src/trotterlib/df_hamiltonian.py', 'src/trotterlib/df_partial_s2.py',
    'src/trotterlib/pr2_s0_s1_validation.py']


def sources(source):
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    if source != head:
        raise ValueError('SOURCE_MUST_BE_FIXED_CURRENT_COMMIT')
    records = []
    for commit, path in [(source, MODULE), (source, CALLER), *[(p0.PLAN, p) for p in REUSED]]:
        raw = p0.blob(ROOT, commit, path)
        if (ROOT/path).read_bytes() != raw:
            raise ValueError('USED_SOURCE_CHANGED: '+path)
        records.append(dict(commit=commit, path=path, sha256=hashlib.sha256(raw).hexdigest()))
    for commit, path in [(p0.SAVED, 'scripts/h46_m5_tol_saved_input.py'),
        (p0.SAVED, 'scripts/h46_new2_core.py'),
        (p0.PLAN, 'src/trottertracks/h6_df19_rpe_d2.py'),
        (p0.M1, 'src/trotterlib/o1_m1_matched_split.py')]:
        records.append(dict(commit=commit, path=path,
            sha256=hashlib.sha256(p0.blob(ROOT, commit, path)).hexdigest()))
    return records


def execute(out, source):
    out = Path(out)
    if out.exists():
        raise ValueError('FRESH_OUTPUT_REQUIRED_NO_RETRY_OR_RESUME')
    out.mkdir(parents=True)
    def save(name, value):
        if name.endswith('.npz'):
            import numpy as np
            with (out/name).open('xb') as f:
                np.savez_compressed(f, **value)
        else:
            # Only the current-run checkpoint is replaced; historical files
            # and final products always use exclusive creation.
            with (out/name).open('w' if name == 'CHECKPOINT.json' else 'x') as f:
                f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    start = time.monotonic()
    result = dict(status='STARTED', plan_source=p0.PLAN, scientific_source=source,
        started_UTC=datetime.now(timezone.utc).isoformat(), prefixes=[], checks=[],
        scope=dict(saved_input_loads=0, full_Fock_extensions=0, Pauli_dictionaries=0,
            mean_evaluations=0, reference_actions=0, eigensolves=0, finite_normalizers=0,
            sampling=0, circuit_build_compile=0, RPE_N_RMSE_C_G_evaluations=0),
        sufficient_N=None, global_RMSE_bound=None, C=None, G=None, winner=None,
        scientific_dictionary_adopted=False, automatic_retry=False, mandatory_stop=True)
    save('EXECUTION.json', result)
    try:
        result['code_sources'] = sources(source)
        meta = p0.metadata(ROOT)
        save('FIXED_INPUTS.json', meta)
        result['binding'] = meta['binding']
        values = p0.load_saved(ROOT, meta)
        result['scope']['saved_input_loads'] = 1
        ham, sector, full_state, state, observed = values
        result['saved_loader_observed'] = observed
        p0.diagnose(ROOT, ham, sector, full_state, state, meta, result, save)
        result['scope']['Pauli_dictionaries'] = len(result['prefixes'])
        for spec in [*meta['inputs'].values(), meta['context_source']]:
            if p0.sha(spec['path']) != spec['sha256']:
                raise ValueError('USED_INPUT_BYTES_CHANGED_DURING_P0')
        import numpy, scipy, openfermion
        from openfermion.config import EQ_TOLERANCE
        result['runtime'] = dict(python=sys.version, numpy=numpy.__version__,
            scipy=scipy.__version__, openfermion=openfermion.__version__)
        result['coefficient_policy'] = dict(adapter_extra_cutoff=0.,
            collector_imaginary_tolerance=1e-12, openfermion_EQ_TOLERANCE=EQ_TOLERANCE,
            openfermion_internal_simplification_preserved=True,
            rounded_imaginary_coefficients_recorded=True, rigorous_operator_certificate=False)
        result['status'] = SUCCESS
    except Exception as error:
        result['status'] = 'STOP_P0_INPUT_OR_NUMERICAL_FAILURE'
        result['error'] = dict(type=type(error).__name__, message=str(error))
        result['scope']['Pauli_dictionaries'] = sum((out/('LD'+str(p)+'_DICTIONARY.json')).exists() for p in (1, 2))
    result['completed_UTC'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.monotonic()-start
    save('result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--source')
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.preflight:
        m = p0.metadata(ROOT)
        print(json.dumps(dict(status='METADATA_PREFLIGHT_ONLY_NO_ARRAY_LOAD',
            binding=m['binding'], prefixes=[1, 2],
            original_DF_configuration_ids=m['original_DF_configuration_ids']), indent=2))
    else:
        if not args.source or not args.out:
            parser.error('--execute requires --source and --out')
        value = execute(args.out, args.source)
        print(json.dumps(dict(status=value['status'], error=value.get('error'),
            elapsed_seconds=value['elapsed_seconds']), indent=2), flush=True)
        sys.exit(0 if value['status'] == SUCCESS else 2)
