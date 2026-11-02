"""Regroup saved F2 costs/probabilities only; never generate quantum circuits."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root', required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--atoms-root')
    source.add_argument('--features')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    from trottertracks.pr_dictionary_validation.f2_attribution import run_saved, write_groups, exact_json
    output = Path(args.output).resolve()
    # Fresh output only; no old grant, marker, ledger or STOP is opened for writing.
    output.mkdir(parents=True, exist_ok=False)
    start, cpu = time.monotonic(), time.process_time()
    result, features, summary = run_saved(args.public_root, args.atoms_root, args.features)
    write_groups(output/'GROUP_CONTRIBUTIONS.csv', result['groups'])
    (output/'ATOM_FEATURES.json').write_text(json.dumps(features, ensure_ascii=False, indent=2)+'\n')
    forbidden = ('qiskit', 'numpy', 'scipy', 'mpmath', 'trotterlib')
    if any(name in sys.modules for name in forbidden):
        raise RuntimeError('Unexpected scientific package imported by saved-only analysis')
    summary['analysis_source_sha256'] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__).resolve(), root/'src/trottertracks/pr_dictionary_validation/f2_attribution.py',
        root/'src/trottertracks/pr_dictionary_validation/f2_execution.py', root/'src/trottertracks/pr_dictionary_validation/f2_plan.py')}
    summary['output_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}
    summary['analysis_cost'] = dict(wall_seconds=time.monotonic()-start, CPU_seconds=time.process_time()-cpu,
        peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        affinity_CPUs=len(os.sched_getaffinity(0)), implementation='streamed saved CSV; integer weighted accumulators; pair metadata cached across axes',
        resource_caps=None, no_quantum_or_numerical_simulation_packages_imported=True)
    (output/'RESULTS.json').write_text(json.dumps(exact_json(summary), ensure_ascii=False, indent=2)+'\n')
    (output/'STOP.json').write_text(json.dumps(dict(status=summary['status'], mandatory_stop=True,
        new_circuits=0, new_compiles=0, grants_consumed=0, next_owner='USER_GPT'), indent=2)+'\n')
    print(json.dumps(dict(status=summary['status'], counts=summary['counts'],
        exact_reconciliation=True, all_group_axes_equal=summary['all_group_axes_equal'], cost=summary['analysis_cost']), indent=2))


if __name__ == '__main__':
    main()
