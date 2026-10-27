#!/usr/bin/env python3
"""P2 artificial Pauli study wiring; never resume a saved experiment."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))


def code_version():
    from trottertracks.pauli_jw.contexts import CodeVersion
    root = Path(__file__).resolve().parents[1]
    env = {k: v for k, v in os.environ.items() if k not in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE')}
    env.update(GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1')
    try:
        head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], env=env,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain=v1',
            '--untracked-files=normal'], env=env, capture_output=True,
            text=True, check=True).stdout
        return CodeVersion(head, 'DIRTY_WORKING_COPY' if dirty else 'CLEAN_CHECKOUT')
    except (OSError, subprocess.CalledProcessError):
        return CodeVersion(None, 'UNVERSIONED_FIXTURE')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('plan', 'synthetic', 'scientific', 'real', 'resume'), default='plan')
    parser.add_argument('--study', choices=('h4-hybrid', 'h6-hybrid', 'h6-rte', 'h2-index'), default='h4-hybrid')
    parser.add_argument('--candidate', help='optional fixed-four H6 Hybrid ID, never a new candidate')
    parser.add_argument('--output-dir', help='optional NEW directory; requires existing legacy core dependencies')
    args = parser.parse_args(argv)
    if args.mode not in ('plan', 'synthetic'):
        parser.error('P2 scientific/real/resume mode is disabled before input/provider/output access')
    if args.study == 'h2-index' and (args.mode != 'plan' or args.output_dir):
        parser.error('H2 is a read-only locator entry; no provider or file output is connected')
    if args.candidate and args.study != 'h6-hybrid':
        parser.error('--candidate is only for the existing fixed-four H6 Hybrid scope')
    from trottertracks.pauli_jw.contexts import PipelineError
    from trottertracks.pauli_jw.adapters import PauliStudyAdapter, h2_saved_index
    from trottertracks.pauli_jw.pipeline import compose, write_new_result
    from trottertracks.pauli_jw.pipeline_fixture import synthetic_context, synthetic_fixture
    try:
        code = code_version()
        if args.study == 'h2-index':
            record = h2_saved_index()
            result = {k: asdict(v) if hasattr(v, '__dataclass_fields__') else v for k, v in record.items()}
            result.update(code=asdict(code), science_executed=False, reproduces_saved_results=False)
        else:
            c = synthetic_context(args.study, code=code, candidate_id=args.candidate)
            adapter = PauliStudyAdapter(args.study)
            if args.mode == 'synthetic':
                data, providers = synthetic_fixture(c)
                out = compose(c, adapter, mode='synthetic', data=data, providers=providers)
            else:
                out = compose(c, adapter)
            if args.output_dir:
                write_new_result(out, args.output_dir)
            result = out.to_dict()
    except (PipelineError, OSError) as exc:
        parser.error(f'Pauli pipeline refused: {exc}')
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result['qualification'] in ('PLAN_ONLY', 'SYNTHETIC_END_TO_END_PASS', 'UNCONNECTED_SOURCE_INDEX') else 1


if __name__ == '__main__':
    raise SystemExit(main())
