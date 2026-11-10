"""Passive A0 readiness/publication intake. No scientific execution mode."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks.rpe_normalization_intake import A0Intake


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--handoff-commit')
    parser.add_argument('--handoff-path')
    parser.add_argument('--result-path')
    args = parser.parse_args()
    if bool(args.handoff_commit) != bool(args.handoff_path) or (args.result_path and not args.handoff_path):
        parser.error('Explicit HANDOFF commit/path required; result uses the same publication commit')
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError('Use a fresh B technical output directory')
    intake = A0Intake(ROOT)
    value = intake.readiness()
    if args.handoff_path:
        value = intake.read_published(dict(commit=args.handoff_commit, path=args.handoff_path),
            None if args.result_path is None else dict(commit=args.handoff_commit, path=args.result_path))
    out.mkdir(parents=True)
    (out / 'INTAKE.json').write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=value['status'], scientific_evaluations=0, mandatory_stop=True)))


if __name__ == '__main__':
    main()
