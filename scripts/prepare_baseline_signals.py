"""Metadata-only baseline request CLI. Never executes science, even on cache miss."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from trottertracks import baseline_signal_pipeline as pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Published B V2 configuration JSON')
    parser.add_argument('--target', required=True, help='Original-H target descriptor JSON')
    parser.add_argument('--references', help='Reference descriptors only; no numerical evaluation')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    read = lambda name: json.loads(Path(name).read_text())
    plan = pipeline.prepare(ROOT, read(args.config), target=read(args.target),
                            references=[] if args.references is None else read(args.references))
    with Path(args.out).open('x') as f:json.dump(plan, f, indent=2, allow_nan=False); f.write('\n')


if __name__ == '__main__':
    main()
