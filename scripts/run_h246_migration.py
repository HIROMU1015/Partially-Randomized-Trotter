"""New-environment input reconstruction, explicit separate grants only."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h246_migration', ROOT/'src/trotterlib/h246_migration.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--private', required=True)
    p.add_argument('--prepare-generation', action='store_true')
    p.add_argument('--generate', choices=('H2', 'H4'))
    a = p.parse_args()
    if a.prepare_generation:
        out = migration.prepare_generation(ROOT, a.private)
    else:
        migration.require(a.generate, 'EXPLICIT_GENERATION_MODE_REQUIRED')
        out = migration.run_generation(ROOT, a.private, a.generate)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    main()
