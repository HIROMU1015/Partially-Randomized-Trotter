"""Explicit LOCAL stage-one entry; the scientific run is separately authorized."""
from pathlib import Path
import argparse
import importlib
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
# Reuse the existing light-package pattern; avoid unrelated eager research imports.
package = types.ModuleType("b2_correspondence_core")
package.__path__ = [str(ROOT/"src/trotterlib")]
sys.modules[package.__name__] = package
module = importlib.import_module(package.__name__+".b2_original_df12_correspondence")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = module.run(args.hdf5, args.output)
    print(result["status"])
    print("elapsed_seconds", result["elapsed_seconds"])
    raise SystemExit(0 if result["status"].startswith("OPERATOR_CHECKS_PASS") else 2)
