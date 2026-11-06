"""Authorized fixed H_phys sector solve, with explicit private/public outputs."""
import argparse
import importlib
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
package = types.ModuleType("b2_state_core")
package.__path__ = [str(ROOT/"src/trotterlib")]
sys.modules[package.__name__] = package
module = importlib.import_module(package.__name__+".b2_hphys_ground_state")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("hdf5", "public-output", "private-output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result = module.run(args.hdf5, args.public_output, args.private_output)
    print(result["status"])
    print("energy_Ha", result["validation"]["energy_Ha"])
    print("elapsed_seconds", result["timings"]["total_seconds"])
    raise SystemExit(0 if result["status"].endswith("CORRECTNESS_PASS") else 2)
