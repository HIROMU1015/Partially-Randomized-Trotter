"""Safe metadata/refusal CLI; numerical execution intentionally unavailable."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from trottertracks.algorithm_codesign.m1s_gate0_independent_v1.control import main

if __name__ == "__main__":
    raise SystemExit(main())
