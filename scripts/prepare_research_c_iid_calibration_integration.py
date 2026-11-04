"""Passive integration preflight only; no scientific execution option."""
import json
from pathlib import Path
from trottertracks.c_iid_calibration_integration import preflight

if __name__ == '__main__':
    print(json.dumps(preflight(Path(__file__).resolve().parents[1]), indent=2, sort_keys=True))
