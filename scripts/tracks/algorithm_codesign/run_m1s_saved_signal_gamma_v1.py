"""Closed candidate status or separately authorized future gamma-domain one-shot."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from trottertracks.algorithm_codesign.m1s_saved_signal_gamma_v1.control import main
if __name__ == '__main__':
    raise SystemExit(main())
