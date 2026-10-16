"""Metadata status or separately authorized future saved-interval comparison."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from trottertracks.algorithm_codesign.m1s_gamma_pi3_v1.control import main
if __name__=='__main__':
    raise SystemExit(main())
