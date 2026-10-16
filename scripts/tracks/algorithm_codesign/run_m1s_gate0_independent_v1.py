"""Safe metadata CLI; scientific dispatch requires separate committed approval."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'src'))
from trottertracks.algorithm_codesign.m1s_gate0_independent_v1.control import main
if __name__ == '__main__':
    raise SystemExit(main())
