"""Fixed-scope entry; STOP guard is implemented in the scientific module."""
import os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
os.environ.setdefault('TRACK_C_H4_COMMON_CACHE',str(root/'artifacts/track_c2_a_pilot/2026-11-06/common.pkl'))
from trottertracks.track_c2_a_pilot import run
if __name__=='__main__':run()
