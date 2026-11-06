"""Approved saved-data-only entry; the scientific module preserves STOPs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trottertracks.track_c2_b_strong_baseline import run
if __name__=='__main__':run()
