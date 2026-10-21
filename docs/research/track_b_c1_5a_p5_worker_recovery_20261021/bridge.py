"""Load unchanged C1-5A dependency in its original support namespace."""
import sys
from common import OLD

def connect():
    old=str(OLD)
    if old not in sys.path:sys.path.insert(0,old)
    import diagnostic
    return diagnostic
