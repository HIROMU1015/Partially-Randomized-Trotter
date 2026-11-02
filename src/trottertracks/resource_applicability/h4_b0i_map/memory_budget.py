"""Current user-approved parallel memory headroom; other jobs remain untouched."""
from ..h4_geometry.identity import require

GiB=2**30
HEADROOM=64*GiB


def validate_cap(cap,workers):
    require(type(cap) is int and cap==32*GiB,'parallel owned worker AS/RSS32GiB')
    require(type(workers) is int and 1<=workers<=12,'bounded parallel worker count')
    return cap


def required_available(workers,cap):
    validate_cap(cap,workers)
    return 8*GiB+workers*cap+HEADROOM+256*2**20
