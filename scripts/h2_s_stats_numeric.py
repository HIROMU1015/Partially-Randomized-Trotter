"""H2 namespace adapter over the unmodified, tested H4 normal-design kernel."""
from copy import deepcopy
from types import FunctionType

import h2_s_stats_contract as s
import h4_tol_s_stats_numeric as shared


def analyze(summary):
    # H2 A stores a ratio instead of H4's separate point_direction field.
    # Convert that saved comparison inside the authorized pass; the shared
    # kernel independently checks its exact Fraction direction against it.
    adapted=deepcopy(summary)
    for endpoint in adapted['endpoints']:
        ratio=endpoint['saved_PR_to_PF_ratio']
        endpoint['saved_point_direction']=(ratio<1)-(ratio>1)
    environment=dict(shared.__dict__,s=s,c=s.c)
    kernel=FunctionType(shared.analyze.__code__,environment,shared.analyze.__name__,shared.analyze.__defaults__)
    return kernel(adapted)
