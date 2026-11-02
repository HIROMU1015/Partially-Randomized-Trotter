"""Saved-value common-P sensitivity, without automatic research decisions."""
import math
from .identity import require
from .signal import resource_point


def common_P_envelope(saved_candidates,epsilon):
    """Analytic affine intersections over common P>=0; preserve exact point ties."""
    lines={}
    for key,saved in saved_candidates.items():
        point=resource_point(saved,epsilon)
        if point['eligible']:
            lines[key]=(point['work'],sum(point['shots']))
    crossings={0.}
    keys=list(lines)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            intercept_a,slope_a=lines[a];intercept_b,slope_b=lines[b]
            if slope_a!=slope_b:
                crossing=(intercept_b-intercept_a)/(slope_a-slope_b)
                if math.isfinite(crossing) and crossing>=0:
                    crossings.add(crossing)
    def active(P):
        if not lines:return []
        work={key:a+b*P for key,(a,b) in lines.items()};minimum=min(work.values())
        return sorted(k for k,v in work.items() if v==minimum)
    boundaries=sorted(crossings)
    points=[{'P':P,'point_minimum_candidates':active(P)} for P in boundaries]
    intervals=[]
    for i,left in enumerate(boundaries):
        right=boundaries[i+1] if i+1<len(boundaries) else None
        probe=(left+right)/2 if right is not None else left+max(1.,abs(left))
        intervals.append({'P_left':left,'P_right':right,'point_minimum_candidates':active(probe)})
    return {'affine_lines':lines,'boundaries':points,'intervals':intervals,
            'research_decision':None,'next_stage_authorized':False}
