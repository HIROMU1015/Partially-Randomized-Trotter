"""Q4 certifiers/optimizer; opt-in functions, never a scientific run on import.

Upper certificates cover continuous center cells and all count boxes. Lower
certificates require physical two-point witnesses, at ONE common center, for
every allocation and the full continuous failure-share domain.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from . import pr_reference_feedback_q4 as g


@dataclass(frozen=True)
class Model:
    rho: F
    bias: F
    prior: F
    target: F
    failure: F
    time: F
    kappa: F
    omega_energy: F
    feedback_error: F
    center_pad: F  # turns; covers supplied normalized center interval width

    def __post_init__(self):
        if not (0 < self.rho <= 1 and 0 <= self.bias < 1 and
                0 < self.prior < 1 and 0 < self.target <= 2*self.prior and
                0 < self.failure < 1 and self.time > 0 and
                min(self.kappa, self.omega_energy, self.feedback_error, self.center_pad) >= 0):
            raise ValueError('Invalid bounded-signal model')

    @property
    def restoration(self):
        return self.time*(self.kappa+self.omega_energy)


@dataclass(frozen=True)
class Allocation:
    nx: int
    ny: int
    share: F

    def __post_init__(self):
        if (type(self.nx) is not int or type(self.ny) is not int or
                min(self.nx, self.ny) < 0 or self.nx+self.ny == 0 or
                not 0 <= self.share <= 1 or
                (self.nx and self.share == 0) or (self.ny and self.share == 1) or
                (not self.nx and self.share != 0) or (not self.ny and self.share != 1)):
            raise ValueError('Invalid prefix-only allocation')

    def alphas(self, total):
        return total*self.share, total*(1-self.share)

    def record(self):
        return dict(nx=self.nx, ny=self.ny, share=str(self.share))


def ci_function(family):
    if family not in ('H', 'E'):
        raise ValueError('Only registered H/E CI families')
    return g.hoeffding if family == 'H' else g.clopper_pearson


def wrapped(cell):
    lo, hi = cell
    if hi-lo >= 1:
        return [(F(0), F(1))]
    j = lo//1; lo -= j; hi -= j
    return [(lo, hi)] if hi <= 1 else [(lo, F(1)), (F(0), hi-1)]


def fit_point(a, model, u, b, psi, cis, geometry):
    """Construct a common rational radius, then verify TRUE axis means in INNER CI."""
    if not -model.prior <= u <= model.prior or not -model.bias <= b <= model.bias:
        return None
    theta = 2*a.iv.pi*a.r(psi)+a.r(u+b)
    offsets = [a.r(0), a.iv.pi/2 if geometry == 'ideal' else a.r(g.NATIVE_A)]
    lo, hi = model.rho, F(1)
    for ci, offset in zip(cis, offsets):
        l, h = ci.inner
        if l > h:
            return None
        v = a.iv.cos(theta-offset); vl, vh = a.bounds(v)
        if vl <= 0 <= vh:
            if vl == vh == 0 and l <= 0 <= h:
                continue
            return None
        lower, upper = ((a.r(l)/v, a.r(h)/v) if vl > 0 else
                        (a.r(h)/v, a.r(l)/v))
        lo = max(lo, a.bounds(lower)[1]); hi = min(hi, a.bounds(upper)[0])
    if lo > hi:
        return None
    point = dict(u=F(u), b=F(b), rho=(lo+hi)/2)
    return point if verify_point(a, model, psi, cis, geometry, point) else None


def verify_point(a, model, psi, cis, geometry, point):
    u, b, rho = (F(point[k]) for k in ('u', 'b', 'rho'))
    if not (-model.prior <= u <= model.prior and -model.bias <= b <= model.bias and
            model.rho <= rho <= 1):
        return False
    theta = 2*a.iv.pi*a.r(psi)+a.r(u+b)
    offsets = [a.r(0), a.iv.pi/2 if geometry == 'ideal' else a.r(g.NATIVE_A)]
    for ci, offset in zip(cis, offsets):
        lo, hi = a.bounds(a.r(rho)*a.iv.cos(theta-offset))
        if lo < ci.inner[0] or hi > ci.inner[1]:
            return False
    return True


def find_pair(a, model, psi, cis, geometry):
    # Candidate search is a heuristic only. Every returned point is separately
    # checked against true-axis interval means. Failure to find one proves nothing.
    if any(c.inner[0] > c.inner[1] for c in cis):
        return None
    inner = [g.Confidence(c.inner[0], c.inner[0], c.inner[1], c.inner[1]) for c in cis]
    regions = g.residual_region(a, inner, model.rho, model.bias,
                                model.prior, (psi, psi), geometry)
    if g.width_of(regions) is None or g.width_of(regions) <= model.target:
        return None
    extremes = []
    for left_side in (True, False):
        ordered = regions if left_side else list(reversed(regions))
        found = None
        for lo, hi in ordered:
            for j in range(128, 0, -1):
                u = lo+(hi-lo)/2**j if left_side else hi-(hi-lo)/2**j
                for b in (model.bias if left_side else -model.bias, F(0),
                          -model.bias if left_side else model.bias):
                    found = fit_point(a, model, u, b, psi, cis, geometry)
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            return None
        extremes.append(found)
    if extremes[1]['u']-extremes[0]['u'] <= model.target:
        return None
    return [{k:str(v) for k,v in p.items()} for p in extremes]


def certify_counts(a, model, allocation, psi, family, geometry):
    """All count boxes, outward geometry; FAIL only with a verified point pair."""
    ci = ci_function(family)
    ax, ay = allocation.alphas(model.failure)
    ns, alphas = (allocation.nx, allocation.ny), (ax, ay)
    @lru_cache(None)
    def single(axis, k):
        return ci(a, ns[axis], k, alphas[axis])
    stack = [(0, ns[0], 0, ns[1])]
    leaves = []; math_only = False; maximum = F(0)
    while stack:
        box = stack.pop()
        bands = []
        for axis, (l, h) in enumerate(((box[0],box[1]), (box[2],box[3]))):
            lower, upper = single(axis,l), single(axis,h)
            bands.append(g.Confidence(lower.lower_lo,lower.lower_hi,
                                      upper.upper_lo,upper.upper_hi,
                                      lower.resolved and upper.resolved))
        region = g.residual_region(a, bands, model.rho, model.bias,
                                   model.prior, psi, geometry)
        width = g.width_of(region)
        if width is None:
            leaves.append(dict(box=list(box),status='INCONSISTENT_COUNTS',regions=[]))
            continue
        maximum = max(maximum,width) if box[0]==box[1] and box[2]==box[3] else maximum
        if width+model.restoration <= model.target:
            maximum = max(maximum,width)
            leaves.append(dict(box=list(box),status='PASS_IMPLEMENTED',
                               regions=[[str(l),str(h)] for l,h in region]))
            continue
        if box[0]==box[1] and box[2]==box[3]:
            center = sum(psi,F(0))/2
            pair = find_pair(a, model, center, bands, geometry)
            if pair:
                return dict(status='FAIL',count=[box[0],box[2]],psi=str(center),pair=pair)
            if width <= model.target:
                math_only = True
                leaves.append(dict(box=list(box),status='MATHEMATICAL_WIDTH_ONLY',
                                   regions=[[str(l),str(h)] for l,h in region]))
                continue
            return dict(status='UNRESOLVED',box=list(box),psi=list(map(str,psi)),
                        width_upper=str(width),reason='Outer width too wide; no feasible pair certified')
        if box[1]-box[0] >= box[3]-box[2]:
            mid=(box[0]+box[1])//2
            stack.extend([(mid+1,box[1],box[2],box[3]),(box[0],mid,box[2],box[3])])
        else:
            mid=(box[2]+box[3])//2
            stack.extend([(box[0],box[1],mid+1,box[3]),(box[0],box[1],box[2],mid)])
    return dict(status='MATHEMATICAL_WIDTH_ONLY' if math_only else 'PASS_IMPLEMENTED',
                allocation=allocation.record(),psi=list(map(str,psi)),
                maximum_width_upper=str(maximum),leaves=leaves)


def exclude_share_box(a, model, nx, ny, share_box, psi, counts, family, geometry):
    # alpha maxima for BOTH axes is an optimistic narrower-CI relaxation of
    # every legal split in this box. A pair in it excludes the whole alpha box.
    lower, upper = share_box
    alphas = [model.failure*upper, model.failure*(1-lower)]
    ns = [nx,ny]
    if any(n and alpha == 0 for n,alpha in zip(ns,alphas)):
        return dict(status='EMPTY_MEASURED_ALPHA_DOMAIN')
    ci = ci_function(family)
    bands = [ci(a,n,k,alpha) for n,k,alpha in zip(ns,counts,alphas)]
    pair = find_pair(a,model,psi,bands,geometry)
    return None if not pair else dict(status='FAIL_ALL_SHARES',share_box=list(map(str,share_box)),
        psi=str(psi),count=list(counts),alpha_max=list(map(str,alphas)),pair=pair)


def search_share(a, model, nx, ny, psi, family, geometry):
    if not nx or not ny:
        allocation=Allocation(nx,ny,F(0) if not nx else F(1))
        result=certify_counts(a,model,allocation,(psi,psi),family,geometry)
        result.update(nx=nx,ny=ny)
        return result
    stack=[(F(0),F(1))]; exclusions=[]; unresolved=[]
    while stack:
        box=stack.pop(); share=sum(box,F(0))/2
        allocation=Allocation(nx,ny,share)
        result=certify_counts(a,model,allocation,(psi,psi),family,geometry)
        if result['status']=='PASS_IMPLEMENTED':
            return result
        if result['status']=='FAIL':
            exclusion=exclude_share_box(a,model,nx,ny,box,psi,result['count'],family,geometry)
            if exclusion:
                exclusions.append(exclusion)
                continue
        if box[1]-box[0]<=g.RESOLUTION:
            # A numerical seam cannot establish an alpha optimum. Retain the
            # entire unresolved box rather than hiding it behind grid pruning.
            unresolved.append(dict(share_box=list(map(str,box)),result=result))
            continue
        stack.extend([(share,box[1]),(box[0],share)])
    return dict(status='UNRESOLVED' if unresolved else 'FAIL_ALL_SHARES',
                nx=nx,ny=ny,psi=str(psi),exclusions=exclusions,unresolved=unresolved)


def lookup_for_total(a, model, total, family, geometry, point_cache=None):
    """Uniform BUDGET total: each center may choose ANY actual N <= total.

    Failing exact-N policies at different centers would NOT exclude this budget.
    A lower witness must exclude ALL smaller totals at one common center too.
    """
    pending=[(F(0),F(1))]; cells=[]
    order=[(nx,n-nx) for n in range(total,0,-1)
           for nx in sorted(range(n+1),key=lambda x:abs(2*x-n))]
    point_cache={} if point_cache is None else point_cache
    while pending:
        cell=pending.pop(); center=sum(cell,F(0))/2
        outcomes=[]; selected=None
        for nx,ny in order:
            key=(nx,ny,center,family,geometry)
            if key not in point_cache:
                point_cache[key]=search_share(a,model,nx,ny,center,family,geometry)
            result=point_cache[key]
            if result['status']=='PASS_IMPLEMENTED':
                selected=Allocation(nx,ny,F(result['allocation']['share']))
                break
            outcomes.append(result)
        if selected is None:
            failed=all(r['status'] in ('FAIL','FAIL_ALL_SHARES') for r in outcomes)
            return dict(status='FAIL' if failed else 'UNRESOLVED',total=total,
                        witness_center=str(center),allocations=outcomes,partial_cells=cells)
        extended=wrapped((cell[0]-model.center_pad,cell[1]+model.center_pad))
        proofs=[certify_counts(a,model,selected,c,family,geometry) for c in extended]
        if all(p['status']=='PASS_IMPLEMENTED' for p in proofs):
            cells.append(dict(domain=list(map(str,cell)),certified_domains=[list(map(str,c)) for c in extended],
                              allocation=selected.record(),proofs=proofs))
        elif cell[1]-cell[0]<=g.RESOLUTION:
            return dict(status='UNRESOLVED',total=total,center_leaf=list(map(str,cell)),
                        proofs=proofs,partial_cells=cells)
        else:
            pending.extend([(center,cell[1]),(cell[0],center)])
    return dict(status='PASS_IMPLEMENTED',total=total,cells=cells)


def select_policy(cells, center_interval, center_pad):
    """Observable rational interval midpoint; never choose an unknown true phase."""
    lo,hi=center_interval
    if hi<lo or hi-lo>2*center_pad:
        raise ValueError('Unqualified normalized center interval')
    center=((lo+hi)/2)%1
    for cell in cells:
        left,right=map(F,cell['domain'])
        if left<=center<=right:
            covers=all(any(F(l)<=u<=v<=F(h) for l,h in cell['certified_domains'])
                       for u,v in wrapped((lo,hi)))
            if covers:
                p=cell['allocation']
                return Allocation(p['nx'],p['ny'],F(p['share']))
    raise ValueError('No certified lookup cell covers supplied center interval')


def center_turn_interval(a,model,energy_center):
    """Normalize a supplied Q3-B center enclosure, not a true energy oracle.

    Keep an unwrapped interval across period cuts; selection uses its observable
    rational midpoint. Alias |hE|<pi and prefix error c imply |t17 Ec|<2^20.
    """
    lo,hi=map(F,energy_center)
    if lo>hi or hi-lo>model.kappa:
        raise ValueError('Energy-center enclosure exceeds saved kappa')
    if max(abs(model.time*lo),abs(model.time*hi))>1<<20:
        raise ValueError('Center outside inherited alias envelope')
    turns=a.bounds(-a.r(model.time)*a.interval(lo,hi)/(2*a.iv.pi))
    if turns[1]-turns[0]>2*model.center_pad:
        raise ValueError('192-bit normalized center enclosure too wide')
    return turns


def solve_xy(a, model, family, geometry, maximum, saved_lower=1):
    results=[]; lower=1; feasible=None; point_cache={}
    for total in range(1,maximum+1):
        if total < saved_lower:
            results.append(dict(total=total,status='FAIL_SAVED_WITNESS'))
            lower=total+1
            continue
        result=lookup_for_total(a,model,total,family,geometry,point_cache)
        results.append(result)
        if result['status']=='FAIL' and total==lower:
            lower+=1
        if result['status']=='PASS_IMPLEMENTED':
            feasible=total
            break
    return dict(lower=lower,feasible=feasible,gap=None if feasible is None else feasible-lower,
                exact=feasible==lower if feasible is not None else False,results=results,
                family=family,geometry=geometry,maximum=maximum)


def certify_q(a, model, n):
    """E matched Q inverse: all counts, saved feedback error retained once."""
    records=[]; maximum=F(0); unresolved=False
    for k in range(n+1):
        ci=g.clopper_pearson(a,n,k,model.failure)
        region=g.q_region(a,ci,model.rho,model.bias+model.feedback_error,model.prior)
        width=g.width_of(region)
        if width is None:
            records.append(dict(k=k,status='INCONSISTENT_COUNTS'))
            continue
        maximum=max(maximum,width)
        if width+model.restoration<=model.target:
            records.append(dict(k=k,status='PASS_IMPLEMENTED',region=[[str(l),str(h)] for l,h in region]))
            continue
        # Outer width alone cannot prove failure. A general sine CI has an
        # INNER preimage; use it to prove an exact-mathematical width violation.
        inner=g.Confidence(ci.lower_hi,ci.lower_hi,ci.upper_lo,ci.upper_lo)
        if inner.lower_lo<=inner.upper_hi:
            inner_region=g.q_region(a,inner,model.rho,model.bias+model.feedback_error,model.prior)
            iw=g.width_of(inner_region)
            # g.q_region is outward; subtract a full endpoint enclosure slack
            # by verifying two explicit feasible sine-model points instead.
            if iw is not None and iw>model.target:
                left,right=inner_region[0][0],inner_region[-1][1]
                for j in range(1,129):
                    us=[left+(right-left)/2**j,right-(right-left)/2**j]
                    if us[1]-us[0]<=model.target:
                        continue
                    points=[]
                    for u,sign in zip(us,(1,-1)):
                        b=sign*(model.bias+model.feedback_error)
                        value=a.iv.sin(a.r(u+b))
                        vl,vh=a.bounds(value)
                        # Select rho through a strict interval quotient.
                        if vl<=0<=vh:
                            break
                        l,h=ci.inner
                        lb,ub=(a.r(l)/value,a.r(h)/value) if vl>0 else (a.r(h)/value,a.r(l)/value)
                        rl=max(model.rho,a.bounds(lb)[1]);rh=min(F(1),a.bounds(ub)[0])
                        if rl>rh:
                            break
                        rho=(rl+rh)/2; ml,mh=a.bounds(a.r(rho)*value)
                        if ml<l or mh>h:
                            break
                        points.append(dict(u=str(u),b=str(b),rho=str(rho)))
                    if len(points)==2:
                        return dict(n=n,status='FAIL',k=k,pair=points)
        unresolved=True
        records.append(dict(k=k,status='MATHEMATICAL_WIDTH_ONLY' if width<=model.target else 'UNRESOLVED',
                            width_upper=str(width)))
    return dict(n=n,status='UNRESOLVED' if unresolved else 'PASS_IMPLEMENTED',
                maximum_width_upper=str(maximum),counts=records)


def solve_q(a,model,maximum):
    lower=1;feasible=None;results=[]
    for n in range(1,maximum+1):
        result=certify_q(a,model,n);results.append(result)
        if result['status']=='FAIL' and n==lower:
            lower+=1
        if result['status']=='PASS_IMPLEMENTED':
            feasible=n
            break
    return dict(lower=lower,feasible=feasible,gap=None if feasible is None else feasible-lower,
                exact=feasible==lower if feasible is not None else False,results=results,maximum=maximum)


def fraction_accounting():
    return g.pure_functions('scripts/run_pr_reference_feedback_q3cd.py',
                            {'pair','add','scale','subtract'},{'F':F})


def native_cost(prefix, prices, allocations):
    """Worst case over prefix policies, expected per-trajectory saved prices."""
    f=fraction_accounting()
    costs=[f['add']([prefix,f['scale'](p.nx,prices['X']),f['scale'](p.ny,prices['Y'])])
           for p in allocations]
    if not costs:
        raise ValueError('No qualified allocation')
    return [max(p[i] for p in costs) for i in (0,1)]


def correlated_saving(q_n, allocation, prices):
    # Use the SAME mu_Y for Q and Y. Negative coefficients reverse endpoints.
    f=fraction_accounting()
    def scaled(n,p):
        return f['scale'](n,p) if n>=0 else [n*p[1],n*p[0]]
    return f['add']([scaled(allocation.nx,prices['X']),scaled(allocation.ny-q_n,prices['Y'])])
