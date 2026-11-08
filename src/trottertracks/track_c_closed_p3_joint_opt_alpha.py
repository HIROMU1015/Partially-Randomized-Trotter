"""Small integer-alpha refinement of the existing continuous allocation.

Directed inverse shot bounds tighten unused alpha after ceil. A local exchange
removes one expensive trial, funding it with cheaper extra trials if needed.
This is a bounded local search, not a global integer optimization certificate.
"""
from fractions import Fraction as F
from functools import lru_cache
import math
import numpy as np
import mpmath as mp
from .track_c_bottleneck import allocate, rigorous_shots
from . import track_c_source_guarantee as sg


def discrete_allocate(points,prior_alpha=None):
    _,proposal=allocate(points,0); shape=proposal.shape
    costs=np.array([p['mean'][:,0] for p in points]).reshape(-1)
    qs=[p['qualification'] for p in points for _ in (0,1)]
    mp.iv.prec=192
    K=[sg.up(4/(sg.I(q['rho_lower'])**2*mp.iv.sin(sg.I(q['beta_stat_lower']))**2)) for q in qs]
    budget=F(1,20)

    @lru_cache(None)
    def required(i,n):
        # Upper K and upper exp make this a sufficient alpha, rather than
        # trusting a float inverse exactly on an integer-shot boundary.
        value=sg.up(2*mp.iv.exp(-sg.I(int(n))/sg.I(K[i])))
        value=math.nextafter(value,math.inf)
        q=qs[i]
        while rigorous_shots(q['rho_lower'],q['beta_stat_lower'],value)>n:value=math.nextafter(value,math.inf)
        return F.from_float(value)

    starts=[proposal.reshape(-1)]
    if prior_alpha is not None:
        starts.append(np.array([[rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],a) for a in row] for p,row in zip(points,prior_alpha)]).reshape(-1))
    best=None
    for start in starts:
        ns=list(map(int,start)); aa=[required(i,n) for i,n in enumerate(ns)]
        while sum(aa)>budget:
            j=max(range(36),key=lambda k:float(aa[k]-required(k,ns[k]+1))/costs[k]); ns[j]+=1;aa[j]=required(j,ns[j])
        while True:
            current=sum(aa); exchange=None
            for i in np.argsort(-costs):
                i=int(i)
                if ns[i]<=1:continue
                trial=ns[:]; alphas=aa[:];trial[i]-=1;alphas[i]=required(i,trial[i]); total=current-aa[i]+alphas[i]; expense=0.
                for _ in range(64):
                    if total<=budget:break
                    options=[j for j in range(36) if j!=i]
                    j=max(options,key=lambda j:float(alphas[j]-required(j,trial[j]+1))/costs[j])
                    expense+=costs[j]
                    if expense>=costs[i]:break
                    old=alphas[j];trial[j]+=1;alphas[j]=required(j,trial[j]);total+=alphas[j]-old
                profit=costs[i]-expense
                if total<=budget and profit>0 and (exchange is None or profit>exchange[0]):exchange=(profit,trial,alphas)
            if exchange is None:break
            _,ns,aa=exchange
        value=float(np.dot(costs,ns))
        if best is None or value<best[0]:best=(value,ns,aa)
    _,ns,aa=best; alpha=np.array(list(map(float,aa))).reshape(shape);ns=np.array(ns,dtype=np.int64).reshape(shape)
    assert sum(F.from_float(float(a)) for a in alpha.flat)<=budget
    for p,ar,nr in zip(points,alpha,ns):
        for a,n in zip(ar,nr):assert rigorous_shots(p['qualification']['rho_lower'],p['qualification']['beta_stat_lower'],a)<=n
    return alpha,ns
