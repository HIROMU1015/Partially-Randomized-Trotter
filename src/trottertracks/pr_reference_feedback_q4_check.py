"""Replay finite Q4 certificate leaves; no optimizer/campaign calls.

This is a separate checking path by the same author, NOT an independent review.
Geometry/CI primitives are shared; artificial analytic cases test those primitives.
"""
from fractions import Fraction as F
from . import pr_reference_feedback_q4 as g
from . import pr_reference_feedback_q4_certify as c


def count_cover(boxes, nx, ny):
    """Exact integer-lattice tiling, no double count or missing lattice point."""
    events={0:[],nx+1:[]}
    for x0,x1,y0,y1 in boxes:
        if not 0<=x0<=x1<=nx or not 0<=y0<=y1<=ny:
            raise ValueError('Count box outside domain')
        events.setdefault(x0,[]).append((y0,y1,1))
        events.setdefault(x1+1,[]).append((y0,y1,-1))
    diff=[0]*(ny+2); xs=sorted(events)
    for x,next_x in zip(xs,xs[1:]):
        for lo,hi,delta in events[x]:
            diff[lo]+=delta;diff[hi+1]-=delta
        if next_x>x:
            active=0
            for y in range(ny+1):
                active+=diff[y]
                if active!=1:
                    raise ValueError('Incomplete or overlapping count certificate')
    return True


def counts_pass(a,model,result,family,geometry):
    if result['status']!='PASS_IMPLEMENTED':
        raise ValueError('Not an implementation PASS')
    p=result['allocation'];allocation=c.Allocation(p['nx'],p['ny'],F(p['share']))
    psi=tuple(map(F,result['psi']));ci=c.ci_function(family)
    alphas=allocation.alphas(model.failure);ns=[allocation.nx,allocation.ny];cache={}
    def checked_ci(axis,k):
        key=(axis,k)
        if key not in cache:
            value=ci(a,ns[axis],k,alphas[axis])
            if family=='E':
                cp_endpoints(ns[axis],k,alphas[axis],value)
            cache[key]=value
        return cache[key]
    count_cover([r['box'] for r in result['leaves']],*ns)
    for row in result['leaves']:
        box=row['box'];bands=[]
        for i,(lo,hi) in enumerate(((box[0],box[1]),(box[2],box[3]))):
            left,right=checked_ci(i,lo),checked_ci(i,hi)
            bands.append(g.Confidence(left.lower_lo,left.lower_hi,right.upper_lo,right.upper_hi))
        regions=g.residual_region(a,bands,model.rho,model.bias,model.prior,psi,geometry)
        width=g.width_of(regions)
        if width is None:
            if row['status']!='INCONSISTENT_COUNTS':
                raise ValueError('Empty set status changed')
        elif row['status']!='PASS_IMPLEMENTED' or width+model.restoration>model.target:
            raise ValueError('Energy width not sufficient')
        if [[str(l),str(h)] for l,h in regions]!=row['regions']:
            raise ValueError('Geometry leaf changed')
    return True


def pair_fail(a,model,result,nx,ny,family,geometry,alphas):
    ci=c.ci_function(family);ns=[nx,ny]
    bands=[ci(a,n,k,alpha) for n,k,alpha in zip(ns,result['count'],alphas)]
    if family=='E':
        for n,k,alpha,value in zip(ns,result['count'],alphas,bands):
            cp_endpoints(n,k,alpha,value)
    psi=F(result['psi']);points=result['pair']
    if len(points)!=2 or F(points[1]['u'])-F(points[0]['u'])<=model.target:
        raise ValueError('No separated pair')
    if not all(c.verify_point(a,model,psi,bands,geometry,p) for p in points):
        raise ValueError('Pair does not fit true common signal/inner CI')
    return True


def share_fail(a,model,result,family,geometry):
    nx,ny=result['nx'],result['ny']
    boxes=[]
    for item in result['exclusions']:
        lo,hi=map(F,item['share_box'])
        expected=[model.failure*hi,model.failure*(1-lo)]
        if list(map(F,item['alpha_max']))!=expected:
            raise ValueError('Invalid optimistic alpha relaxation')
        pair_fail(a,model,item,nx,ny,family,geometry,expected)
        boxes.append((lo,hi))
    if result['unresolved'] or g.merge(boxes)!=[(F(0),F(1))]:
        raise ValueError('Continuous alpha domain not excluded')
    return True


def total_result(a,model,result,family,geometry):
    if result['status']=='PASS_IMPLEMENTED':
        domains=[]
        for cell in result['cells']:
            lo,hi=map(F,cell['domain']);domains.append((lo,hi))
            expected=c.wrapped((lo-model.center_pad,hi+model.center_pad))
            if expected!=[tuple(map(F,p)) for p in cell['certified_domains']]:
                raise ValueError('Missing center-interval padding')
            if len(cell['proofs'])!=len(expected):
                raise ValueError('Missing periodic proof component')
            for proof,domain in zip(cell['proofs'],expected):
                if tuple(map(F,proof['psi']))!=domain or proof['allocation']!=cell['allocation']:
                    raise ValueError('Policy not bound to certificate')
                if proof['allocation']['nx']+proof['allocation']['ny']>result['total']:
                    raise ValueError('Policy exceeds uniform budget')
                counts_pass(a,model,proof,family,geometry)
        if g.merge(domains)!=[(F(0),F(1))]:
            raise ValueError('Center domain incomplete')
    elif result['status']=='FAIL':
        total=result['total'];rows=result['allocations'];psi=F(result['witness_center'])
        indices=[]
        for row in rows:
            if row['status']=='FAIL_ALL_SHARES':
                if F(row['psi'])!=psi or not 1<=row['nx']+row['ny']<=total:
                    raise ValueError('Not one common center for all allocations')
                share_fail(a,model,row,family,geometry);indices.append((row['nx'],row['ny']))
            elif row['status']=='FAIL':
                # Only an unmeasured-axis endpoint allocation can return a
                # direct FAIL, since measured two-axis shares require coverage.
                nx=row.get('nx');ny=row.get('ny')
                if nx is None or ny is None or min(nx,ny)!=0 or not 1<=nx+ny<=total or F(row['psi'])!=psi:
                    raise ValueError('Unbound endpoint allocation')
                alphas=[F(0),model.failure] if nx==0 else [model.failure,F(0)]
                pair_fail(a,model,row,nx,ny,family,geometry,alphas);indices.append((nx,ny))
            else:
                raise ValueError('Unresolved allocation counted as FAIL')
        if sorted(indices)!=sorted((x,n-x) for n in range(1,total+1) for x in range(n+1)):
            raise ValueError('Missing integer allocation')
    else:
        return 'UNRESOLVED_NOT_PROMOTED'
    return True


def rational_tail(n,k,p):
    """Independent exact polynomial, useful for checking root brackets."""
    from math import comb
    return sum((F(comb(n,j))*p**j*(1-p)**(n-j) for j in range(k,n+1)),F(0))


def root_bracket(n,k,target,bracket):
    lo,hi=map(F,bracket)
    if not 0<=lo<=hi<=1 or rational_tail(n,k,lo)>target or rational_tail(n,k,hi)<target:
        raise ValueError('Binomial root not enclosed')
    return True


def cp_endpoints(n,k,alpha,ci):
    if n and k:
        root_bracket(n,k,alpha/2,[(ci.lower_lo+1)/2,(ci.lower_hi+1)/2])
    if n and k<n:
        root_bracket(n,n-k,alpha/2,[(1-ci.upper_hi)/2,(1-ci.upper_lo)/2])
    return True
