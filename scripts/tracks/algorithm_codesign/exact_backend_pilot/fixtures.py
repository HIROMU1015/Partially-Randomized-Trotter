"""Artificial coefficients only. Does not read any research table."""
from fractions import Fraction as F

def strings(value):
    if isinstance(value, list):
        return [strings(x) for x in value]
    return str(F(value))

def lp(name, c, upper, A=(), b=(), H=(), f=(), offset=0, expected='OPTIMAL', tags=()):
    n=len(c)
    assert len(upper)==n and len(A)==len(b) and len(H)==len(f)
    assert all(len(row)==n for row in list(A)+list(H))
    return dict(id=name, c=strings(list(c)), U=strings(list(upper)),
                A=strings(list(map(list,A))), b=strings(list(b)),
                H=strings(list(map(list,H))), f=strings(list(f)),
                c0=strings(offset), expected_status=expected, tags=list(tags), artificial=True)

def shaped(kind, mode):
    # The numbers below are invented, not loaded from or calibrated to RA-D0.
    group_count=8 if kind=='B2' else 7
    rep_sizes=[3,3,2]
    n=3*group_count+(4 if kind=='B2' else 1)
    y=n-1
    def row(): return [F(0)]*n
    cbar=[F(2)+F(g,7) for g in range(group_count)]
    v=[]
    if kind=='B2':
        for size in rep_sizes:
            v += [[F(k+1,size) for k in range(3)] for _ in range(size)]
    else:
        v=[[F(k+1,group_count) for k in range(3)] for _ in range(group_count)]
    H=[]; f=[]
    normal=row()
    for g in range(group_count):
        for p in range(3): normal[3*g+p]=cbar[g]
    H.append(normal);f.append(1)
    if kind=='B2':
        g=0
        for r,size in enumerate(rep_sizes):
            for _ in range(size):
                h=row()
                for p in range(3):h[3*g+p]=1
                h[3*group_count+r]=-1
                H.append(h);f.append(0);g+=1
        h=row();h[y]=-1
        for r in range(3):h[3*group_count+r]=1
        H.append(h);f.append(0)
    for k in range(3):
        h=row();h[y]=-(k+1)
        for g in range(group_count):
            for p in range(3):h[3*g+p]=v[g][k]
        H.append(h);f.append(0)
    A=[];b=[]
    mean=row();mean[y]=-F(1,10000)
    confidence=row();confidence[y]=-F(1,9)
    for g in range(group_count):
        for p in range(3):
            mean[3*g+p]=F(g+1,10**9)
            confidence[3*g+p]=2*F(g+1,(p+1)*10**6)
    # Synthetic reserved-inner and unreserved-outer rows, never original class claims.
    A += [mean,confidence]; b += [-F(1,10**8) if mode=='inner' else 0,
                                  -F(1,200)- (F(1,100000) if mode=='inner' else 0)]
    for q,cap in enumerate([10,12,14]):
        h=row()
        for g in range(group_count):
            for p in range(3):h[3*g+p]=F((g+1)*(p+1)+q,5)
        A.append(h);b.append(cap)
    h=row();h[y]=-1;A.append(h);b.append(-F(1,100))
    if mode=='infeasible':
        A.append(normal);b.append(F(1,2)) # conflicts exactly with normalization=1
    c=row()
    for g in range(group_count):
        for p in range(3):c[3*g+p]=F((g+1)*(p+1),7)
    result=lp(f'{kind}_{mode}',c,[2]*(3*group_count)+[1]*(n-3*group_count),A,b,H,f,
              F(2,11),'INFEASIBLE' if mode=='infeasible' else 'OPTIMAL',
              [kind,'shape_only',mode,'normalization','degree_matching','mean_reserve',
               'confidence_reserve','resource_constraints','finite_bounds'])
    result['shape']={'groups':group_count,'precisions_per_group':3,'degree_rows':3,
                     'B2_representation_group_counts':rep_sizes if kind=='B2' else None,
                     'workspace_preflight':'all synthetic variants use 2 units, cap 3'}
    return result

def fixtures():
    big=F(10**99+123456789,10**99+987654321)
    values=[F(1,3),F(2,7),F(1,2**60),F(1,10**18),big]
    identity=[[int(i==j) for j in range(5)] for i in range(5)]
    cases=[lp('rational_roundtrip',values,[2]*5,H=identity,f=values,offset=-F(2,7),tags=['roundtrip','100_digit'])]
    cases += [lp('unique',[1],[1],[[-1]],[-F(1,3)],offset=F(2,7),tags=['unique','offset']),
              lp('multiple',[0,0],[1,1],H=[[1,1]],f=[1],tags=['multiple']),
              lp('equality_free_multiplier',[-3,2],[1,1],H=[[1,1]],f=[F(1,3)],offset=F(5,7),tags=['equality','negative_cost']),
              lp('upper_active',[-1],[F(2,7)],offset=F(1,3),tags=['active_upper','bound_correction']),
              lp('degenerate',[1,1],[1,1],[[1,1],[2,2]],[1,2],[[1,1],[2,2]],[1,2],tags=['degenerate']),
              lp('inequality_contradiction',[1],[2],[[-1],[1]],[-F(3,5),F(1,2)],expected='INFEASIBLE',tags=['inequality_contradiction']),
              lp('equality_inequality_contradiction',[1],[1],[[1]],[F(2,7)],[[1]],[F(1,3)],expected='INFEASIBLE',tags=['equality_inequality']),
              lp('upper_induced_infeasible',[0],[F(1,4)],[[-1]],[-F(2,7)],expected='INFEASIBLE',tags=['finite_upper_farkas']),
              lp('contradictory_equalities',[0],[1],H=[[1],[1]],f=[F(1,3),F(2,7)],expected='INFEASIBLE',tags=['contradictory_equalities'])]
    for label,gap in [('decimal',F(1,10**18)),('dyadic',F(1,2**60)),('100_digit',F(1,10**99))]:
        bound=big if label=='100_digit' else F(1,3)
        for feasible in [True,False]:
            lower=bound-gap if feasible else bound+gap
            cases.append(lp(f'{label}_{"feasible" if feasible else "infeasible"}',[1],[bound],[[-1]],[-lower],
                            expected='OPTIMAL' if feasible else 'INFEASIBLE',tags=['tiny_gap',label]))
    cases.append(lp('mixed_scales',[10**18,F(1,2**60)],[1,1],
                    [[-2**60,-F(1,10**18)]],[-F(2**60,3)],H=[[1,1]],f=[1],tags=['mixed_scales']))
    cases += [shaped(kind,mode) for kind in ['B2','B3'] for mode in ['inner','outer','infeasible']]
    assert len(cases)==23
    return cases

def wire(problem):
    lines=[f"{len(problem['c'])} {len(problem['A'])} {len(problem['H'])}",problem['c0'],
           ' '.join(problem['c']), ' '.join(problem['U'])]
    for key,rhs in [('A','b'),('H','f')]:
        lines += [' '.join(row+[b]) for row,b in zip(problem[key],problem[rhs])]
    return '\n'.join(lines)+'\n'
