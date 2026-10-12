"""Stdlib, exact, independent certificate checker. No NumPy/CVXPY/producer.

SOS residuals are checked with separate Fock-sector ladder actions, not the
producer's CAR coefficient assembly. This finite small-sector verification is
not a claim of polynomial scaling at general filling.
"""
from fractions import Fraction as F
from itertools import combinations,islice

from .n3_width_contract_v2 import CAPS,BLOCKS,SOS_LEVEL,active_data,decode_case,enc,mat,rat,digest
from .n3_occupation_certificate_audit import Audit
from .n3_saved_width_v1 import check_interval


class Checks(Audit):
    def __init__(self):super().__init__(CAPS['rational_bits']);self.operations=0
    def q(self,x):
        self.operations+=1
        if self.operations>CAPS['checker_operations']:raise ArithmeticError('CHECKER_OPERATION_CAP')
        return super().q(x)


def spectral(A,nu,w,a):
    if set(w)!={'tau','Z','s'}:raise ValueError('SPECTRAL_SCHEMA')
    Z=mat(w['Z'],len(A));tau=rat(w['tau']);s=rat(w['s'])
    a.psd(Z);a.psd([[a.q(A[i][j]+Z[i][j]-(tau if i==j else 0)) for j in range(len(A))] for i in range(len(A))])
    if s!=nu*tau-a.total(Z[i][i] for i in range(len(A))):raise ValueError('SPECTRAL_TRACE')
    return s


def trial_upper(case,d,w):
    h,J,nu,modes,eps=decode_case(case);active=[i for i in range(len(h)) if i!=d]
    expected=[list(x) for x in islice(combinations(active,nu),CAPS['trial_pool'])]
    if set(w)!={'occupations','energies','U'} or w['occupations']!=expected:raise ValueError('TRIAL_POOL_IDENTITY')
    energies=[sum((h[i][i] for i in o),F(0))+sum((J.get((i,j),F(0)) for i,j in combinations(o,2)),F(0)) for o in expected]
    if [rat(x) for x in w['energies']]!=energies or rat(w['U'])!=min(energies):raise ValueError('TRIAL_ENERGY')
    return min(energies)


def verify_bounds(case,d,w,a):
    if set(w)!={'upper','external','c'}:raise ValueError('BOUND_SCHEMA')
    U=trial_upper(case,d,w['upper']);active,A,v,J,E,hdd,nu,eps=active_data(case,d)
    lower=spectral(E,nu-1,w['external'],a)
    c=hdd+lower
    if rat(w['c'])!=c:raise ValueError('EXTERNAL_LOWER_BINDING')
    return U,c


def action(word,bits):
    sign=1
    for creation,i in reversed(word):
        occupied=bool(bits&(1<<i))
        if occupied==creation:return None
        sign*=(-1)**((bits&((1<<i)-1)).bit_count())
        bits=bits|(1<<i) if creation else bits&~(1<<i)
    return bits,sign


def sector_states(m,nu):return [sum(1<<i for i in o) for o in combinations(range(m),nu)]
def ladder_generators(m):
    pairs=list(combinations(range(m),2))
    return {'particle':[((False,i),) for i in range(m)],'hole':[((True,i),) for i in range(m)],
            'D':[((False,j),(False,i)) for i,j in pairs],
            'Q':[((True,i),(True,j)) for i,j in pairs],
            'G':[((True,i),(False,j)) for i in range(m) for j in range(m)]}


def gram_sector(ops,Q,states,a):
    groups={};n=len(states)
    for j,bits in enumerate(states):
        for i,word in enumerate(ops):
            ans=action(word,bits)
            if ans is not None:
                z,sign=ans;groups.setdefault(z,[]).append((j,i,sign))
    out=[[F(0) for _ in states] for _ in states]
    for entries in groups.values():
        for p,i,s in entries:
            for q,j,t in entries:out[p][q]=a.q(out[p][q]+s*t*Q[i][j])
    return out


def one_body_sector(A,states,a):
    ix={b:i for i,b in enumerate(states)};out=[[F(0) for _ in states] for _ in states]
    for q,b in enumerate(states):
        for i in range(len(A)):
            for j in range(len(A)):
                ans=action(((True,i),(False,j)),b)
                if ans is not None:z,sgn=ans;out[ix[z]][q]=a.q(out[ix[z]][q]+sgn*A[i][j])
    return out


def sos_lower(A,v,J,nu,k,w,a):
    if set(w)!={'level','grams','multipliers','ell','residual_bound','s'} or w['level']!=SOS_LEVEL or set(w['grams'])!=set(BLOCKS):
        raise ValueError('SOS_LEVEL_SCHEMA')
    m=len(A);states=sector_states(m,nu);ops=ladder_generators(m)
    if len(w['multipliers'])!=1+m*(m+1)//2:raise ValueError('NUMBER_MULTIPLIER_SCHEMA')
    for x in w['multipliers']:rat(x)  # terms vanish in this fixed number sector
    summed=[[F(0) for _ in states] for _ in states]
    for name in BLOCKS:
        Q=mat(w['grams'][name],len(ops[name]));a.psd(Q)
        S=gram_sector(ops[name],Q,states,a)
        summed=[[a.q(x+S[i][j]) for j,x in enumerate(row)] for i,row in enumerate(summed)]
    K=[[a.q(k*A[i][j]-v[i]*v[j]) for j in range(m)] for i in range(m)]
    target=one_body_sector(K,states,a)
    for i,b in enumerate(states):
        target[i][i]=a.q(target[i][i]+k*sum((x for (p,q),x in J.items() if b&(1<<p) and b&(1<<q)),F(0)))
    ell,e,s=rat(w['ell']),rat(w['residual_bound']),rat(w['s'])
    if e<0 or s!=ell-e:raise ValueError('SOS_SAFE_LOWER')
    residual=[[a.q(target[i][j]-summed[i][j]-(ell if i==j else 0)) for j in range(len(states))] for i in range(len(states))]
    a.require(all(residual[i][j]==residual[j][i] for i in range(len(states)) for j in range(len(states))),'Hermitian residual')
    rowbound=max(a.total(abs(x) for x in row) for row in residual)
    if rowbound>e:raise ValueError('SOS_RESIDUAL_BOUND')
    return s


def verify_witness(case,d,bounds,w):
    a=Checks();active,A,v,J,E,hdd,nu,eps=active_data(case,d);U,c=verify_bounds(case,d,bounds,a)
    if set(w)!={'kind','case_sha256','external','kappa','parameters','s','width'} or w['case_sha256']!=digest(case) or w['external']!=d:
        raise ValueError('CERTIFICATE_BINDING')
    if c<=U:raise ValueError('SCHUR_PRECONDITION')
    k=rat(w['kappa'])
    if k<0:raise ValueError('NEGATIVE_KAPPA')
    K=[[a.q(k*A[i][j]-v[i]*v[j]) for j in range(len(A))] for i in range(len(A))]
    if w['kind']=='NORM':
        if k!=0 or w['parameters']!={}:raise ValueError('NORM_SCHEMA')
        s=-a.total(a.q(x*x) for x in v)
    elif w['kind']=='OLD':
        if set(w['parameters'])!={'mu'}:raise ValueError('OLD_SCHEMA')
        mu=rat(w['parameters']['mu']);a.psd([[a.q(x-(mu if i==j else 0)) for j,x in enumerate(row)] for i,row in enumerate(K)]);s=nu*mu
    elif w['kind']=='OCC':s=spectral(K,nu,w['parameters'],a)
    elif w['kind']=='SOS':s=sos_lower(A,v,J,nu,k,w['parameters'],a)
    else:raise ValueError('CERTIFICATE_KIND')
    if rat(w['s'])!=s:raise ValueError('LOWER_IDENTITY')
    check_interval(c-U,k,k*U-s,w['width'])
    return {'status':'EXACT_CERTIFICATE_PASS','PSD_checks':a.psd_checks,'checks':a.checks,'operations':a.operations,'max_rational_bits':a.max_bits}


def verify_sdp_refutation(A,v,nu,U,c,delta,w):
    a=Checks();m=len(A)
    if set(w)!={'gamma','t','objective'} or c<=U or delta<=0:raise ValueError('REFUTATION_SCHEMA')
    gamma=mat(w['gamma'],m);t=rat(w['t']);a.psd(gamma);a.psd([[F(i==j)-gamma[i][j] for j in range(m)] for i in range(m)])
    if sum(gamma[i][i] for i in range(m))!=nu:raise ValueError('FANTOPE_TRACE')
    energy=sum((A[i][j]*gamma[j][i] for i in range(m) for j in range(m)),F(0))
    if t<max(F(0),energy-U):raise ValueError('EPIGRAPH_T')
    objective=sum((v[i]*gamma[i][j]*v[j] for i in range(m) for j in range(m)),F(0))-(c-U+delta)*t
    if rat(w['objective'])!=objective or objective<=delta*(c-U+delta):raise ValueError('NONSTRICT_OCC_REFUTATION')
    return {'status':'IDEAL_OCC_CONDITION_INFEASIBLE_AT_TARGET','physical_impossibility_claimed':False}


def verify_full_lower(case,d,upper,w):
    a=Checks();h,J,nu,modes,eps=decode_case(case);U=trial_upper(case,d,upper)
    if set(w)!={'spectral','L_H','width'}:raise ValueError('FULL_LOWER_SCHEMA')
    L=spectral(h,nu,w['spectral'],a)
    if rat(w['L_H'])!=L or L>U or rat(w['width'])!=U-L:raise ValueError('FULL_LOWER_INTERVAL')
    return {'status':'FULL_H_ENERGY_BRACKET_PASS','PSD_checks':a.psd_checks,'operations':a.operations,'max_rational_bits':a.max_bits}


def expected_endpoint(case,rows):
    """Independent selector reconstruction after certificate validation."""
    eps=rat(case['epsilon_proj']);summary=[]
    if [r['external'] for r in rows]!=case['candidate_modes']:raise ValueError('MODE_COVERAGE_ORDER')
    for row in rows:
        candidates=row['candidates'];best=min(range(len(candidates)),key=lambda i:(rat(candidates[i]['width']['high']),i)) if candidates else None
        width=candidates[best]['width'] if best is not None else None
        summary.append({'external':row['external'],'status':row['status'],'best_index':best,'width':width,
                        'accepted':width is not None and rat(width['positive_proof_delta'])<=eps})
    accepted=[r for r in summary if r['accepted']]
    selected=min(accepted,key=lambda x:(rat(x['width']['high']),x['external'])) if accepted else None
    reduced=None
    if selected:
        active,A,v,J,E,hdd,nu,eps=active_data(case,selected['external'])
        reduced=enc({'active_to_full':active,'nu':nu,'h':A,'density':[[i,j,x] for (i,j),x in sorted(J.items()) if x]})
    return {'same_mode_endpoints':summary,'H_any_mode_reduction':bool(accepted),
            'selected_external':selected['external'] if selected else None,'modes_removed':1 if selected else 0,
            'reduced_H':reduced,'fallback':None if selected else 'ORIGINAL_FULL_H',
            'actual_projection_error_or_quantum_cost_claimed':False}


def verify_endpoints(case,methods,claimed):
    from .n3_width_contract_v2 import METHODS
    if set(methods)!=set(METHODS):raise ValueError('METHOD_COVERAGE')
    expected={name:expected_endpoint(case,path['mode_records']) for name,path in methods.items()}
    cheap=[];operational=[];origins=[]
    for d in case['candidate_modes']:
        pool=[]
        for name in ('NORM','FIVE','OLD','FULL_H_LOWER'):
            row=next(r for r in methods[name]['mode_records'] if r['external']==d);pool.extend((name,w) for w in row['candidates'])
        cheap.append({'external':d,'status':'OPERATIONAL_CHEAP_OR','candidates':[w for _,w in pool]})
        own=next(r for r in methods['OCC']['mode_records'] if r['external']==d)['candidates'];all_pool=pool+[('OCC_RAW',w) for w in own]
        best=min(all_pool,key=lambda x:(rat(x[1]['width']['high']),x[0])) if all_pool else None
        operational.append({'external':d,'status':'OPERATIONAL_SAFE_OR','candidates':[w for _,w in all_pool]})
        origins.append({'external':d,'origin':best[0] if best else None})
    expected['CHEAP_OR']=expected_endpoint(case,cheap);expected['OCC_OPERATIONAL_OR']=expected_endpoint(case,operational)
    expected['OCC_OPERATIONAL_OR'].update(origins=origins,fallback_is_OCC_specific_gain=False,
        cost_qualification='Coverage OR from saved valid certificates; no separate operational execution time measured')
    for name in ('SDP_EQUIVALENT','DENSITY_SOS'):
        rows=[];other_origins=[]
        for d in case['candidate_modes']:
            pool=[]
            for cheap_name in ('NORM','FIVE','OLD','FULL_H_LOWER'):
                row=next(r for r in methods[cheap_name]['mode_records'] if r['external']==d);pool.extend((cheap_name,w) for w in row['candidates'])
            own=next(r for r in methods[name]['mode_records'] if r['external']==d)['candidates'];pool.extend((name+'_RAW',w) for w in own)
            best=min(pool,key=lambda x:(rat(x[1]['width']['high']),x[0])) if pool else None
            rows.append({'external':d,'status':'OPERATIONAL_SAFE_OR','candidates':[w for _,w in pool]});other_origins.append({'external':d,'origin':best[0] if best else None})
        expected[name+'_OPERATIONAL_OR']=expected_endpoint(case,rows)
        expected[name+'_OPERATIONAL_OR'].update(origins=other_origins,fallback_is_OCC_specific_gain=False,
            cost_qualification='Coverage OR from saved valid certificates; no separate operational execution time measured')
    if claimed!=expected:raise ValueError('SELECTOR_OR_AGGREGATE_ENDPOINT_MISMATCH')
    return True
