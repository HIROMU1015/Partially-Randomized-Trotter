"""Independent JW/Pade/direct-contraction audit; never imports pilot kernels."""
import math
import numpy as np
from scipy import sparse
from scipy.linalg import expm


class AuditError(ValueError):
    pass


def require(condition,reason):
    if not condition:
        raise AuditError(reason)


def jw_transitions(n,labels):
    """Independent tensor-product JW ladder reference in full Fock space."""
    z=sparse.diags([1.,-1.],format='csr');eye=sparse.eye(2,format='csr')
    lower=sparse.csr_matrix([[0.,1.],[0.,0.]])
    annihilators=[]
    for p in range(n):
        op=sparse.csr_matrix([[1.]])
        for q in range(n):
            op=sparse.kron(op,z if q<p else lower if q==p else eye,format='csr')
        annihilators.append(op)
    return np.asarray([(annihilators[p].conj().T@annihilators[q])[labels,:][:,labels].toarray()
                       for p in range(n) for q in range(n)]).reshape(n,n,len(labels),len(labels))


def bracket(a,b):
    return a@b-b@a


def direct_D(parts):
    result=np.zeros_like(parts[0])
    for i,a in enumerate(parts[:-1]):
        tail=sum(parts[i+1:])
        ab=bracket(a,tail)
        result+=bracket(a,ab)/24+bracket(tail,ab)/12
    return result


def pf_nodes(order,h,length):
    p=1/(4-4**(1/3))
    times=[h] if order==2 else [p*h,p*h,(1-4*p)*h,p*h,p*h]
    nodes=[]
    for t in times:
        for j,time in [(j,t/2) for j in range(length-1)]+[(length-1,t)]+[(j,t/2) for j in range(length-2,-1,-1)]:
            if nodes and nodes[-1][0]==j:
                nodes[-1]=(j,nodes[-1][1]+time)
            else:
                nodes.append((j,time))
    return [(j,t) for j,t in nodes if t!=0]


def direct_response(a,gamma):
    n=len(a);g=np.zeros(n);Q=np.zeros((n,n))
    for p in range(n):
        ep=np.zeros((n,n),complex);ep[p,p]=1
        g[p]=np.trace(gamma@bracket(a,bracket(a,ep))).real/12
        for q in range(n):
            eq=np.zeros((n,n),complex);eq[q,q]=1
            Q[p,q]=np.trace(gamma@bracket(ep,bracket(a,eq))).real/24
    return g,Q


def qr_counter(relative):
    # Separate implementation of the fixed primitive decomposition, not a
    # compile call. Same conservative precision policy; no gate elision.
    a=np.asarray(relative,complex).copy();count=0
    for c in range(len(a)):
        for r in range(len(a)-1,c,-1):
            upper,lower=a[r-1,c],a[r,c]
            if lower==0:continue
            norm=math.sqrt(abs(upper)**2+abs(lower)**2)
            matrix=np.array([[upper.conjugate()/norm,lower.conjugate()/norm],[-lower/norm,upper/norm]])
            a[[r-1,r],:]=matrix@a[[r-1,r],:];count+=1
    return np.array([2*count,2*sum(np.angle(z)!=0 for z in np.diag(a)),0,4*count],dtype=np.int64)


def sector_coefficients(etas,weights,shift,particles):
    linears=[];pairs=[];scalars=[]
    for j,eta in enumerate(etas):
        mean=float(eta.mean());centered=eta-mean
        if j==0:
            linear=centered;pair=np.zeros((len(eta),len(eta)));scalar=mean*particles
        else:
            weight=weights[j-1]
            linear=weight*centered**2+2*weight*mean*particles*centered-(shift if j==1 else 0)
            offset=float(linear.mean());linear-=offset
            pair=2*weight*np.outer(centered,centered)
            scalar=weight*mean*mean*particles*particles+offset*particles
        linears.append(linear);pairs.append(pair);scalars.append(scalar)
    return linears,pairs,scalars


def circuit_vector(frames,linears,pairs,nodes,constant,queries):
    phases=[np.array([0,np.count_nonzero(x),np.count_nonzero(np.triu(y,1)),0]) for x,y in zip(linears,pairs)]
    active=[]
    for j,t in nodes:
        if phases[j][1]==0 and phases[j][2]==0:continue
        if active and active[-1][0]==j:active[-1]=(j,active[-1][1]+t)
        else:active.append((j,t))
    nodes=[(j,t) for j,t in active if t!=0]
    if not nodes:return np.array([0,int(constant!=0),0,0],dtype=np.int64)
    total=np.zeros(4,dtype=np.int64);previous=None;cache={}
    for j,t in nodes:
        u=frames[j][::2,::2]
        prev=previous[0] if previous else -1
        if (prev,j) not in cache:
            rel=u.conj().T if previous is None else np.eye(len(u),dtype=complex) if prev==j else u.conj().T@previous[1]
            cache[prev,j]=qr_counter(rel)
        total+=cache[prev,j]+phases[j];previous=(j,u)
    leave=qr_counter(previous[1]);total+=leave
    first=nodes[0][0]
    removed=qr_counter(frames[first][::2,::2].conj().T)+leave+phases[first]
    total=queries*total-(queries-1)*removed
    total[1]+=int(constant!=0)
    require(np.all(total>=0),'NEGATIVE_NATIVE_VECTOR')
    return total


def audit_arrays(data,raw,seal,contract):
    """Small synthetic fixtures invoke this; CLI requires issued attempt marker."""
    require(raw['scope_id']==contract['scope_id'],'SCOPE')
    require(not seal['E0_accessed_by_generator'] and not seal['finite_step_truth_accessed_by_generator'],'SELECTOR_TRUTH_LEAK')
    require(len(raw['h_grid'])==6,'GRID_LENGTH')
    a=data['one_body'];n=len(a);labels=data['labels'];state=data['state'];dim=len(labels)
    ladders=jw_transitions(n,labels)
    lift=lambda m:np.einsum('pq,pqij->ij',m,ladders,optimize=True)
    independent_gamma=np.einsum('i,pqij,j->pq',state.conj(),ladders,state,optimize=True).T
    require(np.linalg.norm(independent_gamma-data['gamma'],'fro')<1e-10,'RDM_CONVENTION')
    original=[lift(a)]+[float(w)*(lift(g)@lift(g)) for w,g in zip(data['lambdas'],data['g_matrices'])]
    H=float(data['constant'])*np.eye(dim)+sum(original)
    require(np.linalg.norm(H-data['H'],'fro')<1e-10,'H_RECONSTRUCTION')
    ev,V=data['H_eigenvalues'],data['H_eigenvectors']
    require(np.all(np.diff(ev)>=0),'H_SPECTRUM_SORT')
    require(np.linalg.norm(H@V-V*ev,'fro')<1e-9 and np.linalg.norm(V.conj().T@V-np.eye(dim),'fro')<1e-9,'H_SPECTRUM_CERTIFICATE')
    require(abs(float(ev[0])-raw['truth']['E0'])<1e-10,'E0_TRUTH_ONLY')
    names=['original','response_selected','known_linear_extraction'];leading={};norm_bounds=[]
    for name in names:
        parts=data['fragments_'+name];frames=data['orbital_frames_'+name]
        require(len(parts)==len(original),'FRAGMENT_COUNT')
        X=(frames[1]*data['shift_'+name])@frames[1].conj().T
        expected=[original[0]+lift(X),original[1]-lift(X),*original[2:]]
        require(np.linalg.norm(np.asarray(expected)-parts)<1e-9,'REPARTITION_IMPLEMENTATION')
        require(np.linalg.norm(sum(parts)+float(data['constant'])*np.eye(dim)-H,'fro')<1e-10,'H_CHANGED')
        require(abs(np.mean(data['shift_'+name]))<1e-10 and np.max(abs(data['shift_'+name][::2]-data['shift_'+name][1::2]))<1e-10,'GAUGE_OR_SPIN')
        D=direct_D(list(parts));leading[name]=float(np.vdot(state,D@state).real)
        require(abs(leading[name]-seal['leading_coefficients'][name])<1e-9,'LEADING_COEFFICIENT')
        require(np.max(abs(np.asarray(seal['shifts'][name])-data['shift_'+name]))<1e-12,'SHIFT_SEAL')
        for j,F in enumerate(frames):
            require(np.linalg.norm(F.conj().T@F-np.eye(n),'fro')<1e-9,'ORBITAL_FRAME_UNITARITY')
            orbital=a+X if j==0 else data['g_matrices'][j-1]
            diagonal=F.conj().T@orbital@F
            require(np.linalg.norm(diagonal-np.diag(np.diag(diagonal)),'fro')<1e-9,'ACTOR_NOT_DIAGONALIZED')
            eta=np.diag(diagonal).real
            require(np.max(abs(eta-data['actor_eta_'+name][j]))<1e-9,'ACTOR_ETA')
            linear=eta if j==0 else data['lambdas'][j-1]*eta**2-(data['shift_'+name] if j==1 else 0)
            pairs=np.zeros((n,n)) if j==0 else 2*data['lambdas'][j-1]*np.outer(eta,eta)
            require(np.max(abs(linear-data['actor_linear_'+name][j]))<1e-9 and np.max(abs(pairs-data['actor_pairs_'+name][j]))<1e-9,'ACTOR_PHASE_COEFFICIENTS')
        local_bound=abs(float(data['constant']))+float(np.sum(abs(data['actor_linear_'+name][0])))
        for j in range(len(data['g_matrices'])):
            F=frames[j+1];eta=np.diag(F.conj().T@data['g_matrices'][j]@F).real
            local_bound+=abs(float(data['lambdas'][j]))*float(np.sum(abs(eta)))**2
        local_bound+=float(np.sum(abs(data['shift_'+name])))
        norm_bounds.append(local_bound)
    u=data['orbital_frames_original'][1]
    g,Q=direct_response(u.conj().T@a@u,u.conj().T@data['gamma']@u)
    record=seal['selection'];x=data['shift_response_selected']
    require(abs(leading['response_selected']-leading['original']-g@x-x@Q@x)<1e-9,'ONE_RDM_RESPONSE')
    if record['status']=='SEALED_LEADING_ONLY':
        direction=g[::2]+g[1::2];direction-=direction.mean();direction=np.repeat(direction/np.max(abs(direction)),2)
        require(np.max(abs(direction-np.asarray(record['direction'])))<1e-9,'DIRECTION_NOT_RDM_GENERATED')
        require(np.max(abs(x-record['t']*direction))<1e-10,'SELECTOR_X')
        table=record['candidates'];require(1<=len(table)<=6,'SELECTOR_FINITE_CAP')
        linear=float(g@direction);quadratic=float(direction@Q@direction);d0=leading['original'];bound=record['bound']
        complete=[0.,-bound,bound]
        if abs(quadratic)>1e-12:
            complete.append(-linear/(2*quadratic));disc=linear*linear-4*quadratic*d0
            if disc>=0:
                term=-.5*(linear+math.copysign(math.sqrt(disc),linear))
                if term:complete.extend([term/quadratic,d0/term])
                else:complete.append(-linear/(2*quadratic))
        elif abs(linear)>1e-12:complete.append(-d0/linear)
        complete=[t for t in complete if -bound<=t<=bound]
        require(all(any(abs(c['t']-t)<1e-9 for c in table) for t in complete),'SELECTOR_CANDIDATES_OMITTED')
        for candidate in table:
            t=candidate['t'];value=leading['original']+t*(g@direction)+t*t*(direction@Q@direction)
            require(-record['bound']<=t<=record['bound'] and abs(value-candidate['leading'])<1e-9,'SELECTOR_TABLE')
        minimum=min(abs(c['leading']) for c in table)
        tied=[c for c in table if abs(c['leading'])<=minimum+1e-12]
        chosen=min(tied,key=lambda c:(abs(c['t']),c['t']))
        require(chosen['t']==record['t'],'SELECTOR_TIE_OR_OBJECTIVE')
    else:
        require(np.max(abs(x))==0,'ZERO_FALLBACK')
        projected=g[::2]+g[1::2];projected-=projected.mean()
        require(np.max(abs(projected))<=1e-12,'INVALID_ZERO_GRADIENT_FALLBACK')
    known=data['actor_linear_original'][1][::2].copy();known-=known.mean()
    bound=.25*max(1e-12,float(np.max(abs(a))));maximum=float(np.max(abs(known)))
    if maximum:known*=min(1.,bound/maximum)
    require(np.max(abs(np.repeat(known,2)-data['shift_known_linear_extraction']))<1e-9,'KNOWN_REPARTITION')
    require(abs(max(norm_bounds)-raw['Lambda_common'])<1e-8*max(1.,raw['Lambda_common']),'COMMON_COEFFICIENT_NORM_BOUND')
    require(all(abs(raw['h_grid'][s]-1/(8*raw['Lambda_common']*2**s))<1e-15 for s in range(6)),'FIXED_GRID_FORMULA')
    require(max(raw['h_grid'])*raw['Lambda_common']*3<math.pi/2,'PHASE_WRAP_GUARD')
    primitives=[r for r in raw['records'] if r['method'].startswith(('S2:','S4:'))]
    require(len(primitives)==36 and len({r['key'] for r in primitives})==36,'PRIMITIVE_COVERAGE')
    exps={};expm_count=0
    for r in primitives:
        name=r['config'];parts=data['fragments_'+name];h=r['h'];key=r['key']
        nodes=pf_nodes(r['order'],h,len(parts));U=np.eye(dim,dtype=complex)
        for j,t in nodes:
            cachekey=(name,j,float(t).hex())
            if cachekey not in exps:
                exps[cachekey]=expm(-1j*t*parts[j]);expm_count+=1
            U=exps[cachekey]@U
        U*=np.exp(-1j*float(data['constant'])*h)
        require(np.linalg.norm(U-data[key+'_U'],'fro')<1e-9,'INDEPENDENT_PADE_PF')
        vectors=data[key+'_vectors'];values=data[key+'_eigenvalues']
        require(np.linalg.norm(U@vectors-vectors*values,'fro')<1e-9 and np.linalg.norm(vectors.conj().T@vectors-np.eye(dim),'fro')<1e-9,'UNITARY_EIGENPAIR_CERTIFICATE')
        indices=r['cluster_indices'];energies=-np.angle(values)/h
        weights=np.abs(vectors.conj().T@state)**2
        clusters=[]
        for k in np.argsort(energies):
            if not clusters or abs(energies[k]-energies[clusters[-1][-1]])>contract['evaluation']['energy_cluster_absolute_tolerance_Ha']:
                clusters.append([int(k)])
            else:clusters[-1].append(int(k))
        scores=[float(sum(weights[c])) for c in clusters]
        ranked=sorted(range(len(scores)),key=lambda j:(-scores[j],j));winner=ranked[0]
        ambiguous=len(ranked)>1 and scores[winner]-scores[ranked[1]]<=contract['evaluation']['overlap_tie_tolerance']
        require(clusters[winner]==indices and ambiguous==r['ambiguous'],'GROUND_BRANCH_OR_AMBIGUITY')
        require(abs(float(np.mean(energies[indices]))-r['energy'])<1e-10,'PHASE_ENERGY')
        require(abs(float(sum(weights[indices]))-r['overlap'])<1e-10,'PF_GROUND_OVERLAP')
        require(abs(r['energy']-raw['truth']['E0']-r['bias'])<1e-10,'ENERGY_BIAS')
    require(expm_count<=contract['limits']['independent_expm_calls'],'AUDIT_EXPM_CAP')
    records={}
    for r in raw['records']:
        records[r['method'],r['h_index']]=r
        if r['method'].startswith('CS2:'):
            base=records.get(('S2:'+r['config'],r['h_index']))
            require(base is not None and abs(r['energy']-base['energy']+base['h']**2*leading[r['config']])<1e-10,'SCALAR_CORRECTION')
        if r['method'].startswith('EXT2:'):
            coarse=records['S2:'+r['config'],r['h_index']];fine=records['S2:'+r['config'],r['h_index']+1]
            require(abs(r['energy']-(4*fine['energy']-coarse['energy'])/3)<1e-10,'RICHARDSON_VALUES')
            require(len(r['experiments'])==2 and abs(sum(abs(x['weight']) for x in r['experiments'])-5/3)<1e-12,'RICHARDSON_COST_AMPLIFICATION')
    require(len(records)==69,'TOTAL_RECORD_COVERAGE')
    prices={p['id']:p for p in contract['resources']['pricing']}
    for cost in raw['costs']:
        r=records[cost['method'],cost['h_index']];target=cost['target_Ha']
        eligible=(not r['ambiguous'] and abs(r['bias'])+r['numerical_allowance']<target*.5 and r['overlap']-1e-10>=.99)
        require(cost['eligible']==eligible,'ELIGIBILITY_BOUNDARY')
        if not eligible:continue
        total=np.zeros(4,dtype=np.int64);name=r['config'];frames=data['orbital_frames_'+name]
        particle_count=int(int(labels[0]).bit_count())
        linears,pairs,sector_scalars=sector_coefficients(data['actor_eta_'+name],data['lambdas'],data['shift_'+name],particle_count)
        # Production conservatively charges one scalar if any summand is
        # nonzero; this also covers a coincidental cancellation of summands.
        scalar_flag=float(data['constant'])!=0 or any(s!=0 for s in sector_scalars)
        amp=sum(abs(e['weight']) for e in r['experiments']);state_preps=0
        for e in r['experiments']:
            q=max(1,math.ceil(math.pi/(e['h']*(target*.25/amp))))
            p=min(1.,e['overlap']);rep=1 if p>=1 else max(1,math.ceil(math.log(.01/len(r['experiments']))/math.log1p(-p)))
            total+=rep*circuit_vector(frames,linears,pairs,e['nodes'],int(scalar_flag),q)
            state_preps+=rep
        actual=cost['primitive_vector'];ordered=[actual[k] for k in ['controlled_Givens','controlled_phase','controlled_pair_phase','fermionic_parity_links']]
        require(np.array_equal(total,ordered),'CONTROLLED_VECTOR')
        price=prices[cost['price_id']];score=int(total[0])*price['controlled_Givens']+int(total[1])*price['controlled_phase']+int(total[2])*price['controlled_pair_phase']
        require(score==cost['conditional_price_total'] and state_preps==cost['state_preparations'],'PRICE_OR_PREPARATION_COST')
        require(cost['information']['gamma_required']==(r['config']=='response_selected'),'UNNECESSARY_INFORMATION_CHARGE')
    require(len(raw['costs'])==276,'COST_COVERAGE')
    expected_keys={(m,i,t,p['id']) for (m,i) in records for t in contract['evaluation']['energy_targets_Ha'] for p in contract['resources']['pricing']}
    observed_keys={(c['method'],c['h_index'],c['target_Ha'],c['price_id']) for c in raw['costs']}
    require(observed_keys==expected_keys and len(observed_keys)==276,'COST_CONTEXTS_MISSING_OR_DUPLICATED')
    cells=raw['result']['cells'];require(len(cells)==4,'COMPARISON_COVERAGE')
    statuses=[]
    for cell in cells:
        candidates=[c for c in raw['costs'] if c['eligible'] and c['target_Ha']==cell['target_Ha'] and c['price_id']==cell['price_id']]
        R=[c for c in candidates if c['method'] in ('S2:response_selected','CS2:response_selected')]
        B=[c for c in candidates if c not in R];key=lambda c:(c['conditional_price_total'],c['method'],c['h_index'])
        require(cell['selected_response']==(min(R,key=key) if R else None) and cell['strong_control']==(min(B,key=key) if B else None),'STRONG_CONTROL_MINIMUM')
        if R and B:
            ratio=min(R,key=key)['conditional_price_total']/min(B,key=key)['conditional_price_total']
            require(abs(ratio-cell['ratio'])<1e-12,'COST_RATIO')
            expected='CONDITIONAL_COMPONENT_GAIN' if ratio<1-contract['evaluation']['registered_effect_fraction']-1e-10 else 'NO_REGISTERED_COMPONENT_GAIN'
        else:expected='UNDETERMINED_ELIGIBLE_GRID_INSUFFICIENT'
        require(cell['status']==expected,'REGISTERED_EFFECT_CLASSIFICATION');statuses.append(expected)
    classification=('CONDITIONAL_COMPONENT_GAIN_IDEAL_INFORMATION' if all(x=='CONDITIONAL_COMPONENT_GAIN' for x in statuses)
                    else 'UNDETERMINED' if any(x.startswith('UNDETERMINED') for x in statuses) else 'NO_UNIFORM_REGISTERED_COMPONENT_GAIN')
    require(raw['result']['classification']==classification and raw['result']['mandatory_stop'],'OVERALL_OR_STOP')
    return {'status':'INDEPENDENT_JW_PADE_DIRECT_RESPONSE_COST_AUDIT_PASS','PF_recompositions':36,
            'expm_calls':expm_count,'independent_RDM_contractions':1,'independent_leading_evaluations':3,
            'new_eigensolvers':0,'compile_calls':0,'formal_research_GO':False,'mandatory_stop':True}
