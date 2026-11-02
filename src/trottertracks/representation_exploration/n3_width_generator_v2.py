"""Finite cold certificate acquisition. Science inputs are runner-gated.

CVXPY/SCS and eigensolvers propose only; every returned certificate is checked
by the stdlib independent checker. No ground-state solver or original ledger.
"""
from contextlib import contextmanager
from fractions import Fraction as F
from itertools import combinations,islice
import math
import time
import resource

import numpy as np

from .n3_width_contract_v2 import CAPS,BLOCKS,DELTA_TIERS,SEARCH_STEPS,SOS_LEVEL,active_data,decode_case,enc,digest
from .n3_width_check_v2 import Checks,verify_witness,verify_full_lower,verify_sdp_refutation
from .n3_saved_width_v1 import interval
from . import n3_width_algebra_v2 as car


class Budget(ArithmeticError):pass


class Engine:
    def __init__(self):
        self.counts={'eigh':0,'solver_calls':0,'PSD_probes':0,'eigh_cache_hits':0}
        self.phase_times={};self.cache={};self.solver_events=[];self.a=Checks();self._phase_stack=[]
        self.RSS_at_engine_start=self.current_RSS()

    @staticmethod
    def current_RSS():
        from pathlib import Path
        return next((int(s.split()[1])*1024 for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('VmRSS:')),None)

    @contextmanager
    def phase(self,name):
        item={'wall':time.perf_counter(),'cpu':time.process_time(),'child_wall':0.,'child_cpu':0.};self._phase_stack.append(item)
        try:yield
        finally:
            wall=time.perf_counter()-item['wall'];cpu=time.process_time()-item['cpu'];self._phase_stack.pop()
            old=self.phase_times.setdefault(name,{'wall':0.,'cpu':0.})
            old['wall']+=wall-item['child_wall'];old['cpu']+=cpu-item['child_cpu']
            if self._phase_stack:self._phase_stack[-1]['child_wall']+=wall;self._phase_stack[-1]['child_cpu']+=cpu

    def eig(self,A):
        key=digest(enc(A))
        if key in self.cache:self.counts['eigh_cache_hits']+=1;return self.cache[key]
        if self.counts['eigh']>=CAPS['eigh_per_mode']:raise Budget('EIGH_CALL_CAP')
        self.counts['eigh']+=1
        with self.phase('numerical_proposal'):
            M=np.asarray(A,dtype=float)
            if not np.isfinite(M).all():raise Budget('NONFINITE_NUMERIC_INPUT')
            answer=np.linalg.eigh(M)
            if not all(np.isfinite(x).all() for x in answer):raise Budget('NONFINITE_EIGENSOLVER')
        self.cache[key]=answer;return answer

    def psd(self,A):
        if self.counts['PSD_probes']>=CAPS['PSD_probes_per_mode']:raise Budget('PSD_PROBE_CAP')
        self.counts['PSD_probes']+=1
        try:self.a.psd(A);return True
        except ValueError:return False

    def solve(self,problem):
        import cvxpy as cp
        if self.counts['solver_calls']>=CAPS['solver_calls_per_mode']:raise Budget('SDP_CALL_CAP')
        self.counts['solver_calls']+=1
        with self.phase('numerical_proposal'):
            problem.solve(solver=cp.SCS,warm_start=False,verbose=False,
                          eps_abs=1e-6,eps_rel=1e-6,max_iters=CAPS['solver_iterations_per_call'],
                          normalize=True,acceleration_lookback=10,use_indirect=False)
        stats=problem.solver_stats
        self.solver_events.append({'status':problem.status,'iterations':stats.num_iters,
            'reported_solve_time':stats.solve_time,'reported_setup_time':stats.setup_time,
            'strict_eigensolver_call_count_inside_solver':'not instrumented'})
        return problem.status

    def stats(self):
        return {'calls':dict(self.counts),'phase_times':self.phase_times,'solver_events':self.solver_events,
                'proposal_exact_PSD_operations':self.a.operations,'proposal_max_rational_bits':self.a.max_bits,
                'RSS_at_engine_start':self.RSS_at_engine_start,'RSS_at_engine_end':self.current_RSS(),
                'peak_process_RSS_bytes_linux':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                'memory_qualification':'boundary RSS and process lifetime highwater; not isolated per-method allocation peak'}


def dyadic(x):
    x=float(x)
    if not math.isfinite(x):raise Budget('NONFINITE_RATIONALIZATION')
    return F(round(x*(1<<CAPS['proposal_bits'])),1<<CAPS['proposal_bits'])


def rounded(M):
    M=np.asarray(M,float);n=len(M)
    if M.shape!=(n,n) or not np.isfinite(M).all():raise Budget('NUMERIC_MATRIX_SCHEMA')
    return [[dyadic((M[i,j]+M[j,i])/2) for j in range(n)] for i in range(n)]


def shift(A,r):return [[x+(r if i==j else 0) for j,x in enumerate(row)] for i,row in enumerate(A)]
def gershgorin_repair(A):return max(F(0),max(sum((abs(x) for j,x in enumerate(row) if j!=i),F(0))-row[i] for i,row in enumerate(A)))


def repaired(matrices,e):
    scale=max(F(1),*(abs(x) for A in matrices for row in A for x in row))
    amounts=[F(0)]+[scale*F(1,1<<b) for b in (36,32,28,24,20,16,12,8,4,0)]
    amounts.append(max(gershgorin_repair(A) for A in matrices))
    with e.phase('rationalization_and_repair'):
        for r in amounts:
            answers=[shift(A,r) for A in matrices]
            if all(e.psd(A) for A in answers):return answers,r
    raise Budget('FINITE_PSD_REPAIR_FAILED')


def spectral_certificate(A,nu,e):
    vals,vecs=e.eig(A);m=len(A)
    tau=dyadic(vals[0]-1 if nu==0 else vals[-1]+1 if nu==m else (vals[nu-1]+vals[nu])/2)
    Zn=(vecs*np.maximum(float(tau)-vals,0))@vecs.T
    Z=rounded(Zn);R=shift([[A[i][j]+Z[i][j] for j in range(m)] for i in range(m)],-tau)
    answers,r=repaired([Z,R],e);Z=answers[0];s=nu*tau-sum(Z[i][i] for i in range(m))
    return enc({'tau':tau,'Z':Z,'s':s})


def upper_certificate(case,d):
    h,J,nu,modes,eps=decode_case(case);active=[i for i in range(len(h)) if i!=d]
    pool=[list(x) for x in islice(combinations(active,nu),CAPS['trial_pool'])]
    values=[sum((h[i][i] for i in o),F(0))+sum((J.get((i,j),F(0)) for i,j in combinations(o,2)),F(0)) for o in pool]
    return enc({'occupations':pool,'energies':values,'U':min(values)})


def acquire_bounds(case,d,e):
    active,A,v,J,E,hdd,nu,eps=active_data(case,d)
    with e.phase('bound_acquisition'):
        upper=upper_certificate(case,d);external=spectral_certificate(E,nu-1,e)
    return {'upper':upper,'external':external,'c':enc(hdd+F(*external['s']))}


def witness(case,d,kind,k,params,s,U,c,e):
    with e.phase('width_and_output'):
        width=interval(c-U,k,k*U-s)
        w={'kind':kind,'case_sha256':digest(case),'external':d,'kappa':enc(k),'parameters':params,'s':enc(s),'width':width}
    return w


def transport_witness(case,d,old_bounds,new_bounds,w):
    """Reuse exact parameters at stronger c; recompute only the fixed-witness width."""
    from copy import deepcopy
    from .n3_width_check_v2 import verify_bounds
    verify_witness(case,d,old_bounds,w)
    U,c=verify_bounds(case,d,old_bounds,Checks());new_U,new_c=verify_bounds(case,d,new_bounds,Checks())
    if new_U!=U or new_c<c:raise ValueError('TRANSPORT_REQUIRES_SAME_U_STRONGER_C')
    ans=deepcopy(w);k=F(*w['kappa']);s=F(*w['s']);ans['width']=interval(new_c-U,k,k*U-s)
    verify_witness(case,d,new_bounds,ans)
    return ans


def search_kappas(A,v,nu,U,D,kind,e):
    scores={}
    def objective(x):
        k=D*F(float(x));K=[[k*A[i][j]-v[i]*v[j] for j in range(len(A))] for i in range(len(A))]
        values,_=e.eig(K);score=float(values[0]*nu if kind=='OLD' else sum(values[:nu]))-float(k*U)
        scores[x]=score;return score
    objective(0.);objective(1.)
    phi=(math.sqrt(5)-1)/2;l,r=0.,1.;x1=r-phi*(r-l);x2=l+phi*(r-l);a,b=objective(x1),objective(x2)
    for _ in range(SEARCH_STEPS):
        if a<b:l=x1;x1=x2;a=b;x2=l+phi*(r-l);b=objective(x2)
        else:r=x2;x2=x1;b=a;x1=r-phi*(r-l);a=objective(x1)
    best=max(scores,key=lambda x:(scores[x],-x))
    scaled=F(best)*(1<<CAPS['proposal_bits']);lo=math.floor(scaled);hi=math.ceil(scaled)
    return sorted({F(0),D,D*F(lo,1<<CAPS['proposal_bits']),D*F(hi,1<<CAPS['proposal_bits'])})


def old_certificate(K,nu,e):
    values,_=e.eig(K);mu=dyadic(values[0]);M=shift(K,-mu);answers,r=repaired([M],e)
    mu-=r
    return {'mu':enc(mu)},nu*mu


def five_candidates(A,v,nu,e):
    # The five shifts use the existing c0 recipe, not a widened scientific scan.
    # Avoid calling c0_bound's additional numerical lower-bound refinement.
    m=len(A);lower=min(A[i][i]-sum((abs(A[i][j]) for j in range(m) if i!=j),F(0)) for i in range(m))
    span=max(A[i][i] for i in range(m))-min(A[i][i] for i in range(m));out=[]
    for rho in (F(0),F(1,20),F(1,10),F(3,20),F(1,5)):
        lam=lower+rho*span;M=shift(A,-lam)
        if not e.psd(M):continue
        # Exact Gaussian elimination; reject singular shifts rather than using float inverse.
        aug=[r[:]+[v[i]] for i,r in enumerate(M)]
        try:
            for j in range(m):
                pivot=aug[j][j]
                if pivot==0:raise ZeroDivisionError
                aug[j]=[e.a.q(x/pivot) for x in aug[j]]
                for i in range(m):
                    if i!=j:
                        fac=aug[i][j];aug[i]=[e.a.q(x-fac*y) for x,y in zip(aug[i],aug[j])]
            k=sum((v[i]*aug[i][-1] for i in range(m)),F(0));out.append((k,lam*k))
        except ZeroDivisionError:continue
    return out


def equivalent_problem(A,v,nu,U):
    import cvxpy as cp
    m=len(A);k=cp.Variable();tau=cp.Variable();Z=cp.Variable((m,m),symmetric=True);D=cp.Parameter(nonneg=True)
    M=k*np.array(A,float)-np.outer(np.array(v,float),np.array(v,float))-tau*np.eye(m)+Z
    constraints=[k>=0,k<=D,Z>>0,M>>0]
    problem=cp.Problem(cp.Minimize(k*float(U)-nu*tau+cp.trace(Z)),constraints)
    return problem,D,k,tau,Z,constraints[-1]


def sdp_candidates(case,d,A,v,nu,U,c,eps,e):
    problem,Dvar,kvar,tvar,Zvar,con=equivalent_problem(A,v,nu,U);out=[];refutations=[]
    for tier in DELTA_TIERS:
        delta=eps*tier;D=c-U+delta;Dvar.value=float(D);e.solve(problem)
        if kvar.value is None or tvar.value is None or Zvar.value is None:continue
        k=min(D,max(F(0),dyadic(kvar.value)));tau=dyadic(tvar.value);Z=rounded(Zvar.value)
        R=shift([[k*A[i][j]-v[i]*v[j]+Z[i][j] for j in range(len(A))] for i in range(len(A))],-tau)
        answers,r=repaired([Z,R],e);Z=answers[0];s=nu*tau-sum(Z[i][i] for i in range(len(A)))
        out.append(witness(case,d,'OCC',k,enc({'tau':tau,'Z':Z,'s':s}),s,U,c,e))
        # PSD-LMI dual supplies a primal gamma proposal. No solver value is trusted.
        if con.dual_value is not None:
            gamma=rounded(con.dual_value);m=len(A);correction=(F(nu)-sum(gamma[i][i] for i in range(m)))/m
            gamma=shift(gamma,correction)
            for mixing in (F(0),F(1,1<<32),F(1,1<<24),F(1,1<<16),F(1,1<<8),F(1)):
                G=[[(1-mixing)*gamma[i][j]+(mixing*F(nu,m) if i==j else 0) for j in range(m)] for i in range(m)]
                I_G=[[F(i==j)-G[i][j] for j in range(m)] for i in range(m)]
                if not e.psd(G) or not e.psd(I_G):continue
                t=max(F(0),sum((A[i][j]*G[j][i] for i in range(m) for j in range(m)),F(0))-U)
                objective=sum((v[i]*G[i][j]*v[j] for i in range(m) for j in range(m)),F(0))-D*t
                if objective>delta*D:
                    ref=enc({'gamma':G,'t':t,'objective':objective})
                    verify_sdp_refutation(A,v,nu,U,c,delta,ref)
                    refutations.append({'delta':enc(delta),'certificate':ref})
                break
    return out,refutations


def sos_problem(A,v,J,nu,U):
    import cvxpy as cp
    from scipy.sparse import coo_matrix
    m=len(A);blocks,multipliers,keys=car.structure(m,nu);ix={w:i for i,w in enumerate(keys)}
    def vector(p):return np.array([float(p.get(w,0)) for w in keys])
    hpoly=car.add(car.one_body(A),car.density(J));Vpoly=car.one_body([[v[i]*v[j] for j in range(m)] for i in range(m)])
    Qs={};summed=np.zeros(len(keys));constraints=[]
    for name,block in blocks.items():
        n=block['size'];Q=cp.Variable((n,n),symmetric=True);Qs[name]=Q;rows=[];cols=[];values=[]
        for i,j,p in block['terms']:
            for w,value in p.items():rows.append(ix[w]);cols.append(i*n+j);values.append(float(value))
        C=coo_matrix((values,(rows,cols)),shape=(len(keys),n*n)).tocsc()
        summed=summed+C@cp.reshape(Q,(n*n,),order='C');constraints.append(Q>>0)
    multipliers_matrix=np.stack([vector(p) for p in multipliers],axis=1)
    eta=cp.Variable(len(multipliers));k=cp.Variable();ell=cp.Variable();D=cp.Parameter(nonneg=True)
    constraints += [summed+multipliers_matrix@eta==k*vector(hpoly)-vector(Vpoly)-ell*vector({():1}),k>=0,k<=D]
    return cp.Problem(cp.Minimize(k*float(U)-ell),constraints),D,k,ell,Qs,eta,(blocks,multipliers,hpoly,Vpoly)


def sos_candidates(case,d,A,v,J,nu,U,c,eps,e):
    with e.phase('SOS_coefficient_assembly'):problem,Dvar,kvar,lvar,Qs,etavar,structure=sos_problem(A,v,J,nu,U)
    blocks,multipliers,hpoly,Vpoly=structure;out=[]
    for tier in DELTA_TIERS:
        D=c-U+eps*tier;Dvar.value=float(D);e.solve(problem)
        if kvar.value is None or lvar.value is None or etavar.value is None or any(Q.value is None for Q in Qs.values()):continue
        k=min(D,max(F(0),dyadic(kvar.value)));ell=dyadic(lvar.value);eta=[dyadic(x) for x in etavar.value];grams={}
        for name,Q in Qs.items():grams[name]=repaired([rounded(Q.value)],e)[0][0]
        with e.phase('rationalization_and_repair'):
            summed=car.add(*(car.assemble_gram(blocks[name],Q) for name,Q in grams.items()),
                           *(car.scale(p,x) for p,x in zip(multipliers,eta)))
            target=car.add(car.scale(hpoly,k),car.scale(Vpoly,-1),{():-ell})
            residual=car.add(target,car.scale(summed,-1));bound=sum((abs(x) for x in residual.values()),F(0));s=ell-bound
        params=enc({'level':SOS_LEVEL,'grams':grams,'multipliers':eta,'ell':ell,'residual_bound':bound,'s':s})
        out.append(witness(case,d,'SOS',k,params,s,U,c,e))
    return out


def acquire_mode(case,d,method,provided_bounds=None):
    e=Engine();started=time.perf_counter();cpu=time.process_time();out=[];refs=[];audits=[];bounds=None;failure=None
    active,A,v,J,E,hdd,nu,eps=active_data(case,d)
    try:
        if method=='FULL_H_LOWER':
            h,fullJ,_,_,_=decode_case(case)
            with e.phase('bound_acquisition'):upper=upper_certificate(case,d);spec=spectral_certificate(h,nu,e)
            U=F(*upper['U']);lower=F(*spec['s']);full=enc({'spectral':spec,'L_H':lower,'width':U-lower})
            with e.phase('independent_checker'):audit=verify_full_lower(case,d,upper,full)
            out=[{'kind':'FULL_H_LOWER','case_sha256':digest(case),'external':d,'upper':upper,'full_lower':full,
                  'width':{'low':enc(U-lower),'high':enc(U-lower),'positive_proof_delta':enc(max(U-lower,F(1,1<<128)))}}];audits=[audit]
        else:
            bounds=provided_bounds or acquire_bounds(case,d,e);U=F(*bounds['upper']['U']);c=F(*bounds['c'])
            if c<=U:
                return {'method':method,'external':d,'status':'BOUND_PRECONDITION_FAILED','bounds':bounds,'candidates':[],
                        'refutations':[],'resources':e.stats(),'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,'audits':[]}
            D=c-U+eps
            if method=='NORM':out=[witness(case,d,'NORM',F(0),{},-sum((x*x for x in v),F(0)),U,c,e)]
            elif method=='FIVE':
                for k,mu in five_candidates(A,v,nu,e):
                    # No old box restriction at eps; fixed-witness width includes k-g.
                    K=[[k*A[i][j]-v[i]*v[j] for j in range(len(A))] for i in range(len(A))]
                    if k>=0 and e.psd(shift(K,-mu)):out.append(witness(case,d,'OLD',k,{'mu':enc(mu)},nu*mu,U,c,e))
            elif method in ('OCC','OLD'):
                for k in search_kappas(A,v,nu,U,D,method,e):
                    K=[[k*A[i][j]-v[i]*v[j] for j in range(len(A))] for i in range(len(A))]
                    if method=='OCC':params=spectral_certificate(K,nu,e);s=F(*params['s']);kind='OCC'
                    else:params,s=old_certificate(K,nu,e);kind='OLD'
                    out.append(witness(case,d,kind,k,params,s,U,c,e))
            elif method=='SDP_EQUIVALENT':out,refs=sdp_candidates(case,d,A,v,nu,U,c,eps,e)
            elif method=='DENSITY_SOS':out=sos_candidates(case,d,A,v,J,nu,U,c,eps,e)
            else:raise ValueError('METHOD_NAME')
            with e.phase('independent_checker'):
                audits=[verify_witness(case,d,bounds,w) for w in out]
    except (Budget,ArithmeticError) as ex:failure='TECHNICAL_STOP:'+str(ex)
    except Exception as ex:failure='SOUNDNESS_OR_IMPLEMENTATION_STOP:'+type(ex).__name__+':'+str(ex)
    valid=out[:len(audits)]
    best=min(range(len(valid)),key=lambda i:(F(*valid[i]['width']['high']),i)) if valid else None
    record={'method':method,'external':d,'status':failure or ('CERTIFICATES_FOUND' if valid else 'UNDETERMINED_FINITE_PROPOSALS'),
            'bounds':bounds,'candidates':valid,'best_index':best,'refutations':refs,'audits':audits,'resources':e.stats(),
            'unverified_proposals_retained':out[len(audits):],
            'wall_seconds':time.perf_counter()-started,'cpu_seconds':time.process_time()-cpu,
            'physical_impossibility_claimed':False,'ideal_superiority_to_equivalent_SDP_claimed':False}
    if len(__import__('json').dumps(record))>CAPS['max_method_record_bytes']:raise Budget('METHOD_RECORD_SIZE_CAP')
    return record
