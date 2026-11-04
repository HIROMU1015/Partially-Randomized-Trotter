"""Phase-compensated Pauli-frame telescope across the existing complete plan.

Local corrections are retained as operators and propagated through shared CX,
rather than independently charged as large mismatches. A floating selector only
chooses a finite Pauli;the chosen residual is always enclosed with Arb192. No
continuous phase fit, Clifford-scalar assumption or state probe is used.
"""
from collections import Counter
from fractions import Fraction as F
from . import c_cx_structural_blocks as blocks
from . import c_compiled_raw_local as local
from . import c_pr_aware_df_fixed_certificate as fixed
from . import c_pr_aware_df_zsxx_enclosure as z
from .c_per_word_population_range import _digest

VERSION='retained_Pauli_frame_CX_telescope_Arb192_v1'


def _body(f):return dict(kind=f['kind'],support=list(range(len(f['support']))),left=f['left'],right=f['right'])


def _pauli(n,x,zz):
    from flint import acb,acb_mat
    d=1<<n
    if not 0<=x<d or not 0<=zz<d:raise ValueError('PAULI_MASK')
    P=acb_mat(d,d);phase=(acb(0,1))**((x&zz).bit_count()%4)
    for j in range(d):P[j^x,j]=phase*(-1 if (j&zz).bit_count()%2 else 1)
    return P


def _choose(a,b,P,n):
    """Finite Pauli selection only;not a numerical bound or fitted phase."""
    import numpy as np
    def midpoint(M):return np.array([[complex(float(M[i,j].real.mid()),float(M[i,j].imag.mid()))
        for j in range(M.ncols())] for i in range(M.nrows())])
    M=midpoint(a)@midpoint(P)@midpoint(b).conj().T
    if not np.isfinite(M).all():raise ValueError('NONFINITE_PAULI_SELECTOR')
    options=[];d=1<<n
    for zz in range(d):
        for x in range(d):
            Q=midpoint(_pauli(n,x,zz));options.append((float(abs(np.vdot(Q,M))),x,zz))
    # Lexicographic tie rule. Correctness does not require the selector to be optimal.
    return max(options,key=lambda v:(v[0],-v[1],-v[2]))[1:]


def _job(job):
    from flint import ctx,arb
    f=job['body'];n=len(f['support']);incoming=job['incoming'];px,pz=incoming['x'],incoming['z']
    if not 1<=n<=4:raise ValueError('REGISTERED_LOCAL_SUPPORT_ONLY')
    with z._BALL_LOCK,ctx.workprec(192):
        cache={};a,ea,pa,ra=blocks._matrix(f['left'],n,cache);b,eb,pb,rb=blocks._matrix(f['right'],n,cache)
        P=_pauli(n,px,pz);qx,qz=_choose(a,b,P,n);Q=_pauli(n,qx,qz)
        pin=F(1,2) if px or pz else F();pout=F(1,2) if qx or qz else F()
        a,delta=local._midpoint_matrix(z._scale(a*P,local._phase(-pin)));ea+=delta
        b,delta=local._midpoint_matrix(z._scale(Q*b,local._phase(-pout)));eb+=delta
        options=[];size=1<<n
        for j in range(size):
            gamma=F(2*j,size);M=z._scale(a,local._phase(gamma))-b
            if n==1:
                norm=F(z._norm(M)['upper_exact'])+ea+eb
                upper=z._fraction((z._real(norm)/arb(2).sqrt()).upper())
            else:upper=blocks._operator_upper(M)+ea+eb
            options.append((upper,j,gamma))
        upper,j,gamma=min(options)
        return dict(upper_exact=str(min(F(2),upper)),output=dict(x=qx,z=qz),incoming=incoming,
            delta_pi_exact=str(pa+pin-pb-pout-gamma),delta_rad_exact=str(ra-rb),
            root_index=j,root_order=size,support_dimension=size,precision_bits=192,
            rounding_enclosed=True,phase_fit=False,selector_is_not_a_bound=True,
            theorem='||Cj Pin-exp(i beta) Pout Rj||2<=epsilon;finite corrections retained')


def _cx(x,zz,c,t):
    """Signed Hermitian Pauli conjugation. Same Boolean rule as Clifford tableaux."""
    xc=(x>>c)&1;xt=(x>>t)&1;zc=(zz>>c)&1;zt=(zz>>t)&1
    sign=xc&zt&(xt^zc^1)
    return x^(xc<<t),zz^(zt<<c),sign


def _local_masks(x,zz,support):
    return sum(((x>>q)&1)<<i for i,q in enumerate(support)),sum(((zz>>q)&1)<<i for i,q in enumerate(support))


def evaluate(left,right,*,saved=None,plan=None,parallel=True):
    from flint import ctx
    z.runtime_model();actual_saved,actual=blocks.make_plan(left,right,saved=saved)
    if plan is not None and plan!=actual:raise ValueError('COMPLETE_SAVED_FACTOR_PLAN_IDENTITY')
    saved,plan=actual_saved,actual
    if not plan['applicable']:return dict(status='NOT_APPLICABLE_BLOCK_SUPPORT',upper_exact=None,
        full_operator_coverage=False,dimension=1<<left['qubits'],dense_fallback_used=False)
    definitions={};dc=Counter()
    for e in saved['expansions']:
        for key,value in e['definitions'].items():
            if key in definitions and {k:v for k,v in value.items() if k!='multiplicity'}!={k:v for k,v in definitions[key].items() if k!='multiplicity'}:
                raise ValueError('PRIMITIVE_DEFINITION_CONFLICT')
            definitions[key]=value;dc[key]+=value['multiplicity']
    dw,dr=local._map(local._definition_job,definitions.values(),parallel);dw=dict(zip(definitions,dw))
    bodies={}
    for f in plan['factors']:
        if f['kind']!='shared_CX':bodies[_digest(_body(f))]=_body(f)
    jobs={}
    # Four possible incoming one-wire Paulis are independent and cheap to cache.
    for key,body in bodies.items():
        if body['kind']!='oneq':continue
        for x in range(2):
            for zz in range(2):
                job=dict(body=body,incoming=dict(x=x,z=zz));jobs[_digest(job)]=job
    cache,resources=local._map(_job,jobs.values(),parallel);cache=dict(zip(jobs,cache))
    x=zz=0;pi=F();rad=F(saved['expansions'][0]['phase_exact_rad'])-F(saved['expansions'][1]['phase_exact_rad'])
    counts=Counter();cx_signs=0;trace=[]
    for index,f in enumerate(plan['factors']):
        if f['kind']=='shared_CX':
            x,zz,s=_cx(x,zz,*f['gate']['qubits']);pi+=s;cx_signs+=s;continue
        body=_body(f);px,pz=_local_masks(x,zz,f['support']);job=dict(body=body,incoming=dict(x=px,z=pz));key=_digest(job)
        if key not in cache:cache[key]=_job(job)
        w=cache[key];counts[key]+=1;pi+=F(w['delta_pi_exact']);rad+=F(w['delta_rad_exact'])
        for i,q in enumerate(f['support']):
            x=(x&~(1<<q))|(((w['output']['x']>>i)&1)<<q)
            zz=(zz&~(1<<q))|(((w['output']['z']>>i)&1)<<q)
        trace.append(dict(factor_index=index,witness_id=key,output_global=dict(x=x,z=zz)))
    with z._BALL_LOCK,ctx.workprec(192):
        phase=local._phase(pi,rad)
        end=z._fraction((phase-1).abs_upper())
        if x or zz:end=max(end,z._fraction((-phase-1).abs_upper()))
        end=min(F(2),end)
    translation=sum((F(dw[k]['upper_exact'])*dc[k] for k in dw),F())
    residual=sum((F(cache[k]['upper_exact'])*counts[k] for k in counts),F())
    resources['method']='independent oneq bodies x all four incoming Pauli frames;internal blocks follow exact dependency order'
    return dict(status='VALIDATED_PAULI_FRAME_NATIVE_PAIR',version=VERSION,kind='rigorous_enclosure',
        norm='full_operator_2',dimension=1<<left['qubits'],full_operator_coverage=True,precision_bits=192,
        rounding_enclosed=True,phase_fit=False,sector_projection=False,columns_evaluated=0,
        left_id=_digest(left),right_id=_digest(right),plan_id=plan['plan_id'],upper_exact=str(min(F(2),translation+residual+end)),
        lower_exact='0',translation_upper_exact=str(translation),residual_upper_exact=str(residual),endpoint_upper_exact=str(end),
        final_frame=dict(x=x,z=zz),global_pi_exact=str(pi),global_rad_exact=str(rad),cx_negative_conjugations=cx_signs,
        definition_witnesses=dw,definition_multiplicities=dict(dc),definition_resources=dr,
        frame_witnesses=cache,used_frame_multiplicities=dict(counts),frame_trace=trace,
        used_unique_frame_jobs=len(counts),precomputed_oneq_jobs=len(jobs),resources=resources,
        internal_blocks=sum(f['kind']=='block' for f in plan['factors']),unique_bodies=len(bodies),
        maximum_support=max(len(f['support']) for f in plan['factors'] if f['kind']!='shared_CX'),
        discrete_operator_corrections_retained=True,nonunitary_literal_target_supported=False,dense_fallback_used=False,
        all_word_uniform=False,calibration_ready=False,Stage_C_ready=False,actual_CA_achieved=None)


def compare(left,right,**kwargs):return evaluate(left,right,**kwargs)


def check_binding(cert,left,right):
    local.validate(left);local.validate(right)
    if (cert.get('status')!='VALIDATED_PAULI_FRAME_NATIVE_PAIR' or cert.get('version')!=VERSION
            or cert.get('left_id')!=_digest(left) or cert.get('right_id')!=_digest(right)
            or left['qubits']!=right['qubits'] or cert.get('dimension')!=1<<left['qubits']
            or cert.get('precision_bits')!=192 or cert.get('rounding_enclosed') is not True
            or cert.get('full_operator_coverage') is not True or cert.get('discrete_operator_corrections_retained') is not True
            or cert.get('columns_evaluated')!=0 or any(cert.get(k) is not False for k in ('phase_fit','sector_projection',
                'nonunitary_literal_target_supported','dense_fallback_used','all_word_uniform','calibration_ready','Stage_C_ready'))
            or cert.get('actual_CA_achieved') is not None):raise ValueError('PAULI_FRAME_TARGET_SCOPE')
    terms=[F(cert[k]) for k in ('translation_upper_exact','residual_upper_exact','endpoint_upper_exact')]
    if min(terms)<0 or F(cert['upper_exact'])!=min(F(2),sum(terms,F())) or F(cert['lower_exact'])!=0:
        raise ValueError('FRAME_TELESCOPE_COMPOSITION')
    return True
