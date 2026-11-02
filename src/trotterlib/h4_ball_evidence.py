"""Validated full-space C numerical evidence; no chemistry, sampling or accounting.

All functions are inert on import. Production inputs come only from a hash-bound
legacy preparation export after the separate controller consumes a fresh grant.
Certificates here are candidates pending scientific acceptance, never N/G gates.
"""
from fractions import Fraction as Q
import hashlib
import json
import math
import struct
from flint import arb, acb, acb_mat, fmpq, ctx


def require(ok, reason):
    if not ok:raise ValueError(reason)


def rational(x):
    if isinstance(x,Q):return x
    if isinstance(x,float):
        require(math.isfinite(x),'NONFINITE_BINARY64');return Q.from_float(x)
    if type(x) is int:return Q(x)
    if isinstance(x,str):return Q(x)
    raise TypeError('EXACT_INPUT_REQUIRED')


def real(x):
    q=rational(x);return arb(fmpq(q.numerator,q.denominator))


def complex_ball(x):
    if isinstance(x,(list,tuple)):
        require(len(x)==2,'COMPLEX_PAIR');return acb(real(x[0]),real(x[1]))
    return acb(real(x))


def dyadic(x):
    require(x.is_finite() and x.is_exact(),'EXACT_FINITE_ENDPOINT_REQUIRED')
    m,e=x.man_exp();return Q(int(m))*Q(2)**int(e)


def endpoints(x):
    require(x.is_finite(),'NONFINITE_ENCLOSURE')
    return dict(lower=str(dyadic(x.lower())),upper=str(dyadic(x.upper())))


def serialize(z):
    return dict(real=endpoints(z.real),imag=endpoints(z.imag))


def identity(n):return acb_mat([[int(i==j) for j in range(n)] for i in range(n)])


def matrix(values):
    require(values and all(len(row)==len(values) for row in values),'SQUARE_MATRIX_REQUIRED')
    return acb_mat([[complex_ball(x) for x in row] for row in values])


def dagger(a):return a.conjugate().transpose()


def frobenius(a):
    # A Frobenius upper bound is also a spectral norm upper bound.
    return sum((abs(a[i,j]).upper()**2 for i in range(a.nrows()) for j in range(a.ncols())),arb(0)).upper().sqrt().upper()


def contraction_norm(a):
    # No unitary assumption: ||A||_2 <= sqrt(1+||A* A-I||_2).
    return (1+frobenius(dagger(a)*a-identity(a.ncols()))).sqrt().upper()


def phase(angle):return acb(0,angle).exp()


def pauli(label):
    n=len(label);dim=1<<n;out=acb_mat(dim,dim)
    require(n>0 and not set(label)-set('IXYZ'),'PAULI_LABEL')
    for col in range(dim):
        row=col;v=acb(1)
        for q,s in enumerate(reversed(label)):
            bit=(col>>q)&1
            if s in 'XY':row^=1<<q
            if s=='Y':v*=acb(0,1 if bit==0 else -1)
            elif s=='Z' and bit:v=-v
        out[row,col]=v
    return out


def one_body(values):
    """Exact Qiskit little-endian a†_p a_q action on all Fock states."""
    n=len(values);out=acb_mat(1<<n,1<<n)
    require(all(len(row)==n for row in values),'ONE_BODY_SHAPE')
    for p in range(n):
        for q in range(n):
            c=complex_ball(values[p][q])
            if c.is_zero():continue
            for ket in range(1<<n):
                if not (ket>>q)&1:continue
                intermediate=ket^(1<<q)
                if (intermediate>>p)&1:continue
                sign=(-1)**((ket&((1<<q)-1)).bit_count()+(intermediate&((1<<p)-1)).bit_count())
                out[intermediate|(1<<p),ket]+=c*sign
    return out


def embed(local,qubits,n):
    require(len(set(qubits))==len(qubits) and all(type(q) is int and 0<=q<n for q in qubits),'GATE_SUPPORT')
    require(local.nrows()==local.ncols()==1<<len(qubits),'LOCAL_GATE_SHAPE')
    dim=1<<n;out=acb_mat(dim,dim);mask=sum(1<<q for q in qubits)
    for col in range(dim):
        sub=sum(((col>>q)&1)<<k for k,q in enumerate(qubits))
        for row_sub in range(1<<len(qubits)):
            row=(col&~mask)|sum(((row_sub>>k)&1)<<q for k,q in enumerate(qubits))
            out[row,col]=local[row_sub,sub]
    return out


def array_hash(values, *, vector=False):
    rows=values if not vector else [values];shape=[len(values)] if vector else [len(values),len(values[0])]
    data=b''.join(struct.pack('<dd',float(v[0]),float(v[1])) for row in rows for v in row)
    payload=dict(shape=shape,dtype='<c16',bytes=hashlib.sha256(data).hexdigest())
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def basis_operators(spec,n):
    described=[];forward=identity(1<<n);inverse=identity(1<<n)
    for op in spec['operations']:
        raw=op['matrix'];invraw=op['inverse_matrix']
        # Match the exact legacy local-matrix fingerprint; not just a display ID.
        data=b''.join(struct.pack('<dd',float(v[0]),float(v[1])) for row in raw for v in row)
        require(hashlib.sha256(data).hexdigest()==op['description']['matrix_sha256'],'BASIS_MATRIX_HASH')
        require(op['description']['qubits']==op['qubits'],'BASIS_SUPPORT_HASH')
        described.append(op['description'])
        A=embed(matrix(raw),op['qubits'],n);Ai=embed(matrix(invraw),op['qubits'],n)
        forward=A*forward;inverse=inverse*Ai
    expected=canonical(dict(num_system_qubits=n,operations=described,hash_policy='ordered_local_basis_operations_v1'))
    require(spec['basis_hash']==expected,'BASIS_HASH')
    return forward,inverse


def primitive_diagonal(spec,n):
    out=acb_mat(1<<n,1<<n)
    for bit in range(1<<n):
        angle=real(spec['global_phase'])
        for q,a in spec['rz']:angle-=real(a)*(1 if not(bit>>q)&1 else -1)/2
        for p,q,a in spec['rzz']:angle-=real(a)*(1 if ((bit>>p)^(bit>>q))&1==0 else -1)/2
        out[bit,bit]=phase(angle)
    return out


def cdf_binary64(probabilities):
    """NumPy 1.26.4 sequential float64 cumsum, terminal division, no RNG."""
    require(probabilities and all(type(x) is float and math.isfinite(x) and x>=0 for x in probabilities),'PROBABILITY_INPUT')
    total=0.;cdf=[]
    for p in probabilities:total=float(total+p);cdf.append(total)
    require(total>0 and math.isfinite(total),'PROBABILITY_TOTAL')
    return [float(x/total) for x in cdf]


def grid_masses(cdf, bits=53):
    require(cdf and cdf[-1]==1. and all(0<=x<=1 for x in cdf),'CDF_ENDPOINTS')
    require(all(a<=b for a,b in zip(cdf,cdf[1:])),'CDF_ORDER')
    size=1<<bits;previous=0;out=[]
    for endpoint in cdf:
        x=rational(endpoint)*size;upper=-(-x.numerator//x.denominator)
        out.append(Q(upper-previous,size));previous=upper
    require(sum(out)==1,'CDF_GRID_MASS')
    return out


def verify_probability_mapping(law):
    require(law['numpy_version']=='1.26.4' and law['uniform_bits']==53,'RUNTIME_PROBABILITY_MAPPING')
    for kind in ('component','order'):
        require(cdf_binary64(law[kind+'_probabilities'])==law[kind+'_cdf'],'CDF_ROUNDING_GRAPH')
    return grid_masses(law['component_cdf']),grid_masses(law['order_cdf'])


def finite_normalization(lam,delta,r,q):
    tau=lam*delta/r
    weights=[abs(tau)**k/math.factorial(k)*(1+(tau/(k+1))**2).sqrt() for k in (0,2,4)]
    BK=sum(weights,arb(0));return dict(tau=tau,weights=weights,B_K=BK,B=BK**(r*q),log_B=r*q*BK.log())


def polynomial5(A):
    I=identity(A.nrows());term=I;total=I
    for k in range(1,6):term=A*term/k;total+=term
    return total


def finite_implemented_mean(components,rotations,probabilities,order_probabilities):
    dim=components[0].nrows();M=acb_mat(dim,dim)
    for p,P in zip(probabilities,components,strict=True):M+=P*real(p)
    F=acb_mat(dim,dim)
    for k,rs,pi in zip((0,2,4),rotations,order_probabilities,strict=True):
        R=acb_mat(dim,dim)
        for p,V in zip(probabilities,rs,strict=True):R+=V*real(p)
        F+=R*(M**k)*real(pi)*((-1)**(k//2))
    return F


def evolve(state,blocks,tail,scalar,q,r):
    out=state
    for _ in range(q):
        for E in blocks:out=E*out
        if tail is not None:
            for _ in range(r):out=tail*out
        for E in reversed(blocks):out=E*out
        out=out*scalar
    return out


def signal(state,out):
    """Contract two column vectors; an unapplied operator is not a signal."""
    require(state.ncols()==out.ncols()==1 and state.nrows()==out.nrows()
            and state.nrows()>0,'SIGNAL_STATE_VECTOR_SHAPE')
    return (dagger(state)*out)[0,0]


def endpoint_compare(error,target):
    """Exact endpoint comparisons, never midpoint-based acceptance."""
    require(error.is_finite(),'NONFINITE_NUMERICAL_ERROR')
    target=rational(target)
    if dyadic(error.upper())<=target:return 'TARGET_MET'
    if dyadic(error.lower())<=target:return 'WIDTH_INSUFFICIENT'
    return 'TARGET_NOT_MET'


class EnclosureEngine:
    def __init__(self,model,budget):
        self.model=model;self.budget=budget;self.cache={}
        require(model['schema']=='h4_frozen_C_export_v1','MODEL_SCHEMA')
        require(model['qubit_order']=='qiskit_little_endian','QUBIT_ORDER')
        require(model['preparation_verified'] is True,'MISSING_FROZEN_PREPARATION')
        require(model['binding_hash']==canonical({k:v for k,v in model.items() if k!='binding_hash'}),'EXPORT_BINDING')
        n=model['n_qubits'];require(type(n) is int and n>0,'DIMENSION')
        require(len(model['state'])==1<<n,'STATE_SHAPE')
        require(array_hash(model['state'],vector=True)==model['state_hash'],'STATE_HASH')
        require(set(model['cases'])=={'PF','PR'},'CASE_COVERAGE')
        for method,case in model['cases'].items():
            require(case['metadata']==case['expected_metadata'],'PREPARATION_METADATA_MISMATCH')
            require(case['config']==dict(T=.8,q=8,r=4,K=4,method=method),'FIXED_EVOLUTION')
            expected={(eps,a) for eps in (.05,.01,.005,.001,.0005) for a in ('cosine','sine')}
            require(len(case['saved_axes'])==10 and {(a['epsilon_sig'],a['axis']) for a in case['saved_axes']}==expected,'AXIS_TARGETS')
            require(all(rational(a['target_exact'])>0 for a in case['saved_axes']),'POSITIVE_TARGET_REQUIRED')
            if model['representation']=='pauli_jw':
                require([b['label'] for b in case['blocks']]==case['metadata']['retained_labels'],'PAULI_BLOCK_ORDER')
            else:
                expected_blocks=['one_body']+[str(i) for i in case['metadata']['deterministic_fragment_indices']]
                require([b['operator_key'] for b in case['blocks']]==expected_blocks,'DF_BLOCK_ORDER')
        self.native_basis_cache={};self.completed={}

    def expm(self,A,key):
        if key not in self.cache:
            self.budget.take('dense_matrix_exponential_requests');self.cache[key]=A.exp()
        return self.cache[key]

    def assemble(self):
        m=self.model;n=m['n_qubits'];dim=1<<n;I=identity(dim)
        state=acb_mat([[complex_ball(z)] for z in m['state']]);norm=signal(state,state).real
        require(norm.lower()>0,'STATE_NORMALIZATION_UNRESOLVED')
        spec=m['hamiltonian'];blocks={}
        if m['representation']=='pauli_jw':
            full=I*real(spec['constant'])
            for index,(label,c) in enumerate(spec['terms']):
                blocks[str(index)]=pauli(label)*real(c);full+=blocks[str(index)]
        else:
            for raw in [spec['one_body'],*spec['g_matrices']]:
                require(all(rational(raw[i][j][0])==rational(raw[j][i][0]) and rational(raw[i][j][1])==-rational(raw[j][i][1]) for i in range(n) for j in range(n)), 'ENCODED_HERMITICITY_UNPROVED')
            one=one_body(spec['one_body']);full=I*real(spec['constant'])+one;blocks['one_body']=one
            for i,(lam,g) in enumerate(zip(spec['lambdas'],spec['g_matrices'],strict=True)):
                a=one_body(g);blocks[str(i)]=a*a*real(lam);full+=blocks[str(i)]
        # Exact Hermiticity is required for the fixed encoded full Hamiltonian.
        require(all((full[i,j]-full[j,i].conjugate()).contains(0) for i in range(dim) for j in range(dim)), 'HAMILTONIAN_HERMITICITY')
        residual=frobenius(full-matrix(m['saved_full_matrix']))
        require(residual.upper()<=real(m['operator_agreement_limit']),'OPERATOR_RECONSTRUCTION_MISMATCH')
        return I,state,norm,full,blocks,residual

    def basis(self,spec):
        key=spec['basis_hash']
        if key not in self.native_basis_cache:self.native_basis_cache[key]=basis_operators(spec,self.model['n_qubits'])
        return self.native_basis_cache[key]

    def semantic_bound(self,case,method):
        # Removing adjacent Ui U pairs differs from explicit concatenation by
        # at most N*d*C^(2N). Keep nonunitary norm amplification explicitly.
        n=self.model['n_qubits'];I=identity(1<<n);q=8;r=4
        specs=[b['basis'] for b in case['blocks'] if 'basis' in b]
        specs += [c['basis'] for c in case['components'] if 'basis' in c]
        unique={s['basis_hash']:s for s in specs};defect=arb(0);bound=arb(1)
        for spec in unique.values():
            U,Ui=self.basis(spec)
            defect=max(defect,frobenius(Ui*U-I))
            bound=max(bound,contraction_norm(U),contraction_norm(Ui))
        applications=q*(2*len(case['blocks'])+(r*5 if method=='PR' else 0))
        reuse=applications*defect*bound**(2*applications)
        # Binary64 gamma_n bound for a finite straight-line phase accumulation.
        # Overcount the maximum local primitive count and phase-sign additions;
        # no stochastic draw or presumed empirical allowance is used.
        gate_count=sum(len(s['operations']) for s in unique.values())
        operations=8*(1+applications)*(1+n*n+gate_count)
        unit=Q(1,1<<52);require(operations*unit<1,'PHASE_BOUND_DOMAIN')
        gamma=real(operations*unit/(1-operations*unit))
        angles=sum((abs(real(x)) for x in case['native_scalar_phases']),arb(0))*q
        for b in case['blocks']:
            if 'coefficient' in b:angles+=2*q*abs(real(b['coefficient'])*real(self.model['step_time'])/2)
            else:
                p=b['primitives'];angles+=2*q*(abs(real(p['global_phase']))+sum((abs(real(x[-1])) for x in p['rz']+p['rzz']),arb(0)))
        if method=='PR':angles+=q*r*(8*arb.pi()+sum((abs(real(a)) for a in case['law']['angles']),arb(0)))
        rounding=gamma*(1+angles)+real(operations*Q(1,1<<1074))
        pi_difference=abs(real(float(math.pi))-arb.pi())*2*q*r if method=='PR' else arb(0)
        return ((reuse+rounding+pi_difference)*bound**(2*applications)).upper()

    def tier(self,bits):
        self.cache={};self.native_basis_cache={}
        with ctx.workprec(bits):
            return self._tier(bits)

    def _tier(self,bits):
        m=self.model;n=m['n_qubits'];delta=real(m['step_time']);T=real(.8);r=4;q=8
        I,state,norm,full,ideal_blocks,op_residual=self.assemble()
        self.budget.take('reference_signal_enclosures')
        reference=signal(state,self.expm(full*acb(0,-T),('reference',bits))*state)
        output=dict(bits=bits,reference=serialize(reference),state_norm=endpoints(norm),
            state_norm_defect=endpoints(abs(norm-1)),operator_snapshot_residual_upper=endpoints(op_residual),
            nominal_law={},implemented_law={},normalization={},axes=[],formal_certificate_accepted=False,
            original_N_G_changed=False)
        statuses=[];model_unresolved=False
        for method in ('PF','PR'):
            if method in self.completed:
                done=self.completed[method]
                output['nominal_law'][method]=done['nominal'];output['implemented_law'][method]=done['implemented'];output['normalization'][method]=done['normalization']
                output['axes'].extend(done['axes']);statuses.extend(['TARGET_MET']*len(done['axes']))
                continue
            case=m['cases'][method];nominal=[];native=[];tail=full-I*real(m['hamiltonian']['constant'])
            for b in case['blocks']:
                H=ideal_blocks[b['operator_key']];tail-=H
                if m['representation']=='pauli_jw':
                    P=pauli(b['label']);angle=real(b['coefficient'])*delta/2
                    E=I*angle.cos()+P*acb(0,-angle.sin())
                    native_E=E  # Gate-parameter rounding bound is recorded by the exporter.
                else:
                    E=self.expm(H*acb(0,-delta/2),('block',b['operator_key'],bits))
                    U,Ui=self.basis(b['basis']);native_E=U*primitive_diagonal(b['primitives'],n)*Ui
                nominal.append(E);native.append(native_E)
            extracted=real(case['extracted_identity']);tail-=I*extracted
            nominal_scalar=phase(-(real(m['hamiltonian']['constant'])+extracted)*delta)
            native_scalar=phase(sum((real(x) for x in case['native_scalar_phases']),arb(0)))
            tailpoly=polynomial5(tail*acb(0,-delta/r)) if method=='PR' else None
            self.budget.take('nominal_law_mean_enclosures')
            nominal_out=evolve(state,nominal,tailpoly,nominal_scalar,q,r)
            nominal_signal=signal(state,nominal_out)
            nominal_bias=nominal_signal-reference
            output['nominal_law'][method]=dict(corrected_mean=serialize(nominal_signal),signed_bias=serialize(nominal_bias),
                finite_degree=5 if method=='PR' else None,nonunitary_norm= endpoints(contraction_norm(tailpoly)) if method=='PR' else None)
            implemented_tail=None;correction=arb(1);norm_status=True
            if method=='PR':
                self.budget.take('PR_normalization_enclosures')
                lam=sum((real(x['coefficient_abs']) for x in case['components']),arb(0))
                proof=finite_normalization(lam,delta,r,q)
                law=case['law'];ps,pis=verify_probability_mapping(law)
                require(ps and len(ps)==len(case['components']),'COMPONENT_COVERAGE')
                components=[];rotations=[[],[],[]];reconstructed=acb_mat(I.nrows(),I.ncols())
                for comp in case['components']:
                    if m['representation']=='pauli_jw':
                        P=pauli(comp['label'])*comp['sign'];U=Ui=None
                    else:
                        U,Ui=self.basis(comp['basis']);label=''.join('Z' if n-1-j in comp['support'] else 'I' for j in range(n))
                        Z=pauli(label)*comp['sign'];P=U*Z*Ui
                    components.append(P);reconstructed+=P*real(comp['coefficient_abs'])
                    for i,angle in enumerate(law['angles']):
                        angle=real(angle)
                        R=I*angle.cos()+P*acb(0,-angle.sin()) if U is None else U*(I*angle.cos()+Z*acb(0,-angle.sin()))*Ui
                        rotations[i].append(R)
                closure=frobenius(reconstructed-tail)
                require(closure.upper()<=real(m['operator_agreement_limit']),'TAIL_OPERATOR_CORRESPONDENCE')
                self.budget.take('implemented_PR_law_mean_enclosures')
                implemented_tail=finite_implemented_mean(components,rotations,ps,pis)
                correction=real(case['saved_normalization']['B'])
                savedupper=real(case['saved_normalization']['log_B_upper'])
                norm_status=bool(proof['log_B'].upper()<=savedupper)
                norm_overlap=bool(proof['log_B'].lower()<=savedupper)
                if not norm_status and not norm_overlap:model_unresolved=True
                output['normalization'][method]=dict(nominal={k:[endpoints(x) for x in v] if isinstance(v,list) else endpoints(v) for k,v in proof.items()},
                    saved=case['saved_normalization'],saved_log_B_upper_encloses=norm_status,
                    lambda_saved_difference=endpoints(abs(lam-real(case['saved_normalization']['lambda_r']))),
                    tail_correspondence_residual_upper=endpoints(closure),
                    component_effective_probabilities=[str(x) for x in ps],order_effective_probabilities=[str(x) for x in pis],
                    IID_uniform_word_assumption=True,PCG64_independence_proved=False,
                    saved_B_difference=endpoints(abs(proof['B']-correction)),
                    range_plus_minus_saved_B_certified=case['exact_event_range_proved'],formal_certificate_accepted=False)
            else:output['normalization'][method]=dict(B='1',log_B='0',deterministic_structure=True,formal_certificate_accepted=False)
            native_out=evolve(state,native,implemented_tail,native_scalar,q,r)
            implemented_signal=signal(state,native_out)*correction
            # Do not normalize or replace the input state. Record both the encoded
            # signal and the same-ray physical expectation as explicit quantities.
            physical_bias=(implemented_signal-reference)/norm
            law_defect=abs(implemented_signal-nominal_signal)
            state_defect=abs((implemented_signal-reference)/norm-(implemented_signal-reference))
            # Exported circuit-rounding/reuse allowance is an analytically bounded
            # worst-case operator deviation; never a presumed zero or empirical u.
            semantic=self.semantic_bound(case,method)*norm*abs(correction)
            if method=='PR':
                output['normalization'][method]['normalization_signal_error_upper']=endpoints(abs(proof['B']-correction)*abs(signal(state,native_out)/norm))
            output['implemented_law'][method]=dict(corrected_encoded_mean=serialize(implemented_signal),
                same_ray_physical_bias=serialize(physical_bias),law_difference=endpoints(law_defect),
                state_normalization_difference=endpoints(state_defect),circuit_semantic_allowance=endpoints(semantic),
                numerical_evidence_only=True,formal_certificate_accepted=False)
            for axis in case['saved_axes']:
                component=physical_bias.real if axis['axis']=='cosine' else physical_bias.imag
                distance=abs(component-real(axis['signed_bias']))
                # The allowance is a known upper error, not an unknown positive
                # error to subtract from a lower endpoint for advancement.
                upper=distance.upper()+semantic.upper()
                lower=max(arb(0),distance.lower()-semantic.upper())
                target=rational(axis['target_exact'])
                status='TARGET_MET' if dyadic(upper.upper())<=target else 'WIDTH_INSUFFICIENT' if dyadic(lower.lower())<=target else 'TARGET_NOT_MET'
                if not norm_status:status='WIDTH_INSUFFICIENT' if norm_overlap else 'MODEL_UNRESOLVED'
                if not case['circuit_correspondence_proved']:status='MODEL_UNRESOLVED';model_unresolved=True
                statuses.append(status)
                output['axes'].append(dict(case=m['representation']+'_'+method,epsilon_sig=axis['epsilon_sig'],axis=axis['axis'],
                    precision_bits=bits,target_exact=str(target),bias_distance=endpoints(distance),total_error_upper=endpoints(upper),status=status,
                    old_N=None,old_G=None,old_status='UNDETERMINED',formal_certificate_accepted=False))
            case_axes=[a for a in output['axes'] if a['case']==m['representation']+'_'+method]
            if all(a['status']=='TARGET_MET' for a in case_axes):
                self.completed[method]=dict(nominal=output['nominal_law'][method],implemented=output['implemented_law'][method],normalization=output['normalization'][method],axes=case_axes)
        output['status']='MODEL_UNRESOLVED' if model_unresolved else 'TARGET_NOT_MET' if 'TARGET_NOT_MET' in statuses else 'WIDTH_INSUFFICIENT' if 'WIDTH_INSUFFICIENT' in statuses else 'TARGET_MET'
        return output
