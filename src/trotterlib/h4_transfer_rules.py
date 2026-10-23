"""Exact formal transfer rules; no saved-H4 numeric or acceptance API.

Reuse phase-T words, phase-sensitive equality and Hadamard/readout rules.
The small polynomial quotient checks diagonal identities with bit^2=bit;
its variables are formal real symbols, never saved molecular parameters.
"""
from fractions import Fraction
import importlib.util
from pathlib import Path

_spec=importlib.util.spec_from_file_location('transfer_frozen_T',Path(__file__).with_name('h4_certificate_gates.py'))
T=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(T)
require=T.require
COVERAGE_STATES=frozenset({'PROVED_DISJOINT','PROVED_COVERED','OVERLAP_UNRESOLVED','MISSING_PROOF','NOT_APPLICABLE'})


class Polynomial:
    """Rational polynomial with occupation-bit idempotence only."""
    def __init__(self,terms):
        self.terms={key:Fraction(value) for key,value in terms.items() if value}

    @classmethod
    def constant(cls,value):return cls({():Fraction(value)})

    @classmethod
    def variable(cls,name):
        require(type(name) is str and name,'FORMAL_VARIABLE')
        return cls({(name,):Fraction(1)})

    def __add__(self,other):
        if not isinstance(other,Polynomial):other=self.constant(other)
        out=dict(self.terms)
        for key,value in other.terms.items():out[key]=out.get(key,0)+value
        return Polynomial(out)

    __radd__=__add__
    def __neg__(self):return Polynomial({key:-value for key,value in self.terms.items()})
    def __sub__(self,other):return self+-other

    def __mul__(self,other):
        if not isinstance(other,Polynomial):other=self.constant(other)
        out={}
        for a,x in self.terms.items():
            for b,y in other.terms.items():
                combined=a+b
                ordinary=[v for v in combined if not v.startswith('bit:')]
                bits={v for v in combined if v.startswith('bit:')}
                key=tuple(sorted(ordinary+list(bits)))
                out[key]=out.get(key,0)+x*y
        return Polynomial(out)

    __rmul__=__mul__

    def encode(self):
        return [{'monomial':list(key),'coefficient':str(value)} for key,value in sorted(self.terms.items())]


def formal_result(name,conclusion,assumptions):
    return dict(status='FORMAL_LEMMA_CHECKED',rule=name,conclusion=conclusion,
                assumptions=assumptions,binding_to_saved_H4='NOT_PROVED',formal_certificate_accepted=False)


def diagonal_identity(kind,n=2,*,phase_sign=1):
    """Prove real one-body/squared diagonal phase, including global phase.

    n_k=(1-Z_k)/2. Squared DF is kept as a square, never an involution.
    phase_sign is a mutation-test hook: an incorrect sign must be rejected.
    """
    require(kind in ('ONE_BODY','DF_SQUARED') and type(n) is int and n>0,'DIAGONAL_RULE_DOMAIN')
    require(type(phase_sign) is int and phase_sign in (-1,1),'PHASE_SIGN')
    var=Polynomial.variable;zero=Polynomial.constant(0)
    t=var('tau');lam=var('lambda');eta=[var('eta:'+str(k)) for k in range(n)]
    bits=[var('bit:'+str(k)) for k in range(n)];z=[1+(-2)*b for b in bits]
    if kind=='ONE_BODY':
        angles=[(-1)*t*x for x in eta]
        phase=sum(angles,zero)*Fraction(1,2)*phase_sign
        actual=phase+sum([(-Fraction(1,2))*a*b for a,b in zip(angles,z)],zero)
        target=(-1)*t*sum([x*b for x,b in zip(eta,bits)],zero)
    else:
        alpha=[(-1)*t*lam*x*x for x in eta];angles=list(alpha)
        phase=sum(alpha,zero)*Fraction(1,2);pairs=[]
        for k in range(n):
            for j in range(k+1,n):
                beta=(-2)*t*lam*eta[k]*eta[j]
                angles[k]=angles[k]+beta*Fraction(1,2)
                angles[j]=angles[j]+beta*Fraction(1,2)
                phase=phase+beta*Fraction(1,4)
                pairs.append((k,j,beta*(-Fraction(1,2))))
        phase=phase*phase_sign
        actual=phase+sum([(-Fraction(1,2))*a*b for a,b in zip(angles,z)],zero)
        actual=actual+sum([(-Fraction(1,2))*a*z[k]*z[j] for k,j,a in pairs],zero)
        occupation=sum([x*b for x,b in zip(eta,bits)],zero)
        target=(-1)*t*lam*occupation*occupation
    require(actual.terms==target.terms,'DIAGONAL_PHASE_IDENTITY_FAILED')
    return dict(formal_result(kind,'Exact phase polynomial equals -tau*H_diagonal',
                ['All tau/lambda/eta are real','Commuting occupation projectors n_k^2=n_k',
                 'No binary64 parameter generation or literal controlled decomposition is proved']),
                polynomial=target.encode(),system_qubits=n,DF_squared_replaced_by_involution=False)


def controlled_word(evolution,zero_branch,one_branch):
    return T.check_symbolic_rule(dict(rule='CONTROLLED_WORD',data=dict(evolution=evolution,zero_branch=zero_branch,one_branch=one_branch)))


def ordered_basis(n,names,inverse_names):
    require(type(names) is list and all(type(x) is str for x in names),'BASIS_NAMES')
    require(inverse_names==names[::-1],'INVERSE_REVERSE_ORDER')
    forward=T.identity_word(n)
    forward['factors']=[dict(kind='UNITARY_SYMBOL',name=T.identifier(x),inverse=False) for x in names[::-1]]
    inverse=T.adjoint_word(forward)
    T.equal_words(T.product(forward,inverse),T.identity_word(n))
    return dict(formal_result('ORDERED_BASIS','F D F-dagger; rightmost factor acts first',
                ['Exact ideal unitary symbols only','Rounded forward/inverse retain separate L errors']),
                forward=forward,inverse=inverse)


def event_phase(order,product_signs,rotation_sign,*,identity=False):
    require(type(order) is int and order in (0,2,4),'FINITE_RTE_EVEN_ORDER')
    require(type(product_signs) is list and len(product_signs)==order and all(type(s) is int and s in (-1,1) for s in product_signs),'PRODUCT_SIGNS')
    require(type(rotation_sign) is int and rotation_sign in (-1,1) and type(identity) is bool,'ROTATION_SIGN')
    # Exact (-i)^order and negative product signs; no floating pi is evaluated.
    quarters=(-order+2*sum(s==-1 for s in product_signs))%4
    return dict(formal_result('SIGNED_EVENT_PHASE','Retain product phase and signed rotation',
                ['Exact pi symbol','rotation_angle=sign*unsigned_angle',
                 'Literal event phase tolerance and binary64 pi remain separate obligations']),
                phase_quarters=quarters,rotation_parameter=f'{rotation_sign}*alpha',
                identity_rotation_phase=f'{-rotation_sign}*alpha' if identity else None,
                signed_identity_is_relative_under_control=True)


def wrapper(axis):
    return T.check_symbolic_rule(dict(rule='HADAMARD_WRAPPER',data=dict(axis=axis,
        operations=['H','CONTROLLED_U']+(['SDG'] if axis=='sine' else [])+['H'])))


def measurement_transfer(*,normalized_ray=True,zero_exact=False):
    require(type(normalized_ray) is bool and type(zero_exact) is bool,'TRANSFER_FLAGS')
    branch='eta1' if zero_exact else 'eta0+eta1+eta0*eta1'
    return dict(formal_result('CROSS_BRANCH_AND_SCALE',
                '|Re/Im <psi|(A0^dagger A1 - V0^dagger V1)|psi>| <= '+branch,
                ['||V0||=||V1||=1','||A0-V0||<=eta0; ||A1-V1||<=eta1',
                 'Nonnegative eta0/eta1; all phases included',
                 'Literal Born-readout identity must be proved separately']),
                cross_difference='(A0-V0)^dagger*A1+V0^dagger*(A1-V1)',
                unscaled_bound=branch,measurement_bound_symbolic='B_saved*('+branch+')',
                expectation_factor='1' if normalized_ray else 'state_norm',
                same_ray_normalization_cancels_state_norm=True,
                B_saved_not_nominal=True,operator_norm_one_not_inferred_from_binary_support=True,
                numerical_B_scaled_bound_evaluated=False)


def validate_term(term):
    T.keys(term,{'id','domain','context','edge','primitives','bound_ref','guarantee'},'COVERAGE_TERM_SCHEMA')
    T.identifier(term['id']);require(term['domain'] in ('FORMAL','SAVED_H4'),'COVERAGE_DOMAIN')
    require(type(term['context']) is str and bool(term['context']),'COVERAGE_CONTEXT')
    T.keys(term['edge'],{'from','to','quantity','state','scale'},'COVERAGE_EDGE_SCHEMA')
    require(all(type(x) is str and x for x in term['edge'].values()),'COVERAGE_EDGE_TYPE')
    require(term['edge']['quantity'] in ('OPERATOR_NORM','SIGNAL_ERROR'),'COVERAGE_QUANTITY')
    require(type(term['primitives']) is list and term['primitives'] and all(type(x) is str for x in term['primitives']) and len(set(term['primitives']))==len(term['primitives']),'COVERAGE_PRIMITIVES')
    require(type(term['bound_ref']) is str and term['bound_ref'],'BOUND_REFERENCE_REQUIRED')
    require(term['guarantee'] in ('FORMAL_HYPOTHESIS','SAVED_UNCHECKED'),'COVERAGE_GUARANTEE')
    require(term['domain']!='SAVED_H4' or term['guarantee']=='SAVED_UNCHECKED','SAVED_FLAGS_ARE_NOT_PROOFS')


def coverage_relation(left,right,*,rule=None,operation='KEEP_SEPARATE'):
    validate_term(left);validate_term(right)
    require(operation in ('KEEP_SEPARATE','APPLY_RULE'),'UNPROVED_SUBTRACTION_OR_ADDITION')
    require(left['context']==right['context'] and left['domain']==right['domain'],'CROSS_TARGET_COVERAGE_REUSE')
    require(left['id']!=right['id'],'DUPLICATE_ERROR_TERM_ID')
    a,b=left['edge'],right['edge'];labels=sorted(set(left['primitives'])&set(right['primitives']))
    status='OVERLAP_UNRESOLVED' if a==b and labels else 'MISSING_PROOF'
    if rule=='IDENTICAL_BOUND':
        require(a==b and left['bound_ref']==right['bound_ref'] and left['primitives']==right['primitives'],'COVERED_BOUND_IDENTITY_REQUIRED')
        status='PROVED_COVERED'
    elif rule=='TELESCOPING_CHAIN':
        require(a['to']==b['from'] and all(a[k]==b[k] for k in ('quantity','state','scale')),'NONCONTIGUOUS_OR_MIXED_NORM_SCALE_CHAIN')
        require(a['quantity']=='OPERATOR_NORM','OPERATOR_TRIANGLE_REQUIRED')
        status='PROVED_DISJOINT' if left['domain']=='FORMAL' else 'MISSING_PROOF'
    elif rule is not None:
        raise T.GateValidationError('UNSUPPORTED_COVERAGE_PROOF_RULE')
    return dict(status=status,label_collisions=labels,same_endpoint_quantity=a==b,
                numerical_combination_performed=False,formal_certificate_accepted=False,
                binding_to_saved_H4='NOT_PROVED',
                meaning='DISJOINT means distinct accounting links in a proved conditional triangle chain, not independent physical noise sources',
                combination='FORMAL_RULE_ONLY' if status.startswith('PROVED') else 'COMBINATION_BLOCKED')


def not_applicable(representation):
    require(representation=='pauli_jw','DF_LOCAL_PROOF_IS_APPLICABLE')
    return dict(status='NOT_APPLICABLE',reason='DF local matrices are not used by Pauli-JW')


def controlled_diagonal_identity(kind):
    """Exact ideal decompositions, including both branches and parity.

    This is a mathematical rule. It does not assert that Qiskit historically
    chose the sequence or that binary64 angle halves had no error.
    """
    require(kind in ('CRZ','RZZ','CONTROLLED_RZZ','PHASE'),'CONTROL_DIAGONAL_RULE')
    v=Polynomial.variable;theta=v('theta');control=v('bit:c');a=v('bit:a');b=v('bit:b')
    za=1+(-2)*a;zb=1+(-2)*b;zc=1+(-2)*control
    if kind=='CRZ':
        actual=(-Fraction(1,4))*theta*za+Fraction(1,4)*theta*zc*za
        target=(-Fraction(1,2))*theta*control*za
        sequence=['RZ(theta/2)','CX(c,t)','RZ(-theta/2)','CX(c,t)']
    elif kind=='RZZ':
        parity=a+b+(-2)*a*b
        actual=(-Fraction(1,2))*theta*(1+(-2)*parity)
        target=(-Fraction(1,2))*theta*za*zb
        sequence=['CX(a,b)','RZ(theta)_b','CX(a,b)']
    elif kind=='CONTROLLED_RZZ':
        parity=a+b+(-2)*a*b
        actual=(-Fraction(1,4))*theta*(1+(-2)*parity)+Fraction(1,4)*theta*zc*(1+(-2)*parity)
        target=(-Fraction(1,2))*theta*control*za*zb
        sequence=['CX(a,b)','CRZ(theta)_(c,b)','CX(a,b)']
    else:
        actual=theta*Fraction(1,2)+(-Fraction(1,2))*theta*za
        target=theta*a
        sequence=['global_phase(theta/2)','RZ(theta)']
    require(actual.terms==target.terms,'CONTROL_DIAGONAL_IDENTITY_FAILED')
    return dict(formal_result(kind,'Exact diagonal phase and restored computational bits',
                ['Exact real theta; CX is exact reversible parity',
                 'Exact symbolic identities do not certify literal floating parameters or actual selected decomposition']),
                chronological_sequence=sequence,phase_polynomial=target.encode(),
                implementation_choice_in_saved_H4='NOT_PROVED')
