"""Finite comparison contract. No numerical workload during import or preflight."""
from fractions import Fraction as F
import hashlib
import json

METHODS = ('NORM', 'FIVE', 'OLD', 'OCC', 'SDP_EQUIVALENT', 'DENSITY_SOS', 'FULL_H_LOWER')
CAPS = {'full_modes': 6, 'trial_pool': 8, 'eigh_per_mode': 128,
        'solver_calls_per_mode': 3, 'solver_iterations_per_call': 20000,
        'PSD_probes_per_mode': 256, 'rational_bits': 4096,
        'checker_operations': 10000000, 'proposal_bits': 40,
        'repair_steps': 12, 'max_method_record_bytes': 4*1024**2}
DELTA_TIERS = (F(1), F(1,2), F(1,4))
SEARCH_STEPS = 32
SOS_LEVEL = 'real_quadratic_P_Q_G_particle_hole_number_v1'
BLOCKS = ('particle', 'hole', 'D', 'Q', 'G')


def enc(x):
    if isinstance(x,F): return [x.numerator,x.denominator]
    if isinstance(x,dict): return {k:enc(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [enc(v) for v in x]
    return x


def canonical(x): return json.dumps(enc(x),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x): return hashlib.sha256(canonical(x)).hexdigest()


def rat(x):
    if not isinstance(x,list) or len(x)!=2 or any(type(v) is not int for v in x) or x[1]<=0:
        raise ValueError('EXACT_RATIONAL_SCHEMA')
    if max(abs(x[0]).bit_length(),x[1].bit_length())>CAPS['rational_bits']:
        raise ArithmeticError('RATIONAL_BIT_CAP')
    return F(*x)


def mat(x,n=None):
    if not isinstance(x,list) or not x or any(not isinstance(r,list) or len(r)!=len(x) for r in x):
        raise ValueError('SQUARE_MATRIX_SCHEMA')
    if n is not None and len(x)!=n: raise ValueError('MATRIX_DIMENSION')
    A=[[rat(v) for v in r] for r in x]
    if any(A[i][j]!=A[j][i] for i in range(len(A)) for j in range(len(A))):
        raise ValueError('REAL_SYMMETRY')
    return A


def decode_case(case):
    if set(case)!={'id','family','seed','modes','nu','h','density','candidate_modes','epsilon_proj'}:
        raise ValueError('CASE_SCHEMA_OR_ORACLE')
    n=case['modes'];nu=case['nu']
    if type(n) is not int or not 3<=n<=CAPS['full_modes'] or type(nu) is not int or not 1<=nu<n:
        raise ValueError('MODE_OR_PARTICLE_SCOPE')
    if not isinstance(case['id'],str) or not case['id'] or type(case['seed']) is not int:
        raise ValueError('CASE_ID')
    h=mat(case['h'],n); J={}
    if not isinstance(case['density'],list): raise ValueError('DENSITY_SCHEMA')
    for row in case['density']:
        if not isinstance(row,list) or len(row)!=3: raise ValueError('DENSITY_ROW')
        i,j,x=row
        if type(i) is not int or type(j) is not int or not 0<=i<j<n or (i,j) in J:
            raise ValueError('UNORDERED_DENSITY_INDEX')
        J[i,j]=rat(x)
        if J[i,j]<0: raise ValueError('NONNEGATIVE_DENSITY_REQUIRED')
    modes=case['candidate_modes']
    if not isinstance(modes,list) or not modes or len(set(modes))!=len(modes) or any(type(d) is not int or not 0<=d<n for d in modes):
        raise ValueError('CANDIDATE_MODES')
    eps=rat(case['epsilon_proj'])
    if eps<=0: raise ValueError('POSITIVE_PROJECTION_BUDGET')
    return h,J,nu,modes,eps


def active_data(case,d):
    h,J,nu,modes,eps=decode_case(case)
    if d not in modes: raise ValueError('MODE_NOT_PREREGISTERED')
    active=[i for i in range(len(h)) if i!=d]
    A=[[h[i][j] for j in active] for i in active];v=[h[i][d] for i in active]
    den={(i,j):J.get(tuple(sorted((active[i],active[j]))),F(0)) for i in range(len(active)) for j in range(i+1,len(active))}
    external=[[A[i][j]+(J.get(tuple(sorted((active[i],d))),F(0)) if i==j else 0) for j in range(len(active))] for i in range(len(active))]
    return active,A,v,den,external,h[d][d],nu,eps
