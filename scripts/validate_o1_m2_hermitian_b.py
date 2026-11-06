"""Prepared B-candidate operator-only runner; separate adoption/execution required.

Import/help do not decode saved data. No state, dynamics, circuit or cost calls.
All numerical criteria below are SOURCE-fixed before any molecular result.
"""
import argparse
from dataclasses import asdict
from fractions import Fraction
import json
import hashlib
import math
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import validate_o1_m2_h4_operator as previous
from trottertracks import o1_m2_hermitian_b as b


def fixed_plan(spatial, frames):
    """Conservative operation envelopes, not fitted observed tolerances.

    Tensor count covers the entire direct five-factor contraction plus scratch
    sums (more than a per-entry dot count). h contractions use 2n+2. At most
    M=1+2n²+4n⁴ spin monomials contribute; quartic JW has <=16 Pauli words.
    Mapper count 32M includes JW collection and dense Pauli accumulation.
    CAR reference uses only exact binary scaling/signs then fsum (4M allowance).
    Counts/guard are engineering bounds, not an interval proof of NumPy BLAS.
    """
    if any(type(v) is not int or v < 1 for v in (spatial,frames)):
        raise ValueError('Positive spatial/frame dimensions required')
    n=spatial; monomials=1+2*n*n+4*n**4
    density_path=8*frames*n*((1<<(2*n))+n*n)
    return b.RoundoffPlan(max(8*frames*n**6+8*n**4,density_path),2*n+2,2*n+2,
                         32*monomials,4*monomials,
                         guard=8.,maximum_relative_mapper_allowance=.01)


H4_PLAN = fixed_plan(4,8)
FORMATION_RELATIVE_LIMIT = .01


def fsum_car(reference, packet):
    """Existing sparse CAR monomials; fsum each entry without JW/spin_lift.

    CAR entries are exact integers 0,+/-1 (spin-summed E can have 2). Scaling
    by signs and 1/2 is exact for normal binary64; fsum rounds the final sum.
    """
    import numpy as np
    grouped={}
    def add(coefficient, matrix):
        if coefficient==0: return
        matrix=matrix.tocoo()
        for row,col,value in zip(matrix.row,matrix.col,matrix.data,strict=True):
            if value.imag != 0 or value.real not in (-2,-1,0,1,2):
                raise ValueError('Independent CAR monomial must be exact integer')
            grouped.setdefault((int(row),int(col)),[]).append(float(coefficient)*float(value.real))
    add(packet.nuclear_constant,reference.identity)
    for p,q in np.ndindex(reference.n,reference.n):
        add(packet.h[p,q],reference.E[p,q])
    for p,q,r,s in np.ndindex((reference.n,)*4):
        for sigma,tau in np.ndindex(2,2):
            term=((2*p+sigma,1),(2*r+tau,1),(2*s+tau,0),(2*q+sigma,0))
            if packet.g[p,q,r,s] != 0:
                add(.5*packet.g[p,q,r,s],reference.monomial(term))
    out=np.zeros((reference.dimension,reference.dimension),dtype=complex)
    for key,values in grouped.items(): out[key]=math.fsum(values)
    if not np.isfinite(out).all(): raise ValueError('Nonfinite CAR reference')
    return out


def exact_formation_diagnostics(candidate):
    """Exact rational P(H_binary64)-P(C_binary64), BEFORE mapper comparison.

    This separates P/subtraction rounding from upstream core contraction.
    Fractions represent saved input bits exactly; no generation-precision or
    exact-J/U-core claim follows. No fsum tolerance is used for these deltas.
    """
    import numpy as np
    raw=candidate.raw
    losses={'canonical':Fraction(), 'projected_raw':Fraction()}
    reference_scale=Fraction()
    coefficient_max=Fraction()
    exact=lambda x:Fraction.from_float(float(x))
    for name,axes in (('h',(1,0)),('g',(1,0,3,2))):
        H,C=getattr(raw.physical,name),getattr(raw.core,name)
        for index in np.ndindex(H.shape):
            adjoint=tuple(index[a] for a in axes)
            value=(exact(H[index])+exact(H[adjoint])-exact(C[index])-exact(C[adjoint]))/2
            reference_scale+=2*abs(value)
            coefficient_max=max(coefficient_max,abs(value))
            for route,packet in (('canonical',candidate.residual),('projected_raw',candidate.projected_raw_residual)):
                losses[route]+=2*abs(exact(getattr(packet,name)[index])-value)
    return dict(reference='exact Fraction of binary64 H0/C0 packets; upstream core not exactified',
        exact_coefficient_scale=float(reference_scale),exact_coefficient_max=float(coefficient_max),
        canonical_formation_error_bound=float(losses['canonical']),
        projected_raw_formation_error_bound=float(losses['projected_raw']),
        exact_coefficient_nonzero=reference_scale!=0)


def evaluate(raw, *, plan=None, projection_limits=None):
    """Explicit array science entry. Tests supply only invented arrays."""
    import numpy as np
    from trottertracks.c1_stage1 import sector_indices
    n=len(raw.physical.h)
    plan=plan or fixed_plan(n,len(raw.general_core.frames))
    candidate=b.derive_hermitian_b(raw)
    reference=previous.CARReference(n)
    names=('physical','core','residual')
    car={name:fsum_car(reference,getattr(candidate,name)) for name in names}
    raw_car={name:fsum_car(reference,getattr(raw,name)) for name in names}
    projected_raw=fsum_car(reference,candidate.projected_raw_residual)
    density=reference.density_core(raw.general_core)
    density_H=(density+density.conj().T)/2
    packets=dict(zip(names,candidate.pauli_parts(),strict=True))
    mapped={name:previous.pauli_matrix(packets[name]) for name in names}
    budget=b.predeclared_budgets(candidate,plan)
    closure=float(np.max(abs(car['core']+car['residual']-car['physical'])))
    formation=b.assess_formation(candidate,plan,closure_error=closure,
                                input_projection_limits=projection_limits)
    mapper=b.assess_residual_mapper(candidate,car['residual'],mapped['residual'],plan)
    exact=exact_formation_diagnostics(candidate)
    residual_size=float(np.max(abs(car['residual'])))
    density_gap=float(np.max(abs(density_H-car['core'])))
    upstream_resolution=(density_gap <= FORMATION_RELATIVE_LIMIT*residual_size
                         if residual_size>0 else density_gap==0.)
    formation_resolution=(exact['canonical_formation_error_bound'] <= FORMATION_RELATIVE_LIMIT*residual_size
        if residual_size>0 else not exact['exact_coefficient_nonzero'])
    spaces={'full':np.arange(reference.dimension)}
    if raw.sector is not None: spaces['sector']=sector_indices(n,*raw.sector)
    checks=[]
    for name in names:
        tolerance=budget['residual_mapper_allowance'] if name=='residual' else budget['formation_allowance']
        for space,indices in spaces.items():
            diff=(mapped[name]-car[name])[np.ix_(indices,indices)]
            maximum=float(np.max(abs(diff)))
            checks.append(dict(name=name+'_Pauli_vs_CAR',space=space,max_entry=maximum,
                               tolerance=tolerance,passed=maximum<=tolerance))
    for name,diff in (('independent_density_core_vs_B_core',density_H-car['core']),
                      ('independent_density_core_plus_R_vs_B_H',density_H+car['residual']-car['physical'])):
        for space,indices in spaces.items():
            maximum=float(np.max(abs(diff[np.ix_(indices,indices)])))
            tolerance=budget['formation_allowance']
            checks.append(dict(name=name,space=space,max_entry=maximum,tolerance=tolerance,passed=maximum<=tolerance))
    identity_error=abs(packets['residual'].constant-float(np.trace(car['residual']).real/reference.dimension))
    bookkeeping=(raw.residual.nuclear_constant==candidate.residual.nuclear_constant==0.
        and candidate.physical.nuclear_constant==candidate.core.nuclear_constant
        and identity_error<=budget['residual_mapper_allowance'])
    return dict(scope='B_CANDIDATE_OPERATOR_ONLY',convention=candidate.convention,
        formal_adoption=False,physical_generation_accuracy_qualified=False,
        operator_checks_passed=(all(c['passed'] for c in checks) and mapper['status']=='PASS'
            and formation['formation_status']=='PASS' and formation_resolution and upstream_resolution and bookkeeping),
        model_change_status=formation['model_change_status'],roundoff_plan=asdict(plan),
        reference_precision='binary64 exact CAR monomials + math.fsum; formation diagnostics exact Fraction',
        formation_relative_limit=FORMATION_RELATIVE_LIMIT,formation_resolution_qualified=formation_resolution,
        upstream_density_gap_max_entry=density_gap,upstream_reference_resolution_qualified=upstream_resolution,
        exact_formation=exact,formation=formation,residual_mapper=mapper,checks=checks,
        canonical_vs_P_raw_R_max_entry=float(np.max(abs(car['residual']-projected_raw))),
        raw_antihermitian_half_max_entry={name:float(np.max(abs(matrix-matrix.conj().T)))/2
                                        for name,matrix in raw_car.items()},
        identity={name:dict(nuclear=getattr(candidate,name).nuclear_constant,electronic=packets[name].constant,
                           nonidentity_terms=len(packets[name].terms)) for name in names},
        residual_identity_error=identity_error,scalar_identity_bookkeeping=bookkeeping,
        sector_dimensions={name:len(indices) for name,indices in spaces.items()},
        stopped_after='operator-only candidate; no adoption/state/dynamics/circuit/shot/cost')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--projection-limits',type=float,nargs=3,required=True,metavar=('H','C','R'))
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh output path required')
    from trottertracks.o1_m2_residual_adapter import load_saved_pair,UPDATE005
    # H4_PLAN was fixed before any load/derive/operator result.
    raw=load_saved_pair(args.core,args.input)
    limits=dict(zip(('physical','core','residual'),args.projection_limits,strict=True))
    result=evaluate(raw,plan=H4_PLAN,projection_limits=limits)
    result['saved_core_binding']=asdict(UPDATE005)
    paths=('scripts/validate_o1_m2_hermitian_b.py','scripts/validate_o1_m2_h4_operator.py',
        'src/trottertracks/o1_m2_residual_adapter.py','src/trottertracks/o1_m2_hermitian_b.py',
        'src/trottertracks/o1_m2_cutoff_free_pauli.py')
    result['source']=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False); stream.write('\n')
    return 0 if result['operator_checks_passed'] and result['model_change_status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
