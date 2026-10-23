#!/usr/bin/env python3
"""C backend connection tests on hard-coded artificial 1x1/2x2 matrices.

No H4/input path argument or loader. Output describes artificial checks only.
Run in this worktree's private python-flint environment.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import importlib.metadata
import importlib.util
from itertools import product
import json
from pathlib import Path
import sys

import flint
from flint import arb,acb,acb_mat,fmpq,ctx

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('rational_c_prototype',ROOT/'src/trotterlib/h4_enclosure_prototype.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def rational(q):
    q=Q(q);return fmpq(q.numerator,q.denominator)


def interval(q):
    return arb(rational(q.lo)).union(arb(rational(q.hi)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    checks=[]
    def check(value,name):
        if not value:raise AssertionError(name)
        checks.append(name)
    original_precision=ctx.prec
    with ctx.workprec(128):
        n,d=(.1).as_integer_ratio()
        check(arb(fmpq(n,d)).contains(fmpq(n,d)),'Exact binary64 rational conversion retained')
        check(not arb(fmpq(n,d)).contains(fmpq(1,10)),'Binary .1 is not silently replaced by decimal 1/10')
        E=acb_mat([[0,fmpq(1,4)],[0,0]]).exp()
        check(E.contains(acb_mat([[1,fmpq(1,4)],[0,1]])),'Nilpotent matrix exponential contains exact I+A')
        poly=acb_mat([[1]]);term=acb_mat([[1]]);x=acb_mat([[fmpq(1,3)]])
        for k in range(1,6):
            term=x*term/k;poly+=term
        wanted=sum((Q(1,3)**k/Q(c.factorial(k)) for k in range(6)),Q(0))
        check(poly[0,0].contains(rational(wanted)),'K4 finite polynomial includes degree5')
    for precision in [64,96,128]:
        with ctx.workprec(precision):
            tau=arb(fmpq(1,10));weights=[tau**k/c.factorial(k)*(1+(tau/(k+1))**2).sqrt() for k in [0,2,4]]
            normalizer=sum(weights);B=normalizer**4;logB=4*normalizer.log()
            proof=c.finite_law_prototype(Q(1,10),repetitions=4,bits=32)
            check(interval(proof['B']).contains(B) and interval(proof['log_B']).contains(logB),
                  f'Artificial finite normalization/log enclosed by exact-rational prototype at {precision} bits')
            t=arb(fmpq(1,4));E=acb_mat([[acb(0,-t),0],[0,acb(0,t)]]).exp()
            # Deliberately coarser independent enclosure: it must contain the
            # entire backend ball, not just the exact value common to both.
            rotation=c.exponential_prototype([[0,Q(-1,4)],[Q(1,4),0]],degree=10)
            check(interval(rotation[0][0]).contains(E[0,0].real) and interval(rotation[1][0]).contains(-E[0,0].imag),
                  f'Artificial quantum phase agrees with rational exponential enclosure at {precision} bits')
    with ctx.workprec(128):
        I=acb_mat([[1,0],[0,1]]);zero=acb_mat(2,2)
        components=[acb_mat([[0,-1],[-1,0]]),acb_mat([[1,0],[0,-1]])]
        probabilities=[fmpq(1,3),fmpq(2,3)]
        M=sum((components[j]*probabilities[j] for j in range(2)),zero)
        tau=arb(fmpq(1,10))
        weights=[tau**k/c.factorial(k)*(1+(tau/(k+1))**2).sqrt() for k in [0,2,4]]
        normalizer=sum(weights);factored=acb_mat(2,2);enumerated=acb_mat(2,2)
        for k,w in zip([0,2,4],weights):
            angle=(tau/(k+1)).atan();phase=(-1)**(k//2)
            factored+=(I*angle.cos()+M*acb(0,-angle.sin()))*(M**k)*(phase*w/normalizer)
            for indices in product(range(2),repeat=k+1):
                product_operator=I;probability=fmpq(1)
                for j in indices:probability*=probabilities[j]
                for j in indices[1:]:product_operator=components[j]*product_operator
                rotation=I*angle.cos()+components[indices[0]]*acb(0,-angle.sin())
                enumerated+=rotation*product_operator*(phase*w*probability/normalizer)
        x=M*acb(0,-tau);poly=I;term=I
        for k in range(1,6):term=x*term/k;poly+=term
        check((enumerated-factored).contains(zero) and (normalizer*factored-poly).contains(zero),
              'Factorized fixed finite law matches complete artificial event enumeration and degree5 polynomial')
    check(ctx.prec==original_precision,'Working-precision context restored')
    def grid_masses(cdf,bits):
        endpoints=[Q(0),*cdf];size=1<<bits
        ceil=lambda q:-(-q.numerator//q.denominator)
        return [Q(ceil(b*size)-ceil(a*size),size) for a,b in zip(endpoints,endpoints[1:])]
    check(grid_masses([Q(1,4),Q(1,2),Q(1)],4)==[Q(1,4),Q(1,4),Q(1,2)] and
          grid_masses([Q(3,10),Q(1)],4)==[Q(5,16),Q(11,16)],
          'Artificial finite-grid CDF mass includes left endpoint and excludes right endpoint')
    dist=importlib.metadata.distribution('python-flint');runtime_files={}
    for file in dist.files:
        path=Path(dist.locate_file(file)).resolve()
        if path.is_file() and path.suffix not in {'.pyc'}:
            runtime_files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    result=dict(status='PASS_C_ARTIFICIAL_BACKEND_ONLY',checks=checks,check_count=len(checks),
        python=sys.version,python_flint=flint.__version__,FLINT=flint.__FLINT_VERSION__,
        runtime_files=runtime_files,real_H4_inputs_loaded=False,real_H4_C_calls=0,
        real_H4_certificates_generated=0,original_native_completion_verified=False,
        implementation_scope='Backend and rational proof prototypes; production H4 assembly is a future implementation task.')
    with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    print(json.dumps(dict(status=result['status'],checks=len(checks),real_H4_C_calls=0)))


if __name__=='__main__':main()
