"""Synthetic-only precision benchmark; no molecular inputs or circuit builds."""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from trotterlib import b2_strong_pf_precision as hp
from trotterlib import b2_p1_strong_pf_h4_diagnosis as diagnostic
from trotterlib.df_hamiltonian import PhysicalSector
from trotterlib.h2_non_df_pilot import reverse_bits


def artificial_input():
    one=np.kron(np.array([[.4,.13j],[-.13j,-.2]]),np.eye(2))
    factors=(np.kron(np.array([[1.,.17j],[-.17j,-.3]]),np.eye(2)),
             np.kron(np.array([[.3,.21],[.21,.6]]),np.eye(2)))
    sector=PhysicalSector.spin_sector(n_qubits=4,nelec_alpha=1,nelec_beta=1)
    indices=reverse_bits(4)[sector.basis_indices]
    psi=np.array([.5,.5j,-.5,.5j])
    return one,(.7,-.35),factors,.371,indices,psi


class IndependentReference(hp.ExponentialArithmetic):
    """Explicit Pauli-JW generators + many-body spectral exponentials.

    Never uses native orbitals/Fock minors, production CAR bilinears, or their
    output operators. Shared composition is the unchanged PF schedule helper.
    """
    def __init__(self,ctx,one,weights,factors,constant,indices):
        super().__init__(ctx);self.cache={};self.spectral_cache={};self.constant=hp.exact_real(ctx,constant)
        n=len(one);eye=np.eye(2,dtype=object);z=np.diag([1,-1]).astype(object)
        lower=np.array([[0,1],[0,0]],dtype=object);ann=[]
        for mode in range(n):
            a=np.array([[1]],dtype=object)
            for q in reversed(range(n)):
                a=np.kron(a,lower if q==mode else z if q<mode else eye)
            ann.append(ctx.matrix(a.tolist()))
        def lift(original):
            a=hp.hp_helpers().mp_matrix(ctx,original);out=ctx.matrix(1<<n)
            for p in range(n):
                for q in range(n):
                    if a[p,q]:out+=a[p,q]*(ann[p].H*ann[q])
            return out
        full=[lift(one)]
        for w,g in zip(weights,factors,strict=True):
            b=lift(g);full.append(hp.exact_real(ctx,w)*(b*b))
        self.generators=tuple(np.asarray([[a[int(i),int(j)] for j in indices] for i in indices],dtype=object) for a in full)

    def exponential(self,time,generator):
        ctx=self.ctx;key=tuple(np.asarray(generator).ravel())
        if key not in self.spectral_cache:
            a=ctx.matrix(np.asarray(generator).tolist())
            if a!=a.H:raise ValueError('REFERENCE_NOT_EXACT_HERMITIAN')
            self.spectral_cache[key]=ctx.eighe(a)
        values,u=self.spectral_cache[key]
        t=hp.exact_real(ctx,Fraction(time))
        return np.asarray((u*ctx.diag([ctx.exp(-ctx.j*t*x) for x in values])*u.H).tolist(),dtype=object)

    def step(self,schedule,delta):
        ctx=self.ctx
        out=hp._ordered_exponentials(self.generators,schedule,self.cache,arithmetic=self)
        return ctx.exp(-ctx.j*self.constant*hp.exact_real(ctx,Fraction(delta)))*ctx.matrix(out.tolist())


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    one,weights,factors,constant,indices,psi64=artificial_input()
    from qiskit import QuantumCircuit
    def forbidden(*args,**kwargs):raise AssertionError('Synthetic-only benchmark')
    records=[]
    with ExitStack() as stack:
        for target,name in ((np,'load'),(hp.legacy.p,'load_private_inputs'),(hp.legacy.p,'public_json'),
            (hp.legacy.saved,'fixed_json'),(QuantumCircuit,'__init__'),
            (hp.legacy.strong.StrongBuilder,'sequence'),(hp.legacy.p.connection.accounting,'_hoeffding_shots')):
            stack.enter_context(patch.object(target,name,forbidden))
        preps={d:hp.prepare_precision(one,weights,factors,constant,digits=d,basis_indices=indices) for d in (80,120)}
        refs={d:IndependentReference(preps[d].ctx,one,weights,factors,constant,indices) for d in preps}
        ctx=preps[120].ctx
        model,_=hp.legacy.strong.prepare_xdf_basis(one,weights,factors,constant,
            source_identity='fixed_artificial_input',input_identity='fixed_artificial_input',basis_policy=hp.legacy.strong.SPIN)
        car=hp.legacy.p.car_reference();ann=car.annihilators(4);ix=np.ix_(indices,indices)
        gens64=[car.one_body(one,ann).toarray()[ix]]
        gens64.extend(w*(car.one_body(g,ann)@car.one_body(g,ann)).toarray()[ix] for w,g in zip(weights,factors,strict=True))
        bases={};relatives={};exps={};psi=hp.hp_helpers().mp_matrix(ctx,psi64[:,None])
        for formula,s in hp.legacy.ARMS:
            pairs={d:hp.precise_step(preps[d],formula=formula,substeps=s) for d in preps}
            schedule=pairs[120].schedule
            ref={d:refs[d].step(schedule,.02) for d in refs}
            n64,_=hp.legacy.native_mean_step(model,schedule,car,indices,basis_cache=bases,relative_cache=relatives)
            c64=np.exp(-1j*constant*.02)*hp.legacy._ordered_exponentials(gens64,schedule,exps)
            methods=dict(native64=hp.hp_helpers().mp_matrix(ctx,n64),CAR64=hp.hp_helpers().mp_matrix(ctx,c64),
                native80=diagnostic.mp_copy(ctx,pairs[80].native),CAR80=diagnostic.mp_copy(ctx,pairs[80].car),
                native120=pairs[120].native,CAR120=pairs[120].car)
            errors={k:hp.maxabs(a-ref[120]) for k,a in methods.items()}
            convergence=hp.maxabs(diagnostic.mp_copy(ctx,ref[80])-ref[120])
            assert convergence<=ctx.mpf(hp.legacy.p.MATCH)
            assert errors['native80']<errors['native64'] and errors['CAR80']<errors['CAR64']
            assert all(errors[k]<=ctx.mpf(hp.legacy.p.MATCH) for k in ('native80','CAR80','native120','CAR120'))
            base={}
            reference_signal=diagnostic.hp_expectation(ref[120],psi)
            for k,a in methods.items():
                sig=diagnostic.hp_expectation(a,psi)
                base[k]=dict(matrix_error=diagnostic.hp_text(ctx,errors[k]),unitarity=diagnostic.hp_text(ctx,hp.unitarity(a)),
                    trace_relative_phase_rad=diagnostic.hp_text(ctx,ctx.arg(ctx.fsum((ref[120].H*a)[i,i] for i in range(a.rows)))),
                    signal=diagnostic.hp_signal_record(ctx,sig),signal_difference=diagnostic.hp_difference(ctx,sig,reference_signal))
            long={};q=2**17;reference_power=ref[120]**q;reference_long=diagnostic.hp_expectation(reference_power,psi)
            for k,a in methods.items():
                power=a**q
                sig=diagnostic.hp_expectation(power,psi)
                long[k]=dict(matrix_error=diagnostic.hp_text(ctx,hp.maxabs(power-reference_power)),
                    signal=diagnostic.hp_signal_record(ctx,sig),signal_difference=diagnostic.hp_difference(ctx,sig,reference_long))
            for k,a in (('native64',n64),('CAR64',c64)):
                power=np.linalg.matrix_power(a,q);z=complex(np.vdot(psi64,power@psi64));sig=ctx.mpc(z.real,z.imag)
                long[k+'_binary64_power']=dict(signal=hp.legacy.p.signal_record(z),
                    signal_difference=diagnostic.hp_difference(ctx,sig,reference_long))
            components={str(d):[{k:diagnostic.hp_text(preps[d].ctx,v) for k,v in row.items()} for row in preps[d].diagnostics] for d in preps}
            record=dict(arm_id=formula+'_s'+str(s),formula=formula,substeps=s,status='SYNTHETIC_PRECISION_IMPROVED',
                schedule=[dict(block=i,time_fraction=str(t)) for i,t in schedule],base=base,
                reference_80_vs_120=diagnostic.hp_text(ctx,convergence),reference_signal=diagnostic.hp_signal_record(ctx,reference_signal),
                component_diagnostics=components,
                pair_diagnostics={str(d):{k:diagnostic.hp_text(preps[d].ctx,v) for k,v in pairs[d].diagnostics.items()} for d in preps},
                long_Q=q,long=long,reference_long_signal=diagnostic.hp_signal_record(ctx,reference_long))
            records.append(record);print(record['arm_id']+' SYNTHETIC_PRECISION_IMPROVED',flush=True)
    paths=['src/trotterlib/b2_strong_pf_precision.py','scripts/validate_b2_strong_pf_precision.py','tests/test_b2_strong_pf_precision.py']
    result=dict(status='SYNTHETIC_PRECISION_IMPROVED',synthetic_only=True,
        SOURCE_SHA256={p:hashlib.sha256((hp.legacy.ROOT/p).read_bytes()).hexdigest() for p in paths},
        fixture_SHA256={k:hp.legacy.p._array_hash(v) for k,v in dict(one_body=one,factor0=factors[0],factor1=factors[1],state=psi64,indices=indices).items()},
        independent_reference='Explicit tensor Pauli-JW full-space generators, then sector restriction; many-body spectral exponentials',
        precision_dps=[80,120],P1_independent_tolerance_unchanged=hp.legacy.p.MATCH,
        H4_evaluated=False,H4_historical_status='3 PASS / 5 STOP unchanged',emitter_correctness='UNASSESSED',
        fixture_description='Artificial two-spatial-orbital exact spin lift, complex hoppings, positive/negative factors, scalar .371, Nalpha=Nbeta=1; fixed binary64 input',
        methods='native/CAR construction at 80/120 digits; old binary64 construction with binary64 or HP power; no projection or phase fit',
        cases=records,environment=dict(host=platform.node(),python=platform.python_version(),numpy=np.__version__),
        elapsed_seconds=time.perf_counter()-started,mandatory_stop=True)
    (output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
