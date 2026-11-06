"""Completion audit of saved C2-B words/results; zero synthesis/backend calls.

Preserve the initial outputs/STOP. Correct combined-disk geometry and bound
the registered two-precision nonuniform baseline by a rational LP relaxation.
"""
from fractions import Fraction as F
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from . import track_c2_b_pilot as p
from .track_c_h4 import save_json


def lower_native(qc,allocation,records,role):
    """Materialize native operations using cached words, preserving phase.

    Native chronological gates are replaced by written-product words in
    reverse order. Negative RZ uses the actual adjoint of the positive word.
    The return basis is obtained with .inverse(), never another synthesis.
    """
    out=QuantumCircuit(qc.num_qubits);out.global_phase=qc.global_phase
    choices={(x['role'],x['angle']):x['exponent'] for x in allocation}
    for inst in qc.data:
        qs=[qc.find_bit(q).index for q in inst.qubits];op=inst.operation
        if op.name=='rz':
            theta=float(op.params[0]);r=records[(abs(theta).hex(),choices[(role,abs(theta).hex())])]
            local=QuantumCircuit(1)
            for g in reversed(r['sequence']):
                if g=='W':local.global_phase+=math.pi/4
                elif g=='t':local.tdg(0)
                else:getattr(local,{'H':'h','T':'t','S':'s','X':'x'}[g])(0)
            if theta<0:local=local.inverse()
            out.compose(local,qubits=qs,inplace=True)
        elif op.name=='sx':out.h(qs[0]);out.s(qs[0]);out.h(qs[0])
        elif op.name=='x':out.x(qs[0])
        elif op.name=='cx':out.cx(*qs)
        else:raise ValueError(op.name)
    return out


def synthetic_checks(records):
    from pygridsynth.domega_unitary import DOmegaUnitary as U
    inverse={'H':'H','T':'t','t':'T','S':'SSS','X':'X','W':'WWWWWWW'}
    sampled=[v for v in records.values() if v['T']>0][:8]
    for row in sampled:
        word=row['sequence'];adj=''.join(inverse[g] for g in reversed(word))
        assert U.from_gates((word+adj).replace('t','TTTTTTT'))==U.identity()
    # Test the emitter and actual inverse without acquiring or compiling angles.
    row=sampled[0];theta=float.fromhex(row['theta_hex']);qc=QuantumCircuit(2)
    qc.rz(theta,0);qc.cx(0,1);qc.rz(-theta,1)
    a=[dict(role='B',angle=theta.hex(),exponent=row['precision_exponent'])]
    built=lower_native(qc,a,records,'B');back=built.inverse();u=Operator(built).data
    unitary=np.linalg.norm(Operator(built.compose(back)).data-np.eye(4),'fro')
    counts=built.count_ops();assert counts.get('t',0)+counts.get('tdg',0)==2*row['T']
    assert unitary<1e-11
    # Full-space conjugation inequality on synthetic two-qubit blocks.
    b=QuantumCircuit(2);b.ry(.31,0);b.cx(0,1);b.rz(-.27,1)
    bt=b.copy();bt.rx(.03,0)
    B=Operator(b).data;Bt=Operator(bt).data;eps=np.linalg.norm(B-Bt,2)
    lam=np.array([-.7,-.2,.3,.9]);W=1.6;maximum=0.
    for tau in (0.,-.01,.01,.7,3.):
        d=np.diag(np.exp(-1j*tau*lam));delta=np.linalg.norm(B@d@B.conj().T-Bt@d@Bt.conj().T,2)
        bound=min(2.,abs(tau)*W)*eps
        assert delta<=bound+1e-14
        maximum=max(maximum,delta)
    return dict(exact_word_adjoint_pairs=len(sampled),emitted_T_count_matches=True,actual_basis_inverse_identity_Frobenius=unitary,
        synthetic_full_space_conjugation_cases=5,D_identity_cancels=True,maximum_synthetic_operator_difference=maximum,
        new_synthesis_calls=0,new_scientific_compiles=0,new_physical_conditions=0)


def allocation_lp_lower(items,budget,records):
    total=sum((F(x['weight']) for x in items),F(0));e=p.exponent(budget,total)
    cost=sum(x['count']*records[(x['angle'],e)]['T'] for x in items)
    room=budget-total*F(10)**e;offers=[]
    for x in items:
        saving=x['count']*(records[(x['angle'],e)]['T']-records[(x['angle'],e+1)]['T'])
        extra=F(x['weight'])*9*F(10)**e
        if saving>0 and extra>0:offers.append((F(saving)/extra,saving,extra))
    answer=F(cost)
    for ratio,saving,extra in sorted(offers,reverse=True):
        take=min(room,extra);answer-=ratio*take;room-=take
        if room==0:break
    return math.ceil(answer)


def run():
    root=p.OUT;verified=root/'verified_final';verified.mkdir(exist_ok=True)
    assert not (verified/'STOP.json').exists()
    requests=json.loads((root/'PRECISION_REQUESTS.json').read_text());records={}
    # Positive sums/products creating compiler allowances have fewer than 4096
    # rounded operations. Inflate their stored centres by gamma_4096, then
    # evaluate geometry and shot ceilings with directed 192-bit intervals.
    from .track_c_compiler_a_18round import I,upper
    u=F(1,2**53);factor=F(1)/(1-4096*u)
    for case in requests['cases']:
        x=F.from_float(case['additional_raw_compiler_disk'])*factor
        case['additional_raw_compiler_disk']=math.nextafter(float(x),math.inf)
    for path in sorted((root/'synthesis').glob('*.json')):
        r=json.loads(path.read_text());assert r['status']=='PASS'
        assert hashlib.sha256(r['sequence'].encode()).hexdigest()==r['sequence_sha256']
        assert F(r['strict_error_upper'])<=F(10)**r['precision_exponent']
        assert r['T']==sum(r['sequence'].count(g) for g in ('T','t'))
        records[(r['theta_hex'],r['precision_exponent'])]=r
    checks=synthetic_checks(records);save_json(verified/'TECHNICAL_CHECKS.json',checks)
    # Finish's geometry uses arcsin of the combined disk (one disk, one phase conversion).
    p.OUT=verified;p.finish(requests['cases'],records,time.monotonic())
    result=json.loads((verified/'RESULTS.json').read_text());certs=[]
    for nonuniform in (False,True):
        fixed=[]
        for m in range(18):
            cc=[x for x in requests['cases'] if not x['structural'] and x['m']==m]
            bounds=[]
            for case in cc:
                budget=F(case['budget']);items=case['items']
                lower=allocation_lp_lower(items,budget,records) if nonuniform else p.allocate(items,budget,records,False)['T']
                phase=p.phase_disk_upper(upper(I(case['synthesis_raw_disk_cap'])+I(case['additional_raw_compiler_disk'])),case['rho_base'])
                rho=case['rho_base']-case['synthesis_raw_disk_cap']-case['additional_raw_compiler_disk'];beta=case['beta_base']-phase
                bounds.append(2*p.shots(rho,beta,1/720)*lower)
            fixed.append(min(bounds))
        certs.append(dict(baseline='gate_sum_'+('nonuniform' if nonuniform else 'uniform'),
            total_T_uniform_alpha_lower_bound=sum(fixed),bound_scope='registered per-round cap options and two per-key precisions; arbitrary feasible discrete selection; uniform alpha'))
    candidate=result['structural_nonuniform']['total_T_uniform_alpha']
    strongest=certs[1]['total_T_uniform_alpha_lower_bound']
    checks['saved_strict_word_records']=len(records)
    checks['all_alpha_exact_budgets_pass']=all(2*sum((F.from_float(x['alpha_per_axis']) for x in v['rounds']),F(0))<=F('.05') for v in result.values())
    assert checks['all_alpha_exact_budgets_pass']
    save_json(verified/'TECHNICAL_CHECKS.json',checks)
    save_json(verified/'BASELINE_BOUND.json',dict(certificates=certs,candidate_T_uniform_alpha=candidate,
        reduction_vs_registered_nonuniform_LP_lower=1-candidate/strongest,
        beats_registered_baseline_lower_bound=candidate<strongest,
        unrestricted_precision_optimizer='not certified; no claim beyond registered finite baseline'))
    summary=json.loads((verified/'SUMMARY.json').read_text())
    summary.update(new_synthesis_keys=7599,reused_synthesis_keys=4,new_synthesis_calls_in_verification=0,
        initial_science_wall_seconds=json.loads((root/'SUMMARY.json').read_text())['wall_seconds'],
        directed_geometry_bits=192,compiler_positive_sum_gamma_operations=4096,
        initial_summary='preserved one directory above; superseded for combined-disk phase geometry',
        beats_registered_nonuniform_LP_lower_uniform_alpha=candidate<strongest,
        verification='strict word records + exact adjoint + full-space synthetic bound + exact alpha budget')
    save_json(verified/'SUMMARY.json',summary)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
