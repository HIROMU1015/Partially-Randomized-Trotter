"""Post-run Q0 checks, including an independent sine-preimage enumeration."""
import argparse
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q0',ROOT/'scripts/diagnose_pr_reference_q0.py')
q=importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
q.I=q.setup()


def overlap(a,b):
    return max(q.endpoints(a)[0],q.endpoints(b)[0]) <= min(q.endpoints(a)[1],q.endpoints(b)[1])


def unpack(item):
    a,b=(F(x) for x in item['exact_interval'])
    return q.I([q.endpoints(q.iv(a))[0],q.endpoints(q.iv(b))[1]])


def sine_preimage(l,h,rho,D,U,offset):
    """General sine inverse with BOTH branches; independent of q.hull().

    The possible sine values are the interval hull of mu/rho for the entire
    rectangle. Its endpoints use both radius endpoints, without a sign case.
    Period -1,0,1 covers the registered domains, including Pauli offsets.
    """
    low=q.minimum(l/rho,l)
    high=q.maximum(h/rho,h)
    if q.endpoints(low)[0]>1 or q.endpoints(high)[1]<-1:
        return None
    a,b=q.asin(low),q.asin(high)
    intervals=[]
    for j in (-1,0,1):
        for left,right in ((a+2*j*q.mp.iv.pi,b+2*j*q.mp.iv.pi),
                           (q.mp.iv.pi-b+2*j*q.mp.iv.pi,q.mp.iv.pi-a+2*j*q.mp.iv.pi)):
            lo=q.maximum(-U,left-offset-D)
            hi=q.minimum(U,right-offset+D)
            if q.endpoints(lo)[0] <= q.endpoints(hi)[1]:
                intervals.append((lo,hi))
    if not intervals:
        return None
    lo=q.minimum(*(x[0] for x in intervals))
    hi=q.maximum(*(x[1] for x in intervals))
    return q.maximum(q.I(0),hi-lo)


def full_preimage(n,rho,D,U):
    eps=q.epsilon(n)
    widths=[]
    for k in range(n+1):
        center=q.iv(F(2*k,n)-1)
        item=sine_preimage(center-eps,center+eps,rho,D,U,q.I(0))
        if item is not None:
            widths.append(item)
    return q.maximum(*widths)


def stronger_xy_witness(rho,B,total):
    """Exclude ALL X/Y allocations up to total for the same CI-hull task.

    Center pi/4; two equal-radius signals at center +/- a, a=.4-B+.01.
    Their residual phases are +/- .41. Each coordinate differs from their
    common midpoint by delta=rho*sin(a)/sqrt(2). Give EACH axis the entire
    m17 failure allowance, an optimistic relaxation of every legal split.
    A nearest count mean lies within 1/n of the midpoint. If eps(n)-1/n
    exceeds delta for every n<=total, both signals fit that count's CI for
    ANY allocation, even with history-aware allocation at this center.
    """
    a=q.iv(F(2,5))-B+q.iv(F(1,100))
    delta=rho*q.mp.iv.sin(a)/q.mp.iv.sqrt(q.I(2))
    gaps=[q.epsilon(n)-q.iv(F(1,n))-delta for n in range(1,total+1)]
    gap=q.minimum(*gaps)
    assert q.endpoints(gap)[0]>0
    midpoint=rho*q.mp.iv.cos(a)/q.mp.iv.sqrt(q.I(2))
    return {'status':'ALL_XY_ALLOCATIONS_UP_TO_Q_N_FAIL_CI_HULL_CRITERION',
            'total_shots_through':total,'center_phase':'pi/4',
            'residual_pair':['-41/100','41/100'],'pair_diameter':'41/50',
            'target_diameter':'4/5','coordinate_midpoint':q.pack(midpoint),
            'half_coordinate_separation':q.pack(delta),
            'minimum_eps_minus_lattice_error_minus_separation':q.pack(gap),
            'all_axis_counts_checked':total,
            'each_axis_given_full_1_over_360_optimistically':True,
            'history_aware_integer_allocations_covered':True,
            'not_information_theoretic_or_all_confidence_methods_lower_bound':True}


def check(folder):
    source=json.loads((folder/'SOURCE.json').read_text())
    result=json.loads((folder/'RESULTS.json').read_text())
    inputs=json.loads((folder/'INPUTS.json').read_text())
    assert source['script_sha256']==hashlib.sha256((ROOT/q.SELF).read_bytes()).hexdigest()
    for r in source['inputs']:
        assert hashlib.sha256(q.raw(r['commit'],r['path'])).hexdigest()==r['sha256']
    assert inputs['target']==[17,131072,32,88]
    assert F(inputs['prefix_failure'])+F(inputs['m17_failure'])==F(inputs['total_failure'])==F(1,20)
    assert inputs['t17_over_t16']==2
    assert result['new_circuit_compile_sampling_RPE_total_price_synthesis_calls']==0
    checks=[]
    for arm in result['arms']:
        row=inputs['arms'][arm['arm']]
        rho=q.I(row['raw_radius_lower'])
        beta=q.I(row['statistical_phase_lower'])
        B=q.iv(F(2,5))-beta
        U,target=q.iv(F(4,5)),q.iv(F(4,5))
        assert overlap(B,unpack(arm['systematic_phase_upper']))
        assert [r['n'] for r in arm['scan']]==list(range(1,1253))
        selected=[r['n'] for r in arm['scan'] if r['status']=='PASS']
        assert selected==arm['passing_n'] and min(selected)==arm['minimum_one_direction_n']
        assert not any(r['status']=='INDETERMINATE' for r in arm['scan'])
        independent=[]
        for n in (arm['minimum_one_direction_n']-1,arm['minimum_one_direction_n'],1252):
            width=full_preimage(n,rho,B,U)
            saved=arm['scan'][n-1]
            assert overlap(width,unpack(saved['worst_phase_diameter']))
            assert q.verdict(width,target)==saved['status']
            independent.append({'n':n,'all_counts':n+1,'width':q.pack(width),'status':saved['status']})
        for n,item in arm['feedback_margins'].items():
            if item.get('status')=='BRACKETED':
                lo,hi=map(F,item['bracket'])
                # Direct all-count preimages verify each boundary, without
                # calling the production envelope or bisection again.
                a=full_preimage(int(n),rho,B+q.iv(lo),U)
                b=full_preimage(int(n),rho,B+q.iv(hi),U)
                assert q.verdict(a,target)=='PASS' and q.verdict(b,target)=='FAIL'
        pc=arm['single_pauli_counterexample']
        low,high=map(unpack,pc['residuals'])
        mu1=q.mp.iv.sin(low+q.mp.iv.pi/4+B)
        mu2=rho*q.mp.iv.sin(high+q.mp.iv.pi/4-B)
        assert overlap(mu1,mu2) and overlap(mu1,rho)
        assert q.verdict(high-low,target)=='FAIL'
        xy=arm['xy_baseline']
        feasible=xy['boundary_totals'][-1]
        assert feasible['status']=='PASS'
        nx,ny=feasible['best_n']
        ax,ay=map(F,feasible['alpha'])
        assert nx+ny==xy['minimum_certified_total'] and ax+ay==F(1,360)
        lhs=q.epsilon(nx,ax)**2+q.epsilon(ny,ay)**2
        rhs=(rho*q.mp.iv.sin(beta))**2
        assert q.verdict(lhs,rhs)=='PASS'
        strong=stronger_xy_witness(rho,B,arm['minimum_one_direction_n'])
        checks.append({'arm':arm['arm'],'independent_all_count_preimages':independent,
                       'feedback_brackets_verified_all_counts':True,
                       'Pauli_equal_mean_counterexample':True,
                       'optimized_XY_feasible_allocation_verified':True,
                       'stronger_XY_CI_hull_lower_bound':strong})
    assert 'qiskit' not in q.sys.modules
    output={'status':'Q0_POSTRUN_CHECKS_PASS','checks':checks,
            'runner_and_fixed_inputs_unchanged':True,'failure_sum_exact':True,
            'new_quantum_scientific_calls':0,'campaign_not_rerun':True,
            'author_technical_check_not_independent_review':True,
            'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'raw_result_sha256':hashlib.sha256((folder/'RESULTS.json').read_bytes()).hexdigest()}
    (folder/'CHECKS.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(output['status'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    check(parser.parse_args().folder)
