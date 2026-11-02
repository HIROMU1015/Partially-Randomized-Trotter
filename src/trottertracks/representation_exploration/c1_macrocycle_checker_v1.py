"""Saved-result audit, independent stage reconstruction and residuals, no eigensolver."""
from fractions import Fraction as F
import hashlib
import json
import math
import os
from pathlib import Path
from .c1_macrocycle_execution_v1 import ARTIFACT,OUTPUT,SCOPE,THREADS
from .c1_macrocycle_algebra_v1 import nested_error
from .c1_macrocycle_contract_v1 import substitute,expectation,l1


def audit(root):
    root=Path(root); out=root/OUTPUT
    receipt=json.loads((out/'launch_receipt.json').read_text())
    if receipt['approval']['scope_id']!=SCOPE or not receipt['attempt_consumed']:
        raise ValueError('AUTHORIZED_SAVED_ATTEMPT_REQUIRED')
    for name in THREADS: os.environ[name]='1'
    import numpy as np
    complete=json.loads((out/'completion.json').read_text())
    seal=(out/'selector_seal.json').read_bytes()
    if hashlib.sha256(seal).hexdigest()!=complete['selector_seal_SHA256']: raise ValueError('SEAL_HASH')
    selected=json.loads(seal)
    inputs=json.loads((root/ARTIFACT/'inputs.json').read_text())['inputs']
    comparison=json.loads((out/'comparisons.json').read_text())
    plan=json.loads((root/ARTIFACT/'pilot_contract.json').read_text())
    P={'I':np.identity(2,dtype=complex),'X':np.array([[0,1],[1,0]],complex),
       'Y':np.array([[0,-1j],[1j,0]],complex),'Z':np.array([[1,0],[0,-1]],complex)}
    def decode(x): return np.asarray(x['real'])+1j*np.asarray(x['imag'])
    def close(a,b,tol=1e-10):
        if not np.allclose(a,b,rtol=0,atol=tol): raise ValueError('NUMERICAL_MISMATCH')
    orders=('ABC','ACB','BAC')
    # Independent stage registry, no generator/macrocycle/cost imports.
    def stages(mid):
        if mid.startswith('M'):
            head,num,den=mid.split('_'); j,k=int(head[1]),int(head[2]); a=F(int(num),int(den))
            blocks=((j,a),(k,1-2*a),(j,a))
        elif mid.startswith('S4'):
            j=int(mid[-1]); p=1/(4-4**(1/3)); blocks=tuple((j,a) for a in (p,p,1-4*p,p,p))
        else:
            j=int(mid[1]); m=int(mid[-1]); blocks=tuple((j,F(1,m)) for _ in range(m))
        raw=[]
        for j,a in blocks:
            x,y,z=orders[j]
            raw.extend(((x,a/2),(y,a/2),(z,a),(y,a/2),(x,a/2)))
        return raw
    cases={}; probes_checked=0; costs_checked=0
    for row in inputs:
        case=json.loads((out/(row['id']+'.json')).read_text()); cases[row['id']]=case
        expected={f'S{j}_m{m}' for j in range(3) for m in range(1,5)}
        expected|={f'S4_{j}' for j in range(3)}
        expected|={f'M{j}{k}_{num}_{den}' for j in range(3) for k in range(3) if j!=k
                    for num,den in ((1,8),(1,4),(1,3),(3,8))}
        if {(r['method'],r['h_index']) for r in case['probes']}!={(m,h) for m in expected for h in range(6)} or len(case['probes'])!=234:
            raise ValueError('FIXED_QUERY_COVERAGE')
        if {(r['id'],r['h_index']) for r in case['C2']}!={(f'C2_{j}',h) for j in range(3) for h in range(6)} or len(case['C2'])!=18:
            raise ValueError('C2_COVERAGE')
        if case['sealed_selector']!=selected[row['id']]: raise ValueError('SELECTOR_CHANGED')
        fragments={n:{p:F(c) for p,c in terms.items()} for n,terms in row['fragments'].items()}
        gammaA=F(row['diagonal_gap']); v=l1(fragments['B'])+l1(fragments['C'])
        coefficient_Lambda=sum(l1(x) for x in fragments.values())
        if F(case['coefficient_Lambda'])!=coefficient_Lambda: raise ValueError('COEFFICIENT_LAMBDA')
        D=[substitute(nested_error(tuple(order)),fragments) for order in orders]
        for price in plan['CX_prices']:
            for state in ('00','11'):
                selector=selected[row['id']][str(price)]['states'][state]
                distance=min(F(2),2*v/(gammaA-v)) if state=='00' and 2*v<gammaA else F(2)
                if F(selector['information']['d_rho'])!=distance: raise ValueError('STATE_INFORMATION')
                expected_gap=str(gammaA-2*v) if state=='00' and 2*v<gammaA else None
                if selector['information']['gamma_lower']!=expected_gap: raise ValueError('COEFFICIENT_GAP')
                for candidate in selector['candidates']:
                    mid=candidate['id']
                    if mid.startswith('M'):
                        head,num,den=mid.split('_'); j,k=int(head[1]),int(head[2]); a=F(int(num),int(den))
                        poly={p:2*a**3*D[j].get(p,0)+(1-2*a)**3*D[k].get(p,0) for p in set(D[j])|set(D[k])}
                        box_norm=2*a**3*l1(D[j])+(1-2*a)**3*l1(D[k])
                    else:
                        j,m=int(mid[1]),int(mid[-1]);poly={p:c/F(m*m) for p,c in D[j].items()};box_norm=l1(poly)
                    d=expectation(poly,state); width=abs(d)+distance*l1(poly); box=abs(d)+distance*box_norm
                    if [F(candidate[k]) for k in ('d_approx','D_l1','joint_width','box_width')]!=[d,l1(poly),width,box]:
                        raise ValueError('INDEPENDENT_BCH_OR_WIDTH')
                    counts=candidate['counts']; g=counts['RZ']+counts['basis_Clifford']+price*counts['CX']
                    if [F(candidate['J2_'+k]) for k in ('empirical','joint','box')]!=[g*g*abs(d),g*g*width,g*g*box]:
                        raise ValueError('SELECTOR_SCORE')
                pools={'best_single':lambda x:x['kind']=='SINGLE',
                       'uniform_order_mixture':lambda x:x['known_uniform_mixture'],
                       'C1':lambda x:True,'strong_known_composition':lambda x:True}
                for score in ('empirical','joint','box'):
                    for label,pred in pools.items():
                        chosen=min((r for r in selector['candidates'] if pred(r)),key=lambda r:(F(r['J2_'+score]),r['id']))['id']
                        if selector['choices'][score][label]!=chosen: raise ValueError('SEALED_SELECTION')
                expected_precision={}
                for eps in (F(1,100000),F(1,1000000)):
                    expected_precision[str(eps)]={}
                    for candidate in selector['candidates']:
                        valid=[]
                        if expected_gap:
                            gamma=F(expected_gap); dn=F(candidate['D_l1']); width=F(candidate['joint_width'])
                            for index in range(6):
                                h=F(1)/(8*coefficient_Lambda*2**index)
                                bound=h*h*width+h**4*(F(2,3)*(4*coefficient_Lambda)**5+2*dn*dn/gamma)
                                if h*h*dn<=gamma/4 and bound<=eps/2: valid.append(index)
                        expected_precision[str(eps)][candidate['id']]=min(valid) if valid else None
                if expected_precision!=selector['coefficient_only_precision_choices']:
                    raise ValueError('ORACLE_FREE_PRECISION_CHOICES')
        A={name:{word:float(F(c))*np.kron(P[word[0]],P[word[1]]) for word,c in terms.items()}
           for name,terms in row['fragments'].items()}
        H=sum(mat for terms in A.values() for mat in terms.values()); close(H,decode(case['H']))
        E=np.asarray(case['reference_energies']); V=decode(case['reference_vectors'])
        close(H@V,V*E); close(V.conj().T@V,np.eye(4))
        if any(E[i]>E[i+1] for i in range(3)): raise ValueError('REFERENCE_SORT')
        ds=[]
        for poly in D:
            mat=sum((float(c)*np.kron(P[word[0]],P[word[1]]) for word,c in poly.items()),np.zeros((4,4),complex))
            ds.append(float(np.vdot(V[:,0],mat@V[:,0]).real))
        close(ds,case['true_kernel_d_diagnostic'])
        lookup={}
        for r in case['probes']:
            lookup[r['method'],r['h_index']]=r
            Lam=sum(abs(float(F(c))) for terms in row['fragments'].values() for c in terms.values())
            close(r['h'],1/(plan['h_denominator']*Lam*2**r['h_index']))
            U=np.eye(4,dtype=complex)
            for name,a in stages(r['method']):
                # Unmerged word reconstructs the saved merged unitary.
                for word,c in row['fragments'][name].items():
                    t=r['h']*float(a)*float(F(c)); Q=np.kron(P[word[0]],P[word[1]])
                    U=U@(np.cos(t)*np.eye(4)-1j*np.sin(t)*Q)
            close(U,decode(r['unitary']))
            val=decode(r['eigenvalues']); vec=decode(r['eigenvectors'])
            close(U@vec,vec*val); close(vec.conj().T@vec,np.eye(4),1e-9)
            close(-np.angle(val)/r['h'],r['quasienergies'])
            connection=np.abs(V[:,0].conj()@vec)**2; close(connection,r['truth_connection'])
            if r['branch']!=int(np.argmax(connection)): raise ValueError('BRANCH_CONNECTION')
            close(r['bias'],r['quasienergies'][r['branch']]-E[0])
            mid=r['method']
            if mid.startswith('M'):
                head,num,den=mid.split('_'); a=float(F(int(num),int(den)))
                d=2*a**3*ds[int(head[1])]+(1-2*a)**3*ds[int(head[2])]
            elif mid.startswith('S4'): d=0.0
            else: d=ds[int(mid[1])]/int(mid[-1])**2
            close(d,r['true_leading_d_diagnostic'])
            close(r['bias']-r['h']**2*d,r['finite_h_remainder_diagnostic'])
            for state,index in (('00',0),('11',3)):
                close(r['approximate_overlap'][state],abs(vec[index,r['branch']])**2)
            residual=float(np.linalg.norm(U@vec-vec*val)); orth=float(np.linalg.norm(vec.conj().T@vec-np.eye(4)))
            unitary=float(np.linalg.norm(U.conj().T@U-np.eye(4)))
            close([residual,orth,unitary],[r['residual'],r['orthogonality'],r['unitarity']])
            safe=(r['branch']==0 and connection[r['branch']]>=.5 and residual<1e-10 and
                  orth<1e-9 and unitary<1e-10 and max(abs(np.angle(val)))<math.pi/4)
            if safe!=r['diagnostic_branch_valid']: raise ValueError('BRANCH_FLAG')
            names=[x[0] for x in stages(r['method'])]
            merged=[x for i,x in enumerate(names) if i==0 or names[i-1]!=x]
            if len(merged)>1 and merged[0]==merged[-1]: merged=merged[:-1]
            rz=cx=basis=0
            for name in merged:
                for word in row['fragments'][name]:
                    support=len(word.replace('I','')); rz+=2 if support else 1; cx+=4*support-2 if support else 0
                    basis+=2*word.count('X')+4*word.count('Y')
            if (r['counts']['RZ'],r['counts']['CX'],r['counts']['basis_Clifford'])!=(rz,cx,basis): raise ValueError('CONTROLLED_COUNTS')
            probes_checked+=1
        for r in case['C2']:
            a,b=[lookup[m,r['h_index']] for m in r['components']]
            close(r['bias'],(4*b['quasienergies'][b['branch']]-a['quasienergies'][a['branch']])/3-E[0])
            if r['counts']!=[a['counts'],b['counts']] or r['overlaps']!=[a['approximate_overlap'],b['approximate_overlap']]:
                raise ValueError('EXTRAPOLATION_COMPONENTS')
            if r['diagnostic_branch_valid']!=(a['diagnostic_branch_valid'] and b['diagnostic_branch_valid']):
                raise ValueError('EXTRAPOLATION_BRANCH')
    # Compare every cost/eligibility/frontier without calling the producer's summary routine.
    if len(comparison)!=32 or len({(r['input_id'],r['state'],r['price'],r['epsilon']) for r in comparison})!=32:
        raise ValueError('SUMMARY_COVERAGE')
    for summary in comparison:
        case=cases[summary['input_id']]; state=summary['state']; price=summary['price']; eps=summary['epsilon']
        stat=pf=eps/2
        exact=selected[summary['input_id']][str(price)]['states'][state]
        gamma=exact['information']['gamma_lower']; gamma=F(gamma) if gamma is not None else None
        exact_rows={r['id']:r for r in exact['candidates']}
        precision={label:{'method':mid,'h_index':exact['coefficient_only_precision_choices'][str(F(str(eps)))][mid]}
                   for label,mid in exact['choices']['joint'].items()}
        if summary['coefficient_only_precision_choices']!=precision: raise ValueError('PRECISION_POINTS')
        probe={(r['method'],r['h_index']):r for r in case['probes']}
        ext={(r['id'],r['h_index']):r for r in case['C2']}
        for row in summary['rows']:
            key=row['method'],row['h_index']
            if key in probe:
                r=probe[key]; pop=r['approximate_overlap'][state]; g=r['counts']['RZ']+r['counts']['basis_Clifford']+price*r['counts']['CX']
                value=(g/r['h']/stat+state.count('1'))/pop if pop else None
                ok=r['diagnostic_branch_valid'] and abs(r['bias'])+1e-10<=pf and pop>=plan['minimum_overlap']
                bound=None
                if gamma and row['method'] in exact_rows:
                    candidate=exact_rows[row['method']]; dn=F(candidate['D_l1'])
                    Lam=F(case['coefficient_Lambda']); h=F(1)/(8*Lam*2**r['h_index'])
                    if h*h*dn<=gamma/4:
                        bound=h*h*F(candidate['joint_width'])+h**4*(F(2,3)*(4*Lam)**5+2*dn*dn/gamma)
                if bound is None:
                    if row['certified_PF_bound'] is not None: raise ValueError('FALSE_CERTIFICATE')
                else:
                    if bound!=F(row['certified_PF_bound']): raise ValueError('EXACT_BOUND')
            else:
                r=ext[key]; pops=[x[state] for x in r['overlaps']]
                aa=[(x['RZ']+x['basis_Clifford']+price*x['CX'])/r['h']/p if p else None for x,p in zip(r['counts'],pops)]
                value=(math.sqrt(aa[0]/3)+math.sqrt(4*aa[1]/3))**2/stat+sum(state.count('1')/p for p in pops) if all(pops) else None
                ok=r['diagnostic_branch_valid'] and abs(r['bias'])+1e-10<=pf and min(pops)>=plan['minimum_overlap']
            if value is None:
                if row['conditional_quantum_cost'] is not None: raise ValueError('NULL_COST')
            else: close(value,row['conditional_quantum_cost'],max(1e-10,abs(value)*1e-12))
            if ok!=row['eligible']: raise ValueError('ELIGIBILITY')
            costs_checked+=1
        choices=selected[summary['input_id']][str(price)]['states'][state]['choices']
        for score,group in choices.items():
            for label,mid in group.items():
                candidates=[r for r in summary['rows'] if r['eligible'] and r['method']==mid]
                best=min(candidates,key=lambda r:r['conditional_quantum_cost']) if candidates else None
                if summary['frontiers'][score+':'+label]!=best: raise ValueError('FRONTIER')
        for label,predicate in (
            ('all_single_subdivision',lambda x:x['method'].startswith('S') and not x['method'].startswith('S4')),
            ('S4',lambda x:x['method'].startswith('S4')),
            ('C2',lambda x:x['method'].startswith('C2'))):
            candidates=[r for r in summary['rows'] if r['eligible'] and predicate(r)]
            best=min(candidates,key=lambda r:r['conditional_quantum_cost']) if candidates else None
            if summary['frontiers']['oracle_diagnostic:'+label]!=best: raise ValueError('BASELINE_FRONTIER')
    if complete['eigensolver_calls']!=940 or sum(c['eigensolver_calls'] for c in cases.values())!=940:
        raise ValueError('EIGENSOLVER_CALLS')
    return {'status':'SAVED_RESIDUAL_AND_COST_AUDIT_PASS','unitary_probes_checked':probes_checked,
            'comparison_rows_checked':costs_checked,'new_eigensolver_calls':0,
            'external_reproduction':False,'formal_novelty_claim':False,'mandatory_stop':True}
