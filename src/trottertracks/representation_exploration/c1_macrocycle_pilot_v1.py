"""Future numerical pilot. Explicit launch receipt precedes any numerical work."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction as F
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import resource
import time
from .c1_macrocycle_algebra_v1 import REPRESENTATIVES,kernel,kernel_error,canonical_stages
from .c1_macrocycle_contract_v1 import validate_input,select,methods,native_counts,l1,analytic_bound,substitute
from .c1_macrocycle_execution_v1 import ARTIFACT,OUTPUT,check_approval,sha


def pack(a): return {'real':a.real.tolist(),'imag':a.imag.tolist()}


def case_job(job):
    # This function is reachable only after the public runner's explicit gate.
    import numpy as np
    started=time.perf_counter(); row,plan,selected=job
    resource.setrlimit(resource.RLIMIT_AS,(2**30,2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(16*2**20,16*2**20))
    fragments=validate_input(row)
    paulis={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),
            'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
    matrices={p:np.kron(paulis[p[0]],paulis[p[1]]).astype(complex)
              for terms in fragments.values() for p in terms}
    block={n:sum(float(c)*matrices[p] for p,c in terms.items()) for n,terms in fragments.items()}
    H=sum(block.values()); energy,ground=np.linalg.eigh(H)
    true_d=[]
    for order in REPRESENTATIVES:
        poly=substitute(kernel_error(order),fragments)
        D=sum((float(c)*np.kron(paulis[word[0]],paulis[word[1]]) for word,c in poly.items()),np.zeros((4,4),complex))
        true_d.append(float(np.vdot(ground[:,0],D@ground[:,0]).real))
    Lam=sum(l1(x) for x in fragments.values())
    specs=methods()
    p=1/(4-4**(1/3))
    for j,order in enumerate(REPRESENTATIVES):
        stages=tuple((n,float(t)*a) for a in (p,p,1-4*p,p,p) for n,t in kernel(order))
        specs.append({'id':f'S4_{j}','kind':'S4','stages':stages})
    probes=[]; calls=1; cache={}
    for hi in range(plan['h_levels']):
        h=1/(plan['h_denominator']*float(Lam)*2**hi)
        for method in specs:
            U=np.eye(4,dtype=complex)
            for n,t in canonical_stages(method['stages']):
                for word,c in fragments[n].items():
                    theta=h*float(t)*float(c); key=(word,theta.hex())
                    if key not in cache:
                        cache[key]=math.cos(theta)*np.eye(4)-1j*math.sin(theta)*matrices[word]
                        if len(cache)>plan['limits']['primitive_exponential_cache_entries_per_case']:
                            raise RuntimeError('CACHE_CAP')
                    U=U@cache[key]
            vals,vec=np.linalg.eig(U); calls+=1
            E=-np.angle(vals)/h
            index=np.argsort(E); E=E[index]; vals=vals[index]; vec=vec[:,index]
            # Truth is used only here, after both state selectors have been sealed.
            connections=np.abs(ground[:,0].conj()@vec)**2; branch=int(np.argmax(connections))
            residue=float(np.linalg.norm(U@vec-vec*vals,ord='fro'))
            orth=float(np.linalg.norm(vec.conj().T@vec-np.eye(4),ord='fro'))
            unitary=float(np.linalg.norm(U.conj().T@U-np.eye(4),ord='fro'))
            safe=(branch==0 and connections[branch]>=.5 and residue<1e-10 and
                  orth<1e-9 and unitary<1e-10 and np.max(np.abs(np.angle(vals)))<math.pi/4)
            counts=native_counts(method['stages'],fragments)
            if method['kind']=='MACRO':
                a=float(method['a']); d=2*a**3*true_d[method['j']]+(1-2*a)**3*true_d[method['k']]
            elif method['kind']=='S4': d=0.0
            else: d=true_d[method['j']]/method['m']**2
            probes.append({'method':method['id'],'kind':method['kind'],'h_index':hi,'h':h,
                'unitary':pack(U),'eigenvalues':pack(vals),'eigenvectors':pack(vec),
                'quasienergies':E.tolist(),'branch':branch,'truth_connection':connections.tolist(),
                'approximate_overlap':{'00':float(abs(vec[0,branch])**2),
                                       '11':float(abs(vec[3,branch])**2)},
                'bias':float(E[branch]-energy[0]),'true_leading_d_diagnostic':d,
                'finite_h_remainder_diagnostic':float(E[branch]-energy[0])-h*h*d,
                'residual':residue,'orthogonality':orth,
                'unitarity':unitary,'diagnostic_branch_valid':bool(safe),'counts':counts})
    lookup={(r['method'],r['h_index']):r for r in probes}
    extrap=[]
    for j in range(3):
        for hi in range(plan['h_levels']):
            a,b=lookup[f'S{j}_m1',hi],lookup[f'S{j}_m2',hi]
            Ea=a['quasienergies'][a['branch']]
            # m2 eigenphase/h is the same branch energy as S(h/2).
            Eb=b['quasienergies'][b['branch']]
            extrap.append({'id':f'C2_{j}','h_index':hi,'bias':(4*Eb-Ea)/3-float(energy[0]),
                'components':[a['method'],b['method']],'weights':['-1/3','4/3'],
                'diagnostic_branch_valid':a['diagnostic_branch_valid'] and b['diagnostic_branch_valid'],
                'h':a['h'],'counts':[a['counts'],b['counts']],
                'overlaps':[a['approximate_overlap'],b['approximate_overlap']]})
    return {'input_id':row['id'],'coefficient_Lambda':str(Lam),'true_kernel_d_diagnostic':true_d,
            'H':pack(H),'reference_energies':energy.tolist(),
            'reference_vectors':pack(ground),'reference_gap':float(energy[1]-energy[0]),
            'probes':probes,'C2':extrap,'sealed_selector':selected,
            'eigensolver_calls':calls,'primitive_exponential_cache_entries':len(cache),
            'wall_seconds':time.perf_counter()-started,'max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def summaries(records,plan):
    out=[]
    for case in records:
        for price in plan['CX_prices']:
            sel=case['sealed_selector'][str(price)]['states']
            for state in ('00','11'):
                choices=sel[state]['choices']; estimates=sel[state]['candidates']
                bound={r['id']:F(r['joint_width']) for r in estimates}
                norms={r['id']:F(r['D_l1']) for r in estimates}
                gamma=sel[state]['information']['gamma_lower']
                gamma=F(gamma) if gamma is not None else None
                for epsilon in plan['energy_targets']:
                    epsilon=float(F(epsilon)); pf=stat=epsilon/2
                    rows=[]
                    for r in case['probes']:
                        g=r['counts']['RZ']+r['counts']['basis_Clifford']+price*r['counts']['CX']; pop=r['approximate_overlap'][state]
                        # Oracle feasibility frontier is a diagnostic, never a selector input.
                        eligible=r['diagnostic_branch_valid'] and abs(r['bias'])+1e-10<=pf and pop>=plan['minimum_overlap']
                        rows.append({'method':r['method'],'h_index':r['h_index'],
                            'eligible':eligible,'g_per_time':g/r['h'],
                            'conditional_quantum_cost':(g/r['h']/stat+state.count('1'))/pop if pop else None,
                            'certified_PF_bound':None})
                        if gamma and r['method'] in bound:
                            value=analytic_bound(F(case['coefficient_Lambda']),gamma,norms[r['method']],bound[r['method']],r['h_index'])
                            rows[-1]['certified_PF_bound']=str(value) if value is not None else None
                    for r in case['C2']:
                        costs=[(x['RZ']+x['basis_Clifford']+price*x['CX'])/r['h'] for x in r['counts']]
                        pops=[x[state] for x in r['overlaps']]
                        cost=None
                        if all(x>0 for x in pops):
                            cost=(math.sqrt(costs[0]/pops[0]/3)+math.sqrt(4*costs[1]/pops[1]/3))**2/stat
                            cost+=sum(state.count('1')/x for x in pops)
                        rows.append({'method':r['id'],'h_index':r['h_index'],
                            'eligible':r['diagnostic_branch_valid'] and abs(r['bias'])+1e-10<=pf and min(pops)>=plan['minimum_overlap'],
                            'g_per_time':None,'conditional_quantum_cost':cost,'certified_PF_bound':None})
                    frontiers={}
                    for score,group in choices.items():
                        for label,chosen in group.items():
                            valid=[x for x in rows if x['method']==chosen and x['eligible']]
                            frontiers[score+':'+label]=min(valid,key=lambda x:x['conditional_quantum_cost']) if valid else None
                    for group,predicate in (
                        ('all_single_subdivision',lambda x:x['method'].startswith('S') and not x['method'].startswith('S4')),
                        ('S4',lambda x:x['method'].startswith('S4')),
                        ('C2',lambda x:x['method'].startswith('C2'))):
                        valid=[x for x in rows if predicate(x) and x['eligible']]
                        frontiers['oracle_diagnostic:'+group]=min(valid,key=lambda x:x['conditional_quantum_cost']) if valid else None
                    certificate_points={label:{'method':mid,'h_index':sel[state]['coefficient_only_precision_choices'][str(F(str(epsilon)))][mid]}
                                        for label,mid in choices['joint'].items()}
                    labels=['KNOWN_COMPOSITION_EXPLAINS_SELECTOR_BY_CONSTRUCTION',
                            'POSTHOC_FRONTIER_NOT_OPERATIONAL_PRECISION_SELECTION',
                            'CONDITIONAL_HEISENBERG_COST_NOT_FULL_QPE_COST']
                    c1=frontiers['empirical:C1']
                    baseline=[frontiers['oracle_diagnostic:'+x] for x in ('all_single_subdivision','S4','C2')]
                    valid=[x for x in baseline if x is not None]
                    if c1 is None: labels.append('NO_ELIGIBLE_SELECTED_C1_POINT')
                    elif not valid: labels.append('NO_ELIGIBLE_BASELINE_POINT')
                    elif c1['method'].startswith('M') and c1['conditional_quantum_cost']<.95*min(x['conditional_quantum_cost'] for x in valid):
                        labels.append('CONDITIONAL_TRADEOFF_REQUIRES_GPT_JUDGMENT')
                    else: labels.append('STRONG_BASELINE_MATCHES_OR_BEATS_WITHIN_5_PERCENT')
                    subdivision=frontiers['oracle_diagnostic:all_single_subdivision']
                    if c1 is not None and subdivision is not None and subdivision['conditional_quantum_cost']<=1.05*c1['conditional_quantum_cost']:
                        labels.append('SUBDIVISION_EXPLAINS_WITHIN_5_PERCENT')
                    if certificate_points['C1']['h_index'] is None:
                        labels.append('COEFFICIENT_PRECISION_BOUND_UNRESOLVED_ON_FIXED_GRID')
                    out.append({'input_id':case['input_id'],'price':price,'state':state,'epsilon':epsilon,
                        'PF_budget':pf,'statistical_budget':stat,'rows':rows,'frontiers':frontiers,
                        'coefficient_only_precision_choices':certificate_points,
                        'classifications':labels,
                        'full_ground_energy_cost':'UNKNOWN','novelty_adopted':False})
    return out


def run(root,out):
    root=Path(root).resolve();out=Path(out).resolve();folder=root/ARTIFACT
    if out!=root/OUTPUT: raise ValueError('FIXED_AUTHORIZED_OUTPUT_REQUIRED')
    receipt=json.loads((out/'launch_receipt.json').read_text())
    execution=json.loads((folder/'execution_plan.json').read_text())
    check_approval(receipt['approval'],{'source_commit':execution['source_commit'],
        'source_manifest_SHA256':sha(folder/'source_manifest.json'),
        'execution_plan_SHA256':sha(folder/'execution_plan.json'),
        'runtime_lock_SHA256':sha(folder/'runtime_lock.json')})
    if not receipt['attempt_consumed']: raise ValueError('CONSUMED_ATTEMPT_RECEIPT_REQUIRED')
    plan=json.loads((folder/'pilot_contract.json').read_text())
    inputs=json.loads((folder/'inputs.json').read_text())['inputs']
    # Seal every selection, including bad-state controls, before creating reference jobs.
    started=time.perf_counter()
    selected={r['id']:{str(p):select(r,p) for p in plan['CX_prices']} for r in inputs}
    selector_seconds=time.perf_counter()-started
    seal=json.dumps(selected,sort_keys=True,indent=2).encode()+b'\n'
    (out/'selector_seal.json').write_bytes(seal)
    jobs=[(row,plan,selected[row['id']]) for row in inputs]
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        records=list(pool.map(case_job,jobs))
    if sum(r['eigensolver_calls'] for r in records)>plan['limits']['eigensolver_calls']:
        raise RuntimeError('CALL_COUNT')
    for row in records:
        (out/(row['input_id']+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    (out/'comparisons.json').write_text(json.dumps(summaries(records,plan),indent=2)+'\n')
    report={'status':'FIXED_PILOT_COMPLETED_PENDING_INDEPENDENT_AUDIT',
        'input_count':len(records),'eigensolver_calls':sum(r['eigensolver_calls'] for r in records),
        'selector_seal_SHA256':hashlib.sha256(seal).hexdigest(),'workers':2,'threads_per_worker':1,
        'BCH_information_selection_wall_seconds':selector_seconds,
        'wall_cap':None,'mandatory_stop':True,'formal_algorithm_adoption':False}
    if sum(p.stat().st_size for p in out.iterdir())>plan['limits']['output_bytes']:
        raise RuntimeError('OUTPUT_CAP')
    (out/'completion.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
