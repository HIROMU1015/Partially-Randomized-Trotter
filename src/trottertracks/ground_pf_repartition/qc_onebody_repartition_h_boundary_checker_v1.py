"""Independent JW/Padé, circular phase, query/price and policy audit.

Does not import the new pilot or its spectral continuation function.
The shared stdlib schedule is a contract, not a numerical kernel.
"""
from pathlib import Path
from collections import OrderedDict
import math,json,resource
import numpy as np
from scipy.linalg import expm
from . import qc_onebody_repartition_checker_v1 as reference
from .qc_onebody_repartition_execution_v1 import digest
from .qc_onebody_repartition_h_boundary_protocol_v1 import initial_grid,refinement_proposals,boundary_status,eligibility,NAMES

def require(ok,reason):
    if not ok:raise ValueError(reason)

def phase_check(r,U,V,z,state,c,K,parent_projectors):
    require(np.isfinite(U).all() and np.isfinite(V).all() and np.isfinite(z).all(),'NONFINITE_SPECTRUM')
    h=r['h'];angles=-np.angle(z);tol=h*c['evaluation']['energy_cluster_absolute_tolerance_Ha']
    sets=[{i} for i in range(len(z))]
    # Connected components in circular distance; independent of sorted-angle code.
    for i in range(len(z)):
        for j in range(i):
            if abs(np.angle(np.exp(1j*(angles[i]-angles[j]))))<=tol:
                left=next(s for s in sets if i in s);right=next(s for s in sets if j in s)
                if left is not right:left.update(right);sets.remove(right)
    inds=r['cluster_indices'];require(set(inds) in sets,'CIRCULAR_CLUSTER')
    weights=abs(V.conj().T@state)**2;scores=[float(sum(weights[list(s)])) for s in sets]
    score=float(sum(weights[inds]));ordered=sorted(scores,reverse=True)
    require(score>=ordered[0]-1e-12,'EXACT_STATE_BRANCH_SELECTION')
    P=V[:,inds]@V[:,inds].conj().T
    principal=float(np.angle(sum(np.exp(1j*angles[inds]))))
    residual=float(np.linalg.norm(U@V-V*z,'fro'));defect=float(np.linalg.norm(U.conj().T@U-np.eye(len(z)),'fro'));orth=float(np.linalg.norm(V.conj().T@V-np.eye(len(z)),'fro'))
    reasons=[];theta=principal;integer=0
    if max(residual,defect,orth)>c['branch']['maximum_spectral_defect']:reasons.append('NUMERICAL_SPECTRAL_DEFECT')
    if len(scores)>1 and ordered[0]-ordered[1]<=c['evaluation']['overlap_tie_tolerance']:reasons.append('INPUT_OVERLAP_TIE')
    if r['left_key'] is None:
        if h*K>=math.pi/8:reasons.append('ANCHOR_PHASE_GUARD')
    else:
        left,Q=parent_projectors[r['left_key']]
        if left['ambiguous']:reasons.append('LEFT_BRANCH_INVALID_NO_RESTART')
        integer=int(round((left['theta']-principal)/(2*math.pi)));theta=principal+integer*2*math.pi
        refs=[(left,Q)]
        if r['right_key'] is not None and not parent_projectors[r['right_key']][0]['ambiguous']:refs.append(parent_projectors[r['right_key']])
        expected_connections=[]
        for prev,Q in refs:
            dh=abs(h-prev['h']);transfer=float(np.trace(P@Q).real/len(inds));other=[float(np.trace((V[:,list(s)]@V[:,list(s)].conj().T)@Q).real/len(s)) for s in sets if s!=set(inds)]
            expected_connections.append((prev['h'],transfer,theta-prev['theta']))
            if dh*K>math.pi/8*(1+1e-10):reasons.append('CONTINUATION_STEP_TOO_LARGE')
            if abs(theta-prev['theta'])>K*dh+1e-8:reasons.append('PHASE_LIPSCHITZ_FAILURE')
            if transfer<c['branch']['minimum_projector_overlap'] or (other and transfer-max(other)<=1e-6):reasons.append('PROJECTOR_CONNECTION_AMBIGUOUS')
            if len(inds)!=prev['cluster_rank']:reasons.append('CLUSTER_RANK_CHANGED')
            if int(round((prev['theta']-principal)/(2*math.pi)))!=integer:reasons.append('LEFT_RIGHT_ALIAS_CONFLICT')
        require(len(r['connections'])==len(expected_connections),'CONNECTION_COUNT')
        for obs,(rh,t,inc) in zip(r['connections'],expected_connections):
            require(obs['h']==rh and abs(obs['transition_overlap']-t)<1e-10 and abs(obs['phase_increment']-inc)<1e-10,'CONNECTION_LEDGER')
    spread=max(abs(np.angle(np.exp(1j*(angles[inds]-principal)))))/h
    allowance=max(c['evaluation']['numeric_absolute_tolerance'],(residual+defect+orth)/h)+spread
    require(r['numerical_allowance']>=c['evaluation']['numeric_absolute_tolerance'],'NUMERICAL_FLOOR_REMOVED')
    require(r['branch_reasons']==sorted(set(reasons)) and r['ambiguous']==bool(reasons),'BRANCH_VALIDITY_OR_REASON')
    require(r['alias_integer']==integer and abs(r['theta']-theta)<1e-10 and abs(r['principal_theta']-principal)<1e-10,'PHASE_UNWRAP')
    require(abs(r['energy']-theta/h)<1e-10 and abs(r['overlap']-score)<1e-10,'ENERGY_OR_OVERLAP')
    require(abs(r['numerical_allowance']-allowance)<1e-10,'NUMERICAL_ALLOWANCE')
    require(r['cluster_rank']==len(inds),'CLUSTER_RANK')
    return P

def audit(data,seal,raw,contract,directory):
    directory=Path(directory);grid,K=initial_grid(contract['schedule']['h_anchor'],contract['schedule']['Lambda_common'],contract['schedule'])
    require(raw['scope_id']==contract['scope_id'] and raw['initial_h_grid']==grid,'SCOPE_INITIAL_GRID')
    require(abs(raw['phase_derivative_bound']-K)<1e-12,'SIGNED_S4_DERIVATIVE_BOUND')
    require(not seal['E0_accessed_by_generator'] and not seal['finite_step_truth_accessed_by_generator'],'SELECTOR_TRUTH')
    labels=data['labels'];state=data['state'];dim=len(labels);n=len(data['one_body']);ladders=reference.jw_transitions(n,labels)
    lift=lambda a:np.einsum('pq,pqij->ij',a,ladders,optimize=True)
    parts0=[lift(data['one_body'])]+[w*lift(g)@lift(g) for w,g in zip(data['lambdas'],data['g_matrices'])]
    H=sum(parts0)+float(data['constant'])*np.eye(dim)
    require(np.linalg.norm(H-data['H'],'fro')<1e-10,'HASHED_H_RECONSTRUCTION')
    ev=data['H_eigenvalues'];V0=data['H_eigenvectors']
    require(np.all(np.diff(ev)>=0) and np.linalg.norm(H@V0-V0*ev,'fro')<1e-9,'SAVED_TRUTH_CERTIFICATE')
    require(abs(raw['E0_reused_only_for_performance_bias']-float(ev[0]))<1e-12,'BIAS_TRUTH_REFERENCE')
    for name in NAMES:
        F=data['orbital_frames_'+name];x=data['shift_'+name];X=(F[1]*x)@F[1].conj().T
        require(np.array_equal(x,np.asarray(seal['shifts'][name])),'FROZEN_SHIFT')
        expected=[parts0[0]+lift(X),parts0[1]-lift(X),*parts0[2:]]
        require(np.linalg.norm(np.asarray(expected)-data['fragments_'+name])<1e-9,'REPARTITION_OR_FRAME')
    primitives=raw['primitive_records']
    require(len(primitives)==6*len(raw['h_grid']) and len(raw['h_grid'])<=contract['schedule']['total_h_cap'],'COVERAGE_CAP')
    expected_keys={(name,order,h.hex()) for name in NAMES for order in (2,4) for h in raw['h_grid']}
    require({(r['config'],r['order'],r['h'].hex()) for r in primitives}==expected_keys,'FAIR_COMMON_H_COVERAGE')
    require(len({r['key'] for r in primitives})==len(primitives),'DUPLICATE_PRIMITIVE')
    new_calls=sum(r['origin']=='NEW_AUTHORIZED_PF_CELL' for r in primitives)
    require(raw['counts']['PF_seed_evaluations']==raw['counts']['PF_Schur_calls']==new_calls,'CALL_COUNTS')
    require(new_calls<=contract['limits']['PF_seed_evaluations'] and len(primitives)<=contract['limits']['independent_PF_recompositions'],'REGISTERED_CALL_CAPS')
    exps=OrderedDict();exp_count=0;parents={};visited=[]
    for r in primitives:
        h=r['h'];name=r['config'];parts=data['fragments_'+name];nodes=reference.pf_nodes(r['order'],h,len(parts));U=np.eye(dim,dtype=complex)
        for j,t in nodes:
            k=(name,j,float(t).hex())
            if k not in exps:
                require(exp_count<contract['limits']['independent_expm_calls'],'EXPM_CAP')
                exps[k]=expm(-1j*t*parts[j]);exp_count+=1
                if len(exps)>contract['cache']['exponential_entries']:exps.popitem(last=False)
            exps.move_to_end(k);U=exps[k]@U
        U*=np.exp(-1j*float(data['constant'])*h)
        file=directory/'cells'/f'{r["key"]}.npz';require(digest(file)==r['array_SHA256'],'CELL_SHA')
        with np.load(file,allow_pickle=False) as z:stored,V,zv=z['U'],z['vectors'],z['eigenvalues']
        require(np.linalg.norm(U-stored,'fro')<1e-9,'INDEPENDENT_PADE_PF')
        # Nearest previously evaluated brackets, rather than E0-based aliases.
        same=sorted((x for x in visited if x['config']==name and x['order']==r['order']),key=lambda x:x['h'])
        lo=[x for x in same if x['h']<h];hi=[x for x in same if x['h']>h]
        require(r['left_key']==(lo[-1]['key'] if lo else None) and r['right_key']==(hi[0]['key'] if hi else None),'CONTINUATION_BRACKETS')
        need=[r[k] for k in ('left_key','right_key') if r[k] is not None];refs={}
        for k in need:
            prior=parents[k]
            with np.load(directory/'cells'/f'{k}.npz',allow_pickle=False) as z:
                vv=z['vectors'][:,prior['cluster_indices']];refs[k]=(prior,vv@vv.conj().T)
        phase_check(r,U,V,zv,state,contract,K,refs)
        require(abs(r['bias']-r['energy']+float(ev[0]))<1e-10,'BIAS')
        require(r['experiments']==[{'weight':1.,'h':h,'overlap':r['overlap'],'nodes':[list(x) for x in nodes]}],'EXPERIMENT_WORD')
        parents[r['key']]=r;visited.append(r)
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<contract['limits']['aggregate_RSS_bytes'],'AUDIT_RSS_CAP')
    records={(r['method'],r['h'].hex()):r for r in raw['records']}
    primitive_lookup={(r['method'],r['h'].hex()):r for r in primitives}
    require(len(records)==len(raw['records']),'DUPLICATE_METHOD_RECORD')
    expected_record_keys={(f'S{o}:'+n,h.hex()) for n in NAMES for o in (2,4) for h in raw['assessment_h_grid']}
    expected_record_keys|={('CS2:'+n,h.hex()) for n in NAMES for h in raw['assessment_h_grid']}
    expected_record_keys|={('EXT2:'+n,h.hex()) for n in NAMES for h in raw['assessment_h_grid']}
    require(set(records)==expected_record_keys,'DERIVED_RECORD_COVERAGE')
    for r in raw['records']:
        if r['method'].startswith(('S2:','S4:')):require(r==primitive_lookup[r['method'],r['h'].hex()],'PRIMARY_PRIMITIVE_RECORD')
        if r['method'].startswith('CS2:'):
            base=records['S2:'+r['config'],r['h'].hex()];d=seal['leading_coefficients'][r['config']]
            require(abs(r['energy']-base['energy']+r['h']**2*d)<1e-10 and abs(r['bias']-base['bias']+r['h']**2*d)<1e-10,'SCALAR_CORRECTION')
            require(r['ambiguous']==base['ambiguous'] and r['overlap']==base['overlap'],'CS2_BRANCH')
            require(r['numerical_allowance']==base['numerical_allowance'] and r['experiments']==base['experiments'],'CS2_NUMERICAL_OR_QUERY_PROPAGATION')
        if r['method'].startswith('EXT2:'):
            coarse=primitive_lookup['S2:'+r['config'],r['h'].hex()];fine=primitive_lookup['S2:'+r['config'],(r['h']/2).hex()]
            require(abs(r['energy']-(4*fine['energy']-coarse['energy'])/3)<1e-10 and abs(r['bias']-(4*fine['bias']-coarse['bias'])/3)<1e-10,'PAIRED_RICHARDSON')
            require(r['ambiguous']==(coarse['ambiguous'] or fine['ambiguous']) and r['overlap']==min(coarse['overlap'],fine['overlap']),'EXT2_BRANCH')
            expected_experiments=[{**fine['experiments'][0],'weight':4/3},{**coarse['experiments'][0],'weight':-1/3}]
            require(r['experiments']==expected_experiments,'EXTRAPOLATION_WORK')
            require(r['numerical_allowance']==(4*fine['numerical_allowance']+coarse['numerical_allowance'])/3,'EXT2_NUMERICAL_PROPAGATION')
        require(r['h_index']==raw['h_grid'].index(r['h']),'H_INDEX')
    # Replay the deterministic eligibility-refinement choices, with all methods.
    current=list(grid);assessment,_=initial_grid(contract['schedule']['h_anchor'],contract['schedule']['Lambda_common'],contract['schedule'],support=False)
    require(raw['initial_assessment_h_grid']==assessment,'INITIAL_ASSESSMENT_DOMAIN')
    for decision in raw['refinement_ledger']:
        stage_records=[r for r in raw['records'] if r['h'] in assessment]
        points,expected=refinement_proposals(stage_records,assessment,contract,current)
        expected.update(round=decision['round'],previous_h_count=len(current))
        require(expected==decision,'REFINEMENT_DECISION_OR_FAIRNESS');current=sorted(current+points)
        assessment=sorted(set(assessment+expected['assessment_new_h']))
    require(current==raw['h_grid'],'REFINEMENT_FINAL_GRID')
    require(assessment==raw['assessment_h_grid'],'FAIR_ASSESSMENT_FINAL_GRID')
    _,remaining=refinement_proposals(raw['records'],assessment,contract,current);require(raw['remaining_refinement']==remaining,'REMAINING_REFINEMENT')
    # Reuse only the algebraic affine query-count grammar, independently of
    # production accounting; actual controlled primitive decomposition is v1 ref.
    grammars={};particle_count=int(int(labels[0]).bit_count())
    for name in NAMES:
        lines,pairs,scalars=reference.sector_coefficients(data['actor_eta_'+name],data['lambdas'],data['shift_'+name],particle_count)
        scalar_flag=float(data['constant'])!=0 or any(v!=0 for v in scalars)
        for order in (2,4):
            nodes=reference.pf_nodes(order,1.,len(lines))
            one=reference.circuit_vector(data['orbital_frames_'+name],lines,pairs,nodes,int(scalar_flag),1)
            two=reference.circuit_vector(data['orbital_frames_'+name],lines,pairs,nodes,int(scalar_flag),2)
            grammars[name,order]=(one,two-one)
    price_table={p['id']:p for p in contract['resources']['pricing']};cost_keys=set()
    for cost in raw['costs']:
        ck=(cost['method'],cost['h'].hex(),cost['target_Ha'],cost['price_id']);require(ck not in cost_keys,'DUPLICATE_COST');cost_keys.add(ck)
        r=records[cost['method'],cost['h'].hex()];target=cost['target_Ha'];ok,reason=eligibility(r,target,contract)
        require(cost['eligible']==ok and cost['reason']==reason,'ELIGIBILITY_OR_MARGIN')
        if not ok:continue
        total=np.zeros(4,dtype=np.int64);preps=0;amp=sum(abs(e['weight']) for e in r['experiments']);experiments=[]
        for e in r['experiments']:
            q=max(1,math.ceil(math.pi/(e['h']*(target*.25/amp))));prob=min(1.,e['overlap'])
            rep=1 if prob>=1 else max(1,math.ceil(math.log(.01/len(r['experiments']))/math.log1p(-prob)))
            order=4 if r['method'].startswith('S4:') else 2;one,slope=grammars[r['config'],order]
            total+=rep*(one+(q-1)*slope);preps+=rep
            experiments.append({'h':e['h'],'weight':e['weight'],'coherent_queries':q,'repeat_trials':rep})
        actual=[cost['primitive_vector'][k] for k in ('controlled_Givens','controlled_phase','controlled_pair_phase','fermionic_parity_links')]
        require(np.array_equal(total,actual) and cost['experiments']==experiments and cost['state_preparations']==preps,'PRIMITIVE_QUERY_REPEAT_COUNTS')
        price=price_table[cost['price_id']];score=sum(int(total[i])*price[k] for i,k in enumerate(('controlled_Givens','controlled_phase','controlled_pair_phase')))
        require(score==cost['conditional_price_total'],'PRICE')
        rotations=int(total[1]+total[2]+2*total[0])
        require(cost['extrapolation_weight_L1']==amp and cost['uniform_synthesis_angle_allowance']==target*.125/max(1,rotations),'SYNTHESIS_OR_EXTRAPOLATION_BUDGET')
        require(cost['information']['gamma_required']==(r['config']=='response_selected'),'INFORMATION_CHARGE')
        require(cost['information']['leading_coefficient_required']==(r['config']=='response_selected' or r['method'].startswith('CS2:')),'LEADING_INFO')
        precision=target*.125/r['h']**2 if r['method'].startswith('CS2:') else None
        require(cost['information']['required_leading_absolute_precision']==precision,'INFORMATION_PRECISION')
    expected_costs={(m,h,t,p) for (m,h) in records for t in contract['evaluation']['energy_targets_Ha'] for p in price_table}
    require(cost_keys==expected_costs,'FULL_COST_COVERAGE')
    statuses=[];resolved=True
    for cell in raw['result']['cells']:
        rows=[r for r in raw['costs'] if r['eligible'] and r['target_Ha']==cell['target_Ha'] and r['price_id']==cell['price_id']]
        R=[r for r in rows if r['method'] in ('S2:response_selected','CS2:response_selected')];B=[r for r in rows if r not in R]
        choose=lambda rr:min(rr,key=lambda r:(r['conditional_price_total'],r['method'],r['h_index']),default=None)
        candidate,control=choose(R),choose(B)
        require(cell['selected_response']==candidate and cell['strong_control']==control,'STRONG_CONTROLS')
        if candidate is None or control is None:status='UNDETERMINED_ELIGIBLE_GRID_INSUFFICIENT'
        else:
            ratio=candidate['conditional_price_total']/control['conditional_price_total'];require(abs(cell['ratio']-ratio)<1e-12,'RATIO')
            status='CONDITIONAL_COMPONENT_GAIN' if ratio<.9-1e-10 else 'NO_REGISTERED_COMPONENT_GAIN'
        require(cell['status']==status,'10PERCENT_GATE');statuses.append(status)
        for role,row in [('selected_response',candidate),('strong_control',control)]:
            expected=boundary_status(records[row['method'],row['h'].hex()],raw['records'],cell['target_Ha'],contract) if row else {'status':'NO_ELIGIBLE_POINT','resolved':False}
            require(cell['boundary_status'][role]==expected,'BOUNDARY_DIAGNOSTIC');resolved&=expected['resolved']
    classification='CONDITIONAL_COMPONENT_GAIN_IDEAL_INFORMATION' if all(s=='CONDITIONAL_COMPONENT_GAIN' for s in statuses) else 'UNDETERMINED' if any(s.startswith('UNDETERMINED') for s in statuses) else 'NO_UNIFORM_REGISTERED_COMPONENT_GAIN'
    require(raw['result']['classification']==raw['result']['discrete_grid_classification']==classification,'CLASSIFICATION')
    require(raw['result']['boundary_classification']==(classification if resolved else 'UNDETERMINED_BOUNDARY_COVERAGE'),'BOUNDARY_CLASSIFICATION')
    expected_secondary=[{'method_family':f,'target_Ha':t,'price_id':p,
        'best_by_configuration':{n:min((r for r in raw['costs'] if r['eligible'] and r['method']==f+':'+n and r['target_Ha']==t and r['price_id']==p),key=lambda r:(r['conditional_price_total'],r['h_index']),default=None) for n in NAMES}}
        for f in ('S2','CS2','S4','EXT2') for t in contract['evaluation']['energy_targets_Ha'] for p in price_table]
    require(raw['result']['secondary_same_family']==expected_secondary,'SECONDARY_PRIMARY_SEPARATION')
    require(not raw['result']['global_continuum_optimum_certified'] and not raw['result']['formal_research_GO'],'CLAIM_BOUNDARY')
    return {'status':'INDEPENDENT_BOUNDARY_JW_PADE_PHASE_COST_POLICY_AUDIT_PASS','PF_recompositions':len(primitives),
            'expm_calls':exp_count,'new_eigensolvers_RDM_leading_optimizer_compile_calls':0,
            'sampled_branch_not_continuum_certificate':True,'formal_research_GO':False,'mandatory_stop':True}
