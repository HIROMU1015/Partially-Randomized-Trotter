"""Frozen shifts, common finite h domain, and sampled ground-phase continuation."""
from pathlib import Path
from collections import OrderedDict
import math,json,resource,time
import numpy as np
from scipy.linalg import schur
from . import qc_onebody_repartition_pilot_v1 as old
from . import qc_onebody_repartition_resources_v1 as prices
from .qc_onebody_repartition_execution_v1 import digest,write_json
from .qc_onebody_repartition_h_boundary_protocol_v1 import initial_grid,eligibility,refinement_proposals,boundary_status,NAMES
from .qc_onebody_repartition_h_boundary_execution_v1 import Context,ARTIFACT

def key(name,order,h):return f'{name}_S{order}_{float(h).hex().replace(".","_").replace("-","m").replace("+","p")}'

def restore(data,seal):
    """Reconstruct cached actors, without a selector/RDM/eigensolver call."""
    occ=old.occupations(len(data['one_body']),data['labels']);particles=int(occ[0].sum());configs=[]
    for name in NAMES:
        shift=data['shift_'+name]
        if not np.array_equal(shift,np.asarray(seal['shifts'][name])):raise ValueError('SHIFT_CHANGED')
        frames=data['orbital_frames_'+name];etas=data['actor_eta_'+name];actors=[]
        for j,(frame,eta) in enumerate(zip(frames,etas)):
            args={'frame':old.exterior_frame(frame,data['labels']),'orbital_frame':frame,'eta':eta,
                  'linear':data['actor_linear_'+name][j],'pairs':data['actor_pairs_'+name][j],
                  'particles':particles}
            if j==0:args.update(kind='onebody',energies=occ@eta)
            else:
                weight=float(data['lambdas'][j-1]);x=shift if j==1 else np.zeros_like(shift)
                args.update(kind='density',weight=weight,shift=x,energies=weight*(occ@eta)**2-occ@x)
            actors.append(args)
        configs.append({'name':name,'shift':shift,'actors':actors,'constant':float(data['constant']),
                        'd':float(seal['leading_coefficients'][name])})
    return {c['name']:c for c in configs}

def circular_clusters(values,h,tolerance):
    angles=np.mod(-np.angle(values),2*math.pi);groups=[]
    for i in np.argsort(angles):
        if not groups or angles[i]-angles[groups[-1][-1]]>h*tolerance:groups.append([int(i)])
        else:groups[-1].append(int(i))
    if len(groups)>1 and angles[groups[0][0]]+2*math.pi-angles[groups[-1][-1]]<=h*tolerance:
        groups=[groups[-1]+groups[0],*groups[1:-1]]
    return groups

def phase_from_spectrum(U,V,values,state,h,E0,contract,K,left=None,right=None):
    residual=float(np.linalg.norm(U@V-V*values,'fro'));defect=float(np.linalg.norm(U.conj().T@U-np.eye(len(U)),'fro'))
    orth=float(np.linalg.norm(V.conj().T@V-np.eye(len(U)),'fro'))
    groups=circular_clusters(values,h,contract['evaluation']['energy_cluster_absolute_tolerance_Ha'])
    weights=np.abs(V.conj().T@state)**2;scores=[float(sum(weights[g])) for g in groups]
    order=sorted(range(len(groups)),key=lambda i:(-scores[i],i));chosen=groups[order[0]]
    principal=float(np.angle(np.mean(np.exp(-1j*np.angle(values[chosen])))))
    P=V[:,chosen]@V[:,chosen].conj().T;theta=principal;integer=0;reasons=[];connections=[]
    if max(residual,defect,orth)>contract['branch']['maximum_spectral_defect']:reasons.append('NUMERICAL_SPECTRAL_DEFECT')
    if len(order)>1 and scores[order[0]]-scores[order[1]]<=contract['evaluation']['overlap_tie_tolerance']:reasons.append('INPUT_OVERLAP_TIE')
    if left is None:
        if h*K>=math.pi/8:reasons.append('ANCHOR_PHASE_GUARD')
    else:
        if left[0]['ambiguous']:reasons.append('LEFT_BRANCH_INVALID_NO_RESTART')
        integer=int(round((left[0]['theta']-principal)/(2*math.pi)));theta=principal+2*math.pi*integer
        for ref in [left]+([right] if right is not None and not right[0]['ambiguous'] else []):
            r,Q=ref;dh=abs(h-r['h']);transition=float(np.trace(P@Q).real/max(1,len(chosen)))
            competitors=[float(np.trace((V[:,g]@V[:,g].conj().T)@Q).real/max(1,len(g))) for g in groups if g!=chosen]
            connections.append({'h':r['h'],'transition_overlap':transition,'phase_increment':theta-r['theta']})
            if dh*K>math.pi/8*(1+1e-10):reasons.append('CONTINUATION_STEP_TOO_LARGE')
            if abs(theta-r['theta'])>K*dh+1e-8:reasons.append('PHASE_LIPSCHITZ_FAILURE')
            if transition<contract['branch']['minimum_projector_overlap'] or (competitors and transition-max(competitors)<=1e-6):reasons.append('PROJECTOR_CONNECTION_AMBIGUOUS')
            if len(chosen)!=r['cluster_rank']:reasons.append('CLUSTER_RANK_CHANGED')
            if int(round((r['theta']-principal)/(2*math.pi)))!=integer:reasons.append('LEFT_RIGHT_ALIAS_CONFLICT')
    spread=float(np.max(np.abs(np.angle(np.exp(-1j*np.angle(values[chosen]))*np.exp(-1j*principal))))/h)
    allowance=max(contract['evaluation']['numeric_absolute_tolerance'],(residual+defect+orth)/h)+spread
    energy=theta/h
    return {'h':h,'theta':theta,'alias_integer':integer,'principal_theta':principal,'energy':energy,'bias':energy-E0,
            'overlap':scores[order[0]],'ambiguous':bool(reasons),'branch_reasons':sorted(set(reasons)),
            'connections':connections,'cluster_indices':chosen,'cluster_rank':len(chosen),'numerical_allowance':allowance,
            'spectral_residual':residual,'unitarity_defect':defect,'eigenvector_orthogonality_defect':orth},P

class SpectralCache:
    def __init__(self,maximum=512):self.cache=OrderedDict();self.maximum=maximum
    def seed(self,name,actors,nodes,constant,h,budget):
        budget.tick('PF_seed_evaluations');U=np.eye(len(actors[0]['frame']),dtype=complex)
        for j,t in nodes:
            k=(name,j,float(t).hex())
            if k not in self.cache:
                a=actors[j];self.cache[k]=(a['frame']*np.exp(-1j*t*a['energies']))@a['frame'].conj().T
                if len(self.cache)>self.maximum:self.cache.popitem(last=False)
            self.cache.move_to_end(k);U=self.cache[k]@U
        return U*np.exp(-1j*constant*h)

def derived(primitives,configs,assessment=None):
    bases=[r for r in primitives if assessment is None or r['h'] in assessment]
    records=list(bases);table={(r['config'],r['order'],r['h'].hex()):r for r in primitives}
    for r in bases:
        if r['order']!=2:continue
        d=configs[r['config']]['d'];correction=r['h']**2*d
        records.append({**r,'method':'CS2:'+r['config'],'energy':r['energy']-correction,'bias':r['bias']-correction,
                        'scalar_correction':correction,'leading_information_error_coefficient':r['h']**2})
        fine=table.get((r['config'],2,(r['h']/2).hex()))
        if fine is None:continue
        records.append({'method':'EXT2:'+r['config'],'config':r['config'],'h':r['h'],'h_index':r['h_index'],
                        'energy':(4*fine['energy']-r['energy'])/3,'bias':(4*fine['bias']-r['bias'])/3,
                        'overlap':min(fine['overlap'],r['overlap']),'ambiguous':fine['ambiguous'] or r['ambiguous'],
                        'numerical_allowance':(4*fine['numerical_allowance']+r['numerical_allowance'])/3,
                        'experiments':[{**fine['experiments'][0],'weight':4/3},{**r['experiments'][0],'weight':-1/3}],
                        'paired_primitive_keys':[fine['key'],r['key']]})
    return records

def cost_rows(records,configs,contract):
    rows=[];grammar={}
    for name,c in configs.items():
        for order in (2,4):
            nodes=old.word(order,1.,len(c['actors']))
            one=prices.circuit_counts(c['actors'],nodes,c['constant'],1)
            two=prices.circuit_counts(c['actors'],nodes,c['constant'],2)
            grammar[name,order]=(one,{k:two[k]-one[k] for k in one})
    for r in records:
        for target in contract['evaluation']['energy_targets_Ha']:
            ok,reason=eligibility(r,target,contract)
            first={'method':r['method'],'h_index':r['h_index'],'h':r['h'],'target_Ha':target,
                   'price_id':contract['resources']['pricing'][0]['id'],'eligible':ok,'reason':reason}
            if ok:
                total=prices.add({},{});trials=0;queries=[];ampl=sum(abs(e['weight']) for e in r['experiments'])
                order=4 if r['method'].startswith('S4:') else 2;one,slope=grammar[r['config'],order]
                for e in r['experiments']:
                    q=max(1,math.ceil(math.pi/(e['h']*(target*.25/ampl))));rep=prices.repeats(min(1.,e['overlap']),.01/len(r['experiments']))
                    vector={k:one[k]+(q-1)*slope[k] for k in one};total=prices.add(total,vector,rep);trials+=rep
                    queries.append({'h':e['h'],'weight':e['weight'],'coherent_queries':q,'repeat_trials':rep})
                price=contract['resources']['pricing'][0];rotations=total['controlled_phase']+total['controlled_pair_phase']+2*total['controlled_Givens']
                response=r['config']=='response_selected';corrected=r['method'].startswith('CS2:')
                first.update(primitive_vector=total,state_preparations=trials,experiments=queries,extrapolation_weight_L1=ampl,
                    conditional_price_total=sum(total[k]*price[k] for k in ('controlled_Givens','controlled_phase','controlled_pair_phase')),
                    uniform_synthesis_angle_allowance=target*.125/max(1,rotations),
                    information={'gamma_required':response,'leading_coefficient_required':response or corrected,
                      'coefficient':r['config'] if response or corrected else None,'acquisition_quantum_cost':'UNDETERMINED_NOT_ZERO_BY_DEFAULT',
                      'required_leading_absolute_precision':target*.125/r['h']**2 if corrected else None},
                    cost_scope='STATIC_CONTROLLED_PRIMITIVE_HEISENBERG_PROXY_IDEAL_INFORMATION_ORACLE_H;NOT_COMPILED_T_OR_FULL_QPE')
            rows.append(first)
            for price in contract['resources']['pricing'][1:]:
                new={**first,'price_id':price['id']}
                if ok:new['conditional_price_total']=sum(first['primitive_vector'][k]*price[k] for k in ('controlled_Givens','controlled_phase','controlled_pair_phase'))
                rows.append(new)
    return rows

def comparisons(rows,records,contract):
    result=prices.comparison(rows,contract);resolved=True
    for cell in result['cells']:
        statuses={}
        for role in ('selected_response','strong_control'):
            row=cell[role]
            if row is None:statuses[role]={'status':'NO_ELIGIBLE_POINT','resolved':False};continue
            record=next(r for r in records if r['method']==row['method'] and r['h_index']==row['h_index'])
            statuses[role]=boundary_status(record,records,cell['target_Ha'],contract)
        cell['boundary_status']=statuses
        cell['grid_point_10percent_gain']=cell['status']=='CONDITIONAL_COMPONENT_GAIN'
        if not all(s['resolved'] for s in statuses.values()):resolved=False;cell['boundary_interpretation']='UNDETERMINED_WINDOW_BRANCH_OR_RESOLUTION'
        else:cell['boundary_interpretation']='SAMPLED_LOCAL_BOUNDARY_ONLY'
    result['discrete_grid_classification']=result['classification']
    result['boundary_classification']=result['classification'] if resolved else 'UNDETERMINED_BOUNDARY_COVERAGE'
    result['global_continuum_optimum_certified']=False
    result['secondary_same_family']=[{'method_family':f,'target_Ha':t,'price_id':p['id'],
        'best_by_configuration':{n:min((r for r in rows if r['eligible'] and r['method']==f+':'+n and r['target_Ha']==t and r['price_id']==p['id']),key=lambda r:(r['conditional_price_total'],r['h_index']),default=None) for n in NAMES}}
        for f in ('S2','CS2','S4','EXT2') for t in contract['evaluation']['energy_targets_Ha'] for p in contract['resources']['pricing']]
    return result

def resource_check(output,c):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss>c['limits']['aggregate_RSS_bytes']:raise ValueError('RSS_CAP')
    if sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())>c['limits']['output_bytes']:raise ValueError('OUTPUT_CAP')
    return rss

def collect(data,seal,parent_raw,contract,output,budget):
    """Authorized wrapper supplies registered data; synthetic tests supply toys."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True);(output/'cells').mkdir(exist_ok=False)
    configs=restore(data,seal);state=data['state'];E0=float(parent_raw['truth']['E0'])
    grid,K=initial_grid(contract['schedule']['h_anchor'],contract['schedule']['Lambda_common'],contract['schedule'])
    assessment,_=initial_grid(contract['schedule']['h_anchor'],contract['schedule']['Lambda_common'],contract['schedule'],support=False)
    primitives=[];seen={};cache=SpectralCache(contract['cache']['exponential_entries']);ledger=[]
    write_json(output/'inherited_selector_seal.json',seal)
    def get_ref(r):
        with np.load(output/'cells'/f'{r["key"]}.npz',allow_pickle=False) as z:
            V=z['vectors'];P=V[:,r['cluster_indices']]@V[:,r['cluster_indices']].conj().T
        return r,P
    def evaluate(h,name,order,stage):
        k=key(name,order,h);c=configs[name];nodes=old.word(order,h,len(c['actors']))
        existing=sorted((r for r in primitives if r['config']==name and r['order']==order),key=lambda r:r['h'])
        lower=[r for r in existing if r['h']<h];upper=[r for r in existing if r['h']>h]
        left=get_ref(lower[-1]) if lower else None;right=get_ref(upper[0]) if upper else None
        anchor=next((r for r in parent_raw.get('records',[]) if r.get('order')==order and r.get('config')==name and r.get('h')==h and r['method']==f'S{order}:{name}'),None)
        if anchor is not None:
            U=data[anchor['key']+'_U'];V=data[anchor['key']+'_vectors'];values=data[anchor['key']+'_eigenvalues'];origin='HASH_BOUND_V1_ANCHOR_CACHE'
        else:
            U=cache.seed(name,c['actors'],nodes,c['constant'],h,budget)
            budget.tick('PF_Schur_calls');T,V=schur(U,output='complex');values=np.diag(T);origin='NEW_AUTHORIZED_PF_CELL'
        r,_=phase_from_spectrum(U,V,values,state,h,E0,contract,K,left,right)
        r.update(key=k,config=name,order=order,method=f'S{order}:{name}',h_index=grid.index(h),stage=stage,origin=origin,
                 left_key=left[0]['key'] if left else None,right_key=right[0]['key'] if right else None,
                 experiments=[{'weight':1.,'h':h,'overlap':r['overlap'],'nodes':nodes}])
        path=output/'cells'/f'{k}.npz';np.savez_compressed(path,U=U,vectors=V,eigenvalues=values);r['array_SHA256']=digest(path)
        primitives.append(r);seen[name,order,h.hex()]=r
        resource_check(output,contract)
    for h in grid:
        for name in NAMES:
            for order in (2,4):evaluate(h,name,order,0)
    initial=list(grid);initial_assessment=list(assessment)
    for round_no in range(1,contract['schedule']['refinement_rounds']+1):
        records=derived(primitives,configs,assessment);new,decision=refinement_proposals(records,assessment,contract,grid)
        decision.update(round=round_no,previous_h_count=len(grid));ledger.append(decision)
        if len(grid)+len(new)>contract['schedule']['total_h_cap']:raise ValueError('TOTAL_H_CAP')
        if not new and not decision['assessment_new_h']:break
        grid=sorted(grid+new)
        assessment=sorted(set(assessment+decision['assessment_new_h']))
        for h in new:
            for name in NAMES:
                for order in (2,4):evaluate(h,name,order,round_no)
        for r in primitives:r['h_index']=grid.index(r['h'])
    records=derived(primitives,configs,assessment);costs=cost_rows(records,configs,contract);result=comparisons(costs,records,contract)
    if len(records)>contract['limits']['method_records'] or len(costs)>contract['limits']['cost_rows']:raise ValueError('RECORD_OR_COST_ROW_CAP')
    unresolved,unresolved_ledger=refinement_proposals(records,assessment,contract,grid)
    write_json(output/'raw.json',{'scope_id':contract['scope_id'],'initial_h_grid':initial,'h_grid':grid,'phase_derivative_bound':K,
               'initial_assessment_h_grid':initial_assessment,'assessment_h_grid':assessment,'primitive_records':primitives,
               'records':records,'costs':costs,'result':result,'refinement_ledger':ledger,'remaining_refinement':unresolved_ledger,
               'counts':dict(budget.counts),'cached_anchor_cells':sum(r['origin']=='HASH_BOUND_V1_ANCHOR_CACHE' for r in primitives),
               'E0_reused_only_for_performance_bias':E0,'no_selector_or_shift_regeneration':True,
               'finite_grid_not_monotonic_or_global_optimum_claim':True,'mandatory_stop':True})
    return {'status':'COLLECTION_COMPLETE_AWAIT_INDEPENDENT_AUDIT','discrete_grid_classification':result['discrete_grid_classification'],
            'boundary_classification':result['boundary_classification'],'h_points':len(grid),'primitive_cells':len(primitives),
            'method_records':len(records),'cost_rows':len(costs),'counts':dict(budget.counts),'peak_RSS_bytes':resource_check(output,contract),'mandatory_stop':True}

def run_registered(context):
    if not isinstance(context,Context):raise ValueError('AUTHORIZED_CONTEXT_REQUIRED')
    start=time.monotonic();budget=old.Budget(context.contract['limits'])
    try:
        budget.tick('saved_science_array_loads')
        with np.load(context.root/context.contract['saved_arrays'],allow_pickle=False) as z:data={k:np.array(z[k]) for k in z.files}
        seal=json.loads((context.root/context.contract['saved_selector_seal']).read_text());parent=json.loads((context.root/context.contract['saved_raw']).read_text())
        out=collect(data,seal,parent,context.contract,context.output,budget)
        write_json(context.output/'result_identity.json',{'scope_id':context.contract['scope_id'],'authorization_SHA256':context.authorization_SHA256,
                   'contract_SHA256':digest(context.root/ARTIFACT/'contract.json'),'source_manifest_SHA256':digest(context.root/ARTIFACT/'source_manifest.json'),
                   'fixed_files':context.contract['fixed_files'],'attempt':1,'retry':0})
        out['elapsed_seconds']=time.monotonic()-start
    except BaseException as error:
        out={'status':'TECHNICAL_OR_RESOURCE_STOP_NO_RETRY','reason':str(error)[:4096],'counts':dict(budget.counts),'elapsed_seconds':time.monotonic()-start,'mandatory_stop':True}
        write_json(context.output/'stop.json',out);raise
    write_json(context.output/'stop.json',out);return out
