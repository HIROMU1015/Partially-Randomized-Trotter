"""Explicit partial circuit envelopes; unresolved functions remain UNKNOWN.

No circuits are generated or compiled. Counts refer to the documented logical
templates; upper envelopes do not establish optimal circuit lower bounds.
"""
import math

from .qc_mixed_access_math_v1 import ContractError, stages


def vector(toffoli=0,rz=0,cx=0,cz=0,clifford=0,workspace=0):
    return dict(Toffoli=toffoli,RZ=rz,CX=cx,CZ=cz,Clifford=clifford,
                known_workspace=workspace)


def add(a,b):
    return {k:max(a[k],b[k]) if k=='known_workspace' else a[k]+b[k] for k in a}


def diagonal_counts(single,pair,scalar=False):
    # Controlled n and nn phases: CP=3 RZ+2 CX; nn uses compute/erase AND.
    return vector(2*pair,3*(single+pair)+int(scalar),2*(single+pair),workspace=int(pair>0))


def excitation_counts(group):
    k=len(group['create'])+len(group['annihilate']); parity=len(group['parity_mask'])
    weights=sum(v!=0 for v in group['g_weights'].values()); constant=int(group['g0']!=0)
    rotations=weights+constant
    valid_toffoli=2 if k==2 else 8
    valid_cx=2 if k==2 else 0
    return vector(valid_toffoli+2*weights,2*rotations,2*(k-1)+2*parity+2*rotations+valid_cx,
                  2 if parity else 0,clifford=2*(k-1),workspace=k+2)


def resource_nodes(nodes):
    """Share theta/rho and cancel adjacent inverse Ry at the R2 center."""
    out=[]; i=0
    while i<len(nodes):
        node=nodes[i]
        if i+1<len(nodes) and node[0]=='R' and nodes[i+1][0]=='R' and node[1]==nodes[i+1][1] and node[3] and not nodes[i+1][3] and node[2]==-nodes[i+1][2]:
            out.append(('RPAIR',node[1],nodes[i+1][2],False)); i+=2
        else: out.append(node); i+=1
    return out


def stage_counts(node,model,w,a,fraction_bits=0):
    kind,index,t,adj=node
    if kind=='SCALAR': return vector(rz=int(model['scalar']!=0)),0
    if kind=='A':
        singles=sum(x!=0 for x in model['eps']); pairs=sum(model['density'][i,j]!=0 for i in range(model['n']) for j in range(i+1,model['n']))
        return diagonal_counts(singles,pairs,model['scalar']!=0),0
    if kind=='P':
        x,z,g=model['paulis'][index]; k=(x|z).bit_count()
        return (vector(rz=1) if k==0 else vector(rz=2,cx=2*k,clifford=4*k)),0
    if kind=='E': return excitation_counts(model['groups'][index]),0
    if kind=='D':
        actor=model['gaussian'][index]; eta=actor['eta']; lam=actor['weight']
        singles=sum(x!=0 for x in eta) if lam is None else sum(lam*x*x!=0 for x in eta)
        pairs=0 if lam is None else sum(lam*eta[i]*eta[j]!=0 for i in range(len(eta)) for j in range(i+1,len(eta)))
        counts=diagonal_counts(singles,pairs)
        # Basis frames are outside control. Each Givens: parity plus CRX pair encoding.
        for op in actor['operations']:
            if op[0]=='sign': counts=add(counts,vector(clifford=2))
            else:
                p,q=op[1:3]; parity=q-p-1
                counts=add(counts,vector(rz=4,cx=2*(4+2*parity),cz=4 if parity else 0))
        return counts,0
    if kind not in ('R','RPAIR'): raise ContractError('RESOURCE_ACTOR')
    group=model['groups'][index]; z=sum(v!=0 for v in group['delta_weights'].values())
    g=sum(v!=0 for v in group['g_weights'].values())
    w=max(3,w); coefficients=list(group['delta_weights'].values())+list(group['g_weights'].values())
    # Known Cuccaro constant-load compute/erase design from the fixed audit.
    # Constants use Clifford X loads, not occupation-controlled adders.
    hamming=sum((int(round(float(v)*2**fraction_bits)) % (1<<w)).bit_count() for v in coefficients if v!=0)
    hamming=min(hamming,w*(z+g))
    rotations=5 if kind=='RPAIR' else 4
    # Each variable rotation, bitwise controlled: compute control AND,
    # CRZ decomposition, erase. Function generation/erasure is excluded here.
    counts=excitation_counts({**group,'g0':0.,'g_weights':{}})
    counts=add(counts,vector(toffoli=(4*w-6)*(z+g)+2*rotations*a,rz=2*rotations*a,
                             cx=(10*w-14)*(z+g)+4*hamming+2*rotations*a,
                             clifford=(4*w-12)*(z+g)+4*w+8*a,
                             workspace=3*w+rotations*a+8))
    return counts,1


def ledger(method,T,r,fraction,model,epsilon=1e-3,model_error=0.):
    residual=epsilon*(1-fraction)-model_error
    if residual<=0: return {'status':'MODEL_RESIDUAL_EXHAUSTS_BUDGET'}
    parts=3 if method[0]=='R' else 2
    coefficient_budget=residual/parts; synthesis_budget=residual/parts
    original_nodes=stages(method,T/r,model); nodes=resource_nodes(original_nodes)
    # Uniform sufficient rounding precision. Includes every elementary known
    # rotation and coefficient occurrences; no observed error tuning.
    base=vector(); relative=0
    for node in nodes:
        c,k=stage_counts(node,model,1,1); base=add(base,c); relative+=k
    values=[abs(float(x)) for x in model['h'].flat]+[abs(float(x)) for x in model['eri'].flat]+[abs(model['scalar'])]
    for group in model['groups']:
        values += [abs(group['delta0']),abs(group['g0'])]+[abs(v) for v in group['delta_weights'].values()]+[abs(v) for v in group['g_weights'].values()]
    values += [abs(float(v)) for v in model['eps']]+[abs(float(v)) for v in model['density'].flat]
    values += [abs(g) for x,z,g in model['paulis']]
    for actor in model['gaussian']:
        values += [abs(float(v)) for v in actor['eta']]
        if actor['weight'] is not None: values.append(abs(actor['weight']))
        values += [abs(math.atan2(op[4],op[3])) for op in actor['operations'] if op[0]=='givens']
    integer_bits=max(2,2+math.ceil(math.log2(max(1.,sum(values)))))
    # Telescoping unitary rounding envelope. The 1024 guard dominates the
    # degree-three DF phase polynomial derivatives for <=8 occupations, affine
    # detuning/amplitude sums, and basis-frame angle occurrences. This is a
    # design bound under exact fixed-point input rounding, not backend evidence.
    coefficient_guard=1024*(1+sum(values))**2*(1+abs(T))*max(1,r*len(nodes))*max(1,len(values))
    common_coefficient_budget=coefficient_budget/2 if method[0]=='R' else coefficient_budget
    fractional_bits=max(1,math.ceil(math.log2(coefficient_guard/common_coefficient_budget)))
    w=integer_bits+fractional_bits
    group_precision=[]
    if method[0]=='R':
        groups=model['groups']; unit_budget=coefficient_budget/(4*max(1,len(groups)))
        for j,group in enumerate(groups):
            times=[node[2] for node in original_nodes if node[0]=='R' and node[1]==j]
            S1=r*sum(abs(t) for t in times); S2=r*sum(t*t for t in times)
            z=sum(v!=0 for v in group['delta_weights'].values()); zg=sum(v!=0 for v in group['g_weights'].values())
            d_range=abs(group['delta0'])+sum(abs(v) for v in group['delta_weights'].values())
            g_range=abs(group['g0'])+sum(abs(v) for v in group['g_weights'].values())
            D=g_range*(z+1)*S2/(4*unit_budget); B=(zg+1)*S1/(2*unit_budget)
            fD=max(0,math.ceil(math.log2(D))) if D else 0
            fB=max(0,math.ceil(math.log2(B))) if B else 0
            f=max(fD,fB)
            guarded_range=max(d_range+(z+1)*2**(-f-1),g_range+(zg+1)*2**(-f-1))
            ib=max(2,2+math.ceil(math.log2(max(1.,guarded_range))))
            group_precision.append({'j':j,'S1':S1,'S2':S2,'z':z,'z_g':zg,'b_max':g_range,
                'detuning_budget':unit_budget,'amplitude_budget':unit_budget,'detuning_fraction_bits':fD,
                'amplitude_fraction_bits':fB,'fraction_bits':f,'integer_sign_guard_bits':ib,'word_bits':max(3,ib+f),
                'eta_delta':(z+1)*2**(-f-1),'eta_b':(zg+1)*2**(-f-1),
                'detuning_error_envelope':g_range*(z+1)*2**(-f-1)*S2/2,
                'amplitude_error_envelope':(zg+1)*2**(-f-1)*S1})
    a=max(1,math.ceil(math.log2(16*math.pi*max(1,r*relative)/(residual/3)))) if relative else 0
    if max([w,a]+[p['word_bits'] for p in group_precision])>128: return {'status':'PRECISION_CAP','word_bits':w,'angle_bits':a}
    total=vector(); relative=0
    for node in nodes:
        node_w=group_precision[node[1]]['word_bits'] if node[0] in ('R','RPAIR') else w
        f=group_precision[node[1]]['fraction_bits'] if node[0] in ('R','RPAIR') else fractional_bits
        c,k=stage_counts(node,model,node_w,a,f)
        total=add(total,c); relative+=k
    eta=synthesis_budget/max(1,r*total['RZ'])
    return {'status':'PARTIAL_ENVELOPE','per_step':total,'r':r,'T':T,'method':method,
            'PF_fraction':fraction,'coefficient_budget':coefficient_budget,
            'function_budget':residual/3 if relative else None,'synthesis_budget':synthesis_budget,
            'word_bits':w,'fractional_bits':fractional_bits,'angle_bits':a,'eta_RZ':eta,
            'group_precision':{'columns':['detuning_fraction_bits','amplitude_fraction_bits','fraction_bits','word_bits'],
                               'by_j':[[p[k] for k in ('detuning_fraction_bits','amplitude_fraction_bits','fraction_bits','word_bits')] for p in group_precision],
                               'S1':group_precision[0]['S1'] if group_precision else None,'S2':group_precision[0]['S2'] if group_precision else None,
                               'per_group_detuning_budget':group_precision[0]['detuning_budget'] if group_precision else None,
                               'per_group_amplitude_budget':group_precision[0]['amplitude_budget'] if group_precision else None},
            'center_theta_sharing':method[0]=='R',
            'coefficient_rounding_guard':coefficient_guard,'precision_status':'sufficient design envelope;backend rounding not implemented',
            'known_native_serial_depth_envelope':sum(total[k] for k in ('Toffoli','RZ','CX','CZ','Clifford')),
            'relative_blocks_per_step':relative,'Q':0,'depth':'UNKNOWN','total_workspace':'UNKNOWN' if relative else total['known_workspace'],
            'F_remaining':'UNKNOWN' if relative else None,'routing':'all-to-all',
            'classification':'documented conservative template envelope,not compiled or optimal'}


def eligible_steps(probes,method,T,fraction,grid,epsilon,margin):
    rows={p['r']:p['error'] for p in probes if p['method']==method and p['T']==T}
    if set(rows)!=set(grid): raise ContractError('INCOMPLETE_ELIGIBILITY_GRID')
    passed=[r for r in grid if all(rows[s]+margin<=fraction*epsilon for s in grid if s>=r)]
    return [r for r in passed if 2*r in rows], ('BOUNDARY_UNCONFIRMED' if grid[-1] in passed and len(passed)==1 else 'GRID_CHECKED')


def modeled_total(row,kappa):
    c=row['per_step']; tr=math.ceil(kappa*math.log2(1/row['eta_RZ']))
    return row['Q']+row['r']*(7*c['Toffoli']+c['RZ']*tr),tr


def evaluate_resources(model,probes,contract,df_norm_residual):
    grid=contract['grid']; rows=[]; frontiers=[]; precision_catalog={}; numerical_ambiguities=[]
    for T in grid['times_atomic']:
        for method in contract['methods']:
            residual=abs(T)*df_norm_residual if method[0]=='D' else 0.
            for fraction in grid['PF_budget_fractions']:
                eligible,boundary=eligible_steps(probes,method,T,fraction,grid['steps'],grid['operator_error_target'],grid['numerical_margin'])
                for r in grid['steps']:
                    row=ledger(method,T,r,fraction,model,grid['operator_error_target'],residual)
                    if 'group_precision' in row:
                        precision_id=f'{method}:T{T}:r{r}:a{fraction}'
                        precision_catalog[precision_id]=row.pop('group_precision')
                        row['precision_id']=precision_id
                    for kappa in contract['resource_model']['rotation_sensitivity_kappa']:
                        entry={**row,'id':f'{method}:T{T}:r{r}:a{fraction}:k{kappa}','kappa':kappa,
                               'eligible':r in eligible,'boundary':boundary}
                        observed=next(p['error'] for p in probes if p['method']==method and p['T']==T and p['r']==r)
                        entry['numerical_state']=('NUMERICAL_UNDETERMINED' if abs(observed-fraction*grid['operator_error_target'])<=grid['numerical_margin'] else
                                                   'PASS_WITH_MARGIN' if observed+grid['numerical_margin']<=fraction*grid['operator_error_target'] else 'FAIL_WITH_MARGIN')
                        if entry['numerical_state']=='NUMERICAL_UNDETERMINED': numerical_ambiguities.append(entry['id'])
                        if row['status']=='PARTIAL_ENVELOPE':
                            entry['known_T_hypothesis'],entry['T_RZ_hypothesis']=modeled_total(row,kappa)
                            entry['symbolic_T']={'Toffoli_term':7*r*row['per_step']['Toffoli'],
                                                 'RZ_multiplier':r*row['per_step']['RZ'],'T_RZ':'UNKNOWN','eta_RZ':row['eta_RZ'],
                                                 'remaining_multiplier':r*row['relative_blocks_per_step']}
                        rows.append(entry)
        for kappa in contract['resource_model']['rotation_sensitivity_kappa']:
            candidates=[x for x in rows if x.get('T')==T and x['kappa']==kappa and x['eligible'] and x['status']=='PARTIAL_ENVELOPE']
            baselines=[x for x in candidates if x['method'][0]!='R']
            baselines.sort(key=lambda x:(x['known_T_hypothesis'],x['r'],contract['methods'].index(x['method']),x['id']))
            for row in [x for x in candidates if x['method'][0]=='R']:
                b=baselines[0] if baselines else None
                slack=(b['known_T_hypothesis']-row['known_T_hypothesis'])/row['r'] if b else None
                frontiers.append({'R_id':row['id'],'BASE_id':b['id'] if b else None,'T':T,'kappa':kappa,
                                  'remaining_T_per_step':slack,'uniform_F_ceiling':slack/row['relative_blocks_per_step'] if slack is not None and row['relative_blocks_per_step'] else None,
                                  'symbolic_difference':{'BASE':b['symbolic_T'] if b else None,'R':row['symbolic_T']},
                                  'status':'POSITIVE_CONDITIONAL_FUNCTION_BUDGET' if slack is not None and slack>0 else ('NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL' if slack is not None else 'UNDETERMINED_BOUNDARY_OR_COST')})
    if len(rows)>contract['limits']['resource_model_rows']: raise ContractError('RESOURCE_ROW_CAP')
    status=('POSITIVE_CONDITIONAL_FUNCTION_BUDGET' if any(x['status']=='POSITIVE_CONDITIONAL_FUNCTION_BUDGET' for x in frontiers) else
            'NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL' if frontiers and all(x['status']=='NO_POSITIVE_BUDGET_IN_REGISTERED_MODEL' for x in frontiers) else 'UNDETERMINED_BOUNDARY_OR_COST')
    if numerical_ambiguities: status='UNDETERMINED_BOUNDARY_OR_COST'
    return {'rows':rows,'frontiers':frontiers,'precision_catalog':precision_catalog,'status':status,'function_backend':'UNKNOWN',
            'numerical_ambiguities':numerical_ambiguities,
            'qualification':'conditional documented envelopes and kappa hypothesis;no universal impossibility or total energy cost claim'}
