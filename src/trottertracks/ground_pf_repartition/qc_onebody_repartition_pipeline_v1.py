"""Registered input collection is callable only after the one-shot gate."""
from pathlib import Path
import hashlib,json,resource,time
import numpy as np
from . import qc_onebody_repartition_pilot_v1 as core
from .qc_onebody_repartition_execution_v1 import AuthorizedContext,LaunchError,write_json,digest,canonical_sha
from .qc_onebody_repartition_resources_v1 import account,comparison


def array_hash(a):
    a=np.ascontiguousarray(a); h=hashlib.sha256()
    h.update(a.dtype.str.encode());h.update(json.dumps(list(a.shape),separators=(',',':')).encode());h.update(a.tobytes())
    return h.hexdigest()


def check_resource(directory,contract):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    if rss>contract['limits']['aggregate_RSS_bytes']:
        raise core.PilotError('RESOURCE_STOP_RSS')
    if sum(p.stat().st_size for p in Path(directory).rglob('*') if p.is_file())>contract['limits']['output_bytes']:
        raise core.PilotError('RESOURCE_STOP_OUTPUT')
    return rss


def load_input(context,budget):
    budget.tick('registered_pack_loads')
    spec=context.contract['input']; path=context.root/spec['path']
    if digest(path)!=spec['SHA256']:
        raise LaunchError('REGISTERED_INPUT_CHANGED')
    with np.load(path,allow_pickle=False) as z:
        arrays={k:np.array(z[k],copy=True) for k in z.files if k!='metadata_json'}
        metadata=json.loads(str(z['metadata_json'].item()))
    for key in ['one_body','lambdas','state_vector','sector_state_vector']:
        if array_hash(arrays[key])!=metadata[key+'_hash']:
            raise core.PilotError('ARRAY_IDENTITY_'+key)
    if [array_hash(a) for a in arrays['g_matrices']]!=metadata['g_matrix_hashes']:
        raise core.PilotError('FRAGMENT_ARRAY_IDENTITY')
    if metadata['hamiltonian_hash']!=spec['Hamiltonian_hash'] or metadata['state_hash']!=spec['state_hash']:
        raise core.PilotError('CANONICAL_INPUT_IDENTITY')
    labels=arrays['sector_basis_indices'];state=arrays['sector_state_vector']
    if array_hash(labels)!=metadata['sector']['basis_indices_hash'] or array_hash(state)!=spec['sector_state_vector_hash']:
        raise core.PilotError('SECTOR_STATE_IDENTITY')
    if not np.array_equal(state,arrays['state_vector'][labels]):
        raise core.PilotError('STATE_ENDIANNESS_OR_SECTOR')
    occ=core.occupations(8,labels)
    if not np.all(occ[:,::2].sum(1)==2) or not np.all(occ[:,1::2].sum(1)==2):
        raise core.PilotError('SPIN_SECTOR')
    for a in arrays.values():
        if not np.isfinite(a).all():
            raise core.PilotError('NONFINITE_PACK')
    return arrays


def collect(design,contract,budget,output):
    """Also exercised on unregistered synthetic arrays; never loads a pack."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seal={'scope_id':contract['scope_id'],'selection':design['selector'],
          'shifts':{c['name']:c['shift'].tolist() for c in design['configurations']},
          'leading_coefficients':{c['name']:c['d'] for c in design['configurations']},
          'gamma_hash':array_hash(design['gamma']),'d0':design['configurations'][0]['d'],
          'E0_accessed_by_generator':False,'finite_step_truth_accessed_by_generator':False}
    write_json(output/'selector_seal.json',seal)  # BEFORE the first H truth solver
    seal_hash=digest(output/'selector_seal.json')
    budget.tick('H_truth_eigh_calls');truth,vectors=np.linalg.eigh(design['H'])
    E0=float(truth[0]);p0=float(abs(np.vdot(vectors[:,0],design['state']))**2)
    residual=float(np.linalg.norm(design['H']@design['state']-E0*design['state']))
    if len(truth)<2 or truth[1]-truth[0]<=1e-8 or p0<1-1e-9 or residual>1e-9:
        raise core.PilotError('CACHED_GROUND_OR_GAP_NOT_VALIDATED')
    Lambda=core.prepare_actors(design,budget)
    h_grid=[1/(8*Lambda*2**s) for s in range(6)]
    data={'one_body':design['a'],'gamma':design['gamma'],'g_matrices':design['g_matrices'],
          'lambdas':design['weights'],'constant':np.array(design['constant']),
          'state':design['state'],'labels':design['labels'],'H':design['H'],
          'H_eigenvalues':truth,'H_eigenvectors':vectors}
    records=[];config_by_name={c['name']:c for c in design['configurations']}
    for config in design['configurations']:
        config['constant']=design['constant']
        data['fragments_'+config['name']]=np.asarray(config['fragments'])
        data['shift_'+config['name']]=config['shift']
        data['orbital_frames_'+config['name']]=np.asarray([a['orbital_frame'] for a in config['actors']])
        data['actor_linear_'+config['name']]=np.asarray([a['linear'] for a in config['actors']])
        data['actor_pairs_'+config['name']]=np.asarray([a['pairs'] for a in config['actors']])
        data['actor_eta_'+config['name']]=np.asarray([a['eta'] for a in config['actors']])
        for order in [2,4]:
            for index,h in enumerate(h_grid):
                nodes=core.word(order,h,len(config['actors']))
                U=core.seed(config['actors'],nodes,design['constant'],h,budget)
                rec=core.phase_record(U,design['state'],h,E0,contract,budget)
                key=f'{config["name"]}_S{order}_h{index}'
                data[key+'_U']=U;data[key+'_vectors']=rec.pop('vectors');data[key+'_eigenvalues']=rec.pop('eigenvalues')
                rec.update({'key':key,'config':config['name'],'method':f'S{order}:'+config['name'],
                            'h':h,'h_index':index,'order':order,
                            'experiments':[{'weight':1.,'h':h,'overlap':rec['overlap'],'nodes':nodes}]})
                records.append(rec)
                check_resource(output,contract)
    primitive=list(records)
    for rec in primitive:
        if rec['order']==2:
            d=config_by_name[rec['config']]['d']
            corrected={**rec,'method':'CS2:'+rec['config'],'bias':rec['bias']-rec['h']**2*d,
                       'energy':rec['energy']-rec['h']**2*d,'scalar_correction':rec['h']**2*d}
            records.append(corrected)
    for config in design['configurations']:
        points=[r for r in primitive if r['config']==config['name'] and r['order']==2]
        for coarse,fine in zip(points[:-1],points[1:]):
            energy=(4*fine['energy']-coarse['energy'])/3
            experiments=[{**fine['experiments'][0],'weight':4/3},{**coarse['experiments'][0],'weight':-1/3}]
            records.append({'method':'EXT2:'+config['name'],'config':config['name'],
                            'h_index':coarse['h_index'],'energy':energy,'bias':energy-E0,
                            'overlap':min(coarse['overlap'],fine['overlap']),
                            'ambiguous':coarse['ambiguous'] or fine['ambiguous'],
                            'numerical_allowance':(4*fine['numerical_allowance']+coarse['numerical_allowance'])/3,
                            'experiments':experiments})
    costs=[account(r,config_by_name,contract,target,price) for r in records
           for target in contract['evaluation']['energy_targets_Ha'] for price in contract['resources']['pricing']]
    result=comparison(costs,contract)
    if digest(output/'selector_seal.json')!=seal_hash:
        raise core.PilotError('SELECTOR_SEAL_CHANGED')
    np.savez_compressed(output/'raw.npz',**data)
    raw={'scope_id':contract['scope_id'],'selector_seal_SHA256':seal_hash,
         'raw_npz_SHA256':digest(output/'raw.npz'),'h_grid':h_grid,'Lambda_common':Lambda,
         'truth':{'E0':E0,'gap':float(truth[1]-truth[0]),'cached_ground_overlap':p0,'residual':residual},
         'records':records,'costs':costs,'result':result,'counts':dict(budget.counts),
         'measurement_costs':'UNKNOWN_IDEAL_CLASSICAL_INFORMATION_LAYER_ONLY',
         'numerical_margin':'DECLARED_DOUBLE_PRECISION_SAFETY_ALLOWANCE_NOT_OUTWARD_INTERVAL_CERTIFICATION'}
    write_json(output/'raw.json',raw)
    check_resource(output,contract)
    return raw


def run_registered(context):
    if not isinstance(context,AuthorizedContext):
        raise LaunchError('AUTHORIZED_CONTEXT_REQUIRED')
    start=time.monotonic();budget=core.Budget(context.contract['limits'])
    try:
        arrays=load_input(context,budget)
        design=core.build_design(arrays['one_body'],arrays['lambdas'],arrays['g_matrices'],
                                 float(arrays['constant']),arrays['sector_basis_indices'],arrays['sector_state_vector'],budget)
        result=collect(design,context.contract,budget,context.output)
        write_json(context.output/'result_identity.json',{'input_SHA256':context.contract['input']['SHA256'],
                   'source_manifest_SHA256':digest(context.root/'artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/source_manifest.json'),
                   'contract_SHA256':digest(context.root/'artifacts/qc_onebody_repartition_finite_step_preparation_v1/2026-10-15/contract.json'),
                   'authorization_SHA256':context.authorization_SHA256,'attempt':1,'retry':0})
        stop={'status':'COLLECTION_COMPLETE_AWAIT_INDEPENDENT_AUDIT','classification':result['result']['classification'],
              'counts':dict(budget.counts),'elapsed_seconds':time.monotonic()-start,
              'peak_RSS_bytes':check_resource(context.output,context.contract),'mandatory_stop':True}
    except BaseException as error:
        stop={'status':'TECHNICAL_OR_RESOURCE_STOP_NO_RETRY','reason':str(error)[:4096],
              'counts':dict(budget.counts),'elapsed_seconds':time.monotonic()-start,'mandatory_stop':True}
        write_json(context.output/'stop.json',stop)
        raise
    write_json(context.output/'stop.json',stop)
    return stop
