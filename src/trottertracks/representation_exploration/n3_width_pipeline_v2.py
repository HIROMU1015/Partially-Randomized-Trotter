"""Two comparison layers and separate same-mode / Hamiltonian-level endpoints."""
from fractions import Fraction as F
import time

from .n3_width_contract_v2 import METHODS,active_data,decode_case,enc,digest
from .n3_width_generator_v2 import Engine,acquire_bounds,acquire_mode,upper_certificate,spectral_certificate
from .n3_width_check_v2 import verify_full_lower
from . import n3_width_algebra_v2 as car

REPETITIONS=3
LAYERS=('kernel','cold')


def full_h_case(case):
    e=Engine();start=time.perf_counter();cpu=time.process_time();h,J,nu,modes,eps=decode_case(case)
    with e.phase('bound_acquisition'):spec=spectral_certificate(h,nu,e)
    L=F(*spec['s']);rows=[]
    for d in modes:
        with e.phase('bound_acquisition'):upper=upper_certificate(case,d)
        U=F(*upper['U']);full=enc({'spectral':spec,'L_H':L,'width':U-L})
        with e.phase('independent_checker'):audit=verify_full_lower(case,d,upper,full)
        w={'kind':'FULL_H_LOWER','case_sha256':digest(case),'external':d,'upper':upper,'full_lower':full,
            'width':{'low':enc(U-L),'high':enc(U-L),'positive_proof_delta':enc(max(U-L,F(1,1<<128)))}}
        rows.append({'method':'FULL_H_LOWER','external':d,'status':'CERTIFICATES_FOUND','bounds':None,
                     'candidates':[w],'best_index':0,'audits':[audit],'refutations':[]})
    return {'method':'FULL_H_LOWER','mode_records':rows,'resources':e.stats(),
            'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
            'full_lower_acquired_once_per_H':True,'comparison_layer':'own_coefficient_lower_path'}


def select_modes(case,mode_records):
    eps=F(*case['epsilon_proj']);rows=[]
    for row in mode_records:
        candidates=row['candidates'];best=min(range(len(candidates)),key=lambda i:(F(*candidates[i]['width']['high']),i)) if candidates else None
        width=candidates[best]['width'] if best is not None else None
        rows.append({'external':row['external'],'status':row['status'],'best_index':best,'width':width,
                     'accepted':width is not None and F(*width['positive_proof_delta'])<=eps})
    accepted=[r for r in rows if r['accepted']]
    chosen=min(accepted,key=lambda x:(F(*x['width']['high']),x['external'])) if accepted else None
    if chosen:
        active,A,v,J,E,hdd,nu,eps=active_data(case,chosen['external'])
        reduced=enc({'active_to_full':active,'nu':nu,'h':A,'density':[[i,j,x] for (i,j),x in sorted(J.items()) if x]})
    else:reduced=None
    return {'same_mode_endpoints':rows,'H_any_mode_reduction':bool(accepted),
            'selected_external':chosen['external'] if chosen else None,'modes_removed':1 if chosen else 0,
            'reduced_H':reduced,'fallback':'ORIGINAL_FULL_H' if not chosen else None,
            'actual_projection_error_or_quantum_cost_claimed':False}


def summarize_block(case,methods):
    answer={name:select_modes(case,rec['mode_records']) for name,rec in methods.items()}
    cheap=('NORM','FIVE','OLD','FULL_H_LOWER');cheap_rows=[];op_rows=[]
    for d in case['candidate_modes']:
        pool=[]
        for name in cheap:
            row=next(x for x in methods[name]['mode_records'] if x['external']==d)
            pool.extend((name,w) for w in row['candidates'])
        cheap_rows.append({'external':d,'status':'OPERATIONAL_CHEAP_OR','candidates':[w for _,w in pool]})
        own=next(x for x in methods['OCC']['mode_records'] if x['external']==d)['candidates']
        all_pool=pool+[('OCC_RAW',w) for w in own]
        best=min(all_pool,key=lambda x:(F(*x[1]['width']['high']),x[0])) if all_pool else None
        op_rows.append({'external':d,'status':'OPERATIONAL_SAFE_OR','candidates':[w for _,w in all_pool],
                        'origin':best[0] if best else None})
    answer['CHEAP_OR']=select_modes(case,cheap_rows);answer['OCC_OPERATIONAL_OR']=select_modes(case,op_rows)
    answer['OCC_OPERATIONAL_OR']['origins']=[{'external':r['external'],'origin':r['origin']} for r in op_rows]
    answer['OCC_OPERATIONAL_OR']['fallback_is_OCC_specific_gain']=False
    answer['OCC_OPERATIONAL_OR']['cost_qualification']='Coverage OR from saved valid certificates; no separate operational execution time measured'
    for name in ('SDP_EQUIVALENT','DENSITY_SOS'):
        rows=[];origins=[]
        for d in case['candidate_modes']:
            pool=[]
            for cheap_name in cheap:
                row=next(x for x in methods[cheap_name]['mode_records'] if x['external']==d);pool.extend((cheap_name,w) for w in row['candidates'])
            own=next(x for x in methods[name]['mode_records'] if x['external']==d)['candidates'];pool.extend((name+'_RAW',w) for w in own)
            best=min(pool,key=lambda x:(F(*x[1]['width']['high']),x[0])) if pool else None
            rows.append({'external':d,'status':'OPERATIONAL_SAFE_OR','candidates':[w for _,w in pool]});origins.append({'external':d,'origin':best[0] if best else None})
        answer[name+'_OPERATIONAL_OR']=select_modes(case,rows)
        answer[name+'_OPERATIONAL_OR'].update(origins=origins,fallback_is_OCC_specific_gain=False,
            cost_qualification='Coverage OR from saved valid certificates; no separate operational execution time measured')
    return answer


def acquire_case(case):
    """Only runner invokes on registered science inputs after authorization."""
    kernel_bounds={};kernel_costs=[]
    for d in case['candidate_modes']:
        e=Engine();kernel_bounds[d]=acquire_bounds(case,d,e);kernel_costs.append({'external':d,'resources':e.stats()})
    blocks=[]
    for layer in LAYERS:
        for repetition in range(REPETITIONS):
            methods={}
            for name in METHODS:
                # Fresh primitive/proof cache per method path; universal CAR cached within that path.
                car.structure.cache_clear();car.normal.cache_clear();start=time.perf_counter();cpu=time.process_time()
                if name=='FULL_H_LOWER':methods[name]=full_h_case(case)
                else:
                    rows=[acquire_mode(case,d,name,kernel_bounds[d] if layer=='kernel' else None) for d in case['candidate_modes']]
                    methods[name]={'method':name,'mode_records':rows,'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                                   'provided_U_c':layer=='kernel'}
                if any('STOP' in r['status'] for r in methods[name]['mode_records']):
                    return {'input_id':case['id'],'case_sha256':digest(case),'status':'TECHNICAL_OR_SOUNDNESS_STOP',
                            'partial_block':{'layer':layer,'repetition':repetition,'methods':methods},'completed_blocks':blocks,
                            'mandatory_stop':True,'next_stage_authorized':False}
            blocks.append({'layer':layer,'repetition':repetition,'methods':methods,'endpoints':summarize_block(case,methods)})
    return {'input_id':case['id'],'case_sha256':digest(case),'status':'FINITE_COMPARISON_COMPLETE',
            'common_kernel_bound_costs':kernel_costs,'blocks':blocks,'timing_repetitions_are_independent_inputs':False,
            'mandatory_stop':True,'next_stage_authorized':False,'central_theme':None}
