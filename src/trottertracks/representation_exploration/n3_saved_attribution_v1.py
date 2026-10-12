"""Saved-literal indexing and separately gated posthoc arithmetic.

No producer, numerical solver or eigensolver imports. metadata() copies saved
fields and pointers; it does not evaluate certificates or compute new timings.
Arithmetic functions are used only by a separately authorized runner or tests.
"""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

TARGETS = ('OCC', 'SDP_EQUIVALENT', 'DENSITY_SOS')
TIERS = ([1,100], [1,200], [1,400])
BITS = 4096


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pointer(value, path):
    for part in path.strip('/').split('/'):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def rational(value):
    if not isinstance(value,list) or len(value)!=2 or any(type(x) is not int for x in value) or value[1]<=0:
        raise ValueError('RATIONAL_SCHEMA')
    if max(abs(x).bit_length() for x in value)>BITS:
        raise ArithmeticError('RATIONAL_BIT_CAP')
    return Fraction(*value)


def encoded(value):
    if max(abs(x).bit_length() for x in (value.numerator,value.denominator))>BITS:
        raise ArithmeticError('RATIONAL_BIT_CAP')
    return [value.numerator,value.denominator]


def metadata(case, record, relative_path):
    if record['input_id']!=case['id'] or record['case_sha256']!=digest(case):
        raise ValueError('SAVED_CASE_IDENTITY')
    if record['status']!='FINITE_COMPARISON_COMPLETE' or record['mandatory_stop'] is not True:
        raise ValueError('SAVED_COMPLETION_AND_STOP')
    rows=[];events=[];candidates=[]
    for bi,block in enumerate(record['blocks']):
        for name in TARGETS:
            path=block['methods'][name];base=f'/blocks/{bi}/methods/{name}'
            if path['method']!=name:raise ValueError('METHOD_IDENTITY')
            for ri,row in enumerate(path['mode_records']):
                pre=base+f'/mode_records/{ri}'
                if row['method']!=name:raise ValueError('ROW_METHOD_IDENTITY')
                resource=row['resources'];solver=resource['solver_events']
                if len(solver)!=(0 if name=='OCC' else 3):raise ValueError('FROZEN_EVENT_COUNT')
                if name!='OCC' and len(row['candidates'])!=3:
                    raise ValueError('MISSING_TIER_CANDIDATE_MAPPING_UNDETERMINED')
                common={'input_id':case['id'],'layer':block['layer'],'repetition':block['repetition'],
                        'method':name,'external':row['external'],'file':relative_path}
                rows.append({**common,'pointer':pre,'status':row['status'],'best_index':row['best_index'],
                    'candidate_count':len(row['candidates']),'refutation_count':len(row['refutations']),
                    'saved_U':deepcopy(row['bounds']['upper']['U']),'saved_c':deepcopy(row['bounds']['c']),
                    'row_wall_seconds':row['wall_seconds'],'row_cpu_seconds':row['cpu_seconds'],
                    'resources':deepcopy(resource),'method_path_times':{k:v for k,v in path.items() if k.endswith('seconds')},
                    'new_arithmetic_performed':False})
                for ei,event in enumerate(solver):
                    events.append({**common,'pointer':pre+f'/resources/solver_events/{ei}',
                                   'event_index':ei,'tier_delta':deepcopy(TIERS[ei]),'saved_event':deepcopy(event),
                                   'mapping_basis':'Frozen source loop order; all three events and candidate slots present.'})
                for ci,candidate in enumerate(row['candidates']):
                    params=candidate['parameters']
                    candidates.append({**common,'pointer':pre+f'/candidates/{ci}','candidate_index':ci,
                        'selected_in_saved_row':ci==row['best_index'],'kind':candidate['kind'],
                        'kappa':deepcopy(candidate['kappa']),'s':deepcopy(candidate['s']),
                        'saved_width':deepcopy(candidate['width']),
                        'saved_parameter_scalars':{k:deepcopy(params[k]) for k in ('tau','s','ell','residual_bound') if k in params},
                        'raw_float_proposal':'NOT_SAVED','pre_repair_objective':'NOT_SAVED',
                        'repair_shift_r':'NOT_SAVED','pre_repair_matrix':'NOT_SAVED',
                        'solver_numerical_residuals_and_gap':'NOT_SAVED',
                        'origin_tier_delta':deepcopy(TIERS[ci]) if name!='OCC' else None})
    return {'kind':'SAVED_LITERAL_EXTRACTION_NO_NEW_NUMERIC_EVALUATION','input_id':case['id'],
            'rows':rows,'solver_events':events,'candidates':candidates,'new_producer_calls':0,
            'new_eigensolver_calls':0,'new_solver_calls':0,'new_PSD_certification_calls':0,
            'new_numeric_posthoc_evaluations':0}


def phase_summary(row):
    phases=row['resources']['phase_times']
    for times in phases.values():
        for v in times.values():
            if type(v) not in (float,int) or not math.isfinite(v) or v<0:
                raise ValueError('TIMING_SCHEMA')
    wall=sum(x['wall'] for x in phases.values());cpu=sum(x['cpu'] for x in phases.values())
    events=row['resources']['solver_events']
    solve=sum(e['reported_solve_time'] for e in events if e['reported_solve_time'] is not None)
    setup=sum(e['reported_setup_time'] for e in events if e['reported_setup_time'] is not None)
    return {'exclusive_phase_wall_sum':wall,'exclusive_phase_cpu_sum':cpu,
            'row_wall_minus_exclusive_phase_sum_unassigned':row['wall_seconds']-wall,
            'row_cpu_minus_exclusive_phase_sum_unassigned':row['cpu_seconds']-cpu,
            'reported_backend_solve_time_sum':solve,'reported_backend_setup_time_sum':setup,
            'numerical_proposal_minus_backend_times_mixed_unassigned':phases.get('numerical_proposal',{}).get('wall',0)-solve-setup,
            'negative_remainders_retained':True,
            'refutation_only_time':'NOT_SEPARATELY_MEASURED',
            'CVXPY_problem_canonicalization_interface_postprocess_split':'NOT_SEPARATELY_MEASURED',
            'hypothetical_refutation_free_or_one_call_time':'NOT_COMPUTED'}


def candidate_summary(candidate, bounds, nu):
    k=rational(candidate['kappa']);s=rational(candidate['s'])
    U=rational(bounds['upper']['U']);c=rational(bounds['c']);g=c-U
    if type(nu) is not int or nu<=0 or k<0 or g<=0:raise ValueError('FIXED_DOMAIN')
    q=k*U-s;box=max(Fraction(0),k-g)
    if q<=0:
        active='BOX' if box else 'ZERO_FLOOR'
    else:
        val=box*(g+box)
        active='BOX' if val>q else 'BOX_ROOT_TIE' if val==q else 'ROOT'
    result={'kappa':encoded(k),'saved_lower_s':encoded(s),'U':encoded(U),'c':encoded(c),
            'g':encoded(g),'q_final':encoded(q),'box_term':encoded(box),'active_term':active,
            'saved_width':deepcopy(candidate['width']),
            'pre_repair_proposal_and_shift':'NOT_SAVED_CAUSAL_REPAIR_ATTRIBUTION_UNDETERMINED',
            'ideal_minimum_width_or_new_certificate':'NOT_COMPUTED'}
    params=candidate['parameters']
    if candidate['kind']=='SOS':
        ell=rational(params['ell']);residual=rational(params['residual_bound'])
        if residual<0 or ell-residual!=s:raise ValueError('SOS_SAVED_RESIDUAL_BINDING')
        result.update(saved_ell=encoded(ell),saved_residual_bound=encoded(residual),
                      q_using_saved_rounded_ell=encoded(k*U-ell),
                      residual_penalty_in_q=encoded(residual),
                      ell_interpretation='Saved rounded ell; not the unsaved raw numerical/pre-repair optimum.')
    else:
        tau=rational(params['tau']);Z=params['Z'];trace=sum((rational(Z[i][i]) for i in range(len(Z))),Fraction())
        if nu*tau-trace!=s:raise ValueError('OCC_SAVED_TRACE_BINDING')
        result.update(saved_tau=encoded(tau),saved_trace_Z=encoded(trace),
                      repair_penalty_m_times_r='UNDETERMINED_R_NOT_SAVED')
    return result


def analyze(case,record,relative_path):
    indexed=metadata(case,record,relative_path);rows=[];cand=[]
    for item in indexed['rows']:
        raw=pointer(record,item['pointer']);rows.append({**item,'posthoc_phase_summary':phase_summary(raw),
                                                        'new_arithmetic_performed':True})
    for item in indexed['candidates']:
        raw=pointer(record,item['pointer']);mode=pointer(record,item['pointer'].rsplit('/candidates/',1)[0])
        cand.append({**item,'posthoc_scalar_summary':candidate_summary(raw,mode['bounds'],case['nu'])})
    groups={}
    for bi,block in enumerate(record['blocks']):
        for name in TARGETS:
            times=block['methods'][name];key=block['layer']+':'+name
            groups.setdefault(key,[]).append(times['task_wall_including_selector_serialization'])
    return {'kind':'NEW_NUMERIC_POSTHOC_OF_SAVED_DATA_NOT_NEW_SCIENCE_BATCH','input_id':case['id'],
            'rows':rows,'candidates':cand,'solver_events':indexed['solver_events'],
            'path_wall_medians_and_ranges':{k:{'minimum':min(v),'median':statistics.median(v),'maximum':max(v)} for k,v in groups.items()},
            'new_producer_calls':0,'new_eigensolver_calls':0,'new_solver_calls':0,
            'causal_time_or_width_difference_fully_identified':False,'mandatory_stop':True}


def assemble_connection(case,record,relative_path,source_sha):
    """Copy existing witnesses and pointers only; no new acceptance/PSD check."""
    if case['id']!='V2_two_external_stress_5':raise ValueError('CONNECTION_CASE')
    block=record['blocks'][0]
    if block['layer']!='kernel' or block['repetition']!=0:raise ValueError('CONNECTION_BLOCK')
    def selected(name,d):
        rows=block['methods'][name]['mode_records'];ri=next(i for i,r in enumerate(rows) if r['external']==d)
        r=rows[ri];ci=r['best_index'];base=f'/blocks/0/methods/{name}/mode_records/{ri}'
        return r,base,ci
    sos,sbase,si=selected('DENSITY_SOS',4);dual,dbase,_=selected('SDP_EQUIVALENT',4)
    other,obase,oi=selected('OCC',0)
    refi=next(i for i,r in enumerate(dual['refutations']) if r['delta']==[1,400])
    if sos['bounds']!=dual['bounds']:raise ValueError('CONNECTION_SAME_U_C')
    witnesses=[('SOS_external4',sbase+f'/candidates/{si}',sos['candidates'][si]),
               ('OCC_refutation_external4',dbase+f'/refutations/{refi}',dual['refutations'][refi]),
               ('OCC_winner_external0',obase+f'/candidates/{oi}',other['candidates'][oi])]
    return {'kind':'EXISTING_PROOF_POINTER_AND_LITERAL_CONNECTION_NOT_RECERTIFIED',
            'case':deepcopy(case),'target_delta':[1,400],'source_file':relative_path,'source_file_sha256':source_sha,
            'fixed_result_commit':'6d8f769b857e50c8d262e7a4e94f8277a8967597','bounds_external4':deepcopy(sos['bounds']),
            'bounds_external0':deepcopy(other['bounds']),
            'witnesses':[{'label':label,'pointer':pre,'sha256':digest(value),'value':deepcopy(value)} for label,pre,value in witnesses],
            'existing_audit':'artifacts/n3_width_comparison_v2/2026-10-12/run1_audit/independent_saved_audit.json',
            'new_PSD_certification_calls':0,'new_numeric_posthoc_evaluations':0,
            'review_conclusion':'Reviewed same-mode SOS/ideal-OCC separation; other external0 also admits one-mode H reduction.',
            'matrices_rounded_or_modified':False}


def recertify_connection(bundle):
    """Authorized posthoc only: replay three saved proofs, create no witnesses."""
    from .n3_width_contract_v2 import active_data
    from .n3_width_check_v2 import Checks,verify_bounds,verify_witness,verify_sdp_refutation
    case=bundle['case'];values={w['label']:w['value'] for w in bundle['witnesses']};bounds=bundle['bounds_external4']
    sos=values['SOS_external4'];delta=rational(bundle['target_delta'])
    positive=verify_witness(case,4,bounds,sos)
    U,c=verify_bounds(case,4,bounds,Checks());k=rational(sos['kappa']);s=rational(sos['s'])
    if not (0<=k<=c-U+delta and k*U-s<=delta*(c-U+delta)):
        raise ValueError('SOS_TARGET_NOT_CERTIFIED')
    _,A,v,J,E,hdd,nu,eps=active_data(case,4);ref=values['OCC_refutation_external4']
    if ref['delta']!=bundle['target_delta']:raise ValueError('TARGET_BINDING')
    negative=verify_sdp_refutation(A,v,nu,U,c,delta,ref['certificate'])
    other=verify_witness(case,0,bundle['bounds_external0'],values['OCC_winner_external0'])
    if rational(values['OCC_winner_external0']['width']['positive_proof_delta'])>delta:
        raise ValueError('OTHER_MODE_TARGET')
    return {'status':'EXISTING_SOS_POSITIVE_OCC_NEGATIVE_SAME_QUERY_AND_OTHER_MODE_PROOFS_RECHECKED',
            'SOS_audit':positive,'OCC_dual_audit':negative,'other_mode_OCC_audit':other,
            'target_delta':bundle['target_delta'],'physical_impossibility_claimed':False,
            'additional_H_reduction_claimed':False,'new_solver_calls':0,'new_certificate_searches':0}
