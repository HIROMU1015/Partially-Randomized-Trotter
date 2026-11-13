"""Complete only the 22 missing Pauli means; retain P1 records verbatim.

No new controller, physics kernel, dictionary, N/RMSE or price calculation.
Canonical schedules and all physical/numerical conditions remain P1's.
"""
from copy import deepcopy
from fractions import Fraction

from . import df_head_pauli_tail_p1 as A

P1='e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7'
P1DIR='docs/research/df_head_pauli_tail_p1_results_20261114/'
DIRECTORY='docs/research/df_head_pauli_tail_all_round_results_20261114/'
MODULE='src/trottertracks/df_head_pauli_tail_completion.py'
PREFIXES=(1,2)
REUSED=(0,6)
MISSING=tuple(m for m in range(13) if m not in REUSED)
LOCAL='LOCAL_FINITE_MEAN_NOT_RMSE_QUALIFIED'
SUCCESS='ALL_26_PAULI_MEANS_AVAILABLE_NOT_RPE_QUALIFIED'


def load_fixed(root):
    """Metadata/content matching only, with no old numerical reanalysis."""
    meta=A.metadata(root);b=A.b_api(root)
    packet=A.read(root,P1,P1DIR+'MEAN_HANDOFF.json')
    configs={};reused={}
    if len(packet['packets'])!=2:raise ValueError('P1_TWO_PAULI_PACKETS_REQUIRED')
    for entry in packet['packets']:
        prefix=entry['prefix'];config=entry['configuration'];s=config['spec']
        b.validate(config)
        if (prefix not in PREFIXES or prefix in configs or entry['evaluated_rounds']!=list(REUSED)
                or len(entry['rounds'])!=13 or s['L_D']!=prefix or s['M']!=12
                or s['formula']!='2nd' or s['finite_K']!=2
                or A.provider._scalar(s['h'])!=Fraction(.1)
                or A.provider._scalar(s['delta_pf'])!=Fraction(.1)
                or A.provider._scalar(s['kappa']['value'])!=Fraction(meta['saved']['original_exact_kappa'])
                or s['binding']!=meta['saved']['binding']
                or s['profile']!=meta['original_DF_configurations'][prefix]['spec']['profile']
                or s['rmse_policy']!=meta['original_DF_configurations'][prefix]['spec']['rmse_policy']
                or s['split']['tail_dictionary']!=A.dictionary_identity(meta['dictionaries'][prefix])):
            raise ValueError('P1_CANONICAL_FIXED_CONDITIONS_DIFFER')
        configs[prefix]=deepcopy(config)
        for row,canonical in zip(entry['rounds'],config['rounds']):
            m=row['m']
            if (row['identity']!=canonical['identity'] or row['law_binding']!=canonical['law_binding']
                    or row['candidate_id']!=canonical['candidate']['id']):
                raise ValueError('P1_ROUND_IDENTITY_OR_LAW')
            if m in REUSED:
                record=row['mean_record'];path=P1DIR+'run01/LD'+str(prefix)+'_m'+str(m)+'_kappa.json'
                original=A.read(root,P1,path)['record']
                if (record!=original or record['status']!=LOCAL
                        or record['numerical'].get('gate_pass') is not True
                        or record['B_candidate_id']!=canonical['candidate']['id']
                        or record['B_identity']!=canonical['identity']
                        or record['B_law_binding']!=canonical['law_binding']
                        or record['source']['commit']!='05447d1d4f1e3c0e9e335a7085aa58a877c139f2'):
                    raise ValueError('P1_SAVED_MEAN_SOURCE_OR_STATUS')
                reused[prefix,m]=dict(commit=P1,path=path,selector=['record'],
                    scientific_source=record['source']['commit'])
            elif row['mean_record'] is not None or row['status']!='NOT_EVALUATED':
                raise ValueError('P1_REMAINING_ROUND_NOT_MISSING')
    if set(configs)!=set(PREFIXES) or len(reused)!=4:
        raise ValueError('P1_COVERAGE')
    return meta,b,configs,packet,reused


def run_missing(jobs,evaluate,save,result):
    """Fixed round-major order; first FAIL/exception stops all further calls."""
    expected={(p,m) for p in PREFIXES for m in MISSING}
    if set(jobs)!=expected:raise ValueError('ONLY_THE_22_FIXED_MISSING_JOBS')
    for m in MISSING:
        for prefix in PREFIXES:
            if len(result['records'])>=22:raise ValueError('SCIENTIFIC_CALL_SCOPE_EXHAUSTED')
            record=evaluate(prefix,m,jobs[prefix,m])
            item=dict(prefix=prefix,m=m,space='sector36',kind='COMMON_KAPPA',record=record)
            result['records'].append(item)
            save('LD'+str(prefix)+'_m'+str(m)+'.json',item)
            save('CHECKPOINT.json',result)
            job=jobs[prefix,m]
            if (record.get('B_candidate_id')!=job['B_candidate_id']
                    or record.get('B_configuration_id')!=job['B_configuration_id']
                    or record.get('candidate',{}).get('id')!=job['B_candidate_id']
                    or record.get('m')!=m
                    or record.get('B_identity')!=job['B_identity']
                    or record.get('B_law_binding')!=job['B_law_binding']
                    or record.get('source')!=job['physical_context']['numerical_recipe']['source']):
                result.update(status='STOP_RETURNED_IDENTITY_SOURCE_OR_LAW_MISMATCH',
                    error=dict(prefix=prefix,m=m,reason='Returned record differs from fixed request'))
                return False
            if record['status']!=LOCAL or record.get('numerical',{}).get('gate_pass') is not True:
                result['status']='STOP_NUMERICAL_FAILURE'
                return False
    result['status']=SUCCESS
    return True


def join_packets(previous,new_items,*,raw_commit,science_source):
    """Source-bearing packet assembly only; never recalculate stored values."""
    bykey={}
    for item in new_items:
        key=item['prefix'],item['m']
        if key in bykey or key[0] not in PREFIXES or key[1] not in MISSING:
            raise ValueError('DUPLICATE_OR_OUTSIDE_NEW_SCOPE')
        bykey[key]=item['record']
    packets=[]
    for old in previous['packets']:
        packet=deepcopy(old);prefix=packet['prefix'];config=packet['configuration']
        packet.update(kind='A_PAULI_ALL_ROUND_MEANS_ONLY_NOT_N_PACKET',
            scientific_source=science_source,completion_scientific_source=science_source,
            P1_scientific_source=old['scientific_source'],per_round_source_is_authoritative=True,
            reused_rounds=list(REUSED),new_rounds=[],evaluated_rounds=[],failed_rounds=[])
        for row,canonical in zip(packet['rounds'],config['rounds']):
            m=row['m'];key=prefix,m
            if m in REUSED:
                row['mean_source']=dict(commit=P1,path=P1DIR+'run01/LD'+str(prefix)+'_m'+str(m)+'_kappa.json',
                    selector=['record'],scientific_source=row['mean_record']['source']['commit'])
                row['reused_without_reanalysis']=True
                packet['evaluated_rounds'].append(m)
                continue
            if key not in bykey:continue
            record=bykey[key]
            if (record['B_configuration_id']!=config['id'] or record['B_candidate_id']!=canonical['candidate']['id']
                    or record['B_identity']!=canonical['identity'] or record['B_law_binding']!=canonical['law_binding']
                    or record['candidate']['id']!=canonical['candidate']['id']
                    or record['source']['commit']!=science_source):
                raise ValueError('NEW_MEAN_CANONICAL_ID_SOURCE_OR_LAW_MISMATCH')
            for k in ('N_X','N_Y','C','G','global_RMSE_bound','scientific_PASS_FAIL'):
                if record.get(k) is not None:raise ValueError('UNAUTHORIZED_QUALIFICATION_OR_COST')
            row.update(status=record['status'],mean_record=deepcopy(record),
                mean_source=dict(commit=raw_commit,path=DIRECTORY+'run01/LD'+str(prefix)+'_m'+str(m)+'.json',
                    selector=['record'],scientific_source=science_source),reused_without_reanalysis=False)
            packet['new_rounds'].append(m)
            if record['status']==LOCAL and record.get('numerical',{}).get('gate_pass') is True:
                packet['evaluated_rounds'].append(m)
            else:packet['failed_rounds'].append(m)
        packet['signal_coverage_complete']=(packet['evaluated_rounds']==list(range(13)))
        if not packet['signal_coverage_complete']:
            packet['kind']='A_PAULI_PARTIAL_MEANS_ONLY_NOT_N_PACKET'
        packets.append(packet)
    complete=all(p['signal_coverage_complete'] for p in packets)
    return dict(kind='A_PAULI_ALL_ROUND_SIGNAL_HANDOFF' if complete else 'A_PAULI_PARTIAL_SIGNAL_HANDOFF',
        status=SUCCESS if complete else 'INCOMPLETE_NUMERICAL_OR_INPUT_STOP',packets=packets,
        raw_result_source=dict(commit=raw_commit,path=DIRECTORY+'run01/result.json'),
        reused_source=dict(commit=P1,path=P1DIR+'MEAN_HANDOFF.json'),
        P1_fixed_R_full_control_source=dict(commit=P1,path=P1DIR+'MEAN_HANDOFF.json',selector=['control_evidence']),
        sufficient_N=None,global_RMSE_bound=None,C=None,G=None,winner=None,mandatory_stop=True)
