"""Collect only a completed fresh confirmation campaign; no n8 pooling."""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

from h46_confirmation_contract import GROUPS, PAIRS, PROFILE, PROPOSED_N, task_id, sha
from h46_higher_pf_adapter import kernel


def compare(summaries):
    """Exact point signs plus explicitly approximate independent-candidate diagnostics."""
    if set(summaries)!=set(PROPOSED_N):raise ValueError('THREE_COMPLETE_CANDIDATES_REQUIRED')
    critical=NormalDist().inv_cdf(1-.05/(2*2));rows=[]
    for left,right in PAIRS:
        a,b=summaries[left],summaries[right]
        difference=Fraction(a['mean_exact'])-Fraction(b['mean_exact'])
        if a['n']!=PROPOSED_N[left] or b['n']!=PROPOSED_N[right]:raise ValueError('NO_EXPLORATION_POOLING_OR_SAMPLE_EXPANSION')
        se=math.sqrt(float(Fraction(a['variance_exact']))/a['n']+float(Fraction(b['variance_exact']))/b['n'])
        if not math.isfinite(se):raise ValueError('NONFINITE_DIFFERENCE_SE')
        interval=[float(difference)-critical*se,float(difference)+critical*se]
        rows.append({'left':left,'right':right,'difference_exact':str(difference),'difference':float(difference),
            'point_order':'REPRODUCED' if difference>0 else 'TIE' if difference==0 else 'NOT_REPRODUCED',
            'independent_candidate_SE':se,'normal_Bonferroni_approx_interval':interval,
            'normal_critical':critical,'family_alpha':.05,'family_size':2,
            'approx_nominal_interval_excludes_zero_in_expected_direction':interval[0]>0 if se>0 else None,
            'zero_sample_variance_is_population_certainty':False,'coverage_guarantee':False,
            'independence_assumption':'DISTINCT_CANDIDATE_SEED_STREAMS_NO_CRN_COVARIANCE_ASSUMED',
            'population_ranking_certified':False})
    return rows


def collect(config):
    out=Path(config['output']);k=kernel();candidates={};raremetadata={};checks_total=0;ledger_expected=set()
    for group in GROUPS:
        gid=group['id'];binding=config['group_binding'][gid]
        identity={'SOURCE':config['SOURCE'],'input_sha256':config['plan']['input_sha256'],
                  'preparation_hash':binding['preparation_hash'],'compiler_settings_hash':PROFILE,
                  'series':k.SERIES,'control':'ordinary','formula_id':k.FORMULAS[group['label']],
                  'system':'H6','prefix':5,'q':group['q'],'stage':'CONFIRMATION'}
        records=[];strata={'s0':0,'s2_single':0,'s2_multi':0,'s4':0}
        for sample in range(PROPOSED_N[gid]):
            directory=out/task_id(config['plan'],group,sample)
            raw=json.loads((directory/'result.json').read_text());env=json.loads((directory/'confirmation.json').read_text())
            if env.get('status')!='CONFIRMATION_COST_VALIDATED' or env.get('stage')!='CONFIRMATION' or env.get('study_id')!=config['plan']['study_id']:
                raise ValueError('CONFIRMATION_ENVELOPE_REQUIRED')
            for key,value in [('SOURCE',config['SOURCE']),('plan_sha256',config['plan_sha256']),('group',group),('sample',sample)]:
                if env.get(key)!=value or raw.get(key)!=value:raise ValueError('CONFIRMATION_IDENTITY:'+key)
            if env.get('seed_scope_digest')!=config['plan']['seed_scope_digest'] or env.get('raw_native_result_sha256')!=sha(directory/'result.json'):
                raise ValueError('CONFIRMATION_RAW_SEAL')
            if (raw.get('status')!='ACQUIRED_VALIDATED_NO_ACCOUNTING' or raw.get('forced_events_enter_cost_sample') is not False or
                raw.get('preparation_hash')!=identity['preparation_hash'] or raw.get('input_sha256')!=identity['input_sha256'] or
                raw.get('seed_namespace')!=config['plan']['seed_namespace'] or raw.get('seed_registry_sha256')!=config['seed_registry_sha256'] or
                raw.get('provider_SOURCE')!=config['provider_SOURCE']):raise ValueError('RAW_PROVENANCE')
            checks=raw['validation_checks'];expected=12 if sample==0 else 5
            if len(checks)!=expected or any(not math.isfinite(v['error']) or v['error']<0 or v['error']>1e-9 for v in checks):
                raise ValueError('NATIVE_CONTROL_NOT_PASS')
            checks_total+=len(checks)
            sealed_paths={str(Path(e['path']).resolve()) for e in config['sealed_files']}
            if not raw.get('imported_provider_modules') or any(str(Path(path).resolve()) not in sealed_paths for path in raw['imported_provider_modules'].values()):
                raise ValueError('PROVIDER_IMPORT_OUTSIDE_SEALED_CLOSURE')
            events=json.loads((directory/'events_PRIVATE.json').read_text())
            if events.get('conditional') is not False or events.get('seed_namespace')!=config['plan']['seed_namespace']:raise ValueError('UNCONDITIONAL_FRESH_EVENTS')
            event_hash=hashlib.sha256(json.dumps(events['events'],sort_keys=True).encode()).hexdigest()
            orders=[v['taylor_order'] for occurrence in events['events'] for v in occurrence]
            if any(v not in (0,2,4) for v in orders):raise ValueError('TAYLOR_ORDER')
            key='s4' if 4 in orders else 's2_multi' if orders.count(2)>1 else 's2_single' if 2 in orders else 's0'
            strata[key]+=1
            if len(raw['records'])!=2 or {r['axis'] for r in raw['records']}!={'cosine','sine'}:raise ValueError('PAIRED_AXES')
            for record in raw['records']:
                axis=record['axis']
                if record['event_digest']!=event_hash or record['compiler_settings_hash']!=PROFILE or record['sample']!=sample:
                    raise ValueError('EVENT_COMPILER_SAMPLE')
                if json.loads((directory/('cost_'+axis+'.json')).read_text())!=record:raise ValueError('AXIS_RECEIPT')
                ledger_name=gid+'__%02d__'%sample+axis+'.json';ledger_expected.add(ledger_name)
                entry=json.loads((Path(config['call_ledger'])/ledger_name).read_text())
                expected_entry={'state':'ENTERED_NO_REFUND','group':gid,'sample':sample,'axis':axis,
                                'event_digest':event_hash,'fingerprint':record['fingerprint'],'SOURCE':config['SOURCE']}
                if entry!=expected_entry:raise ValueError('NONREFUNDABLE_LEDGER_IDENTITY')
                records.append({**identity,'replica_id':str(sample),'axis':axis,'event_digest':event_hash,
                    'seed_namespace':config['plan']['seed_namespace'],'forced_event':False,'cost_kind':'MEASURED',**record['metrics']})
        shots={a:binding['shots']['axes'][a]['shots'] for a in ('real','imag')}
        candidates[gid]={'identity':identity,'shots_reused_without_recomputation':shots,
                         **k.paired_costs(records,shots,identity)}
        raremetadata[gid]={'observed_trajectory_strata':strata,'unobserved_stratum_cost':None,'population_variance_upper_bound':None}
    if {p.name for p in Path(config['call_ledger']).glob('*.json')}!=ledger_expected or len(ledger_expected)!=206 or checks_total!=536:
        raise ValueError('COMPLETE_FIXED_BUDGET_OR_CHECKS')
    comparisons=compare({gid:v['summaries']['RZ'] for gid,v in candidates.items()})
    return {'status':'COMPLETE_PURPOSE_A_DIAGNOSTIC','SOURCE':config['SOURCE'],'plan_sha256':config['plan_sha256'],
        'stage':'CONFIRMATION','candidates':candidates,'comparisons':comparisons,'rare_event_coverage':raremetadata,
        'point_order_status':'BOTH_REPRODUCED' if all(r['point_order']=='REPRODUCED' for r in comparisons) else 'ONE_OR_MORE_NOT_REPRODUCED_OR_TIED',
        'control_checks':checks_total,'compiled_axes':206,'fresh_trajectories':103,'exploration_pooled':False,
        'shot_recomputed':False,'new_q_selection':False,'population_ranking_certified':False,
        'rare_event_mean_guarantee':False,'strict_bias_guarantee':False,'P':0,'automatic_n_increase':False,'mandatory_STOP':True}
