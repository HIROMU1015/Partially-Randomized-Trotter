"""Saved RC2/PR scalar shots only. Passive preflight; no physics/circuit entry.

Reuse C1 qualifications/accuracy/energy and the existing shot-range extension.
Scientific evaluation requires separate chat permission; imports do no work.
"""
from fractions import Fraction
import csv
import importlib.metadata
import math
from pathlib import Path
import resource
import subprocess
import sys
import time

from . import c_rc2_pr_stage_b as stage_b
from . import c_pr_aware_df_stage_c as c1
from . import c1_stage3_shots_screen as screen

io, save, accuracy = stage_b.io, stage_b.save, c1.accuracy
BASE = 'e33ab6e03344c8179ff9b378dc446bf4cfc55e10'
STAGE_B_SOURCE = 'd6b17b79461a39bd8ba2e6b7389a1f4fad4d74e6'
VERSION = 'saved_rc2_pr_conditional_shots_v1'
SCOPE = 'C_RC2_PR_SAVED_CONDITIONAL_SHOTS_WEIGHT_V1'
FAMILY = 'artifacts/research_c_rc2_pr_conditional_shots/2026-11-01'
FRESH = FAMILY + '/run01'
ROLES, LANES, TARGET = stage_b.ROLES, c1.LANES, c1.TARGET
# Same numerical assumptions as C1, without its old B2 cost-proxy assumption.
ASSUMPTIONS = {s['id']:[x for x in s['assumptions'] if 'proxy' not in x] for s in c1.SCENARIOS}
SCENARIOS = [dict(s,assumptions=ASSUMPTIONS[s['id']]) for s in c1.SCENARIOS]
LIMITS = dict(saved_bundle_load=1,identity_gate=1,target_validation=1,arm_attachment=2,
    signal_qualification=6,round_accuracy=78,energy_assessment=6,
    shots_weight_summary=6,lane_comparison=3,result_summary=1)
MAX_CALLS = sum(LIMITS.values())


def load_bundle(root,c):
    root = Path(root)
    return {k:io.read(root/p) for k,p in c['records'].items()}


def identity_gate(bundle,c):
    """Metadata/coverage/finite-value checks only; no shots or energy evaluation."""
    r, stop, source, old = (bundle[k] for k in ('result','STOP','SOURCE','contract'))
    if (r['status'] != 'SAVED_ENDPOINT_STAGE_B_COMPLETE' or stop['failed']
            or stop['scientific_units'] != 143 or stop['call_counts'] != stage_b.LIMITS
            or stop['retry'] or stop['resume'] or not stop['mandatory_stop']
            or source['execution_commit'] != STAGE_B_SOURCE
            or source['files'] != bundle['manifest']['files']
            or old['policy'] != stage_b.POLICY or old['fit_source'] != stage_b.FIT_SOURCE
            or old['identities'] != c['identities'] or set(r['arms']) != set(ROLES)):
        raise ValueError('STAGE_B_SOURCE_POLICY_IDENTITY')
    if len(r['matched_rounds']) != 13 or [x['m'] for x in r['matched_rounds']] != list(range(13)):
        raise ValueError('MATCHED_13_ROUND_COVERAGE')
    for role in ROLES:
        arm = r['arms'][role]
        if (arm['identity'] != c['identities'][role] or arm['arm'] != role
                or arm['DF_budget'] != 'PASS' or arm['optimizer_qualification'] != 'UNQUALIFIED'
                or arm['optimizer']['success'] is not False
                or arm['classification']['status'] != 'POTENTIALLY_FEASIBLE'
                or [x['m'] for x in arm['rounds']] != list(range(13))):
            raise ValueError('SAVED_ARM_QUALIFICATION_OR_COVERAGE')
        accuracy._finite(arm['saved_DF_bound_Ha'],'saved DF guard-inclusive bound')
        if arm['saved_DF_bound_Ha'] > old['policy']['DF_reserve_Ha']:
            raise ValueError('SAVED_DF_RESERVE')
        if bundle[role+'_ground']['status'] != 'PASS' or bundle[role+'_ground']['alias_margin'] <= 0:
            raise ValueError('SAVED_GROUND_ALIAS')
        for row in arm['rounds']:
            m = row['m']
            if (row['arm'] != role or row != r['matched_rounds'][m][role]
                    or row['coherent_repetitions'] != 1 << m
                    or row['physical_time_Ha_inverse'] != .1*(1 << m)
                    or row['feasibility']['status'] != 'POTENTIALLY_FEASIBLE'
                    or row['synthesis_error'] is not None or row['implemented_law_error'] is not None):
                raise ValueError('SAVED_ROUND_IDENTITY_OR_ERROR_QUALIFICATION')
            for key in ('pf_operator_error','rte_corrected_operator_error','attenuation'):
                accuracy._finite(row['reference'][key],key)
            if row['reference']['attenuation'] > 1: raise ValueError('SAVED_ATTENUATION')
    return dict(status='PASS',rounds=26,DF_budget='SAVED_PASS',optimizer_qualification='UNQUALIFIED',
        stage_B_SOURCE=STAGE_B_SOURCE,old_prices_transferred=False)


def preflight(root,path):
    """Only bytes/JSON/schema/package metadata, never assess_round/energy/NPZ."""
    root = Path(root).resolve()
    c = io.read(path)
    for k,want in dict(scope=SCOPE,implementation=VERSION,base_commit=BASE,
            stage_B_SOURCE=STAGE_B_SOURCE,target=TARGET,scenarios=SCENARIOS,
            roles=list(ROLES),lanes=list(LANES),call_limits=LIMITS,
            maximum_logical_units=MAX_CALLS,fresh_output=FRESH,retry=0,resume=0).items():
        if c[k] != want: raise ValueError('FIXED_SHOTS_SCOPE:'+k)
    if io.sha(root/c['source_manifest']) != c['source_manifest_sha256']:
        raise ValueError('SOURCE_MANIFEST')
    for group in (io.read(root/c['source_manifest'])['files'],c['saved_inputs']):
        for name,h in group.items():
            p = (root/name).resolve()
            if not p.is_relative_to(root) or io.sha(p) != h: raise ValueError('SOURCE_OR_INPUT:'+name)
    if not set(c['records'].values()) <= c['saved_inputs'].keys(): raise ValueError('UNPINNED_RECORD')
    old = io.read(root/c['records']['contract'])
    if c['identities'] != old['identities']: raise ValueError('ARM_EXPORT_IDENTITY')
    for role in ROLES:
        ident = c['identities'][role]
        if ident['role'] != role or c['saved_inputs'][ident['export']] != ident['sha256']:
            raise ValueError('SAVED_FIT_IDENTITY')
    identity_gate(load_bundle(root,c),c)
    for k,want in c['packages'].items():
        if importlib.metadata.version(k) != want: raise ValueError('ENVIRONMENT:'+k)
    if (root/FRESH).exists(): raise ValueError('FRESH_SHOTS_NO_RETRY_RESUME')
    return c


def attach(bundle,c,role):
    a = bundle['result']['arms'][role]
    return dict(arm=role,target_id=role+':'+c['identities'][role]['sha256'],
        identity=c['identities'][role],rounds=a['rounds'],df_bound_Ha=a['saved_DF_bound_Ha'],
        optimizer_qualification=a['optimizer_qualification'],error_evidence=None)


def qualification(packet,lane):
    q = c1.qualification(packet,lane)
    if lane == 'evidence':
        q = dict(q,reason='No all-law synthesis or implemented-law raw-signal bounds for these saved endpoints')
    return dict(q,lane=lane,assumptions=[] if lane == 'evidence' else ASSUMPTIONS[lane],
        native_cost_proxy_assumed=False,scope='shots_and_coherent_weight_only')


def _range_round(target,packet,q,row):
    """Existing A=1 geometry + exact products, as in shots_screen; no new law.

    Same binary64 phase/log constants, not a transcendental enclosure. Raw
    products which round to zero are null with exact rational diagnostics.
    """
    ref = row['reference']; A = ref['attenuation']
    scenario = next(s for s in SCENARIOS if s['id'] == q['lane'])
    e = accuracy.RoundEvidence(row['m'],1.,ref['pf_operator_error'],ref['rte_corrected_operator_error'],
        scenario['synthesis_fraction_of_A'],scenario['law_fraction_of_A'],packet['target_id'],
        'own_DF_ground_state_oracle',True,True,law_model=q['law_model'])
    unit = accuracy.assess_round(target,e)
    if unit['status'] != 'CONDITIONAL_ROUND_PASS':
        delta=math.fsum((ref['pf_operator_error'],ref['rte_corrected_operator_error'],
            scenario['synthesis_fraction_of_A'],scenario['law_fraction_of_A']))
        unit=dict(unit,raw_signal_error=delta,radius_lower=1.-delta)
        n,method=None,'normalized_rejected_geometry'
    else:
        n,method = screen._count(unit['epsilon_coordinate'],A,target.alpha_axis)
    value = dict(unit,shots=n,qualification=q['kind'],assumption_applied=True,
        range_extension=True,arithmetic=method,normalized_raw_signal_error=unit['raw_signal_error'],
        normalized_radius_lower=unit['radius_lower'],unit_coordinate=unit.get('epsilon_coordinate'))
    for key in ('raw_signal_error','radius_lower','epsilon_coordinate'):
        if key not in unit: continue
        exact = Fraction.from_float(A)*Fraction.from_float(unit[key])
        rounded = float(exact)
        value[key] = None if exact != 0 and rounded == 0 else rounded
        value[key+'_fraction_binary64_constants'] = str(exact)
    return value


def assess_round(target,packet,q,row):
    # Validate references even when the evidence lane has unknown extra errors.
    for k in ('attenuation','pf_operator_error','rte_corrected_operator_error'):
        accuracy._finite(row['reference'][k],k)
    A = row['reference']['attenuation']
    if A > 1: raise ValueError('INVALID_ATTENUATION')
    if q['kind'] == 'ASSUMED_SCENARIO' and 0 < A < sys.float_info.min:
        value = _range_round(target,packet,q,row)
    else:
        value = c1.assess_round(target,packet,q,row)
        if value['status'] == 'UNRESOLVED_NUMERICAL_SHOT_RANGE':
            value = _range_round(target,packet,q,row)
        elif value['status'] == 'CONDITIONAL_ROUND_PASS':
            n,method = screen._count(value['epsilon_coordinate'],1.,target.alpha_axis)
            value = dict(value,shots=n,arithmetic=method,range_extension=False)
    n = value.get('shots')
    if n is not None and (type(n) is not int or n < 1): raise ValueError('POSITIVE_PYTHON_INTEGER_SHOTS')
    # Legacy rejected-round records omit delta/rho. Supply only these stated
    # signal-accounting diagnostics, without changing its decision/shot formula.
    if q['kind'] == 'ASSUMED_SCENARIO' and 'raw_signal_error' not in value:
        errors=q['round_errors'][row['m']]
        delta=math.fsum((A*row['reference']['pf_operator_error'],
            A*row['reference']['rte_corrected_operator_error'],errors['synthesis'],errors['law']))
        value=dict(value,raw_signal_error=delta,radius_lower=A-delta)
    return dict(value,lane=q['lane'],arm=packet['arm'],target_id=packet['target_id'],
        raw_signal_error=value.get('raw_signal_error'),radius_lower=value.get('radius_lower'),
        beta_deterministic=value.get('beta_deterministic'),epsilon_coordinate=value.get('epsilon_coordinate'),
        attenuation=A,pf_operator_error=row['reference']['pf_operator_error'],
        rte_corrected_operator_error=row['reference']['rte_corrected_operator_error'],
        shots_decimal=None if n is None else str(n),
        axes={axis:dict(shots_decimal=None if n is None else str(n),alpha_axis=target.alpha_axis) for axis in ('X','Y')},
        beta_stat=target.beta_stat,actual_CA_achieved=None,guaranteed_actual_shots=None,
        actual_synthesis_signal_error=None,actual_law_signal_error=None,
        all_law_bound_qualified=False,optimizer_qualification='UNQUALIFIED')


def shots_weight_summary(packet,q,plans,energy):
    common = dict(N_total_decimal=None,W_decimal=None,axis_total_decimal=None,groups=None,
        mu=None,S=None,total_resources=None,actual_CA_achieved=None,guaranteed_shots=None,
        optimizer_qualification=packet['optimizer_qualification'],integer_schema='decimal strings authoritative; JSON numeric shots are exact only in arbitrary-integer consumers')
    if q['kind'] == 'UNQUALIFIED': return dict(common,status='EVIDENCE_UNQUALIFIED')
    if not energy['same_accuracy']:
        return dict(common,status='ONE_OR_BOTH_INFEASIBLE' if any(p['status'].startswith('INFEASIBLE') for p in plans)
            or energy['status'] in ('DF_RESERVE_EXCEEDED','FAILURE_PROBABILITY_EXCEEDED','ENERGY_TARGET_FAIL') else 'EVIDENCE_UNQUALIFIED',
            reason=energy['status'])
    if [p['m'] for p in plans] != list(range(13)): raise ValueError('13_ROUND_COVERAGE')
    for p in plans:
        if type(p['shots']) is not int or p['shots'] < 1 or str(p['shots']) != p['shots_decimal']:
            raise ValueError('EXACT_INTEGER_SHOTS')
        if any(p['axes'][a]['shots_decimal'] != p['shots_decimal'] for a in ('X','Y')):
            raise ValueError('MATCHING_XY_SHOT_COUNTS')
    axis = {a:sum(int(p['axes'][a]['shots_decimal']) for p in plans) for a in ('X','Y')}
    groups = {}
    for label,indices in dict(m0_to9=range(10),m10_to12=range(10,13),m12=[12]).items():
        groups[label] = dict(N_decimal=str(sum(2*plans[m]['shots'] for m in indices)),
            W_decimal=str(sum((1 << m)*2*plans[m]['shots'] for m in indices)))
    return dict(common,status='CONDITIONAL_SHOTS_AVAILABLE',N_total_decimal=str(sum(axis.values())),
        W_decimal=str(sum((1 << p['m'])*2*p['shots'] for p in plans)),
        axis_total_decimal={a:str(v) for a,v in axis.items()},groups=groups,
        kernel_occurrence_rule='2^m; s=1; r=128 stays INSIDE kernel price, never an extra W factor',
        future_native_accounting='C_k = W*mu_arm,k + S_arm,k; both prices unmeasured')


def compare(lane,arms):
    summaries = {r:arms[r]['shots_weight'] for r in ROLES}
    statuses = [s['status'] for s in summaries.values()]
    status = ('ONE_OR_BOTH_INFEASIBLE' if 'ONE_OR_BOTH_INFEASIBLE' in statuses else
        'EVIDENCE_UNQUALIFIED' if 'EVIDENCE_UNQUALIFIED' in statuses else 'CONDITIONAL_SHOTS_AVAILABLE')
    delta = None
    if status == 'CONDITIONAL_SHOTS_AVAILABLE':
        delta = {k:str(int(summaries['pr'][k])-int(summaries['rc2'][k])) for k in ('N_total_decimal','W_decimal')}
    return dict(lane=lane,status=status,arms=summaries,PR_minus_RC2_decimal=delta,
        guaranteed_advantage=None,native_price=None,total_resources=None,actual_CA_achieved=None)


def write_csv(path,results):
    fields=('arm','lane','m','axis','attenuation','pf_operator_error','rte_corrected_operator_error',
        'raw_signal_error','radius_lower','beta_deterministic','beta_stat','epsilon_coordinate',
        'raw_signal_error_fraction_binary64_constants','radius_lower_fraction_binary64_constants',
        'epsilon_coordinate_fraction_binary64_constants','range_extension','arithmetic','shots_decimal','status','reason')
    with Path(path).open('x',newline='') as f:
        w = csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for role in ROLES:
            for lane in LANES:
                for p in results[role][lane]['rounds']:
                    for axis in ('X','Y'):
                        values={k:p.get(k) for k in fields}
                        values.update(arm=role,lane=lane,axis=axis,shots_decimal=p['axes'][axis]['shots_decimal'])
                        w.writerow(values)


def run(root,path,output):
    root = Path(root).resolve();c = preflight(root,path);output = Path(output).resolve()
    if output != root/FRESH or output.exists(): raise ValueError('FRESH_SHOTS_NO_RETRY_RESUME')
    output.mkdir(parents=True);calls=stage_b.old.integrated.Calls(output,LIMITS,MAX_CALLS);start=time.monotonic()
    try:
        save(output/'contract.json',c)
        save(output/'SOURCE.json',dict(io.read(root/c['source_manifest']),
            execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()))
        bundle = calls('saved_bundle_load','saved scalar JSON only',lambda:load_bundle(root,c))
        save(output/'identity.json',calls('identity_gate','new endpoints and Stage B26',lambda:identity_gate(bundle,c)))
        def target():
            t = accuracy.AccuracyTarget(**TARGET)
            if t.horizon()['maximum_round_index_M'] != 12: raise ValueError('FIXED_HORIZON')
            return t
        t = calls('target_validation','same CA/RPE/failure',target)
        results = {r:{} for r in ROLES}
        for role in ROLES:
            packet = calls('arm_attachment',role,lambda:attach(bundle,c,role))
            for lane in LANES:
                label = role+':'+lane
                q = calls('signal_qualification',label,lambda:qualification(packet,lane))
                plans=[]
                for row in packet['rounds']:
                    plan = calls('round_accuracy',label+':m'+str(row['m']),lambda:assess_round(t,packet,q,row))
                    plans.append(plan);save(output/f'{role}_{lane}_m{row["m"]:02d}.json',plan)
                energy = calls('energy_assessment',label,lambda:c1.energy(t,packet,plans))
                summary = calls('shots_weight_summary',label,lambda:shots_weight_summary(packet,q,plans,energy))
                value = dict(arm=role,lane=lane,identity=packet['identity'],qualification=q,rounds=plans,
                    conditional_energy_assessment=energy,shots_weight=summary,
                    actual_CA_achieved=None,total_resources=None,mu=None,S=None)
                results[role][lane]=value;save(output/f'{role}_{lane}.json',value)
        comparisons={lane:calls('lane_comparison',lane,lambda lane=lane:compare(lane,{r:results[r][lane] for r in ROLES})) for lane in LANES}
        result=calls('result_summary','shots/W only; prices unknown',lambda:dict(
            status='STAGE_C_SHOTS_COMPLETE',arms=results,comparisons=comparisons,
            actual_CA_achieved=None,native_price=None,total_resources=None,
            new_fits=0,ground_recalculation=0,PF_RTE_recalculation=0,native_build=0,
            compile=0,sampling=0,quantum_shots_drawn=0,mandatory_stop=True))
        save(output/'result.json',result);write_csv(output/'axes156.csv',results)
        return result
    except BaseException as error:
        calls.stop.set();save(output/'failure.json',dict(status='TECHNICAL_FAILURE',error=str(error),first_failure=calls.first_failure));raise
    finally:
        save(output/'STOP.json',dict(failed=calls.stop.is_set(),scientific_units=sum(calls.counts.values()),
            call_counts=dict(calls.counts),retry=0,resume=0,mandatory_stop=True))
        save(output/'resources.json',dict(wall_seconds=time.monotonic()-start,CPU_seconds=time.process_time(),
            peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
