"""Adopted zero-ZZ saved-only reassessment and one fixed empirical/model scenario."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
import subprocess
import time

import account_full_h_reference_scenario as C

BASE = 'docs/research/full_h_m0_zero_zz_admission_total_20261116'
MODULE = 'scripts/reassess_full_h_m0_zero_zz.py'
POLICY = dict(commit='481c4b7abffbf070347f697980f2ec051e6a6538',
              path='docs/research/full_h_m0_zero_zz_rule_20261116/PROPOSAL.json')
D1 = dict(commit='74cd9c2625a6338793f1beef7b434c765a3a7e67',
          path='docs/research/d_full_h_m0_reference_20261115/HANDOFF_C.json')
PRIOR = dict(commit='426b8c4f14b4a00869c73f7b2ae4ecdc638a0107',
             path='docs/research/full_h_m0_reference_total_20261115/run01/RESULT.json')
RULE_ID = 'FULL_H_KAPPA2_NORMAL_PATH_ZERO_ZZ_ADMISSION_PROPOSAL_V1'


def common_conditions(rule, policy, evidence, handoff, prior):
    C.equal(policy['rule_id'], RULE_ID, 'Adopted rule identity')
    C.equal(policy['original_rule'], C.RULE, 'Original empirical price rule')
    C.equal(handoff['rule_source'],C.RULE,'D1 original rule binding')
    C.require(policy['alternative_parity_evidence']['applies_only_to_independent_N_zero'] is True,
              'Alternative evidence restricted to independently proven zero')
    for obj in (evidence, handoff):
        C.equal(obj['physical'], rule['physical_candidate'], 'Original physical identity')
        C.equal((obj['cohort'],obj['path_index'],obj['seeds'],obj['independent_paths']),
                (rule['proposed_acquisition']['cohort'],0,[rule['proposed_acquisition']['seed_u64']],1),
                'Same pre-fixed normal path')
    C.equal(evidence['execution_source'], handoff['execution_source'], 'Original producer SOURCE')
    C.equal(evidence['result_commit'], handoff['result_commit'], 'Original result publication')
    C.equal(evidence['D1_handoff'], D1['commit'], 'Evidence-to-original-handoff binding')
    C.equal(prior['status'], 'HELD_NO_TOTAL', 'Old C reassessment preserved')
    C.require(prior['T'] is None and prior['m0_partial'] is None, 'Old null accounting preserved')
    z=evidence['zero_target_evidence']
    C.equal(z['target_count_before_substitution'],0,'Independent pre-substitution target count')
    C.require(z['target_counter_used_as_basis'] is False,'Not inferred from replacement counter')
    C.equal(z['no_deterministic_blocks'],0,'No omitted deterministic head')
    C.require(bool(z['full_wrapper_scope']),'Whole generation scope documented')
    C.equal(len(z['applications']),rule['physical_candidate']['s'],'Every saved application covered')
    for app in z['applications']:
        C.equal((app['role'],app['is_identity'],len(app['central_support']),app['basis_system_only'],app['replacement_target']),
                ('rotation',False,1,True,False),'Legal non-ZZ application and system-only basis')
        C.equal(app['phase'],dict(real=1.0,imag=0.0),'Original event relative phase')
    original=evidence['original_saved']
    C.equal((original['context_check'],original['native_status'],original['sampler_calls']),
            ('PASS','PASS',1),'Original input/numeric/normal-draw evidence')
    C.equal(original['composite_gate'],'STOP','Preserve original composite STOP')
    for key in ('parity_return','scalar_event_return','XY_return','runtime_lowering_marker'):
        C.equal(original[key],'NOT_PERSISTED','Do not fabricate historical '+key)
    C.equal(len(original['numeric_checks']),7,'Original seven numerical checks')
    C.require(all(g['phase_alignment'] is False for g in original['numeric_checks']), 'Original phase convention')
    current=evidence['current_read_only_checks']
    C.equal(current['legacy_new_equivalence'],
            'FIXED_SOURCE_BRANCH_ARGUMENT_ONLY_NO_LEGACY_CIRCUIT_COMPARISON','Adopted alternative SOURCE evidence')
    xy=current['phase_and_XY']
    C.equal(xy['status'],'PASS','Saved-QPY phase/XY check')
    C.require(xy['same_controlled_evolution'] is True,'Same controlled U in X/Y')


def reassess_axis(axis, rule, policy, evidence, handoff, prior, receipt, native):
    common_conditions(rule,policy,evidence,handoff,prior)
    rows=[r for r in handoff['axis_rows'] if r['axis']==axis]
    checks=[r for r in evidence['current_read_only_checks']['axes'] if r['axis']==axis]
    C.equal((len(rows),len(checks)),(1,1),'One receipt and one saved check per axis')
    row,check=rows[0],checks[0]
    unit=next(u for u in rule['units'] if u['axis']==axis)
    C.equal(row['admission'],unit['admission'],'Original C physical/decoder bridge')
    C.equal((row['physical'],row['physical_id'],row['axis'],row['m'],row['cohort'],row['path_index'],row['seeds']),
            (rule['physical_candidate'],rule['physical_candidate']['id'],axis,0,handoff['cohort'],0,handoff['seeds']),
            'Axis and original candidate/path binding')
    C.equal(row['source']['commit'],handoff['execution_source'],'Axis producer SOURCE')
    C.require(row['per_shot_RZ'] is None,'Original held price remains null')
    C.equal(check['compiler_profile'],rule['profile'],'Compiler/control/wrapper/P0')
    for key,other in (('receipt','compile_record'),('wrapper','wrapper'),('compiled_circuit','compiled_circuit')):
        C.equal(check[key],row[other],'Original axis artifact '+key)
    C.equal((receipt['compiler'],receipt['fingerprint'],receipt['metrics']['rz_count']),
            (rule['profile']['compiler'],row['fingerprint'],row['observed_compiled_RZ']),'Original price receipt')
    C.equal(check['fingerprint'],row['fingerprint'],'Separate original axis fingerprint')
    C.equal(check['observed_RZ_quoted_from_original_record'],row['observed_compiled_RZ'],'Saved quoted price')
    C.equal(native['status'],'PASS','Original native numerical status')
    C.equal(native['checks'],evidence['original_saved']['numeric_checks'],'Saved numerical checks unchanged')
    C.equal((native['preparation_hash'],native['scalar']),
            (rule['target_preparation_hash'],rule['scalar']['total']),'Original preparation/scalar')
    C.equal(row['measurement']['axis'],axis,'Axis measurement binding')
    C.require(row['measurement']['include_measurement'] is True,'Full measured wrapper')
    C.equal(check['measurement'],[[596 if axis=='X' else 597,[8],[0]]],'Saved ancilla/cbit measurement')
    C.require(check['all_operations_at_most_two_wires'] is True,'No unaccounted opaque controlled operation')
    expected={'crz','h','measure','p','xx_plus_yy'} | ({'sdg'} if axis=='Y' else set())
    C.equal(set(check['unique_top_level_operation_names']),expected,'Closed wrapper operation set')
    C.equal(check['global_phase'],0.0,'Saved global phase')
    phase=check['current_read_only_phase_check']
    C.equal(phase,evidence['current_read_only_checks']['phase_and_XY']['axes']['cosine' if axis=='X' else 'sine'],
            'Axis check and shared XY check')
    C.equal((phase['status'],phase['scalar'],phase['expected_scalar'],phase['ancilla_p'],phase['event_phases']),
            ('PASS',rule['scalar']['total'],rule['scalar']['total'],1,[]),'Scalar/event phase conditions')
    C.require(phase['legacy_scalar_once'] is True,'Scalar applied once at total T')
    C.equal(check['original_implementation_gate_return'],'NOT_PERSISTED','Historical parity still missing')
    value=receipt['metrics']['rz_count']
    C.require(type(value) is int and value>=0,'Original nonnegative integer RZ unit price')
    return value,deepcopy(row)


def read_and_reassess(root, rule, policy, evidence, handoff, prior):
    rows=[];admissions=[]
    for unit in rule['units']:
        axis=unit['axis'];record=dict(axis=axis,status='HELD',accepted_per_shot_RZ=None,reason=None,
            policy_source=POLICY,post_observation_rule_adoption=True,original_parity_return='NOT_PERSISTED',
            original_runtime_lowering_marker='NOT_PERSISTED',
            current_integrity_evidence='INDEPENDENT_ZERO_TARGET_FIXED_SOURCE_ALTERNATIVE_AND_SAVED_QPY',
            ZZ_usage_coverage='NOT_APPLICABLE_ONLY_IF_INDEPENDENT_ZERO_PROVEN')
        output=dict(m=0,axis=axis,physical_id=unit['physical_id'],N=96,C_ref=None,status='HELD',
            price_origin='SAME_NORMAL_PATH_AXIS_DIRECT_EMPIRICAL_REFERENCE',qualified_E_C=False,
            physical_candidate=deepcopy(rule['physical_candidate']),
            **{k:deepcopy(unit[k]) for k in ('original_configuration_id','total_configuration_id',
                'original_candidate_id','total_candidate_id','B_join')})
        try:
            original=next(r for r in handoff['axis_rows'] if r['axis']==axis)
            # Reuse the original receipt/QPY bindings, without loading QPY or counting gates.
            for key in ('compile_record','wrapper','compiled_circuit'):
                ref=original[key];raw=C.G.C.blob(root,ref['commit'],ref['path'])
                C.equal(hashlib.sha256(raw).hexdigest(),ref['sha256'],'Original '+axis+' '+key+' bytes')
            receipt=C.G.evidence(root,original['compile_record']);native=C.G.evidence(root,original['native_trial'])
            value,original=reassess_axis(axis,rule,policy,evidence,handoff,prior,receipt,native)
            output.update(C_ref=C.exact(value),status='ACCEPTED_SAVED_REFERENCE',
                source=C.at(D1,'axis_rows',handoff['axis_rows'].index(original)),cohort=original['cohort'],
                path_index=original['path_index'],seeds=original['seeds'],fingerprint=original['fingerprint'],
                classification='DIRECT_COMPILED_FINITE_SAMPLE',expectation='FINITE_COHORT_ONLY')
            record.update(status='ACCEPTED_EMPIRICAL_REFERENCE_UNDER_ADOPTED_ZERO_ZZ_RULE',
                accepted_per_shot_RZ=value,original_record=original,source=output['source'])
        except (KeyError,ValueError,OSError,StopIteration) as error:
            output['reason']=record['reason']=str(error)
        rows.append(output);admissions.append(record)
    return rows,admissions


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    if not parser.parse_args().execute:parser.error('Explicit --execute required for the adopted finite scope')
    root=C.ROOT;source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    for path in (MODULE,C.MODULE):
        C.equal((root/path).read_bytes(),C.G.C.blob(root,source,path),'Fixed execution SOURCE '+path)
    C.equal((root/C.MODULE).read_bytes(),C.G.C.blob(root,PRIOR['commit'],C.MODULE),'Reuse original accounting unchanged')
    out=root/BASE/'run01';C.require(not out.exists(),'Fresh reassessment; never overwrite old results')
    out.mkdir(parents=True);start=time.perf_counter();print('BEGIN saved-only zero-ZZ reassessment',flush=True)
    try:
        policy=C.G.evidence(root,POLICY);rule=C.G.evidence(root,C.RULE);prior=C.G.evidence(root,PRIOR)
        handoff=C.G.evidence(root,D1);evidence=C.G.evidence(root,policy['D1_evidence'])
        saved=[deepcopy(r) for r in prior['rows'] if r['m']>0]
        C.equal(len(saved),24,'Reuse all 24 already-connected F1 axes')
        C.require(all(r['N']==C.SHOTS[r['m']] and r['price_origin']=='SAVED_F1_MODEL_REFERENCE' for r in saved),
                  'Original fixed allocation and saved price tier')
        benchmarks=[deepcopy(r['benchmark']) for r in prior['comparisons']]
        C.equal({r['label']:r['T_PF'] for r in benchmarks},C.PF_VALUES,'Current two PF comparison references')
        m0,admissions=read_and_reassess(root,rule,policy,evidence,handoff,prior)
        print('END admission; BEGIN existing exact accounting',flush=True)
        result=C.account(m0+saved,benchmarks)
        result.update(schema='FULL_H_ZERO_ZZ_ADOPTED_EMPIRICAL_MODEL_SCENARIO_V1',execution_source=source,
            adopted_rule_id=RULE_ID,sources=dict(policy=POLICY,D1_handoff=D1,D1_evidence=policy['D1_evidence'],
                prior_C_result=PRIOR,F1=C.F1,PF=C.PF,original_rule=C.RULE),m0_admissions=admissions,
            accepted_m0_axes=sum(r['accepted_per_shot_RZ'] is not None for r in admissions),
            independent_m0_paths=1,shared_path_for_XY=True,reused_F1_model_axes=24,
            new_sampling=0,new_builds=0,new_compiles=0,new_simulations=0,new_price_recounts=0,new_predictions=0,
            new_fits=0,new_Q3=0,total_accounting_calls=1 if result['T'] is not None else 0,
            original_C_and_D1_STOP_preserved=True,original_F1_negative_prices_and_null_total_preserved=True,
            model_qualification_changed=False,E_C=None,CI=None,law_distance=None,finite_cohort_uncertainty=None,
            shared_covariance=None,price_tolerance=None,G_qualified=None,winner=None,
            wall_seconds=time.perf_counter()-start,mandatory_stop=True)
        (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        with (out/'AXES.csv').open('w',newline='') as handle:
            writer=csv.writer(handle,lineterminator='\n');writer.writerow(
                ['m','axis','physical_id','N','price_numerator','price_denominator','weighted_numerator',
                 'weighted_denominator','price_origin','status'])
            for r in result['rows']:
                p=r.get('C_ref') or {};w=r.get('weighted_RZ') or {}
                writer.writerow([r['m'],r['axis'],r['physical_id'],r['N'],p.get('numerator'),p.get('denominator'),
                    w.get('numerator'),w.get('denominator'),r['price_origin'],r['status']])
        print(json.dumps({k:result[k] for k in ('status','accepted_m0_axes','T','m0_partial','m0_percent')}),flush=True)
    except Exception as error:
        (out/'STOP.json').write_text(json.dumps(dict(status='STOP_NO_AUTOMATIC_RETRY',source=source,
            reason=repr(error),mandatory_stop=True),indent=2)+'\n');raise


if __name__=='__main__':main()
