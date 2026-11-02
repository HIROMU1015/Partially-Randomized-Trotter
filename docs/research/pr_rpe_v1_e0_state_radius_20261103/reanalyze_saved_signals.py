#!/usr/bin/env python3
"""V1: scalar/complex reanalysis of the fixed saved H4/DF12 signals only.

No trotterlib/scientific-runner import, matrix/state/operator generation, RNG,
shot/probability/branch/energy-estimator/cost calculation, or optimization.
Phase/radius definitions follow the fixed finite_rte_signal_validation helpers.
"""
from __future__ import annotations
import argparse
import cmath
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import unittest

D = '4591b8d75a2a6a55c74a51225f0c61648023c8ee'
PRIOR = '953b3bbcd05c495ff24d29c7b7e26ea0714907b7'
PRIOR_DIR = 'docs/research/pr_rpe_energy_bridge_readiness_20261103/'
HERE = Path(__file__).resolve().parent
PHYSICAL = 'physical_df_ground_state'
CONTROL = 'effective_partial_s2_eigenstate_control'
KINDS = ('exact_signal','pf_signal','corrected_finite_rte_signal','attenuated_event_mean_signal')
SOURCE_IDS = ('FINITESRC','PFDELTASRC','ABLATIONSRC','ACCOUNT','RPE4SRC','OPT2EXTSRC','WRAPPER','DFSTEP','REPEATED','ABLRUN')
DATA_IDS = ('ABLGRID','ABLJSON','PF3','PF12','OPT2JSON','INITPROV','EXTPROV')
NA = 'NOT_APPLICABLE'
NR = 'NOT_RECORDED'
UNRESOLVED = 'UNRESOLVED'


def decode(raw):
    return json.loads(raw, parse_int=str, parse_float=str)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(',',':'), allow_nan=False)


def finite(value):
    if value is None or isinstance(value, bool):
        raise ValueError('Missing or invalid numeric value')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite numeric value')
    return result


def saved_complex(payload):
    if not isinstance(payload,dict) or set(payload) != {'real','imag'}:
        raise ValueError('Expected an explicit saved real/imag signal')
    return complex(finite(payload['real']), finite(payload['imag']))


def phase_delta(left, right):
    """Signed principal Arg(left*conj(right)); None for a zero signal.

    The fixed _phase_distance returns abs of this expression, also with a
    zero-signal guard. No unwrap, frequency fit, or branch reconstruction.
    """
    if left == 0 or right == 0:
        return None
    return cmath.phase(left * right.conjugate())


def tuple_key(row):
    return tuple(str(row[k]) for k in ('q_m','r_m','K_m'))


def select_point(points, model_round):
    matches = [(i,p) for i,p in enumerate(points) if tuple_key(p)==tuple_key(model_round)]
    if len(matches)!=1:
        raise ValueError('Missing/duplicate exact q,r,K tuple')
    return matches[0]


def select_state(point, label):
    if label not in (PHYSICAL,CONTROL):
        raise ValueError('Unknown state label')
    matches = [(i,s) for i,s in enumerate(point['state_results']) if s['state_label']==label]
    if len(matches)!=1:
        raise ValueError('Missing/duplicate state label')
    return matches[0]


def signal_record(state, kind, reference, saved_radius=None):
    if state.get('state_label') not in (PHYSICAL,CONTROL):
        raise ValueError('Unknown state label')
    if kind not in KINDS or kind not in state:
        raise ValueError('Missing/unknown signal kind')
    raw = state[kind]; signal = saved_complex(raw)
    signed = phase_delta(signal, reference); radius = abs(signal)
    return dict(state_label=state['state_label'],signal_kind=kind,
        saved_complex=raw,reanalyzed_radius=radius,
        signed_principal_residual_to_common_E0=signed,
        absolute_principal_residual_to_common_E0=None if signed is None else abs(signed),
        saved_radius=saved_radius,
        reanalyzed_minus_saved_radius=None if saved_radius is None else radius-finite(saved_radius))


def comparison(actual, budget, allowance):
    if actual is None:
        return dict(status=UNRESOLVED,actual=None,budget=budget,excess=None,
                    within_without_allowance=None,allowance=allowance)
    limit=finite(budget); excess=actual-limit
    return dict(status='NUMERICALLY_CONSISTENT' if excess<=allowance else 'INCONSISTENT',
                actual=actual,budget=budget,excess=excess,
                within_without_allowance=excess<=0,allowance=allowance)


class Evidence:
    def __init__(self, repo):
        self.repo=str(repo)
        self.env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
        self.manifest=json.loads(self.git('show',PRIOR+':'+PRIOR_DIR+'EVIDENCE_INDEX.json'))
        self.metadata={}; self.data={}
        for ident in DATA_IDS+SOURCE_IDS:
            item=self.manifest['artifacts'][ident]
            raw=self.read(item)
            self.metadata[ident]={**item}
            if item['path'].endswith('.json'):
                self.data[ident]=decode(raw)
        self.identity_checks=self.check_identity()

    def git(self,*args):
        return subprocess.check_output(['git','-C',self.repo,*args],env=self.env)

    def read(self,item):
        oid=self.git('rev-parse',item['commit']+':'+item['path']).decode().strip()
        if oid!=item['blob']:
            raise ValueError('Fixed commit/path/blob mismatch')
        raw=self.git('cat-file','blob',oid)
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:
            raise ValueError('Fixed input bytes mismatch')
        return raw

    def check_identity(self):
        g,p3,p12=(self.data[k] for k in ('ABLGRID','PF3','PF12'))
        for key in ('hamiltonian_hash','ground_energy','sector_n_electrons','sector_dimension'):
            if len({j['hamiltonian'][key] for j in (g,p3,p12)})!=1:
                raise ValueError('Physical Hamiltonian/reference identity mismatch')
        if len({j['states'][PHYSICAL]['state_fingerprint'] for j in (g,p3,p12)})!=1:
            raise ValueError('Physical state fingerprint mismatch')
        if g['request']['ld']!='3' or g['request']['delta_time']!='0.02':
            raise ValueError('Grid outside fixed LD/delta')
        if g['hamiltonian']['sector_n_electrons']!='4' or g['hamiltonian']['n_df_blocks']!='12':
            raise ValueError('Sector/DF rank mismatch')
        a=self.data['ABLJSON']; snap=a['source_evidence']['snapshot']
        for p in (p3,p12):
            if p['provenance']['hamiltonian_input']['sha256']!=snap['sha256']:
                raise ValueError('Snapshot reference mismatch')
        for k in ('INITPROV','EXTPROV'):
            if self.data[k]['input_sha256'][snap['path']]!=snap['sha256']:
                raise ValueError('Compute snapshot reference mismatch')
        if a['source_evidence']['ld3_selected_schedule_signal_grid']['sha256']!=self.metadata['ABLGRID']['sha256']:
            raise ValueError('Grid source binding mismatch')
        for ld in ('3','12'):
            candidate=self.data['OPT2JSON']['coherent_reoptimization']['best_by_ld'][ld]
            if candidate['delta_time']!='0.02' or candidate['q_max']!='131072' or len(candidate['best']['rounds'])!=18:
                raise ValueError('Final model selection mismatch')
        return dict(recorded_H_E0_sector_fingerprints_match=True,
            snapshot_reference_hashes_match=True,grid_file_binding_matches=True,
            full_diagnostic_state_fingerprint='NOT_RECORDED_IN_DIAGNOSTIC; provenance only',
            full_delta0p02_eigenstate_control=UNRESOLVED,
            NPZ_decoded=False,states_or_operators_generated=False)

    def reference(self,ident,selector):
        item=self.metadata[ident]
        return dict(id=ident,commit=item['commit'],path=item['path'],blob=item['blob'],selector=selector,url=item['url'])

    def round(self,ld,model_index):
        g=self.data['ABLGRID']; a=self.data['ABLJSON']
        candidate=self.data['OPT2JSON']['coherent_reoptimization']['best_by_ld'][ld]
        if candidate['ld']!=ld or candidate['delta_time']!='0.02':
            raise ValueError('Model LD/delta mismatch')
        best=candidate['best']; m=best['rounds'][model_index]
        if m['round_index']!=str(model_index):
            raise ValueError('Round index mismatch')
        energy=finite(g['hamiltonian']['ground_energy']); delta=finite(candidate['delta_time'])
        time=finite(m['q_m'])*delta; reference=cmath.exp(-1j*energy*time)
        atol=finite(g['request']['numerical_atol']); budget_atol=1e-15
        model_radius=finite(m['conservative_radius_lower_bound'])
        saved_radius_map={'pf_signal':'reference_signal_radius',
            'corrected_finite_rte_signal':'corrected_finite_rte_radius',
            'attenuated_event_mean_signal':'observed_attenuated_radius'}
        records=[]; controls=[]; checks=[]
        if ld=='3':
            pi,point=select_point(g['points'],m)
            for label,target in ((PHYSICAL,records),(CONTROL,controls)):
                si,state=select_state(point,label)
                for kind in KINDS:
                    record=signal_record(state,kind,reference,state.get(saved_radius_map.get(kind,'')))
                    record['source']=self.reference('ABLGRID',f'/points/{pi}/state_results/{si}/{kind}')
                    target.append(record)
                if any(v['signed_principal_residual_to_common_E0'] is None for v in target):
                    raise ValueError('UNRESOLVED_ZERO_SIGNAL: no numerical phase assigned')
                pf=saved_complex(state['pf_signal']); ex=saved_complex(state['exact_signal'])
                corr=saved_complex(state['corrected_finite_rte_signal'])
                pf_saved_error=abs(phase_delta(pf,ex)); rte_saved_error=abs(phase_delta(corr,pf))
                checks.extend([dict(state_label=label,field='pf_vs_exact_phase_error',
                    calculated=pf_saved_error,saved=state['pf_vs_exact_phase_error'],
                    difference=pf_saved_error-finite(state['pf_vs_exact_phase_error'])),
                    dict(state_label=label,field='finite_rte_phase_error',
                    calculated=rte_saved_error,saved=state['finite_rte_phase_error'],
                    difference=rte_saved_error-finite(state['finite_rte_phase_error']))])
            _,physical=select_state(point,PHYSICAL)
            pf=saved_complex(physical['pf_signal']); corr=saved_complex(physical['corrected_finite_rte_signal'])
            observed=abs(saved_complex(physical['attenuated_event_mean_signal']))
            radius_pf=abs(pf); rte_error=abs(phase_delta(corr,pf))
            finite_saved=physical['finite_rte_phase_error']; finite_bound=physical['finite_rte_phase_error_bound']
            old_pf=physical['pf_vs_exact_phase_error']; saved_lb=physical['conservative_radius_lower_bound']
            refs=[self.reference('ABLGRID',f'/points/{pi}')]
            control_fp=g['states'][CONTROL]['state_fingerprint']
            state_identity='MATCHED_RECORDED_PHYSICAL_FINGERPRINT'
            input_fp=g['states'][PHYSICAL]['state_fingerprint']
            epsilon=point['round_signal_error_bound']
            applicability=finite(epsilon)<radius_pf and radius_pf>0
            actual_vs_saved_lb=observed-finite(saved_lb)
            finite_status=comparison(rte_error,best['beta_rte_budget'],budget_atol)
            finite_bound_status=comparison(finite(finite_bound),best['beta_rte_budget'],budget_atol)
            control_status='CONDITIONAL'
            attenuation_phase_difference=abs(phase_delta(saved_complex(physical['attenuated_event_mean_signal']),corr))
        else:
            matches=[(i,p) for i,p in enumerate(a['signal_diagnostics']['12']['points']) if p['q_m']==m['q_m']]
            if len(matches)!=1 or m['r_m']!='0' or m['K_m']!='0':
                raise ValueError('Full diagnostic tuple mismatch')
            i,point=matches[0]
            state={'state_label':PHYSICAL,'pf_signal':point['signal']}
            record=signal_record(state,'pf_signal',reference,point['observed_radius'])
            record['source']=self.reference('ABLJSON',f'/signal_diagnostics/12/points/{i}/signal')
            if record['signed_principal_residual_to_common_E0'] is None:
                raise ValueError('UNRESOLVED_ZERO_SIGNAL: no numerical phase assigned')
            records=[record]; refs=[self.reference('ABLJSON',f'/signal_diagnostics/12/points/{i}')]
            pf=saved_complex(point['signal']); radius_pf=observed=abs(pf)
            rte_error=finite_saved=finite_bound=epsilon=NA; old_pf=saved_lb=NR
            applicability=actual_vs_saved_lb=attenuation_phase_difference=NA
            finite_status=finite_bound_status={'status':NA}
            control_fp=NR; control_status=UNRESOLVED
            input_fp=NR; state_identity='CONDITIONAL_SOURCE_PROVENANCE; fingerprint only in PF12 reference'
        pf_error=abs(phase_delta(pf,reference)); pf_status=comparison(pf_error,best['beta_pf_budget'],budget_atol)
        value_differences=[v['difference'] for v in checks]
        value_differences += [v['reanalyzed_minus_saved_radius'] for v in records+controls if v['reanalyzed_minus_saved_radius'] is not None]
        stored_match=all(abs(v)<=atol for v in value_differences)
        if not stored_match:
            raise ValueError('Saved phase/radius reproduction differs beyond fixed numerical_atol')
        radius_margin=observed-model_radius
        differences=[]
        if controls:
            for physical_record,control_record in zip(records,controls):
                if physical_record['signal_kind']!=control_record['signal_kind']:
                    raise ValueError('Control/physical signal-kind mismatch')
                left=saved_complex(physical_record['saved_complex']);right=saved_complex(control_record['saved_complex'])
                differences.append(dict(signal_kind=physical_record['signal_kind'],physical_minus_control_radius=abs(left)-abs(right),
                    physical_minus_control_principal_phase=phase_delta(left,right)))
            control_pf=saved_complex(select_state(point,CONTROL)[1]['pf_signal'])
            control_corr=saved_complex(select_state(point,CONTROL)[1]['corrected_finite_rte_signal'])
            control_pf_comparison=comparison(abs(phase_delta(control_pf,reference)),best['beta_pf_budget'],budget_atol)
            control_rte_comparison=comparison(abs(phase_delta(control_corr,control_pf)),best['beta_rte_budget'],budget_atol)
        else:
            control_pf_comparison=control_rte_comparison={'status':UNRESOLVED}
        row=dict(row_id=f'LD{ld}-M{m["round_index"]}',ld=ld,delta_time=candidate['delta_time'],
            round_index=m['round_index'],q_m=m['q_m'],r_m=m['r_m'],K_m=m['K_m'],
            hamiltonian_hash=g['hamiltonian']['hamiltonian_hash'],snapshot_sha256=a['source_evidence']['snapshot']['sha256'],
            sector_n_electrons=g['hamiltonian']['sector_n_electrons'],saved_E0_ha=g['hamiltonian']['ground_energy'],
            saved_physical_state_fingerprint=input_fp,physical_identity_status=state_identity,
            PF12_reference_physical_fingerprint=self.data['PF12']['states'][PHYSICAL]['state_fingerprint'],
            saved_control_fingerprint=control_fp,control_data_status=control_status,
            reference_kind='MATHEMATICAL_REFERENCE_FROM_SAVED_E0_NOT_NEW_PHYSICAL_SIGNAL',
            calculated_time=time,calculated_E0_reference_json=json_text(dict(real=reference.real,imag=reference.imag)),
            physical_signals_json=json_text(records),control_signals_json=json_text(controls),
            physical_vs_control_differences_json=json_text(differences),
            control_PF_budget_comparison_json=json_text(control_pf_comparison),
            control_RTE_budget_comparison_json=json_text(control_rte_comparison),
            saved_control_physical_overlap=g['states'][CONTROL]['physical_state_overlap'] if ld=='3' else NR,
            calculated_physical_PF_signed_residual=phase_delta(pf,reference),
            calculated_physical_PF_abs_residual=pf_error,saved_PF_vs_exact_phase_error=old_pf,
            saved_model_empirical_PF_proxy=m['empirical_pf_phase_proxy'],
            saved_beta_PF=best['beta_pf_budget'],PF_budget_comparison_json=json_text(pf_status),
            calculated_finite_RTE_vs_PF_abs_residual=rte_error,saved_finite_RTE_phase_error=finite_saved,
            saved_finite_RTE_phase_bound=finite_bound,saved_model_finite_RTE_phase_bound=m['finite_rte_phase_bound'],
            saved_beta_RTE=best['beta_rte_budget'],finite_RTE_budget_comparison_json=json_text(finite_status),
            saved_finite_bound_budget_comparison_json=json_text(finite_bound_status),
            saved_beta_stat_reference_only=best['beta_stat_budget'],
            calculated_physical_PF_radius=radius_pf,calculated_PF_radius_minus_unit=radius_pf-1.0,
            calculated_observed_radius=observed,saved_physical_observed_radius=physical['observed_attenuated_radius'] if ld=='3' else point['observed_radius'],
            saved_physical_radius_lower_bound=saved_lb,calculated_observed_minus_saved_physical_bound=actual_vs_saved_lb,
            saved_model_radius_lower_bound=m['conservative_radius_lower_bound'],
            calculated_observed_minus_model_radius=radius_margin,
            observed_ge_model_radius_without_allowance=radius_margin>=0,
            observed_ge_model_radius_with_grid_atol=radius_margin>=-atol,
            model_radius_numeric_status='NUMERICALLY_CONSISTENT' if radius_margin>=-atol else 'INCONSISTENT',
            saved_model_attenuation=m['attenuation'],saved_epsilon_Z=epsilon,
            numerical_finite_phase_bound_applicability=applicability,
            calculated_attenuation_phase_difference=attenuation_phase_difference,
            unit_radius_guarantee_status='CONDITIONAL',
            model_input_assumption='exact_effective_partial_s2_eigenstate_input_assumed; unit_unattenuated_survival_signal_radius_assumed',
            saved_value_checks_json=json_text(checks),saved_value_reproduction='NUMERICALLY_CONSISTENT',
            numerical_atol=g['request']['numerical_atol'],budget_comparison_atol='1e-15',
            phase_budget_numeric_status='INCONSISTENT' if any(c['status']=='INCONSISTENT' for c in (pf_status,finite_status,finite_bound_status)) else 'NUMERICALLY_CONSISTENT',
            V1_energy_interpretation_status='CONDITIONAL',energy_branch_status='UNRESOLVED_BRANCH',
            unresolved='No final branch/success certification; unit reference radius not rigorous'+('; Full delta0.02 control and diagnostic state fingerprint not recorded' if ld=='12' else ''),
            evidence_refs_json=json_text(refs),model_ref_json=json_text(self.reference('OPT2JSON',f'/coherent_reoptimization/best_by_ld/{ld}/best/rounds/{model_index}')))
        return row


def numeric_summary(rows):
    output={}
    for ld in ('3','12'):
        rr=[r for r in rows if r['ld']==ld]
        if not rr: continue
        output[ld]=dict(round_count=len(rr),
            maximum_PF_abs_residual=max(r['calculated_physical_PF_abs_residual'] for r in rr),
            minimum_PF_radius=min(r['calculated_physical_PF_radius'] for r in rr),
            maximum_PF_radius=max(r['calculated_physical_PF_radius'] for r in rr),
            maximum_abs_PF_radius_deviation=max(abs(r['calculated_PF_radius_minus_unit']) for r in rr),
            maximum_model_radius_shortfall=max(0.0,max(-r['calculated_observed_minus_model_radius'] for r in rr)),
            model_radius_raw_fail_rows=[r['row_id'] for r in rr if not r['observed_ge_model_radius_without_allowance']],
            model_radius_beyond_grid_atol_rows=[r['row_id'] for r in rr if not r['observed_ge_model_radius_with_grid_atol']],
            phase_budget_inconsistent_rows=[r['row_id'] for r in rr if r['phase_budget_numeric_status']=='INCONSISTENT'])
        if ld=='3':
            output[ld]['maximum_physical_control_PF_phase_difference']=max(abs(json.loads(r['physical_vs_control_differences_json'])[1]['physical_minus_control_principal_phase']) for r in rr)
            output[ld]['maximum_control_PF_abs_residual']=max(json.loads(r['control_PF_budget_comparison_json'])['actual'] for r in rr)
            output[ld]['maximum_control_PF_radius_deviation']=max(abs(json.loads(r['control_signals_json'])[1]['reanalyzed_radius']-1.0) for r in rr)
            output[ld]['maximum_actual_finite_RTE_phase_error']=max(r['calculated_finite_RTE_vs_PF_abs_residual'] for r in rr)
            output[ld]['maximum_saved_finite_RTE_phase_bound']=max(finite(r['saved_finite_RTE_phase_bound']) for r in rr)
            output[ld]['maximum_physical_exact_reference_residual']=max(json.loads(r['physical_signals_json'])[0]['absolute_principal_residual_to_common_E0'] for r in rr)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evidence-repo',required=True)
    ap.add_argument('--sanity',action='store_true',help='Only the first/final saved round of each candidate')
    args=ap.parse_args(); evidence=Evidence(args.evidence_repo)
    test_stream=io.StringIO()
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_saved_signal_reanalysis.py')
    test_result=unittest.TextTestRunner(stream=test_stream).run(suite)
    if not test_result.wasSuccessful():
        raise RuntimeError(test_stream.getvalue())
    tests=dict(run=test_result.testsRun,failures=len(test_result.failures),errors=len(test_result.errors),status='PASS')
    pairs=[(ld,i) for ld in ('3','12') for i in ((0,17) if args.sanity else range(18))]
    rows=[evidence.round(ld,i) for ld,i in pairs]
    summary=numeric_summary(rows)
    if args.sanity:
        print(json.dumps(dict(stage='SAVED_POINT_SANITY',saved_value_reproduction='PASS',summary=summary),allow_nan=False))
        return
    with (HERE/'V1_ROUND_RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    validation=dict(schema_version='RPE-V1-SAVED-REANALYSIS-1',environment='LOCAL',
        prior_fixed_commit=PRIOR,evidence_carrier_commit=D,
        inputs_and_sources=evidence.metadata,input_identity=evidence.identity_checks,
        input_checks='PASS',saved_tuple_count=36,partial_matched_grid_tuples=18,full_matched_signal_q=18,
        saved_value_reproduction='PASS',artificial_tests=tests,
        summary=summary,interpretation='CONDITIONAL',branch_status='UNRESOLVED_BRANCH',
        preserved_saved_PF_diagnostic_flags={ld:evidence.data['PF'+ld]['summary']['qpe_spectral_diagnostics_pass'] for ld in ('3','12')},
        scientific_operations_performed=['principal phase residuals of saved signals','absolute values of saved complex signals','fixed-budget and saved-value comparisons'],
        prohibited_operations_executed=dict(H_or_state_or_operator_generation=0,simulation_or_sampling=0,
            RNG_replay=0,build_compile_transpile=0,shot_recalculation=0,alpha_beta_optimization=0,
            branch_reconstruction=0,new_success_probability=0,new_final_energy_estimate=0,cost_reaccounting_or_reranking=0),
        runtime=dict(python=platform.python_version(),cpu_affinity_count=len(os.sched_getaffinity(0)),
            configuration='stdlib scalar loop; first/final sanity precedes full 36 rows; no artificial resource cap'),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256((HERE/'test_saved_signal_reanalysis.py').read_bytes()).hexdigest(),
        source_reuse='Fixed phase-distance, complex-radius definitions and tuple/state selection; minimal stdlib implementation avoids scientific module imports',
        publication='VERIFY_REMOTE_SHA_AND_FULL_PUBLIC_BYTES_AFTER_COMMIT',mandatory_stop=True)
    (HERE/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='V1_REANALYSIS_COMPLETE',rows=len(rows),summary=summary),allow_nan=False))


if __name__=='__main__':
    main()
