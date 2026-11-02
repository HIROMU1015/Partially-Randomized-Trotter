#!/usr/bin/env python3
"""Fixed V2 saved-mean reconstruction and analytic binomial probabilities.

Only selected pure functions are loaded from immutable SOURCE. No repository
scientific module import, physical signal/state/operator generation, sampling,
RNG replay, compile, shot optimization or cost accounting is performed.
"""
from __future__ import annotations
import argparse
import ast
import cmath
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import unittest
import numpy as np
import scipy
from scipy.stats import binom

HERE=Path(__file__).resolve().parent
PHYSICAL='physical_df_ground_state'
V1_COMMIT='e520e6e396f7e0def9e522ea1cff58eb10ddfa8f'
D='4591b8d75a2a6a55c74a51225f0c61648023c8ee'
FIXED = {'V1REPORT': ('e520e6e396f7e0def9e522ea1cff58eb10ddfa8f',
              'docs/research/pr_rpe_v1_e0_state_radius_20261103/REPORT.md',
              '4fcc99a5460c6f9cf72d1313ed5f0fec5dd847fe'),
 'V1ROWS': ('e520e6e396f7e0def9e522ea1cff58eb10ddfa8f',
            'docs/research/pr_rpe_v1_e0_state_radius_20261103/V1_ROUND_RESULTS.csv',
            '40df7c102e9ab8aba52862f734dc20b008ad361b'),
 'V1VALID': ('e520e6e396f7e0def9e522ea1cff58eb10ddfa8f',
             'docs/research/pr_rpe_v1_e0_state_radius_20261103/VALIDATION.json',
             '8c3fabc4d60b03bd2cf8a476454587879737fbda'),
 'V1HELPER': ('e520e6e396f7e0def9e522ea1cff58eb10ddfa8f',
              'docs/research/pr_rpe_v1_e0_state_radius_20261103/reanalyze_saved_signals.py',
              'c84bb778b386181125bda81021c7e0240f7af7f2'),
 'V1TEST': ('e520e6e396f7e0def9e522ea1cff58eb10ddfa8f',
            'docs/research/pr_rpe_v1_e0_state_radius_20261103/test_saved_signal_reanalysis.py',
            '18e2c06b97f1199688dfbed62beccf088e97ad0e'),
 'ABLGRID': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
             'artifacts/research_direction_ablation/2026-09-21/wp04_ld3_full_schedule_signal_grid_v1.json',
             '1abb90ebd647356223160f3fb889c89112a3294a'),
 'ABLJSON': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
             'artifacts/research_direction_ablation/2026-09-21/wp04_finite_rte_statistical_ablation_v1.json',
             'cbd4646e0f64b47d5d502f22fad5e3ea846f5bd9'),
 'OPT2JSON': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
              'artifacts/research_direction_full_opt2/2026-09-25/wp11_all_r_opt2_coherent_analysis_20260925_065827.json',
              'fa6f6e154fd60e135a9e6901a186976ad8bf34be'),
 'WP01': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
          'artifacts/research_direction_prevalidation/2026-09-21/wp01s_model_conditional_screening_v1.json',
          '1aea2a7ed656753ead908a9fc5e41db5f682e6af'),
 'RPE4JSON': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
              'artifacts/rpe_four_round_phase_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_q1_q2_q4_q8_physical_branch_v1.json',
              '9271f6b6fd86c7cfc3cbab8714dfbaa527b5d477'),
 'HORIZONJSON': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                 'artifacts/rpe_target_round_horizon_validation/2026-09-20/h4_sto3g_d100_rank12_ld3_dt0p1_r4_k2_ca_over_10_horizon_v1.json',
                 '30db39a773c5b7b11940967d6448137bcdf29c87'),
 'METHOD': ('9a0bc7f17f47823e46c3945734127e9f0d09154c',
            'docs/research/研究方法・解析手順.md',
            'a03b048d3fd5150bbc3ea55946ae92f2e84a8c75'),
 'PLAN': ('9a0bc7f17f47823e46c3945734127e9f0d09154c',
          'docs/research/数値実験・評価計画.md',
          '4a38150f75238224375b37e44300312d7edd8669'),
 'RPE4SRC': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
             'src/trotterlib/rpe_four_round_phase_validation.py',
             '928e662fe9734ab267674ee9d6f4ff62931ec2b1'),
 'RPE4TEST': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
              'tests/test_rpe_four_round_phase_validation.py',
              'e5292c371e1c1dc55556c7e42929d17e30758d6b'),
 'FAILURESRC': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                'src/trotterlib/rpe_hadamard_failure_validation.py',
                '6c3137985c81c8d69995b64bde5bbeab3d953167'),
 'FAILURETEST': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                 'tests/test_rpe_hadamard_failure_validation.py',
                 'cbf00e894b82af7065867a14ed64f3709b9a0b99'),
 'ACCOUNTSRC': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                'src/trotterlib/rpe_resource_accounting.py',
                '95fd914943d4ee91779e03ba13e83b6f33bdaa87'),
 'ACCOUNTTEST': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                 'tests/test_rpe_resource_accounting.py',
                 'ea0ecfff1d1f58054a682f15a2f003235f0d7dde'),
 'ABLATIONSRC': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                 'src/trotterlib/research_direction_ablation.py',
                 '75395da286ce094bd5c4c1f1fcf820502c9de932'),
 'ABLATIONTEST': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                  'tests/test_research_direction_ablation.py',
                  '18956cb04727601d1b5468918854806fe03695ae'),
 'FINITESRC': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
               'src/trotterlib/finite_rte_signal_validation.py',
               '4480bf602d600be3c69f96e3f036564fc1f92e57'),
 'FINITETEST': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                'tests/test_finite_rte_signal_validation.py',
                '8686991f7ce502d33069e7f796d6408efb536797'),
 'OPT2EXTSRC': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                'src/trotterlib/research_direction_full_opt2_extension_analysis.py',
                'b288da652a9f5e4f6d5e27504cb767d2f59f6abd'),
 'OPT2EXTTEST': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                 'tests/test_research_direction_full_opt2_extension_analysis.py',
                 '9b7397d1d4c1f7df75eef50031553568a87feb7b'),
 'HORIZONSRC': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                'src/trotterlib/rpe_target_round_horizon_validation.py',
                'b344348784cdbcadd60bff7fb1324998b787581a'),
 'HORIZONTEST': ('4591b8d75a2a6a55c74a51225f0c61648023c8ee',
                 'tests/test_rpe_target_round_horizon_validation.py',
                 '1adffc3a1c42094ba2e286e02e52360dc16779bc')}


def literal_json(raw):
    return json.loads(raw,parse_int=str,parse_float=str)


def text_json(value):
    return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def select_point(points,m):
    matches=[(i,p) for i,p in enumerate(points) if all(p[k]==m[k] for k in ('q_m','r_m','K_m'))]
    if len(matches)!=1:raise ValueError('Missing/duplicate fixed q,r,K tuple')
    return matches[0]


def physical_state(point):
    found=[(i,s) for i,s in enumerate(point['state_results']) if s['state_label']==PHYSICAL]
    if len(found)!=1:raise ValueError('Missing/duplicate physical state label')
    return found[0]


def number(value):
    if value is None or isinstance(value,bool):raise ValueError('Missing numeric input')
    result=float(value)
    if not math.isfinite(result):raise ValueError('Nonfinite input')
    return result


def complex_value(value):
    return complex(number(value['real']),number(value['imag']))


def selected_functions(source,names):
    """Execute unchanged FunctionDef nodes only; original imports/runners omitted."""
    tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('SOURCE function set mismatch')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    ast.fix_missing_locations(module)
    namespace={'np':np,'math':math,'binom':binom,'__name__':'fixed_v2_functions'}
    exec(compile(module,'<fixed-source-functions>','exec'),namespace)
    records={n.name:dict(first_line=n.lineno,last_line=n.end_lineno,
        function_text_sha256=hashlib.sha256(ast.get_source_segment(source,n).encode()).hexdigest(),
        unchanged_ast=True) for n in nodes}
    return {n:namespace[n] for n in names},records


class Evidence:
    def __init__(self,repo):
        self.repo=str(repo);self.env=dict(os.environ,GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1')
        self.raw={};self.meta={};self.data={}
        for key,(commit,path,expected_blob) in FIXED.items():
            blob=self.git('rev-parse',commit+':'+path).decode().strip()
            if blob!=expected_blob:raise ValueError('Fixed Git commit/path identity mismatch')
            raw=self.git('cat-file','blob',blob)
            self.raw[key]=raw
            self.meta[key]=dict(commit=commit,path=path,blob=blob,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                url='https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/'+commit+'/'+path)
            if path.endswith('.json'):self.data[key]=literal_json(raw)
        self.v1_rows=list(csv.DictReader(io.StringIO(self.raw['V1ROWS'].decode())))
        self.f,self.function_records=selected_functions(self.raw['RPE4SRC'].decode(),
            ('_wrap_phase','_phase_distance','reconstruct_rpe_phase','_vectorized_reconstruction','_oracle_branch'))
        functions,records=selected_functions(self.raw['FAILURESRC'].decode(),
            ('_axis_distribution','_round_exact_phase_probability'))
        self.f.update(functions);self.function_records.update(records)
        self.tasks=self.bind_tasks()

    def git(self,*args):
        return subprocess.check_output(['git','-C',self.repo,*args],env=self.env)

    def ref(self,key,selector):
        return {**self.meta[key],'id':key,'selector':selector}

    def bind_tasks(self):
        v=self.data['V1VALID'];g=self.data['ABLGRID'];a=self.data['ABLJSON'];o=self.data['OPT2JSON'];w=self.data['WP01']
        if v['interpretation']!='CONDITIONAL' or len(self.v1_rows)!=36:raise ValueError('V1 interpretation/inventory mismatch')
        for key in ('ABLGRID','ABLJSON','OPT2JSON'):
            if self.meta[key]['sha256']!=v['inputs_and_sources'][key]['sha256']:raise ValueError('V1 input binding mismatch')
        self.energy=number(g['hamiltonian']['ground_energy']);self.delta=0.02
        self.beta=number(w['configuration']['beta_rpe']);self.alpha=number(w['configuration']['alpha_total'])
        self.energy_target=number(w['configuration']['target_energy_precision_ha'])
        if g['request']['delta_time']!='0.02' or g['hamiltonian']['sector_n_electrons']!='4' or g['hamiltonian']['n_df_blocks']!='12':raise ValueError('Fixed physical conditions mismatch')
        tasks=[]
        for ld in ('3','12'):
            c=o['coherent_reoptimization']['best_by_ld'][ld];best=c['best']
            if c['ld']!=ld or c['delta_time']!='0.02' or c['q_max']!='131072' or len(best['rounds'])!=18:raise ValueError('Fixed model mismatch')
            for i,m in enumerate(best['rounds']):
                vr=[r for r in self.v1_rows if r['ld']==ld and r['round_index']==m['round_index']]
                if len(vr)!=1 or any(vr[0][k]!=m[k] for k in ('q_m','r_m','K_m')):raise ValueError('V1 tuple mismatch')
                vr=vr[0]
                if ld=='3':
                    pi,p=select_point(g['points'],m);si,s=physical_state(p)
                    raw=s['attenuated_event_mean_signal'];kind='attenuated_event_mean_signal'
                    ref=self.ref('ABLGRID',f'/points/{pi}/state_results/{si}/{kind}')
                else:
                    found=[(pi,p) for pi,p in enumerate(a['signal_diagnostics']['12']['points']) if p['q_m']==m['q_m']]
                    if len(found)!=1 or m['r_m']!='0' or m['K_m']!='0':raise ValueError('Full signal binding mismatch')
                    pi,p=found[0];raw=p['signal'];kind='physical_PF_signal'
                    ref=self.ref('ABLJSON',f'/signal_diagnostics/12/points/{pi}/signal')
                saved_v1=next(r for r in literal_json(vr['physical_signals_json']) if r['signal_kind']==('attenuated_event_mean_signal' if ld=='3' else 'pf_signal'))
                if raw!=saved_v1['saved_complex']:raise ValueError('Saved signal differs from V1')
                signal=complex_value(raw)
                if abs(signal)==0:raise ValueError('Zero true mean: meaningful phase model unresolved')
                target=complex_value(literal_json(vr['calculated_E0_reference_json']))
                expected=cmath.exp(-1j*self.energy*(int(m['q_m'])*self.delta))
                if abs(target-expected)>1e-10:raise ValueError('Common E0 phase reference mismatch')
                tasks.append(dict(ld=ld,round_index=m['round_index'],q_m=m['q_m'],r_m=m['r_m'],K_m=m['K_m'],
                    model=m,best=best,v1=vr,signal=signal,raw_signal=raw,signal_kind=kind,target=target,
                    signal_ref=ref,model_ref=self.ref('OPT2JSON',f'/coherent_reoptimization/best_by_ld/{ld}/best/rounds/{i}')))
        return tasks


def axis_tail_cdf_sf(axis):
    """Stable numerical cross-check for the original PMF/mask event, no new N."""
    means=axis['_sample_means'];mask=axis['_failure_mask'];mu=axis['mean'];n=axis['shots'];p=axis['probability_plus_one']
    left=np.flatnonzero(mask & (means<mu));right=np.flatnonzero(mask & (means>=mu))
    low=0.0 if not left.size else float(binom.cdf(int(left[-1]),n,p))
    high=0.0 if not right.size else float(binom.sf(int(right[0])-1,n,p))
    return low+high


def evaluate_round(task,f,beta):
    m=task['model'];best=task['best'];z=task['signal'];eps=number(m['epsilon_coordinate'])
    axes={};distributions={}
    for name,mu in (('cosine',z.real),('sine',z.imag)):
        saved=m['axes'][name];n=int(saved['shots']);alpha=number(saved['alpha'])
        if n<=0 or not 0<alpha<1 or eps<=0:raise ValueError('Invalid fixed N/alpha/epsilon')
        d=f['_axis_distribution'](mu,n,eps);distributions[name]=d
        mass=math.fsum(d['_probabilities']);cross=axis_tail_cdf_sf(d)
        axes[name]={k:v for k,v in d.items() if not k.startswith('_')}
        axes[name].update(saved_N=saved['shots'],saved_alpha=saved['alpha'],saved_epsilon_coordinate=m['epsilon_coordinate'],
            coordinate_probability_le_saved_alpha=d['exact_coordinate_failure_probability']<=alpha,
            hoeffding_coordinate_upper_bound=2.0*math.exp(-n*eps*eps/2.0),
            binomial_mass_sum=mass,binomial_mass_residual=mass-1.0,
            cdf_sf_coordinate_tail=cross,pmf_vs_cdf_sf_difference=d['exact_coordinate_failure_probability']-cross,
            raw_p_plus=(1+mu)/2.0,clipping_applied=(1+mu)/2.0!=d['probability_plus_one'],
            zero_PMF_entries=int(np.count_nonzero(d['_probabilities']==0)),
            floating_zero_tail_is_exact_zero_certified=False)
    co=distributions['cosine'];si=distributions['sine']
    bstat=number(best['beta_stat_budget'])
    stat,imp=f['_round_exact_phase_probability'](co,si,z,bstat)
    total,total_imp=f['_round_exact_phase_probability'](co,si,task['target'],beta)
    zero_c=math.fsum(co['_probabilities'][co['_sample_means']==0])
    zero_s=math.fsum(si['_probabilities'][si['_sample_means']==0])
    systematic=float(abs(np.angle(z*np.conj(task['target']))))
    alpha_round=number(m['axes']['cosine']['alpha'])+number(m['axes']['sine']['alpha'])
    return dict(ld=task['ld'],round_index=task['round_index'],q_m=task['q_m'],r_m=task['r_m'],K_m=task['K_m'],
        signal_kind=task['signal_kind'],saved_complex_signal=task['raw_signal'],axes=axes,
        identity={k:task['v1'][k] for k in ('hamiltonian_hash','snapshot_sha256','sector_n_electrons','saved_E0_ha','saved_physical_state_fingerprint','physical_identity_status')},
        saved_model_attenuation=m['attenuation'],
        saved_beta_pf=best['beta_pf_budget'],saved_beta_rte=best['beta_rte_budget'],saved_beta_stat=best['beta_stat_budget'],
        saved_model_radius=m['conservative_radius_lower_bound'],V1_physical_radius=task['v1']['calculated_observed_radius'],
        V1_model_radius_status=task['v1']['model_radius_numeric_status'],
        V1_physical_PF_residual=task['v1']['calculated_physical_PF_abs_residual'],
        V1_finite_RTE_residual=task['v1']['calculated_finite_RTE_vs_PF_abs_residual'],
        statistical_phase_failure_probability=stat,coordinate_success_but_stat_phase_failure_grid_points=imp,
        total_phase_failure_to_common_E0_probability=total,coordinate_success_but_total_phase_failure_grid_points=total_imp,
        zero_estimated_signal_probability=zero_c*zero_s,
        actual_systematic_phase_to_E0=systematic,saved_round_alpha_sum=alpha_round,
        statistical_phase_failure_le_round_alpha=stat<=alpha_round,
        total_phase_failure_le_round_alpha=total<=alpha_round,
        probability_kind='FLOAT64_FINITE_BINOMIAL_SUM_CONDITIONAL_ON_SAVED_MEANS_AND_INDEPENDENT_AXIS_SHOTS',
        status='CONDITIONAL_SUCCESS' if total<=alpha_round else 'CONDITIONAL_FAILURE',
        exact_joint_decoder_failure_probability='UNRESOLVED_NOT_EVALUATED',
        signal_ref=task['signal_ref'],model_ref=task['model_ref'])


def reconstruct_saved_means(tasks,evidence):
    f=evidence.f;q=[int(t['q_m']) for t in tasks];phases=[float(np.angle(t['signal'])) for t in tasks]
    # Deliberately no E0/reference argument to the actual reconstructor.
    actual=f['reconstruct_rpe_phase'](phases,q)
    reference=-evidence.energy*evidence.delta
    reference_wrapped=f['_wrap_phase'](reference)
    for row,phase,qm in zip(actual['rounds'],phases,q):
        oracle=int(f['_oracle_branch'](np.asarray(phase),qm,reference_wrapped))
        row['oracle_branch_diagnostic_only']=oracle
        row['selected_matches_oracle_on_saved_mean']=row['selected_branch']==oracle
        if qm==1:
            row['ambiguity_within_principal_window']=False
            row['initial_alias_is_external_condition']=True
        else:
            row['ambiguity_within_principal_window']=bool(np.isclose(row['distance_from_previous'],row['next_smallest_branch_distance'],rtol=0.0,atol=1e-12))
            row['best_runner_up_distance_gap']=row['next_smallest_branch_distance']-row['distance_from_previous']
    final=actual['final_phase_estimate'];tol=evidence.beta/q[-1]
    ehat=-final/evidence.delta
    window={'base_phase_lower':-math.pi,'base_phase_upper_exclusive':math.pi,
        'energy_lower_exclusive':-math.pi/evidence.delta,'energy_upper_inclusive':math.pi/evidence.delta,
        'target_inside_principal_window':-math.pi<=reference<math.pi,
        'reference_distance_to_phase_window_edges':min(reference+math.pi,math.pi-reference),
        'external_alias_assumption_not_physical_certification':True}
    alias_guard=window['target_inside_principal_window'] and window['reference_distance_to_phase_window_edges']>tol
    unique=not any(r['ambiguity_within_principal_window'] for r in actual['rounds'])
    error=float(f['_phase_distance'](final,reference_wrapped));energy_error=abs(ehat-evidence.energy)
    actual.update(ld=tasks[0]['ld'],input_kind='DETERMINISTIC_SAVED_MEANS_NOT_NEW_MEASUREMENT_SAMPLES',
        actual_reconstructor_uses_E0=False,reference_base_phase_evaluation_only=reference,
        circular_final_phase_error=error,final_phase_tolerance=tol,
        reconstructed_energy_in_initial_principal_window=ehat,saved_E0=evidence.energy,
        energy_absolute_error=energy_error,saved_energy_target=evidence.energy_target,
        nominal_energy_resolution=evidence.beta/(q[-1]*evidence.delta),
        phase_within_saved_tolerance=error<=tol,energy_within_saved_target=energy_error<=evidence.energy_target,
        unique_mean_branches=unique,all_mean_branches_match_oracle=all(r['selected_matches_oracle_on_saved_mean'] for r in actual['rounds']),
        alias_window=window,local_energy_unwrapping_guard=alias_guard,
        status='NUMERICALLY_VALIDATED' if unique and alias_guard and error<=tol and energy_error<=evidence.energy_target else 'CONDITIONAL_FAILURE',
        full_state_identity='CONDITIONAL_AS_RECORDED_IN_V1')
    return actual


def aggregate_probabilities(rounds,branches,evidence):
    out={}
    for ld in ('3','12'):
        rr=[r for r in rounds if r['ld']==ld];br=branches[ld]
        coord=math.fsum(a['exact_coordinate_failure_probability'] for r in rr for a in r['axes'].values())
        stat=math.fsum(r['statistical_phase_failure_probability'] for r in rr)
        total=math.fsum(r['total_phase_failure_to_common_E0_probability'] for r in rr)
        alpha=math.fsum(number(a['saved_alpha']) for r in rr for a in r['axes'].values())
        hoeffding=math.fsum(a['hoeffding_coordinate_upper_bound'] for r in rr for a in r['axes'].values())
        applicable=evidence.beta<math.pi/3 and br['local_energy_unwrapping_guard']
        success=max(0.0,1.0-total) if applicable else None
        out[ld]=dict(axis_coordinate_failure_union_bound=min(1.0,coord),round_statistical_phase_failure_union_bound=min(1.0,stat),
            common_E0_round_phase_failure_union_bound=min(1.0,total),
            saved_alpha_sum=alpha,saved_alpha_budget=evidence.alpha,alpha_sum_within_budget_without_allowance=alpha<=evidence.alpha,
            hoeffding_coordinate_union_bound=min(1.0,hoeffding),
            conditional_energy_success_lower_bound_numerical=success,
            conditional_branch_failure_upper_bound_from_sufficient_phase_event_union=min(1.0,total) if applicable else None,
            branch_bound_kind='SUFFICIENT_PHASE_EVENT_UNION_BOUND_NOT_EXACT_JOINT_DECODER_FAILURE',
            meets_saved_failure_target_numerically=applicable and total<=evidence.alpha,
            branch_sufficient_condition_applicable=applicable,
            all_coordinate_to_stat_phase_grid_implications_hold=all(r['coordinate_success_but_stat_phase_failure_grid_points']==0 for r in rr),
            all_coordinate_to_total_phase_grid_implications_hold=all(r['coordinate_success_but_total_phase_failure_grid_points']==0 for r in rr),
            maximum_abs_PMF_mass_residual=max(abs(a['binomial_mass_residual']) for r in rr for a in r['axes'].values()),
            maximum_PMF_CDF_SF_difference=max(abs(a['pmf_vs_cdf_sf_difference']) for r in rr for a in r['axes'].values()),
            any_probability_clipping=any(a['clipping_applied'] for r in rr for a in r['axes'].values()),
            exact_joint_branch_or_energy_failure_probability='UNRESOLVED_NOT_COMPUTED',
            guarantees_not_promoted=['exact unit PF radius','rigorous PF error bound','physical state preparation','backend/noise','final cost superiority'],
            status='CONDITIONAL_SUCCESS' if applicable and total<=evidence.alpha else 'UNRESOLVED' if not applicable else 'CONDITIONAL_FAILURE')
    return out


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--evidence-repo',required=True);args=ap.parse_args()
    started=time.perf_counter();e=Evidence(args.evidence_repo)
    tests=unittest.defaultTestLoader.discover(str(HERE),pattern='test_saved_v2.py')
    stream=io.StringIO();test_result=unittest.TextTestRunner(stream=stream).run(tests)
    if not test_result.wasSuccessful():raise RuntimeError(stream.getvalue())
    sanity_ids={('3','0'),('3','17'),('12','0'),('12','17')}
    sanity={}
    for t in e.tasks:
        if (t['ld'],t['round_index']) in sanity_ids:sanity[(t['ld'],t['round_index'])]=evaluate_round(t,e.f,e.beta)
    # A saved two-round prefix sanity precedes either full 18-round decoder.
    prefix_checks={ld:reconstruct_saved_means([t for t in e.tasks if t['ld']==ld][:2],e)['unique_mean_branches'] for ld in ('3','12')}
    if not all(prefix_checks.values()):raise ValueError('Ambiguous saved-prefix branch; STOP')
    remaining=[t for t in e.tasks if (t['ld'],t['round_index']) not in sanity]
    workers=min(len(remaining),len(os.sched_getaffinity(0)))
    with ThreadPoolExecutor(max_workers=workers) as pool:more=list(pool.map(lambda t:evaluate_round(t,e.f,e.beta),remaining))
    all_results=list(sanity.values())+more
    all_results.sort(key=lambda r:(0 if r['ld']=='3' else 1,int(r['round_index'])))
    branches={ld:reconstruct_saved_means([t for t in e.tasks if t['ld']==ld],e) for ld in ('3','12')}
    summary=aggregate_probabilities(all_results,branches,e)
    rows=[]
    for r in all_results:
        i=int(r['round_index']);br=branches[r['ld']]['rounds'][i]
        rows.append(dict(ld=r['ld'],round_index=r['round_index'],q_m=r['q_m'],r_m=r['r_m'],K_m=r['K_m'],
            delta_time='0.02',**r['identity'],saved_model_attenuation=r['saved_model_attenuation'],
            physical_input='Original H physical sector ground state; Full fingerprint conditional as V1',
            saved_signal_kind=r['signal_kind'],saved_complex_json=text_json(r['saved_complex_signal']),
            saved_beta_PF=r['saved_beta_pf'],saved_beta_RTE=r['saved_beta_rte'],saved_beta_stat=r['saved_beta_stat'],
            V1_physical_PF_residual=r['V1_physical_PF_residual'],V1_finite_RTE_residual=r['V1_finite_RTE_residual'],
            axes_results_json=text_json(r['axes']),statistical_phase_failure=r['statistical_phase_failure_probability'],
            total_phase_failure_to_common_E0=r['total_phase_failure_to_common_E0_probability'],
            coordinate_success_stat_phase_implication_violations=r['coordinate_success_but_stat_phase_failure_grid_points'],
            coordinate_success_total_phase_implication_violations=r['coordinate_success_but_total_phase_failure_grid_points'],
            zero_estimated_signal_probability=r['zero_estimated_signal_probability'],actual_systematic_phase=r['actual_systematic_phase_to_E0'],
            V1_observed_radius=r['V1_physical_radius'],saved_model_radius=r['saved_model_radius'],V1_model_radius_status=r['V1_model_radius_status'],
            saved_mean_branch=br['selected_branch'],oracle_branch_diagnostic_only=br['oracle_branch_diagnostic_only'],
            saved_mean_ambiguity=br['ambiguity_within_principal_window'],saved_mean_base_phase=br['base_phase_estimate'],
            probability_kind=r['probability_kind'],V2_status=r['status'],
            unresolved='Alias and IID measurement assumptions; V1 conditional identity/radius; joint decoder failure not evaluated',
            evidence_ref_json=text_json(r['signal_ref']),model_ref_json=text_json(r['model_ref'])))
    with (HERE/'V2_ROUND_RESULTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write_json(HERE/'BRANCH_RESULTS.json',dict(method='UNCHANGED_RPE4_SCALAR_NEAREST_BRANCH_ON_SAVED_MEANS',results=branches,
        E0_usage='evaluation and separate oracle diagnostic only; no E0 argument to actual reconstructor'))
    write_json(HERE/'SUCCESS_PROBABILITY_RESULTS.json',dict(method='FINITE_BINOMIAL_ENUMERATION_AND_UNION_BOUNDS_NO_SAMPLING',rounds=all_results,summary=summary,
        assumptions=['Saved complex values are the true marginal means for the stated physical input',
        'Independent bounded shots within each axis; independent cosine/sine experiments',
        'Partial uses fresh IID RTE trajectory per hypothetical quantum shot',
        'External alias-free principal energy window and fixed strict branch sufficient condition'],
        interpretation='Conditional numerical probability evaluation, not interval-certified physical guarantee; no exact full joint decoder distribution',
        Monte_Carlo_or_RNG_replay_performed=False))
    validation=dict(schema_version='PR-RPE-V2-1',environment='LOCAL',V1_fixed_commit=V1_COMMIT,source_carrier_commit=D,
        fixed_files=e.meta,source_functions_reused=e.function_records,
        function_loading='Unchanged selected FunctionDef AST nodes; original repository module imports/runners omitted',
        input_identity='PASS: V1 object/input SHA binding, physical tuples and saved signals',
        artificial_tests=dict(run=test_result.testsRun,failures=len(test_result.failures),errors=len(test_result.errors),status='PASS'),
        saved_point_sanity=dict(cells=sorted([list(k) for k in sanity]),two_round_prefix_unique=prefix_checks,followed_by_full_36_rows=True),
        operations_performed=['saved mean nearest-branch reconstruction','fixed-N binomial axis and round phase probabilities','union bounds','conditional energy estimate/precision evaluation'],
        prohibited_operations_executed=dict(new_H_state_operator_or_physical_signal=0,sampling_RNG_replay=0,build_compile_transpile=0,shot_reoptimization=0,
            alpha_beta_or_schedule_changes=0,cost_reaccounting_or_reranking=0),
        numerical_precision=dict(dtype='float64',signal_atol=1e-10,branch_ambiguity_atol=1e-12,coordinate_mean_range_atol=1e-12,
            budget_allowance_reference=1e-15,PMF_totals_and_CDF_SF_differences_in_summary=True,interval_arithmetic_certificate=False),
        V1_preserved=dict(interpretation='CONDITIONAL',Full_radius_inconsistent_rounds=e.data['V1VALID']['summary']['12']['model_radius_beyond_grid_atol_rows'],
            unit_radius_not_rigorous=True,Full_control_missing=True,original_scientific_status_unchanged=True),
        summary=summary,runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            available_CPU_affinity=len(os.sched_getaffinity(0)),parallel_probability_workers=workers,
            elapsed_seconds=time.perf_counter()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            artificial_resource_caps=False,shared_environment_modified=False),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_sha256=hashlib.sha256((HERE/'test_saved_v2.py').read_bytes()).hexdigest(),
        mandatory_STOP_after_publication_verification=True,publication='Verify remote SHA and full public bytes after commit; report in final chat')
    write_json(HERE/'VALIDATION.json',validation)
    print(json.dumps(dict(phase='V2_ANALYSIS_COMPLETE',tests=validation['artificial_tests'],probabilities=summary,
        branch_energy={ld:{k:b[k] for k in ('status','energy_absolute_error','reconstructed_energy_in_initial_principal_window','unique_mean_branches','all_mean_branches_match_oracle')} for ld,b in branches.items()}),allow_nan=False))


if __name__=='__main__':main()
