#!/usr/bin/env python3
"""Postprocess the complete C6a pilot into a conditional C6b planning proposal.

No sampling, circuits, ground solves, fits, N/RPE accounting or holdout execution.
Reads the frozen runner's existing statistics/model; normal planning assumptions
are proposals, not a qualified confidence interval or long-q price claim.
"""
from pathlib import Path
import hashlib
import json
import math
from statistics import NormalDist

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/track_d_c6a_paired_price/2026-11-06/run01'


def main():
    result=json.loads((OUT/'result.json').read_text())
    if result['status']!='PILOT_COMPLETE_MANDATORY_STOP' or len(result['completed_ids'])!=32:
        raise ValueError('COMPLETE_PLANNED_C6A_REQUIRED')
    target=OUT/'C6b_planning_proposal.json'
    if target.exists():raise ValueError('FRESH_POSTPROCESS_OUTPUT')
    summary=json.loads((OUT/'paired_statistics.json').read_text())
    fit=json.loads((OUT/'exploratory_affine.json').read_text())
    labels=[tuple(x) for x in summary['by_q']['2']['sample_covariance_labels'] if x[2]!='peak_qubits']
    n=16
    def covariance(q,x,y):
        group=summary['by_q'][str(q)];all_labels=[tuple(z) for z in group['sample_covariance_labels']]
        return group['sample_covariance'][all_labels.index(x)][all_labels.index(y)]/n
    parameter_labels=[[*x,p] for x in labels for p in ['slope','intercept']]
    parameter_cov=[]
    for x in labels:
        for p in ['slope','intercept']:
            row=[]
            for y in labels:
                v2,v4=covariance(2,x,y),covariance(4,x,y)
                for r in ['slope','intercept']:
                    row.append(.25*(v2+v4) if p==r=='slope' else
                               4*v2+v4 if p==r=='intercept' else -v2-.5*v4)
            parameter_cov.append(row)
    model={(m,{'cosine':'X','sine':'Y'}[r['axis']],r['metric']):r for m,rows in fit['models'].items() for r in rows}
    alpha=.05;family=36;z=NormalDist().inv_cdf(1-alpha/(2*family));relative=.05;absolute=1.
    proposals=[]
    for axis in ['X','Y']:
        for g in ['rz_count','cx_count','rz_depth','cx_depth','total_depth','circuit_size']:
            sx=('symmetric_directional',axis,g);rx=('reference',axis,g)
            prediction={x[0]:8*model[x]['slope']+model[x]['intercept'] for x in [sx,rx]}
            stats={r['metric']:r for r in summary['by_q']['4']['rows'] if r['axis']==axis}[g]
            for quantity,field in [('symmetric_directional','symmetric'),('reference','reference'),('paired_delta','paired_delta')]:
                if quantity=='paired_delta':
                    v2=covariance(2,rx,rx)+covariance(2,sx,sx)-2*covariance(2,rx,sx)
                    v4=covariance(4,rx,rx)+covariance(4,sx,sx)-2*covariance(4,rx,sx)
                    f=prediction['reference']-prediction['symmetric_directional']
                    scale=abs(prediction['symmetric_directional'])
                else:
                    x=(quantity,axis,g);v2=covariance(2,x,x);v4=covariance(4,x,x)
                    f=prediction[quantity];scale=abs(f)
                calibration_variance=max(0.,4*v2+9*v4)
                sigma8_plan=2*stats[field]['sample_SD']
                tau=absolute+relative*scale
                denominator=tau*tau-z*z*calibration_variance
                n_formula=(math.ceil(z*z*sigma8_plan*sigma8_plan/denominator) if denominator>0 else None)
                proposals.append(dict(axis=axis,metric=g,quantity=quantity,
                    hypothetical_affine_q8_value_for_planning_only=f,absolute_width_component=absolute,
                    relative_width_component=relative,pre_holdout_scale=scale,tau=tau,
                    calibration_variance_conditional=calibration_variance,
                    nominal_calibration_halfwidth=z*math.sqrt(calibration_variance),
                    assumed_sigma8_for_planning=sigma8_plan,denominator=denominator,
                    conditional_n_formula=n_formula,proposed_n_with_pilot_floor=max(32,n_formula) if n_formula is not None else None,
                    status='CONDITIONAL_SIZE_ONLY_NOT_GUARANTEED' if n_formula is not None else 'CALIBRATION_FLOOR_BLOCKS_THIS_WIDTH_EVEN_AS_HOLDOUT_N_INCREASES'))
    blocked=[r for r in proposals if r['conditional_n_formula'] is None]
    packet=dict(status='PROPOSAL_NOT_ADOPTED_NOT_EXECUTED',source_commit=result['source_commit'],
        postprocess_source=dict(path='scripts/summarize_track_d_c6a.py',sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        calibration_sources=dict(statistics_sha256=hashlib.sha256((OUT/'paired_statistics.json').read_bytes()).hexdigest(),
            exploratory_model_sha256=hashlib.sha256((OUT/'exploratory_affine.json').read_bytes()).hexdigest()),
        confidence_proposal=dict(classical_family_alpha=alpha,family_size=family,two_sided_normal_z=z,
            qualified_CI=False,assumptions='pseudo-independent q/whole-path calibration and holdout; normal approximation; no proven support/tail/finite-bit-law distance'),
        design=dict(relative_price_width=.05,absolute_price_width=1,
            paired_scale='symmetric prediction magnitude, never relative to near-zero paired difference',
            peak_qubits='separate width, no additive cost or normal interval; structural9/17 to check',
            proposed_min_holdout_paths=32,minimum_is_a_design_choice_not_sufficiency=True,
            q8_SD_transport='2 * q4 sample SD; unvalidated planning assumption',
            actual_q8_samples=0,actual_q8_circuits=0,thresholds_frozen_before_future_holdout=True),
        conditional_parameter_covariance=dict(labels=parameter_labels,matrix=parameter_cov,
            formula='b=(mean4-mean2)/2, a=2mean2-mean4; Cov(meanq)=Sq/16; cross-q covariance=0 ASSUMED conditional on disjoint pseudo-streams',
            shared_across_future_round_predictions=True,rigorous_IID_or_model_bias_bound=False),
        per_quantity=proposals,blocked_quantities=len(blocked),
        conditional_max_holdout_paths_if_all_floors_feasible=(max(r['proposed_n_with_pilot_floor'] for r in proposals) if not blocked else None),
        guaranteed_required_holdout_paths=None,long_q_qualified=False,confirmed_model_PASS=False,
        decision_needed='User/GPT chooses price widths/confidence/statistical premises; if floor blocked, wider tolerance or additional calibration needs separate permission before holdout. No automatic new data.',
        acceptance_proposal='correctness PASS prerequisite; fixed model/threshold/samplecount before independent q8; nominal |mean8-f8|+z*sqrt(s8^2/n+vcal)<=tau for every declared quantity; failure saved, no reselection/refit/threshold relaxation',
        limitations='n16 pilot cannot prove normal coverage, ideal IID bridge, support/tails, q8 variance transport or affine model bias; no total-RPE/N evaluation')
    target.write_text(json.dumps(packet,indent=2)+'\n')
    print(json.dumps(dict(status=packet['status'],blocked_quantities=len(blocked),conditional_max_holdout_paths=packet['conditional_max_holdout_paths_if_all_floors_feasible'])))


if __name__=='__main__':main()
