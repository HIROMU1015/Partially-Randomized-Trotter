#!/usr/bin/env python3
"""Metadata-only decoder-statistics preparation and D receipt-coverage intake.

No real means are passed to a decoder, RNG, RMSE calculation or N solver.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_decoder_statistics as S
from trottertracks import rpe_d_evidence_intake as D
from trottertracks import abc_s0_published_intake as H


def build():
    store=S.p.SavedArtifacts(ROOT)
    config=store.read(dict(commit=H.B_RELEASE,path=H.BDIR+'CONFIGURATION_V2.json'))
    packet_ref=dict(commit='3e035c0d61386010f59ad7ea792bc7a5cefd18ac',path='docs/research/h4_pr_m1_rmse_results_20261113/run01/N_HANDOFF.json')
    packet=store.read(packet_ref)
    shots=[]
    for row,supplied in zip(config['rounds'],packet['rounds']):
        H.H.same(row['identity'],supplied['identity'],'STATISTICS_NOMINAL_PACKET_IDENTITY_MISMATCH')
        H.H.same(row['law_binding'],supplied['law_binding'],'STATISTICS_NOMINAL_PACKET_LAW_MISMATCH')
        shots.append(dict(m=row['m'],N_X=supplied['nominal_N_X'],N_Y=supplied['nominal_N_Y']))
    # No target value is extracted/recomputed; A must provide it with error and
    # original-H/state SOURCE. Nominal counts alone cannot run a simulation.
    target=dict(quantity='ORIGINAL_DF_H_TARGET_ENERGY',binding=deepcopy(config['spec']['binding']),energy_shift_ha=0,
        energy_ha=None,uncertainty_ha=None,source={'requirement':'A_ORIGINAL_H_TARGET_REFERENCE_AND_STATE_ERROR'})
    prepared=S.prepare(config,[],shots,target=target,shot_policy=dict(meaning='NOMINAL',source=packet_ref))
    metadata=dict(configuration_id=config['id'],M=config['spec']['M'],h=config['spec']['h'],kappa=config['spec']['kappa'],
        rmse_policy=config['spec']['rmse_policy'],decoder=prepared['decoder'],prior=prepared['prior'],
        target=target,shot_policy=prepared['shot_policy'],decoder_SOURCE=S.SOURCE,
        mean_values_imported_for_statistics=False,input_complete=False,statistics_executed=False,
        nominal_rounds=[r['m'] for r in shots if r['N_X'] is not None and r['N_Y'] is not None],
        sufficient_N=None,global_RMSE_bound=None,empirical_H4_RMSE=None,G=None,
        rows=[dict(m=r['m'],candidate=r['physical_candidate'],issues=r['issues'],shots=r['shots'],
            canonical_candidate_id=r['identity']['candidate']['id'],
            mean_input='NOT_SUPPLIED_TO_STATISTICS',normalization_qualification=None) for r in prepared['rows']],
        unresolved_loss_policy=dict(kind='UNDEFINED',source='Existing decoder provides no unresolved-trial energy/loss'),
        new_scientific_simulation=False,mandatory_stop=True)
    previous=store.read(dict(commit=D.COVERAGE,path=D.COVERAGE_JSON))
    current=store.read(dict(commit=D.RECEIPT_RELEASE,path=D.RECEIPT_DIR+'COVERAGE_V2.json'))
    receipts=store.read(dict(commit=D.RECEIPT_RELEASE,path=D.RECEIPT_DIR+'RECEIPTS.json'))
    accepted=D.accept_receipt_coverage(previous,current,receipts)
    return dict(PREPARATION=metadata,D_RECEIPTS_ACCEPTANCE=accepted,SOURCE_FILES=store.sources)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():parser.error('Fresh output directory required')
    products=build()
    args.out.mkdir(parents=True)
    for name,value in products.items():
        (args.out/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(statistics_executed=False,mean_values_imported=False,D_exact=products['D_RECEIPTS_ACCEPTANCE']['totals']['X'],G=None)))


if __name__=='__main__':main()
