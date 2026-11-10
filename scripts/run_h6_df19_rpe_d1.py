"""Explicit H6 D1 scientific entry; invoke only within the permitted campaign.

No chemistry construction, state loading, eigensolve, signal evolution,
trajectory, circuit, compilation, price fit or full-RPE resource sum.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h6_df19_rpe_d1 as d1

RECEIPT_HASHES=dict(
    snapshot_receipt='3dc5b1c695f1a042ee0c2b5e3762ae0032af90f03017667541ea470577f35e3f',
    df_receipt='89cc00ab1c1b6ff65804781b5904b127e5d2e81acd15d66b0c87a86adb4c5f55')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','snapshot-receipt','df-receipt','saved-points','output'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Use a fresh D1 result path')
    saved=json.loads(Path(args.saved_points).read_bytes())
    original=subprocess.check_output(['git','show',d1.PREPARATION+
        ':docs/research/h6_df19_rpe_preparation_20261110/SAVED_POINTS.json'],cwd=ROOT)
    if original!=Path(args.saved_points).read_bytes():
        raise ValueError('Fixed saved-point projection differs')
    for name,expected in RECEIPT_HASHES.items():
        if sha(getattr(args,name))!=expected:raise ValueError('Original receipt identity differs')
    receipt=json.loads(Path(args.snapshot_receipt).read_bytes())
    df=json.loads(Path(args.df_receipt).read_bytes());meta=receipt['metadata']
    if sha(args.snapshot)!=receipt['sha256'] or sha(args.snapshot)!=d1.connection.BINDING['snapshot']:
        raise ValueError('Snapshot scientific identity differs')
    if (meta['hamiltonian_hash']!=d1.connection.BINDING['hamiltonian']
            or meta['state_hash']!=d1.connection.BINDING['state']
            or df['hamiltonian_hash']!=meta['hamiltonian_hash'] or df['actual_rank']!=19):
        raise ValueError('Saved H/state/DF binding differs')
    points=saved['points']
    if len(points)!=4 or {x['candidate']['id'] for x in points}!=set(d1.IDS):
        raise ValueError('Exactly four fixed H6 D1 points required')
    cache={};hydrated=[]
    for point in points:
        d1.connection.inspect_saved_point(point)
        src=point['allowance_source'];key=(src['commit'],src['path'])
        if key not in cache:
            data=subprocess.check_output(['git','show',key[0]+':'+key[1]],cwd=ROOT)
            cache[key]=json.loads(gzip.decompress(data))
        matches=[e for e in cache[key] if e['candidate']['id']==point['candidate']['id']
                 and e['epsilon']==src['epsilon_signal']]
        if len(matches)!=1:raise ValueError('Missing or duplicate original allowance row')
        hydrated.append(d1.hydrate_saved_allowances(point,matches[0]))
    coeff=d1.load_coefficient_payload(args.snapshot)
    prior=d1.coefficient_prior(*coeff)
    config=d1.configuration(prior['energy_prior_ha'])
    rows=[];partial=[]
    if config['status']=='CONDITIONAL_D1_CONFIGURATION':
        rows=[d1.analyze_point(x,config,state_energy=meta['solver_energy'],
              state_residual=meta['solver_residual']) for x in hydrated]
        partial=d1.partial_resources(hydrated,rows,config)
    files=['src/trottertracks/h6_df19_rpe_d1.py','src/trottertracks/h4_df10_rpe_d1.py',
        'src/trottertracks/h6_df19_rpe_connection.py','src/trottertracks/h4_df10_rpe_connection.py',
        'src/trottertracks/c1_stage3_accuracy.py','src/trotterlib/rpe_resource_accounting.py',
        'src/trotterlib/rpe_target_round_horizon_validation.py','scripts/run_h6_df19_rpe_d1.py']
    result=dict(status='H6_D1_COEFFICIENT_PRIOR_AND_SAVED_SCALAR_CAMPAIGN_COMPLETED'
        if rows else config['status'],
        scientific_source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        preparation_source=d1.PREPARATION,H4_reuse_source='ffe8a71da98bec330ce1af109a9606f14e1fc598',
        prior=prior,configuration=config,rows=rows,partial_resources=partial,
        input_bindings=dict(snapshot_sha256=sha(args.snapshot),
            snapshot_receipt_sha256=sha(args.snapshot_receipt),df_receipt_sha256=sha(args.df_receipt),
            saved_points_sha256=sha(args.saved_points),binding=d1.connection.BINDING,
            allowance_sources=[dict(commit=c,path=p) for c,p in cache]),
        source_files={name:sha(ROOT/name) for name in files},
        scope=dict(coefficient_prior_evaluations=1,horizon_evaluations=int(bool(rows)),
            saved_point_evaluations=len(rows),partial_cost_tables=len(partial),
            state_vector_loaded=False,full_H_assembled=False,trial_state_vector_built=False,
            eigensolves=0,new_signals=0,new_trajectories=0,circuit_builds=0,circuit_compiles=0,
            price_refits=0,full_RPE_resource_sums=0,full_RPE_simulations=0,H4_scientific_calls=0),
        full_RPE_qualified=False,unconditional_ground_identification_error=None,mandatory_stop=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=out.name,status=result['status'],saved_point_evaluations=len(rows))))


if __name__=='__main__':main()
