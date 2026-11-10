"""Explicit approved D1 entry: saved scalar/coefficients only; no provider import."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import h4_df10_rpe_d1 as d1


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--saved-points',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output)
    if out.exists():raise FileExistsError('Use fresh D1 output; no result overwrite')
    saved=json.loads(Path(args.saved_points).read_text());receipt=json.loads(Path(args.receipt).read_text())
    original=subprocess.check_output(['git','show',d1.PREPARATION+':docs/research/h4_df10_rpe_preparation_20261110/SAVED_POINTS.json'],cwd=ROOT)
    if original!=Path(args.saved_points).read_bytes():raise ValueError('Fixed preparation input differs')
    if sha(args.receipt)!=saved['state_metadata']['receipt_sha256']:
        raise ValueError('Original H4 receipt hash differs from preparation')
    if sha(args.snapshot)!=receipt['sha256']:raise ValueError('Saved snapshot/receipt identity differs')
    points=[x for x in saved['points'] if x['candidate']['id'] in d1.IDS]
    if {x['candidate']['id'] for x in points}!=set(d1.IDS) or len(points)!=2:raise ValueError('Fixed two D1 points required')
    for x in points:
        if x['binding']['hamiltonian']!=receipt['hamiltonian_hash'] or x['binding']['state']!=receipt['state_hash']:
            raise ValueError('Saved H/state binding differs')
    # Decode saved coefficient arrays only: no Hamiltonian, state vector or solve.
    import numpy as np
    with np.load(args.snapshot,allow_pickle=False) as z:
        h=z['one_body'].tolist();weights=z['lambdas'].tolist();g=z['g_matrices'].tolist()
        constant=float(z['constant'].reshape(-1)[0]);mask=int(z['sector_basis_indices'][0])
    occupied=[i for i in range(len(h)) if mask>>i&1]
    if len(occupied)!=4 or sum(i%2==0 for i in occupied)!=2 or sum(i%2==1 for i in occupied)!=2:
        raise ValueError('Prior determinant must belong to the saved H4 spin sector')
    prior=d1.coefficient_prior(constant,h,weights,g,occupied)
    config=d1.configuration(prior['energy_prior_ha'])
    rows=[d1.analyze_point(x,config,state_energy=receipt['state_energy'],state_residual=receipt['state_residual']) for x in points]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result=dict(status='D1_SAVED_T08_CONDITIONAL_NUMERICAL_PILOT_COMPLETED',
        scientific_source=commit,preparation_source=d1.PREPARATION,
        prior=prior,configuration=config,rows=rows,
        input_bindings=dict(snapshot_sha256=sha(args.snapshot),receipt_sha256=sha(args.receipt),
            saved_points_sha256=sha(args.saved_points),hamiltonian_hash=receipt['hamiltonian_hash'],state_hash=receipt['state_hash']),
        source_files={name:sha(ROOT/name) for name in ['src/trottertracks/h4_df10_rpe_d1.py',
            'scripts/run_h4_df10_rpe_d1.py','src/trottertracks/h4_df10_rpe_connection.py',
            'src/trottertracks/c1_stage3_accuracy.py','src/trotterlib/rpe_target_round_horizon_validation.py',
            'src/trotterlib/rpe_resource_accounting.py']},
        scope=dict(coefficient_prior_evaluations=1,horizon_evaluations=1,saved_point_evaluations=2,
            state_vector_loaded=False,full_H_assembled=False,eigensolves=0,new_signals=0,
            new_trajectories=0,circuit_builds=0,circuit_compiles=0,price_refits=0,
            resource_products=0,full_RPE_simulations=0,H6_numerical_evaluations=0),
        full_RPE_qualified=False,mandatory_stop=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(output=out.name,status=result['status'],base_time=config['base_time'],
        round_count=config['horizon']['round_count_M_plus_one'],D1_m=config['D1_m'],
        rows=[dict(id=r['candidate_id'],status=r['ideal_numerical_pilot']['status'],
                   N_X=r['ideal_numerical_pilot']['N_X'],N_Y=r['ideal_numerical_pilot']['N_Y']) for r in rows])))


if __name__=='__main__':main()
