"""Artificial scalar/mock demonstration only. No molecular input or science CLI."""
from copy import deepcopy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from trottertracks import pr_rpe_rmse as model


def artificial_design():
    return dict(system='TOY',base_time=.1,delta=.1,formula='2nd',prefix=1,K=2,
        PF=dict(method='FULL',prefix=2),binding=dict(system='TOY',hamiltonian='ARTIFICIAL_H',state='ARTIFICIAL_STATE',snapshot='ARTIFICIAL_SNAPSHOT'),
        profile=dict(compiler='ARTIFICIAL_COMPILER'),contract=dict(kind='ARTIFICIAL_ONLY'),
        energy_prior_ha=[-1.,1.],decoder='consecutive_dyadic_nearest_branch_principal_energy_v1',
        state_assumption='ARTIFICIAL_EXACT_STATE_MODEL',source=dict(kind='ARTIFICIAL_ONLY'),early_rounds=[])


def mock_provider(rows):
    out=[]
    for r in rows:
        c=r['identity']['candidate'];st=r['identity']['structure']
        log_B=0. if model.common.arm(c)=='PF' else .05
        z=dict(real=.9,imag=.1)
        signal=dict(reference=z,corrected=z,raw=z,exact_tail=z,log_B=log_B)
        record=dict(candidate=c,signal=signal,**{k:st[k] for k in ('signed_weights','allocation','event_count')})
        out.append(model.mean_evidence(r,record,dict(kind='MOCK_ONLY')))
    return out


def demo():
    design=artificial_design()
    policy=model.rmse_policy(target_rmse='1',qpe_rmse='1/2',evolution_rmse='1/10',
        error_rule='paper_quadrature',normalization_rule='finite_B_diagnostic',source=dict(kind='ARTIFICIAL_ONLY'))
    tail=dict(lambda_R='1/2',prefix=1,binding=deepcopy(design['binding']),source=dict(kind='ARTIFICIAL_ONLY'))
    proposal=model.schedule(M=3,kappa_paper='2',design=design,tail_norm=tail,policy=policy)
    result=model.evaluate(proposal,mean_provider=mock_provider)
    return dict(evidence_domain='ARTIFICIAL_ONLY',schedule=proposal,result=result,
        B_handoff=model.b_handoff(proposal,result,{'kind':'ARTIFICIAL_ONLY'},evidence_domain='SYNTHETIC_ONLY'),
        real_science_calls=0,mandatory_stop=True)


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: demo_pr_rpe_rmse.py <fresh-output.json> (ARTIFICIAL ONLY)')
    path=Path(sys.argv[1])
    if path.exists():raise FileExistsError('Fresh output required')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(demo(),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
