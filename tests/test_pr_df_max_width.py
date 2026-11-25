"""Exercise exact mapper identities and the one-body endpoint with no real signals."""
import json
from pathlib import Path
import subprocess
from copy import deepcopy
from fractions import Fraction
import pytest

from trottertracks import pr_df_max_width as W
from trottertracks import pr_price_rules as prices

ROOT=Path(__file__).resolve().parents[1]
BASE='0defa286c53a1b9acf513be453907bcad83102b3:docs/research/d_pr_df_ld7_v0_results_20261116/generation/CONFIGURATION.json'

@pytest.fixture(scope='module')
def base():return json.loads(subprocess.check_output(['git','show',BASE],cwd=ROOT))

def config(base,ld,width,k=8):
    tail=dict(base['spec']['A_input']['tail_norm']);tail['lambda_R']=.375
    return W.build(ROOT,base,dict(L_D=ld,formula='2nd',K=2,delta_pf=W.encode(Fraction(.1)*width),kappa=str(k)),
                   tail=tail,tail_source={'fixture':'tail'},source='ARTIFICIAL_TEST')

@pytest.mark.parametrize('width',range(1,11))
def test_requested_width_and_round_steps(base,width):
    cfg=config(base,7,width);W.validate(ROOT,cfg)
    assert W.exact(cfg['spec']['delta_pf'])==Fraction(.1)*width
    assert all(W.actual_step_matches(r) for r in cfg['rounds'])
    for r in cfg['rounds']:
        assert prices.point(r)['q']==r['q_PF']
        assert r['physical_candidate']['delta'].hex()==r['mapping']['actual_delta_binary64']
    corrupt=deepcopy(cfg);corrupt['rounds'][0]['mapping']['requested_delta_pf']='1/1'
    with pytest.raises(ValueError):W.validate(ROOT,corrupt)

def test_one_body_endpoint_plan_is_not_pure_rte(base):
    cfg=config(base,0,3);W.validate(ROOT,cfg)
    assert cfg['spec']['split']['one_body_policy']=='DETERMINISTIC'
    assert cfg['spec']['method']=='PR'
    plan=W.jobs(cfg,{r['m']:r['physical_candidate'] for r in cfg['rounds']},b_validate=lambda c:W.validate(ROOT,c),
        b_exact_scalar=W.exact,numerical_recipe=dict(source={'fixture':True},deterministic_PF='DIRECT_EXPM',finite_RTE='FIXTURE',sampling=False))
    assert len(plan['jobs'])==13 and all(j['physical_context']['physical_candidate']['prefix']==0 for j in plan['jobs'])
