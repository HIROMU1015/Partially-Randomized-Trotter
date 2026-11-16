"""Read-only acceptance of published width metadata; no numerical H4 actions.

This checks catalog identity/price reuse and request generation, not signal or
RMSE performance. Array loading is forbidden in this process and DF children.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from trottertracks import rpe_method_optimizer as O
from trottertracks import pauli_calibration_request as C


def check():
    import numpy as np
    with patch.object(np, 'load', side_effect=AssertionError('No H/state array loads in saved acceptance')):
        p = O.load_json(ROOT/'configs/rpe_method_optimizer/h4_df10_problem.json')
        reader, inputs, cat = O.load_catalog(ROOT, p)
        facts = O.hamiltonian_facts(p); O.check_comparison(p, facts, inputs['comparison'])
        old = O.enumerate_candidates(reader, p, facts, cat)
        scoped = deepcopy(p)
        for method in ('PR_PAULI', 'PR_DF'):
            scoped['methods'][method]['requested_delta_pf_values'] = [O.enc(Fraction(.1))]
        new = O.enumerate_candidates(reader, scoped, facts, cat)
        reuse = {}
        for method in ('PR_PAULI', 'PR_DF'):
            before = [v for v in old if v['method'] == method]
            after = [v for v in new if v['method'] == method]
            assert before == after, method+' single-width catalog changed'
            # Each price vector / saved point ID is passed through untouched.
            reuse[method] = dict(candidates=len(before), saved_point_ids=[x for v in before for x in v['saved_points']],
                configuration_ids=[v['configuration_id'] for v in before], candidates_identical=True)
        generated = []
        for method in ('PR_PAULI', 'PR_DF'):
            point = next(v for v in cat['candidates'] if v['method'] == method)
            source = point['configuration_source']; base = reader.read(source)
            got = O.run_generator(ROOT, p['methods'][method]['generation'],
                [dict(item_id='SAVED_SINGLE_WIDTH',base=source,changes=dict(delta_over_h='1'))])['items'][0]
            assert got['status'] == 'GENERATED' and got['base_reproduced'] and got['configuration'] == base
            generated.append(dict(method=method, source=source, configuration_id=base['id'],
                all_rounds_identical=True, base_reproduced=True))

        folder = ROOT/'docs/research/rpe_method_optimizer_h4_20261116/kappa_abs_round04/width_requests'
        rows = O.load_json(folder/'WIDTH_REQUESTS.json')['rows']; quotes = O.load_json(folder/'QUOTES.json')['items']
        expected = O.load_json(folder/'BATCH.json'); requests = []
        for row in rows:
            config = O.load_json(folder/'configurations'/('LD%d_%s_kappa_2.json'%(row['L_D'],row['width']['label'])))
            q = next(v['raw'] for v in quotes if v['configuration_id'] == config['id'])
            requests.append(dict(configuration=config, quote=q))
        got = C.prepare_many(ROOT, requests)
        assert got['unique_calibration_requests'] == expected['unique_calibration_requests'] == 4
        checks = []
        for r, e, inp in zip(got['results'], expected['results'], rows):
            assert r['configuration_id'] == e['configuration_id'] and r['status'] == e['status']
            assert r['round_actions'] == e['round_actions']
            assert [g['request_id'] for g in r['calibration_requests']] == [g['request_id'] for g in e['calibration_requests']]
            checks.append(dict(L_D=inp['L_D'], width=inp['width'], configuration_id=r['configuration_id'],
                status=r['status'], actions=[dict(m=a['m'],action=a['action']) for a in r['round_actions']],
                calibration_request_ids=[g['request_id'] for g in r['calibration_requests']],
                covered_rounds=[g['covered_rounds'] for g in r['calibration_requests']], identical_to_round04=True))
        assert got['science_calls'] == 0 and got['execution_authorized'] is False

    # Fresh DF service interpreter prevents provider module graphs mixing. It
    # still forbids all numpy array loads; canonical mapper is metadata only.
    code = """import json,sys,numpy as np
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=['src','scripts']
import pr_candidate_services as S
with patch.object(np,'load',side_effect=AssertionError('No H/state array loads')):
    r=json.loads(sys.argv[1]);out=S.df_generate(r)
    if out['status']=='CONFIGURATION_READY':
        c=out['configuration'];out={'status':out['status'],'configuration_id':c['id'],'h':c['spec']['h'],'M':c['spec']['M'],'requested':c['spec']['delta_pf'],'rounds':[{'m':v['m'],'q':v['q_PF'],'R':v['R_outer'],'actual_step':v['actual_delta_pf'],'physical_id':v['candidate']['id']} for v in c['rounds']]}
    print(json.dumps(out))
"""
    df = []
    for delta in (.1, .05, .2, '1/5'):
        req = dict(design=dict(L_D=7,formula='2nd',delta_pf=delta,kappa='2',K=2))
        result = json.loads(subprocess.check_output([sys.executable,'-B','-c',code,json.dumps(req)],cwd=ROOT))
        if delta in (.1,.05):
            assert result['status'] == 'CONFIGURATION_READY'
            assert [r['q'] for r in result['rounds']] == [(1 if delta==.1 else 2)*2**m for m in range(13)]
            assert all(Fraction(**r['actual_step']) == Fraction(delta) for r in result['rounds'])
        elif delta == .2:
            assert result['status'] == 'UNSUPPORTED' and result['reason'] == 'DF_MAPPER_REJECTED_REQUESTED_STEP_WIDTH'
        else:
            assert result['status'] == 'UNSUPPORTED' and result['reason'] == 'REQUESTED_DF_DELTA_NOT_EXACT_BINARY64'
        df.append(dict(request=req,result=result))
    old_round = subprocess.check_output(['git','show','d8e806265c3ded8e1184181c37c2d7ec71f08ade:configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json'],cwd=ROOT)
    assert old_round == (ROOT/'configs/rpe_method_optimizer/rounds/h4_pr_pauli_schedule_wide_round05.json').read_bytes()
    return dict(scope='SAVED_METADATA_AND_CANONICAL_GENERATORS_ONLY_NO_SCIENCE',
        baseline='d8e806265c3ded8e1184181c37c2d7ec71f08ade',single_width_catalog=reuse,
        single_width_generators=generated,pauli_round04=checks,unique_pauli_cells=4,df_mapper=df,
        original_round05_sha256=hashlib.sha256(old_round).hexdigest(),original_round05_unchanged=True,
        new_H4_actions=0,paths=0,compiles=0,fits=0,Q3=0)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path)
    args=ap.parse_args();result=check()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',single_width_catalog={k:v['candidates'] for k,v in result['single_width_catalog'].items()},
        pauli_requests=len(result['pauli_round04']),df_cases=len(result['df_mapper']),new_science_calls=0)))
