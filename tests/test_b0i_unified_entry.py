"""The same schedule CLI routes saved B0I metadata and artificial completion.

Saved H4 is plan-only.  The executed provider/Q3 test has artificial signals,
prices and MSE, and cannot be admitted to the production catalog.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from trottertracks import rpe_method_optimizer as O
from trottertracks.saved_rpe_cost_catalog import Catalog
import run_rpe_schedule_search as ENTRY
import test_pf_auto_entry as FIXTURE

PIN='5efa6e97faf5906320c14d6fff58220f259b6670'
DIR='docs/research/b0i_prefix_auto_selection_20261117'
PROVIDER=ROOT.parent/'pr-width-b0i-plan-source-36ad-20261117'
PROVIDER_PIN='36ad751d4ebf0ec160532615da6425b268a0c18a'


def read_published(name):
    return json.loads(subprocess.check_output(['git','show',PIN+':'+DIR+'/'+name],cwd=ROOT))


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(O.jsonable(value),indent=2)+'\n')


def digests(paths):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}


def invoke(problem_file,round_file,run,plan_only=False):
    argv=['--problem',str(problem_file),'--round',str(round_file),'--run-dir',str(run)]
    if plan_only:argv.append('--plan-only')
    with redirect_stdout(io.StringIO())as capture:
        ENTRY.main(argv)
    with(run.parent/'ENTRY_STDOUT.log').open('a')as stream:stream.write(capture.getvalue())


def saved_acceptance(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROVIDER,text=True).strip()
    if actual!=PROVIDER_PIN:raise AssertionError('Fixed D plan provider changed')
    problem=read_published('PROBLEM.json');spec=read_published('SAVED_PLAN_ROUND.json')
    if 'selected'in spec['pf_completion']:raise AssertionError('No manual p selection allowed')
    # D's existing source-bound output rule is preserved. Use this integration's
    # own fixed provider checkout, never another担当's execution directory.
    owned=PROVIDER/'artifacts/local_unified_b0i'/hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:16]
    spec['pf_completion'].update(entry=str(PROVIDER/'scripts/prepare_pf_candidate.py'),
        output_root=str(owned/'provider_output'),registry_file=str(owned/'REGISTRY.json'))
    pf=directory/'PROBLEM.json';rf=directory/'ROUND.json';run=directory/'run'
    put(pf,problem);put(rf,spec)
    invoke(pf,rf,run,plan_only=True)
    selected=json.loads((run/'pf_auto/SELECTION.json').read_text())
    frozen=[run/'pf_auto/SELECTION.json',run/'pf_auto/pf_completion/REQUEST.json',run/'pf_auto/pf_completion/RESPONSE.json']
    before=digests(frozen)
    invoke(pf,rf,run,plan_only=True)
    after=digests(frozen)
    choice=selected['selected'];key=choice['key']
    if (key['p'],key['formula'],str(key['delta_over_h']))!=(9,'4th','32/5'):
        raise AssertionError('Published saved H4 p9/Yoshida4/32h/5 selection changed: '+str(key))
    if before!=after:raise AssertionError('Completed plan request/response changed on resume')
    if list(run.glob('jobs/*/*/LAUNCH.json')):raise AssertionError('Saved plan launched Q3')
    status=json.loads((run/'pf_auto/STATUS.json').read_text())
    if status['status']!='D_PLAN_ONLY_NO_SCIENCE':raise AssertionError(status['status'])
    result=dict(schema='B0I_UNIFIED_SAVED_PLAN_ACCEPTANCE_V1',scientific_calls=0,
        source=dict(commit=PIN,problem=DIR+'/PROBLEM.json',round=DIR+'/SAVED_PLAN_ROUND.json'),
        provider_commit=actual,entry='scripts/run_rpe_schedule_search.py',
        selected=choice['key'],configuration_id=choice['configuration_id'],
        no_manual_p='selected'not in spec['pf_completion'],resume_request_response_unchanged=before==after,
        plan_hashes=before,Q3_launches=0,production_catalog_modified=False)
    put(directory/'CHECKS.json',result);return result


def artificial_acceptance(directory,scheduler='concurrent'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    problem,spec,saved,index,facts,full,saved_full=FIXTURE.make_inputs(directory,scheduler)
    # Use the same public candidate metadata and generic priority rule as the
    # saved case. Signals, prices and MSE remain entirely artificial fixtures.
    published=read_published('PROBLEM.json')
    idx=published['evidence']['pf_candidate_index']
    index=json.loads(subprocess.check_output(['git','show',idx['commit']+':'+idx['path']],cwd=ROOT))
    for method in problem['methods']:
        problem['methods'][method].update(formulas=['2nd','4th','4th(m5_best)'],delta_over_h_range=['1/2','16'])
    problem['evidence']['pf_candidate_index']=idx
    pf=directory/'PROBLEM.json';rf=directory/'ROUND.json';run=directory/'run'
    put(pf,problem);put(rf,spec)
    reader=FIXTURE.FixtureReader(index)
    with patch.object(O,'hamiltonian_facts',return_value=facts), \
         patch.object(O,'load_catalog',side_effect=lambda *_:(reader,dict(comparison={}),dict(candidates=deepcopy([saved,saved_full])))), \
         patch.object(O,'check_comparison'):
        invoke(pf,rf,run)
        first=json.loads((run/'ROUND_RESULT.json').read_text())
        launches=digests(run.glob('jobs/*/*/LAUNCH.json'))
        invoke(pf,rf,run)
        second=json.loads((run/'ROUND_RESULT.json').read_text())
    selection=json.loads((run/'pf_auto/SELECTION.json').read_text());selected=selection['selected']
    if (selected['key']['p'],selected['key']['formula'],str(selected['key']['delta_over_h']))!=(9,'4th','32/5'):
        raise AssertionError('Artificial response did not follow the public metadata selection')
    jobs=list(run.glob('jobs/TRUNCATED_PF/*/REQUEST.json'))
    requests=[json.loads(p.read_text())for p in jobs]
    outcomes=[json.loads((p.parent/'OUTCOME.json').read_text())for p in jobs]
    calls=sum(len(p.read_text().splitlines())for p in run.glob('pf_auto/provider_output/*/PROVIDER_CALLS.jsonl'))
    checks=dict(no_manual_prefix='selected'not in spec['pf_completion'],selected_p9=selected['key']['p']==9,
        provider_calls=calls,artificial_Q3_launches=len(jobs),real_Q3_launches=0,
        normal_shot_seed=bool(requests)and all(r['origin']['kind']=='GENERATOR_SEED'for r in requests),
        all_artificial=all(r['route']['kind']=='artificial'for r in requests)and all(o.get('artificial')for o in outcomes),
        resume_new_calls=second['methods']['TRUNCATED_PF']['new_calls'],
        launch_files_unchanged=launches==digests(run.glob('jobs/*/*/LAUNCH.json')),
        selected_returned_to_search=first['methods']['TRUNCATED_PF']['best']['configuration_id']==selected['configuration_id'],
        full_saved_best_preserved=first['methods']['FULL_PF']['best']['configuration_id']==full['id'],
        no_full_Q3_launches=not list(run.glob('jobs/FULL_PF/*/LAUNCH.json')),
        no_production_consumer_join=not list(directory.rglob('C_RESULT_JOIN.json')))
    if calls!=1 or len(jobs)!=1 or checks['resume_new_calls']!=0:raise AssertionError(checks)
    for k in ('no_manual_prefix','selected_p9','normal_shot_seed','all_artificial','launch_files_unchanged',
              'selected_returned_to_search','full_saved_best_preserved','no_full_Q3_launches','no_production_consumer_join'):
        if checks[k]is not True:raise AssertionError(k)
    status=json.loads((run/'pf_auto/STATUS.json').read_text())
    registry=json.loads(Path(status['response']['registry_file']).read_text())
    for artifact in (registry,first):
        class Consumer:
            root=ROOT
            def commit(self,v):return v
            def read(self,s):return artifact
        try:Catalog(Consumer(),{}).add(dict(commit='ARTIFICIAL',path='docs/artificial/ROUND_RESULT.json'))
        except ValueError as error:
            if 'artificial'not in str(error).lower():raise
        else:raise AssertionError('Production catalog admitted artificial data')
    methods={m:dict(configuration_id=r['best']['configuration_id'],cost=r['best']['cost'],status=r['best']['status'])
        for m,r in first['methods'].items()}
    result=dict(schema='B0I_UNIFIED_ARTIFICIAL_ACCEPTANCE_V1',artificial_only=True,scientific_calls=0,
        scheduler=scheduler,entry='scripts/run_rpe_schedule_search.py',checks=checks,
        selected=selected['key'],method_comparison=methods,cost_quantity='ARTIFICIAL_ONLY_NOT_COMPILED_RZ',
        first_new_calls=first['methods']['TRUNCATED_PF']['new_calls'],resume_new_calls=checks['resume_new_calls'],
        production_catalog_modified=False)
    put(directory/'FIRST_RESULT.json',first);put(directory/'RESUMED_RESULT.json',second)
    put(directory/'CHECKS.json',result);return result


class UnifiedB0I(unittest.TestCase):
    def test_fixed_saved_H4_metadata_selects_p9_without_scientific_calls(self):
        with tempfile.TemporaryDirectory(prefix='unified-b0i-saved-')as tmp:
            result=saved_acceptance(tmp);self.assertEqual(result['Q3_launches'],0)

    def test_artificial_completion_shots_Q3_comparison_and_resume(self):
        for scheduler in ('legacy','concurrent'):
            with self.subTest(scheduler=scheduler),tempfile.TemporaryDirectory(prefix='unified-b0i-artificial-')as tmp:
                result=artificial_acceptance(tmp,scheduler);self.assertEqual(result['resume_new_calls'],0)


if __name__=='__main__':
    if '--output'in sys.argv:
        import argparse
        p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
        result=dict(saved=saved_acceptance(args.output/'saved'),
                    artificial={s:artificial_acceptance(args.output/('artificial_'+s),s)for s in ('legacy','concurrent')})
        put(args.output/'SUMMARY.json',result)
        print(json.dumps(result,indent=2))
    else:unittest.main(verbosity=2)
