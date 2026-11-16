"""Normal CLI -> automatic prefix -> fixture completion -> artificial Q3 -> resume.

Only initial facts/catalog are fixtures. Canonical configuration metadata, the
ordinary CLI, fixed shot generator, preparation subprocess, registry join and
existing artificial Q3 worker all run through their usual implementations.
"""
from copy import deepcopy
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import pf_candidate_space as P,pf_optimizer_connection as PF,rpe_method_optimizer as O
from trottertracks.saved_rpe_cost_catalog import Catalog


def enc(value):
    value=Fraction(value);return dict(numerator=value.numerator,denominator=value.denominator)


class FixtureReader:
    root=ROOT
    def __init__(self,index,registry=None):self.index=index;self.registry=registry
    def commit(self,value):return value
    def document(self,commit,path):return deepcopy(self.index)
    def read(self,source):
        if self.registry is None:raise ValueError('Only fixture index is available')
        return deepcopy(self.registry)


def make_inputs(directory,scheduler):
    p7=P.configure(ROOT,'TRUNCATED_PF',7,'2nd','2');full=P.configure(ROOT,'FULL_PF',10,'2nd','2')
    price_axes=[dict(m=m,axis=a,unit_price_RZ=enc(10**6))for m in range(13)for a in ('X','Y')]
    allocation=[4]*13
    saved=dict(method='TRUNCATED_PF',configuration_id=p7['id'],L_D=7,formula='2nd',delta_pf=p7['spec']['delta_pf'],h=p7['spec']['h'],
        label='ARTIFICIAL_SAVED_P7',point_id='ARTIFICIAL_SAVED_POINT',status='FEASIBLE',q3_status=O.PASS,
        conditional_cost=enc(sum(allocation)*2*10**6),N_X=allocation,N_Y=allocation,
        MSE_lower=enc(Fraction(1,1000)),MSE_upper=enc(Fraction(1,1000)),price_axes=price_axes,
        result_sources=[dict(artificial=True)],signal_source=dict(artificial=True,kind='SAVED_P7_SIGNAL_FIXTURE'),
        binding_id='ARTIFICIAL_SAVED_P7_BINDING',total_configuration_id='ARTIFICIAL_SAVED_P7_TOTAL')
    saved_full=deepcopy(saved)
    saved_full.update(method='FULL_PF',configuration_id=full['id'],L_D=10,label='ARTIFICIAL_SAVED_FULL_PF',point_id='ARTIFICIAL_SAVED_FULL_POINT',
        conditional_cost=enc(sum(allocation)*2*2*10**6),price_axes=[dict(m=m,axis=a,unit_price_RZ=enc(2*10**6))for m in range(13)for a in ('X','Y')],
        signal_source=dict(artificial=True,kind='SAVED_FULL_PF_SIGNAL_FIXTURE'),binding_id='ARTIFICIAL_SAVED_FULL_BINDING',
        total_configuration_id='ARTIFICIAL_SAVED_FULL_TOTAL')
    index=dict(candidates=[])
    for method,p in [('B0I_P7',7),('FULL_PF',10)]:
        for formula in ('2nd','4th'):
            for width in ('1','2'):
                cfg=P.configure(ROOT,'TRUNCATED_PF'if p<10 else'FULL_PF',p,formula,width)
                index['candidates'].append(dict(method=method,formula=formula,requested_delta_over_h=width,
                    configuration_id=cfg['id'],source_bound_13round_packet=None))
    methods={m:dict(family='PF',index_method='B0I_P7'if m=='TRUNCATED_PF'else'FULL_PF',formulas=['2nd','4th'],
        delta_over_h_range=['1','2'],caps=[4]*13,p_values=list(range(11))if m=='TRUNCATED_PF'else[10])for m in ('TRUNCATED_PF','FULL_PF')}
    route=dict(kind='artificial',interpreter=sys.executable,configuration_ids='*',view_commit=None,
               artificial=dict(weights=[1]*13,scale='1/10000',tau='1/100'))
    problem=dict(schema='ARTIFICIAL_PF_AUTO_ENTRY_PROBLEM',artificial_only=True,methods=methods,
        accuracy=dict(epsilon_E_decimal='.1',scope='ARTIFICIAL_ONLY'),target=dict(energy_ha=0),rpe=dict(M=12),
        evidence=dict(pf_candidate_index=dict(commit='ARTIFICIAL',path='INDEX.json')),
        generator=dict(commit='e094b2d60c5c4468711dea244fa2f91fb587ac12',path='scripts/run_pauli_ld2_reallocation_q3_fixed.py',function='generate'),
        execution_routes={m:[deepcopy(route)]for m in methods},
        search=dict(decrement_steps=[1],large_step_headroom='9/10',seed_rules=['B_PRICE','A_SHAPE'],max_steps_per_method=1,max_new_q3_per_method=0))
    spec=dict(round_id='ARTIFICIAL_AUTOMATIC_PF_ENTRY_'+scheduler,methods=['TRUNCATED_PF','FULL_PF'],initial='all',science_approved=True,
        max_new_q3_per_method={'TRUNCATED_PF':1,'FULL_PF':0},max_steps_per_method=1,
        pf_completion=dict(mode='artificial',artificial_fixture=True,max_candidates=1,
            entry=str(ROOT/'scripts/artificial_pf_preparation_fixture.py'),interpreter=sys.executable),
        executor=dict(scheduler=scheduler,max_concurrent_jobs=2))
    binding=p7['spec']['binding'];facts=dict(fragment_count=10,snapshot_sha256=binding['snapshot'],hamiltonian_hash=binding['hamiltonian'],state_hash=binding['state'])
    return problem,spec,saved,index,facts,full,saved_full


def run_acceptance(directory,scheduler='concurrent'):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    problem,spec,saved,index,facts,full,saved_full=make_inputs(directory,scheduler)
    problem_file=directory/'PROBLEM.json';round_file=directory/'ROUND.json';run=directory/'run'
    for path,value in [(problem_file,problem),(round_file,spec)]:path.write_text(json.dumps(value,indent=2)+'\n')
    module_spec=importlib.util.spec_from_file_location('pf_auto_normal_cli_fixture_'+scheduler,ROOT/'scripts/run_rpe_method_optimizer.py')
    cli=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(cli)
    reader=FixtureReader(index);contexts=[];real_prepare=O.prepare
    def prepare(*args,**kwargs):
        context=real_prepare(*args,**kwargs);contexts.append(context);return context
    command=[str(ROOT/'scripts/run_rpe_method_optimizer.py'),'--problem',str(problem_file),'--round',str(round_file),'--run-dir',str(run)]
    with patch.object(O,'hamiltonian_facts',return_value=facts), \
         patch.object(O,'load_catalog',side_effect=lambda *_:(reader,dict(comparison={}),dict(candidates=deepcopy([saved,saved_full])))), \
         patch.object(O,'check_comparison'),patch.object(O,'prepare',side_effect=prepare),patch.object(sys,'argv',command):
        cli.main()
        first=json.loads((run/'ROUND_RESULT.json').read_text())
        (directory/'FIRST_RESULT.json').write_text(json.dumps(first,indent=2)+'\n')
        launches={str(p):p.read_bytes()for p in run.glob('jobs/*/*/LAUNCH.json')}
        cli.main()
        second=json.loads((run/'ROUND_RESULT.json').read_text())
        (directory/'RESUMED_RESULT.json').write_text(json.dumps(second,indent=2)+'\n')
    selection=json.loads((run/'pf_auto/SELECTION.json').read_text())
    chosen=selection['selected'];cid=chosen['configuration_id']
    status=json.loads((run/'pf_auto/STATUS.json').read_text())
    registry=json.loads(Path(status['response']['registry_file']).read_text())
    jobs=list(run.glob('jobs/TRUNCATED_PF/*/REQUEST.json'))
    requests=[json.loads(p.read_text())for p in jobs]
    outcomes=[json.loads((p.parent/'OUTCOME.json').read_text())for p in jobs]
    checks=dict(artificial_only=True,real_scientific_calls=0,
        no_caller_selected_prefix='selected'not in spec['pf_completion'],
        multiple_prefixes_in_selection=len({r['key']['p']for r in selection['candidates']})>1,
        multiple_formulas_in_selection=len({r['key']['formula']for r in selection['candidates']})>1,
        multiple_widths_in_selection=len({r['key']['delta_over_h']for r in selection['candidates']})>1,
        chosen_was_unseen=chosen['key']['p']not in selection['evaluated_prefixes'],
        chosen_is_not_saved_prefix=cid!=saved['configuration_id'],
        selected_canonical_id_matches_return=cid in registry['configurations'],
        selected_ready_after_completion=status['status']=='READY_RETURNED_TO_SAME_OPTIMIZER',
        ordinary_generator_seed=bool(requests)and all(q['origin']['kind']=='GENERATOR_SEED'for q in requests),
        all_fake_routes=all(q['route']['kind']=='artificial'for q in requests),
        all_fake_outcomes=all(o.get('artificial')is True for o in outcomes),
        fake_q3_count=len(outcomes),
        best_uses_selected_own_prices=first['methods']['TRUNCATED_PF']['best']['configuration_id']==cid,
        truncated_cost_below_its_saved_p7=O.frac(first['methods']['TRUNCATED_PF']['best']['cost'])<O.frac(saved['conditional_cost']),
        full_saved_best_retained=first['methods']['FULL_PF']['best']['configuration_id']==full['id']
            and O.frac(first['methods']['FULL_PF']['best']['cost'])==O.frac(saved_full['conditional_cost']),
        full_new_calls_zero=first['methods']['FULL_PF']['new_calls']==second['methods']['FULL_PF']['new_calls']==0,
        full_no_Q3_launch=not list(run.glob('jobs/FULL_PF/*/LAUNCH.json')),
        resume_no_duplicate_launch=launches=={str(p):p.read_bytes()for p in run.glob('jobs/*/*/LAUNCH.json')},
        resume_new_calls_zero=second['methods']['TRUNCATED_PF']['new_calls']==0,
        p7_saved_prices_preserved=all(next(c for c in ctx['candidates']if c['configuration_id']==saved['configuration_id'])['prices']==[Fraction(10**6)]*26 for ctx in contexts),
        p10_full_alias_once=all(sum(c['configuration_id']==full['id']for c in ctx['candidates'])==1
                                and not any(c['method']=='TRUNCATED_PF'and c['key']['p']==10 for c in ctx['candidates'])for ctx in contexts),
        provider_called_once=sum(len(p.read_text().splitlines())for p in run.glob('pf_auto/provider_output/*/PROVIDER_CALLS.jsonl'))==1,
        no_C_RESULT_JOIN=not list(directory.rglob('C_RESULT_JOIN.json')))
    for key,value in checks.items():
        if key not in ('artificial_only','real_scientific_calls','fake_q3_count')and value is not True:
            raise AssertionError(key)
    if checks['fake_q3_count']!=1:raise AssertionError('One fake Q3 was expected')
    # Scientific C consumers must refuse both the marked registry and the
    # top-level artificial ROUND_RESULT rather than comparing fake costs.
    for artifact in (registry,first):
        class ConsumerReader:
            root=ROOT
            def commit(self,value):return value
            def read(self,source):return artifact
        catalog=Catalog(ConsumerReader(),{})
        try:catalog.add(dict(commit='ARTIFICIAL',path='docs/artificial/ROUND_RESULT.json'))
        except ValueError as error:
            if 'artificial'not in str(error).lower():raise AssertionError('Explicit artificial rejection required')from error
        else:raise AssertionError('Scientific C consumer admitted artificial evidence')
    checks['scientific_C_rejects_artificial']=True
    summary=dict(schema='ARTIFICIAL_PF_NORMAL_ENTRY_ACCEPTANCE_V1',artificial_only=True,checks=checks,selection=selection,
                 cost_quantity='ARTIFICIAL_FIXTURE_UNITS_NOT_COMPILED_RZ',
                 first_new_calls=first['methods']['TRUNCATED_PF']['new_calls'],resume_new_calls=second['methods']['TRUNCATED_PF']['new_calls'],
                 selected_configuration_id=cid,request_sources=[str(p)for p in jobs],production_catalog_modified=False,
                 method_comparison={m:dict(configuration_id=r['best']['configuration_id'],cost=r['best']['cost'],status=r['best']['status'],
                    artificial_only=True)for m,r in first['methods'].items()})
    (directory/'CHECKS.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


class AutoEntry(unittest.TestCase):
    def test_normal_cli_selects_unseen_prefix_completes_and_resumes(self):
        for scheduler in ('legacy','concurrent'):
            with self.subTest(scheduler=scheduler),tempfile.TemporaryDirectory()as tmp:
                result=run_acceptance(tmp,scheduler)
                self.assertTrue(result['checks']['chosen_was_unseen'])
                self.assertEqual(result['checks']['real_scientific_calls'],0)


if __name__=='__main__':
    if '--output'in sys.argv:
        import argparse
        parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
        args=parser.parse_args();print(json.dumps(run_acceptance(args.output)['checks'],indent=2))
    else:unittest.main(verbosity=2)
