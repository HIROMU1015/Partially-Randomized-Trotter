"""Fixed saved LD1/LD2, one whole vector each; explicit no-MC science caller.

prepare writes selected scientific inputs. execute evaluates the two entries
once into a fresh directory. Re-execution needs its own permitted scope.
Numerical methods are exclusively the existing deterministic-risk module.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from trottertracks import rpe_deterministic_risk as D
# Reuse ONLY the existing pinned pure A/C metadata functions. Importing this
# helper does not run unittest, synthetic(), MC, signal or price producers.
from check_rpe_signal_statistics_caller import I

BASE = '5e8f1664d9612998beeb83c3078691d7616390eb'
ENGINE = 'd71721f30c7b7fc13ba0789eefffd6238aca3ebc'
DIRECTORY = 'docs/research/pr_ld1_ld2_deterministic_risk_results_20261113'
INPUT = DIRECTORY+'/INPUTS.json'
SOURCE_METADATA = 'docs/research/pr_ld1_ld2_deterministic_application_preparation_20261113/SOURCE_AND_CONDITIONS.json'
N = [64,60,56,52,47,43,38,34,30,25,21,17,12]


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def prepare():
    metadata = json.loads(git('show', BASE+':'+SOURCE_METADATA))
    store = D.p.SavedArtifacts(ROOT)
    cases = []
    for entry in metadata['configurations']:
        bundle = store.read(entry['bundle_source'])
        config = store.read(entry['configuration_source'])
        assert bundle['configuration_id'] == config['id'] == entry['configuration_id']
        assert config['spec']['M'] == 12 and bundle['shots'] is None
        cases.append(dict(label=entry['label'], bundle_source=entry['bundle_source'],
            configuration_id=config['id'], configuration_source=entry['configuration_source'],
            allocation=dict(allocation_id='H4_'+entry['label']+'_CONTROL_WHOLE_VECTOR_01',
                shot_policy='USER_SUPPLIED',
                shots=[dict(m=m, N_X=n, N_Y=n) for m,n in enumerate(N)]),
            evaluation_target=dict(kind='SAVED_FULL_DF_H_SOLVER_CONDITIONAL_PROXY',
                energy_ha=bundle['target']['reference_solver_energy_ha'],
                source=deepcopy(bundle['target']['source']),
                truth_proxy_source=deepcopy(bundle['target']['truth_proxy_source']),
                qualification_scope='RAW_MODEL_DIAGNOSTIC_ONLY_NOT_CERTIFIED_GROUND_ENERGY',
                original_target_source=deepcopy(bundle['target']['source']),
                binding=deepcopy(bundle['target']['binding']))))
    assert [c['label'] for c in cases] == ['LD1','LD2']
    assert cases[0]['evaluation_target'] == cases[1]['evaluation_target']
    h = float(D.a.exact_scalar(metadata['shared']['h']))
    edge = math.nextafter(math.pi/h, math.inf)
    target = D.a.exact_scalar(cases[0]['evaluation_target']['energy_ha'])
    upper = max((D.a.exact_scalar(edge)-target)**2, (-D.a.exact_scalar(edge)-target)**2)
    cells = [(n+1)**2 for n in N]
    inputs = dict(engine_source=dict(commit=ENGINE,path='src/trottertracks/rpe_deterministic_risk.py'),
        preparation_source=dict(commit=BASE,path=SOURCE_METADATA),
        scope='TWO_FIXED_RAW_MODEL_DIAGNOSTICS_NO_N_QUALIFICATION_OR_G',
        cases=cases, original_h=metadata['shared']['h'],
        statistical_contract=dict(id='H4_LD1_LD2_DETERMINISTIC_RAW_CONTROL_01',
            claim_level='RAW_MODEL_DIAGNOSTIC_ONLY', allow_conditional_unqualified=True,
            simulation_policy=dict(kind=D.S.IID, target_law='ideal_finite_RTE_law',
                interpretation='CONDITIONAL_IID_SHOTS_AXES_ROUNDS_NO_MC_REPETITIONS'),
            summary_options=dict(unresolved_loss=dict(kind='BOUNDED_INTERVAL',lower=0,
                upper=D.a.exact_value(upper)))),
        options=dict(probability_rule=D.EXACT_AFFINE, phase_radii=[1.]*13,
            maximum_round_outcomes=max(cells),
            maximum_verification_pairs=sum(a*b for a,b in zip(cells,cells[1:])),
            model_loss_contract=dict(adopted=True, scope='RAW_MODEL_DIAGNOSTIC_LOSS_FAMILY_ONLY')),
        loss_definition=dict(energy_edge_binary64=edge,
            formula='max((nextafter(pi/float(original_h),+inf)-E_proxy)^2,(-nextafter(pi/float(original_h),+inf)-E_proxy)^2)',
            arithmetic='EXACT_RATIONAL_DIFFERENCE_AND_SQUARE_OF_STORED_BINARY64_SCALARS',
            unresolved_energy_imputation=None),
        sufficient_N=None, global_RMSE_bound=None, G=None, winner=None)
    path=ROOT/INPUT; path.parent.mkdir(parents=True,exist_ok=True)
    write_new(path,inputs)
    print('PREPARED two fixed input entries; no count/risk evaluation',flush=True)


def evaluate_one(case, common, source):
    start=time.perf_counter()
    with ExitStack() as forbidden:
        for module,name in ((D.S,'simulate'),(D.S,'simulate_prepared'),(D.K,'diagnose'),
                            (D.W,'diagnose'),(D.p,'run_missing'),(D,'compare'),
                            (D.W,'compare'),(D.B,'cost'),(D.B,'select')):
            forbidden.enter_context(patch.object(module,name,
                side_effect=AssertionError('Outside the fixed two deterministic evaluations: '+name)))
        accepted=D.K.accept_saved(ROOT,bundle_source=case['bundle_source'],integration=I)
        config=accepted['request']['configuration']; packet=accepted['signal_packet']
        assert config['id'] == case['configuration_id']
        assert packet['target']['truth_proxy_adopted_for_new_simulation'] is False
        assert all(m['qualification']=='UNQUALIFIED' for m in packet['means'])
        allocation=deepcopy(case['allocation']); allocation['source']=deepcopy(source)
        contract=deepcopy(common['statistical_contract']); contract['source']=deepcopy(source)
        contract['simulation_policy']['source']=deepcopy(source)
        contract['summary_options']['unresolved_loss']['source']=deepcopy(source)
        options=deepcopy(common['options']); options['source']=deepcopy(source)
        options['model_loss_contract']['source']=deepcopy(source)
        # The only scientific evaluation call in a worker. No retry or expansion.
        records=D.evaluate(config,packet,[allocation],evaluation_target=case['evaluation_target'],
            statistical_contract=contract,integration=I,source=source,options=options)
    assert len(records)==1
    record=records[0]; risk=record['diagnostic']['deterministic_risk']
    derived=dict(global_tie_probability=None,global_tie_qualification='UNKNOWN_NOT_HISTORY_ENUMERATED')
    if risk['decoder_certificate'] and risk['decoder_certificate']['status']=='VERIFIED_FINITE_BINARY64_SUPPORT':
        unsafe=1-D.a.exact_scalar(risk['all_good']['exact'])
        zero=D.a.exact_scalar(risk['any_zero']['exact'])
        derived.update(any_unsafe=D.a.exact_value(unsafe),
            zero_free_global_tie_bounds=dict(lower=0,upper=D.a.exact_value(unsafe-zero),
                meaning='ZERO_PRECEDENCE_AND_ALL_GOOD_CLOSURE_BOUND_NOT_EXACT_HISTORY_TIE'))
    for name in ('n_packet','sufficient_N','global_RMSE_bound','G'):
        assert record[name] is None
    return dict(label=case['label'],source=source,record=record,derived=derived,
        performance=dict(wall_seconds=time.perf_counter()-start,
            peak_worker_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        MC_calls=0,deltaE_addition_performed=False,formal_energy_precision_test_performed=False,
        sufficient_N=None,global_RMSE_bound=None,G=None,winner=None)


def execute():
    source_commit=git('rev-parse','HEAD')
    assert git('show',source_commit+':'+INPUT)+'\n' == (ROOT/INPUT).read_text()
    assert git('show',source_commit+':scripts/'+Path(__file__).name)+'\n' == Path(__file__).read_text()
    assert not git('diff',BASE,'--name-only','--','src','tests')
    inputs=json.loads((ROOT/INPUT).read_text())
    assert len(inputs['cases'])==2 and [v['label'] for v in inputs['cases']]==['LD1','LD2']
    output=ROOT/DIRECTORY/'run01'; output.mkdir()  # Existing results are never overwritten.
    source=dict(commit=source_commit,path=INPUT,caller='scripts/'+Path(__file__).name)
    write_new(output/'EXECUTION.json',dict(source=source,engine_source=inputs['engine_source'],
        configurations=[v['configuration_id'] for v in inputs['cases']],
        evaluations_requested=2,MC_calls=0,prices_or_G_requested=False))
    try:
        start=time.perf_counter(); completed=[]
        # Two workers because there are exactly two independent tasks; no
        # artificial CPU/RAM/thread settings or shared environment changes.
        with ProcessPoolExecutor(max_workers=len(inputs['cases'])) as pool:
            futures={pool.submit(evaluate_one,case,inputs,source):case['label'] for case in inputs['cases']}
            for future in as_completed(futures):
                label=futures[future]; result=future.result()
                write_new(output/('RESULT_'+label+'.json'),result)
                completed.append(label)
                print(label+': '+result['record']['diagnostic']['deterministic_risk']['status'],flush=True)
        write_new(output/'COMPLETION.json',dict(status='TWO_DIAGNOSTICS_COMPLETED',
            completed=sorted(completed),wall_seconds=time.perf_counter()-start,
            sufficient_N=None,global_RMSE_bound=None,G=None,winner=None,mandatory_stop=True))
    except Exception as exc:
        write_new(output/'ERROR.json',dict(type=type(exc).__name__,message=str(exc),
            automatic_retry=False,mandatory_stop=True))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    mode=parser.parse_args().mode
    (prepare if mode=='prepare' else execute)()
