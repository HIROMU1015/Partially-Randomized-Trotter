#!/usr/bin/env python3
"""Execute the single adopted S3 case using pinned B numerical APIs.

Scientific execution requires the user's separate instruction. This caller
keeps the adopted n/seed/nominal allocation fixed; no search or automatic retry.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
CODE = 'ab12b69120323379b0aae572e7f62692a5819125'
INPUT = 'bf515a976f1987ad6a19c61d14689b27b0894542'
PLAN = 'ee1d456dcd1cb47ae1f404b01a62447ede22a8c1'
INPUT_PATH = 'docs/research/rpe_s3_readiness_20261113/metadata01/STATISTICS_INPUTS.json'
CONFIG_PATH = 'docs/research/h4_s0_v2_integration_20261113/run01/CONFIGURATION_V2.json'
PLAN_PATH = 'docs/research/rpe_s3_final_plan_20261113/SCIENTIFIC_PLAN.md'
CONFIG_ID = '587d20f7ed906d25841615dc9746674af872675b895bd438c06d3e74abf0e911'
NOMINAL = [64,60,56,52,47,43,38,34,30,25,21,17,12]
REPETITIONS = 43819
SEED = 20261113


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Fresh output directory required; no overwrite or retry')
    # Source comparison concerns this run's code, not other worktrees.
    subprocess.run(['git','diff','--exit-code',CODE,'--','src'],cwd=ROOT,check=True,capture_output=True)
    input_bytes = git('show', INPUT+':'+INPUT_PATH)
    prepared = json.loads(input_bytes)
    config = json.loads(git('show', INPUT+':'+CONFIG_PATH))
    sys.path.insert(0, str(ROOT/'src'))
    from trottertracks import rpe_decoder_statistics as S
    from trottertracks import rpe_decoder_fast as F
    # Metadata identity comparisons only. No mean/N re-evaluation.
    if config['id'] != CONFIG_ID or prepared['configuration_id'] != CONFIG_ID or prepared['M'] != 12:
        raise ValueError('Fixed canonical configuration mismatch')
    if not prepared['input_complete'] or prepared['decoder'] != S.DECODER:
        raise ValueError('Missing input or decoder mismatch')
    if not S.p.same(prepared['h'],config['spec']['h']) or not S.p.same(
            prepared['prior'],config['spec']['rmse_policy']['error_contract']['energy_prior_ha']):
        raise ValueError('Fixed h/prior mismatch')
    if [r['m'] for r in prepared['rows']] != list(range(13)):
        raise ValueError('All thirteen rounds required')
    for row, canonical, n in zip(prepared['rows'],config['rounds'],NOMINAL):
        for k in ('identity','law_binding','physical_candidate'):
            if not S.p.same(row[k],canonical[k]):
                raise ValueError('Prepared/canonical mismatch: '+k)
        if row['law_binding']['definition']['model'] != 'ideal_finite_RTE_law':
            raise ValueError('Fixed ideal law mismatch')
        for axis in ('X','Y'):
            value=row['shots']['N_'+axis]
            if type(value) is not int or value != n:
                raise ValueError('Fixed nominal allocation mismatch')
    proxy = prepared['target']['saved_truth_proxy']['reference_solver_energy_ha']
    if not math.isfinite(proxy) or abs(proxy)>32:
        raise ValueError('Proxy outside adopted support')
    plan_source = dict(commit=PLAN,path=PLAN_PATH)
    policy = dict(kind=S.IID,source=plan_source,target_law='ideal_finite_RTE_law')
    import numpy as np
    settings = dict(configuration_id=CONFIG_ID,code_source=CODE,
        caller_source=dict(commit=git('rev-parse','HEAD').decode().strip(),path='scripts/run_h4_s3_statistics.py'),
        input_source=dict(commit=INPUT,path=INPUT_PATH,sha256=hashlib.sha256(input_bytes).hexdigest()),
        configuration_source=dict(commit=INPUT,path=CONFIG_PATH),plan_source=plan_source,
        A_physical_contract='bba150834ca30bdb186fbfd58c55a8d51177b855',
        A_thirteen_round_results='169d3316dd008eb98dd9be56fe4ecefc5c1a7d60',
        scientific_conditions_adopted=True,repetitions=REPETITIONS,seed=SEED,RNG='PCG64',
        nominal_N_XY=NOMINAL,policy=policy,count_caller=S.COUNT_RULE,
        batch_decoder='trottertracks.rpe_decoder_fast.decode_sampled_counts',
        target=dict(kind='CONDITIONAL_SAVED_SOLVER_PROXY',energy_ha=proxy,
            source=prepared['target']['source'],energy_shift_ha=0,ground_state_certified=False),
        primary_unresolved_loss='UNDEFINED',include_loss_envelope=True,
        envelope_unresolved_loss_interval_Ha2=[0.,4096.],energy_support_Ha=[-32.,32.],
        uniform_loss_upper_Ha2=4096.,planned_MSE_endpoint_half_width_Ha2=40.96,
        component_confidence=.975,joint_nominal_confidence=.95,
        numerical_scope='BINARY64_FIXED_SUPPLIED_MEANS_PROXY_IID_MODEL_NOT_PHYSICAL_CERTIFICATION',
        physical_error_contract_source=prepared['physical_contract_source'],
        original_prepared_target_and_qualification_unchanged=True,
        python=sys.version,numpy_version=np.__version__,platform=platform.platform(),
        sufficient_N=None,global_RMSE_bound=None,G=None)
    args.out.mkdir(parents=True)
    write_json(args.out/'EXECUTION.json', settings)
    simulation = None
    timings = {}
    try:
        print('START fixed S3: n=43819, seed=20261113, 13 rounds, one simulation', flush=True)
        start=time.perf_counter()
        simulation=S.simulate_prepared(prepared,repetitions=REPETITIONS,seed=SEED,policy=policy,
            allow_conditional_unqualified=True,batch_decoder=F.decode_sampled_counts)
        timings['simulation_seconds']=time.perf_counter()-start
        # Save every original trial before summaries/checks; no discarded failures.
        start=time.perf_counter()
        with (args.out/'SIMULATION.json.gz').open('xb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as compressed:
                import io
                with io.TextIOWrapper(compressed,encoding='utf-8') as f:
                    json.dump(simulation,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
                    f.write('\n')
        timings['archive_seconds']=time.perf_counter()-start
        trials=simulation['trials']
        if len(trials)!=REPETITIONS:
            raise ValueError('Incomplete trial ensemble')
        for row in trials:
            if row['status']==S.DECODED:
                value=row['energy_estimate_ha']
                if not math.isfinite(value) or not -32<=value<=32 or (value-proxy)**2>4096:
                    raise ValueError('Adopted support exceeded; no clipping')
            elif row['status'] not in ('UNRESOLVED_ZERO_COORDINATES','UNRESOLVED_BRANCH_TIE') or row['energy_estimate_ha'] is not None:
                raise ValueError('Unknown decoder status or supplied fallback')
        common=dict(target_energy_ha=proxy,iid_repetitions=True,confidence=.975)
        start=time.perf_counter()
        primary=S.summarize(trials,**common,unresolved_loss=dict(kind='UNDEFINED',source=plan_source))
        write_json(args.out/'PRIMARY.json',primary)
        envelope=S.summarize(trials,**common,
            unresolved_loss=dict(kind='BOUNDED_INTERVAL',lower=0.,upper=4096.,source=plan_source),
            loss_support=dict(upper=4096.,source=plan_source))
        write_json(args.out/'LOSS_ENVELOPE.json',envelope)
        timings['summary_seconds']=time.perf_counter()-start
        result=dict(status='COMPLETE_FIXED_CONDITIONAL_S3',requested_trials=REPETITIONS,
            completed_trials=len(trials),simulation_calls=1,scientific_RMSE_PASS=None,
            full_estimator_population_RMSE=None,conditional_RMSE_CI=None,
            support_checks='PASS_ADOPTED_ASSUMPTION_NOT_INTERVAL_CERTIFICATION',
            envelope_is_loss_completion_sensitivity=True,timings=timings,
            peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            sufficient_N=None,global_RMSE_bound=None,qualified_compiled_C=None,G=None,
            mandatory_stop=True)
        write_json(args.out/'RESULT.json',result)
        # Scientific artifact integrity; no approval proof or user-text storage.
        artifacts=[]
        for name in ('EXECUTION.json','SIMULATION.json.gz','PRIMARY.json','LOSS_ENVELOPE.json','RESULT.json'):
            path=args.out/name;digest=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
            artifacts.append(dict(path=name,bytes=path.stat().st_size,sha256=digest.hexdigest()))
        write_json(args.out/'ARTIFACTS.json',dict(artifacts=artifacts))
        print(json.dumps(dict(status=result['status'],trials=len(trials),
            status_counts=primary['status_counts'],conditional_RMSE=primary['conditional_on_decoded_RMSE'],
            unresolved_probability_CI=primary['unresolved_probability_CI'],
            envelope_population_CI=envelope['population_RMSE_CI'],timings=timings),allow_nan=False),flush=True)
    except Exception as error:
        if not (args.out/'RESULT.json').exists():
            write_json(args.out/'RESULT.json',dict(status='STOPPED_NO_AUTOMATIC_RETRY',
                error=repr(error),traceback=traceback.format_exc(),
                returned_trials=None if simulation is None else len(simulation.get('trials') or []),
                sufficient_N=None,global_RMSE_bound=None,G=None,mandatory_stop=True))
        raise


if __name__=='__main__':main()
