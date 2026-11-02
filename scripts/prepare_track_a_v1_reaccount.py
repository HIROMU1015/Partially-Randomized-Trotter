"""Materialize one saved-only reaccounting config; no scientific arithmetic."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import track_a_v1_reaccount as m

ROOT=m.ROOT
PUBLIC=ROOT/'artifacts/resource_applicability'


def put(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase-b-config',type=Path,required=True);p.add_argument('--private-directory',type=Path,required=True)
    p.add_argument('--public-directory',type=Path,required=True);a=p.parse_args()
    for path in (a.private_directory,a.public_directory):
        if not path.is_absolute() or path.resolve()!=path or path.exists() or not path.is_relative_to('/home/AbeHiromu/projects'):
            raise ValueError('NEW_ABSOLUTE_DIRECTORY_REQUIRED')
    old=m.r.read_json(a.phase_b_config);old_root=Path(old['provider_root']).parent/'h6-hash-reuse-eight-worker-20261012/artifacts/resource_applicability'
    files={
        'frozen':PUBLIC/'track_a_v1_phase_b_readiness/2026-10-16/frozen_model.json',
        'validation':PUBLIC/'track_a_v1_phase_b_run01/2026-10-16/validation_complete.json',
        'phase_b_terminal':PUBLIC/'track_a_v1_phase_b_run01/2026-10-16/execution_terminal.json',
        'phase_b_launcher':PUBLIC/'track_a_v1_phase_b_run01/2026-10-16/launcher_terminal.json',
        'phase_a_terminal':PUBLIC/'track_a_v1_phase_a_run02/2026-10-16/execution_terminal_v1.json',
        'additional':PUBLIC/'track_a_v1_phase_a_run02/2026-10-16/signal_results_v1.json.gz',
        'old_signals':old_root/'track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/signal_summary.json',
        'exploration_wrappers':old_root/'track_a_h6_exploration_summary_v1/2026-10-12/wrapper_metrics_v1.csv',
        'confirmation_wrappers':old_root/'track_a_h6_confirmation_summary_v1/2026-10-12/wrapper_metrics_v1.csv',
        'old_accounting':PUBLIC/'track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/candidate_accounting_v1.json',
        'b0i_accounting':PUBLIC/'track_a_h6_native_b0i_full_family_result_v1/2026-10-15/b0i_accounting_v1.json',
        'b0i_axes':PUBLIC/'track_a_h6_native_b0i_full_family_result_v1/2026-10-15/axis_compiled_metrics_v1.csv',
        'old_comparison':PUBLIC/'track_a_h6_native_b0i_full_family_result_v1/2026-10-15/comparison_and_P_sensitivity_v1.json'}
    if m.r.sha(files['frozen'])!=m.plan()['frozen_model_sha256']:raise ValueError('FROZEN_MODEL_HASH')
    provenance=m.r.read_json(PUBLIC/'track_a_d0_saved_value_reanalysis_result_v1/2026-10-14/provenance_v1.json')
    for key,alias in [('old_signals','h6_signals.json'),('exploration_wrappers','h6_exploration_wrappers.csv'),
                      ('confirmation_wrappers','h6_confirmation_wrappers.csv')]:
        expected=next(x for x in provenance['input_receipts'] if x['alias']==alias)
        if m.r.sha(files[key])!=expected['sha256'] or files[key].stat().st_size!=expected['bytes']:
            raise ValueError('OLD_RAW_RECEIPT:'+alias)
    private_base=ROOT.parent.parent/'h6-handoff-evidence/20261015'
    pilot=private_base/'h6-native-b0i-pilot6-run04-v3/run04';family=private_base/'h6-native-b0i-full-family-run04/run01'
    signal_records=[];signal_sources=[]
    for prefix in (5,10,15):
        for q in m.c.QS:
            saved=(pilot/f'case_{prefix}_{q}_scalar_signal.json' if (prefix,q) in ((5,1),(10,2),(15,8))
                   else family/f'case_{prefix}_{q}_screen.json')
            doc=m.r.read_json(saved);signal_records.append(doc['signal'])
            signal_sources.append(dict(logical_name=f'b0i_saved_signal_p{prefix}_q{q}',path=str(saved),
                bytes=saved.stat().st_size,sha256=m.r.sha(saved)))
    if {x['cell']['id'] for x in signal_records}!={x['id'] for x in m.c.registrations()['existing'] if x['method']=='B0I'}:
        raise ValueError('OLD_B0I_SIGNAL_DOMAIN')
    a.private_directory.mkdir(mode=0o700);a.public_directory.mkdir(parents=True)
    put(a.private_directory/'saved_b0i_signals.json',signal_records)
    files['b0i_signals']=a.private_directory/'saved_b0i_signals.json'
    sources=list(old['source_hashes'])+['scripts/track_a_v1_reaccount.py','scripts/prepare_track_a_v1_reaccount.py']
    config=dict(schema='track_a_v1_reaccount_config_v1',plan_sha256=m.c.digest(m.plan()),
        python=old['python'],runtime_file_hashes=old['runtime_file_hashes'],
        source_hashes={name:m.r.sha(ROOT/name) for name in sources},resources=m.plan()['resources'],
        input_mapping={key:str(path) for key,path in files.items()},
        files=[dict(logical_name=key,path=str(path),bytes=path.stat().st_size,sha256=m.r.sha(path)) for key,path in files.items()]+signal_sources,
        frozen_model_sha256=m.plan()['frozen_model_sha256'],approved=False,runtime_authorization=False,
        allowed_cpus=[],absolute_launch_command=None,output=str(a.private_directory/'output01'),
        lock=str(a.private_directory/'reaccount_ONE_SHOT'),control=str(a.private_directory/'control01'))
    put(a.private_directory/'config_draft.json',config);put(a.public_directory/'plan.json',m.plan())
    put(a.public_directory/'input_binding.json',dict(plan_sha256=config['plan_sha256'],
        frozen_model_sha256=config['frozen_model_sha256'],private_config_sha256=m.r.sha(a.private_directory/'config_draft.json'),
        source_hashes=config['source_hashes'],runtime_closure_digest=m.c.digest(config['runtime_file_hashes']),
        input_receipts=[{k:v for k,v in x.items() if k!='path'} for x in config['files']],
        copied_old_B0I_saved_records=12,old_signal_overlay_recomputed=False,
        new_signal=0,new_compile=0,new_fit=0,grant_issued=False,next_stage_authorized=False))
    print(json.dumps(dict(status='CONFIG_PREPARED_NOT_LAUNCHED',input_files=len(config['files']),
        source_files=len(config['source_hashes']),plan_sha256=config['plan_sha256']),indent=2))


if __name__=='__main__':main()
