"""Bind accepted saved PF inputs and dormant source; never launch science."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import pf_direct_compile_contract as c
import pf_direct_execution as e

PROJECTS=Path('/home/AbeHiromu/projects'); WT=PROJECTS/'partially-randomized-trotter-worktrees'
H2=WT/'h2-pr-resource-preparation-20261019/.h2_pr_resource_private/2026-10-19'
H4=WT/'h4-tol-only-main-phase0-20261019/.h4_tol_main_private'
LEGACY=WT/'h4-dp-phase0-20261019/.h4_dp_phase0_private'
CONFIGS={
'H2_DF3':H2/'fp_d_a_preparation01/D_spec_PENDING.json',
'H4_DF10':H4/'rank10_native_config_PRIVATE.json',
'H6_DF19':PROJECTS/'h6-dba-validation/20261018/private/config_D_sealed.json',
'H8_DF28':PROJECTS/'h8-dba-validation/20261018/private/config_D_sealed.json',
'H4_LEGACY_DF12':LEGACY/'native_for_D.json'}


def read_signals(path):
    raw=Path(path).read_bytes()
    value=json.loads(gzip.decompress(raw) if raw[:2]==b'\x1f\x8b' else raw)
    return value if isinstance(value,list) else value['signals']


def bind(source):
    c.require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=e.ROOT).decode().strip()==source,'COMMITTED_SOURCE_REQUIRED')
    target=e.PRIVATE/'sealed01';target.mkdir(exist_ok=False)
    approval=e.PRIVATE/'scientific_approval01.json';c.require(approval.is_file(),'EXACT_USER_APPROVAL_RECEIPT_REQUIRED')
    dependencies={};systems={};inputs={}
    for system,path in CONFIGS.items():
        raw=c.read_json(path);cfg=raw['native_config'] if system=='H2_DF3' else raw
        if system=='H2_DF3':
            runtime=c.read_json(raw['runtime_manifest'])
            provider={str(Path(cfg['provider_root'])/r['role']):r['sha256'] for r in raw['provider_files']}
            input_files={str(p):e.sha(p) for p in map(Path,cfg['mapping'].values())}
            profile=dict(commit='87b6b9f46b7e8ba9abe132ee21478c114b6f46f7',
                path='artifacts/resource_applicability/track_a_h6_server_preparation_v2/2026-10-12/environment_compiler_profile_v2.json',
                sha256='4e32bb9ba57c38280562390100ebf916e9858218534504ead1519625a609bd75')
            provider_source='217c971dd537b1cc7fbc13742bc0f6c263ff7f45'
        else:
            runtime=raw['runtime_file_hashes'];provider={str(Path(cfg['provider_root'])/name):h for name,h in raw['provider_hashes'].items()}
            input_files={r['path']:r['sha256'] for r in raw['inputs']}
            profile=raw.get('profile_reference',dict(sha256=raw.get('profile_sha256')))
            provider_source=raw['provider_source']
        if system=='H8_DF28':
            origin=PROJECTS/'h6-handoff-evidence/20261015/h8-native-b0i-full-family-run01/final_contract_v1.json'
            parent=c.read_json(origin)
            c.require(c.digest(parent['profile'])==profile['sha256']==parent['expected_profile_digest'],'H8_PROFILE_CONTENT')
            c.require(parent['profile']['runtime_file_hashes']==runtime,'H8_PROFILE_RUNTIME')
            input_files[str(origin)]=e.sha(origin)
            profile=dict(profile,content_digest=True,reference=parent['profile']['full_critical_hashes_reference'])
        files={**runtime,**provider,**input_files,str(path):e.sha(path)}
        for name,h in files.items():
            c.require(name not in dependencies or dependencies[name]==h,'CONFLICTING_INPUT_PROFILE')
            dependencies[name]=h
        # Verify provider against its original Git objects, not a mutable HEAD.
        for name,h in provider.items():
            rel=str(Path(name).relative_to(cfg['provider_root']))
            blob=subprocess.check_output(['git','show',provider_source+':'+rel],cwd=cfg['provider_root'])
            c.require(hashlib.sha256(blob).hexdigest()==h,'PROVIDER_GIT_BLOB')
        if profile.get('commit'):
            blob=subprocess.check_output(['git','show',profile['commit']+':'+profile['path']],cwd=e.ROOT)
            c.require(hashlib.sha256(blob).hexdigest()==profile['sha256'],'PROFILE_GIT_BLOB')
        systems[system]=dict(config=cfg,python=raw['python'],provider_SOURCE=provider_source,
            input_hashes=input_files,profile=profile,provider_hashes=provider)
        inputs[system]=set(files)
    sources={name:e.sha(e.ROOT/name) for name in e.CODE}
    # Select saved control records only; no new oracle/signal evaluation.
    signal_sources={
      'H2_DF3':[H2/'attempts/D_run01/output/signals.json'],
      'H4_DF10':[H4/'D_seal01_20261019/run01/output/signals.json'],
      'H6_DF19':[e.ROOT/'artifacts/resource_applicability/track_a_v1_phase_a_run02/2026-10-16/signal_results_v1.json.gz',
        WT/'h6-hash-reuse-eight-worker-20261012/artifacts/resource_applicability/track_a_h6_matched_accuracy_v2/2026-10-11/launch_v1/signal/signal_summary.json'],
      'H8_DF28':[PROJECTS/'h8-dba-validation/20261018/private/inputs/H8_phase_A_signals.json',
        PROJECTS/'h6-handoff-evidence/20261014/h8-resource-map-resume02-v1/run01/signal_summary.json'],
      'H4_LEGACY_DF12':[LEGACY.parent/'artifacts/resource_applicability/track_a_h4_dp_phase0/2026-10-19/saved_signal_catalog.json']}
    oracles={};provenance={}
    for system,paths in signal_sources.items():
        records={}
        for path in paths:
            dependencies[str(path)]=e.sha(path);inputs[system].add(str(path))
            for row in read_signals(path):
                cid=row['cell']['id']
                if cid not in records: records[cid]=dict(record=row,source=str(path),source_sha256=e.sha(path))
        # Existing B0I saved rotations; no new rotation or scalar estimation.
        for prefix in ((5,) if system=='H6_DF19' else (7,) if system=='H8_DF28' else ()):
            for q in (1,2,4,8):
                cid=f'B0I_p{prefix}_q{q}_2nd'
                if cid in records:continue
                family='h6-native-b0i-full-family-run04' if system=='H6_DF19' else 'h8-native-b0i-full-family-run01'
                directory=PROJECTS/'h6-handoff-evidence/20261015'/family/'run01'
                if system=='H6_DF19' and q==1:
                    path=PROJECTS/'h6-handoff-evidence/20261015/h6-native-b0i-pilot6-run04-v3/run04/case_5_1_scalar_signal.json'
                else:path=directory/f'case_{prefix}_{q}_screen.json'
                doc=c.read_json(path)
                row=doc['signal'];c.require(row['cell']['id']==cid,'SAVED_ORACLE_CANDIDATE')
                records[cid]=dict(record=row,saved_outer_scalar=doc.get('scalar'),source=str(path),source_sha256=e.sha(path))
                dependencies[str(path)]=e.sha(path);inputs[system].add(str(path))
        oracles[system]=records
    e.save(target/'oracles.json',oracles);dependencies[str(target/'oracles.json')]=e.sha(target/'oracles.json')
    audit=e.hash_audit(dependencies);e.save(target/'byte_gate.json',audit)
    c.require(audit['status']=='PASS','DEPENDENCY_BYTES_STOP')
    for group in e.GROUPS:
        _,selected=e.expected_tasks(group);series=sorted({r['system'] for r in selected})
        missing=[r['task_id'] for r in selected if r['candidate_id'] not in oracles[r['system']]]
        spec=dict(schema='PF_DIRECT_PRODUCTION_SEALED_V1',contract_id=group,SOURCE=source,base_commit=e.BASE,
            sealed=True,selected_task_ids=[r['task_id'] for r in selected],call_caps=c.call_caps(selected),
            excluded_conditional_axes=48,compiler_hash=c.COMPILER,T=.8,P=0,control='symmetric_directional',
            source_hashes=sources,bound_files={k:v for k,v in dependencies.items() if k==str(target/'oracles.json') or any(k in inputs[s] for s in series)},
            systems={s:systems[s] for s in series},oracles=str(target/'oracles.json'),
            missing_saved_oracles=missing,output_root=str(e.PRIVATE/'run01'/group),
            input_reconstruction_policy='same saved DF and state; original native preparation restoration; no SCF/DF/state solver',
            control_oracle_policy='existing classical corrected and independent_forward if saved; no new reference/signal',
            one_shot=True,retry=False,next_stage_authorized=False,resource_limits=None)
        grant=dict(action='PF_DIRECT_COMPILE_SCIENTIFIC_EXECUTION',approved_by_user=True,runtime_authorization=True,
            contract_id=group,SOURCE=source,spec_digest=c.digest(spec),retry=False,next_stage_authorized=False,
            approval_receipt=str(approval),approval_sha256=e.sha(approval))
        e.authorize(spec,grant)
        e.save(target/(group+'_spec.json'),spec);e.save(target/(group+'_grant.json'),grant)
    print(json.dumps(dict(status='SEALED',SOURCE=source,dependency_files=len(dependencies),
        oracle_counts={s:len(v) for s,v in oracles.items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--SOURCE',required=True)
    bind(p.parse_args().SOURCE)
