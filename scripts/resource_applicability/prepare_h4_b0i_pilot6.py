#!/usr/bin/env python3
"""Byte-only preparation of an unapproved six-task pilot; never run a circuit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys

ROOT=Path(__file__).absolute().parents[2];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();resource.setrlimit(resource.RLIMIT_AS,(512*2**20,512*2**20))
    resource.setrlimit(resource.RLIMIT_CPU,(60,60));signal.alarm(90)
    from trottertracks.resource_applicability.h4_b0i import adapter as a,contract as c
    from trottertracks.resource_applicability.h4_geometry.prelaunch_audit import private_path,streaming_sha,host_readonly,cpu_proposal,environment_profile,compiler_profile
    out=private_path(args.output);out.mkdir(exist_ok=False)
    old=ROOT/'artifacts/resource_applicability/track_a_h4_geometry_scalar_diagnostic/2026-10-13'
    projection_path='artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/saved_candidates_v1.json'
    projection=json.loads((ROOT/projection_path).read_text())
    phases=json.loads((old/'saved_b0_phase_sensitivity_v1.json').read_text())
    receipts=json.loads((old/'source_input_provenance_v1.json').read_text())
    sealed=json.loads((ROOT/'artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/plan_sealed_v7.json').read_text())
    input_root=private_path(sealed['input_root']);tasks=[];input_audit={}
    for g,prefix in c.SELECTION:
        entry=next(i for i in receipts['inputs'] if i['geometry']==g)
        input_file=dict(path=str(input_root/entry['file']),bytes=entry['bytes'],sha256=entry['sha256'])
        actual=streaming_sha(Path(input_file['path']))
        if actual!={'bytes':input_file['bytes'],'sha256':input_file['sha256']}:raise ValueError('input byte receipt differs')
        input_audit[g]=actual
        phase=next(v for v in phases if v['geometry']==g and v['prefix']==prefix and '-q1-' in v['original_template_id'])
        candidate=next(v for v in projection['candidates'] if v['geometry']==g and v['template_id']==phase['original_template_id'])
        for index,axis in enumerate(('cosine','sine')):
            task=dict(geometry=g,prefix=prefix,axis=axis,template=a.template(prefix,1),input_file=input_file,
                input_identity=entry['input_identity'],tail_identity=phase['tail_identity'],
                phase_angle=phase['phase_angle'],original_B0_template=phase['original_template_id'],
                old_B0_reference_RZ=candidate['saved']['paired_costs'][0][index],
                saved_original_signal=phase['saved_corrected_signal'],saved_phase_rotated_signal=phase['scalar_phase_counterfactual'],
                saved_target=phase['target'],saved_bias_B0I=phase['counterfactual_bias'],normalization=1.,
                trajectory_seed=None,trajectory_index=None,numeric_circuit_fingerprint=None,
                validation_numeric_fingerprint='DEFERRED_UNTIL_APPROVED_BUILD',compiled_cost='UNMEASURED')
            task['task_id']=c.digest(task);tasks.append(task)
    env_path='artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/environment_profile_v2.json'
    comp_path='artifacts/resource_applicability/track_a_h4_native_proof_seal/2026-10-09/compiler_profile_v2.json'
    env=json.loads((ROOT/env_path).read_text());comp=json.loads((ROOT/comp_path).read_text())
    observed_env=environment_profile(env['dependencies'],[Path(p) for p in env['installed_sources']])
    observed_comp=compiler_profile(comp['explicit_options'],comp['inherited_defaults'])
    profile_audit={'environment_matches_frozen':observed_env==env,'compiler_matches_frozen':observed_comp==comp,
        'environment_fingerprint':observed_env['fingerprint'],'compiler_fingerprint':observed_comp['fingerprint'],
        'old_compiler_output_equivalence':False,'install_upgrade_setting_changes':0}
    (out/'profile_audit.json').write_text(json.dumps(profile_audit,indent=2)+'\n')
    if observed_env!=env or observed_comp!=comp:raise ValueError('profile drift; files preserved, no repair/install')
    paths=[*sorted((ROOT/'src/trottertracks/resource_applicability/h4_geometry').glob('*.py')),
           *sorted((ROOT/'src/trottertracks/resource_applicability/h4_b0i').glob('*.py')),
           ROOT/'src/trottertracks/__init__.py',ROOT/'src/trottertracks/resource_applicability/__init__.py',
           ROOT/'tests/tracks/resource_applicability/test_h4_b0i_preparation.py',
           *[ROOT/'scripts/resource_applicability'/n for n in ('run_h4_b0i_pilot6.py','run_h4_b0i_preparation_tests.py','prepare_h4_b0i_pilot6.py')]]
    relative=[str(p.relative_to(ROOT)) for p in paths]
    subprocess.run(['git','cat-file','-e',args.source+'^{commit}'],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--exit-code',args.source,'--',*relative],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    def stamp(p):
        data=p.read_bytes();return dict(path=str(p.relative_to(ROOT)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    closure=[stamp(p) for p in paths]
    bound_paths=[env_path,comp_path,projection_path,
        'artifacts/resource_applicability/track_a_h4_geometry_results/2026-10-13/summary.json',
        *[str((old/n).relative_to(ROOT)) for n in ('source_input_provenance_v1.json','saved_b0_phase_sensitivity_v1.json','tail_scalar_metadata_v1.json')]]
    pilot_parent=out.parent/'future-pilot6';pilot_parent.mkdir(exist_ok=True)
    host=host_readonly(out,sample_seconds=1)
    proposed=cpu_proposal(host,workers=1)
    plan=dict(schema_version='h4-b0i-pilot6-plan-v1',namespace=a.NAMESPACE,method=a.METHOD,semantics=a.SEMANTICS,
        original_science_source=a.ORIGINAL_SOURCE,source_commit=args.source,
        source_closure=closure,bound_public_files=[stamp(ROOT/p) for p in bound_paths],
        tasks=tasks,compiler_options=comp['explicit_options'],environment_profile=env_path,compiler_profile=comp_path,
        output=str(pilot_parent/a.NAMESPACE),control=str(pilot_parent/(a.NAMESPACE+'-control')),
        one_shot_lock=str(pilot_parent/(a.NAMESPACE+'.lock')),caps=c.CAPS,carry=c.CARRY,
        approved=False,runtime_authorization=False,allowed_cpus=[],absolute_launch_command=None,
        proposed_role_cpus=proposed,pressure_policy='h4-memory-pressure-policy-v1 (strict host<1%, nonroot0, unchanged OOM)',
        numeric_fingerprints='not computed on real H4 during preparation',**c.FLAGS)
    c.validate_plan(plan)
    auth=dict(schema_version='h4-b0i-pilot6-authorization-v1',namespace=a.NAMESPACE,source_commit=args.source,
        plan_digest=c.digest(plan),approved=False,runtime_authorization=False,allowed_cpus=[],role_cpus={},
        absolute_launch_command=None,user_approval_evidence_sha256=None,**c.FLAGS)
    review=dict(schema_version='h4-b0i-pilot6-review-v1',approved=False,plan_digest=c.digest(plan),
        authorization_digest=c.digest(auth),scientific_baseline_adopted=False,launch_instruction_received=False)
    outputs={'plan.json':plan,'authorization_draft.json':auth,'review_draft.json':review,'registry.json':tasks,
        'source_closure.json':closure,'input_byte_audit.json':input_audit,'host_readonly.json':host,
        'binding.json':{'plan_digest':c.digest(plan),'authorization_digest':c.digest(auth),'review_digest':c.digest(review),
                        'source_commit':args.source,'technical_bound':True,'runtime_sealed':False}}
    for name,value in outputs.items():(out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PREPARATION_ONLY_BOUND_NOT_RUNTIME_AUTHORIZED',source=args.source,tasks=6,
        source_files=len(closure),frozen_input_bytes_verified=sum(v['bytes'] for v in input_audit.values()),
        proposed_cpus=proposed,actual_new_compile_calls=0)))


if __name__=='__main__':main()
