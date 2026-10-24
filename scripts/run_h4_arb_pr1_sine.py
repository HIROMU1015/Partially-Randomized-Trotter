#!/usr/bin/env python3
"""PR1 sine-only one-shot; reuse PR1 authority/native and saved Sdg inspection."""
import argparse
import ast
import multiprocessing.popen_fork
from pathlib import Path
import runpy
import sys
import types

ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType('arb_pr1_sine_private');pkg.__path__=[str(ROOT/'src/trotterlib')];sys.modules['arb_pr1_sine_private']=pkg
from arb_pr1_sine_private import h4_arb_pr1_sine as A
C,B,BASE=A.C,A.B,A.BASE


def controller():
    prior=runpy.run_path(str(ROOT/'scripts/run_h4_arb_pr1_cosine.py'))['controller']()
    env={**prior,**globals()}
    env['CODE']=prior['CODE']+['src/trotterlib/h4_arb_pr1_sine.py','src/trotterlib/h4_r1_sine.py',
        'scripts/run_h4_arb_pr1_sine.py','tests/test_h4_arb_pr1_sine.py','tests/test_h4_r1_sine.py',
        'artifacts/non_df_h4_central_arb_pr1_sine/2026-10-25/prepare_metadata.py']
    tree=ast.parse((ROOT/'scripts/run_h4_arb_pr0_cosine.py').read_text())
    names=('resources','runtime','verify','prepare','check','worker','native_witness','launch')
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    C.require(len(nodes)==len(names),'REUSED_AUTHORITY_FUNCTIONS')
    exec(compile(ast.fix_missing_locations(A.SineScope().visit(ast.Module(body=nodes,type_ignores=[]))),
        '<same native/one-shot control, PR1 sine namespace>','exec'),env)
    return env


def inspect():
    C.hashes({str(ROOT/p):h for p,h in C.read(BASE/'inputs.json')['files'].items()})
    # Four original PR1 source/control/cosine QPY are READ ONLY. No cosine walk.
    ct,model,join,prior,binding=A.R1.preflight()
    manifest=C.read(A.S.BASE/'inputs.json')
    target=next(t for t in manifest['targets'] if t['key']=='PR1_sine')
    C.require(next(t for t in manifest['cosine_targets'] if t['key']=='PR1_cosine')==ct,'SAME_SAVED_TRAJECTORY')
    C.require(target['trajectory_id']==A.TID and target['axis']=='sine' and target['index']==1,'PR1_SINE_ONLY')
    loaded={}
    for role in ('input_qpy','compiled_qpy'):
        path=ROOT/target[role];C.require(C.sha(path)==manifest['files'][target[role]],'ACTUAL_SINE_QPY_SHA')
        with path.open('rb') as stream:items=A.S.R.api().qpy.load(stream)
        C.require(len(items)==1,'SINGLE_SINE_QPY');loaded[role]=items[0]
    folder=(ROOT/target['input_qpy']).parent
    inp,metrics=C.read(folder/'input.json'),C.read(folder/'metrics.json')
    A.S.P.check_saved_tags(inp,metrics,A.S.P.AG);A.S.check_sine_metadata('PR1',target,inp,metrics)
    sidecar=C.read((ROOT/ct['input_qpy']).parent/'input.json')['sidecar']
    sine=A.S.validate_sine(prior['source_body'],prior['controlled_body'],prior['input_qpy'],loaded['input_qpy'],sidecar)
    C.require(sine['aggregate_phase']==target['aggregate_source_phase']==A.R1.SAVED_PHASE,'SAME_PR1_RELATIVE_PHASE')
    for role,prefix in (('input_qpy','input'),('compiled_qpy','output')):
        signature=A.S.R.validate_program(loaded[role],compiled=role=='compiled_qpy',n=8)
        C.require(A.S.R.api().fingerprint(loaded[role])==target[prefix+'_fingerprint'],'SINE_QPY_FINGERPRINT')
        C.require(dict(loaded[role].count_ops())==metrics['pretranspile_histogram' if role=='input_qpy' else 'posttranspile_histogram'],'SINE_HISTOGRAM')
        sine[role]=dict(sha256=C.sha(ROOT/target[role]),fingerprint=target[prefix+'_fingerprint'],global_phase=signature['global_phase'],instructions=len(loaded[role].data))
    plan=A.build_plan(model,join)
    program=B.descriptor_from_circuit(loaded['compiled_qpy'],axis='sine')
    program['origin']=dict(qpy_sha256=C.sha(ROOT/target['compiled_qpy']),fingerprint=target['output_fingerprint'],decode_only=True,state_actions=0,compile=0)
    C.require(program['origin']['qpy_sha256']==A.QPY_SHA and program['origin']['fingerprint']==A.QPY_FP,'FIXED_SINE_COMPILED')
    guard=next(g for g in C.read(A.R1.P.BASE/'guard_inputs_quoted.json')['axes']
        if (g['case'],g['epsilon_sig'],g['axis'])==('pauli_jw_PR','0.0005','sine'))
    state=C.read(A.R1.R0BASE/'run01/state_identity.json')
    for name,value in [('PR1_target_QUOTED.json',target),('PR1_guard_QUOTED.json',guard),
            ('PR1_selected_C_plan_NO_EXECUTION.json',plan),('PR1_compiled_program_NO_EXECUTION.json',program),('state_identity_QUOTED.json',state)]:C.save(BASE/name,value)
    C.save(BASE/'structural_inspection.json',dict(status='PASS',QPY_decodes=6,state_actions=0,cosine_state_actions=0,
        prior_PR1_SOURCE_binding=binding,sine_binding=sine,order2=binding['order2'],state_hash=model['state_hash']))
    A.load_inputs()
    C.save(BASE/'structural_preflight.json',dict(status='PASS_PR1_ACTUAL_QPY_PLAN_DESCRIPTOR_IDENTITY',QPY_decodes=6,state_actions=0,new_compile=0,
        axis='sine',convention='+Im using Sdg before final H',cosine_state_actions=0,compiled_sha256=A.QPY_SHA,fingerprint=A.QPY_FP,
        C_target_unscaled=True,B_saved_exact=A.SCALE,trajectory_id=A.TID,
        exact_plan_sha256=C.sha(BASE/'PR1_selected_C_plan_NO_EXECUTION.json'),descriptor_sha256=C.sha(BASE/'PR1_compiled_program_NO_EXECUTION.json')))
    inputs=C.read(BASE/'inputs.json')
    for name in ('PR1_target_QUOTED.json','PR1_guard_QUOTED.json','PR1_selected_C_plan_NO_EXECUTION.json','PR1_compiled_program_NO_EXECUTION.json','state_identity_QUOTED.json'):
        inputs['files'][str((BASE/name).relative_to(ROOT))]=C.sha(BASE/name)
    C.save(BASE/'inputs.json',inputs,exclusive=False)
    print(A.S.R.json.dumps(dict(status='PASS_PR1_SINE_STATIC_ONLY',QPY_decodes=6,state_actions=0)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True)
    for mode in ('inspect','prepare','launch','worker'):group.add_argument('--'+mode,action='store_true')
    args=parser.parse_args();env=controller()
    if args.inspect:inspect()
    elif args.prepare:env['prepare']()
    elif args.launch:sys.exit(env['launch']())
    else:env['worker']()
